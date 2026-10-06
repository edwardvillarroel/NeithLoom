"""Registro de recursos (RAM y CPU) alrededor de la conversión a puntadas.

`stitch_generator.generate_stitches` es la etapa más pesada del programa
(imagen + colores de hilo -> matriz de puntadas), así que este módulo la
envuelve automáticamente con un muestreo cada `SAMPLE_INTERVAL_S` segundos:

- RSS del proceso y % de CPU del proceso actual.
- % de uso de cada núcleo a nivel de sistema (`psutil.cpu_percent(percpu=True)`).

Cada ejecución crea su propio `logs/recursos_<timestamp>.csv` con las columnas
`tiempo_s`, `ram_mb`, `cpu_proceso_pct` y una columna por núcleo (`core_0`,
`core_1`, ...). Al terminar imprime por consola un resumen: RAM pico, RAM
promedio, CPU pico del proceso, duración total y cuántos núcleos pasaron del
50 % de uso en algún momento (indicador de single/multi-hilo).

OJO: los porcentajes por núcleo son de TODO el sistema, no solo del proceso,
así que otros programas abiertos también cuentan.
"""

from __future__ import annotations

import csv
import functools
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

SAMPLE_INTERVAL_S = 0.5
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
CPU_CORE_THRESHOLD = 50.0

_lock = threading.Lock()
_active = False


def _unique_csv_path(log_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"recursos_{stamp}.csv"
    counter = 1
    while path.exists():
        path = log_dir / f"recursos_{stamp}_{counter}.csv"
        counter += 1
    return path


class ResourceLogger:
    """Muestrea RAM/CPU cada 0.5 s en un hilo aparte y escribe un CSV."""

    def __init__(self, label: str, interval: float = SAMPLE_INTERVAL_S):
        self.label = label
        self.interval = interval
        self.path: Path | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._writer = None
        self._n_cores = 0
        self._process: psutil.Process | None = None
        self._started_at = 0.0
        self._rows = 0
        self._ram_peak = 0.0
        self._ram_sum = 0.0
        self._cpu_peak = 0.0
        self._core_peak: list[float] = []

    # ------------------------------------------------------------------
    def start(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = _unique_csv_path(LOG_DIR)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        # Primera llamada a cpu_percent devuelve 0.0: se calienta aquí para
        # que el primer muestreo ya refleje el uso real del intervalo. El
        # objeto Process se reutiliza: el estado entre llamadas vive en él.
        # (El calentamiento de los PORCENTAJES POR NÚCLEO se hace en `_loop`,
        # porque ese estado de psutil es por hilo.)
        self._process = psutil.Process()
        self._n_cores = len(psutil.cpu_percent(percpu=True))
        self._process.cpu_percent()
        self._core_peak = [0.0] * self._n_cores

        self._writer.writerow(
            ["tiempo_s", "ram_mb", "cpu_proceso_pct"]
            + [f"core_{i}" for i in range(self._n_cores)]
        )
        self._file.flush()

        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, name="resource-logger", daemon=True
        )
        self._thread.start()
        print(f"[recursos] midiendo '{self.label}' -> {self.path}")
        return self.path

    def _loop(self) -> None:
        # El baseline de `cpu_percent(percpu=True)` se guarda POR HILO: si se
        # calienta en `start()` (hilo principal) la primera muestra de este
        # hilo compara contra su propio arranque y sale 0.0 en todas las
        # columnas. Hay que calentar aquí, dentro del hilo muestreado.
        psutil.cpu_percent(percpu=True)
        while not self._stop_event.wait(self.interval):
            self._sample()
        self._sample()  # muestra final justo al detener

    def _sample(self) -> None:
        elapsed = time.perf_counter() - self._started_at
        process = self._process
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_pct = process.cpu_percent()
        cores = psutil.cpu_percent(percpu=True)

        self._writer.writerow(
            [f"{elapsed:.2f}", f"{ram_mb:.1f}", f"{cpu_pct:.1f}"]
            + [f"{value:.1f}" for value in cores]
        )
        self._file.flush()

        self._rows += 1
        self._ram_sum += ram_mb
        if ram_mb > self._ram_peak:
            self._ram_peak = ram_mb
        if cpu_pct > self._cpu_peak:
            self._cpu_peak = cpu_pct
        for index, value in enumerate(cores):
            if index < len(self._core_peak) and value > self._core_peak[index]:
                self._core_peak[index] = value

    # ------------------------------------------------------------------
    def stop(self) -> Path | None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 4 + 1)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._print_summary()
        return self.path

    # ------------------------------------------------------------------
    def _print_summary(self) -> None:
        duration = time.perf_counter() - self._started_at
        ram_avg = self._ram_sum / self._rows if self._rows else 0.0
        over = [i for i, peak in enumerate(self._core_peak) if peak > CPU_CORE_THRESHOLD]
        kind = "multi-hilo" if len(over) > 1 else "single-hilo"
        print("")
        print(f"===== Resumen de recursos: {self.label} =====")
        print(f"CSV                       : {self.path}")
        print(f"Duracion total            : {duration:.2f} s ({self._rows} muestras)")
        print(f"RAM pico                  : {self._ram_peak:.1f} MB")
        print(f"RAM promedio              : {ram_avg:.1f} MB")
        print(f"CPU proceso pico          : {self._cpu_peak:.1f} %")
        print(
            f"Nucleos > 50% en algun momento: {len(over)} de {self._n_cores} -> {kind}"
        )
        if over:
            print(f"  (nucleos: {', '.join(str(i) for i in over)})")
        print("================================================")


# ----------------------------------------------------------------------
def measure_resources(label: str):
    """Decorador: registra RAM/CPU mientras dura la llamada a la función."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _active
            with _lock:
                if _active:  # llamada anidida: ya se está midiendo
                    return func(*args, **kwargs)
                _active = True
            logger = ResourceLogger(label)
            logger.start()
            try:
                return func(*args, **kwargs)
            finally:
                logger.stop()
                with _lock:
                    _active = False

        return wrapper

    return decorator
