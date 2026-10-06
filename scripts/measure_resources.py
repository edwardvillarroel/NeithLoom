"""Ejecuta NeithLoom de principio a fin con las opciones más pesadas y deja el
CSV de recursos que `core.resource_logger` genera automáticamente alrededor
de `stitch_generator.generate_stitches` (la etapa de conversión imagen ->
matriz de puntadas).

Opciones más pesadas usadas (las que ofrece el programa hoy):

- Imagen de mayor resolución del repositorio (auto, o la que se pase como
  argumento). El programa la reduce a `image_loader.MAX_DIMENSION` igual que
  al cargarla desde la interfaz.
- "Máximo de colores" al tope de `color_processor.COLOR_OPTIONS` (12): se
  elige con el combo del paso 2 (el mismo control que vería el usuario).
- Bastidor de mayor área de `hoops.HOOPS` (25 x 25 cm).
- Densidad "Alta", la más densa de `stitch_generator.DENSITY_OPTIONS`.

No cambia ninguna lógica de procesamiento: solo selecciona opciones y
conduce la propia `MainWindow` por sus cuatro pasos.

Uso:
    .\\venv\\Scripts\\python.exe scripts\\measure_resources.py [imagen]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from core import color_processor, hoops, stitch_generator
from core.resource_logger import LOG_DIR
from ui.main_window import LAST_SAVE_DIR_KEY, MainWindow

TICK_MS = 200
DEADLINE_S = 900.0
EXPORT_DIR = ROOT / "scripts"
EXPORT_NAME = "bench_recursos"
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg")


def biggest_image(explicit: str | None) -> Path:
    """Imagen de mayor número de píxeles del repo (sin contar el venv)."""
    if explicit:
        return Path(explicit).resolve()
    best: Path | None = None
    best_pixels = -1
    for base in (ROOT, ROOT / "images"):
        if not base.is_dir():
            continue
        for pattern in IMAGE_PATTERNS:
            for path in base.glob(pattern):
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                except Exception:  # noqa: BLE001
                    continue
                if width * height > best_pixels:
                    best_pixels = width * height
                    best = path
    if best is None:
        raise SystemExit("No se encontró ninguna imagen de prueba en el repo.")
    return best.resolve()


def main(argv: list[str]) -> int:
    image_path = biggest_image(argv[0] if argv else None)
    max_colors = max(color_processor.COLOR_OPTIONS)
    hoop = max(hoops.HOOPS, key=lambda item: item["width_mm"] * item["height_mm"])
    density = stitch_generator.DENSITY_OPTIONS[-1]

    app = QApplication(sys.argv)
    window = MainWindow()
    # Antes se fijaba en tiempo de ejecución `color_processor.DEFAULT_COLORS`
    # porque el combo no existía; ahora se elige con el mismo control que
    # vería el usuario (el cambio llega a `process_image` vía currentData).
    max_colors_index = window._max_colors_combo.findData(max_colors)
    if max_colors_index < 0:
        raise SystemExit(
            f"{max_colors} no es una opción de COLOR_OPTIONS: "
            f"{color_processor.COLOR_OPTIONS}"
        )
    window._max_colors_combo.setCurrentIndex(max_colors_index)
    window.show()

    before = set(LOG_DIR.glob("recursos_*.csv")) if LOG_DIR.is_dir() else set()
    stored_save_dir = QSettings().value(LAST_SAVE_DIR_KEY)

    state: dict = {"phase": "load", "t0": time.perf_counter(), "error": None}

    print("=== Medición de recursos con las opciones más pesadas ===")
    print(f"imagen      : {image_path}")
    print(f"colores     : {max_colors} (max de COLOR_OPTIONS)")
    print(
        f"bastidor    : {hoop['name']} "
        f"({hoop['width_mm']:.0f} x {hoop['height_mm']:.0f} mm)"
    )
    print(f"densidad    : {density}")

    def stop(reason: str, code: int) -> None:
        timer.stop()
        window._usb_timer.stop()
        gallery = getattr(window, "_gallery_thread", None)
        if gallery is not None and gallery.isRunning():
            gallery.requestInterruption()
            gallery.wait(5000)
        # `export_image` recuerda la carpeta en QSettings: se restaura para
        # no dejar la configuración del usuario cambiada por esta corrida.
        settings = QSettings()
        if stored_save_dir is None:
            settings.remove(LAST_SAVE_DIR_KEY)
        else:
            settings.setValue(LAST_SAVE_DIR_KEY, stored_save_dir)
        new_csvs = sorted(set(LOG_DIR.glob("recursos_*.csv")) - before)
        print(f"\n=== Fin: {reason} ===")
        for csv_path in new_csvs:
            print(f"CSV generado: {csv_path}")
        app.exit(code)

    def fail(message: str) -> None:
        state["error"] = message
        stop(message, 1)

    def on_tick() -> None:
        if state["error"] is not None:
            return
        elapsed = time.perf_counter() - state["t0"]
        if elapsed > DEADLINE_S:
            fail(f"timeout de {DEADLINE_S:.0f}s en la fase '{state['phase']}'")
            return

        modal = QApplication.activeModalWidget()
        if isinstance(modal, QMessageBox):
            fail(f"dialogo de error: {modal.text()}")
            return

        phase = state["phase"]
        if phase == "load":
            window.load_image(str(image_path))
            window.process_image()
            state["phase"] = "wait_colors"
        elif phase == "wait_colors":
            if window._processed_result is not None:
                window._on_hoop_card_clicked(hoop)
                index = window._density_combo.findData(density)
                window._density_combo.setCurrentIndex(index)
                # La única pregunta de la interfaz (guía de zonas) se salta
                # para que la corrida tenga exactamente una generación.
                window._guide_asked = True
                window.generate_stitches()
                state["phase"] = "wait_stitches"
        elif phase == "wait_stitches":
            if window._stitch_pattern is not None and window._stitch_thread is None:
                window._selected_save_dir = str(EXPORT_DIR)
                window._design_name = EXPORT_NAME
                window.export_image()
                state["phase"] = "wait_export"
        elif phase == "wait_export":
            if window._export_thread is None:
                state["phase"] = "close_dialog"
        elif phase == "close_dialog":
            if modal is not None:
                modal.close()
            elif window._export_status_label.text().startswith("Guardado"):
                stop("pipeline completo", 0)
            else:
                stop("exportación no confirmada", 1)

    timer = QTimer()
    timer.setInterval(TICK_MS)
    timer.timeout.connect(on_tick)
    timer.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
