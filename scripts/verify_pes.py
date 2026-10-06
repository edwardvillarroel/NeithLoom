"""Verificador de archivos .pes: cuentas JUMP/TRIM y saltos largos sin corte.

Diagnostica el problema de "hilo suelto bajo la tela": el programa debe
cortar el hilo (TRIM) antes de saltar entre zonas separadas del mismo color.
Referencia medida con pyembroidery: NeithLoom.pes tenía 3150 JUMP y 0 TRIM,
mientras que un archivo de un digitalizador externo tenía 6 JUMP y 3 TRIM.

Por defecto usa el mismo umbral que el exportador (TRIM_MIN_JUMP_DISTANCE_MM)
para decidir que un salto corresponde a una zona separada. Código de salida:
0 = todo correcto; 1 = hay al menos un salto largo sin TRIM cercano (regresión).

Uso:
    python scripts/verify_pes.py archivo.pes [--threshold-mm 6.0] [--trim-window 6]

Nota: las coordenadas PES que devuelve pyembroidery están en décimas de
milímetro, igual que las escribe core/exporters/pes_exporter.py (x10).
"""

import argparse
import math
import os
import sys

import pyembroidery as pe

# Unidades nativas PES = décimas de milímetro.
_COORD_UNIT_MM = 0.1


def _default_threshold_mm():
    """Umbral del exportador, si se puede importar; si no, 6.0 mm."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.exporters.pes_exporter import (  # noqa: PLC0415
            TRIM_MIN_JUMP_DISTANCE_MM,
        )

        return TRIM_MIN_JUMP_DISTANCE_MM
    except Exception:
        return 6.0


def _kind(command: int) -> int:
    return command & 0xFF


def analyze(commands, threshold_mm, trim_window):
    """Recorre los comandos y calcula métricas de saltos sin TRIM cercano.

    Un "salto largo" es una racha de comandos JUMP consecutivos (entre dos
    comandos que no son JUMP) cuya distancia total supera `threshold_mm`.
    La racha inicial previa a la primera puntada (posicionamiento de la aguja
    desde el reposo) no cuenta: no hay hilo que cortar.
    """
    threshold = threshold_mm / _COORD_UNIT_MM
    totals = {"stitch": 0, "jump": 0, "trim": 0, "color_change": 0, "other": 0}
    long_runs = []   # dicts: run[0]=start_idx, run[1]=end_idx, run[2]=dist_mm
    violations = []  # long_runs sin TRIM cerca

    run = None  # (x0, y0, start_idx) con x0,y0 = posición de la aguja ANTES del run
    px = 0.0
    py = 0.0
    sewn = False
    n = len(commands)

    def flush(end_idx):
        nonlocal run
        if run is None:
            return
        x0, y0, start_idx = run
        x1, y1 = commands[end_idx][0], commands[end_idx][1]
        dist = math.hypot(x1 - x0, y1 - y0)
        run_obj = None
        if dist > threshold and sewn:
            lo = max(0, start_idx - trim_window)
            hi = min(n - 1, end_idx + trim_window)
            has_trim = any(_kind(commands[k][2]) == pe.TRIM for k in range(lo, hi + 1))
            run_obj = (start_idx, end_idx, dist * _COORD_UNIT_MM, has_trim)
            long_runs.append(run_obj)
            if not has_trim:
                violations.append(run_obj)
        run = None

    for i, (x, y, raw) in enumerate(commands):
        kind = _kind(raw)
        if kind == pe.STITCH:
            flush(i - 1)
            sewn = True
            totals["stitch"] += 1
        elif kind == pe.JUMP:
            if run is None:
                run = (px, py, i)
            totals["jump"] += 1
        elif kind == pe.TRIM:
            flush(i - 1)
            totals["trim"] += 1
        elif kind == pe.COLOR_CHANGE:
            flush(i - 1)
            totals["color_change"] += 1
        elif kind == pe.END or kind == pe.STOP:
            flush(i - 1)
        else:
            totals["other"] += 1
        px, py = x, y
    flush(n - 1)

    return totals, long_runs, violations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pes_file", help="archivo .pes a analizar")
    parser.add_argument(
        "--threshold-mm",
        type=float,
        default=_default_threshold_mm(),
        help="distancia mínima (mm) para considerar un salto una zona separada",
    )
    parser.add_argument(
        "--trim-window",
        type=int,
        default=6,
        help="número de comandos alrededor de un salto para considerar un TRIM 'cercano'",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=20,
        help="cuántas rachas detallar en pantalla (resto se resume)",
    )
    args = parser.parse_args(argv)

    pattern = pe.read_pes(args.pes_file)
    commands = list(pattern.stitches)
    totals, long_runs, violations = analyze(
        commands, args.threshold_mm, args.trim_window
    )

    print(f"Archivo: {args.pes_file}")
    print(f"Puntadas (STITCH):        {totals['stitch']}")
    print(f"JUMP:                     {totals['jump']}")
    print(f"TRIM:                     {totals['trim']}")
    print(f"Cambios de color:         {totals['color_change']}")
    print(f"Umbral de 'zona separada': {args.threshold_mm} mm")
    print(f"Saltos largos (>umbral):  {len(long_runs)}")
    for start, end, dist_mm, has_trim in long_runs[: args.max_list]:
        print(
            f"  - comandos [{start}..{end}]: {dist_mm:6.1f} mm, "
            f"TRIM cerca: {'SI' if has_trim else 'NO'}"
        )
    hidden = len(long_runs) - min(len(long_runs), args.max_list)
    if hidden > 0:
        print(f"  ... y {hidden} saltos adicionales (usa --max-list para verlos todos)")
    if violations:
        print(f"\nSaltos largos SIN TRIM: {len(violations)} (BUG: hilo suelto bajo la tela)")
        print("RESULTADO: ERROR")
        return 1
    print("\nSaltos largos SIN TRIM: 0")
    print("RESULTADO: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())