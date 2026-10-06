"""Diagnóstico de los cortes de hilo (TRIM) dentro de una misma letra.

Reconstruye el flujo exacto de la GUI sobre un logo (Free Fire por defecto),
genera las puntadas y analiza el flujo YA enrutado por `order_tatami_rows`:

  - cada centinela de movimiento del flujo final (`JUMP_SENTINEL` o
    `TRIM_SENTINEL`) corresponde a una decisión del enrutado (en
    `_route_one_color`, por distancia o por salida de la máscara: aguja abajo
    solo si se cumplen ambos criterios);
  - un `TRIM_SENTINEL` se convierte SIEMPRE en TRIM en el exportador PES
    (cruce de hueco/contador real), mientras que un `JUMP_SENTINEL` solo si su
    distancia supera `TRIM_MIN_JUMP_DISTANCE_MM` (el valor configurado en el
    exportador);

y clasifica cada TRIM según si los dos puntos del salto pertenecen a la
MISMA componente conexa del color o a componentes DISTINTAS (mapeando las
coordenadas mm de vuelta a la matriz de códigos tras morph+min_area, que es
la que rige el relleno).

Uso:
    python scripts/diagnose_trim_cuts.py ["ruta/al/logo"] [--colors N]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from core import color_processor, hoops, image_loader, stitch_generator  # noqa: E402
from core.exporters.pes_exporter import TRIM_MIN_JUMP_DISTANCE_MM  # noqa: E402

DENSITY = "Media"
SEED = 42


def build_labels(codes):
    """Por color: idx de la componente conexa (0-based, -1 = fondo/None)."""
    mapping = {}
    for code in stitch_generator._unique_color_codes(codes):
        labels, n = ndimage.label(codes == code)
        idx = np.full(codes.shape, -1, dtype=np.int32)
        idx[labels > 0] = labels[labels > 0] - 1
        mapping[code] = (n, idx)
    return mapping


def component_of(labels, code, x_mm, y_mm, mmp):
    entry = labels.get(code)
    if entry is None:
        return None
    _, idx = entry
    r = int(round(y_mm / mmp))
    c = int(round(x_mm / mmp))
    if 0 <= r < idx.shape[0] and 0 <= c < idx.shape[1]:
        return int(idx[r, c])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logo",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Free_Fire_Logo.jpg",
        ),
    )
    parser.add_argument("--colors", type=int, default=8)
    args = parser.parse_args(argv)

    hoop = hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX]
    width_mm, height_mm = hoop["width_mm"], hoop["height_mm"]

    img = image_loader.load_resized(args.logo)
    rng = np.random.default_rng(SEED)
    processed = color_processor.map_to_threads(
        color_processor.reduce_palette(img, args.colors, rng=rng)
    )
    image, threads_used = processed.image, processed.threads_used
    mmp = min(width_mm / image.width, height_mm / image.height)
    print(f"Logo: {args.logo}  ({img.width}x{img.height}) | colores {len(threads_used)}")
    print(f"Bastidor: {hoop['name']} | mmp {mmp:.4f} mm/px")

    pattern = stitch_generator.generate_stitches(
        image,
        threads_used,
        density=DENSITY,
        width_mm=width_mm,
        height_mm=height_mm,
        morph_kernel=3,
        outline_running=True,
        rotate_fill=True,
        min_area_mm2=0.5,
        codes_median=None,
    )
    stitches = pattern.stitches

    # Matriz REAL (morph + drop_tiny) que decide el relleno: la misma con la
    # que trabajó generate_stitches, porque los puntos mm se mapean a píxeles.
    codes = stitch_generator.build_color_codes(image, threads_used)
    codes = stitch_generator._morph_close_codes(codes, 3)
    codes = stitch_generator._drop_tiny_regions(codes, mmp, 0.5)
    labels = build_labels(codes)
    n_comp_total = sum(n for n, _ in labels.values())

    # Recorrido exacto del flujo final: cada centinela de movimiento de
    # `_route_one_color` marca una decisión de levantar la aguja. Un
    # TRIM_SENTINEL (salto forzado por la máscara de fondo) se convierte
    # SIEMPRE en TRIM en el exportador; un JUMP_SENTINEL solo si su distancia
    # supera TRIM_MIN_JUMP_DISTANCE_MM. Se clasifica por las componentes de los
    # dos puntos reales que conecta el salto.
    cuts = []  # (dist_mm, (x0,y0,comp0), (x1,y1,comp1), sentinel, forzado)
    prev = None  # (x, y, comp) real anterior
    pending_jump = False
    forced = False
    n_stitch = 0
    n_sentinel = 0
    n_forced = 0
    for x, y, code in stitches:
        if code in stitch_generator.MOVE_SENTINELS:
            pending_jump = True
            forced = code == stitch_generator.TRIM_SENTINEL
            if forced:
                n_forced += 1
            continue
        comp = component_of(labels, code, x, y, mmp)
        n_stitch += 1
        if prev is not None:
            d = math.hypot(x - prev[0], y - prev[1])
            if pending_jump:
                n_sentinel += 1
                if forced or d > TRIM_MIN_JUMP_DISTANCE_MM:
                    cuts.append(
                        (d, (prev[0], prev[1], prev[2]), (x, y, comp), True, forced)
                    )
            elif d > TRIM_MIN_JUMP_DISTANCE_MM:
                # Sin sentinel: puntadas consecutivas con aguja abajo (hebra
                # continua). Se registra como contraste, no como corte.
                cuts.append(
                    (d, (prev[0], prev[1], prev[2]), (x, y, comp), False, False)
                )
        prev = (x, y, comp)
        pending_jump = False
        forced = False

    n_cuts = sum(1 for c in cuts if c[3])
    n_within = sum(
        1
        for _, a, b, s, _f in cuts
        if s and a[2] == b[2] and a[2] is not None and b[2] is not None
    )
    n_cross = n_cuts - n_within

    print(f"\nPuntadas reales: {n_stitch} | componentes conexas (fill): {n_comp_total}")
    print(f"Saltos (centinelas) del enrutado: {n_sentinel}")
    print(f"  - forzados por máscara (TRIM obligatorio): {n_forced}")
    print(
        f"Cortes en el exportador (forzados o salto > "
        f"{TRIM_MIN_JUMP_DISTANCE_MM} mm): {n_cuts}"
    )
    print(f"  - dentro de la MISMA componente: {n_within}")
    print(f"  - entre componentes DISTINTAS:   {n_cross}")

    for tag, arr in (
        ("MISMA componente", [c for c in cuts if c[3] and c[1][2] == c[2][2]]),
        ("componentes distintas", [c for c in cuts if c[3] and not (c[1][2] == c[2][2])]),
    ):
        if not arr:
            print(f"\n[{tag}] sin cortes")
            continue
        ds = sorted(c[0] for c in arr)
        print(
            f"\n[{tag}] {len(arr)} cortes | "
            f"dist min {ds[0]:.1f} mediana {ds[len(ds)//2]:.1f} max {ds[-1]:.1f} mm"
        )
        for d, a, b, _s, forced in arr[:10]:
            print(
                f"   {d:5.1f} mm{' [forzado]' if forced else '           '}  "
                f"({a[0]:6.1f},{a[1]:6.1f}) -> ({b[0]:6.1f},{b[1]:6.1f})"
            )
        if len(arr) > 10:
            hist = np.histogram(ds, bins=6)[0]
            print(f"   ... y {len(arr) - 10} más | histograma {hist.tolist()}")

    # Contraste: gaps largos con aguja abajo (hebra corrida, no son cortes).
    sewn_gaps = sorted(c[0] for c in cuts if not c[3])
    if sewn_gaps:
        print(
            f"\n[contraste] gaps aguja-abajo > {TRIM_MIN_JUMP_DISTANCE_MM} mm "
            f"(conectados): {len(sewn_gaps)} | mediana {sewn_gaps[len(sewn_gaps)//2]:.1f} mm"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())