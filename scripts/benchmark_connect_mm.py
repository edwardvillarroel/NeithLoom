"""Benchmark del umbral de no-cortar (`connect_mm`) del enrutado.

Reproduce el flujo exacto de la GUI sobre el logo de Free Fire (o cualquier
otra imagen) y entrega, para cada valor de `connect_mm`:

  - conteos STITCH/JUMP/TRIM del .pes exportado (pyembroidery, umbral configurado en el exportador)
  - tiempo de máquina estimado con la fórmula del usuario:
        STITCH*0.09 s + TRIM*2.5 s
  - preview byte-idéntica o no respecto a la generada con `connect_mm=None`
    (el relleno y el contorno NO cambian con este parámetro; solo cambia la
    decisión de levantar la aguja entre puntadas del mismo color)

Uso:
    python scripts/benchmark_connect_mm.py ["ruta/al/logo"] [--colors N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pyembroidery as pe  # noqa: E402

from core import color_processor, hoops, image_loader, stitch_generator  # noqa: E402
from core.exporters.pes_exporter import TRIM_MIN_JUMP_DISTANCE_MM, export_pes  # noqa: E402
from verify_pes import analyze  # noqa: E402

DENSITY = "Media"
SEED = 42
THRESHOLD_MM = TRIM_MIN_JUMP_DISTANCE_MM  # mismo umbral que el exportador (zona separada)


def run_case(image, threads_used, hoop_mm, connect_mm, out_pes):
    t0 = time.perf_counter()
    pattern = stitch_generator.generate_stitches(
        image,
        threads_used,
        density=DENSITY,
        width_mm=hoop_mm["width"],
        height_mm=hoop_mm["height"],
        morph_kernel=3,
        outline_running=True,
        rotate_fill=True,
        min_area_mm2=0.5,
        codes_median=None,
        connect_mm=connect_mm,
    )
    gen_sec = time.perf_counter() - t0
    export_pes(pattern.stitches, threads_used, name="Bench", out_path=out_pes)
    pattern2 = pe.read_pes(out_pes)
    totals, _, violations = analyze(list(pattern2.stitches), THRESHOLD_MM, 6)
    return pattern, totals, len(violations), gen_sec


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
    hoop_mm = {"width": hoop["width_mm"], "height": hoop["height_mm"], "mmp": 1.0}

    img = image_loader.load_resized(args.logo)
    rng = np.random.default_rng(SEED)
    processed = color_processor.map_to_threads(
        color_processor.reduce_palette(img, args.colors, rng=rng)
    )
    image, threads_used = processed.image, processed.threads_used
    hoop_mm["mmp"] = min(hoop_mm["width"] / image.width, hoop_mm["height"] / image.height)
    print(f"Logo: {args.logo}  ({img.width}x{img.height}) | bastidor {hoop['name']}")
    print(f"Colores: {len(threads_used)} | mmp {hoop_mm['mmp']:.4f} mm/px")
    print(f"Umbral PES de zona separada: {THRESHOLD_MM} mm | semilla {SEED}")

    values = [None, 2.5, 3.0, 4.5, 6.0, 8.0]
    print(f"\n{'connect_mm':>11}{'STITCH':>9}{'JUMP':>7}{'TRIM':>7}"
          f"{'maq_min':>10}{'violac':>8}{'gen_s':>8}{'prev_igual':>11}")
    base = None
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    for v in values:
        tag = "adapt" if v is None else f"{v:g}".replace(".", "_")
        out_pes = os.path.join(scripts_dir, f"bench_connect_{tag}.pes")
        pattern, totals, nviol, gen_sec = run_case(
            image, threads_used, hoop_mm, v, out_pes
        )
        stitch = totals["stitch"]
        trim = totals["trim"]
        jump = totals["jump"]
        est_min = stitch * 0.09 / 60 + trim * 2.5 / 60
        same = "-"
        if base is not None:
            same = "SI" if np.array_equal(
                np.asarray(base.preview), np.asarray(pattern.preview)
            ) else "NO"
        else:
            base = pattern
        disp = "adapt." if v is None else f"{v:g}"
        print(
            f"{disp:>11}{stitch:>9}{jump:>7}{trim:>7}"
            f"{est_min:>10.1f}{nviol:>8}{gen_sec:>8.2f}{same:>11}"
            f"   -> {os.path.basename(out_pes)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())