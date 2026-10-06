"""Benchmark del relleno orientado por región (+ contorno) del generador.

Compara sobre el mismo logo (Free Fire por defecto) cuatro modos del
generador: "plain" (relleno único horizontal, la versión previa),
"outline" (solo running stitch de contorno), "rotate" (solo orientación por
eje mayor) y "ambos" (lo que genera la GUI por defecto). Para cada modo:

  - puntadas/TRIM/JUMP del .pes (pyembroidery) y violaciones de hilo suelto
  - tiempo de generación (imagen -> puntadas) y RAM pico (tracemalloc)
  - diferencia de la vista previa respecto a "plain"
  - diagnóstico por región de color: nº de componentes, ángulo del eje
    mayor (momentos), comprobación de que el relleno sigue el eje cuando se
    rota (paralelo vs perpendicular) y nº de puntadas de contorno (si
    existen) posicionadas sobre el borde de la región.

Uso:
    python scripts/benchmark_region_fill.py ["ruta/al/logo.png"]
"""

import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyembroidery as pe
from PIL import Image

from core import color_processor, hoops, image_loader, stitch_generator
from core.exporters.pes_exporter import TRIM_MIN_JUMP_DISTANCE_MM, export_pes
from verify_pes import analyze

DEFAULT_COLORS = 8
DENSITY = "Media"
SEED = 42

MODES = (
    ("plain", dict(outline_running=False, rotate_fill=False)),
    ("outline", dict(outline_running=True, rotate_fill=False)),
    ("rotate", dict(outline_running=False, rotate_fill=True)),
    ("ambos", dict(outline_running=True, rotate_fill=True)),
)


def region_diagnosis(image, threads_used, hoop_mm):
    """Diagnóstico por región sobre la matriz (morph cerrada) del logo."""
    codes_before = stitch_generator.build_color_codes(image, threads_used)
    codes = stitch_generator._morph_close_codes(codes_before, 3)
    mmp = hoop_mm["mmp"]
    step = stitch_generator._step_mm(DENSITY)
    diag = []
    for code in stitch_generator._unique_color_codes(codes):
        mask = codes == code
        labels, ncomp = stitch_generator.ndimage.label(mask)
        if ncomp == 0:
            continue
        for i, sl in enumerate(stitch_generator.ndimage.find_objects(labels)):
            comp = labels[sl] == (i + 1)
            r0, c0 = sl[0].start, sl[1].start
            ang = stitch_generator._region_major_axis_deg(comp)
            n_px = int(comp.sum())
            if n_px < 4:
                continue
            # Fuerza de relleno rotado: los deltas dentro de una misma fila
            # scanline (distancia <= step) deben ser paralelos al eje.
            ev = stitch_generator._fill_region_tatami(
                comp, r0, c0, code, mmp, step, None, rotate=True
            )
            A = np.array([(x, y) for x, y, t, _r in ev if t == code])
            var = {"code": code, "px": n_px, "ang": ang, "rotated": bool(abs(ang) >= 5.0)}
            if len(A) > 4:
                import math

                d = np.diff(A, axis=0)
                nrm = np.hypot(d[:, 0], d[:, 1])
                small = nrm < 0.5
                if small.sum():
                    a = math.radians(ang)
                    ca = np.abs(
                        (d[small, 0] * math.cos(a) + d[small, 1] * math.sin(a))
                        / (nrm[small] + 1e-9)
                    ).mean()
                    cp = np.abs(
                        (d[small, 0] * -math.sin(a) + d[small, 1] * math.cos(a))
                        / (nrm[small] + 1e-9)
                    ).mean()
                    var["paralelo"] = float(ca)
                    var["perp"] = float(cp)
            ol = stitch_generator._outline_stitches(comp, r0, c0, code, mmp, 0.8)
            n_ol = sum(1 for e in ol if e[2] != stitch_generator.JUMP_SENTINEL)
            if n_ol:
                from scipy import ndimage as nd

                border = comp & ~nd.binary_erosion(comp)
                ok = 0
                for e in ol:
                    if e[2] == stitch_generator.JUMP_SENTINEL:
                        continue
                    r = int(round(e[1] / mmp)) - r0
                    c = int(round(e[0] / mmp)) - c0
                    if 0 <= r < comp.shape[0] and 0 <= c < comp.shape[1] and border[r, c]:
                        ok += 1
                var["outline_stitches"] = n_ol
                var["outline_on_border"] = ok
            diag.append(var)
    return codes, diag


def run_case(image, threads_used, hoop_mm, mode_kw, out_pes):
    tracemalloc.start()
    t0 = time.perf_counter()
    pattern = stitch_generator.generate_stitches(
        image,
        threads_used,
        density=DENSITY,
        width_mm=hoop_mm["width"],
        height_mm=hoop_mm["height"],
        morph_kernel=3,
        **mode_kw,
    )
    gen_sec = time.perf_counter() - t0
    _cur, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.perf_counter()
    export_pes(pattern.stitches, threads_used, name="Bench", out_path=out_pes)
    export_sec = time.perf_counter() - t1

    pattern2 = pe.read_pes(out_pes)
    totals, _long_runs, violations = analyze(
        list(pattern2.stitches), TRIM_MIN_JUMP_DISTANCE_MM, 6
    )
    return {
        "gen_sec": gen_sec,
        "export_sec": export_sec,
        "ram_peak": ram_peak,
        "pes_stitch": totals["stitch"],
        "pes_trim": totals["trim"],
        "pes_jump": totals["jump"],
        "violations": len(violations),
        "preview": pattern.preview,
    }


def preview_diff(a, b):
    if a is None or b is None:
        return None
    pa = np.asarray(a.convert("RGB")).astype(np.int16)
    pb = np.asarray(b.convert("RGB")).astype(np.int16)
    diff = np.abs(pa - pb)
    return {
        "mean_abs": float(diff.mean()),
        "diff_pixels": int((diff.sum(axis=2) > 0).sum()),
        "total_pixels": int(diff.shape[0] * diff.shape[1]),
    }


def main(argv=None):
    if not argv:
        argv = sys.argv[1:]
    logo_path = argv[0] if argv else os.path.expanduser(
        r"~\Downloads\Logo_of_Garena_Free_Fire.png"
    )

    hoop = hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX]
    hoop_mm = {"width": hoop["width_mm"], "height": hoop["height_mm"], "mmp": 1.0}

    img = image_loader.load_resized(logo_path)
    print(f"Logo: {logo_path}  ({img.width}x{img.height})")
    rng = np.random.default_rng(SEED)
    reduced = color_processor.reduce_palette(img, DEFAULT_COLORS, rng=rng)
    processed = color_processor.map_to_threads(reduced)
    image = processed.image
    threads_used = processed.threads_used
    hoop_mm["mmp"] = min(hoop_mm["width"] / image.width, hoop_mm["height"] / image.height)
    print(f"Colores mapeados: {len(threads_used)} | Bastidor: {hoop['name']} | mmp {hoop_mm['mmp']:.4f}")

    codes, regions = region_diagnosis(image, threads_used, hoop_mm)
    print(f"\nRegiones de color ({len(regions)}):")
    rot = [r for r in regions if r["rotated"]]
    for r in regions:
        tag = "ROTADO" if r["rotated"] else "horizontal"
        row = (
            f"  {r['code']}: px={r['px']:>6} ang={r['ang']:>6.1f} ({tag})"
        )
        if "paralelo" in r:
            row += f" | relleno: paralelo={r['paralelo']:.2f} perp={r['perp']:.2f}"
            if r["rotated"]:
                row += " -> " + ("SIGUE EJE" if r["paralelo"] > r["perp"] else "NO")
        if "outline_stitches" in r:
            row += (f" | contorno: {r['outline_stitches']} puntadas"
                    f" (sobre borde {r['outline_on_border']}/{r['outline_stitches']})")
        print(row)
    print(f"Regiones no horizontales/verticales que se rotan: {len(rot)}/{len(regions)}")

    results = {}
    for tag, kw in MODES:
        out_pes = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"bench_region_{tag}.pes"
        )
        results[tag] = run_case(image, threads_used, hoop_mm, kw, out_pes)
        r = results[tag]
        print(
            f"--- modo {tag:<8} -> STITCH {r['pes_stitch']} | TRIM {r['pes_trim']} | "
            f"JUMP {r['pes_jump']} | violac {r['violations']} | gen {r['gen_sec']*1000:.0f} ms | "
            f"export {r['export_sec']*1000:.0f} ms | RAM {r['ram_peak']/1e6:.1f} MB"
        )

    plain = results["plain"]
    print("\n=== comparacion vs plain ===")
    print(
        f"{'modo':<9}{'TRIM':>8}{'JUMP':>8}{'STITCH':>9}{'violac':>8}"
        f"{'gen_ms':>9}{'RAM_MB':>9}{'preview_diff_px':>17}"
    )
    for tag, _ in MODES:
        r = results[tag]
        pd = preview_diff(r["preview"], plain["preview"])
        px = "-" if pd is None else str(pd["diff_pixels"])
        print(
            f"{tag:<9}{r['pes_trim']:>8}{r['pes_jump']:>8}{r['pes_stitch']:>9}"
            f"{r['violations']:>8}{r['gen_sec']*1000:>9.0f}{r['ram_peak']/1e6:>9.1f}"
            f"{px:>17}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())