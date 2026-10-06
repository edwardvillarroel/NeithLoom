"""Benchmark de la limpieza morfológica por color del generador de puntadas.

Reproduce el flujo de la GUI (reduce_palette -> map_to_threads ->
generate_stitches -> export_pes) sobre un logo real y compara, para cada
kernel de closing (None = actual, 3, 5):

  - puntadas/TRIM/JUMP del .pes resultante (leído con pyembroidery)
  - saltos largos sin TRIM (violaciones de hilo suelto)
  - componentes conexas por color antes/después del closing (fidelidad:
    no perder ni fusionar islas genuinas)
  - tramos por color antes/después (prueba de que caen los fragmentos falsos)
  - tiempo de generación (imagen -> puntadas) y RAM pico (tracemalloc)
  - diferencia de la vista previa respecto a sin closing

Uso:
    python scripts/benchmark_mask_cleanup.py "ruta/al/logo.png"
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
KERNELS = (None, 3, 5)
SEED = 42


def run_case(image, threads_used, kernel, hoop_mm, out_pes):
    """Genera puntadas + exporta, midiendo tiempo y RAM del generador."""
    codes_before = stitch_generator.build_color_codes(image, threads_used)
    codes_after = (
        stitch_generator._morph_close_codes(codes_before, kernel)
        if kernel
        else codes_before
    )
    diagnosis = stitch_generator.diagnose_color_masks(
        codes_before,
        codes_after,
        kernel_size=kernel or 3,
        mmp=hoop_mm["mmp"],
        step=stitch_generator._step_mm(DENSITY),
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    pattern = stitch_generator.generate_stitches(
        image,
        threads_used,
        density=DENSITY,
        width_mm=hoop_mm["width"],
        height_mm=hoop_mm["height"],
        morph_kernel=kernel,
        outline_running=False,
        rotate_fill=False,
    )
    gen_sec = time.perf_counter() - t0
    _cur, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.perf_counter()
    export_pes(pattern.stitches, threads_used, name="Bench", out_path=out_pes)
    export_sec = time.perf_counter() - t1

    pattern2 = pe.read_pes(out_pes)
    totals, long_runs, violations = analyze(
        list(pattern2.stitches), TRIM_MIN_JUMP_DISTANCE_MM, 6
    )

    real = sum(1 for s in pattern.stitches if s[2] not in stitch_generator.MOVE_SENTINELS)
    return {
        "kernel": kernel,
        "diagnosis": diagnosis,
        "gen_sec": gen_sec,
        "export_sec": export_sec,
        "ram_peak": ram_peak,
        "real_stitches": real,
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


def report_diagnosis(diagnosis, kernel=3):
    print("  componentes conexas (label) y tramos (scanline) por color:")
    min_area = 9 if (kernel or 0) <= 3 else 16
    for code, d in diagnosis.items():
        lost_flag = ""
        if d["lost_max_area_px"] >= min_area:
            lost_flag = f"  <-- ISLA GENUINA PERDIDA (area {d['lost_max_area_px']}px)"
        print(
            f"    {code}: comps {d['components_before']}->{d['components_after']}"
            f" | perdidas {d['lost_components']}"
            f" (max area {d['lost_max_area_px']}px){lost_flag} | "
            f"tramos {d['spans_before']}->{d['spans_after']}"
            f" (delta {d['spans_delta']:+d})"
        )


def main(argv=None):
    if not argv:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print(__doc__)
        return 2
    logo_path = argv[0]

    hoop = hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX]
    hoop_mm = {
        "width": hoop["width_mm"],
        "height": hoop["height_mm"],
        "mmp": 1.0,
    }

    img = image_loader.load_resized(logo_path)
    print(f"Logo: {logo_path}  ({img.width}x{img.height})")
    rng = np.random.default_rng(SEED)
    reduced = color_processor.reduce_palette(img, DEFAULT_COLORS, rng=rng)
    processed = color_processor.map_to_threads(reduced)
    image = processed.image
    threads_used = processed.threads_used
    print(f"Colores mapeados: {len(threads_used)} | Bastidor: {hoop['name']}")

    hoop_mm["mmp"] = min(
        hoop_mm["width"] / image.width, hoop_mm["height"] / image.height
    )

    results = {}
    for k in KERNELS:
        tag = "none" if k is None else str(k)
        out_pes = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"bench_morph_k{k}.pes"
        )
        results[k] = run_case(image, threads_used, k, hoop_mm, out_pes)
        r = results[k]
        print(f"\n--- kernel closing = {tag} ---")
        print(f"  .pes:   {out_pes}")
        print(
            f"  puntadas reales flujo: {r['real_stitches']} | "
            f"PES: STITCH {r['pes_stitch']} | TRIM {r['pes_trim']} | "
            f"JUMP {r['pes_jump']}"
        )
        print(f"  saltos largos sin TRIM: {r['violations']}")
        print(
            f"  tiempo: generacion {r['gen_sec']*1000:.0f} ms | "
            f"export {r['export_sec']*1000:.0f} ms | "
            f"RAM pico {r['ram_peak']/1e6:.1f} MB"
        )
        report_diagnosis(r["diagnosis"], kernel=r["kernel"])

    none_r = results[None]
    print("\n=== comparacion vs sin closing (None) ===")
    print(
        f"{'kernel':<6}{'TRIM':>8}{'JUMP':>8}{'STITCH':>9}{'violac':>8}"
        f"{'gen_ms':>9}{'RAM_MB':>9}{'preview_diff_px':>17}"
    )
    for k in KERNELS:
        r = results[k]
        pd = preview_diff(r["preview"], none_r["preview"])
        px = "-" if pd is None else str(pd["diff_pixels"])
        print(
            f"{str(k):<6}{r['pes_trim']:>8}{r['pes_jump']:>8}{r['pes_stitch']:>9}"
            f"{r['violations']:>8}{r['gen_sec']*1000:>9.0f}{r['ram_peak']/1e6:>9.1f}"
            f"{px:>17}"
        )
    if pd is not None:
        print(
            f"  preview diff (None vs 3): {pd['mean_abs']:.4f} media por canal, "
            f"{100.0*pd['diff_pixels']/pd['total_pixels']:.2f}% de pixeles"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())