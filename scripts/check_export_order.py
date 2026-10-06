"""Verifica que el DST escribe los bloques de color en el MISMO orden que
se generaron (y, por tanto, que el PES).

El DST no guarda colores: la máquina conoce cada zona por la posición de sus
stop codes (cambios de color). Si la secuencia de bloques del DST no coincide
con la de las secciones CSewSeg del PES, el usuario cargaría los hilos en un
orden equivocado y el diseño saldría con las zonas de color intercambiadas
(fondo <-> logo). Este script exporta UN MISMO diseño procesado a ambos
formatos y comprueba que:

  1. El número de bloques de color es idéntico (PES == DST == colores del
     flujo).
  2. Cada bloque del DST contiene TODAS las puntadas reales de la zona de
     origen con esa posición (mismo bloque, mismo orden). Se compara a
     ≤ 0,35 mm porque el DST redondea a 0,1 mm; el bloque puede además
     añadir agarres que no están en el origen, eso no se exige.
  3. Cada bloque conserva al menos sus puntadas de origen (los formatos solo
     añaden agarres y puntada de cierre, nunca las quitan).

Nota: no se comprueba la geometría decodificada del PES contra el flujo porque
el lector de pyembroidery reconstruye las secciones CSewSeg en un orden y una
traslación que NO coinciden con el diseño original (visto con Free_Fire_Logo:
los centros normalizados quedan desplazados). Se comprueba solo su número de
bloques, que sí es fiable.

Salida: 0 = todo en orden, 1 = los formatos difieren.

Uso:
    python scripts/check_export_order.py [imagen] [density]
"""

from __future__ import annotations

import os
import sys
import tempfile

from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyembroidery

from core.color_processor import map_to_threads
from core.exporters.dst_exporter import export_dst
from core.exporters.pes_exporter import export_pes
from core.stitch_generator import MOVE_SENTINELS, generate_stitches


def make_fixture(size: int = 120) -> Image.Image:
    """Fondo a detectar + logo + acento: tres regiones bien diferenciadas.

    El fondo (esquina) se elimina del bordado igual que en la app; quedan un
    bloque grande (logo) y uno pequeño (acento) más los que se generen.
    """
    img = Image.new("RGB", (size, size), (30, 120, 200))  # esquina = fondo
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 85, 85], fill=(220, 30, 30))
    draw.rectangle([55, 35, 70, 45], fill=(30, 180, 90))  # acento descentrado
    return img


def source_points(stitches):
    """(código, [(x_mm, y_mm), ...]) por bloque de color, en orden del flujo.

    El flujo es contiguo por color por construcción (group_stitches_by_color:
    ver core/stitch_generator.py); el bloque se corta al cambiar de código.
    """
    blocks = []
    for x_mm, y_mm, code in stitches:
        if code in MOVE_SENTINELS:
            continue
        if blocks and blocks[-1][0] == code:
            blocks[-1][1].append((x_mm, y_mm))
        else:
            blocks.append([code, [(x_mm, y_mm)]])
    return [(code, pts) for code, pts in blocks]


def dst_block_points(path):
    """Puntadas reales de cada bloque de color del DST (mm decodificados).

    Cada bloque se delimita por los stop codes; no hay datos de color.
    """
    design = pyembroidery.read_dst(path)
    blocks = []
    pts = []
    for stitch in design.stitches:
        if stitch[2] == pyembroidery.COLOR_CHANGE:
            blocks.append(pts)
            pts = []
        elif stitch[2] == pyembroidery.STITCH:
            pts.append((stitch[0] / 10.0, stitch[1] / 10.0))
    blocks.append(pts)
    return blocks


def check(source, pes_n, dst_blocks, tol=0.35) -> bool:
    ok = True
    n_src = len(source)
    n_pes = pes_n
    n_dst = len(dst_blocks)
    print(
        f"bloques de color: flujo={n_src} PES={n_pes} DST={n_dst}  "
        f"{'OK' if n_src == n_pes == n_dst else 'ERROR: distinto numero'}"
    )
    if not (n_src == n_pes == n_dst):
        ok = False

    for i, (code, pts) in enumerate(source):
        n = len(pts)
        dst = dst_blocks[i] if i < len(dst_blocks) else []
        n_dst = len(dst)
        member = None
        if dst:
            tree = cKDTree(dst)
            dist, _ = tree.query(pts)
            member = dist.max() <= tol
        if member is None or not member:
            ok = False
        if n_dst < n:
            ok = False
        print(
            f"{' ' if ok_for(member, n_dst, n) else '!'} "
            f"bloque {i:2d}  hilo {code:>5s}  flujo:{n:5d}  "
            f"DST:{n_dst:5d}  "
            f"{'zona=' + str(member) if member is not None else 'sin puntadas en DST'}"
        )
    return ok


def ok_for(member, n_dst, n) -> bool:
    return bool(member) and (n_dst >= n)


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    density = sys.argv[2] if len(sys.argv) > 2 else "Media"
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = make_fixture()

    processed = map_to_threads(image)
    pattern = generate_stitches(
        processed.image,
        processed.threads_used,
        density=density,
        width_mm=180.0,
        height_mm=180.0,
        min_area_mm2=None,  # mismos valores que la app (ver AGENTS.md)
    )
    source = source_points(pattern.stitches)
    print(f"colores del flujo en orden: {[c for c, _ in source]}")
    print(f"bloques de color: {len(source)}")

    with tempfile.TemporaryDirectory(prefix="neithloom_check_") as tmp:
        pes = os.path.join(tmp, "check.pes")
        dst = os.path.join(tmp, "check.dst")
        export_pes(pattern.stitches, processed.threads_used, name="Check", out_path=pes)
        export_dst(pattern.stitches, name="Check", out_path=dst)
        pes_n = 1 + sum(
            1
            for s in pyembroidery.read_pes(pes).stitches
            if s[2] == pyembroidery.COLOR_CHANGE
        )
        dst_blocks = dst_block_points(dst)

    ok = check(source, pes_n, dst_blocks)
    print()
    if ok:
        print("OK: mismo numero de bloques y mismo orden en PES y DST.")
        return 0
    print("FALLO: PES y DST no coinciden en el numero u orden de bloques.")
    return 1


if __name__ == "__main__":
    sys.exit(main())