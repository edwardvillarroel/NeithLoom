"""Genera un .pes de prueba con dos zonas separadas del mismo color.

Sirve para comprobar que el exportador corta el hilo (TRIM) al saltar entre
zonas no contiguas del mismo color: la imagen de prueba son dos cuadrados
rojos separados por una zona blanca, procesados con el pipeline real
(mapeo de color + generación de puntadas + exportador PES).

Uso:
    python scripts/make_test_pes.py [destino.pes]
"""

import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_processor import map_to_threads
from core.exporters.pes_exporter import export_pes
from core.stitch_generator import MOVE_SENTINELS, generate_stitches


def make_test_image(size: int = 100) -> Image.Image:
    """Fondo blanco con dos cuadrados rojos separados (zonas no contiguas)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 38, 38], fill=(255, 0, 0))
    draw.rectangle([62, 62, 92, 92], fill=(255, 0, 0))
    return img


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "test_separated_zones.pes"
    processed = map_to_threads(make_test_image())
    pattern = generate_stitches(
        processed.image,
        processed.threads_used,
        density="Media",
        width_mm=50.0,
        height_mm=50.0,
    )
    export_pes(
        pattern.stitches,
        processed.threads_used,
        name="TestZones",
        out_path=out,
    )
    real = sum(1 for s in pattern.stitches if s[2] not in MOVE_SENTINELS)
    print(f"OK -> {out} ({real} puntadas reales en el flujo)")


if __name__ == "__main__":
    main()