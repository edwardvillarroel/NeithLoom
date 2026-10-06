"""Regenera `images/icon.png` a partir del diseño original en la misma carpeta.

El icono original era un dibujo oscuro pegado arriba a la izquierda de un
lienzo blanco (226x256 px dentro de 256x256) y a 16/32 px salía descentrado
y achatado sobre un fondo blanco. Este script recorta el motivo, conserva su
proporción y lo centra sobre un lienzo cuadrado transparente con aire (85% de
la altura), que es el formato que Windows pinta bien en barra de tareas,
título y Alt+Tab.

Uso:
    .\\venv\\Scripts\\python.exe scripts\\make_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "images" / "icon.png"
CANVAS = 256
FILL = 0.85


def main() -> int:
    im = Image.open(SRC)
    arr = np.asarray(im).astype(int)
    if arr.ndim == 3 and arr.shape[2] == 4 and (arr[:, :, 3] == 0).any():
        print("El icono ya está centrado y con fondo transparente; no hago nada.")
        print("Para regenerarlo, dale primero el PNG original con fondo blanco.")
        return 0
    im = im.convert("RGB")
    arr = np.asarray(im).astype(int)
    # Píxel del dibujo: cualquier canal > 8 por debajo del blanco (incluye el
    # antialiasing del borde). Es el recorte que llevaría el motivo al lienzo.
    ink = (255 - arr).max(axis=2) > 8
    ys, xs = np.where(ink)
    art = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    width, height = art.size
    if width == 0 or height == 0:
        print("No se encontró el motivo: ¿el icono ya es transparente?")
        return 1

    target_h = int(round(CANVAS * FILL))
    target_w = max(1, int(round(width * target_h / height)))
    art = art.resize((target_w, target_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(art, ((CANVAS - target_w) // 2, (CANVAS - target_h) // 2))
    canvas.save(SRC)

    print(f"motivo original: {width}x{height} px")
    print(f"icono nuevo    : {CANVAS}x{CANVAS} RGBA, motivo {target_w}x{target_h} centrado")
    return 0


if __name__ == "__main__":
    sys.exit(main())