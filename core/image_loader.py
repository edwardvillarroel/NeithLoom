from PIL import Image
from PIL import ImageOps
from pathlib import Path
from typing import Iterable, List

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_DIMENSION = 1000
THUMB_SIZE = (120, 120)
GALLERY_LIMIT = 100
USER_DIR_NAMES = ("Pictures", "Downloads", "Desktop", "Documents")


def default_search_dirs() -> List[Path]:
    home = Path.home()
    return [home / name for name in USER_DIR_NAMES if (home / name).is_dir()]


def list_recent_images(
    dirs: Iterable[Path] | None = None, limit: int = GALLERY_LIMIT
) -> List[str]:
    """Devuelve rutas de imágenes .jpg/.jpeg/.png ordenadas por fecha de modificación (más recientes primero)."""
    if dirs is None:
        dirs = default_search_dirs()

    entries = []
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        try:
            for child in directory.iterdir():
                if not child.is_file():
                    continue
                if child.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                try:
                    mtime = child.stat().st_mtime
                except OSError:
                    continue
                entries.append((mtime, child))
        except OSError:
            continue

    entries.sort(key=lambda entry: entry[0], reverse=True)
    return [str(entry) for _, entry in entries[:limit]]


def load_resized(path: str | Path, max_dimension: int = MAX_DIMENSION) -> Image.Image:
    """Carga la imagen y la reduce manteniendo la proporción (el lado más largo <= max_dimension)."""
    with Image.open(path) as img:
        img.load()
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def make_thumbnail(path: str | Path, size: tuple = THUMB_SIZE) -> Image.Image:
    """Genera una miniatura pequeña sin decodificar la imagen completa (JPEG con draft)."""
    with Image.open(path) as img:
        img.draft("RGB", size)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return ImageOps.exif_transpose(img).convert("RGB")