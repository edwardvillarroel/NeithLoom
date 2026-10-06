"""Procesado de color para bordado.

Reduce la paleta de una imagen y mapea los colores resultantes al
catálogo de hilos Brother más cercano por distancia euclidiana en RGB.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from core import threads

COLOR_OPTIONS = (4, 6, 8, 10, 12)
DEFAULT_COLORS = 8
_KMEANS_MAX_ITER = 6
_KMEANS_SAMPLE = 20000


@dataclass
class ThreadUse:
    code: str
    name: str
    rgb: tuple
    usage_fraction: float


@dataclass
class ProcessResult:
    image: Image.Image
    threads_used: list = field(default_factory=list)


def _kmeans_palette(colors: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(colors) < n:
        n = len(colors)
    rand = rng.permutation(len(colors))[:n]
    centroids = colors[rand].astype(np.float64)

    if len(colors) > _KMEANS_SAMPLE:
        idx = rng.choice(len(colors), _KMEANS_SAMPLE, replace=False)
        sample = colors[idx].astype(np.float64)
    else:
        sample = colors.astype(np.float64)

    for _ in range(_KMEANS_MAX_ITER):
        diff = sample[:, None, :] - centroids[None, :, :]
        dist = np.einsum("ijk,ijk->ij", diff, diff)
        labels = dist.argmin(axis=1)
        new_centroids = centroids.copy()
        for i in range(n):
            members = sample[labels == i]
            if len(members):
                new_centroids[i] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1.0):
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.clip(np.rint(centroids), 0, 255).astype(np.uint8)


def reduce_palette(
    image: Image.Image,
    max_colors: int,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Reduce la imagen a un máximo de `max_colors` colores (K-Means ligero).

    `rng` opcional fija la semilla del K-Means (para resultados
    reproducibles en pruebas/benchmarks); por defecto es aleatorio.
    """
    image = image.convert("RGB")
    small = image.resize((image.width // 2, image.height // 2), Image.Resampling.BILINEAR)
    arr = np.asarray(small)
    pixels = arr.reshape(-1, 3).astype(np.uint8)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    top = np.argsort(counts)[-min(len(counts), 512):][::-1]
    colors = unique[top]

    palette = _kmeans_palette(
        colors, int(max_colors), rng if rng is not None else np.random.default_rng()
    )

    arr = np.asarray(image).reshape(-1, 3).astype(np.int32)
    best = np.zeros(arr.shape[0], dtype=np.int32)
    best_dist = np.full(arr.shape[0], np.inf)
    for i, p in enumerate(palette.astype(np.int32)):
        d = ((arr - p) ** 2).sum(axis=1)
        mask = d < best_dist
        best[mask] = i
        best_dist[mask] = d[mask]

    reduced = palette[best].reshape(image.height, image.width, 3)
    return Image.fromarray(reduced, "RGB")


def thread_use_from_code(code: str) -> ThreadUse:
    """Crea un ThreadUse de un hilo del catálogo Brother."""
    return ThreadUse(
        code=code,
        name=threads.THREAD_RGB_NAME[code],
        rgb=threads.THREAD_RGB[code],
        usage_fraction=0.0,
    )


def _find_nearest_thread(color: np.ndarray) -> tuple:
    diffs = np.asarray(threads.THREADS_RGB) - color
    dist = (diffs ** 2).sum(axis=1)
    idx = int(np.argmin(dist))
    return threads.THREADS[idx]


def map_to_threads(image: Image.Image) -> ProcessResult:
    """Mapea cada color de la imagen al hilo Brother más cercano en RGB."""
    arr = np.asarray(image.convert("RGB"))
    pixels = arr.reshape(-1, 3)
    colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )
    n_pixels = pixels.shape[0]

    thread_for_color = []
    result_colors = np.zeros_like(colors)
    thread_counts = {}
    for i, c in enumerate(colors):
        code, name, rgb = _find_nearest_thread(c)
        thread_for_color.append(code)
        result_colors[i] = rgb
        thread_counts[code] = thread_counts.get(code, 0) + int(counts[i])

    result = result_colors[inverse].reshape(arr.shape)

    uses = []
    for code, count in thread_counts.items():
        name = threads.THREAD_RGB_NAME[code]
        rgb = threads.THREAD_RGB[code]
        uses.append(
            ThreadUse(code=code, name=name, rgb=rgb, usage_fraction=count / n_pixels)
        )
    uses.sort(key=lambda u: u.usage_fraction, reverse=True)

    return ProcessResult(image=Image.fromarray(result, "RGB"), threads_used=uses)