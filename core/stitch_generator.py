"""Generación de puntadas por vectorización + relleno inteligente.

Convierte la imagen procesada (colores de hilos) en una lista de puntadas
con coordenadas en milímetros dentro del tamaño de bastidor elegido: la
imagen se escala proporcionalmente para que el bordado no supere el bastidor.

El motor vectorial reemplaza al scanline denso: cada componente conexa de
color se convierte en formas vectoriales (anillos exteriores + agujeros) con
`cv2.findContours(RETR_TREE)`, simplificados con `approxPolyDP` y suavizados
con Chaikin para quitar el escalón de píxel de los bordes. La JERARQUÍA de
esos contornos (`_contour_hierarchy`) separa el borde exterior de la forma de
sus huecos interiores (contadores de letra, agujeros): profundidad par = área
bordable, profundidad impar = hueco. Un hueco nunca se rellena, nunca recibe
underlay y su interior es infranqueable para el enrutado. Por cada forma se
bordan, en orden:

  1. Contorno (running stitch): el recorrido del borde exterior y el de los
     agujeros, muestreado a `OUTLINE_STEP_MM`. El borde del hueco se cose
     (sin rellenar su interior) para que el contador quede definido.
  2. Underlay ligero: un edge-run del borde de la región (la silueta de la
     máscara erosionada, ~`UNDERLAY_OFFSET_MM` hacia dentro) muestreado a
     `UNDERLAY_STEP_MM` (2.5 mm: un borde interior punteado y muy ligero,
     ~1/3 de una segunda pasada completa), para estabilizar el borde sin
     mucho hilo extra.
  3. Relleno Tatami diagonal: las filas scanline se trazan en una dirección
    FIJA de `FILL_ANGLE_DEG` (45° respecto a la horizontal de la imagen, la
     usada por los tatami profesionales) en lugar de seguir el eje mayor de
     cada forma, y las puntadas salen de intersecciones exactas con el polígono
     (reglas par-impar sobre anillos y agujeros), así cada fila llega justo al
     borde y no deja la franja despuntada del scanline de píxeles. Fila par e
     impar alternan el medio paso (offset Tatami) y dentro de cada tramo las
     puntadas se cosen de forma continua (`STITCH_LENGTH_DEFAULT_MM`), de modo
     que la hebra cubre todo el tramo sin huecos.

Antes de exportar, las puntadas se agrupan por color (todas las de un mismo
color juntas, una sola prueba de hilo por color) y dentro de cada color las
regiones se ordenan por centroides con nearest-neighbor, sin fusionarlas.
Después se reordenan sus etapas/filas en zigzag (ida y vuelta) minimizando los
saltos (`order_tatami_rows`/`_route_one_color`): la aguja permanece abajo
mientras las puntadas consecutivas están próximas y la transición entre
etapas de una misma región solo se salta si cruza fondo real. Cuando el salto
está forzado por la máscara (hueco/contador) se marca con `TRIM_SENTINEL` y el
exportador corta siempre; los saltos largos por distancia mantienen el umbral
`TRIM_MIN_JUMP_DISTANCE_MM` del exportador.

La matriz de códigos pasa antes por la limpieza morfológica ligera
(`_morph_close_codes`), el despeckle opcional (`_despeckle_codes`) y el
filtro de tamaño mínimo por región (`_drop_tiny_regions`), que ya existían y
no cambian. Las funciones scanline de la versión previa (`_tatami_fill`,
`_fill_region_tatami`, `_outline_stitches`, `_rotate_comp_mask`) se conservan
(no las usa `generate_stitches`) para que los benchmarks de `scripts/` sigan
funcionando.
"""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

from core import mockup as mockup_module
from core import resource_logger

DENSITY_OPTIONS = ("Baja", "Media", "Alta")

# Código centinela para los movimientos con aguja arriba (no se dibujan).
# Los códigos de hilo son cadenas ('001', ...), así que un entero no choca.
JUMP_SENTINEL = -1

# Igual que JUMP_SENTINEL pero obliga al exportador a cortar el hilo ANTES del
# salto. Lo emite `_route_one_color` cuando el segmento sale de la máscara
# sólida del color (hueco/contador real): ahí el salto NO depende de la
# distancia, así que no puede quedar esperando a `TRIM_MIN_JUMP_DISTANCE_MM`.
TRIM_SENTINEL = -2

# Todos los centinelas de movimiento: no son puntadas reales ni puntos de color.
MOVE_SENTINELS = (JUMP_SENTINEL, TRIM_SENTINEL)

# Separación entre puntadas según densidad, en milímetros. La separación
# vertical entre filas nunca es mayor que la horizontal (se ajusta hacia
# abajo al paso de píxel); las filas impares se desplazan media separación
# para tapar huecos (off-set Tatami).
DENSITY_STEP_MM = {
    "Baja": 0.40,
    "Media": 0.34,
    "Alta": 0.28,
}

# Separación por defecto (mm) entre puntadas a lo LARGO de cada tramo de
# relleno. Un tatami profesional usa puntadas largas de 2-3.5 mm: la hebra
# recorre el tramo de forma continua (la aguja baja y cose el tramo sin
# levantarse), así que reducir el nº de pinchazos de 0.2-0.4 mm a 3 mm no
# cambia la cobertura de hilo, solo el coste en máquina. La separación ENTRE
# FILAS la controla la densidad (`DENSITY_STEP_MM` -> `row_step` en
# `_tatami_fill`) y no se toca.
STITCH_LENGTH_DEFAULT_MM = 3.0

# Área mínima (mm²) que debe tener una región para merecer su propio pass de
# contorno+relleno. Las motas subs-mm (anti-aliasing o ruido de la reducción
# de paleta) no se pueden bordar de forma significativa y costaban UN TRIM de
# replegado cada una (en el logo de prueba tras el closing quedaban ~104 de
# 1-13 px = 0.02-0.22 mm²). En unidades de mm² el umbral es invariante a la
# resolución de la imagen. No se toca el conteo de componentes de la máscara
# original: los píxeles conservan su código de hilo, solo se omiten del
# relleno las regiones por debajo del umbral.
MIN_REGION_AREA_MM2 = 0.5

MAX_SIZE_MM = 250.0  # 25 cm
PREVIEW_SCALE_PX_PER_MM = 2.0
PREVIEW_BLOCK = 3
DEFAULT_TOLERANCE = 25  # ±25 en cada canal RGB para el fondo manual

# Umbral mínimo (mm) para decidir que dos puntadas consecutivas del mismo
# color están lo bastante cerca como para que la aguja siga abajo (el
# desplazamiento se dibuja como una puntada corta más) en lugar de levantar
# la aguja. El umbral efectivo es adaptativo: 1.8 x la mediana del paso
# horizontal entre filas de ese color (el relleno siempre deja ~2*step de
# separación, que depende de la resolución), nunca por debajo de este piso.
# Los huecos o zonas separadas del mismo color quedan por encima del umbral y
# se emiten como saltos (movimientos con aguja arriba que no se dibujan).
ROW_CONNECT_MM = 1.0

# Umbral opcional (mm) de "no cortar": distancia entre el ÚLTIMO punto cosido
# y el PRIMERO del siguiente tramo/región del mismo color por debajo de la
# cual la aguja NO se levanta dentro de la máscara sólida, en lugar de saltar
# con aguja arriba y cortar el hilo.
# Se suma (como piso) al criterio adaptativo existente de `_route_one_color`
# (max(ROW_CONNECT_MM, 1.8*pitch)). La continuidad entre etapas de una misma
# región se trata en `_route_one_color` y solo se permite dentro de esa máscara.
#
# `None` o <= 0 deja el comportamiento actual (solo el umbral adaptativo).
# Con puntadas largas de ~3 mm los puntos de la aguja quedan ~3 mm dentro
# de cada trazo, así que un hueco visible de 3 mm entre trazos de la misma
# letra aparece como un salto de ~6 mm entre puntos de aguja: NO bajar de ahí
# con la esperanza de coser los trazos separados de una letra. Subirlo vuelve
# a coser puentes de más en los huecos entre letras/zonas realmente separadas
# y puede reaparecer el problema original de hilo suelto en la parte
# inferior del bordado: cada subida del umbral debe validarse COSIENDO el
# diseño en la máquina real, subiendo de a poco, nunca de una sola vez.
CONNECT_NO_TRIM_MM = None

INTRA_REGION_CONNECTOR_MAX_MM = 12.0

# Lado del kernel cuadrado (píxeles) del closing morfológico aplicado a la
# máscara de cada color antes de coser. Un cuadrado de lado N cierra huecos de
# ancho ~N/2 que estén totalmente rodeados por el mismo color, y `None`/<2
# desactiva la limpieza. Un closing no une islas separadas por fondo, así que
# no daña detalles genuinos (ojos, letras, zonas independientes).
MORPH_KERNEL = 3

# Erosión (píxeles) aplicada a la máscara de huecos antes de usarla como
# barrera en el enrutado. El running stitch del borde del hueco se cose justo
# SOBRE el contorno, así que sin este margen cada puntada del contador se
# detectaría como "cruce de hueco" y se cortaría el hilo. Con 1 px el interior
# real del hueco (contadores de letra de varios píxeles) sigue prohibido.
HOLE_BARRIER_ERODE_PX = 1

# Ángulo muerto (grados): regiones cuyo eje mayor forma menos de este ángulo
# con la horizontal no se rotan (el relleno scanline ya sigue el eje y la
# salida coincide exactamente con la versión previa). Evita pagar la rotación
# en la región principal de logos de eje horizontal.
ROTATION_DEAD_ANGLE = 5.0

# Racha máxima, en mm, de píxeles que NO son del color que se está cosiendo que
# un segmento puede atravesar con la aguja abajo antes de darse por roto. Por
# debajo de este valor el hueco es redondeo de borde: el relleno se cose pegado
# al contorno y su segmento de enganche toca una columna de fondo, que no es
# separación real. Por encima, el hilo se estaría tendiendo sobre otro color o
# sobre el fondo, que es justo lo que esta constante impide.
#
# OJO con el nombre: "ajeno" incluye el FONDO, no solo los otros colores de la
# tabla. `own_mask` es `codes == c`, así que cualquier píxel que no sea este
# color cuenta, Includedo el fondo. Por eso la función es `_segment_off_color` y
# no "crosses_color": este proyecto ya sufrió por confundir las dos cosas (ver
# la nota de `bridge_gap_mm` en `_route_one_color`).
#
# Calibrado contra la distribución medida de las rachas CON LA AGUJA ABAJO en los
# tres logos de prueba, después de que esta comprobación gobierne el enrutado:
#
#   racha ajena   McDonalds   Elefante   NeithLoom
#   mediana        0.26 mm     0.29 mm   0.35 mm
#   p90           1.17 mm     0.58 mm   0.88 mm
#   p99           1.30 mm     1.16 mm   1.24 mm
#   max           1.30 mm     1.45 mm   2.12 mm
#
# Antes de que esta comprobación existiera, el máximo era 10.27 / 11.63 /
# 10.26 mm: es decir, el hilo se tendía hasta 11 mm sobre otro color o sobre el
# fondo y el enrutado lo daba por bueno porque la máscara dilatada 5x5 tapaba
# el hueco. La cota de 1.5 mm es el valle entre el ruido de borde (mediana 0.3)
# y los cruces reales. Bajarla a 1.0 parte el relleno en más intersecciones;
# subirla a 2.0 deja pasar hilos de hasta 2 mm. NeithLoom todavía reporta 5 de
# 1622 puentes por encima de 1.5 mm (máximo 2.12): es el margen de
# discretización de un lettering de 0.18 mm/px, no un cruce de zona.
OFF_COLOR_MAX_MM = 1.5

# Dirección del relleno Tatami del motor vectorial, en grados respecto a la
# horizontal de la imagen. 45° es la diagonal usada por los tatami
# profesionales: reparte mejor la tensión que el 0°/90° y evita que todas las
# filas del bordado queden paralelas al borde de la tela. Se aplica a TODAS las
# regiones (con `rotate_fill=True`, el default) en lugar de seguir el eje
# mayor de cada forma.
FILL_ANGLE_DEG = 45.0

# Separación (mm) entre puntadas del running stitch de contorno que se borda
# antes del relleno de cada región. Se elige por debajo de ROW_CONNECT_MM para
# que el enrutado por filas trate las puntadas contiguas del borde como
# conectadas (aguja abajo) y no genere un TRIM por puntada de contorno.
OUTLINE_STEP_MM = 1.2

# Motor vectorial: tolerancia (mm) de `approxPolyDP` al simplificar cada
# anillo (el mínimo del píxel se respeta porque no se baja de ~1 px) y número
# de pasadas de suavizado Chaikin que quitan el escalón de píxel del borde.
# La simplificación se subió (0.15→0.30) y el Chaikin se redujo (2→1) para
# aligerar el contorno: menos nodos/segmentos pequeños, la forma se mantiene
# (approxPolyDP conserva las esquinas dominantes) y el running de contorno
# sigue cubriendo el perímetro de forma continua (la hebra corre de puntada a
# puntada: baja el nº de pinchazos, no la cobertura de hilo).
VECTOR_EPS_MM = 0.30
CHAIKIN_STEPS = 1

# Underlay ligero: edge-run del borde hacia DENTRO de la región. La silueta
# se obtiene erosionando la máscara esta distancia (mm) y el running stitch
# se muestrea CAMBIANDO la separación respecto al contorno: `UNDERLAY_STEP_MM`
# (2.5 mm, el mismo paso que los pinchazos del relleno) baja el underlay a un
# simple borde interior punteado (~1/3 de puntadas que su versión a 0.8 mm),
# que es lo que pide el requisito "underlay mucho más ligero". Es seguro
# respecto a la regla de máscara porque el edge-run va sobre la silueta
# EROSIONADA (dentro del sólido real), nunca sobre el borde anti-aliasado.
UNDERLAY_OFFSET_MM = 0.7
UNDERLAY_STEP_MM = 2.5


@dataclass
class StitchPattern:
    stitches: list = field(default_factory=list)
    preview: Image.Image | None = None


def _step_mm(density: str) -> float:
    return DENSITY_STEP_MM.get(density, DENSITY_STEP_MM["Media"])  # mm entre puntadas


def _detect_background(arr: np.ndarray) -> tuple:
    """Devuelve el color de fondo según las 4 esquinas de la imagen."""
    corners = [
        tuple(arr[0, 0].tolist()),
        tuple(arr[0, -1].tolist()),
        tuple(arr[-1, 0].tolist()),
        tuple(arr[-1, -1].tolist()),
    ]
    return max(set(corners), key=corners.count)


def build_color_codes(
    image: Image.Image,
    threads_used: list,
    background_rgb: tuple | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
    background_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Devuelve la matriz de códigos de hilo (str) o None (fondo).

    Misma lógica que usaba `generate_stitches`: cada píxel se mapea al código
    del hilo Brother más cercano en el catálogo ya procesado; el fondo manual
    (`background_rgb` ± `tolerance`) y las zonas marcadas en
    `background_mask` quedan a `None`.
    """
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    rgb_to_code = {tuple(t.rgb): t.code for t in threads_used}

    if background_rgb is not None:
        bg_target = np.asarray(background_rgb, dtype=np.int16)
        is_bg = (np.abs(arr.astype(np.int16) - bg_target) <= tolerance).all(axis=2)
    elif background_mask is not None:
        is_bg = background_mask
    else:
        bg_color = np.asarray(_detect_background(arr), dtype=np.int16)
        is_bg = (np.abs(arr.astype(np.int16) - bg_color) <= 0).all(axis=2)
    if background_mask is not None:
        is_bg = is_bg | background_mask

    flat = arr.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    code_for = np.empty(len(unique), dtype=object)
    for ci, u in enumerate(unique):
        code_for[ci] = rgb_to_code.get(tuple(int(v) for v in u))
    codes = code_for[inverse].reshape(h, w)
    codes[is_bg] = None
    return codes


def _unique_color_codes(codes: np.ndarray) -> list:
    """Códigos de hilo distintos (excluido `None`), ordenados.

    `np.unique` sobre un array de tipo object falla si mezcla `None` y str
    (no se pueden comparar al ordenar), así que se filtran los `None` antes.
    """
    not_bg = codes != None  # noqa: E711
    if not not_bg.any():
        return []
    return list(np.unique(codes[not_bg]))


def _morph_close_codes(
    codes: np.ndarray, kernel_size: int = MORPH_KERNEL
) -> np.ndarray:
    """Cierre morfológico ligero por color sobre la matriz de códigos.

    Aplica un closing (dilatación + erosión) con un cuadrado de `kernel_size`
    píxeles a la máscara de cada color por separado y reconstruye la matriz.
    Un closing NUNCA une dos islas separadas por fondo abierto: rellena solo
    huecos o muescas estrechas (de ancho ~kernel_size/2) totalmente rodeados
    por el mismo color, que es la fragmentación falsa que causan el
    anti-aliasing y la compresión (tramos rotos de 1-2 píxeles que disparan
    TRIM de más). `kernel_size < 2` devuelve la matriz sin cambios.

    Opera sobre un mapa de índices enteros (por color) y comparaciones
    booleanas para no arrastrar el coste de los array de tipo object: el
    coste es O(colores x píxeles), vectorizado y sin estructuras que escalen
    con el número de fragmentos.
    """
    codes = np.asarray(codes)
    if kernel_size < 2:
        return codes
    colors = _unique_color_codes(codes)
    if not colors:
        return codes

    # Mapa de píxel -> índice de color con UNA pasada de np.unique (los
    # `None` se descartan aparte): evita las comparaciones elemento a
    # elemento sobre arrays de tipo object en los bucles del closing.
    idx = np.full(codes.shape, -1, dtype=np.int32)
    valid = codes != None  # noqa: E711
    uniq, inverse = np.unique(codes[valid], return_inverse=True)
    idx[valid] = inverse
    if not np.array_equal(uniq, colors):
        # Códigos que np.unique devuelve en otro orden del esperado: mapear.
        remap = {c: i for i, c in enumerate(colors)}
        for i, c in enumerate(uniq.tolist()):
            idx[codes == c] = remap[c]

    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    claimed = np.zeros(idx.shape, dtype=bool)
    out_idx = idx.copy()
    for i in range(len(colors)):
        closed = ndimage.binary_closing(idx == i, structure=structure)
        changed = closed & ~claimed
        out_idx[changed] = i
        claimed |= closed

    out = codes.copy()
    changed = out_idx != idx
    if changed.any():
        code_arr = np.array(colors, dtype=object)
        out[changed] = code_arr[out_idx[changed]]
    return out


def _count_spans(mask: np.ndarray, row_step: int = 1) -> int:
    """Tramos horizontales contiguos en las filas que visita `_tatami_fill`.

    Cuenta, sobre las filas `mask[::row_step]`, cuántos tramos contiguos de
    `True` hay por fila (transiciones False->True), el mismo criterio con el
    que `_tatami_fill` detecta los tramos de color. Vectorizado: suma
    comienzos de tramo sobre toda la matriz (una comparación por píxel).
    """
    rows = mask[::row_step]
    left = np.zeros_like(rows)
    left[:, 1:] = rows[:, :-1]
    return int((rows & ~left).sum())


def _lost_components(before: np.ndarray, after: np.ndarray) -> tuple:
    """Componentes de `before` que desaparecen de `after`, y su mayor área.

    Un closing es extensivo (`closing(X) ⊇ X`), así que un componente solo
    puede "desaparecer" si todos sus píxeles fueron absorbidos por otra
    región. Devuelve (n_components_perdidas, area_max_px) usando el mismo
    criterio de conectividad que `scipy.ndimage.label`.
    """
    labels, n = ndimage.label(before)
    if n == 0:
        return 0, 0
    present = np.zeros(n + 1, dtype=bool)
    present[1:] = False
    for label in np.unique(labels[after]):
        present[label] = True
    alive = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    lost = np.where(~present[1:])[0] + 1
    if not len(lost):
        return 0, 0
    areas = alive[lost - 1]
    return int(len(lost)), int(np.max(areas))


def diagnose_color_masks(
    codes_before: np.ndarray,
    codes_after: np.ndarray,
    kernel_size: int = MORPH_KERNEL,
    mmp: float = 1.0,
    step: float | None = None,
) -> dict:
    """Diagnóstico de la limpieza morfológica por color (una sola pasada).

    Para cada color distinto (excluido el fondo) reporta:
      - `components_before` / `components_after`: componentes conexas
        (`scipy.ndimage.label`) antes y después del closing.
      - `lost_components` / `lost_max_area_px`: componentes de ANTES que ya
        no existen DESPUÉS y el área (en píxeles) del mayor de los perdidos.
        Casi todo lo perdido son motas de 1-2 px (anti-aliasing) absorbidas
        por la región que las rodea: si `lost_max_area_px` es pequeño
        (del orden de kernel_size^2) ninguna isla genuina del diseño se ha
        perdido. Un ojo o una letra reales nunca se absorben.
      - `spans_before` / `spans_after`: tramos que detectaría el scanline de
        `_tatami_fill` (con `row_step` derivado de `mmp`/`step`) antes y
        después. Su descenso es la prueba de que se eliminan fragmentos
        falsos y, con una salvedad, de que caerá el conteo de TRIM.

    Devuelve un dict `{código: {...}}` ordenado por código.
    """
    if step is None:
        step = _step_mm("Media")
    row_step = max(1, int(step / mmp)) if mmp else 1
    uniq = set(_unique_color_codes(codes_before)) | set(
        _unique_color_codes(codes_after)
    )
    report = {}
    for code in sorted(uniq):
        before = (codes_before == code).astype(bool)
        after = (codes_after == code).astype(bool)
        n_before = int(ndimage.label(before)[1])
        n_after = int(ndimage.label(after)[1])
        lost_n, lost_area = _lost_components(before, after)
        report[code] = {
            "components_before": n_before,
            "components_after": n_after,
            "components_delta": n_after - n_before,
            "lost_components": lost_n,
            "lost_max_area_px": lost_area,
            "spans_before": _count_spans(before, row_step),
            "spans_after": _count_spans(after, row_step),
            "spans_delta": _count_spans(after, row_step) - _count_spans(before, row_step),
        }
    return report


def _fill_span(
    events: list,
    px0: int,
    px1: int,
    y: int,
    code: str,
    mmp: float,
    step: float,
    odd: bool,
    stitch_len: float | None = None,
) -> None:
    """Coloca las puntadas cortas de un tramo horizontal de color.

    El tramo va del píxel `px0` al `px1` de la fila `y`. Sobre él se traza
    una rejilla con separación `step` mm y se recorre DE FORMA CONTINUA: la
    aguja baja al primer punto y cose sin levantarse entre puntos
    consecutivos, de modo que el hilo cubre todo el tramo sin huecos
    horizontales (no se dejan tramos sin coser entre puntadas sueltas).
    Las filas impares se desplazan media separación (offset Tatami) para
    intercalar el hilo entre filas y tapar los huecos en vertical.

    `stitch_len` (mm), si se indica, amplía la separación ENTRE puntadas a lo
    largo del tramo (3 mm frente a los ~0.3 mm de la densidad): reduce el
    conteo de puntadas de cada tramo al precio de una cobertura horizontal
    más espaciada (la hebra recorre el tramo de forma continua entre
    puntadas, así que la cobertura de hilo no cambia; solo el nº de pinchazos
    de aguja). Si es `None` se usa la separación por defecto
    `STITCH_LENGTH_DEFAULT_MM` (3.0 mm). La separación entre filas
    (`row_step`, aparte) no cambia.

    Con la rejilla de separación `spacing`, `off` vale `spacing/2` en filas
    impares (offset Tatami): el desplazamiento intercala los pinchazos de una
    fila con los de la siguiente, cumpliéndose también con el paso por
    defecto de 3 mm.

    Todos los tramos válidos generan al menos una puntada, incluidos los de
    un solo píxel: si la rejilla no alcanzara a entrar (tramos más cortos que
    la separación), se borda una puntada en el centro del tramo.
    """
    spacing = stitch_len if stitch_len and stitch_len > 0 else STITCH_LENGTH_DEFAULT_MM
    x_start = px0 * mmp
    off = spacing / 2.0 if odd else 0.0
    points = []
    last_px = -1
    i = 0
    while True:
        xx = x_start + off + i * spacing
        px = int(round(xx / mmp))
        if px > px1:
            break
        if px >= px0 and px != last_px:
            points.append((xx, px))
            last_px = px
        i += 1

    ym = y * mmp
    if not points:
        # Tramo corto: la rejilla no cabe, se borda la puntada central
        # para no descartar el tramo.
        mid_mm = ((px0 + px1) / 2.0) * mmp
        events.append((mid_mm, ym, JUMP_SENTINEL))
        events.append((mid_mm, ym, code))
        return

    # Recorrido continuo: salto (aguja arriba) solo al inicio del tramo y
    # una puntada por punto de la rejilla, sin saltos entre ellos.
    events.append((points[0][0], ym, JUMP_SENTINEL))
    for xx, _px in points:
        events.append((xx, ym, code))


def _tatami_fill(
    codes: np.ndarray,
    mmp: float,
    step: float,
    stitch_len: float | None = None,
) -> list:
    """Relleno Tatami (scanline con offset) en puntadas cortas.

    Recorre la imagen fila por fila (eje Y) y, dentro de cada fila, detecta
    los tramos horizontales de cada color (`code` != None) y los rellena con
    puntadas cortas separadas `step` mm, de forma continua dentro de cada
    tramo (cada tramo queda completamente cubierto). Las filas pares no se
    desplazan y las impares se desplazan media separación (half-offset).

    La separación vertical entre filas se redondea hacia abajo al paso de
    píxel (nunca es mayor que la horizontal) y jamás se salta una fila de
    píxeles: cada fila de la imagen que contenga un color válido genera
    puntadas.

    `codes` es una matriz (h, w) de objetos: el código de hilo (str) o None
    para fondo/sin hilo. `stitch_len` (mm), si se indica, amplía la
    separación entre puntadas dentro de cada tramo (ver `_fill_span`);
    `None` usa la separación por defecto `STITCH_LENGTH_DEFAULT_MM`. La
    separación entre filas no cambia (la controla la densidad). Devuelve la
    lista de puntadas (x_mm, y_mm, código) con el marcador JUMP_SENTINEL en
    los movimientos de aguja arriba.
    """
    h, w = codes.shape
    row_step = max(1, int(step / mmp))  # sep. vertical <= horizontal
    events = []
    odd = False
    for y in range(0, h, row_step):
        row = codes[y]
        x = 0
        while x < w:
            code = row[x]
            if code is None:
                x += 1
                continue
            x1 = x + 1
            while x1 < w and row[x1] == code:
                x1 += 1
            _fill_span(events, x, x1 - 1, y, code, mmp, step, odd, stitch_len)
            x = x1
        odd = not odd
    return events


def _region_major_axis_deg(mask: np.ndarray) -> float:
    """Ángulo (grados) del eje mayor de una componente segun sus momentos.

    Usa los momentos centrales de segundo orden de la máscara
    (`0.5*atan2(2*mu11, mu20-mu02)`) en el sistema de imagen (Y hacia
    abajo). Es una fórmula cerrada calculada sobre los píxeles de la
    componente (O(área), el mismo recorrido que ya hace `ndimage.label`),
    sin iterar nada extra: identifica el eje largo de una letra/diagonal.
    Devuelve 0.0 para una máscara vacía o puntual.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0
    cy = float(ys.mean())
    cx = float(xs.mean())
    mu20 = float(((xs - cx) ** 2).sum())
    mu02 = float(((ys - cy) ** 2).sum())
    if mu20 + mu02 == 0.0:
        return 0.0
    mu11 = float(((xs - cx) * (ys - cy)).sum())
    return 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))


def _rotate_comp_mask(mask: np.ndarray, deg: float) -> tuple:
    """Máscara rotada para dejar el eje mayor horizontal + mapeo para volver.

    Rota la ventana `mask` con `scipy.ndimage.rotate` (order=0, `reshape=True`,
    la máscara queda binaria y el resultado contenido sin recortar) por el
    ángulo `deg` que alinea su eje mayor con la horizontal. Devuelve
    `(rotada, C)` donde `C` es una matriz afín 3x2 que mapea las coordenadas
    (fila, columna) de la salida rotada a las de la ventana original:

        p_in = C @ [r', c', 1]   (fila, columna) en la ventana

    La transformación se MIDE, no se asume: se rota un marcador en el centro
    de la ventana (una rotación igual preserva el centro) y dos marcadores
    desplazados una distancia `off` del centro en +fila y +columna, resueltos
    en tres rotaciones separadas de la misma ventana (O(ventana) cada una,
    mismas del relleno). Los marcadores son bloques 7x7 rotados con
    interpolación bilineal para medir bien su centro. Con el centro y las dos
    direcciones del marco la afín queda determinada de forma exacta y
    consistente con el frame de `ndimage.rotate`.
    """
    rot = ndimage.rotate(mask.astype(np.uint8), deg, order=0, reshape=True) > 0
    H, W = mask.shape
    c_in = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    off = float(max(4, min(H, W) // 4))
    rad = 3
    offset_rel = [np.zeros(2), np.array([0.0, off]), np.array([off, 0.0])]
    pts = []
    ok = True
    for rel in offset_rel:
        m = np.zeros((H, W), dtype=np.uint8)
        r = min(max(int(round(c_in[0] + rel[0])), 0), H - 1)
        c = min(max(int(round(c_in[1] + rel[1])), 0), W - 1)
        m[max(0, r - rad):min(H, r + rad + 1), max(0, c - rad):min(W, c + rad + 1)] = 1
        rm = ndimage.rotate(m, deg, order=1, reshape=True)
        lab, nl = ndimage.label(rm > 0.1)
        if nl < 1:
            ok = False
            break
        cm = np.asarray(ndimage.center_of_mass(rm, lab, range(1, nl + 1)))
        pts.append(cm[0])
    if not ok:
        return rot, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
    c_out, pa, pb = pts
    d_c = (pa - c_out) / off
    d_r = (pb - c_out) / off
    R = np.column_stack([d_r, d_c])  # (2,2) delta_in -> delta_out
    if abs(np.linalg.det(R)) < 1e-9:
        # rotación degenerada (ventana diminuta colapsada)
        return rot, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
    inv = np.linalg.inv(R)
    b = c_in - inv @ c_out
    C = np.vstack([inv.T, b.reshape(1, 2)])  # (3,2): [r,c,1] -> in
    return rot, C


def _fill_region_tatami(
    comp: np.ndarray,
    r0: int,
    c0: int,
    code: str,
    mmp: float,
    step: float,
    stitch_len: float | None,
    rotate: bool,
) -> list:
    """Relleno Tatami (el mismo `_tatami_fill`) de una componente a coords globales.

    Recorta la ventana `comp`, la rellena con `_tatami_fill` y devuelve las
    puntadas en coordenadas globales (mm). Si `rotate` es True y el eje mayor
    no es (casi) horizontal, la ventana se rota primero para dejar el eje
    horizontal, el relleno corre sin cambios sobre el frame rotado y las
    puntadas se rotan de vuelta al sistema de la imagen (ver
    `_rotate_comp_mask`): las filas scanline quedan paralelas al eje de la
    forma en lugar de siempre horizontales. Con `rotate` False o ángulos
    dentro de `ROTATION_DEAD_ANGLE` devuelve exactamente el relleno previo.

    Las puntadas se devuelven como 4-tuplas `(x, y, code, row)` donde `row`
    es la FILA scanline lógica que las produjo: la coordenada de la fila en el
    frame (rotado o no) sobre el que corrió el relleno. El enrutado la usa
    como clave de agrupación (rowkey) para reconstruir el zigzag ida-y-vuelta
    incluso cuando las puntadas de una misma fila ya no comparten Y global.
    """
    if rotate:
        ang = _region_major_axis_deg(comp)
        if abs(ang) >= ROTATION_DEAD_ANGLE:
            rot, C = _rotate_comp_mask(comp, ang)
            roi = np.full(rot.shape, None, dtype=object)
            roi[rot] = code
            events = _tatami_fill(roi, mmp, step, stitch_len)
            H, W = comp.shape
            n = len(events)
            xs = np.asarray([e[0] for e in events], dtype=np.float64)
            ys = np.asarray([e[1] for e in events], dtype=np.float64)
            prow = np.rint(ys / mmp).astype(int)  # fila scanline del frame rotado
            p_out = np.stack([ys / mmp, xs / mmp], axis=1)  # (fila,col) rotado
            aug = np.hstack([p_out, np.ones((n, 1))])
            p_in = aug @ C  # (fila,col) en la ventana
            row = np.clip(p_in[:, 0], 0.0, float(H - 1))
            col = np.clip(p_in[:, 1], 0.0, float(W - 1))
            out = []
            for e, rw, cl, rr in zip(events, row, col, prow):
                out.append(((c0 + cl) * mmp, (r0 + rw) * mmp, e[2], int(rr)))
            return out

    roi = np.full(comp.shape, None, dtype=object)
    roi[comp] = code
    events = _tatami_fill(roi, mmp, step, stitch_len)
    dc, dr = c0 * mmp, r0 * mmp
    return [(x + dc, y + dr, t, int(round(y / mmp))) for x, y, t in events]


def _outline_stitches(
    comp: np.ndarray,
    r0: int,
    c0: int,
    code: str,
    mmp: float,
    outline_step_mm: float = OUTLINE_STEP_MM,
) -> list:
    """Running stitch sobre el contorno de la componente (borde tal cual).

    Usa `cv2.findContours` (borde exterior, puntos de píxel sin suavizar ni
    ajustar: coste O(perímetro), mucho menor que el propio relleno) y genera
    un running stitch que recorre los puntos del contorno con separación
    `outline_step_mm`, bajando la aguja al primer punto. Un salto lead-in al
    inicio y puntadas seguidas: como la separación es menor que el umbral de
    conexión del enrutado, el borde se cose de forma continua sin TRIM
    internos. Devuelve eventos en coordenadas globales (mm).
    """
    cnts, _ = cv2.findContours(
        comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    k = max(1, int(round(outline_step_mm / mmp)))
    out = []
    for cnt in cnts:
        pts = cnt[:, 0, :]  # (N,2) con (x=col, y=row) en coords de la ventana
        sel = pts[::k]
        if len(sel) < 2:
            continue
        gx0 = (c0 + int(sel[0, 0])) * mmp
        gy0 = (r0 + int(sel[0, 1])) * mmp
        out.append((gx0, gy0, JUMP_SENTINEL))
        for p in sel[1:]:
            out.append((((c0 + int(p[0])) * mmp), ((r0 + int(p[1])) * mmp), code))
    return out


def _chaikin_closed(pts: np.ndarray, iters: int) -> np.ndarray:
    """Suaviza un polígono cerrado con corner-cutting de Chaikin.

    Cada iteración sustituye cada vértice `P_i` por los puntos
    `0.75*P_i + 0.25*P_{i+1}` y `0.25*P_i + 0.75*P_{i+1}` (cierre cíclico):
    redondea los vértices y acorta los segmentos sin tocar el área aparente
    de la forma, eliminando el escalón de píxel que deja `cv2.findContours`.
    Opera en coordenadas flotantes, independiente de la resolución.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return pts
    out = pts
    for _ in range(iters):
        n = len(out)
        nxt = np.roll(out, -1, axis=0)
        a = 0.75 * out + 0.25 * nxt
        b = 0.25 * out + 0.75 * nxt
        out = np.empty((2 * n, 2), dtype=np.float64)
        out[0::2] = a
        out[1::2] = b
    return out


def _rotate_pts(pts: np.ndarray, angle_deg: float, origin) -> np.ndarray:
    """Rota puntos (x, y) `angle_deg` grados alrededor de `origin`.

    Rotación afín exacta (cos/sin), la misma para píxeles o milímetros
    porque la escala es isótropa. Una rotación `+a` se deshace con `-a`.
    """
    ar = math.radians(angle_deg)
    ca, sa = math.cos(ar), math.sin(ar)
    d = np.asarray(pts, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return (
        np.column_stack(
            [ca * d[:, 0] - sa * d[:, 1], sa * d[:, 0] + ca * d[:, 1]]
        )
        + np.asarray(origin, dtype=np.float64)
    )


def _approximate_ring(ring, eps: float) -> np.ndarray:
    """Simplifica un contorno de OpenCV (N,1,2) a un polígono (M,2) float."""
    pts = ring[:, 0, :].astype(np.float64)
    if len(pts) < 3:
        return pts
    approx = cv2.approxPolyDP(
        pts.astype(np.float32).reshape(-1, 1, 2), eps, closed=True
    )
    if len(approx) < 3:
        return pts
    return approx[:, 0, :].astype(np.float64)


@dataclass
class ContourHierarchy:
    """Jerarquía de contornos de UNA componente conexa de un color.

    Cada anillo es un polígono (N,2) float en píxeles de la ventana de la
    componente, con su profundidad de anidamiento:

      - profundidad 0: borde EXTERIOR de la forma,
      - profundidad 1: hueco interior (contador de letra, agujero, ojo),
      - profundidad 2: isla del mismo color dentro de un hueco,
      - y así sucesivamente.

    Un anillo es hueco cuando su profundidad es IMPAR; un anillo es rellenable
    cuando es PAR. `sewable` es la máscara rellenable (regla par-impar sobre
    todos los anillos) y `hole_mask` la de los huecos realmente vacíos (anillos
    pares dentro de un hueco), que es la zona que nunca se rellena ni se cruza.
    """

    rings: list
    outer: list
    holes: list
    sewable: np.ndarray
    hole_mask: np.ndarray

    @property
    def has_holes(self) -> bool:
        return bool(self.holes)


def _contour_hierarchy(mask: np.ndarray, mmp: float) -> ContourHierarchy:
    """Detecta los contornos de una componente CON jerarquía y clasifica cada
    anillo en exterior/hueco según su profundidad de anidamiento.

    Usa `cv2.findContours(RETR_TREE)`, que devuelve la jerarquía completa: el
    padre de cada contorno permite calcular su profundidad caminando la cadena
    de padres. Con RETR_CCOMP la información se limita a dos niveles y un
    anillo externo de una isla dentro de un hueco sería indistinguible de un
    borde exterior; con RETR_TREE la paridad de la profundidad sí lo es.

    Los anillos se simplifican con `approxPolyDP` (eps = `VECTOR_EPS_MM` en
    mm, sin bajar de ~1 px) y se suavizan con Chaikin, igual que antes. Las
    máscaras `sewable`/`hole_mask` se rasterizan desde ESOS MISMOS anillos con
    la regla par-impar, de modo que la geometría que rellena el motor y la que
    prohíbe cruzar un hueco son exactamente la misma.
    """
    empty = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return ContourHierarchy([], [], [], empty, empty)
    cnts, hier = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if hier is None or not len(cnts):
        return ContourHierarchy([], [], [], empty, empty)

    eps = max(1.0, VECTOR_EPS_MM / mmp)
    rings: list = []
    for i, c in enumerate(cnts):
        ring = _approximate_ring(c, eps)
        if len(ring) < 3:
            continue
        ring = _chaikin_closed(ring, CHAIKIN_STEPS)
        if len(ring) < 3:
            continue
        depth = 0
        parent = int(hier[0, i, 3])
        while parent >= 0:
            depth += 1
            parent = int(hier[0, parent, 3])
        rings.append((ring, depth))

    # Regla par-impar REAL: un contador de anidamiento, no dos lienzos. Cada
    # anillo relleno suma 1; un píxel está dentro de la forma (rellenable) si
    # lo contiene un número IMPAR de anillos, y está en un hueco si lo contiene
    # un número PAR >= 2 (dentro de la forma, pero no rellenable). Un contador
    # 0 está fuera del componente, así que no cuenta como hueco.
    # Importante: separar "exteriores" y "huecos" en dos imágenes NO sirve,
    # porque rellenar el anillo exterior pinta también el hueco que contiene.
    count = np.zeros(mask.shape, dtype=np.uint8)
    ring_layer = np.zeros(mask.shape, dtype=np.uint8)
    for ring, _depth in rings:
        # `fillPoly` ASIGNA el valor, no lo suma: cada anillo se rasteriza en
        # un lienzo limpio y se acumula, que es lo que produce el conteo de
        # anidamiento.
        ring_layer.fill(0)
        cv2.fillPoly(ring_layer, [np.round(ring).astype(np.int32)], 1)
        count += ring_layer
    count = count.astype(np.int16)
    sewable = (count % 2) == 1
    # Una isla del mismo color dentro de un hueco (profundidad par) tiene un
    # número impar de anillos encima y por eso vuelve a ser rellenable: la
    # paridad ya lo restaura sin necesidad de restar `~sewable` a mano.
    hole_mask = (count % 2 == 0) & (count >= 2)

    outer = [r for r, d in rings if d == 0]
    holes = [r for r, d in rings if d % 2 == 1]
    outer.sort(key=lambda r: (float(r[:, 1].min()), float(r[:, 0].min())))
    return ContourHierarchy(rings, outer, holes, sewable, hole_mask)


def _vectorize_component(mask: np.ndarray, mmp: float) -> tuple:
    """Anillos vectoriales de una componente: `(externos, agujeros)`.

    Envoltorio de `_contour_hierarchy` para los llamadores que solo necesitan
    la lista de anillos: exteriores = profundidad par 0, agujeros = profundidad
    impar. Devuelve arrays (N,2) float en coordenadas de píxel (x=columna,
    y=fila) de la ventana de la componente.
    """
    hierarchy = _contour_hierarchy(mask, mmp)
    return hierarchy.outer, hierarchy.holes


def _resample_ring(ring, step_mm: float, mmp: float) -> list:
    """Muestrea un anillo cerrado a intervalos de `step_mm` (running stitch).

    Recorre el perímetro del polígono y emite los puntos del recorrido
    separados `step_mm` mm de distancia de arco (el último tramo se corta
    antes de cerrar). Devuelve coordenadas de píxel de la ventana.
    """
    pts = np.asarray(ring, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return []
    nxt = np.roll(pts, -1, axis=0)
    seg = np.hypot(nxt[:, 0] - pts[:, 0], nxt[:, 1] - pts[:, 1]) * mmp
    tb = np.maximum.accumulate(np.concatenate([[0.0], np.cumsum(seg)]))
    total = float(tb[-1])
    if total <= 0:
        return []
    xs = np.concatenate([pts[:, 0], [pts[0, 0]]])
    ys = np.concatenate([pts[:, 1], [pts[0, 1]]])
    s = np.arange(0.0, total, step_mm)
    return list(
        zip(
            np.interp(s, tb, xs).tolist(),
            np.interp(s, tb, ys).tolist(),
        )
    )


def _outline_vectorize(rings, r0, c0, code, mmp, step_mm) -> list:
    """Running stitch sobre los anillos vectoriales en coords globales (mm).

    Por cada anillo: un lead-in (JUMP_SENTINEL) al primer punto y puntadas
    consecutivas a `step_mm` (por debajo de ROW_CONNECT_MM, de modo que el
    enrutado cose el borde de forma continua sin TRIM internos).
    """
    out = []
    for ring in rings:
        pts = _resample_ring(ring, step_mm, mmp)
        if len(pts) < 2:
            continue
        out.append(((c0 + pts[0][0]) * mmp, (r0 + pts[0][1]) * mmp, JUMP_SENTINEL))
        for x, y in pts:
            out.append(((c0 + x) * mmp, (r0 + y) * mmp, code))
    return out


def _point_inside_mask(point, r0, c0, mask, mmp) -> bool:
    """¿El punto (x_px, y_px) de una ventana cae dentro de `mask`?

    Traduce a píxeles de la ventana de la componente (r0, c0 son su origen
    global) y consulta la máscara booleana. Fuera de la ventana se considera
    que NO está dentro: el hueco solo existe dentro del componente.
    """
    if mask is None:
        return False
    r = int(round(r0 + point[1]))
    c = int(round(c0 + point[0]))
    if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]:
        return False
    return bool(mask[r, c])


def _underlay_edge_run(
    mask, r0, c0, code, mmp, step_mm, hole_mask=None
) -> list:
    """Underlay ligero: edge-run del borde de la región hacia dentro.

    Erosiona la máscara `UNDERLAY_OFFSET_MM` mm para obtener la silueta
    interior y le da un running stitch a `step_mm` (= `UNDERLAY_STEP_MM`,
    2.5 mm: un borde interior punteado y ligero, ~1/3 de una segunda pasada
    completa). Si la región es demasiado fina y la erosión la borra, no se
    genera underlay.

    `hole_mask` es la máscara de huecos de la jerarquía de contornos: se usa
    para descartar cualquier puntada que caiga dentro de un contador. Hace
    falta porque al erosionar una forma con agujero (una "O", una "R") la
    silueta interior puede partirse en trozos y el borde del hueco pasa a ser
    un contorno EXTERIOR de uno de ellos; sin este filtro el underlay
    redondearía el contador.
    """
    k = max(1, int(round(UNDERLAY_OFFSET_MM / mmp)))
    nmask = ndimage.binary_erosion(mask, iterations=k)
    if not nmask.any():
        return []
    ext, _ = _vectorize_component(nmask, mmp)
    events = _outline_vectorize(ext, r0, c0, code, mmp, step_mm)
    if hole_mask is not None and hole_mask.any():
        events = [
            e
            for e in events
            if not _point_inside_mask((e[0], e[1]), r0, c0, hole_mask, mmp)
        ]
    return events


def _assign_fill_strips(row_spans: list, strip_prev: list) -> list:
    """Asigna a cada tramo de una fila su "calle" (strip id) estable.

    Un tramo es el intervalo `[xa, xb]` que la fila scanline atraviesa dentro de
    la región. Una misma fila puede tener VARIOS tramos separados por fondo real
    o por otro color (los dos arcos de una "M", las dos patas de un elefante,
    dosordes de un "8"): en ese caso el enrutado NO debe coser de un tramo al
    siguiente, porque el hueco entre ellos es separación de verdad.

    Para que el enrutado pueda distinguirlos, cada tramo lleva su propio
    identificador de calle, que viaja en la rowkey. La calle se hereda de la
    fila anterior cuando el tramo se solapa con ella, de forma que una columna
    que baja por la región conserva su identificador y el hilo sigue cosiendo
    la misma "columna" sin cortes; los tramos que aparecen de nuevo reciben
    identificadores nuevos.

    `strip_prev` es la lista `(sid, xa, xb)` de la fila anterior; vacía en la
    primera fila de cada bloque. Devuelve un id por tramo, en el mismo orden.
    """
    strips = []
    used = set()
    for xa, xb in row_spans:
        best_sid = None
        best_ov = 0.0
        for sid, pa, pb in strip_prev:
            if sid in used:
                continue
            overlap = min(xb, pb) - max(xa, pa)
            if overlap > best_ov:
                best_ov = overlap
                best_sid = sid
        if best_sid is None:
            best_sid = max((sid for sid, _, _ in strip_prev), default=-1) + 1
        used.add(best_sid)
        strips.append(best_sid)
    return strips


def _span_events(
    events, x0, x1, y, code, spacing, odd, mmp, strip: int = 0
) -> None:
    """Puntadas continuas de un tramo [x0, x1] en la fila scanline `y`.

    Rejilla de separación `spacing` mm con medio paso en filas impares
    (offset Tatami); la aguja baja en `x0` (JUMP lead-in) y cose todos los
    puntos de la rejilla de forma continua, de modo que la hebra cubre el
    tramo entero. Un último punto en `x1` (si queda más de ~0.3 mm libre)
    ancla el borde del relleno al polígono.

    `strip` es la calle del tramo (ver `_assign_fill_strips`) y se propaga en
    la rowkey de cada punto, de modo que dos tramos de la misma fila nunca se
    cosen entre sí como si fueran una fila continua.
    """
    step_px = spacing / mmp
    off_px = step_px / 2.0 if odd else 0.0
    events.append((x0, y, JUMP_SENTINEL, strip))
    last = None
    for i in range(int((x1 - x0 - off_px) / step_px) + 2):
        px = x0 + off_px + i * step_px
        if px > x1:
            break
        events.append((px, y, code, strip))
        last = px
    if last is None:
        events.append(((x0 + x1) / 2.0, y, code, strip))
    elif (x1 - last) * mmp > 0.3:
        events.append((x1, y, code, strip))


def _fill_region_vector(
    mask,
    r0,
    c0,
    code,
    mmp,
    step,
    stitch_len,
    rotate,
    hierarchy: "ContourHierarchy | None" = None,
) -> list:
    """Relleno Tatami vectorial con dirección diagonal fija.

    Vectoriza la forma con su jerarquía de contornos y, en un frame rotado
    `FILL_ANGLE_DEG` (45°) para que las filas scanline queden en diagonal
    respecto a la imagen, barre filas separadas `step` mm (exactas, no
    cuantizadas a píxel). Cada fila intersecta los segmentos de TODOS los
    anillos (exteriores + huecos + islas, por profundidad) y las intersecciones
    se emparejan con la regla par-impar: así el interior de cada hueco queda
    sin rellenar y el tramo llega exactamente al borde de la forma. Como
    última garantía, `hierarchy.hole_mask` descarta cualquier puntada que
    hubiera caído dentro de un hueco (el suavizado Chaikin puede recortar una
    concavidad y empujar el anillo un píxel hacia dentro del contador). Las
    puntadas se rotan de vuelta al sistema de la imagen. Devuelve 4-tuplas
    `(x_mm, y_mm, t, fila)` donde `fila` identifica la línea scanline del
    frame (rotado o no) para que el enrutado reconstruya el zigzag.
    """
    hierarchy = hierarchy if hierarchy is not None else _contour_hierarchy(mask, mmp)
    rings = [ring for ring, _depth in hierarchy.rings]
    if not rings:
        return []
    hole_guard = None
    if hierarchy.hole_mask.any():
        # Erosión de 1 px: no descarta las puntadas que se apoyan justo en el
        # borde del hueco (el running stitch del contador se cose encima).
        hole_guard = ndimage.binary_erosion(
            hierarchy.hole_mask, structure=np.ones((3, 3))
        )
        if not hole_guard.any():
            hole_guard = None
    frame_ang = -FILL_ANGLE_DEG if rotate else 0.0
    org = (hierarchy.outer or rings)[0].mean(axis=0)

    X1, Y1, X2, Y2 = [], [], [], []
    for r in rings:
        rr = _rotate_pts(r, frame_ang, org) if frame_ang else r
        X1.append(rr[:, 0])
        Y1.append(rr[:, 1])
        X2.append(np.roll(rr[:, 0], -1))
        Y2.append(np.roll(rr[:, 1], -1))
    X1, Y1 = np.concatenate(X1), np.concatenate(Y1)
    X2, Y2 = np.concatenate(X2), np.concatenate(Y2)
    dy_all = Y2 - Y1

    ymin, ymax = float(Y1.min()), float(Y1.max())
    if ymax - ymin < 1e-9:
        return []
    spacing = stitch_len if stitch_len and stitch_len > 0 else STITCH_LENGTH_DEFAULT_MM
    row_step = float(step / mmp)
    n_rows = int(math.floor((ymax - ymin) / row_step)) + 1

    out = []
    strip_last: list = []
    for ri in range(n_rows):
        y = ymin + ri * row_step
        ok = (dy_all != 0) & ((Y1 <= y) != (Y2 <= y))
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            strip_last = []
            continue
        xs = X1[idx] + (y - Y1[idx]) * (X2[idx] - X1[idx]) / dy_all[idx]
        xs.sort()
        row_spans = []
        for s in range(0, len(xs) - 1, 2):
            xa, xb = float(xs[s]), float(xs[s + 1])
            if xb - xa > 0.1 / mmp:
                row_spans.append((xa, xb))
        if not row_spans:
            strip_last = []
            continue
        strips = _assign_fill_strips(row_spans, strip_last)
        evs = []
        for (xa, xb), sid in zip(row_spans, strips):
            _span_events(evs, xa, xb, y, code, spacing, ri % 2, mmp, sid)
        if not evs:
            strip_last = []
            continue
        strip_last = [(sid, xa, xb) for (xa, xb), sid in zip(row_spans, strips)]
        if frame_ang:
            pts = _rotate_pts(
                np.asarray([[e[0], e[1]] for e in evs], dtype=np.float64),
                -frame_ang,
                org,
            )
            for e, (px, py) in zip(evs, pts):
                if hole_guard is not None and _point_inside_mask(
                    (px, py), 0, 0, hole_guard, mmp
                ):
                    continue
                out.append(((c0 + px) * mmp, (r0 + py) * mmp, e[2], (ri, e[3])))
        else:
            for e in evs:
                if hole_guard is not None and _point_inside_mask(
                    (e[0], e[1]), 0, 0, hole_guard, mmp
                ):
                    continue
                out.append(((c0 + e[0]) * mmp, (r0 + e[1]) * mmp, e[2], (ri, e[3])))
    return out


def _ordered_region_slices(labels: np.ndarray, mmp: float) -> list:
    """Ordena componentes por centroides con nearest-neighbor, sin fusionarlas."""
    candidates = []
    for label, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        comp = labels[sl] == label
        ys, xs = np.nonzero(comp)
        if len(xs):
            center_x = (sl[1].start + float(xs.mean())) * mmp
            center_y = (sl[0].start + float(ys.mean())) * mmp
        else:
            center_x = (sl[1].start + sl[1].stop) * 0.5 * mmp
            center_y = (sl[0].start + sl[0].stop) * 0.5 * mmp
        candidates.append((label, sl, center_x, center_y))

    remaining = candidates
    current_x = 0.0
    current_y = 0.0
    ordered = []
    while remaining:
        index = min(
            range(len(remaining)),
            key=lambda idx: (
                math.hypot(
                    remaining[idx][2] - current_x,
                    remaining[idx][3] - current_y,
                ),
                remaining[idx][0],
            ),
        )
        label, sl, center_x, center_y = remaining.pop(index)
        ordered.append((label, sl))
        current_x = center_x
        current_y = center_y
    return ordered


def _vector_fill_regionwise(
    codes: np.ndarray,
    mmp: float,
    step: float,
    stitch_len: float | None = None,
    outline: bool = True,
    underlay: bool = True,
    rotate: bool = True,
    outline_step_mm: float = OUTLINE_STEP_MM,
) -> list:
    """Motor vectorial: por región, contorno → underlay → relleno dirigido.

    Recorre cada componente conexa de cada color (`scipy.ndimage.label`) y
    por cada una emite, en orden y con su "rowkey":
      - tipo 0: contorno (running stitch de los anillos exterior + agujeros),
      - tipo 1: underlay ligero (edge-run interior, sin entrar en los huecos),
      - tipo 2: relleno Tatami vectorial a `FILL_ANGLE_DEG` (45°).
    El enrutado (`order_tatami_rows`) agrupa por color y zigzaguea por filas
    conservando el recorrido nativo de cada región/elemento. Las regiones se
    ordenan por centroides con nearest-neighbor, sin fusionarlas, de modo que
    el orden "contorno → underlay → relleno" se mantiene y los saltos/TRIMs se
    limitan a zonas realmente separadas del mismo color.

    Devuelve `(eventos, barreras_de_hueco)`. La segunda salida es un dict
    `{código: máscara_bool}` a imagen completa con el interior de los huecos
    que la jerarquía de contornos detectó, erosionado `HOLE_BARRIER_ERODE_PX`
    para no penalizar el running stitch que se cose sobre el borde del hueco.
    El enrutado lo usa para impedir que un puente con la aguja abajo atraviese
    un contador aunque la máscara dilatada lo dé por bueno.
    """
    events: list = []
    hole_masks: dict = {
        c: np.zeros(codes.shape, dtype=bool) for c in _unique_color_codes(codes)
    }

    def add(ev, reg, kind, key):
        events.append((ev[0], ev[1], ev[2], (reg, kind, key)))

    for code in _unique_color_codes(codes):
        mask = codes == code
        labels, ncomp = ndimage.label(mask)
        if ncomp == 0:
            continue
        for reg, (label, sl) in enumerate(_ordered_region_slices(labels, mmp)):
            comp = labels[sl] == label
            r0, c0 = sl[0].start, sl[1].start
            hierarchy = _contour_hierarchy(comp, mmp)
            if hierarchy.hole_mask.any():
                barrier = ndimage.binary_erosion(
                    hierarchy.hole_mask,
                    structure=np.ones((2 * HOLE_BARRIER_ERODE_PX + 1,) * 2),
                )
                if barrier.any():
                    hole_masks[code][sl] |= barrier
            if outline:
                for k, ev in enumerate(
                    _outline_vectorize(
                        hierarchy.outer + hierarchy.holes,
                        r0,
                        c0,
                        code,
                        mmp,
                        outline_step_mm,
                    )
                ):
                    add(ev, reg, 0, k)
            if underlay:
                for k, ev in enumerate(
                    _underlay_edge_run(
                        comp,
                        r0,
                        c0,
                        code,
                        mmp,
                        UNDERLAY_STEP_MM,
                        hierarchy.hole_mask,
                    )
                ):
                    add(ev, reg, 1, k)
            for ev in _fill_region_vector(
                comp, r0, c0, code, mmp, step, stitch_len, rotate, hierarchy
            ):
                add((ev[0], ev[1], ev[2]), reg, 2, ev[3])
    # Una isla del mismo color que cae dentro del hueco de OTRA componente (un
    # punto dentro de la "O" de la "Q", el corazón de un "@") no es un hueco:
    # la componente que aporta el hueco no la ve, así que hay que restar el
    # sólido del color DESPUÉS de recorrer todas las regiones. Sin esta resta
    # el enrutado trataría la isla como infranqueable y cortaría de más.
    for code, mask in hole_masks.items():
        if mask.any():
            hole_masks[code] = mask & ~(codes == code)
    return events, {c: m for c, m in hole_masks.items() if m.any()}


def _drop_tiny_holes(
    hole_masks: dict,
    mmp: float,
    min_area_mm2: float | None,
) -> dict:
    """Quita de las barreras de hueco los componentes por debajo de `min_area_mm2`.

    Aplica SOLO sobre las barreras que consume el enrutado (`hole_masks`), nunca
    sobre lo que ya se bordó: el relleno y el underlay se generaron con el
    hueco dentro, así que la geometría de la costura no cambia en absoluto.
    Lo único que cambia es si el enrutado considera el hueco infranqueable.

    Motivo (medido): la jerarquía de contornos ve huecos de 1 px que sobreviven
    a `HOLE_BARRIER_ERODE_PX` porque el erode se aplica a la máscara del
    componente, no a cada hueco. En `Captura de logo free fire.png` son 160
    huecos de los que 105 miden menos de 4 mm² y muchos son un único píxel de
    0.29 mm. Cada uno obligaba a un `TRIM_SENTINEL` para mover la aguja ~1.2 mm
    (`OUTLINE_STEP_MM`): cortar el hilo para avanzar un milímetro. Un hueco de
    esa área no es el contador de una letra, es el redondeo de un borde, y la
   aguja puede cruzarlo con el hilo echado sin que se note.

    `min_area_mm2` en mm², con `mmp` como mm por píxel del plano. `None` o
    <= 0 devuelve los mapas sin tocar (comportamiento previo intacto).
    """
    if not hole_masks or min_area_mm2 is None or min_area_mm2 <= 0 or not mmp or mmp <= 0:
        return hole_masks
    min_px = min_area_mm2 / (mmp * mmp)
    out = {}
    for code, mask in hole_masks.items():
        if mask is None or not mask.any():
            out[code] = mask
            continue
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[:2]
        if count <= 1:
            out[code] = mask
            continue
        # stats[:, 0] = label, [:, 1] = area en píxeles.
        areas = np.bincount(labels.ravel(), minlength=count)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = areas[1:] >= min_px
        out[code] = keep[labels]
    return out


def _despeckle_codes(
    codes: np.ndarray,
    size: int = 5,
) -> np.ndarray:
    """Mediana `size`x`size` sobre el plano de códigos de hilo (fuente opcional).

    Elimina las motas/contiguos de unos píxeles que deja una imagen JPEG
    comprimida (ruido de bloques y anillos en bordes de letra): son regiones
    de 3-6 mm² que ni el closing morfológico ni `_drop_tiny_regions` quitan
    y que disparan TRIMs de replegado (en `Free_Fire_Logo.jpg` quedaban ~120
    por color). El píxel toma el código mayoritario de su vecindario; si el
    fondo es mayoría, se vuelve fondo.

    A DIFERENCIA de `_morph_close_codes` y `_drop_tiny_regions`, esto SÍ
    cambia el contenido del diseño (elimina componentes de borde), por eso es
    opcional y no se aplica por defecto. Coste O(píxeles) (una mediana
    separable de scipy; no escala con el número de regiones).
    """
    if size < 2:
        return codes
    uniq = sorted(_unique_color_codes(codes))
    lut = {c: i for i, c in enumerate(uniq)}
    arr = np.full(codes.shape, -1, dtype=np.int16)
    for c, i in lut.items():
        arr[codes == c] = i
    med = ndimage.median_filter(arr, size=size, mode="constant", cval=-1)
    out = codes.copy()
    for c, i in lut.items():
        out[med == i] = c
    out[med < 0] = None
    return out


def _drop_tiny_regions(
    codes: np.ndarray,
    mmp: float,
    min_area_mm2: float | None = MIN_REGION_AREA_MM2,
) -> np.ndarray:
    """Omite del relleno las regiones con área < `min_area_mm2` mm².

    Devuelve una COPIA de `codes` con los píxeles de esas regiones puestos a
    None (fondo). En unidades de mm² el umbral es invariante a la resolución:
    a 1000 px (mmp 0.13) un píxel son ~0.017 mm²; a 4167 px (mmp 0.031) ~0.001
    mm², así que la misma mota de anti-aliasing se sigue descartando a
    cualquier escala. Coste O(píxeles): un `ndimage.label` por color (no
    escala con el número de regiones). No altera las estadísticas de la
    máscara original (p. ej. `diagnose_color_masks`): la "isla real" se
    conserva en `codes` sin cambios.

    `None` o <= 0 desactiva el filtro.

    Un color NUNCA se borra entero: la región más grande de cada color se
    conserva aunque esté por debajo del umbral. Sin esta salvaguarda un color
    claro que la reducción de paleta deja fragmentado en motas (un turquesa de
    borde, un gris claro anti-aliaseado) desaparece del bordado por completo y
    el usuario ve un área grande sin rellenar, cuando en realidad lo que se ha
    descartado es el ruido que lo rodeaba. Descartar el ruido que sí rodea a una
    zona real es lo que hace el filtro; borrar el color entero es un cambio de
    diseño, y para eso está el despeckle, que es optativo.
    """
    if min_area_mm2 is None or min_area_mm2 <= 0:
        return codes
    thresh_px = min_area_mm2 / (mmp * mmp)
    if thresh_px <= 1:
        return codes
    out = codes.copy()
    for code in _unique_color_codes(codes):
        mask = codes == code
        labels, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        keep = set(np.flatnonzero(sizes >= thresh_px) + 1)
        # La mayor región del color se conserva siempre: el color no desaparece.
        keep.add(int(np.argmax(sizes)) + 1)
        small = np.flatnonzero(sizes < thresh_px) + 1
        drop = np.array([lb for lb in small if lb not in keep], dtype=int)
        if len(drop):
            out[np.isin(labels, drop)] = None
    return out


def _tatami_fill_regionwise(
    codes: np.ndarray,
    mmp: float,
    step: float,
    stitch_len: float | None = None,
    outline: bool = True,
    rotate: bool = True,
    outline_step_mm: float = OUTLINE_STEP_MM,
) -> list:
    """Relleno por región (orientación propia + contorno) que reutiliza `_tatami_fill`.

    Recorre cada componente conexa de cada color (`scipy.ndimage.label`) y
    por cada una emite, en orden: (opcional) el running stitch de contorno
    (`_outline_stitches`) y luego el relleno Tatami (opcionalmente rotado
    según su eje mayor, `_fill_region_tatami`). Antes de devolver las
    puntadas en el mismo formato que `_tatami_fill` (eventos con
    `JUMP_SENTINEL`), que el enrutado sigue agrupando por color. Coste total
    O(píxeles) para el label + O(área)/O(perímetro) por región, sin
    estructuras que escalen con el número de regiones.

    Monta una "rowkey" (4-tupla) a cada evento mientras dura el enrutado:
    `(índice_de_región, tipo, clave)` para que `_route_one_color` conserve el
    recorrido de cada región ANTES de decidir salto o continuidad. El tipo 0
    son los contornos (clave = orden nativo del running stitch, para que el
    borde se recorra sin reordenar) y el tipo 1 son las filas scanline del
    relleno (clave = fila del frame posiblemente rotado sobre el que corrió
    el relleno, para reconstruir el zigzag ida-y-vuelta entre direcciones
    aunque las puntadas de una fila ya no compartan Y global). Así el
    enrutado no fragmenta regiones en filas de una puntada ni dispara TRIM
    entre filas.
    """
    events: list = []

    def add(ev, reg, kind, key):
        events.append((ev[0], ev[1], ev[2], (reg, kind, key)))

    for code in _unique_color_codes(codes):
        mask = codes == code
        labels, ncomp = ndimage.label(mask)
        if ncomp == 0:
            continue
        for reg, (label, sl) in enumerate(_ordered_region_slices(labels, mmp)):
            comp = labels[sl] == label
            r0, c0 = sl[0].start, sl[1].start
            if outline:
                for k, ev in enumerate(
                    _outline_stitches(comp, r0, c0, code, mmp, outline_step_mm)
                ):
                    add(ev, reg, 0, k)
            for ev in _fill_region_tatami(
                comp, r0, c0, code, mmp, step, stitch_len, rotate
            ):
                add(ev, reg, 1, ev[3])
    return events


def group_stitches_by_color(stitches: list) -> list:
    """Reagrupa las puntadas por código de color para minimizar cambios.

    Todas las puntadas de un mismo color quedan juntas en el primer orden de
    aparición (estable). Dentro de cada color se conserva el orden original
    (scanline de arriba hacia abajo). Cada movimiento de salto (marcador
    JUMP_SENTINEL o TRIM_SENTINEL) se mantiene pegado a la puntada que le
    precede/sigue, con
    lo que los desplazamientos con aguja arriba entre zonas del mismo color
    se conservan sin dibujarse y los cambios de color quedan cerca del
    número de colores usados (11-20) en lugar de miles.
    """
    groups = {}          # código -> lista de eventos (saltos y puntadas)
    pending = []         # saltos agrupados con la siguiente puntada
    for item in stitches:
        if item[2] in MOVE_SENTINELS:
            pending.append(item)
            continue
        code = item[2]
        if code not in groups:
            groups[code] = []
        groups[code].extend(pending)
        pending.clear()
        groups[code].append(item)
    if pending and groups:
        next(reversed(groups.values())).extend(pending)

    ordered = []
    for group in groups.values():
        ordered.extend(group)
    return ordered


def _segment_in_solid(
    a: tuple,
    b: tuple,
    solid_mask: np.ndarray,
    mmp: float,
) -> bool:
    """True si el segmento recto a→b (mm, marco global) queda ÍNTEGRO dentro
    de la máscara sólida de ESTE color.

    Convierte ambos extremos a píxeles globales (px = mm/mmp, redondeados) y
    muestrea la recta en pasos de 1 px. Devuelve False si CUALQUIER muestra
    cae fuera de la máscara (un píxel de fondo, o el interior de un
    hueco/contador de letra que el relleno nunca toca). Es el test de cruce de
    fondo del puente (regla nº1/nº4 del proyecto): el enrutado solo une con la
    aguja abajo cuando el desplazamiento queda DENTRO de la misma región
    sólida continua; si el segmento cruza un hueco/contador o una zona
    realmente separada del mismo color se fuerza un salto con TRIM_SENTINEL
    (corte obligatorio) para no cubrir nunca contadores ni interiores vacíos.
    `None`/`mmp <= 0` significan "sin criterio de máscara" → True (solo
    distancia).
    """
    if solid_mask is None or mmp is None or mmp <= 0:
        return True
    h, w = solid_mask.shape
    x0, y0 = round(a[0] / mmp), round(a[1] / mmp)
    x1, y1 = round(b[0] / mmp), round(b[1] / mmp)
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    for k in range(n + 1):
        x = round(x0 + (x1 - x0) * k / n) if n else x0
        y = round(y0 + (y1 - y0) * k / n) if n else y0
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        if not solid_mask[y, x]:
            return False
    return True


def _segment_off_color(
    a: tuple,
    b: tuple,
    own_mask: np.ndarray | None,
    mmp: float | None,
    max_mm: float = OFF_COLOR_MAX_MM,
) -> bool:
    """True si el segmento a→b se tiende sobre un píxel que NO es este color.

    Cubre los dos casos a la vez, y esa es la razón del nombre "off_color": otro
    color de la tabla Y el fondo. `own_mask` es la máscara del color que se está
    bordando SIN dilatar, así que cualquier píxel ajeno cuenta. Distinguirlos
    exigiría dos máscaras y no vale la pena: el enrutado los trata igual (la
    aguja se levanta en los dos casos) y lo que NO puede hacer es tratar el
    segundo como si fuera el primero, que es lo que le pasaba a la silueta de
    `core/mockup.py`.

    Es la autoridad que impide coser a través de otro hilo: con solo la máscara
    dilatada 5x5 el hilo quedaba tendido hasta 11 mm sobre el bordado. Se mide
    la racha MÁS LARGA de muestras ajenas en lugar de contar el total, porque el
    relleno se cose pegado al contorno y su segmento de enganche siempre toca
    alguna columna de fondo en el borde: sin ese margen, cada fila del relleno
    se partiría en varias piezas y volvería el problema de exceso de cortes que
    esto viene a arreglar.

    `max_mm` es la racha tolerada (`OFF_COLOR_MAX_MM`, ver la nota de
    calibración). `own_mask`/`mmp` a `None` significan "sin plano de colores" →
    False (no se inventa un cruce), que es lo que pasa en las rutas sin máscara.
    """
    if own_mask is None or mmp is None or mmp <= 0 or max_mm <= 0:
        return False
    h, w = own_mask.shape
    x0, y0 = round(a[0] / mmp), round(a[1] / mmp)
    x1, y1 = round(b[0] / mmp), round(b[1] / mmp)
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    if n <= 1:
        return False
    tol_px = max_mm / mmp
    run = 0
    for k in range(n + 1):
        x = round(x0 + (x1 - x0) * k / n)
        y = round(y0 + (y1 - y0) * k / n)
        if x < 0 or y < 0 or x >= w or y >= h:
            return True
        if own_mask[y, x]:
            run = 0
        else:
            run += 1
            if run > tol_px:
                return True
    return False


def _segment_interior_in_solid(
    a: tuple,
    b: tuple,
    solid_mask: np.ndarray | None,
    mmp: float | None,
) -> bool:
    """Como `_segment_in_solid` pero ignora los dos píxeles extremos.

    El contorno de una región corre justo sobre su borde, así que el redondeo
    a píxel de un extremo puede caer en la primera columna de fondo y hacer
    fallar un tramo que en realidad NO cruza fondo (típico en las transiciones
    contorno → underlay → relleno de cada componente). Aquí solo se exige que
    el recorrido INTERIOR quede dentro de la máscara sólida; si el interior
    tiene un hueco de fondo (una separación real) devuelve False.
    `None`/`mmp <= 0` → True (sin criterio de máscara).
    """
    if solid_mask is None or mmp is None or mmp <= 0:
        return True
    h, w = solid_mask.shape
    x0, y0 = round(a[0] / mmp), round(a[1] / mmp)
    x1, y1 = round(b[0] / mmp), round(b[1] / mmp)
    n = max(abs(x1 - x0), abs(y1 - y0))
    if n <= 1:
        return True
    for k in range(1, n):
        x = round(x0 + (x1 - x0) * k / n)
        y = round(y0 + (y1 - y0) * k / n)
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        if not solid_mask[y, x]:
            return False
    return True


def _segment_in_hole(
    a: tuple,
    b: tuple,
    hole_mask: np.ndarray | None,
    mmp: float | None,
) -> bool:
    """True si el segmento recto a→b ENTRA en el interior de un hueco.

    Usa la máscara de huecos que produce la jerarquía de contornos
    (`_contour_hierarchy`), que es precisa: a diferencia de la máscara sólida
    dilatada 5×5, no puentea ni un hueco estrecho. Es el criterio que impide
    que un puente con la aguja abajo cruce el interior de un contorno hijo
    (contador de letra, agujero). `None`/`mmp <= 0` → False (sin huecos).
    """
    if hole_mask is None or mmp is None or mmp <= 0:
        return False
    return _segment_in_solid(a, b, hole_mask, mmp)


def _is_intra_region_stage_transition(previous: tuple, current: tuple) -> bool:
    if len(previous) < 4 or len(current) < 4:
        return False
    previous_key = previous[3]
    current_key = current[3]
    if not isinstance(previous_key, tuple) or not isinstance(current_key, tuple):
        return False
    if len(previous_key) < 2 or len(current_key) < 2:
        return False
    return previous_key[0] == current_key[0] and previous_key[1] != current_key[1]


def _is_same_region(previous: tuple, current: tuple) -> bool:
    """True si ambas puntadas pertenecen a la MISMA componente conexa.

    La rowkey que emite `_vector_fill_regionwise` es `(region, etapa, clave)`,
    así que basta comparar `region` (el primer campo) sin exigir el mismo
    tipo de etapa. Una componente puede fragmentarse en varias filas de
    relleno (`kind 2`) o anillos de contorno (`kind 0`) que el enrutado
    recorre por separado; mientras el segmento recto quede dentro de la
    máscara sólida, unirlas con la aguja abajo es seguro aunque superen el
    umbral de distancia.
    """
    if len(previous) < 4 or len(current) < 4:
        return False
    previous_key = previous[3]
    current_key = current[3]
    if not isinstance(previous_key, tuple) or not isinstance(current_key, tuple):
        return False
    if len(previous_key) < 1 or len(current_key) < 1:
        return False
    return previous_key[0] == current_key[0]


def _append_intra_region_move(out: list, previous: tuple, current: tuple, code: str) -> None:
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    steps = max(
        1,
        math.ceil(max(abs(dx), abs(dy)) / INTRA_REGION_CONNECTOR_MAX_MM),
    )
    for step in range(1, steps + 1):
        ratio = step / steps
        out.append(
            (
                previous[0] + dx * ratio,
                previous[1] + dy * ratio,
                code,
            )
        )


def _rowkey_sort(key) -> tuple:
    """Clave de ordenación total para las rowkeys del enrutado.

    Las rowkeys son `(region, etapa, clave)`. La `clave` del relleno vectorial es
    a su vez `(fila, calle)` (`_assign_fill_strips`), mientras que la del
    contorno y el underlay es un entero plano. Sin normalizar, comparar un
    entero con una tupla lanza `TypeError` al ordenar, así que aquí ambas
    formas se aplanan a la misma tupla de cuatro enteros.

    El orden es `(region, etapa, CALLE, fila)`, y es deliberado: ordenar por
    fila primero haría que el enrutado zigzagueara de un tramo al siguiente
    DENTRO de cada fila y volviera al primero en la fila siguiente, eso es, un
    vaivén horizontal por cada fila con tantos saltos como tramos tenga, y en
    una forma con muchas calles (las patas de un elefante, los dos arcos de
    una "M") esos saltos son largos. Ordenando por calle primero se recorre
    una columna entera de arriba abajo con la aguja abajo y solo se vuelve al
    empezar la columna siguiente: los tramos de una misma fila dejan de
    coserse entre sí y el número de saltos baja al de columnas, no al de filas.
    """
    if not isinstance(key, tuple):
        return (key, 0, 0, 0)
    region = key[0] if len(key) > 0 else 0
    kind = key[1] if len(key) > 1 else 0
    rest = key[2] if len(key) > 2 else 0
    if isinstance(rest, tuple):
        row = rest[0] if len(rest) > 0 else 0
        strip = rest[1] if len(rest) > 1 else 0
        return (region, kind, strip, row)
    return (region, kind, 0, rest)


def _route_one_color(
    tokens: list,
    code: str,
    connect_mm: float | None = None,
    solid_mask: np.ndarray | None = None,
    mmp: float | None = None,
    hole_mask: np.ndarray | None = None,
    mockup=None,
    bridge_gap_mm: float = 0.0,
    own_mask: np.ndarray | None = None,
) -> list:
    """Reordena los eventos de un único color en filas zigzag.

    `tokens` son parejas (inicio, fin) de las puntadas cortas generadas por
    el relleno, ya todas del mismo código. Se agrupan por fila (mismo Y) y
    dentro de cada fila se recorren de izquierda a derecha y, en la fila
    siguiente, de derecha a izquierda (ida y vuelta). Entre dos puntadas
    consecutivas del mismo color:

    - Si la distancia es <= el umbral (1.8x la mediana del paso de la fila,
      nunca menos de ROW_CONNECT_MM, y nunca menos de `connect_mm` si se
      indica) la aguja sigue abajo y se emite solo el final de la puntada (el
      desplazamiento se dibuja como una puntada corta más, sin salto).
    - Si la distancia es mayor pero los dos puntos pertenecen a la MISMA
      componente conexa (mismo `region` en la rowkey) y el segmento recto
      queda dentro de la máscara sólida —basta el interior del segmento: se
      tolera el redondeo de los extremos al borde—, se mantiene la
      continuidad con la aguja abajo. Cubre tanto las transiciones entre
      etapas (contorno → underlay → relleno) como las de una misma etapa
      (fila → fila del relleno, anillo → anillo del contorno): sin esta regla
      cada una gastaría un salto extra.
    - Si el segmento sale de la máscara sólida (fondo real entre zonas
      separadas del mismo color) se emite el par salto + puntada con aguja
      arriba marcado como JUMP_SENTINEL: el exportador decide el corte por
      distancia (TRIM solo si supera TRIM_MIN_JUMP_DISTANCE_MM). Un salto
      sobre fondo NO obliga a cortar la hebra; forzar un TRIM por cada
      transición era la principal fuente de trims/1000 frente al perfil.
    - Si el segmento cruza el interior de un hueco/contador (`hole_mask`) el
      salto se marca TRIM_SENTINEL: el corte es obligatorio para no tender
      hilo sobre el vacío del contador, y no depende de la distancia.

    `mockup` + `bridge_gap_mm` (opcionales, ambos necesitan estar presentes)
    añaden la única excepción a esa regla: si el salto ya está descartado por
    cruzar fondo, pero la silueta simplificada (`core.mockup`) dice que entre
    los dos puntos hay MENOS de `bridge_gap_mm` de vacío, entonces ese vacío no
    es separación real sino el redondeo de una esquina, y el hilo sigue abajo.
    El tramo se divide con `_append_intra_region_move` para no activar el
    fallback de salto del exportador. Se excluye por construcción el cruce de
    un hueco real: la comprobación es `not crosses_hole`, que siempre gana.

    El umbral se calibra contra la resolución de la propia silueta, no contra
    un número redondo: con 1.56 mm/px, 2.0 mm es ~1.3 px, el tamaño del error de
    discretización. Vacíos de 6-12 mm (la mayoría de los cortes que quedan) son
    separación real y este umbral deliberadamente no los toca.

    `hole_mask` es el interior de los huecos que detectó la jerarquía de
    contornos. Un puente con la aguja abajo NUNCA puede entrar en él, ni
    siquiera cuando la máscara sólida dilatada lo daría por bueno: es el
    criterio que mantiene abiertos los contadores de las letras.

    `own_mask` es la máscara del color SIN dilatar y es la autoridad sobre
    `solid_mask`: mientras el segmento no se tienda sobre otro color o sobre el
    fondo no se cose. Sin ella la máscara dilatada 5×5 daba por bueno cruzar
    separaciones de hasta 4 px y el hilo quedaba unido entre los dos arcos de
    una "M" o entre el cuerpo y una pata, tendido sobre hilo ya cosido. Se tolera
    una racha de `OFF_COLOR_MAX_MM` de píxeles ajenos porque el relleno se cose
    pegado al contorno y su enganche siempre toca el borde
    (`_segment_off_color`).

    `connect_mm` (piso configurable, ver CONNECT_NO_TRIM_MM) amplía el umbral
    normal hacia arriba para NO cortar la hebra entre trazos/regiones del
    mismo color separados por pocos milímetros. `None` o <= 0 deja solo el
    criterio adaptativo. La continuidad intra-región no depende de
    `connect_mm`: solo exige que el segmento quede dentro de la máscara.

    No se modifica la densidad ni el número de puntadas de relleno: solo
    cambia el orden de recorrido y cuántos saltos sobran. Un conector
    intraregión excepcionalmente largo se divide en tramos de puntada para no
    activar el fallback de salto del exportador.

    Los eventos pueden ser 3-tuplas (x, y, código) o 4-tuplas con una
    "rowkey" (x, y, código, rowkey). Con 3-tuplas las filas se agrupan por la
    coordenada Y exacta (relleno horizontal clásico). Con 4-tuplas la rowkey
    identifica la fila del relleno local (p. ej. la fila scanline de una
    región ya rotada, cuyas puntadas no comparten Y global): rowkeys únicas
    conservan el recorrido contiguo de cada región y el zigzag alterna solo
    entre aquellas que comparten fila.
    """
    rows = {}
    for j, s in tokens:
        key = s[3] if len(s) > 3 else s[1]
        rows.setdefault(key, []).append((j, s))

    # Umbral adaptativo al paso real de este color: la distancia euclídea
    # entre puntos CONSECUTIVOS de cada fila en ORDEN de emisión (no entre
    # puntos ordenados por X). Ordenar por X deja gaps degenerados de 0 mm
    # cuando la región rotada corre vertical (todas las puntadas de una fila
    # comparten X global) y la mediana se hunde a 0 -> el enrutado volvería a
    # tratar cada puntada larga (~3 mm) como un salto. Se descartan los
    # enlaces <0.5 mm (duplicados/proyecciones colapsadas) para que la
    # mediana capture el paso largo del relleno cuando existe.
    gaps = []
    for y in rows:
        pts = [s for _, s in rows[y]]
        for a, b in zip(pts, pts[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            if d >= 0.5:
                gaps.append(d)
    pitch = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    limit = max(ROW_CONNECT_MM, 1.8 * pitch)
    if connect_mm and connect_mm > 0:
        limit = max(limit, connect_mm)
    limit2 = limit * limit

    out = []
    prev = None
    for r, key in enumerate(sorted(rows, key=_rowkey_sort)):
        plist = rows[key]
        plist.sort(key=lambda js: js[1][0])
        if r % 2:
            plist.reverse()
        for j, s in plist:
            connected = False
            intra_region_stage = False
            solid = False
            crosses_hole = False
            off_color = False
            if prev is not None:
                distance_ok = (s[0] - prev[0]) ** 2 + (s[1] - prev[1]) ** 2 <= limit2
                # Cruzar OTRO color con la aguja abajo tendería el hilo sobre un
                # hilo que ya está cosido. Va antes que la máscara dilatada y es
                # el criterio que más manda: solo la racha mínima de píxeles
                # ajenos del enganche al borde es tolerable.
                off_color = _segment_off_color(prev, s, own_mask, mmp)
                # El interior de un contador/hueco es infranqueable con la
                # aguja abajo y, además, obliga a cortar (ver rama else).
                crosses_hole = _segment_in_hole(prev, s, hole_mask, mmp)
                # Región continua Y sin entrar en el interior de un hueco: los
                # dos criterios son obligatorios para coser con la aguja abajo.
                # Se tolera que el borde redondee algún extremo fuera de la
                # máscara: si el interior del segmento sí está dentro, no se
                # cruza fondo real (evita saltos/TRIM artificiales por cada
                # transición contorno → underlay → relleno de cada componente).
                solid = (
                    (
                        solid_mask is None
                        or mmp is None
                        or _segment_in_solid(prev, s, solid_mask, mmp)
                        or _segment_interior_in_solid(prev, s, solid_mask, mmp)
                    )
                    and not crosses_hole
                    and not off_color
                )
                intra_region_stage = _is_intra_region_stage_transition(prev, s)
                same_region = _is_same_region(prev, s)
                # Continuidad intra-región: si el segmento recto queda dentro de
                # la máscara sólida, se cose con la aguja abajo aunque supere el
                # umbral de distancia. Vale para CUALQUIER transición de la
                # misma componente (fila→fila de relleno, anillo→anillo del
                # contorno), no solo contorno→underlay→relleno: son las 171
                # separaciones por distancia que no cruzaban fondo.
                long_connector = False
                if distance_ok and solid:
                    connected = True
                elif (
                    same_region
                    and solid_mask is not None
                    and mmp is not None
                    and solid
                ):
                    connected = True
                    long_connector = True
                elif (
                    mockup is not None
                    and bridge_gap_mm > 0.0
                    and not crosses_hole
                    and not off_color
                    and mmp is not None
                ):
                    # Fallo del test exacto de máscara, pero la silueta dice
                    # que el vacío es de redondeo: se puentea con aguja abajo.
                    # `crosses_hole` ya queda excluido arriba, así que un
                    # contador real nunca entra por esta rama. `off_color`
                    # también, y por una razón que conviene no olvidar: la
                    # silueta es de 160 px y NO distingue "otro color" de
                    # "fondo", así que no puede ser la autoridad para tender el
                    # hilo sobre un hilo que ya está cosido.
                    #
                    # OJO: en la práctica esta rama es casi inalcanzable con
                    # `OFF_COLOR_MAX_MM = 1.5`. Si `off_color` es False, el
                    # segmento no tiene ninguna racha ajena de más de 1.5 mm, y
                    # la máscara dilatada 5×5 (2 px por lado) absorbe de sobra
                    # una racha así, o sea que `solid` ya sale True y la aguja se
                    # mantiene abajo por la vía normal. Medido: los tres modos de
                    # guía dan streams byte a byte iguales en McDonalds y en el
                    # elefante, y difieren en 1 puntada en NeithLoom. Antes de que
                    # `off_color` gobernara, esta rama recortaba un 13-36% de
                    # los cortes; ahora ese recorte ya lo hace la comprobación
                    # exacta. No la desconectes esperando que vuelva a hacer
                    # trabajo: si alguna vez hay que recuperarlo, hay que subir
                    # `OFF_COLOR_MAX_MM`, no quitar el `and not off_color`.
                    if mockup.gap_mm((prev[0], prev[1]), (s[0], s[1])) <= bridge_gap_mm:
                        connected = True
                        long_connector = True
            if connected:
                if (
                    (intra_region_stage or long_connector)
                    and solid_mask is not None
                    and mmp is not None
                ):
                    _append_intra_region_move(out, prev, s, code)
                else:
                    out.append((s[0], s[1], code))
            else:
                # Cortar la hebra es obligatorio en dos casos, y ambos son
                # "el hilo quedaría tendido sobre algo que no es esta pieza":
                # cruzar el interior de un hueco/contador, o cruzar otro color o
                # el fondo. El segundo no depende de la distancia: un salto de
                # 3 mm por encima de otra zona igual deja hilo sobre el bordado,
                # así que se marca TRIM aunque el exportador no lo haría por su
                # regla de los 6 mm.
                force_trim = (crosses_hole or off_color) and prev is not None
                marker = TRIM_SENTINEL if force_trim else JUMP_SENTINEL
                out.append((j[0], j[1], marker))
                out.append((s[0], s[1], code))
            prev = s
    return out


def order_tatami_rows(
    stitches: list,
    connect_mm: float | None = None,
    solid_masks: dict | None = None,
    mmp: float | None = None,
    hole_masks: dict | None = None,
    mockup=None,
    bridge_gap_mm: float = 0.0,
    own_masks: dict | None = None,
) -> list:
    """Reagrupa por color y reordena cada color en filas zigzag (ida y vuelta).

    Reutiliza `group_stitches_by_color` para dejar cada color en un único
    bloque (un cambio de hilo por color) y dentro de cada bloque reordena las
    puntadas por filas alternando la dirección, de modo que la aguja apenas
    se levanta (ver `_route_one_color`) y el número de saltos cae de uno por
    puntada a unos pocos por color sin tocar la densidad ni el relleno.

    `connect_mm` se delega a `_route_one_color`: piso opcional (mm) del umbral
    de no-cortar, uniforme dentro de la fila y entre componentes del mismo
    color (ver CONNECT_NO_TRIM_MM).

    `solid_masks` (opcional, `{código: máscara_bool}`) + `mmp` activan el
    test de cruce de fondo del puente (regla nº1/nº4 del proyecto): cuando
    están presentes, el enrutado NO corta solo por distancia, sino que además
    exige que el segmento recto entre la última puntada y la siguiente quede
    dentro de la máscara sólida de ese color. Cualquier transición entre
    puntos de la MISMA componente (mismo `region` de la rowkey: etapa → etapa,
    fila → fila del relleno, anillo → anillo del contorno) se mantiene
    continua aunque supere el umbral normal si el segmento sigue dentro de la
    máscara. Si el segmento cruza un hueco/contador real, se conserva el salto
    con TRIM_SENTINEL (corte obligatorio); si solo cruza fondo entre zonas
    separadas del mismo color, el salto queda como JUMP_SENTINEL y el
    exportador decide el corte por distancia. Un puente nunca cubre fondo ni
    huecos aunque la distancia esté por debajo del umbral. Sin
    `solid_masks`/`mmp` el comportamiento es idéntico al anterior (solo umbral
    de distancia).

    `hole_masks` (opcional, `{código: máscara_bool}`) añade el criterio de la
    jerarquía de contornos: el interior de un hueco detectado dentro de una
    región (contador de letra, agujero) es infranqueable con la aguja abajo.
    Es más estricto que `solid_masks`, que va dilatada 5×5 y por eso puentea
    huecos estrechos; aquí manda la geometría real del contorno hijo.
    Sin `hole_masks` el enrutado se comporta como antes.

    `mockup` + `bridge_gap_mm` (opcionales) delegan en `core.mockup` la última
    decisión: cuando el tramo ya está descartado por cruzar fondo, pero la
    silueta simplificada dice que entre los dos puntos hay menos de
    `bridge_gap_mm` de vacío, se puentea con la aguja abajo porque ese vacío es
    redondeo de borde y no separación real. Un hueco real nunca se puentea (el
    criterio de `hole_masks` manda siempre). Sin `mockup` o con
    `bridge_gap_mm <= 0` el comportamiento es idéntico al anterior.
    """
    grouped = group_stitches_by_color(stitches)
    out = []
    i, n = 0, len(grouped)
    while i < n:
        code = None
        k = i
        while k < n and grouped[k][2] in MOVE_SENTINELS:
            k += 1
        if k >= n:
            break
        code = grouped[k][2]
        tokens = []
        j = i
        while j < n:
            if grouped[j][2] == code:
                tokens.append((grouped[j], grouped[j]))
                j += 1
            elif (
                grouped[j][2] in MOVE_SENTINELS
                and j + 1 < n
                and grouped[j + 1][2] == code
            ):
                tokens.append((grouped[j], grouped[j + 1]))
                j += 2
            elif grouped[j][2] in MOVE_SENTINELS:
                j += 1
            else:
                break
        if tokens:
            out.extend(
                _route_one_color(
                    tokens,
                    code,
                    connect_mm,
                    solid_masks.get(code) if solid_masks else None,
                    mmp,
                    hole_masks.get(code) if hole_masks else None,
                    mockup,
                    bridge_gap_mm,
                    own_masks.get(code) if own_masks else None,
                )
            )
        i = j
    return out


@resource_logger.measure_resources("generate_stitches")
def generate_stitches(
    image: Image.Image,
    threads_used: list,
    density: str = "Media",
    background_rgb: tuple | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
    background_mask: np.ndarray | None = None,
    width_mm: float | None = None,
    height_mm: float | None = None,
    morph_kernel: int | None = MORPH_KERNEL,
    stitch_length_mm: float | None = None,
    outline_running: bool = True,
    underlay: bool = True,
    rotate_fill: bool = True,
    outline_step_mm: float = OUTLINE_STEP_MM,
    min_area_mm2: float | None = MIN_REGION_AREA_MM2,
    codes_median: int | None = None,
    connect_mm: float | None = None,
    guide: str = "auto",
) -> StitchPattern:
    """Genera puntadas con el motor vectorial (contorno + underlay + relleno).

    Cada región de color se vectoriza en anillos (exterior + agujeros) y se
    borda en tres pasadas: contorno (`outline_running`), underlay ligero
    (`underlay`, edge-run interior ~`UNDERLAY_OFFSET_MM`) y relleno Tatami
    diagonal (`rotate_fill`, filas a `FILL_ANGLE_DEG` = 45°). Ver el docstring
    del módulo para los detalles del motor; las puntadas muestreadas a
    `STITCH_LENGTH_DEFAULT_MM` cubren el tramo de forma continua.

    El color de fondo se omite: si `background_rgb` se indica, se usa ese
    color con una tolerancia de ±`tolerance` por canal; si no, se detecta
    por las esquinas de la imagen. `background_mask` es una máscara booleana
    (mismo alto y ancho que la imagen) que marca píxeles que también deben
    tratarse como fondo (zonas eliminadas); se suma a lo anterior. Todos los
    demás colores (incluidos los blancos/claros) generan puntadas.

    `width_mm`/`height_mm` son las dimensiones del bastidor: la imagen se
    escala de forma proporcional (sin deformar) para que el bordado quepa
    dentro del bastidor. Si no se indican se usa el límite de 25x25 cm por
    defecto. Las coordenadas están en milímetros con densidad horizontal y
    vertical iguales.

    `morph_kernel` controla la limpieza morfológica previa al relleno (ver
    `_morph_close_codes`): un closing de lado `morph_kernel` píxeles por
    color que rellena los huecos de 1-2 px de anti-aliasing y evita que el
    scanline fragmente una misma región y el exportador dispare TRIM
    innecesarios. `None` o < 2 desactiva la limpieza (comportamiento previo).

    `stitch_length_mm` (mm), si se indica, amplía la separación entre
    puntadas a lo largo de cada tramo (p. ej. 3 mm) para reducir el conteo
    de puntadas; la separación entre filas de la densidad no cambia. Por
    defecto (`None`) se usa `STITCH_LENGTH_DEFAULT_MM` (3.0 mm, puntadas
    largas de tatami profesional en lugar del pinchado denso de ~0.3 mm).

    `min_area_mm2` (mm², default `MIN_REGION_AREA_MM2` = 0.5) omite del
    relleno las regiones subs-mm (`_drop_tiny_regions`): en unidades
    invariantes a la resolución, sin tocar el conteo de componentes de la
    máscara original. `codes_median` (None por defecto) aplica además un
    despeckle de mediana `codes_median`x`codes_median` sobre el plano de
    códigos (`_despeckle_codes`): quita el ruido de bloques de un JPEG
    comprimido (contiguos de 3-6 mm² en bordes de letra) que disparan TRIMs;
    A DIFERENCIA de `min_area_mm2`, esto SÍ elimina fragmentos del diseño,
    por eso va desactivado y se expone en la GUI como opción.

    `rotate_fill` (True por defecto) orienta el relleno de cada región a
    `FILL_ANGLE_DEG` (45°) en lugar de siempre en horizontal: los anillos se
    rotan (afín exacta) para alinear esa diagonal con la horizontal del frame,
    las filas scanline corren paralelas a ella y las puntadas se rotan de vuelta
    al sistema de la imagen.
    `outline_running` (True por defecto) borda además un running
    stitch sobre el contorno vectorizado (anillo exterior + agujeros, sin el
    escalón de píxel gracias a `approxPolyDP` + Chaikin) ANTES del underlay y
    del relleno de cada región, con separación `outline_step_mm` (1.2 mm por
    defecto, aligerado desde 0.8; el running se remuestrea por arco, así que
    esta es la palanca real de puntadas de borde y sigue por debajo del umbral
    de conexión del enrutado para que el borde no genere TRIM internos).
    `underlay` (True por defecto) añade entre ambos un edge-run interior
    (silueta erosionada `UNDERLAY_OFFSET_MM` mm) muestreado a
    `UNDERLAY_STEP_MM` (2.5 mm, ~1/3 de una segunda pasada completa: mucho más
    ligero que la versión heredada que copiaba el paso del contorno). Con
    `outline_running`, `underlay` y `rotate_fill` a False se obtiene el
    relleno único horizontal de la versión anterior vía `_tatami_fill` (los
    demás pasos iguales).

    `connect_mm` (`None` por defecto = solo el umbral adaptativo de
    `_route_one_color`) es el piso opcional, en milímetros, de la distancia
    a partir de la cual el enrutado deja de coser de forma continua y emite
    un salto (que el exportador convierte en TRIM solo si supera su umbral de
    corte por distancia). Desde la regla de máscara (regla nº1/nº4 del
    proyecto), el
    puente (aguja abajo) solo se emite cuando el desplazamiento queda ÍNTEGRO
    dentro de la misma región sólida continua del color. La única excepción
    de distancia es la transición entre etapas contiguas de una misma región:
    si el segmento queda dentro de la máscara, se divide en conectores cortos
    para evitar un TRIM artificial. `connect_mm` ya NO une zonas realmente
    separadas del mismo color (eso cubriría huecos y contadores), solo permite
    coser de forma continua hasta ese piso dentro de la propia región
    (cierra huecos de discretización del borde y micro-gaps de 1px del
    anti-alias). Los cruces reales de fondo/hueco siempre fuerzan un salto con
    TRIM_SENTINEL, que el exportador convierte en TRIM siempre; los saltos por
    distancia mantienen el umbral `TRIM_MIN_JUMP_DISTANCE_MM`. El
    default de la GUI (8.0 mm) se mantiene como piso de continuidad interior,
    no como puente entre zonas.

    `guide` (ver `core.mockup`) es la única palanca nueva y mueve una sola
    cosa: el área mínima (mm²) que un hueco debe tener para seguir siendo
    infranqueable con la aguja abajo, vía `_drop_tiny_holes`. No toca la
    densidad, ni el número de puntadas, ni el relleno, ni las coordenadas ni
    la escala, así que el bordado sale en las mismas coordenadas y con el
    mismo aspecto, solo que algunos huecos de ruido ya no obligan a cortar el
    hilo. `"faithful"` deja las barreras intactas y reproduce exactamente el
    comportamiento previo; `"auto"` (default) quita solo el ruido de un píxel;
    `"simple"` quita también los contadores pequeños. El mock interno
    (`core.mockup.build_mockup`) se construye una vez aquí para decidir si el
    diseño es simple —en cuyo caso el guide no se aplica y el resultado es el
    de siempre— y se descarta al terminar: no se guarda en el `StitchPattern`.
    """
    if width_mm is None:
        width_mm = MAX_SIZE_MM
    if height_mm is None:
        height_mm = MAX_SIZE_MM

    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return StitchPattern()

    # Escala mm por píxel: ajusta la imagen dentro del bastidor sin deformarla
    mmp = min(width_mm / w, height_mm / h)

    step = _step_mm(density)

    codes = build_color_codes(
        image, threads_used, background_rgb, tolerance, background_mask
    )

    # Cierre morfológico ligero por color: rellena huecos de un par de
    # píxeles (fragmentación falsa por anti-aliasing) sin unir islas reales.
    if morph_kernel:
        codes = _morph_close_codes(codes, morph_kernel)

    # Despeckle opcional de fuente (mediana `codes_median`×`codes_median`):
    # quita el ruido de bloques de un JPEG comprimido (contiguos de 3-6 mm² en
    # bordes de letra) que closing y área mínima no tocan y que generan TRIMs.
    # Cambia el diseño (elimina fragmentos de borde), por eso va desactivado.
    if codes_median:
        codes = _despeckle_codes(codes, codes_median)

    # Filtro de tamaño mínimo por región (default 0.5 mm²): las motas
    # subs-mm no se bordan y evitaban un TRIM de replegado cada una. Solo
    # afecta al relleno; la máscara de códigos original queda intacta.
    codes = _drop_tiny_regions(codes, mmp, min_area_mm2)

    hole_masks: dict = {}
    if outline_running or underlay or rotate_fill:
        stitches, hole_masks = _vector_fill_regionwise(
            codes, mmp, step, stitch_length_mm, outline_running, underlay,
            rotate_fill, outline_step_mm,
        )
    else:
        stitches = _tatami_fill(codes, mmp, step, stitch_length_mm)
    # Regla nº1/nº4 (cruce de fondo del puente): se entrega al enrutado la
    # máscara sólida REAL por color (el plano de códigos ya cerrado,
    # despeckleado y sin regiones subs-mm, el mismo que se borda) para que el
    # puente solo se emita cuando el desplazamiento queda ÍNTEGRO dentro de
    # esa región continua. Si el segmento cruza un píxel de fondo o de hueco
    # (contador de letra/zona separada del mismo color) el enrutado fuerza un
    # salto con TRIM_SENTINEL, que el exportador corta siempre. Así los
    # huecos y el interior de los contadores NUNCA se cubren.
    solid_masks = {
        c: ndimage.binary_dilation(
            codes == c, structure=np.ones((5, 5))
        )
        for c in _unique_color_codes(codes)
    }
    # Mismas máscaras SIN dilatar: son la autoridad para no coser sobre otro
    # color. La dilatación 5×5 de arriba sirve para tolerar el redondeo de los
    # extremos al borde, pero da por buena cualquier separación de hasta 4 px y
    # por eso no puede ser la que decide si el hilo se tiende sobre otro color
    # (`OFF_COLOR_MAX_MM`, `_segment_off_color`).
    own_masks = {c: (codes == c) for c in _unique_color_codes(codes)}
    # Jerarquía de contornos: `hole_masks` es el interior real de los huecos
    # (contadores de letra, agujeros) detectado por `_contour_hierarchy`. Se
    # aplica ADEMÁS de la máscara sólida, que al ir dilatada 5×5 puede dar por
    # bueno un puente sobre un hueco estrecho: el interior de un contorno hijo
    # es infranqueable con la aguja abajo, y si no hay forma de conectar dos
    # puntos sin cruzarlo se emite TRIM.
    #
    # `guide`: aquí es donde el mockup decide dos cosas, y solo si el usuario no
    # pidió dejarlo todo como estaba. Se construye UNA silueta de baja
    # resolución (`core.mockup`) a partir del plano de códigos que ya se borda y
    # se descarta al terminar; no se guarda en el patrón ni en ningún caché.
    #
    # 1. Si un hueco obliga a cortar. Los huecos que sobreviven a
    #    `HOLE_BARRIER_ERODE_PX` incluyen ruido de discretización de un píxel
    #    (medido: 105 de 160 huecos del logo de prueba miden <4 mm², con un
    #    mínimo de 0.09 mm²), y cada uno costaba un corte de hilo para mover la
    #    aguja ~1 mm. `_drop_tiny_holes` se los quita de la barrera — SOLO de la
    #    barrera: el relleno y el underlay ya se generaron con el hueco dentro,
    #    así que la geometría bordada no cambia.
    # 2. Si un salto entre dos zonas puede seguir con la aguja abajo. La silueta
    #    dice cuánto vacío real hay entre las dos puntadas; si es menos de un
    #    píxel de silueta (~2 mm a 1.56 mm/px), es redondeo de una esquina y el
    #    hilo no necesita tenderse sobre el fondo. Aquí está la mayor parte del
    #    ahorro: de los 1747 saltos de más de 6 mm del logo de texto, 232
    #    cruzan menos de 2 mm de vacío según la silueta.
    #
    # Ninguno de los dos mandos depende de un umbral global de "complejidad":
    # cada decisión se toma tramo a tramo y es autorestrictiva, así que en un
    # diseño limpio no hay ni un salto que cruce vacío y el resultado es
    # idéntico al de siempre. `GUIDE_FAITHFUL` no construye nada y devuelve el
    # comportamiento original byte a byte.
    mockup = None
    hole_min = 0.0
    bridge_gap = 0.0
    if guide != mockup_module.GUIDE_FAITHFUL:
        mockup = mockup_module.build_mockup(codes, mmp, background_mask)
        if mockup is not None:
            hole_min = mockup_module.hole_min_mm2(guide)
            bridge_gap = mockup_module.bridge_gap_mm(guide)
    if hole_min > 0.0:
        hole_masks = _drop_tiny_holes(hole_masks, mmp, hole_min)
    stitches = order_tatami_rows(
        stitches, connect_mm, solid_masks, mmp, hole_masks, mockup, bridge_gap,
        own_masks,
    )
    # El mock era solo guía: se va fuera de scope al terminar esta función y
    # no se guarda en el patrón (ver `core/mockup`).
    del mockup

    return StitchPattern(
        stitches=stitches,
        preview=render_preview(stitches, threads_used, width_mm, height_mm),
    )


def render_preview(
    stitches: list,
    threads_used: list,
    width_mm: float = MAX_SIZE_MM,
    height_mm: float = MAX_SIZE_MM,
) -> Image.Image:
    """Dibuja la vista previa del patrón sobre un fondo blanco del bastidor."""
    size_x = max(1, int(width_mm * PREVIEW_SCALE_PX_PER_MM))
    size_y = max(1, int(height_mm * PREVIEW_SCALE_PX_PER_MM))
    canvas = np.full((size_y, size_x, 3), 255, dtype=np.uint8)

    code_to_rgb = {t.code: np.asarray(t.rgb, dtype=np.uint8) for t in threads_used}

    # Los saltos (marcadores) no son puntadas reales y no se dibujan
    real = [s for s in stitches if s[2] not in MOVE_SENTINELS]
    if real:
        xs = np.asarray([s[0] for s in real], dtype=np.int32)
        ys = np.asarray([s[1] for s in real], dtype=np.int32)
        cols = np.stack([code_to_rgb[s[2]] for s in real])
        px = (xs * PREVIEW_SCALE_PX_PER_MM).astype(np.int32)
        py = (ys * PREVIEW_SCALE_PX_PER_MM).astype(np.int32)
        for dy in range(PREVIEW_BLOCK):
            for dx in range(PREVIEW_BLOCK):
                xx = px + dx
                yy = py + dy
                ok = (xx >= 0) & (xx < size_x) & (yy >= 0) & (yy < size_y)
                canvas[yy[ok], xx[ok]] = cols[ok]

    preview = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(preview)
    # Trazado real de la hebra: con separación de puntadas de ~3 mm los
    # puntos aislados dejan huecos en la vista; se dibuja la línea que la
    # aguja cose entre puntadas consecutivas del MISMO color mientras está
    # bajada (dos puntadas reales seguidas sin salto entre medias). Con el
    # relleno denso previo estas líneas coinciden con los bloques y el aspecto
    # es idéntico.
    s = PREVIEW_SCALE_PX_PER_MM
    prev = None
    for run in real:
        x, y, code = run
        if prev is not None:
            pv_x, pv_y, pv_code = prev
            if pv_code == code:
                draw.line(
                    [pv_x * s, pv_y * s, x * s, y * s],
                    fill=tuple(int(v) for v in code_to_rgb[code]),
                    width=1,
                )
        prev = run
    # Marcar cortes de hilo (TRIM_SENTINEL). Solo indicador visual, no cambia generación.
    try:
        trim_positions = []
        for st in stitches:
            if st[2] == TRIM_SENTINEL:
                trim_positions.append((st[0], st[1]))
        if trim_positions:
            marker_r = max(2, int(1.5 * s)) if s > 0 else 3
            for tx, ty in trim_positions:
                px_t = tx * s
                py_t = ty * s
                draw.line(
                    [(px_t - marker_r, py_t), (px_t + marker_r, py_t)],
                    fill=(220, 50, 50),
                    width=2,
                )
                draw.line(
                    [(px_t, py_t - marker_r), (px_t, py_t + marker_r)],
                    fill=(220, 50, 50),
                    width=2,
                )
    except Exception:
        pass
    draw.rectangle([0, 0, size_x - 1, size_y - 1], outline=(150, 150, 150))
    return preview