"""Zonas con muchos cortes de hilo: dónde el programa puede alargar el bordado.

Este módulo NO borda ni decide el diseño: mira la lista de puntadas ya
generada, cuenta los cortes de hilo y agrupa los que caen en la misma zona del
bastidor. Sirve para dos cosas concretas:

1. Decirle a la interfaz **dónde** mirar (un rectángulo en milímetros, que la
   vista previa resalta), para que la pregunta que se le hace al usuario tenga
   algo que señalar.
2. Alimentar el mensaje de estado de la interfaz con `cuts_per_1000`, que
   traduce "muchas paradas de máquina" sin tecnicismos.

No decide el diseño: qué modo usar es cosa de `core.mockup.suggest_guide`, que
solo necesita saber si hay silueta con la que trabajar.

Reutiliza las constantes del motor y del exportador (`TRIM_SENTINEL`,
    `JUMP_SENTINEL`, `MOVE_SENTINELS`, `TRIM_MIN_JUMP_DISTANCE_MM`), de modo que
    un "corte" contado aquí es exactamente un corte en la máquina.
"""

from dataclasses import dataclass

from core.exporters.pes_exporter import TRIM_MIN_JUMP_DISTANCE_MM
from core.stitch_generator import MOVE_SENTINELS, TRIM_SENTINEL

# Lado de la celda del histograma, en mm. 12 mm sobre un bastidor de 13x18 cm
# cae en una cuadrícula de ~11x15 celdas: bastante fino para distinguir "las
# letras del centro" de "la esquina del emblema", bastante grueso para que dos
# cortes separados por un píxel caigan en la misma celda y se sumen.
CELL_MM = 12.0

# Mínimo de cortes en una zona para que merezca la pena mencionarla. Con 3 ya
# hay un patrón (tres paradas seguidas de la máquina en el mismo trozo); con 1 o
# 2 es ruido normal de cualquier diseño.
MIN_CUTS = 3

# Margen en mm alrededor del rectángulo de la zona al reportarla, para que el
# resaltado englobe el recorrido y no solo el punto exacto de parada.
PAD_MM = 3.0

# Distancia mínima (mm) entre el centro de dos zonas que se reportan. Dos
# cuadros a 20 mm uno del otro se leen como uno solo en la vista previa, así que
# señalarlos los dos no aporta nada; con 45 mm se garantiza que quien mira ve
# dos lugares distintos en el bastidor.
MIN_SEPARATION_MM = 45.0


@dataclass(frozen=True)
class BusyZone:
    """Una zona del bastidor con cortes de hilo acumulados.

    `x0, y0, x1, y1` van en milímetros con Y hacia abajo (la convención de las
    coordenadas de bordado, la misma que usa `render_preview` y las máscaras del
    motor). `cuts` es cuántos cortes caen dentro. `travel_mm` es la distancia
    media que recorre la aguja entre los puntos de esos cortes, y `cover` la
    fracción de diseño que la silueta del mock ve a lo largo de ellos (`0.0`
    si no se pasó mock): alto = hay estructura real debajo, bajo = el corte
    está sobre un hueco de redondeo.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    cuts: int
    travel_mm: float
    cover: float

    @property
    def width_mm(self) -> float:
        return self.x1 - self.x0

    @property
    def height_mm(self) -> float:
        return self.y1 - self.y0

    @property
    def side_mm(self) -> float:
        """Lado mayor del rectángulo: el tamaño que ve quien cose."""
        return max(self.width_mm, self.height_mm)

    def padded(self, pad_mm: float = PAD_MM) -> "BusyZone":
        """El mismo rectángulo con `pad_mm` de margen por los cuatro lados."""
        return BusyZone(
            x0=self.x0 - pad_mm,
            y0=self.y0 - pad_mm,
            x1=self.x1 + pad_mm,
            y1=self.y1 + pad_mm,
            cuts=self.cuts,
            travel_mm=self.travel_mm,
            cover=self.cover,
        )

    def describe(self) -> str:
        """Descripción en lenguaje cotidiano, sin jerga de bordado.

        Describe la zona por su tamaño y por cuántas veces se para la máquina,
        nunca por la causa técnica ni por las palabras del dominio.
        """
        side = self.side_mm
        where = "pequeña" if side < 25 else "mediana" if side < 60 else "grande"
        return f"una zona {where}, de unos {side:.0f} mm, con {self.cuts} paradas seguidas"


def count_cuts(stitches: list, threshold_mm: float = TRIM_MIN_JUMP_DISTANCE_MM) -> list:
    """Un corte por entrada de la lista, como `(origen, destino, distancia, forzado)`.

    Se recorre la lista como lo haría la máquina. Un corte ocurre en el comando
    cuyo centinela es `TRIM_SENTINEL` (corta siempre), o en un `JUMP_SENTINEL`
    cuyo desplazamiento supera `threshold_mm` (el exportador lo convierte en
    corte). Se devuelven ambos extremos porque el hilo que queda tendido es el
    tramo `origen → destino`, que es justo lo que hay que consultar en el mock.
    """
    out = []
    previous = (0.0, 0.0, None)
    for point in stitches:
        code = point[2]
        if code in MOVE_SENTINELS:
            origin = (previous[0], previous[1])
            destination = (point[0], point[1])
            dx = destination[0] - origin[0]
            dy = destination[1] - origin[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if code == TRIM_SENTINEL or distance > threshold_mm:
                out.append((origin, destination, distance, code == TRIM_SENTINEL))
        previous = point
    return out


def find_busy_zones(
    stitches: list,
    width_mm: float,
    height_mm: float,
    mockup=None,
    max_zones: int = 2,
    min_cuts: int = MIN_CUTS,
    cell_mm: float = CELL_MM,
    min_separation_mm: float = MIN_SEPARATION_MM,
) -> list:
    """Las zonas con más cortes de las que merece la pena enseñar, peor primero.

    Los cortes se echan en una cuadrícula de `cell_mm` y cada celda se evalúa
    por separado. **No se fusionan celdas vecinas**: en un texto largo los
    cortes forman una cadena continua de celda en celda, y cualquier
    flood-fill acaba devolviendo el bastidor entero como una sola zona, que no
    señala nada. Lo que se busca es el punto más denso, no la región conectada.

    Para eso, las celdas con al menos `min_cuts` se ordenan de más a menos
    cortes y se van eligiendo greedy, saltándose las que estén a menos de
    `min_separation_mm` de una ya elegida (dos cuadros juntos se leen como uno y
    no aportan información). Así las zonas que salen son locales y distintas
    entre sí, que es lo que sirve para señalar en la vista previa.

    De cada celda elegida se toma el rectángulo envolvente de los puntos de
    llegada, el número de cortes, la distancia media recorrida y —si se pasó
    `mockup`— la cobertura media que la silueta ve a lo largo de esos tramos.

    Devuelve como mucho `max_zones` (2 por defecto: menos preguntas y menos
    ruido visual en la vista previa), ordenadas por número de cortes
    descendente. Lista vacía = no hay nada que preguntar.
    """
    cuts = count_cuts(stitches)
    if not cuts or max_zones <= 0 or min_cuts <= 0:
        return []

    columns = max(1, int(width_mm / cell_mm) + 1)
    rows = max(1, int(height_mm / cell_mm) + 1)

    cells = {}
    for origin, destination, distance, forced in cuts:
        col = min(columns - 1, max(0, int(destination[0] / cell_mm)))
        row = min(rows - 1, max(0, int(destination[1] / cell_mm)))
        cells.setdefault((row, col), []).append((origin, destination, distance, forced))

    dense = [
        (key, group)
        for key, group in cells.items()
        if len(group) >= min_cuts
    ]
    dense.sort(key=lambda item: (-len(item[1]), -max(c[2] for c in item[1])))

    chosen = []
    zones = []
    for (row, col), group in dense:
        if len(zones) >= max_zones:
            break
        centre = ((col + 0.5) * cell_mm, (row + 0.5) * cell_mm)
        if any(
            ((centre[0] - other[0]) ** 2 + (centre[1] - other[1]) ** 2) ** 0.5
            < min_separation_mm
            for other in chosen
        ):
            continue
        xs = [destination[0] for _, destination, _, _ in group]
        ys = [destination[1] for _, destination, _, _ in group]
        distances = [distance for _, _, distance, _ in group]
        if mockup is not None:
            cover = sum(
                mockup.cover_ratio(origin, destination) for origin, destination, _, _ in group
            ) / len(group)
        else:
            cover = 0.0
        chosen.append(centre)
        zones.append(
            BusyZone(
                x0=min(xs),
                y0=min(ys),
                x1=max(xs),
                y1=max(ys),
                cuts=len(group),
                travel_mm=sum(distances) / len(distances),
                cover=cover,
            )
        )

    return zones


def cuts_per_1000(stitches: list, threshold_mm: float = TRIM_MIN_JUMP_DISTANCE_MM) -> float:
    """Cortes por cada 1000 puntadas reales, para el mensaje de la interfaz.

    Es el número que explica "el diseño tiene muchas paradas" sin entrar en
    tecnicismos: una densidad, no un total. 0.0 si no hay puntadas reales.
    """
    real = sum(1 for s in stitches if s[2] not in MOVE_SENTINELS)
    if real <= 0:
        return 0.0
    return 1000.0 * len(count_cuts(stitches, threshold_mm)) / real