"""Mockup interno: silueta simplificada de baja resolución (guía, no producto).

NO es una vista previa ni un resultado de bordado: es un mapa binario de
"diseño / vacío" del mismo diseño, reducido a unos pocos píxeles por lado y
suavizado hasta perder el detalle pequeño. Sirve para una sola pregunta:
cuando el enrutado (`core.stitch_generator`) duda entre seguir cosiendo o
levantar la aguja, la silueta le dice **cuánto vacío real hay que atravesar**.

Por qué ayuda: el motor real decide con geometría exacta (jerarquía de
contornos, máscaras dilatadas), y esas decisiones tienen un error de
discretización del tamaño de un píxel. Un hueco de 1-2 mm en el plano de
códigos no es un contador real, es el redondeo de un borde; obligar a cortar
ahí consume un corte de hilo para mover la aguja un milímetro. La silueta,
que a propósito borró ese detalle, separa los dos casos: hueco de 1 mm
(ruido, se puede seguir) y contador de 6 mm (real, hay que cortar).

Por eso contesta a DOS preguntas del enrutado, ambas con la misma silueta:

1. `hole_min_mm2(guide)`: ¿un hueco es un contador real o ruido de borde?
   Decide si ese hueco obliga a levantar la aguja y cortar.
2. `bridge_gap_mm(guide)`: ¿cuánto vacío real atraviesa el tramo entre dos
   puntadas? Si el vacío es menor que un píxel de silueta, es redondeo y el
   hilo puede seguir abajo; si ocupa varios píxeles, son zonas separadas de
   verdad y hay que levantar la aguja.

Ambas son consultas locales sobre un tramo recto, así que no hace falta
guardar el mock entre designs: se construye, se usa y se tira.

Características (requisitos de diseño):

- **Liviano**: `MOCK_MAX_PX` px en el lado largo y `bool` de un solo canal.
  64x64 bool = 4 KB. Se genera una vez por diseño y no se guarda en
  `StitchPattern` ni en ningún caché: el llamador lo descarta tras usarlo.
- **Sin colores ni texturas**: solo "hay diseño aquí o no". Los códigos de
  hilo, la densidad y la textura se pierden a propósito.
- **Conserva la forma general**: al reducir con `INTER_AREA` y quedarse con
  el Majority de cada celda, una región debe cubrir ~la mitad de su celda
  para sobrevivir, así que la silueta mantiene contornos principales y
  agujeros importantes y descarta motas.
- **Proporción**: se reescala de forma uniforme, así que la silueta mantiene
  las proporciones reales y `mm_per_px` es constante.

Este módulo no importa el motor de bordado (ni el revés): solo depende de
NumPy y OpenCV, que el motor ya necesita.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

# Lado largo máximo de la silueta, en píxeles. Medido: por debajo de 128 px la
# silueta no resuelve un hueco de 1 mm y confunde el redondeo de un borde con el
# contador de una letra; 160 px separa los dos casos en todos los logos de
# prueba. Coste: 8 KB para una imagen apaisada, 25 KB para un cuadrado, contra
# los ~3 MB que ocupa el plano de códigos que ya se borda.
MOCK_MAX_PX = 160

# Longitud de la arista del openings+closing, en píxeles DE LA SILUETA.
# 3 px a 160 px de lado ~= 2% del lado: borra muescas y motas finas sin
# comerse un agujero real, y a 1.56 mm/px equivale a ~4.7 mm de radio, justo
# por debajo de un contador de letra de cuerpo normal.
MOCK_CLEAN_PX = 3

# Un diseño se considera "complejo" (el mock interviene) cuando la silueta
# tiene más de este número de pedazos separados: muchas letras, muchos
# contadores, bordes muy troceados. Un círculo o una barra tienen 1-2.
MOCK_COMPLEX_PIECES = 12


@dataclass(frozen=True)
class Mockup:
    """Silueta binaria del diseño y su escala, lista para consultas en mm.

    `mask` es `True` donde hay diseño (no fondo) a `MOCK_MAX_PX` de lado
    largo. `mm_per_px` traduce de coordenadas de bordado (mm) a píxeles de
    la silueta, de modo que se puede preguntar por un punto del patrón sin
    arrastrar ninguna transformación del llamador.
    """

    mask: np.ndarray
    mm_per_px: float

    @property
    def width_px(self) -> int:
        return int(self.mask.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.mask.shape[0])

    @property
    def pieces(self) -> int:
        """Número de pedazos separados de diseño (componentes conexas)."""
        labels, n = ndimage.label(self.mask)
        del labels
        return int(n)

    def _index(self, x_mm: float, y_mm: float) -> tuple[int, int] | None:
        """Píxel de la silueta que contiene (x_mm, y_mm), o None si cae fuera."""
        if self.mm_per_px <= 0:
            return None
        # Las coordenadas del bordado están en mm con Y hacia abajo (misma
        # convención que `render_preview` y que las máscaras del motor), así
        # que basta con dividir sin invertir ningún eje.
        col = int(x_mm / self.mm_per_px)
        row = int(y_mm / self.mm_per_px)
        if row < 0 or col < 0 or row >= self.mask.shape[0] or col >= self.mask.shape[1]:
            return None
        return row, col

    def samples(self, a: tuple, b: tuple, step_mm: float | None = None) -> np.ndarray:
        """Muestrea el segmento a→b en la silueta. Devuelve 1D `bool`.

        `False` = el punto cae en vacío (o fuera de la silueta). El paso es
        medio píxel de silueta por defecto: suficiente para no dejar pasar
        un hueco de un píxel sin detectarlo, y ~10 muestras por tramo
        típico. Devuelve un array vacío si el segmento no toca la silueta.
        """
        if step_mm is None:
            step_mm = max(self.mm_per_px * 0.5, 1e-6)
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0.0:
            idx = self._index(float(a[0]), float(a[1]))
            if idx is None:
                return np.zeros(0, dtype=bool)
            return np.array([bool(self.mask[idx])], dtype=bool)
        steps = max(2, int(length / step_mm) + 1)
        out = np.empty(steps, dtype=bool)
        for k in range(steps):
            ratio = k / (steps - 1)
            idx = self._index(float(a[0]) + dx * ratio, float(a[1]) + dy * ratio)
            out[k] = bool(self.mask[idx]) if idx is not None else False
        return out

    def gap_mm(self, a: tuple, b: tuple) -> float:
        """Vacío real más largo que el segmento a→b atraviesa, en mm.

        Es la pregunta que el motor no puede responder bien por su cuenta:
        `_segment_in_hole` solo sabe si el segmento ENTRA en un hueco, no
        si ese hueco es un contador de 8 mm o el redondeo de 1 mm de un
        borde. Devuelve la longitud de la run más larga de `False` en el
        muestreo, en mm; 0.0 si el segmento va íntegro sobre diseño.
        """
        s = self.samples(a, b)
        if s.size == 0 or bool(s.all()):
            return 0.0
        # Run más larga de vacío dentro del muestreo.
        longest = 0
        current = 0
        for value in s:
            if value:
                current = 0
            else:
                current += 1
                if current > longest:
                    longest = current
        # `samples` avanza ~medio píxel de silueta por punto.
        return longest * max(self.mm_per_px * 0.5, 1e-6)

    def cover_ratio(self, a: tuple, b: tuple) -> float:
        """Fracción del segmento a→b que cae sobre diseño (0.0 a 1.0)."""
        s = self.samples(a, b)
        if s.size == 0:
            return 0.0
        return float(s.mean())


def build_mockup(
    codes: np.ndarray,
    mmp: float,
    background: bool | np.ndarray | None = None,
) -> Mockup | None:
    """Construye la silueta de `codes` en baja resolución.

    `codes` es el plano de códigos de hilo (el mismo que se borda), con
    `None` en el fondo. No se leen los colores: solo interesa la silueta
    binaria "diseño / vacío", que es lo que separa un borde redondeado de
    un contador real.

    `mmp` son los mm por píxel del plano original (el `mmp` que ya calcula
    `generate_stitches`), necesario para devolver la escala real.

    `background` es opcional y solo se usa para acotar el recorte: si se
    pasa una máscara booleana (las zonas que el usuario marcó como
    eliminadas), se ignoran como diseño. `None` = usa `codes` tal cual.

    Devuelve `None` si la silueta no sirve (plano vacío, escala inválida o
    módulo de cv2 sin INTER_AREA), para que el llamador siga con el flujo
    anterior sin mock. Nunca lanza.
    """
    if codes is None or mmp is None or mmp <= 0:
        return None
    try:
        design = codes != None  # noqa: E711 - comparison intentional, None = fondo
        if background is not None:
            if getattr(background, "shape", None) == design.shape:
                design = design & ~background.astype(bool)
        if not design.any():
            return None

        h, w = design.shape[:2]
        scale = min(1.0, MOCK_MAX_PX / float(max(h, w)))
        if scale >= 1.0:
            small = design
        else:
            small_w = max(1, int(round(w * scale)))
            small_h = max(1, int(round(h * scale)))
            # INTER_AREA sobre uint8 promedia la cobertura de la celda; luego
            # el Majority decide si esa celda es diseño. Así una mota que no
            # llega a la mitad de su celda desaparece sola, sin morfología.
            area = cv2.resize(
                design.astype(np.uint8),
                (small_w, small_h),
                interpolation=cv2.INTER_AREA,
            )
            small = area >= 0.5

        # Limpieza simétrica: abre (quita motas y flecos de un píxel) y
        # cierra (une muescas de un píxel). El opening es lo que borra los
        # agujeros minúsculos que la silueta debe ignorar; el closing lo que
        # evita que una rotura de discretización se vuelva un "contador".
        if MOCK_CLEAN_PX >= 2 and min(small.shape[:2]) > MOCK_CLEAN_PX:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (MOCK_CLEAN_PX, MOCK_CLEAN_PX)
            )
            small = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel).astype(bool)

        if not small.any():
            return None

        return Mockup(
            mask=np.ascontiguousarray(small, dtype=bool),
            mm_per_px=float(mmp) / float(scale),
        )
    except Exception:  # noqa: BLE001 - el mock es una guía, nunca un requisito
        return None


# ---------------------------------------------------------------------------
# Decisiones simples: los tres modos que el usuario puede elegir
# ---------------------------------------------------------------------------
# Son los únicos valores que acepta `stitch_generator.generate_stitches(guide=)`.
# Los nombres son internos; la interfaz los traduce a lenguaje cotidiano (ver
# `ui/main_window.py`), nunca muestra estas cadenas.

GUIDE_FAITHFUL = "faithful"  # "Dejarlo como está"
GUIDE_SIMPLE = "simple"      # "Simplificar estas zonas"
GUIDE_AUTO = "auto"          # "Dejar que el programa decida"

GUIDE_OPTIONS = (GUIDE_FAITHFUL, GUIDE_SIMPLE, GUIDE_AUTO)

# Área mínima (mm²) que un hueco debe tener para seguir siendo infranqueable con
# la aguja abajo. Es el ÚNICO mando que el guide mueve: no toca la densidad, ni
# el número de puntadas, ni el relleno, ni las coordenadas, solo si un hueco
# obliga a cortar el hilo.
#
# Calibrado contra los holes reales de los logos de prueba:
#   - 0.0 mm² = "faithful": conserva los 160 huecos, incluidos los de un píxel.
#   - 1.0 mm² = "auto": elimina solo el ruido puro (un píxel erosionado mide
#     0.09-0.3 mm²), que es el 92% de los huecos < 2 mm². Invisible a simple vista.
#   - 6.0 mm² = "simple": elimina también los contadores pequeños de las
#     letras petites, que es lo que de verdad se nota en un bordado de 13x18 cm.
HOLE_MIN_MM2_BY_GUIDE = {
    GUIDE_FAITHFUL: 0.0,
    GUIDE_AUTO: 1.0,
    GUIDE_SIMPLE: 6.0,
}


def hole_min_mm2(guide: str | None) -> float:
    """Umbral de barrera de hueco, en mm², para el modo indicado.

    `None` o un modo desconocido cae en `GUIDE_AUTO`: es el modo que solo
    quita ruido (ruido que nadie quiere bordado) y por eso es el default
    seguro. `GUIDE_FAITHFUL` devuelve 0.0, que deja los mapas intactos.
    """
    if guide == GUIDE_FAITHFUL:
        return 0.0
    return HOLE_MIN_MM2_BY_GUIDE.get(guide, HOLE_MIN_MM2_BY_GUIDE[GUIDE_AUTO])


# Vacío máximo (mm) que un tramo puede atravesar y seguir cosiendo con la aguja
# abajo, según el modo. 0.0 = no se puentea nunca (comportamiento original).
#
# Calibrado contra la propia resolución de la silueta, no contra un número
# redondo: los logos de prueba dan 1.56 mm/px, así que 2.0 mm es ~1.3 px y
# 3.0 mm es ~2 px. Ese es exactamente el rango del error de discretización:
# por debajo, el "vacío" es el redondeo de una esquina viva; por encima, hay
# separación real entre dos zonas y el hilo no debe tenderse sobre el fondo.
#
# Efecto medido sobre los saltos de más de 6 mm (los únicos que el exportador
# convierte en corte), con la silueta como autoridad:
#   - logo de texto 857x295 (1.56 mm/px): 1747 saltos -> 232 con hueco <= 2 mm
#   - logo 735x575 (1.56 mm/px): 377 saltos -> 136 con hueco <= 2 mm
# Es decir, el 13% y el 36% de los cortes son redondeo de borde, no costuras
# entre zonas separadas. Los de 6-12 mm (47% y 42%) son separación real y este
# umbral NO los toca: seguir ahí sería coser sobre el fondo.
MOCK_BRIDGE_GAP_MM_BY_GUIDE = {
    GUIDE_FAITHFUL: 0.0,
    GUIDE_AUTO: 2.0,
    GUIDE_SIMPLE: 3.0,
}


def bridge_gap_mm(guide: str | None) -> float:
    """Vacío máximo (mm) que el hilo puede cruzar con la aguja abajo.

    `None` o modo desconocido cae en `GUIDE_AUTO`. `GUIDE_FAITHFUL` devuelve
    0.0, que desactiva el puente por completo y deja el enrutado exactamente
    como estaba.
    """
    if guide == GUIDE_FAITHFUL:
        return 0.0
    return MOCK_BRIDGE_GAP_MM_BY_GUIDE.get(guide, MOCK_BRIDGE_GAP_MM_BY_GUIDE[GUIDE_AUTO])


def is_complex(mockup: "Mockup | None") -> bool:
    """True si el diseño es de los complejos, donde el guide aporta.

    Un diseño simple (un círculo, una barra, un texto corto sin contadores)
    sale más rápido y más fiel por el flujo de siempre; ahí el mock no se
    construye siquiera. El criterio es la silueta partida en muchos pedazos:
    muchas letras, muchos contadores, bordes muy troceados.
    """
    if mockup is None:
        return False
    return mockup.pieces > MOCK_COMPLEX_PIECES


def suggest_guide(mockup: "Mockup | None", busy_zones: list | None = None) -> str:
    """Qué modo conviene por defecto, sin preguntarle al usuario.

    La regla es deliberadamente corta: `GUIDE_AUTO` es seguro por construcción,
    así que es la respuesta siempre que haya una silueta con la que decidir.
    El modo automático solo hace dos cosas, y las dos son independientes de
    cómo esté dibujado el diseño:

    - un hueco menor de 1 mm² deja de obligar a cortar (es ruido de un píxel);
    - un tramo que atraviesa menos de 2 mm de vacío en la silueta sigue con la
      aguja abajo (es redondeo de una esquina).

    Ninguna de las dos puede abrir un contador real ni coser sobre el fondo: para
    que eso ocurriera haría falta que el vacío midiera de verdad ~2 mm, que es
    justo lo que este modo no permite. Por eso no hace falta un criterio más
    astuto para decidir si conviene: aplicarlo siempre es correcto, y en un
    diseño limpio no cambia ni una puntada.

    - Sin mock (`None`) no hay con qué decidir → `GUIDE_FAITHFUL`, que es
      exactamente el comportamiento anterior al agregar esta guía.
    - Con mock, haya zonas con cortes acumulados o no → `GUIDE_AUTO`.

    `GUIDE_SIMPLE` no se propone solo: sube ambos umbrales (3 mm de vacío y
    6 mm² de hueco) y eso sí cambia el dibujo en el lettering pequeño, así que
    queda como elección explícita del usuario.
    """
    if mockup is None:
        return GUIDE_FAITHFUL
    return GUIDE_AUTO