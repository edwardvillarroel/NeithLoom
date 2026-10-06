import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import (
    Qt,
    QPoint,
    QPointF,
    QRect,
    QSettings,
    QSize,
    QRectF,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core import (
    color_processor,
    hoops,
    image_loader,
    mockup,
    stitch_generator,
    threads,
    trim_zones,
)
from core.exporters.dst_exporter import export_dst
from core.exporters.pes_exporter import export_pes

IMAGE_FILTER = "Imágenes (*.jpg *.jpeg *.png)"

# Rejilla del paso 1. `image_loader.list_recent_images` acota el total a
# GALLERY_LIMIT, así que por carpeta se limita a un número parecido para que
# ninguna se trague el panel y no se generen miniaturas fuera de la vista.
GALLERY_PER_DIR = 40
THUMB_GRID_ICON = 132
THUMB_GRID_CELL_W = 152
THUMB_GRID_CELL_H = 168
THUMB_GRID_SPACING = 8
MAX_DIMENSION = image_loader.MAX_DIMENSION
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICON_SUFFIXES = (".png", ".ico", ".svg")

# Clave de QSettings donde se recuerda la elección del usuario. Opcional a
# propósito: si nunca se marca "recordar", no se escribe nada y cada diseño se
# vuelve a preguntar.
GUIDE_SETTING_KEY = "gui/guia_zonas"

# Por encima de este número de paradas por cada 1000 no se pregunta nada: son
# pocas y el usuario no va a notar la diferencia. El corte está muy por encima
# de los 81/1000 del logo de prueba más ruidoso, así que ese diseño sí pregunta.
GUIDE_ASK_MIN_PER_1000 = 45.0

# Carpeta de destino del último guardado, para preseleccionarla al volver.
LAST_SAVE_DIR_KEY = "gui/last_save_dir"

# Donación opcional. Se apoya en el mismo QSettings() desnudo que el resto de
# la app: se recuerda si el usuario pidió no volver a ver el mensaje y, con un
# contador simple de flujos completados, se muestra 1 de cada DONATION_EVERY_N
# finalizaciones exitosas (nunca la primera).
DONATION_SUPPRESS_KEY = "donacion/no_volver_a_mostrar"
DONATION_COUNT_KEY = "donacion/flujos_completados"
DONATION_EVERY_N = 5
PAYMENT_LINK = "https://link.mercadopago.com.ar/neithloom"
QR_IMAGE_PATH = PROJECT_ROOT / "images" / "donacion" / "mp_qr.png"


def _donation_settings() -> QSettings:
    # Con org/aplicación vacíos el QSettings() desnudo de PySide6 es un objeto
    # nulo que no persiste nada; la forma explícita sí escribe en el registro.
    return QSettings("NeithLoom", "NeithLoom")

_PICK_BG_TEXT = "Seleccionar fondo"
_RECOLOR_TEXT = "Cambiar color de zona"
_REMOVE_TEXT = "Eliminar zona"
_CANCEL_TEXT = "Cancelar selección"
_EDITING_HEADER = "Edición"

# ---------------------------------------------------------------------------
# Tema visual
# ---------------------------------------------------------------------------
# Un solo color de acento, cálido y de hilo (naranja terracota). Solo se usa
# donde significa algo: la acción principal de cada paso, el borde de lo que
# está seleccionado y el visto de los pasos completados. Todo lo demás se
# queda en grises para que el acento resalte entre tanto color.
ACCENT = "#c2410c"  # naranja terracota
ACCENT_HOVER = "#9a3412"
ACCENT_PRESSED = "#7c2d12"
ACCENT_TINT = "#fdf0e6"  # fondo suave del elemento seleccionado
ACCENT_ON = "#ffffff"  # texto/icono sobre el acento

NEUTRAL_BORDER = "#cccccc"
NEUTRAL_TEXT = "#666666"


def _primary_button_style() -> str:
    """Estilo del botón de acción principal de un paso."""
    return (
        f"QPushButton {{ background: {ACCENT}; color: {ACCENT_ON};"
        " border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600; }"
        f"QPushButton:hover:enabled {{ background: {ACCENT_HOVER}; }}"
        f"QPushButton:pressed:enabled {{ background: {ACCENT_PRESSED}; }}"
        "QPushButton:disabled { background: #d9d9d9; color: #9a9a9a; }"
    )


def _selected_card_style() -> str:
    """Borde del elemento elegido (tarjeta de bastidor o de destino)."""
    return (
        f"QFrame {{ border: 2px solid {ACCENT}; border-radius: 8px;"
        f" background: {ACCENT_TINT}; }}"
    )


def _unselected_card_style() -> str:
    return "QFrame { border: 2px solid #c8ccd4; border-radius: 8px; background: #ffffff; }"


# ---------------------------------------------------------------------------
# Iconos
# ---------------------------------------------------------------------------
# Se dibujan con QPainter sobre un pixmap en vez de añadir una librería de
# iconos: son cuatro formas muy simples, el proyecto no trae ninguna y meter
# una dependencia (con su fuente y su registro de iconos) solo para un
# carrete, una aguja y un disquete no compensa.


def _make_icon(kind: str, color: str, size: int = 20) -> QIcon:
    """Dibuja un icono de línea simple y lo devuelve como `QIcon`.

    Cada icono se pinta sobre un fondo transparente y usa un trazo de 1.8 px
    para que se lea bien tanto sobre el acento (botones principales) como
    sobre gris claro (botones normales).
    """
    scale = 4  # se dibuja en una rejilla de 4x4 y se escala al tamaño pedido
    grid = 4
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color), 1.8 * scale)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    def pt(x: float, y: float) -> QPointF:
        return QPointF(x * scale, y * scale)

    if kind == "image":
        # Marco de foto con un sol: una imagen.
        painter.drawRect(QRectF(pt(0.6, 1.1), pt(3.4, 3.0)))
        painter.drawLine(pt(1.0, 2.7), pt(1.8, 1.9))
        painter.drawLine(pt(1.8, 1.9), pt(2.5, 2.6))
        painter.drawLine(pt(2.5, 2.6), pt(3.0, 2.1))
        painter.drawEllipse(QRectF(pt(0.95, 1.35), pt(1.55, 1.95)))
    elif kind == "spool":
        # Carrete de hilo: dos alas y el hilo enrollado en medio.
        painter.drawLine(pt(0.5, 1.2), pt(3.5, 1.2))
        painter.drawLine(pt(0.5, 3.0), pt(3.5, 3.0))
        painter.drawLine(pt(1.2, 1.2), pt(1.2, 3.0))
        painter.drawLine(pt(2.8, 1.2), pt(2.8, 3.0))
        painter.drawLine(pt(1.2, 1.9), pt(2.8, 1.9))
        painter.drawLine(pt(1.2, 2.3), pt(2.8, 2.3))
    elif kind == "needle":
        # Aguja con hilo en la punta.
        painter.drawLine(pt(2.9, 0.4), pt(1.3, 2.9))
        painter.drawLine(pt(1.3, 2.9), pt(0.7, 3.5))
        painter.drawEllipse(QRectF(pt(0.35, 3.25), pt(1.0, 3.9)))
        painter.drawLine(pt(2.2, 0.7), pt(2.9, 1.4))
    elif kind == "save":
        # Disquete con su ventana y su etiqueta.
        painter.drawRoundedRect(QRectF(pt(0.6, 0.5), pt(3.4, 3.5)), 0.5, 0.5)
        painter.drawRect(QRectF(pt(1.1, 0.7), pt(2.9, 1.7)))
        painter.drawRect(QRectF(pt(1.3, 2.2), pt(2.7, 3.3)))
    elif kind == "check":
        # Marca de completada.
        painter.drawLine(pt(0.7, 2.1), pt(1.7, 3.1))
        painter.drawLine(pt(1.7, 3.1), pt(3.3, 0.9))
    else:
        painter.drawEllipse(QRectF(pt(1.2, 1.2), pt(2.8, 2.8)))

    painter.end()
    icon = QIcon(pixmap)
    icon.addPixmap(pixmap)
    return icon


def find_app_icon() -> QIcon | None:
    """Busca un icono .png/.ico/.svg dentro de la carpeta images/ del proyecto."""
    images_dir = PROJECT_ROOT / "images"
    if not images_dir.is_dir():
        return None
    for child in sorted(images_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in ICON_SUFFIXES:
            return QIcon(str(child))
    return None


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(image))


def short_name(name: str, metrics: QFontMetrics, max_px: int) -> str:
    """Recorta el nombre de un archivo a `max_px` píxeles de ancho.

    Recortar por número de caracteres no vale: el ancho de cada letra depende
    de la fuente, del idioma y de los DPI, y con 18 caracteres un nombre largos
    medía 216 px en una celda de 152 px (se salía de la celda). El ancho en
    píxeles es lo único que se puede comparar con el ancho de la celda.
    """
    return metrics.elidedText(name, Qt.TextElideMode.ElideRight, max_px)


class ThreadPickerDialog(QDialog):
    """Lista de hilos Brother con muestras de color para elegir un hilo."""

    def __init__(self, parent=None, title: str = "Elegir hilo Brother"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(300, 420)
        self._selected_code: str | None = None

        layout = QVBoxLayout(self)

        self._list = QListWidget(self)
        for code, name, rgb in threads.THREADS:
            item = QListWidgetItem(f"{code}  {name}")
            item.setData(Qt.ItemDataRole.UserRole, code)
            swatch = QPixmap(40, 20)
            swatch.fill(QColor(*rgb))
            item.setIcon(QIcon(swatch))
            self._list.addItem(item)
        self._list.itemDoubleClicked.connect(self._accept_selection)
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Aplicar")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_selection(self, item=None) -> None:
        current = self._list.currentItem()
        if current is None:
            return
        self._selected_code = current.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def selected_code(self) -> str | None:
        return self._selected_code


_HELP_TEXT = """1. Abre una imagen
   Pulsa "Abrir imagen..." o elige una de la galería de la izquierda.

2. Elige el fondo (opcional)
   Si la imagen tiene un color de fondo que no quieres bordar, pulsa
   "Seleccionar fondo" y haz clic sobre ese color.

3. Elige los colores de hilo
   Pulsa "Ver los colores de hilo". Cada color encontrado aparece
   como un círculo con su nombre de Brother. "Máximo de colores
   del bordado" limita cuántos hilos como máximo se pueden usar.

4. Genera el bordado
   Elige una densidad (Baja / Media / Alta), el tamaño de bastidor que
   vayas a usar y pulsa "Generar bordado". La imagen se ajusta sola
   para que quepa en el bastidor.

5. Revisa la vista previa
   Verás una simulación del bordado dentro del bastidor elegido.

6. Edita si lo necesitas (opcional)
   - "Cambiar color de zona": haz clic en una zona y elige otro hilo.
   - "Cambiar densidad": aplica al bordado la densidad elegida.
   - "Eliminar zona": haz clic en una zona para no bordarla.

7. Guarda el archivo
   Elige la carpeta y el nombre, y pulsa "Guardar bordado".
   Ese archivo se lleva a la máquina de bordar."""


class HelpDialog(QDialog):
    """Guía de uso breve para quien no conoce el bordado digital."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guía de uso")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        title = QLabel("Cómo usar NeithLoom")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        text = QLabel(_HELP_TEXT)
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Cerrar")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class EmptyState(QWidget):
    """Estado vacío: un icono grande con un mensaje corto debajo.

    Sustituye al rectángulo gris vacío. El mensaje dice qué hacer a
    continuación en lugar de describir lo que falta, para que el usuario sepa
    dónde pulsar sin abrir la ayuda.
    """

    def __init__(
        self,
        icon_kind: str,
        title: str,
        hint: str = "",
        icon_size: int = 64,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        self._icon_label = QLabel()
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setPixmap(
            _make_icon(icon_kind, ACCENT, icon_size).pixmap(icon_size, icon_size)
        )
        layout.addWidget(self._icon_label)

        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 14px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(self._title_label)

        self._hint_label = QLabel(hint)
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet(
            f"color: {NEUTRAL_TEXT}; background: transparent;"
        )
        if hint:
            layout.addWidget(self._hint_label)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class PickImageLabel(QLabel):
    """Vista previa con forma de aro de bastidor.

    Dibuja el diseño dentro de una elipse (el aro) en vez de un rectángulo,
    que es como el usuario ve el trabajo en la máquina. El círculo se deforma
    con la proporción real del bastidor elegido, de forma que un 13x18 se ve
    más alargado que un 20x20 y el usuario compara tamaños de un vistazo.

    En modo cuentagotas reporta el píxel (x, y) de la imagen original. El
    clic se proyecta sobre la elipse: fuera del aro no hay diseño, así que
    no cuenta como píxel válido.
    """

    pixel_clicked = Signal(int, int)

    RING_COLOR = QColor(150, 150, 155)
    RING_WIDTH = 5

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._src_w = 0
        self._src_h = 0
        self._view_w = 0
        self._view_h = 0
        self._pick_mode = False
        self._source_pixmap: QPixmap | None = None
        self._empty_state: QWidget | None = None
        # Proporción del bastidor (ancho / alto). Por defecto 1:1 para que el
        # aro se vea redondo hasta que se elige un bastidor rectangular.
        self._hoop_aspect = 1.0

    def set_source_size(self, w: int, h: int) -> None:
        self._src_w = w
        self._src_h = h

    def set_view_size(self, w: int, h: int) -> None:
        self._view_w = w
        self._view_h = h

    def set_hoop_aspect(self, width_mm: float, height_mm: float) -> None:
        """Adapta la elipse a la proporción real del bastidor."""
        if width_mm > 0 and height_mm > 0:
            self._hoop_aspect = float(width_mm) / float(height_mm)
        self.update()

    def set_pick_mode(self, on: bool) -> None:
        self._pick_mode = on
        self.setCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor
        )

    def _hoop_rect(self) -> QRectF:
        """Caja elíptica centrada que respeta la proporción del bastidor.

        Se reserva el grosor del aro para que el metal no tape el diseño.
        """
        pad = self.RING_WIDTH + 4
        available_w = max(1.0, self.width() - 2 * pad)
        available_h = max(1.0, self.height() - 2 * pad)

        if available_w / available_h > self._hoop_aspect:
            # El ancho sobra: manda la altura.
            height = available_h
            width = height * self._hoop_aspect
        else:
            width = available_w
            height = width / self._hoop_aspect

        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def paintEvent(self, event) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            super().paintEvent(event)
            return

        hoop = self._hoop_rect()
        inner = hoop.adjusted(
            self.RING_WIDTH, self.RING_WIDTH, -self.RING_WIDTH, -self.RING_WIDTH
        )
        if inner.width() <= 0 or inner.height() <= 0:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # El diseño se recorta con la elipse: lo que cae fuera del aro no se
        # borda, así que tampoco se muestra.
        path = QPainterPath()
        path.addEllipse(inner)
        painter.setClipPath(path)
        painter.drawPixmap(inner, self._source_pixmap, self._source_pixmap.rect())

        painter.setClipping(False)
        painter.setPen(QPen(self.RING_COLOR, self.RING_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(hoop)
        painter.end()

        self._view_w = int(inner.width())
        self._view_h = int(inner.height())

    def set_source_pixmap(self, pixmap: QPixmap | None) -> None:
        self._source_pixmap = pixmap
        self.update()

    def set_empty_state(self, widget: QWidget) -> None:
        """Encuaja un estado vacío sobre el label.

        Un `QLabel` no reparte geometría a sus hijos (no tiene layout), así que
        el estado vacío se ajusta a mano y se recoloca en cada `resizeEvent`.
        """
        self._empty_state = widget
        widget.setGeometry(self.rect())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._empty_state is not None:
            self._empty_state.setGeometry(self.rect())

    def mousePressEvent(self, event) -> None:
        if not self._pick_mode or self._source_pixmap is None:
            super().mousePressEvent(event)
            return
        if self._src_w <= 0 or self._src_h <= 0:
            return

        hoop = self._hoop_rect()
        inner = hoop.adjusted(
            self.RING_WIDTH, self.RING_WIDTH, -self.RING_WIDTH, -self.RING_WIDTH
        )
        if inner.width() <= 0 or inner.height() <= 0:
            return

        # Fuera del aro no hay diseño que tomar: se ignora el clic.
        if not inner.contains(event.position()):
            return

        local = event.position() - inner.topLeft()
        x = int(local.x() * self._src_w / inner.width())
        y = int(local.y() * self._src_h / inner.height())
        if 0 <= x < self._src_w and 0 <= y < self._src_h:
            self.pixel_clicked.emit(x, y)


class ThreadSwatch(QWidget):
    """Círculo con el color del hilo y su nombre del catálogo Brother."""

    def __init__(self, rgb: tuple, code: str, name: str, usage: float, parent=None):
        super().__init__(parent)
        self._rgb = tuple(int(v) for v in rgb[:3])
        self._code = code
        self._name = name
        self._usage = max(0.0, float(usage))
        self.setFixedHeight(22)
        self.setMinimumWidth(150)
        self.setToolTip(f"{code} {name} - {self._usage * 100:.1f}% del diseño")

    def sizeHint(self):  # noqa: N802 (Qt API)
        text = self.fontMetrics().horizontalAdvance(f"{self._name}")
        return QSize(46 + text + 16, 22)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        radius = 9
        center = QPointF(radius + 2, self.height() / 2)
        painter.setPen(QPen(QColor(90, 92, 98), 1))
        painter.setBrush(QColor(*self._rgb))
        painter.drawEllipse(center, radius, radius)

        # El blanco puro sobre fondo claro necesita borde; el resto no.
        if min(self._rgb) > 235:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(120, 122, 128), 1))
            painter.drawEllipse(center, radius, radius)

        painter.setPen(QPen(QColor(40, 42, 48)))
        painter.drawText(
            QRectF(2 * radius + 4, 0, self.width() - 2 * radius - 4, self.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._name,
        )
        painter.end()


class HoopCard(QFrame):
    """Tarjeta clicable de un bastidor con su aro dibujado a escala.

    Cada tarjeta muestra el óvalo del bastidor con la misma proporción real
    (ancho/alto en mm) y con el tamaño relativo frente al bastidor más grande
    del catálogo, para que 10x10 se vea claramente más pequeño que 25x20.
    """

    clicked = Signal(object)

    OVAL_MAX_W = 74
    OVAL_MAX_H = 48

    def __init__(self, hoop: dict, selected: bool = False, parent=None):
        super().__init__(parent)
        self._hoop = hoop
        self._selected = selected
        self._ratio = 1.0
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(118)
        self.setMinimumWidth(116)
        self.setToolTip(self._tooltip_text())
        self._apply_style()

    def _tooltip_text(self) -> str:
        if self._hoop.get("custom"):
            return (
                f"{self._hoop['name']} - elige ancho y alto entre "
                f"{hoops.MIN_CUSTOM_CM:g} y {hoops.MAX_CUSTOM_CM:g} cm"
            )
        return (
            f"{self._hoop['name']} - "
            f"{self._hoop['width_mm']:.0f}x{self._hoop['height_mm']:.0f} mm"
        )

    def refresh(self) -> None:
        """Repinta la tarjeta tras cambiar las medidas de su bastidor."""
        self.setToolTip(self._tooltip_text())
        self.update()

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(_selected_card_style())
        else:
            self.setStyleSheet(_unselected_card_style())

    def set_ratio(self, ratio: float) -> None:
        """Tamaño del óvalo respecto al bastidor más grande (0..1)."""
        self._ratio = max(0.05, min(1.0, ratio))
        self.update()

    def set_selected(self, selected: bool) -> None:
        if self._selected != selected:
            self._selected = selected
            self._apply_style()

    @property
    def hoop(self) -> dict:
        return self._hoop

    def _oval_rect(self) -> QRectF:
        aspect = float(self._hoop["width_mm"]) / float(self._hoop["height_mm"])
        max_w = self.OVAL_MAX_W * self._ratio
        max_h = self.OVAL_MAX_H * self._ratio
        if max_w / max_h > aspect:
            height = max_h
            width = height * aspect
        else:
            width = max_w
            height = width / aspect
        return QRectF(
            (self.width() - width) / 2,
            10 + (self.OVAL_MAX_H * self._ratio - height) / 2,
            width,
            height,
        )

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self._oval_rect()
        if rect.width() > 0 and rect.height() > 0:
            painter.setPen(QPen(QColor(120, 124, 132), 3))
            painter.setBrush(QColor(246, 247, 250))
            painter.drawEllipse(rect)

        # Nombre del bastidor bajo el aro: "13 x 18 cm" es lo que el usuario
        # reconoce en la máquina, sin el nombre largo entre comillas. El
        # catálogo guarda milímetros, así que aquí se convierten a cm. En el
        # personalizado no hay cifras fijas: las pone el usuario debajo.
        if self._hoop.get("custom"):
            size = "Personalizado"
        else:
            size = (
                f"{self._hoop['width_mm'] / 10:.0f} x "
                f"{self._hoop['height_mm'] / 10:.0f} cm"
            )
        painter.setPen(QPen(QColor(30, 32, 38)))
        painter.drawText(
            QRectF(0, rect.bottom() + 4, self.width(), 18),
            int(Qt.AlignmentFlag.AlignCenter),
            size,
        )
        painter.end()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._hoop)


class FlowLayout(QLayout):
    """Layout que salta de línea cuando se llena el ancho disponible.

    Qt no trae un flow layout, así que se implementa el mínimo necesario:
    coloca los widgets de izquierda a derecha y quebra la línea al llegar al
    borde. Se usa para los círculos de hilo y las tarjetas de bastidor.
    """

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item) -> None:  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 (Qt API)
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802 (Qt API)
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802 (Qt API)
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt API)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt API)
        return self._arrange(QRect(0, 0, width, 0), dry_run=True)

    def setGeometry(self, rect) -> None:  # noqa: N802 (Qt API)
        super().setGeometry(rect)
        self._arrange(rect, dry_run=False)

    def sizeHint(self):  # noqa: N802 (Qt API)
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802 (Qt API)
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.sizeHint())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _arrange(self, rect, dry_run: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = area.x()
        y = area.y()
        line_height = 0
        for item in self._items:
            item_size = item.sizeHint()
            next_x = x + item_size.width() + self._spacing
            if next_x - self._spacing > area.right() and line_height > 0:
                x = area.x()
                y += line_height + self._spacing
                next_x = x + item_size.width() + self._spacing
                line_height = 0
            if not dry_run:
                item.setGeometry(QRect(QPoint(x, y), item_size))
            x = next_x
            line_height = max(line_height, item_size.height())
        return y + line_height - rect.y() + margins.bottom()


class GalleryWorker(QThread):
    """Genera miniaturas en segundo plano sin bloquear la interfaz."""

    thumb_ready = Signal(str, QImage)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self._paths = paths

    def run(self):
        for path in self._paths:
            if self.isInterruptionRequested():
                break
            try:
                # A la resolución exacta que la grilla pinta: generar la
                # miniatura más pequeña obligaría a estirarla al pintar y se
                # vería borrosa en la vista previa.
                thumb = image_loader.make_thumbnail(
                    path, (THUMB_GRID_ICON, THUMB_GRID_ICON)
                )
            except Exception:
                continue
            self.thumb_ready.emit(path, pil_to_qimage(thumb))


class ProcessWorker(QThread):
    """Procesa colores en segundo plano: reduce paleta y mapea a hilos."""

    status = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, image: Image.Image, max_colors: int, parent=None):
        super().__init__(parent)
        self._image = image
        self._max_colors = max_colors

    def run(self):
        try:
            self.status.emit("Reduciendo colores...")
            reduced = color_processor.reduce_palette(self._image, self._max_colors)
            if self.isInterruptionRequested():
                return
            self.status.emit("Mapeando hilos...")
            result = color_processor.map_to_threads(reduced)
            if self.isInterruptionRequested():
                return
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class StitchWorker(QThread):
    """Genera las puntadas (scanline) en segundo plano."""
    
    progress = Signal(int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image: Image.Image,
        threads_used,
        density: str,
        background_rgb: tuple | None = None,
        background_mask: np.ndarray | None = None,
        width_mm: float | None = None,
        height_mm: float | None = None,
        codes_median: int | None = None,
        min_area_mm2: float | None = None,
        connect_mm: float | None = None,
        guide: str = mockup.GUIDE_AUTO,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._threads_used = threads_used
        self._density = density
        self._background_rgb = background_rgb
        self._background_mask = background_mask
        self._width_mm = width_mm
        self._height_mm = height_mm
        self._codes_median = codes_median
        self._min_area_mm2 = min_area_mm2
        self._connect_mm = connect_mm
        self._guide = guide

    def run(self):
        try:
            pattern = stitch_generator.generate_stitches(
                self._image,
                self._threads_used,
                self._density,
                background_rgb=self._background_rgb,
                background_mask=self._background_mask,
                width_mm=self._width_mm,
                height_mm=self._height_mm,
                codes_median=self._codes_median,
                min_area_mm2=self._min_area_mm2,
                connect_mm=self._connect_mm,
                guide=self._guide,
            )
            if self.isInterruptionRequested():
                return
            # Las zonas se buscan sobre el resultado ya generado, que es donde
            # se puede saber cuántas veces se detiene la máquina de verdad. El
            # mockup no se conserva: `core.mockup` lo tira tras usarlo, así que
            # aquí la cobertura se calcula sin él y queda en 0.0, que la
            # interfaz no usa (solo cuenta y tamaño).
            zones = trim_zones.find_busy_zones(
                pattern.stitches,
                self._width_mm or stitch_generator.MAX_SIZE_MM,
                self._height_mm or stitch_generator.MAX_SIZE_MM,
                mockup=None,
                max_zones=2,
            )
            per_1000 = trim_zones.cuts_per_1000(pattern.stitches)
            self.done.emit((pattern, zones, per_1000))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExportWorker(QThread):
    """Escribe el archivo de bordado (DST o PES) en segundo plano."""

    done = Signal(str)
    failed = Signal(str)

    export_formats = {
        "PES": lambda stitches, threads, name, path: export_pes(
            stitches, threads, name=name, out_path=path
        ),
    }

    def __init__(
        self,
        stitches,
        threads_used,
        export_format: str,
        name: str,
        out_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self._stitches = stitches
        self._threads_used = threads_used
        self._export_format = export_format
        self._name = name
        self._out_path = out_path

    def run(self):
        try:
            if self._export_format == "PES":
                path = export_pes(
                    self._stitches,
                    self._threads_used,
                    name=self._name,
                    out_path=self._out_path,
                )
            elif self._export_format == "DST":
                path = export_dst(
                    self._stitches, name=self._name, out_path=self._out_path
                )
            else:
                raise ValueError("Formato de exportación no soportado")
            if self.isInterruptionRequested():
                return
            self.done.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ZonesDialog(QDialog):
    """La única pregunta que se le hace al usuario sobre el diseño.

    Se muestra UNO de estos diálogos por diseño, como mucho, y solo cuando el
    programa detecta que la máquina se va a detener muchas veces en un par de
    zonas concretas. El texto es deliberadamente cotidiano: describe cuántas
    veces se detiene la máquina y en cuántas zonas, sin nombrar el mecanismo
    interno ni las opciones técnicas.
    """

    def __init__(self, zones, per_1000: float, remembered: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zonas con muchas paradas")
        self.setModal(True)
        self._choice = remembered or mockup.GUIDE_AUTO
        self._remember = remembered is not None

        layout = QVBoxLayout(self)

        count = len(zones)
        title = QLabel(
            f"En este diseño la máquina se detiene unas {per_1000:.0f} veces "
            f"por cada 1000 pasos, casi todo en {count} "
            f"{'zona' if count == 1 else 'zonas'} marcadas en la vista previa."
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        detail = QLabel("\n".join(f"• {z.describe()}" for z in zones))
        detail.setWordWrap(True)
        layout.addWidget(detail)

        ask = QLabel("¿Qué prefieres?")
        layout.addWidget(ask)

        buttons = QDialogButtonBox(self)
        self._keep_btn = buttons.addButton(
            "Dejarlo como está", QDialogButtonBox.ButtonRole.RejectRole
        )
        self._simple_btn = buttons.addButton(
            "Simplificar esas zonas", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._auto_btn = buttons.addButton(
            "Decídelo tú", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._keep_btn.setToolTip("No se toca nada: el diseño queda exactamente igual.")
        self._simple_btn.setToolTip(
            "Quita más detalle del habitual en esas zonas. Adecuado para letras pequeñas."
        )
        self._auto_btn.setToolTip(
            "Solo se quita lo que no se ve: redondeos de un milímetro."
        )
        self._keep_btn.clicked.connect(lambda: self._choose(mockup.GUIDE_FAITHFUL))
        self._simple_btn.clicked.connect(lambda: self._choose(mockup.GUIDE_SIMPLE))
        self._auto_btn.clicked.connect(lambda: self._choose(mockup.GUIDE_AUTO))
        layout.addWidget(buttons)

        self._remember_box = QCheckBox("Recordar mi elección para los próximos diseños")
        self._remember_box.setChecked(self._remember)
        layout.addWidget(self._remember_box)

        self._highlight_default()

    def _highlight_default(self) -> None:
        """Marca el botón que corresponde a lo que se haría sin preguntar."""
        for button, mode in (
            (self._keep_btn, mockup.GUIDE_FAITHFUL),
            (self._simple_btn, mockup.GUIDE_SIMPLE),
            (self._auto_btn, mockup.GUIDE_AUTO),
        ):
            button.setDefault(mode == self._choice)

    def _choose(self, mode: str) -> None:
        self._choice = mode
        self.accept()

    @property
    def choice(self) -> str:
        return self._choice

    @property
    def remember(self) -> bool:
        return self._remember_box.isChecked()


def draw_zone_boxes(preview: Image.Image, zones, scale_px_per_mm: float):
    """Devuelve una COPIA del preview con las zonas marcadas.

    El recuadro va sobre la vista previa, no sobre el patrón: `pattern.preview`
    solo se usa para mostrar en pantalla (la exportación PES/DST se escribe a
    partir de `pattern.stitches`), así que marcar aquí no altera el archivo que
    sale a la máquina.
    """
    if not zones:
        return preview
    out = preview.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    s = scale_px_per_mm
    for zone in zones:
        box = zone.padded()
        rect = [
            box.x0 * s,
            box.y0 * s,
            box.x1 * s,
            box.y1 * s,
        ]
        draw.rectangle(rect, outline=(220, 30, 30), width=max(2, int(s)))
    return out


def _format_bytes(size: float) -> str:
    """Bytes en una unidad legible para el usuario (GB, MB...)."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def _volume_label(root: str) -> str:
    """Nombre bonito del volumen ("SANDISK", "Mi USB"), con la letra de recurso.

    En Windows se pregunta a la API por la etiqueta del volumen; si no la hay
    se usa la letra de unidad, que es lo que el usuario ve en el explorador.
    """
    fallback = os.path.basename(root.rstrip("\\/")) or root
    if platform.system() != "Windows":
        return fallback
    try:
        import ctypes

        name = ctypes.create_unicode_buffer(261)
        filesystem = ctypes.create_unicode_buffer(261)
        serial = ctypes.c_ulong()
        max_component = ctypes.c_ulong()
        flags = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            name,
            ctypes.sizeof(name),
            ctypes.byref(serial),
            ctypes.byref(max_component),
            ctypes.byref(flags),
            filesystem,
            ctypes.sizeof(filesystem),
        )
        if ok and name.value:
            return name.value
    except Exception:  # noqa: BLE001 - la etiqueta es un extra, nunca un fallo
        pass
    return fallback


def _list_removable_drives() -> list:
    """Unidades extraíbles conectadas ahora, con su espacio libre.

    Se usa `GetDriveTypeW` de la API de Windows en vez de una dependencia
    externa: el proyecto prioriza no añadir librerías, y `ctypes` ya viene en
    la biblioteca estándar. Fuera de Windows se devuelven las fijas montadas
    bajo `/Volumes` (macOS) o `/media` (Linux), que es lo equivalente.
    """
    drives: list = []
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes

            get_drives = ctypes.windll.kernel32.GetLogicalDrives
            get_type = ctypes.windll.kernel32.GetDriveTypeW
            mask = get_drives()
            for index in range(26):
                if not mask & (1 << index):
                    continue
                letter = chr(ord("A") + index)
                root = f"{letter}:\\"
                # 2 = DRIVE_REMOVABLE. Las de red (4) y fijas (3) no cuentan:
                # el usuario busca un USB, no un disco interno.
                if get_type(ctypes.c_wchar_p(root)) != 2:
                    continue
                try:
                    usage = shutil.disk_usage(root)
                except OSError:
                    continue
                drives.append(
                    {"root": root, "label": _volume_label(root), "free": usage.free}
                )
        except Exception:  # noqa: BLE001 - sin unidades extraíbles no hay tarjetas
            return []
        return drives

    roots: list = []
    if system == "Darwin":
        roots = sorted(Path("/Volumes").glob("*"))
    elif system == "Linux":
        roots = sorted(Path("/media").glob("*")) + sorted(Path("/mnt").glob("*"))
    for root in roots:
        if not root.is_dir():
            continue
        try:
            usage = shutil.disk_usage(str(root))
        except OSError:
            continue
        drives.append(
            {"root": str(root), "label": root.name, "free": usage.free}
        )
    return drives


def _open_folder_in_explorer(folder: str) -> None:
    """Abre `folder` en el explorador de archivos del sistema."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(folder)  # noqa: S606 - API propia de Windows
        elif system == "Darwin":
            subprocess.run(["open", folder], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)
    except Exception:  # noqa: BLE001 - abrir el explorador es un extra
        pass


class StepIndicator(QWidget):
    """Barra de los 4 pasos: marca el actual, los hechos y deja volver atrás.

    Cada paso es un botón plano. Solo se habilitan los que ya se pudieron
    completar, así que no se puede saltar a un paso whose requisitos no se
    cumplen. El clic emite `step_clicked` y es `MainWindow` quien decide si
    acepta el salto.
    """

    step_clicked = Signal(int)

    def __init__(self, titles, parent=None):
        super().__init__(parent)
        self._titles = list(titles)
        self._current = 0
        self._done = [False] * len(self._titles)
        self._buttons: list = []
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for index, title in enumerate(self._titles):
            if index:
                layout.addWidget(self._make_separator())
            button = QPushButton()
            button.setFlat(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(
                lambda _checked=False, i=index: self.step_clicked.emit(i)
            )
            self._buttons.append(button)
            layout.addWidget(button)

        self.set_state(0, [False] * len(self._titles))

    def _make_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"color: {NEUTRAL_BORDER};")
        return line

    def set_state(self, current: int, done: list) -> None:
        """`current` es el paso visible; `done[i]` si el paso i está completo."""
        self._current = current
        self._done = list(done)

        for index, button in enumerate(self._buttons):
            done = self._done[index]
            current = index == self._current
            # El visto del completado es el mismo acento que los botones: el
            # recorrido se lee de un vistazo sin cambiar de color.
            button.setIcon(
                _make_icon("check", ACCENT if current else ACCENT_HOVER, 13)
                if done
                else QIcon()
            )
            button.setText(f"{index + 1}. {self._titles[index]}")

            if current:
                button.setStyleSheet(
                    f"QPushButton {{ font-weight: bold; color: {ACCENT};"
                    f" border-bottom: 2px solid {ACCENT}; }}"
                )
            elif done:
                button.setStyleSheet(
                    f"QPushButton {{ color: {ACCENT_HOVER}; }}"
                )
            else:
                button.setStyleSheet(f"QPushButton {{ color: {NEUTRAL_TEXT}; }}")

            # Hacia atrás siempre se puede; hacia adelante solo a pasos ya
            # completados (o al inmediatamente siguiente, que aún no está
            # hecho pero es el que se está intentando completar).
            reachable = index <= current or any(self._done[:index])
            button.setEnabled(index != current and reachable)
            if index == current:
                button.setEnabled(False)


class FixedCellDelegate(QStyledItemDelegate):
    """Fuerza que TODA celda de la rejilla mida lo mismo.

    Sin esto, `QListWidget` dimensiona cada item con lo que devuelve su
    `sizeHint`, y el de fábrica usa el ancho del texto: con nombres largos la
    celda mide 152 px y con nombres cortos 138 px. Las columnas dejan de estar
    alineadas, quedan huecos entre ellas y el `QListWidget` deja de dibujar
    item en el `indexAt()` de un punto cae en ese hueco, así que el clic no
    selecciona nada. Medido: 21% de la superficie sin item y anchos de celda
    138/152 mezclados; con este delegate, 0.5%.
    """

    def sizeHint(self, option, index):  # noqa: N802 (Qt API)
        return QSize(
            self.parent().gridSize()
            if isinstance(self.parent(), QListWidget)
            else QSize(THUMB_GRID_CELL_W, THUMB_GRID_CELL_H)
        )


class ThumbnailGrid(QListWidget):
    """Rejilla de miniaturas que ocupa justo el alto de su contenido.

    `QListWidget` no sirve para esto: su `sizeHint()` es un (256, 192) de
    fábrica que no depende de los items, así que dentro de un layout el alto se
    lo reparte el layout y solo queda visible una fila, con el resto escondido
    detrás de una barra de scroll diminuta. Desplazar es tarea del `QScrollArea`
    de alrededor, que sí funciona con la columna entera.
    """

    def _content_height(self) -> int:
        # `FixedCellDelegate` hace que la celda mida exactamente `gridSize()`
        # y el view ya no añade espaciado entre celdas: los pasos reales son
        # 152 px en horizontal y 168 en vertical. Hay que usar el paso tal cual,
        # sin sumarle `spacing()`, o se cuentan menos columnas de las que caben
        # (638 // 160 = 3 en vez de 638 // 152 = 4) y el grupo queda con cientos
        # de píxeles de blanco al final, zona que no pertenece a ninguna
        # miniatura y por tanto no es seleccionable.
        pitch_y = self.gridSize().height()
        pitch_x = self.gridSize().width()
        width = max(self.viewport().width(), pitch_x)
        columns = max(1, width // pitch_x)
        rows = max(1, -(-self.count() // columns))
        return rows * pitch_y

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        base = super().sizeHint()
        return QSize(max(base.width(), self.gridSize().width()), self._content_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        # El `QScrollArea` decide si aparece la barra vertical comparando el
        # widget con su `minimumSizeHint`, así que el alto de contenido tiene
        # que estar aquí también, no solo en `sizeHint`.
        return QSize(
            max(super().minimumSizeHint().width(), self.gridSize().width()),
            self._content_height(),
        )


class ImageDropArea(QFrame):
    """Zona donde se arrastra una imagen para cargarla sin usar el explorador."""

    image_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(70)
        self.setStyleSheet(
            "QFrame { background-color: #fafafa; border: 2px dashed #c2c2c2;"
            " border-radius: 8px; }"
            f"QFrame:hover {{ border-color: {ACCENT}; }}"
        )

        layout = QVBoxLayout(self)
        text = QLabel("Arrastra una imagen aquí")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setStyleSheet("color: #777777; border: none; background: transparent;")
        layout.addWidget(text)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.image_dropped.emit(path)
                event.acceptProposedAction()
                return


class MainWindow(QMainWindow):
    """Ventana única de NeithLoom, organizada como un flujo guiado de 4 pasos.

    La lógica de procesamiento no cambió: solo se muestra un paso a la vez.
    Los workers, el generador de puntadas y los exportadores son los mismos.
    """

    STEP_TITLES = ("Elegir imagen", "Ajustar", "Generar bordado", "Guardar")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeithLoom")
        self.setMinimumSize(1000, 680)

        icon = find_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)

        self._image: Image.Image | None = None
        self._pixmap: QPixmap | None = None
        self._current_path: str | None = None
        self._gallery_thread: GalleryWorker | None = None
        self._process_thread: ProcessWorker | None = None
        # True cuando el proceso se relanza desde el paso 2 (Máximo de
        # colores) y no hay que arrastrar al usuario al paso 3 al terminar.
        self._process_in_place = False
        self._stitch_thread: StitchWorker | None = None
        self._export_thread: ExportWorker | None = None
        self._processed_image: Image.Image | None = None
        self._processed_result: color_processor.ProcessResult | None = None
        self._stitch_pattern: stitch_generator.StitchPattern | None = None
        self._busy_zones: list = []
        # Modo con el que se generó por última vez. Arranca con la elección
        # recordada si la hay, y si no con el automático, que es el que solo
        # quita redondeos y por tanto no cambia un diseño limpio.
        self._guide_choice: str = (
            QSettings().value(GUIDE_SETTING_KEY) or mockup.GUIDE_AUTO
        )
        # Guía con la que se generó el patrón actual. Sirve para no regenerar
        # en bucle: solo se vuelve a generar si el usuario elige otro modo.
        self._guide_used: str = self._guide_choice
        # Una sola pregunta por diseño: se vuelve a armar al cargar otra imagen.
        self._guide_asked = False
        self._design_name: str = "NeithLoom"
        self._background_rgb: tuple | None = None
        self._background_mask: np.ndarray | None = None
        self._picking_bg = False
        self._edit_mode: str | None = None  # None | "recolor" | "remove"
        self._saved_pixmap: QPixmap | None = None
        self._hoop: dict = hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX]

        self._save_cards: list = []
        self._usb_cards: list = []
        self._selected_save_dir: str = self._read_last_save_dir()
        self._export_format = "PES"

        self._build_ui()
        self._load_gallery()

        # Las unidades extraíbles se detectan cada pocos segundos: aparecen y
        # desaparecen mientras el usuario trabaja y no hay señal de Qt para
        # avisar, así que se sondea con un temporizador barato.
        self._usb_timer = QTimer(self)
        self._usb_timer.timeout.connect(self._refresh_usb_cards)
        self._usb_timer.start(2000)
        self._refresh_usb_cards()

    # ------------------------------------------------------------------
    # Armado de la interfaz
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._step_bar = StepIndicator(self.STEP_TITLES, self)
        self._step_bar.step_clicked.connect(self._on_step_clicked)
        layout.addWidget(self._step_bar)

        body = QHBoxLayout()
        body.setSpacing(6)

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._build_step_image())
        self._stack.addWidget(self._build_step_adjust())
        self._stack.addWidget(self._build_step_generate())
        self._stack.addWidget(self._build_step_save())
        body.addWidget(self._stack)

        # La vista previa se mantiene visible en todos los pasos porque es
        # también la superficie del "cuentagotas" (fondo y zonas), y el
        # usuario necesita ver el diseño mientras ajusta.
        self._image_label = PickImageLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setStyleSheet(
            "background-color: #fafafa; border: 1px solid #e0e0e0;"
            " border-radius: 8px;"
        )
        self._image_label.pixel_clicked.connect(self._on_pixel_clicked)
        body.addWidget(self._image_label, stretch=1)
        # El aro de la vista previa nace con la proporción del bastidor actual.
        self._apply_hoop_to_preview()

        # Estado vacío de la vista previa: carrete de hilo + qué hacer. Se
        # superpone al label y se oculta en cuanto hay diseño que enseñar.
        self._preview_empty = EmptyState(
            "spool",
            "Aquí verás tu bordado",
            "Elige una imagen para empezar",
            icon_size=72,
            parent=self._image_label,
        )
        self._image_label.set_empty_state(self._preview_empty)
        self._preview_empty.show()

        layout.addLayout(body, stretch=1)

        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        self._back_button = QPushButton("Atrás")
        self._back_button.clicked.connect(self._go_back)
        self._next_button = QPushButton("Continuar")
        self._next_button.setDefault(True)
        self._next_button.clicked.connect(self._go_next)
        self._help_button = QPushButton("Ayuda")
        self._help_button.clicked.connect(self._show_help)
        nav.addWidget(self._back_button)
        nav.addStretch()
        nav.addWidget(self._next_button)
        nav.addWidget(self._help_button)
        layout.addLayout(nav)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._refresh_steps()

    # ----- Paso 1: elegir imagen -------------------------------------

    def _build_step_image(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._drop_area = ImageDropArea()
        self._drop_area.image_dropped.connect(self._on_image_dropped)
        layout.addWidget(self._drop_area)

        # Un panel por carpeta, cada uno con su rejilla de miniaturas. Se
        # rellenan todos en `_load_gallery`, que lanza un único worker.
        self._gallery_sections: list[tuple[QGroupBox, QListWidget]] = []
        self._gallery_box = QWidget()
        self._gallery_layout = QVBoxLayout(self._gallery_box)
        self._gallery_layout.setContentsMargins(0, 0, 0, 0)
        self._gallery_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._gallery_box)
        layout.addWidget(scroll, stretch=1)

        self._open_button = QPushButton("Buscar en otra carpeta...")
        self._open_button.setIcon(_make_icon("image", NEUTRAL_TEXT))
        self._open_button.clicked.connect(self.open_image)
        layout.addWidget(self._open_button)

        return page

    def _load_gallery(self) -> None:
        """Crea un panel por carpeta y lanza un worker para todas las miniaturas.

        Se conserva el orden por fecha de modificación (más reciente primero)
        y el límite de `image_loader`, pero agrupando por carpeta de origen.
        """
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        self._gallery_sections.clear()
        pending: list[str] = []

        for directory in image_loader.default_search_dirs():
            paths = image_loader.list_recent_images([directory], limit=GALLERY_PER_DIR)
            group = QGroupBox(directory.name)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(4, 4, 4, 4)
            # El grupo se tiene que añadir AL LAYOUT, no solo crearse: sin esta
            # línea los paneles quedan huérfanos, se rellenan igual pero no se
            # ven nunca, y la columna se queda vacía.
            self._gallery_layout.addWidget(group)

            if not paths:
                # Una carpeta sin imágenes se dice en voz alta en vez de
                # desaparecer: si no, el usuario cree que la app no la ha visto.
                empty = QLabel("No hay imágenes en esta carpeta")
                empty.setStyleSheet("color: #777; padding: 6px;")
                group_layout.addWidget(empty)
            else:
                grid = ThumbnailGrid()
                grid.setViewMode(QListView.ViewMode.IconMode)
                grid.setIconSize(QSize(THUMB_GRID_ICON, THUMB_GRID_ICON))
                grid.setGridSize(QSize(THUMB_GRID_CELL_W, THUMB_GRID_CELL_H))
                grid.setWrapping(True)
                grid.setSpacing(THUMB_GRID_SPACING)
                grid.setMovement(QListView.Movement.Static)
                grid.setResizeMode(QListView.ResizeMode.Adjust)
                # Delegate de celda fija: sin esto Qt mide cada item con el
                # ancho de su texto y quedan huecos sin item entre columnas,
                # donde el clic no selecciona nada.
                grid.setItemDelegate(FixedCellDelegate(grid))
                # SIN `setUniformItemSizes(True)`: congela el alto de la celda
                # con el primer item, y como los items se añaden sin icono
                # todavía, ese alto queda en el de una línea de texto (12 px).
                # Los iconos llegan después del worker y no lo recomputan: la
                # celda se queda en una franja de 12 px y casi todo el ancho de
                # la miniatura deja de ser clicable.
                grid.setFrameShape(QFrame.Shape.NoFrame)
                # La rejilla no se desplaza ella misma: ocupa todo su alto y es
                # el `QScrollArea` del panel el que baja por las carpetas.
                grid.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                grid.itemClicked.connect(self._on_gallery_click)
                for path in paths:
                    # El nombre se recorta al ancho real de la celda: con el
                    # nombre entero, 38 de 44 miniaturas se salían de ella.
                    item = QListWidgetItem(
                        short_name(
                            Path(path).name,
                            grid.fontMetrics(),
                            THUMB_GRID_CELL_W - 8,
                        )
                    )
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    item.setToolTip(path)
                    grid.addItem(item)
                group_layout.addWidget(grid)
                self._gallery_sections.append((group, grid))
                pending.extend(paths)

        self._gallery_layout.addStretch()

        # El worker recorre `pending` en orden: se ordena por fecha entre TODAS
        # las carpetas, no carpeta por carpeta. Si no, la primera carpeta del
        # panel bloquea al resto con sus imágenes viejas (una captura de 88
        # megapíxeles tardaba 895 ms sola) y las miniaturas nuevas, que son las
        # que el usuario quiere, aparecían al final.
        pending.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        if pending:
            self._gallery_thread = GalleryWorker(pending, self)
            self._gallery_thread.thumb_ready.connect(self._on_thumb_ready)
            self._gallery_thread.start()

    def _on_thumb_ready(self, path: str, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        for _, grid in self._gallery_sections:
            for row in range(grid.count()):
                item = grid.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == path:
                    item.setIcon(QIcon(pixmap))
                    return

    def _on_gallery_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.load_image(path)

    def _on_image_dropped(self, path: str) -> None:
        if Path(path).suffix.lower() not in image_loader.SUPPORTED_SUFFIXES:
            self._status_label.setText("Ese archivo no es una imagen (.jpg o .png)")
            return
        self.load_image(path)

    # ----- Paso 2: ajustar -------------------------------------------

    def _build_step_adjust(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        question = QLabel("¿Qué colores de hilo quieres usar?")
        question.setWordWrap(True)
        layout.addWidget(question)

        self._process_button = QPushButton("Ver los colores de hilo")
        self._process_button.setIcon(_make_icon("spool", ACCENT_ON))
        self._process_button.setStyleSheet(_primary_button_style())
        self._process_button.clicked.connect(self.process_image)
        layout.addWidget(self._process_button)

        # Hilos detectados: un círculo del color real más el nombre más
        # próximo del catálogo Brother (no un código suelto que no se entiende).
        threads_group = QGroupBox("Hilos que vas a usar")
        threads_layout = QVBoxLayout(threads_group)
        self._threads_flow = QWidget()
        self._threads_flow_layout = FlowLayout(self._threads_flow, spacing=8)
        threads_layout.addWidget(self._threads_flow)
        threads_group.setVisible(False)
        self._threads_group = threads_group
        layout.addWidget(threads_group)

        # Máximo de colores: decisión que ya tenía la interfaz de 7 pasos
        # ("Máximo de colores" en vez de "K") y que se perdió al rehacer la
        # ventana en 4 pasos, cuando el valor quedó clavado en el código.
        colors_hint = (
            "Cuántos colores de hilo como máximo puede usar el diseño.\n"
            "Al cambiarlo se vuelven a calcular los colores de hilo."
        )
        colors_row = QHBoxLayout()
        colors_label = QLabel("Máximo de colores del bordado")
        colors_label.setToolTip(colors_hint)
        colors_row.addWidget(colors_label)
        self._max_colors_combo = QComboBox()
        for count in color_processor.COLOR_OPTIONS:
            self._max_colors_combo.addItem(f"{count} colores", count)
        self._max_colors_combo.setCurrentIndex(
            max(0, self._max_colors_combo.findData(color_processor.DEFAULT_COLORS))
        )
        self._max_colors_combo.setToolTip(colors_hint)
        self._max_colors_combo.currentIndexChanged.connect(
            self._on_max_colors_changed
        )
        colors_row.addWidget(self._max_colors_combo, stretch=1)
        layout.addLayout(colors_row)

        # Bastidores: tarjetas con el óvalo a escala, seleccionables con clic.
        hoop_group = QGroupBox("¿De qué tamaño es tu bastidor?")
        hoop_outer = QVBoxLayout(hoop_group)
        self._hoops_host = QWidget()
        self._hoops_flow = FlowLayout(self._hoops_host, spacing=8)
        hoop_outer.addWidget(self._hoops_host)

        biggest = max(
            h["width_mm"] * h["height_mm"] for h in hoops.HOOPS
        )
        self._hoop_area_max = biggest
        default_hoop = hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX]
        self._custom_default_size = (
            float(default_hoop["width_mm"]),
            float(default_hoop["height_mm"]),
        )
        self._custom_hoop = hoops.custom_hoop(*self._custom_default_size)
        self._hoop_cards: list[HoopCard] = []
        for index, hoop in enumerate([*hoops.HOOPS, self._custom_hoop]):
            custom = bool(hoop.get("custom"))
            card = HoopCard(
                hoop, selected=not custom and index == hoops.DEFAULT_HOOP_INDEX
            )
            card.set_ratio((hoop["width_mm"] * hoop["height_mm"] / biggest) ** 0.5)
            card.clicked.connect(self._on_hoop_card_clicked)
            self._hoop_cards.append(card)
            self._hoops_flow.addWidget(card)
            if custom:
                self._custom_card = card

        # Medidas libres: solo aparecen cuando "Personalizado" es el bastidor
        # elegido, justo debajo de las tarjetas.
        self._custom_fields = QWidget()
        fields_layout = QHBoxLayout(self._custom_fields)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        fields_layout.setSpacing(6)
        fields_layout.addWidget(QLabel("Ancho"))
        self._custom_width = self._make_size_spin(self._custom_default_size[0] / 10)
        fields_layout.addWidget(self._custom_width)
        fields_layout.addWidget(QLabel("Alto"))
        self._custom_height = self._make_size_spin(self._custom_default_size[1] / 10)
        fields_layout.addWidget(self._custom_height)
        fields_layout.addStretch()
        self._custom_fields.setVisible(False)
        hoop_outer.addWidget(self._custom_fields)

        self._custom_error = QLabel("")
        self._custom_error.setWordWrap(True)
        self._custom_error.setStyleSheet("color: #b00020; font-weight: bold;")
        self._custom_error.setVisible(False)
        hoop_outer.addWidget(self._custom_error)

        self._hoop_info_label = QLabel("")
        self._hoop_info_label.setStyleSheet("color: #666;")
        self._hoop_info_label.setWordWrap(True)
        hoop_outer.addWidget(self._hoop_info_label)
        layout.addWidget(hoop_group)

        self._hoop = hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX]
        self._update_hoop_info()

        layout.addStretch()
        return page

    def _make_size_spin(self, value_cm: float) -> QDoubleSpinBox:
        """Campo de medida en cm.

        El rango de entrada es amplio a propósito: si el tope se mete en el
        Qt solo, el valor que se sale se recorta en silencio y el usuario
        nunca llega a ver por qué no se aplicó. Aquí el número entra entero y
        avisa `self._custom_error` de que está fuera de rango.
        """
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" cm")
        spin.setMinimumWidth(96)
        spin.setValue(value_cm)
        # Sin teclado en vivo: el cambio se emite al salir del campo o con
        # Enter, para no recalcular el bordado con cada dígito tecleado.
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._on_custom_size_changed)
        return spin

    # ----- Paso 3: generar -------------------------------------------

    def _build_step_generate(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        group = QGroupBox("Generar bordado")
        form = QVBoxLayout(group)

        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("¿Qué tan apretado quieres el bordado?"))
        self._density_combo = QComboBox()
        for density in stitch_generator.DENSITY_OPTIONS:
            self._density_combo.addItem(density, density)
        self._density_combo.setToolTip(
            "Alta = puntadas más juntas y másogre para el resultado.\n"
            "Baja = menos puntadas y archivo más ligero."
        )
        density_row.addWidget(self._density_combo, stretch=1)
        self._density_button = QPushButton("Cambiar")
        self._density_button.clicked.connect(self.change_density)
        density_row.addWidget(self._density_button)
        form.addLayout(density_row)

        self._stitch_button = QPushButton("Generar bordado")
        self._stitch_button.setIcon(_make_icon("needle", ACCENT_ON))
        self._stitch_button.setStyleSheet(_primary_button_style())
        self._stitch_button.clicked.connect(self.generate_stitches)
        form.addWidget(self._stitch_button)

        layout.addWidget(group)

        edit = QGroupBox("Editar el diseño")
        edit_layout = QVBoxLayout(edit)

        self._pick_bg_button = QPushButton(_PICK_BG_TEXT)
        self._pick_bg_button.clicked.connect(self._start_pick_background)
        edit_layout.addWidget(self._pick_bg_button)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Fondo que estamos quitando"))
        self._bg_swatch = QLabel("")
        self._bg_swatch.setFixedSize(40, 20)
        self._bg_swatch.setFrameShape(QFrame.Shape.StyledPanel)
        bg_row.addWidget(self._bg_swatch)
        bg_row.addStretch()
        edit_layout.addLayout(bg_row)

        self._recolor_button = QPushButton(_RECOLOR_TEXT)
        self._recolor_button.clicked.connect(self._start_recolor)
        edit_layout.addWidget(self._recolor_button)

        self._remove_button = QPushButton(_REMOVE_TEXT)
        self._remove_button.clicked.connect(self._start_remove)
        edit_layout.addWidget(self._remove_button)

        layout.addWidget(edit)

        # Estado vacío del paso, al final donde de verdad queda el hueco: los
        # controles arriba y abajo el mensaje de lo que falta por hacer.
        self._generate_empty = EmptyState(
            "needle",
            "Todavía no hay bordado",
            "Pulsa «Generar bordado» para crear las puntadas",
            icon_size=56,
        )
        layout.addWidget(self._generate_empty, stretch=1)
        return page

    # ----- Paso 4: guardar -------------------------------------------

    def _build_step_save(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        dest = QGroupBox("¿Dónde lo guardamos?")
        dest_layout = QVBoxLayout(dest)
        self._cards_layout = QVBoxLayout()
        dest_layout.addLayout(self._cards_layout)

        home = Path.home()
        for title, directory in (
            ("Descargas", home / "Downloads"),
            ("Imágenes", home / "Pictures"),
        ):
            self._cards_layout.addWidget(self._make_save_card(title, directory))

        self._usb_cards_box = QWidget()
        self._usb_cards_layout = QVBoxLayout(self._usb_cards_box)
        self._usb_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._usb_cards_layout.setSpacing(6)
        dest_layout.addWidget(self._usb_cards_box)

        self._other_folder_button = QPushButton("Elegir otra carpeta...")
        self._other_folder_button.clicked.connect(self._choose_other_folder)
        dest_layout.addWidget(self._other_folder_button)

        layout.addWidget(dest)

        file_box = QGroupBox("El archivo")
        file_layout = QVBoxLayout(file_box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Ponle un nombre"))
        self._design_name_edit = QLineEdit(self._design_name)
        self._design_name_edit.textChanged.connect(self._update_design_name)
        name_row.addWidget(self._design_name_edit, stretch=1)
        file_layout.addLayout(name_row)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("¿Para qué máquina?"))
        self._format_combo = QComboBox()
        self._format_combo.addItem("PES (Brother)", "PES")
        self._format_combo.addItem("DST (Tajima)", "DST")
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_row.addWidget(self._format_combo, stretch=1)
        file_layout.addLayout(format_row)

        self._export_button = QPushButton("Guardar bordado")
        self._export_button.setIcon(_make_icon("save", ACCENT_ON))
        self._export_button.setStyleSheet(_primary_button_style())
        self._export_button.clicked.connect(self.export_image)
        file_layout.addWidget(self._export_button)

        self._export_status_label = QLabel("")
        self._export_status_label.setWordWrap(True)
        file_layout.addWidget(self._export_status_label)

        layout.addWidget(file_box)
        layout.addStretch()
        return page

    def _make_save_card(
        self,
        title: str,
        directory: Path,
        subtitle: str | None = None,
        enabled: bool = True,
    ) -> QFrame:
        """Tarjeta de destino: nombre, ruta y espacio libre del volumen."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        head = QLabel(f"<b>{title}</b>")
        layout.addWidget(head)

        if subtitle is not None:
            detail = QLabel(subtitle)
            detail.setStyleSheet("color: #666;")
            layout.addWidget(detail)
        else:
            detail = QLabel(str(directory))
            detail.setStyleSheet("color: #666;")
            layout.addWidget(detail)

        free = QLabel("")
        free.setStyleSheet("color: #666;")
        layout.addWidget(free)

        card._path = str(directory)
        card.setEnabled(enabled)
        if enabled:
            self._paint_save_card(card, selected=str(directory) == self._selected_save_dir)
            # Un `QFrame` no tiene click propio: se conecta el evento del frame
            # a un manejador que elige esta tarjeta.
            card.mousePressEvent = (
                lambda _event, path=card._path: self._select_save_dir(path)
            )
        else:
            card.setStyleSheet("QFrame { color: #999; }")

        if enabled:
            self._update_free_label(free, directory)
        self._save_cards.append(card)
        return card

    def _update_free_label(self, label: QLabel, directory: Path) -> None:
        """Muestra el espacio libre del volumen, no el de la carpeta.

        `shutil.disk_usage` acepta un archivo o una carpeta y devuelve el del
        volumen, que es lo que interesa: un USB del 8 GB libre muestra 8 GB
        aunque la carpeta destino esté casi vacía.
        """
        try:
            usage = shutil.disk_usage(str(directory))
        except OSError:
            label.setText("")
            return
        label.setText(f"{_format_bytes(usage.free)} libres")

    def _refresh_usb_cards(self) -> None:
        """Redibuja las tarjetas de USB con lo que hay conectado ahora mismo.

        Se llama al abrir la ventana y luego cada 2 s. Solo reconstruye si la
        lista cambió, para no perder el hover ni repintar en cada tictac.
        """
        drives = _list_removable_drives()
        signature = [(d["root"], d["label"]) for d in drives]
        if signature == getattr(self, "_usb_signature", None):
            return
        self._usb_signature = signature

        while self._usb_cards_layout.count():
            item = self._usb_cards_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._usb_cards.clear()

        if not drives:
            # La tarjeta se muestra igual, deshabilitada: el usuario ve que la
            # opción existe y solo falta enchufar algo.
            self._usb_cards.append(
                self._make_save_card(
                    "USB",
                    Path(""),
                    subtitle="Conecta un USB para guardar aquí",
                    enabled=False,
                )
            )
            return

        for drive in drives:
            root = Path(drive["root"])
            self._usb_cards.append(
                self._make_save_card(
                    drive["label"],
                    root,
                    subtitle=f"{drive['label']} — {root}",
                )
            )

    def _select_save_dir(self, path: str) -> None:
        self._selected_save_dir = path
        for card in self._save_cards + self._usb_cards:
            self._paint_save_card(card, selected=getattr(card, "_path", None) == path)
        self._status_label.setText(f"Destino: {Path(path).name or path}")

    def _paint_save_card(self, card: QFrame, selected: bool) -> None:
        card.setStyleSheet(_selected_card_style() if selected else _unselected_card_style())

    def _choose_other_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Elige la carpeta de destino", self._selected_save_dir
        )
        if not folder:
            return
        self._selected_save_dir = folder
        self._remember_save_dir(folder)
        for card in self._save_cards + self._usb_cards:
            self._paint_save_card(
                card, selected=getattr(card, "_path", None) == folder
            )
        card = self._make_save_card(Path(folder).name or folder, Path(folder))
        self._paint_save_card(card, selected=True)
        self._status_label.setText(f"Destino: {folder}")

    def _remember_save_dir(self, path: str) -> None:
        """Guarda la última carpeta usada para preseleccionarla la próxima vez."""
        QSettings().setValue(LAST_SAVE_DIR_KEY, path)

    def _read_last_save_dir(self) -> str:
        stored = QSettings().value(LAST_SAVE_DIR_KEY)
        if stored and Path(stored).is_dir():
            return str(stored)
        downloads = Path.home() / "Downloads"
        return str(downloads if downloads.is_dir() else Path.home())

    # ------------------------------------------------------------------
    # Navegación entre pasos
    # ------------------------------------------------------------------

    def _step_done(self, step: int) -> bool:
        """Cada paso se completa solo cuando hay algo real hecho, no solo visible."""
        if step == 0:
            return self._image is not None
        if step == 1:
            # El paso no queda hecho si el bastidor Personalizado está elegido
            # con una medida fuera del rango permitido.
            return self._processed_result is not None and self._custom_size_valid()
        if step == 2:
            return self._stitch_pattern is not None
        return True

    def _first_incomplete_step(self) -> int:
        for step in range(len(self.STEP_TITLES) - 1):
            if not self._step_done(step):
                return step
        return len(self.STEP_TITLES) - 1

    def _refresh_steps(self) -> None:
        current = self._stack.currentIndex()
        self._step_bar.set_state(current, [self._step_done(i) for i in range(4)])
        self._back_button.setEnabled(current > 0)
        self._next_button.setEnabled(self._step_done(current))
        # El estado vacío del paso 3 solo aparece cuando no hay patrón todavía.
        self._generate_empty.setVisible(self._stitch_pattern is None)
        if current == 3:
            self._next_button.setText("Guardar")
            self._next_button.setEnabled(
                self._stitch_pattern is not None and self._export_thread is None
            )
        else:
            self._next_button.setText("Continuar")

    def _on_step_clicked(self, step: int) -> None:
        """Clic en el indicador: solo se permite hacia atrás o a un paso hecho."""
        current = self._stack.currentIndex()
        if step == current:
            return
        if step < current or self._step_done(step - 1):
            self._stack.setCurrentIndex(step)
            self._refresh_steps()

    def _go_next(self) -> None:
        current = self._stack.currentIndex()
        if current == 1 and self._processed_result is None:
            self.process_image()
            return
        if current == 2 and self._stitch_pattern is None:
            self.generate_stitches()
            return
        if current == 3:
            self.export_image()
            return
        if self._step_done(current):
            self._stack.setCurrentIndex(current + 1)
            self._refresh_steps()

    def _go_back(self) -> None:
        current = self._stack.currentIndex()
        if current > 0:
            self._stack.setCurrentIndex(current - 1)
            self._refresh_steps()

    def _show_help(self) -> None:
        HelpDialog(self).exec()

    # ------------------------------------------------------------------
    # Carga de imagen
    # ------------------------------------------------------------------

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", str(Path.home()), IMAGE_FILTER
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        try:
            image = image_loader.load_resized(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "No se pudo abrir", str(exc))
            return

        self._image = image
        self._pixmap = pil_to_qpixmap(image)
        self._current_path = path
        self._processed_image = None
        self._processed_result = None
        self._stitch_pattern = None
        self._background_rgb = None
        self._background_mask = None
        self._busy_zones = []
        self._guide_asked = False
        self._exit_pick_mode()
        self._update_design_name_from_path()

        self._refresh_thread_swatches([])
        self._export_status_label.setText("")
        self._refresh_preview()
        self._status_label.setText(f"Imagen cargada: {Path(path).name}")
        self._go_to_step(1)

    def _update_design_name_from_path(self) -> None:
        if self._current_path:
            self._design_name = Path(self._current_path).stem
            self._design_name_edit.setText(self._design_name)

    def _go_to_step(self, step: int) -> None:
        self._stack.setCurrentIndex(step)
        self._refresh_steps()

    def _update_design_name(self) -> None:
        text = self._design_name_edit.text().strip()
        if text:
            self._design_name = text
        elif self._current_path:
            self._design_name = Path(self._current_path).stem
        else:
            self._design_name = "NeithLoom"

    def _on_format_changed(self) -> None:
        self._export_format = str(self._format_combo.currentData())

    def _refresh_preview(self) -> None:
        # Sin imagen no hay pixmap: se muestra el estado vacío del carrete en
        # lugar de un rectángulo gris con un texto suelto.
        self._preview_empty.setVisible(self._pixmap is None)
        if self._pixmap is None:
            self._image_label.set_source_pixmap(None)
            self._image_label.update()
            return
        if self._processed_image is not None:
            self._image_label.set_source_size(
                self._processed_image.width, self._processed_image.height
            )
        # El recorte elíptico lo hace el propio label en paintEvent; aquí solo
        # se le pasa el pixmap original a escala completa.
        self._image_label.set_source_pixmap(self._pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._image is not None:
            self._refresh_preview()

    # ------------------------------------------------------------------
    # Procesado de color
    # ------------------------------------------------------------------

    def process_image(self, in_place: bool = False) -> None:
        """Calcula la paleta de hilos de la imagen cargada.

        `in_place` es para recalcular desde el propio paso 2 (cambio del
        "Máximo de colores"): no hay que sacar al usuario de la pantalla en
        la que acaba de tocar nada.
        """
        if self._image is None or self._process_thread is not None:
            return

        self._process_in_place = in_place
        self._process_button.setEnabled(False)
        self._status_label.setText("Procesando...")

        self._process_thread = ProcessWorker(
            self._image, self._max_colors_combo.currentData(), self
        )
        self._process_thread.status.connect(self._status_label.setText)
        self._process_thread.done.connect(self._on_process_done)
        self._process_thread.failed.connect(self._on_process_failed)
        self._process_thread.finished.connect(self._on_process_finished)
        self._process_thread.start()

    def _on_process_done(self, result: color_processor.ProcessResult) -> None:
        self._processed_image = result.image
        self._processed_result = result
        self._stitch_pattern = None
        self._busy_zones = []
        self._background_mask = None
        self._background_rgb = None
        self._pixmap = pil_to_qpixmap(result.image)

        self._refresh_thread_swatches(result.threads_used)
        self._status_label.setText("Listo para generar el bordado")
        self._export_status_label.setText("")

        self._refresh_preview()
        self._refresh_steps()
        if not self._process_in_place:
            self._go_to_step(2)

    def _on_process_failed(self, message: str) -> None:
        self._status_label.setText("Error al procesar")
        QMessageBox.warning(self, "Error", f"No se pudo procesar la imagen:\n{message}")

    def _on_process_finished(self) -> None:
        self._process_button.setEnabled(self._image is not None)
        self._process_thread = None
        self._refresh_steps()

    def _on_max_colors_changed(self, _index: int) -> None:
        """Cambio en "Máximo de colores": se repite el paso 3 con el tope nuevo.

        Solo hay algo que recalcular si ya se había procesado; si no, el nuevo
        valor se usará la primera vez que se pulse "Ver los colores de hilo".
        """
        if self._processed_result is None or self._process_thread is not None:
            return
        self.process_image(in_place=True)

    # ------------------------------------------------------------------
    # Cuentagotas: fondo y edición de zonas
    # ------------------------------------------------------------------

    def _start_pick_background(self) -> None:
        if self._processed_image is None or self._edit_mode is not None:
            return
        if self._picking_bg:
            self._exit_pick_mode()
            self._status_label.setText("Selección cancelada")
            return
        self._picking_bg = True
        self._enter_pick_state(
            self._pick_bg_button, "Haz clic en el color que quieres como fondo"
        )

    def _start_recolor(self) -> None:
        if self._processed_image is None or self._picking_bg:
            return
        if self._edit_mode == "recolor":
            self._exit_pick_mode()
            self._status_label.setText("Selección cancelada")
            return
        if self._edit_mode is not None:
            return
        self._edit_mode = "recolor"
        self._enter_pick_state(
            self._recolor_button, "Haz clic en la zona cuyo color quieres cambiar"
        )

    def _start_remove(self) -> None:
        if self._processed_image is None or self._picking_bg:
            return
        if self._edit_mode == "remove":
            self._exit_pick_mode()
            self._status_label.setText("Selección cancelada")
            return
        if self._edit_mode is not None:
            return
        self._edit_mode = "remove"
        self._enter_pick_state(
            self._remove_button, "Haz clic en la zona que quieres eliminar"
        )

    def _enter_pick_state(self, button: QPushButton, message: str) -> None:
        self._saved_pixmap = self._pixmap
        self._pixmap = pil_to_qpixmap(self._processed_image)
        self._refresh_preview()
        self._image_label.set_pick_mode(True)
        button.setText(_CANCEL_TEXT)
        self._status_label.setText(message)

    def _exit_pick_mode(self) -> None:
        if not self._picking_bg and self._edit_mode is None:
            return
        self._picking_bg = False
        self._edit_mode = None
        self._image_label.set_pick_mode(False)
        self._pick_bg_button.setText(_PICK_BG_TEXT)
        self._recolor_button.setText(_RECOLOR_TEXT)
        self._remove_button.setText(_REMOVE_TEXT)
        if self._saved_pixmap is not None:
            self._pixmap = self._saved_pixmap
            self._saved_pixmap = None
            self._refresh_preview()

    def _on_pixel_clicked(self, x: int, y: int) -> None:
        if self._processed_image is None:
            return
        if self._picking_bg:
            self._on_background_picked(x, y)
        elif self._edit_mode == "recolor":
            self._on_recolor_picked(x, y)
        elif self._edit_mode == "remove":
            self._on_remove_picked(x, y)

    def _on_background_picked(self, x: int, y: int) -> None:
        self._background_rgb = self._processed_image.getpixel((x, y))[:3]
        self._bg_swatch.setStyleSheet(
            "background-color: rgb({},{},{});".format(*self._background_rgb)
        )
        self._exit_pick_mode()
        self._status_label.setText("Fondo actualizado")

        # Si ya hay puntadas, regenerarlas con el nuevo fondo (con un pequeño
        # retardo para que el usuario vea el mensaje "Fondo actualizado").
        if self._stitch_pattern is not None:
            QTimer.singleShot(400, self.generate_stitches)

    def _apply_recolor(self, old_rgb: tuple, new_code: str, new_rgb: tuple) -> None:
        arr = np.asarray(self._processed_image.convert("RGB")).copy()
        zone = (arr == old_rgb).all(axis=2)
        if not bool(zone.any()):
            self._status_label.setText("No se encontró esa zona")
            return
        arr[zone] = new_rgb
        self._processed_image = Image.fromarray(arr)
        codes = {t.code for t in self._processed_result.threads_used}
        if new_code not in codes:
            self._processed_result.threads_used.append(
                color_processor.thread_use_from_code(new_code)
            )
        self._pixmap = pil_to_qpixmap(self._processed_image)
        self._refresh_preview()
        if self._stitch_pattern is not None:
            self._status_label.setText("Regenerando el bordado...")
            QTimer.singleShot(300, self.generate_stitches)
        else:
            self._status_label.setText("Zona recoloreada")

    def _on_recolor_picked(self, x: int, y: int) -> None:
        old_rgb = tuple(self._processed_image.getpixel((x, y))[:3])
        self._exit_pick_mode()
        if self._stitch_thread is not None:
            self._status_label.setText("Todavía se está generando el bordado")
            return
        dialog = ThreadPickerDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._status_label.setText("Cambio de color cancelado")
            return
        new_code = dialog.selected_code()
        if new_code is None:
            self._status_label.setText("Cambio de color cancelado")
            return
        new_rgb = threads.THREAD_RGB[new_code]
        if old_rgb == new_rgb:
            self._status_label.setText("Ese color ya corresponde a ese hilo")
            return
        self._apply_recolor(old_rgb, new_code, new_rgb)

    def _on_remove_picked(self, x: int, y: int) -> None:
        rgb = tuple(self._processed_image.getpixel((x, y))[:3])
        self._exit_pick_mode()
        if self._stitch_thread is not None:
            self._status_label.setText("Todavía se está generando el bordado")
            return
        if self._background_mask is None:
            self._background_mask = np.zeros(
                (self._processed_image.height, self._processed_image.width),
                dtype=bool,
            )
        zone = (np.asarray(self._processed_image) == rgb).all(axis=2)
        if not bool(zone.any()):
            self._status_label.setText("No se encontró esa zona")
            return
        self._background_mask |= zone
        if self._stitch_pattern is not None:
            self._status_label.setText("Regenerando el bordado...")
            QTimer.singleShot(300, self.generate_stitches)
        else:
            self._status_label.setText("Zona eliminada")

    def change_density(self) -> None:
        if self._stitch_pattern is None or self._stitch_thread is not None:
            return
        density = str(self._density_combo.currentData())
        self._status_label.setText(f"Aplicando densidad {density}...")
        self.generate_stitches()

    def _update_hoop_info(self) -> None:
        self._hoop_info_label.setText(f"Bastidor: {self._hoop['name']}")

    def _refresh_thread_swatches(self, threads_used) -> None:
        """Pinta un círculo por hilo detectado con su nombre del catálogo."""
        flow = self._threads_flow_layout
        for index in reversed(range(flow.count())):
            item = flow.takeAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        threads = list(threads_used)
        # Sin hilos detectados el grupo sobra: se oculta en vez de dejar un
        # recuadro vacio con un texto de instruccion.
        self._threads_group.setVisible(bool(threads))
        for use in threads:
            flow.addWidget(
                ThreadSwatch(
                    use.rgb, use.code, use.name, getattr(use, "usage_fraction", 0.0)
                )
            )

    def _apply_hoop_to_preview(self) -> None:
        """La elipse de la vista previa adopta la proporción del bastidor."""
        self._image_label.set_hoop_aspect(
            self._hoop["width_mm"], self._hoop["height_mm"]
        )

    def _on_hoop_card_clicked(self, hoop: dict) -> None:
        if hoop is self._hoop:
            return
        self._apply_hoop(hoop)

    def _apply_hoop(self, hoop: dict) -> None:
        """Único camino por el que cambia el bastidor elegido.

        Lo comparten los clics de las tarjetas y la edición del tamaño
        personalizado: tarjetas, vista previa, información, visibilidad de los
        campos libres y, si ya había un bordado generado, regeneración con las
        medidas nuevas.
        """
        self._hoop = hoop
        custom = bool(hoop.get("custom"))
        self._custom_fields.setVisible(custom)
        if not custom:
            self._custom_error.setVisible(False)
        for card in self._hoop_cards:
            card.set_selected(card.hoop is hoop)
        self._apply_hoop_to_preview()
        self._update_hoop_info()
        self._exit_pick_mode()
        if custom:
            self._validate_custom_size()
        if self._stitch_pattern is not None and self._stitch_thread is None:
            self._status_label.setText("Bastidor cambiado - regenerando el bordado...")
            QTimer.singleShot(300, self.generate_stitches)
        elif self._image is not None:
            self._status_label.setText(f"Bastidor: {self._hoop['name']}")
        self._refresh_steps()

    def _custom_size_valid(self) -> bool:
        """Falso solo con "Personalizado" elegido y una medida fuera de rango."""
        if not self._hoop.get("custom"):
            return True
        low, high = hoops.MIN_CUSTOM_CM, hoops.MAX_CUSTOM_CM
        return (
            low <= self._custom_width.value() <= high
            and low <= self._custom_height.value() <= high
        )

    def _validate_custom_size(self) -> bool:
        """Escribe el aviso de rango (o lo borra) y devuelve si los campos valen."""
        low, high = hoops.MIN_CUSTOM_CM, hoops.MAX_CUSTOM_CM
        out = []
        if not low <= self._custom_width.value() <= high:
            out.append("ancho")
        if not low <= self._custom_height.value() <= high:
            out.append("alto")
        if len(out) == 2:
            message = f"El ancho y el alto deben estar entre {low:g} y {high:g} cm."
        elif out:
            message = f"El {out[0]} debe estar entre {low:g} y {high:g} cm."
        else:
            message = ""
        self._custom_error.setText(message)
        self._custom_error.setVisible(bool(message))
        return not message

    def _on_custom_size_changed(self, _value: float) -> None:
        # Los campos están ocultos y no mandan salvo que "Personalizado" sea
        # el bastidor elegido: al volver a un tamaño del catálogo sus cambios
        # se ignoran, y el tamaño libre se restaura al reiniciar el diseño.
        if not self._hoop.get("custom"):
            return
        if not self._validate_custom_size():
            # Fuera de rango se queda el último tamaño bueno, el que ya está
            # aplicado en el aro y en la vista previa.
            self._refresh_steps()
            return
        width_mm = self._custom_width.value() * 10
        height_mm = self._custom_height.value() * 10
        if (
            width_mm == self._custom_hoop["width_mm"]
            and height_mm == self._custom_hoop["height_mm"]
        ):
            return
        hoops.set_custom_size(self._custom_hoop, width_mm, height_mm)
        self._custom_card.set_ratio(
            (width_mm * height_mm / self._hoop_area_max) ** 0.5
        )
        self._custom_card.refresh()
        self._apply_hoop(self._custom_hoop)

    # ------------------------------------------------------------------
    # Generación de puntadas
    # ------------------------------------------------------------------

    def generate_stitches(self) -> None:
        if self._processed_result is None or self._stitch_thread is not None:
            return
        self._stitch_button.setEnabled(False)
        self._density_button.setEnabled(False)
        self._export_button.setEnabled(False)
        self._status_label.setText("Generando puntadas...")
        self._guide_used = self._guide_choice
        self._stitch_thread = StitchWorker(
            self._processed_image,
            self._processed_result.threads_used,
            str(self._density_combo.currentData()),
            background_rgb=self._background_rgb,
            background_mask=self._background_mask,
            width_mm=self._hoop["width_mm"],
            height_mm=self._hoop["height_mm"],
            guide=self._guide_used,
            parent=self,
        )
        self._stitch_thread.progress.connect(
            lambda value: self._status_label.setText(f"Generando puntadas... {value}%")
        )
        self._stitch_thread.done.connect(self._on_stitch_done)
        self._stitch_thread.failed.connect(self._on_stitch_failed)
        self._stitch_thread.finished.connect(self._on_stitch_finished)
        self._stitch_thread.start()

    def _on_stitch_done(self, result: tuple) -> None:
        pattern, zones, per_1000 = result
        self._stitch_pattern = pattern
        self._busy_zones = zones

        # Una sola pregunta por diseño, y solo si las paradas son tantas que se
        # notan. Si el usuario cambia de opinión se regenera, pero una vez: el
        # guía usado queda guardado para no entrar en bucle.
        if not self._guide_asked and per_1000 >= GUIDE_ASK_MIN_PER_1000:
            self._guide_asked = True
            settings = QSettings()
            remembered = settings.value(GUIDE_SETTING_KEY)
            dialog = ZonesDialog(zones, per_1000, remembered, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._guide_choice = dialog.choice
                if dialog.remember:
                    settings.setValue(GUIDE_SETTING_KEY, dialog.choice)
            if self._guide_choice != self._guide_used:
                self.generate_stitches()
                return

        preview = pattern.preview
        if preview is not None and zones:
            preview = draw_zone_boxes(
                preview, zones, stitch_generator.PREVIEW_SCALE_PX_PER_MM
            )
        if preview is not None:
            self._pixmap = pil_to_qpixmap(preview)

        self._status_label.setText("Bordado listo para guardar")
        self._stitch_button.setEnabled(True)
        self._density_button.setEnabled(True)
        self._export_button.setEnabled(True)
        self._refresh_preview()
        self._refresh_steps()
        self._go_to_step(3)

    def _on_stitch_failed(self, message: str) -> None:
        self._status_label.setText("Error al generar puntadas")
        QMessageBox.warning(self, "Error", f"No se pudo generar el bordado:\n{message}")

    def _on_stitch_finished(self) -> None:
        self._stitch_thread = None
        self._refresh_steps()

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def export_image(self) -> None:
        if self._stitch_pattern is None or self._export_thread is not None:
            return

        directory = Path(self._selected_save_dir)
        if not directory.is_dir():
            QMessageBox.warning(
                self,
                "No se pudo guardar",
                f"La carpeta {self._selected_save_dir} ya no existe.\n"
                "Elige otro destino.",
            )
            return

        extension = self._export_format.lower()
        name = f"{self._design_name}.{extension}"
        out_path = str(directory / name)

        self._export_button.setEnabled(False)
        self._export_status_label.setText("Guardando...")
        self._remember_save_dir(str(directory))

        self._export_thread = ExportWorker(
            self._stitch_pattern.stitches,
            self._processed_result.threads_used,
            self._export_format,
            self._design_name,
            out_path,
            self,
        )
        self._export_thread.done.connect(self._on_export_done)
        self._export_thread.failed.connect(self._on_export_failed)
        self._export_thread.finished.connect(self._on_export_finished)
        self._export_thread.start()

    def _on_export_done(self, path: str) -> None:
        self._export_status_label.setText(f"Guardado en {path}")
        self._show_export_confirmation(path)
        self._maybe_show_donation()

    def _on_export_failed(self, message: str) -> None:
        self._export_status_label.setText("No se pudo guardar")
        QMessageBox.warning(self, "Error", f"No se pudo guardar el archivo:\n{message}")

    def _on_export_finished(self) -> None:
        self._export_thread = None
        self._export_button.setEnabled(self._stitch_pattern is not None)
        self._refresh_steps()

    def _show_export_confirmation(self, path: str) -> None:
        """Confirmación final con tres salidas: abrir la carpeta de destino,
        quedarse en la pantalla final o empezar otra matriz desde cero."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Bordado guardado")
        dialog.setMinimumWidth(460)
        layout = QVBoxLayout(dialog)

        title = QLabel("<b>Listo para bordar</b>")
        title.setStyleSheet("font-size: 14px;")
        layout.addWidget(title)

        detail = QLabel(
            f"Archivo: <b>{Path(path).name}</b><br>"
            f"Carpeta: {Path(path).parent}"
        )
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(detail)

        buttons = QDialogButtonBox()
        open_button = buttons.addButton(
            "Abrir carpeta", QDialogButtonBox.ButtonRole.ActionRole
        )
        done_button = buttons.addButton(
            "Listo", QDialogButtonBox.ButtonRole.AcceptRole
        )
        again_button = buttons.addButton(
            "Generar otra matriz", QDialogButtonBox.ButtonRole.ActionRole
        )

        def _open_folder() -> None:
            _open_folder_in_explorer(str(Path(path).parent))

        def _start_again() -> None:
            dialog.accept()
            self._reset_flow()

        open_button.clicked.connect(_open_folder)
        done_button.clicked.connect(dialog.accept)
        again_button.clicked.connect(_start_again)

        layout.addWidget(buttons)
        dialog.exec()

    def _maybe_show_donation(self) -> None:
        """Decide si toca mostrar el mensaje de donación tras un guardado.

        Se cuenta un flujo completado por cada exportación exitosa. El mensaje
        aparece 1 de cada `DONATION_EVERY_N` veces y nunca en la primera, a
        menos que el usuario haya pedido no volver a verlo.
        """
        settings = _donation_settings()
        if settings.value(DONATION_SUPPRESS_KEY, False, type=bool):
            return
        count = int(settings.value(DONATION_COUNT_KEY, 0, type=int) or 0) + 1
        settings.setValue(DONATION_COUNT_KEY, count)
        if count % DONATION_EVERY_N != 0:
            return
        self._show_donation_ask()

    def _show_donation_ask(self) -> None:
        """Primera ventana: mensaje breve + 'Donar' / 'No, gracias' + casilla."""
        dialog = QDialog(self)
        dialog.setWindowTitle("NeithLoom es gratis")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        message = QLabel(
            "<b>NeithLoom es y será siempre gratis.</b><br><br>"
            "Hacer bordados no tiene costo para ti, pero si el programa te "
            "resulta útil, un aporte voluntario nos ayuda a seguir mejorándolo. "
            "Es 100% opcional: puedes cerrar esta ventana y seguir igual."
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        suppress = QCheckBox("No volver a mostrar este mensaje")
        layout.addWidget(suppress)

        buttons = QDialogButtonBox()
        donate_button = buttons.addButton(
            "Donar", QDialogButtonBox.ButtonRole.AcceptRole
        )
        no_button = buttons.addButton(
            "No, gracias", QDialogButtonBox.ButtonRole.RejectRole
        )
        donate_button.clicked.connect(dialog.accept)
        no_button.clicked.connect(dialog.reject)
        layout.addWidget(buttons)

        result = dialog.exec()
        if suppress.isChecked():
            _donation_settings().setValue(DONATION_SUPPRESS_KEY, True)
        if result == QDialog.DialogCode.Accepted:
            self._show_donation_pay()

    def _show_donation_pay(self) -> None:
        """Segunda ventana: código QR fijo + copiar link + cerrar."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Donar")
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)

        if QR_IMAGE_PATH.is_file():
            pixmap = QPixmap(str(QR_IMAGE_PATH))
            pixmap = pixmap.scaled(
                240,
                240,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            qr_label = QLabel()
            qr_label.setPixmap(pixmap)
            qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(qr_label)

        hint = QLabel(
            "Escanea el código QR y escribe el monto que quieras aportar."
        )
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        copy_status = QLabel("")
        copy_status.setStyleSheet("color: gray;")
        copy_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copy_status)

        def _copy_link() -> None:
            QGuiApplication.clipboard().setText(PAYMENT_LINK)
            copy_status.setText("Link copiado")

        buttons = QDialogButtonBox()
        copy_button = buttons.addButton(
            "Copiar link de pago", QDialogButtonBox.ButtonRole.ActionRole
        )
        close_button = buttons.addButton(
            "Cerrar", QDialogButtonBox.ButtonRole.AcceptRole
        )
        copy_button.clicked.connect(_copy_link)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def _reset_flow(self) -> None:
        """Vuelve al paso 1 con todo limpio, como si la app acabara de abrir.

        Se restauran también bastidor, densidad y formato: el botón pide
        empezar de cero, no solo quitar la imagen de la vista previa. La carpeta
        de destino NO se toca: viene de `gui/last_save_dir` y sobrevive al
        cierre de la app.
        """
        if (
            self._process_thread is not None
            or self._stitch_thread is not None
            or self._export_thread is not None
        ):
            return

        self._image = None
        self._pixmap = None
        self._current_path = None
        self._processed_image = None
        self._processed_result = None
        self._stitch_pattern = None
        self._background_rgb = None
        self._background_mask = None
        self._busy_zones = []
        self._saved_pixmap = None
        self._guide_asked = False
        self._exit_pick_mode()

        # El tamaño personalizado vuelve a sus valores iniciales y se ocultan
        # sus campos: `_apply_hoop` es quien decide la visibilidad, y se pone
        # el bastidor del catálogo antes de tocar los campos para que su
        # `valueChanged` se descarte en vez de reaplicar un tamaño libre.
        self._apply_hoop(hoops.HOOPS[hoops.DEFAULT_HOOP_INDEX])
        hoops.set_custom_size(self._custom_hoop, *self._custom_default_size)
        self._custom_width.setValue(self._custom_default_size[0] / 10)
        self._custom_height.setValue(self._custom_default_size[1] / 10)
        self._custom_card.set_ratio(
            (
                self._custom_default_size[0]
                * self._custom_default_size[1]
                / self._hoop_area_max
            )
            ** 0.5
        )
        self._custom_card.refresh()
        # El combo de densidad no dispara nada: solo se lee al generar, así que
        # volver al primer índice no regenera nada.
        self._density_combo.setCurrentIndex(0)
        self._format_combo.setCurrentIndex(0)
        self._export_format = "PES"
        self._design_name = "NeithLoom"
        self._design_name_edit.setText(self._design_name)

        self._refresh_thread_swatches([])
        self._export_status_label.setText("")
        self._refresh_preview()
        self._status_label.setText("Elige una imagen para empezar de nuevo")
        self._go_to_step(0)

    def closeEvent(self, event) -> None:
        for worker in (
            self._gallery_thread,
            self._process_thread,
            self._stitch_thread,
            self._export_thread,
        ):
            if worker is not None:
                worker.requestInterruption()
                worker.wait(300)
        super().closeEvent(event)