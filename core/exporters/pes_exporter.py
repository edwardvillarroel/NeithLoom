"""Exportador de puntadas al formato PES v1 (Brother).

Replica la estructura del escritor v1 de pyembroidery: cabecera binaria con
bloques CEmbOne/CSewSeg y un bloque PEC (colores + puntadas comprimidas).
Las coordenadas de entrada están en milímetros; el formato usa décimas de
milímetro, por lo que se multiplican por 10.
"""

import math
import struct
from pathlib import Path

from core.exporters.pec_palette import build_unique_palette
from core.stitch_generator import MOVE_SENTINELS, TRIM_SENTINEL

PES_VERSION_1_SIGNATURE = b"#PES0001"
EMB_ONE = "CEmbOne"
EMB_SEG = "CSewSeg"

PEC_ICON_WIDTH = 48
PEC_ICON_HEIGHT = 38

MASK_07_BIT = 0b01111111
JUMP_CODE = 0b00010000
TRIM_CODE = 0b00100000
FLAG_LONG = 0b10000000
MAX_JUMP_DISTANCE = 2047  # valor largo de 12 bits con signo (±20.47 mm)
MAX_STITCH_DISTANCE = 127  # más allá de esto se trata como salto (12.7 mm)

# Distancia mínima (mm) de un salto para considerarlo un desplazamiento entre
# zonas separadas y cortar el hilo (TRIM) antes de reposicionar. Saltos más
# cortos se tratan como reposicionamientos pequeños que no requieren corte.
TRIM_MIN_JUMP_DISTANCE_MM = 6.0

STITCH = 0
JUMP = 1
COLOR_CHANGE = 2
END = 3
TRIM = 4

# Área máxima de bordado de la Brother PE830DL (mm).
# El diseño exportado NUNCA puede superar estos límites.
MAX_STITCH_AREA_WIDTH_MM = 130.0
MAX_STITCH_AREA_HEIGHT_MM = 180.0

# Zona de seguridad: en lugar de quedar justo en el límite de la máquina,
# el diseño se escala para caber en este área algo menor (evita rechazos
# por tolerancias de la máquina o redondeos de las puntadas).
SAFE_AREA_WIDTH_MM = 125.0
SAFE_AREA_HEIGHT_MM = 175.0


def _write_int8(fh, value):
    fh.write(struct.pack("<b", value))


def _write_int16le(fh, value):
    v = bytes(bytearray([(value >> 0) & 0xFF, (value >> 8) & 0xFF]))
    fh.write(v)


def _write_int24le(fh, value):
    data = value & 0xFFFFFF
    fh.write(bytes(bytearray([data & 0xFF, (data >> 8) & 0xFF, (data >> 16) & 0xFF])))


def _write_int32le(fh, value):
    fh.write(struct.pack("<i", value))


def _write_float32le(fh, value):
    fh.write(struct.pack("<f", float(value)))


def _write_pes_string_8(fh, text):
    if text is None:
        _write_int8(fh, 0)
        return
    if not isinstance(text, bytes):
        text = text.encode("utf-8")
    if len(text) > 255:
        text = text[:255]
    _write_int8(fh, len(text))
    fh.write(text)


def _write_pes_string_16(fh, text):
    if text is None:
        _write_int16le(fh, 0)
        return
    if not isinstance(text, bytes):
        text = text.encode("utf-8")
    _write_int16le(fh, len(text))
    fh.write(text)


def _group_by_color(stitches):
    """Agrupa puntadas (mm) por código en el primer orden de aparición.

    Los movimientos con aguja arriba (JUMP_SENTINEL / TRIM_SENTINEL) no son
    puntadas reales y se descartan: solo interesan para la secuencia de
    comandos.

    Devuelve lista de (código, lista de (x, y) en décimas de mm).
    """
    order = []
    groups = {}
    for x_mm, y_mm, code in stitches:
        if code in MOVE_SENTINELS:
            continue
        x = int(round(x_mm * 10.0))
        y = int(round(y_mm * 10.0))
        if code not in groups:
            groups[code] = []
            order.append(code)
        groups[code].append((x, y))
    return [(code, groups[code]) for code in order]


def _split_move(dx, dy):
    """Divide un desplazamiento largo en tramos <= MAX_JUMP_DISTANCE.

    El valor largo del PEC admite ±2047; el salto se divide en pasos enteros
    que sumen exactamente (dx, dy).
    """
    dist = max(abs(dx), abs(dy))
    if dist <= MAX_JUMP_DISTANCE:
        return [(dx, dy)]
    steps = -(-dist // MAX_JUMP_DISTANCE)  # ceil(dist/MAX) >= 2
    return [
        (
            dx * i // steps - dx * (i - 1) // steps,
            dy * i // steps - dy * (i - 1) // steps,
        )
        for i in range(1, steps + 1)
    ]


def _is_separated_zone(dx, dy):
    """¿El desplazamiento (dx, dy) en décimas de mm describe una zona separada?

    Solo los saltos por distancia entre zonas no contiguas (más largos que el
    umbral TRIM_MIN_JUMP_DISTANCE_MM) reciben un corte de hilo; los
    reposicionamientos pequeños se cosen sin cortar para no malgastar hilo ni
    tiempo de máquina. Los saltos marcados con TRIM_SENTINEL (cruce de hueco o
    fondo real detectado por el enrutado) se cortan siempre, sin importar la
    distancia.
    """
    return math.hypot(dx, dy) > TRIM_MIN_JUMP_DISTANCE_MM * 10.0


def build_commands(stitches):
    """Genera la secuencia de comandos (absolutos en décimas de mm).

    Recorre el flujo (mm) EN ORDEN, preservando la posición de los saltos
    (JUMP_SENTINEL / TRIM_SENTINEL) entre tramos de un mismo color: produce
    comandos JUMP ahí donde el patrón marcó aguja arriba, un comando TRIM antes
    de cada salto real entre zonas separadas y de cada TRIM_SENTINEL (evita
    hilo suelto bajo la tela y garantiza el corte al cruzar huecos), cambios de
    color al cambiar el código y puntadas de agarre tras un salto largo. Los
    desplazamientos largos se dividen en saltos.

    Comandos: (tipo, x, y).
    """
    commands = []
    px = 0
    py = 0
    current_code = None
    for x_mm, y_mm, code in stitches:
        x = int(round(x_mm * 10.0))
        y = int(round(y_mm * 10.0))
        if code in MOVE_SENTINELS:
            dx = x - px
            dy = y - py
            # Zona separada (no mero reposicionamiento) o salto forzado por la
            # máscara (TRIM_SENTINEL): cortar el hilo antes de saltar, excepto
            # en el salto inicial desde la aguja en reposo (aún no hay nada
            # cosido que cortar).
            force_trim = code == TRIM_SENTINEL
            if commands and (force_trim or _is_separated_zone(dx, dy)):
                commands.append((TRIM, px, py))
            for sx, sy in _split_move(dx, dy):
                px += sx
                py += sy
                commands.append((JUMP, px, py))
            continue
        if code != current_code:
            if current_code is not None:
                commands.append((COLOR_CHANGE, px, py))
            current_code = code
        dx = x - px
        dy = y - py
        if max(abs(dx), abs(dy)) > MAX_STITCH_DISTANCE:
            # Puntada real demasiado lejos del punto anterior sin salto
            # marcado: se corta el hilo y se reposiciona con saltos antes de
            # coser la puntada de agarre (mismo criterio que JUMP_SENTINEL).
            if _is_separated_zone(dx, dy):
                commands.append((TRIM, px, py))
            for sx, sy in _split_move(dx, dy):
                px += sx
                py += sy
                commands.append((JUMP, px, py))
            commands.append((STITCH, x, y))  # puntada de agarre
        else:
            commands.append((STITCH, x, y))
        px = x
        py = y
    commands.append((STITCH, px, py))
    commands.append((END, px, py))
    return commands


def _write_value(fh, value, long=False, flag=0):
    data = []
    if not long and -64 < value < 63:
        data.append(value & MASK_07_BIT)
    else:
        value &= 0b0000111111111111
        value |= 0b1000000000000000
        value |= flag << 8
        data.append((value >> 8) & 0xFF)
        data.append(value & 0xFF)
    fh.write(bytes(bytearray(data)))


def _write_jump(fh, dx, dy):
    _write_value(fh, dx, long=True, flag=JUMP_CODE)
    _write_value(fh, dy, long=True, flag=JUMP_CODE)


def _write_stitch(fh, dx, dy):
    _write_value(fh, dx, long=False)
    _write_value(fh, dy, long=False)


def _write_pec_encode(fh, commands):
    color_two = True
    jumping = True
    xx = 0
    yy = 0
    for command in commands:
        kind, x, y = command
        dx = int(round(x - xx))
        dy = int(round(y - yy))
        xx += dx
        yy += dy
        if kind == STITCH:
            if jumping:
                if dx != 0 and dy != 0:
                    _write_stitch(fh, 0, 0)
                jumping = False
            _write_stitch(fh, dx, dy)
        elif kind == JUMP:
            jumping = True
            _write_jump(fh, dx, dy)
        elif kind == TRIM:
            # Corte de hilo en el mismo sitio, con la aguja arriba, justo
            # antes del salto de reposicionamiento. Replica el patrón de bits
            # observado en archivos de digitalizadores externos: un JUMP(0,0)
            # seguido de un TRIM(0,0) codificado con la bandera TRIM_CODE
            # (sin JUMP_CODE), de modo que los decodificadores (p.ej.
            # pyembroidery) lo reconozcan como un comando TRIM.
            _write_jump(fh, 0, 0)
            _write_value(fh, 0, long=True, flag=TRIM_CODE)
            _write_value(fh, 0, long=True, flag=TRIM_CODE)
            jumping = True
        elif kind == COLOR_CHANGE:
            if jumping:
                _write_stitch(fh, 0, 0)
                jumping = False
            fh.write(b"\xfe\xb0")
            fh.write(b"\x02" if color_two else b"\x01")
            color_two = not color_two
        elif kind == END:
            fh.write(b"\xff")
            break


def _write_pec_header(fh, name, groups, color_index_list):
    fh.write(b"LA:%-16s\r" % str(name)[:8].encode("ascii", "ignore"))
    fh.write(b"\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\xFF\x00")
    _write_int8(fh, int(PEC_ICON_WIDTH / 8))  # 6 → stride de bytes
    _write_int8(fh, int(PEC_ICON_HEIGHT))  # 38
    current_thread_count = len(color_index_list)
    if current_thread_count != 0:
        fh.write(b"\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20")
        add_value = current_thread_count - 1
        color_index_list = [add_value] + list(color_index_list)
        fh.write(bytes(bytearray(color_index_list)))
    else:
        fh.write(b"\x20\x20\x20\x20\x64\x20\x00\x20\x00\x20\x20\x20\xFF")
    for _ in range(current_thread_count, 463):
        fh.write(b"\x20")


def _write_pec_block(fh, commands, width, height):
    start = fh.tell()
    fh.write(b"\x00\x00")
    _write_int24le(fh, 0)  # placeholder longitud
    fh.write(b"\x31\xff\xf0")
    _write_int16le(fh, int(round(width)))
    _write_int16le(fh, int(round(height)))
    _write_int16le(fh, 0x1E0)
    _write_int16le(fh, 0x1B0)
    _write_pec_encode(fh, commands)
    block_length = fh.tell() - start
    current = fh.tell()
    fh.seek(start + 2, 0)
    _write_int24le(fh, block_length)
    fh.seek(current, 0)


_PEC_BLANK_GRAPHIC = bytes.fromhex(
    "000000000000f0ffffffff0f080000000010040000000020020000000040"
    "020000000040020000000040020000000040020000000040020000000040"
    "020000000040020000000040020000000040020000000040020000000040"
    "020000000040020000000040020000000040020000000040020000000040"
    "020000000040020000000040020000000040020000000040020000000040"
    "020000000040020000000040020000000040020000000040020000000040"
    "020000000040020000000040020000000040020000000040040000000020"
    "080000000010f0ffffffff0f000000000000"
)


def _graphic_mark_bit(graphic, x, y, stride):
    width = PEC_ICON_WIDTH
    height = PEC_ICON_HEIGHT
    if 0 <= x < width and 0 <= y < height:
        graphic[y * stride + x // 8] |= 1 << (x % 8)


def _draw_points_into(graphic, extends, points, stride, buffer):
    """Dibuja las puntadas (décimas de mm) como bits en el bitmap 48x38."""
    left, top, right, bottom = extends
    width = stride * 8
    height = PEC_ICON_HEIGHT
    diagram_width = right - left if right != left else 1
    diagram_height = bottom - top if bottom != top else 1
    scale = min(
        (width - buffer) / float(diagram_width),
        (height - buffer) / float(diagram_height),
    )
    center_x = (right + left) / 2.0
    center_y = (bottom + top) / 2.0
    translate_x = -center_x * scale + width / 2.0
    translate_y = -center_y * scale + height / 2.0
    for x, y in points:
        _graphic_mark_bit(
            graphic,
            int(math.floor(x * scale + translate_x)),
            int(math.floor(y * scale + translate_y)),
            stride,
        )


def _write_pec_graphics(fh, groups, extends):
    """Miniaturas PEC (48x38) que la máquina muestra en su lista de diseños.

    Replica el layout de pyembroidery: un bitmap con todas las puntadas y un
    bitmap por color, sobre un marco rombo estándar. Contar como píxel
    marcado las puntadas reales hace que el diseño sea reconocible en la
    pantalla del aro sin modificar la geometría ni las puntadas exportadas.
    """
    stride = int(PEC_ICON_WIDTH / 8)
    all_points = [p for _, points in groups for p in points]
    graphic = bytearray(_PEC_BLANK_GRAPHIC)
    _draw_points_into(graphic, extends, all_points, stride, 4)
    fh.write(bytes(graphic))
    for _, points in groups:
        graphic = bytearray(_PEC_BLANK_GRAPHIC)
        _draw_points_into(graphic, extends, points, stride, 5)
        fh.write(bytes(graphic))


def _write_pes_sewsegheader(fh, left, top, right, bottom):
    width = right - left
    height = bottom - top
    hoop_height = 1800
    hoop_width = 1300
    for _ in range(8):
        _write_int16le(fh, 0)
    trans_x = float(350) + hoop_width / 2 - width / 2
    trans_y = float(100) + height + hoop_height / 2 - height / 2
    for value in (1.0, 0.0, 0.0, 1.0, trans_x, trans_y):
        _write_float32le(fh, value)
    _write_int16le(fh, 1)
    _write_int16le(fh, 0)
    _write_int16le(fh, 0)
    _write_int16le(fh, int(width))
    _write_int16le(fh, int(height))
    fh.write(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    placeholder = fh.tell()
    _write_int16le(fh, 0)  # placeholder nº de secciones
    return placeholder


def _write_pec(fh, name, groups, color_index_list, commands, extends):
    """Escribe el bloque PEC completo tal y como lo lee Brother."""
    width = extends[2] - extends[0]
    height = extends[3] - extends[1]
    _write_pec_header(fh, name, groups, color_index_list)
    _write_pec_block(fh, commands, width, height)
    _write_pec_graphics(fh, groups, extends)


def export_pes(stitches, threads_used, name="NeithLoom", out_path=None):
    """Escribe un archivo PES v1 y devuelve la ruta generada.

    Sistema de coordenadas para la Brother PE830DL (área máxima de bordado
    130x180 mm) que interpreta (0,0) como una esquina del área:

    1. Si el diseño no cabe en el área de seguridad (125x175 mm) se escala
       de forma proporcional para que quepa dentro de ella (nunca supera
       130x180 mm).
    2. El diseño se traslada a su esquina mínima: todas las coordenadas son
       positivas y la esquina del diseño queda en (0,0). No se centra.
    3. El ancho y alto reales del diseño (ancho = max_x - min_x, alto =
       max_y - min_y, ya escalado y trasladado) se escriben en la cabecera
       en décimas de milímetro (x10), coherentes con las puntadas escritas.
    4. Verificación interna: ninguna puntada del flujo PEC queda fuera de la
       zona de seguridad (por tanto nunca del área 130x180 mm).
    Cuando hay puntadas, el archivo codifica las coordenadas trasladadas; al
    no haber puntadas se escribe un diseño vacío.
    """
    # 1) Escala al área de seguridad (proporcional, sin deformar)
    real = [s for s in stitches if s[2] not in MOVE_SENTINELS]
    if real:
        xs = [s[0] for s in real]
        ys = [s[1] for s in real]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        scale = min(
            1.0,
            SAFE_AREA_WIDTH_MM / width if width > 0 else float("inf"),
            SAFE_AREA_HEIGHT_MM / height if height > 0 else float("inf"),
        )
        scaled = [
            (x_mm * scale, y_mm * scale, code)
            for x_mm, y_mm, code in stitches
        ]

        # 2) Trasladar a la esquina mínima: la esquina del diseño queda
        #    en (0,0) y todas las coordenadas son positivas. La máquina
        #    Brother PE830DL interpreta (0,0) como una esquina del área de
        #    bordado, así que el diseño se coloca a partir de ahí.
        sx = [s[0] for s in scaled if s[2] not in MOVE_SENTINELS]
        sy = [s[1] for s in scaled if s[2] not in MOVE_SENTINELS]
        min_x = min(sx)
        min_y = min(sy)
        translated = [
            (x_mm - min_x, y_mm - min_y, code)
            for x_mm, y_mm, code in scaled
        ]
    else:
        translated = stitches

    groups = []
    commands = build_commands(translated)
    extends = (0, 0, 0, 0)
    for _ in range(4):
        groups = _group_by_color(translated)
        if not groups:
            break
        commands = build_commands(translated)
        tx = [c[1] for c in commands]
        ty = [c[2] for c in commands]
        min_tx = min(tx)
        min_ty = min(ty)
        # La esquina mínima del diseño debe quedar en (0,0). El redondeo de
        # los saltos puede dejar el flujo PEC ligeramente negativo; se
        # desplaza lo justo para que la esquina quede exactamente en (0,0).
        if min_tx < 0 or min_ty < 0:
            translated = [
                (x_mm - min_tx / 10.0, y_mm - min_ty / 10.0, code)
                for x_mm, y_mm, code in translated
            ]
            continue
        ext10 = (min(tx), min(ty), max(tx), max(ty))
        w10 = ext10[2] - ext10[0]
        h10 = ext10[3] - ext10[1]
        # 4) Verificación sobre el flujo que realmente se escribe en el.
        #    bloque PEC (0.1mm): nada fuera de la zona de seguridad de
        #    125x175 mm (por tanto nunca del área 130x180 mm de la
        #    máquina). El escalado se hace respecto a la esquina (0,0),
        #    manteniendo la esquina del diseño fija. Si el redondeo de las
        #    puntadas o de los saltos dejara algo fuera, se reduce el
        #    diseño y se vuelve a encajar.
        if w10 > SAFE_AREA_WIDTH_MM * 10 or h10 > SAFE_AREA_HEIGHT_MM * 10:
            shrink = min(
                SAFE_AREA_WIDTH_MM * 10 / w10,
                SAFE_AREA_HEIGHT_MM * 10 / h10,
            )
            translated = [
                (x_mm * shrink, y_mm * shrink, code)
                for x_mm, y_mm, code in translated
            ]
            continue
        extends = ext10
        break

    color_index = build_unique_palette(
        [t for t in threads_used if t.code in {c for c, _ in groups}]
    )
    color_index_list = [
        color_index[code] for code, _ in groups if code in color_index
    ]

    cx = (extends[2] + extends[0]) / 2.0
    cy = (extends[3] + extends[1]) / 2.0
    left = extends[0] - cx
    top = extends[1] - cy
    right = extends[2] - cx
    bottom = extends[3] - cy

    if out_path is None:
        out_path = Path.cwd() / "NeithLoom.pes"
    out_path = Path(out_path)

    with open(out_path, "wb") as fh:
        fh.write(PES_VERSION_1_SIGNATURE)
        placeholder_pec_block = fh.tell()
        _write_int32le(fh, 0)  # placeholder posición bloque PEC

        if not stitches:
            _write_int16le(fh, 0x01)
            _write_int16le(fh, 0x01)
            _write_int16le(fh, 0)
            _write_int16le(fh, 0x0000)
            _write_int16le(fh, 0x0000)
        else:
            _write_int16le(fh, 0x01)  # scale to fit
            _write_int16le(fh, 0x01)  # hoop 130x180
            _write_int16le(fh, 1)  # distinct block objects
            _write_int16le(fh, 0xFFFF)
            _write_int16le(fh, 0x0000)

            _write_pes_string_16(fh, EMB_ONE)
            placeholder_sections = _write_pes_sewsegheader(
                fh, left, top, right, bottom
            )
            _write_int16le(fh, 0xFFFF)
            _write_int16le(fh, 0x0000)

            _write_pes_string_16(fh, EMB_SEG)
            section = 0
            colorlog = []
            previous_code = -1
            adjust_x = left + cx
            adjust_y = bottom + cy
            first = True
            for code, points in groups:
                index = color_index.get(code, 0)
                if not first:
                    _write_int16le(fh, 0x8003)  # fin de sección anterior
                first = False
                _write_int16le(fh, 0)  # flag = 0 (stitch run)
                _write_int16le(fh, index)  # código de color
                _write_int16le(fh, len(points))
                for x, y in points:
                    _write_int16le(fh, int(x - adjust_x))
                    _write_int16le(fh, int(y - adjust_y))
                if previous_code != index:
                    colorlog.append([section, index])
                    previous_code = index
                section += 1

            _write_int16le(fh, len(colorlog))
            for log_item in colorlog:
                _write_int16le(fh, log_item[0])
                _write_int16le(fh, log_item[1])

            current = fh.tell()
            fh.seek(placeholder_sections, 0)
            _write_int16le(fh, section)
            fh.seek(current, 0)

            _write_int16le(fh, 0x0000)
            _write_int16le(fh, 0x0000)

        pec_position = fh.tell()
        fh.seek(placeholder_pec_block, 0)
        _write_int32le(fh, pec_position)
        fh.seek(pec_position, 0)

        _write_pec(
            fh,
            name,
            groups,
            color_index_list,
            commands,
            extends,
        )

    return str(out_path)