"""Exportador de puntadas al formato DST (Tajima).

Sigue la codificación de registros de pyembroidery (bit-encoding ±1/±3/±9/
±27/±81 por eje). Las coordenadas de entrada están en milímetros; el formato
usa décimas de milímetro, por lo que se multiplican por 10.
"""

from pathlib import Path

from core.stitch_generator import MOVE_SENTINELS, group_stitches_by_color

MAX_STITCH_DISTANCE = 121  # décimas de mm por registro (12.1 mm)
DST_HEADER_SIZE = 512

# Bandera de comando
STITCH = 0
JUMP = 1
COLOR_CHANGE = 2
END = 3


def _bit(n):
    return 1 << n


def encode_record(dx, dy, flags):
    """Codifica un registro DST de 3 bytes (deltas en décimas de mm)."""
    y = -dy  # el eje Y del DST crece hacia arriba
    b0 = 0
    b1 = 0
    b2 = 0
    if flags == JUMP:
        b2 += _bit(7)  # jumpstitch
    if flags == STITCH or flags == JUMP:
        b2 += _bit(0)
        b2 += _bit(1)
        x = dx
        if x > 40:
            b2 += _bit(2)
            x -= 81
        if x < -40:
            b2 += _bit(3)
            x += 81
        if x > 13:
            b1 += _bit(2)
            x -= 27
        if x < -13:
            b1 += _bit(3)
            x += 27
        if x > 4:
            b0 += _bit(2)
            x -= 9
        if x < -4:
            b0 += _bit(3)
            x += 9
        if x > 1:
            b1 += _bit(0)
            x -= 3
        if x < -1:
            b1 += _bit(1)
            x += 3
        if x > 0:
            b0 += _bit(0)
            x -= 1
        if x < 0:
            b0 += _bit(1)
            x += 1
        if x != 0:
            raise ValueError("El desplazamiento X excede el máximo permitido.")
        if y > 40:
            b2 += _bit(5)
            y -= 81
        if y < -40:
            b2 += _bit(4)
            y += 81
        if y > 13:
            b1 += _bit(5)
            y -= 27
        if y < -13:
            b1 += _bit(4)
            y += 27
        if y > 4:
            b0 += _bit(5)
            y -= 9
        if y < -4:
            b0 += _bit(4)
            y += 9
        if y > 1:
            b1 += _bit(7)
            y -= 3
        if y < -1:
            b1 += _bit(6)
            y += 3
        if y > 0:
            b0 += _bit(7)
            y -= 1
        if y < 0:
            b0 += _bit(6)
            y += 1
        if y != 0:
            raise ValueError("El desplazamiento Y excede el máximo permitido.")
    elif flags == COLOR_CHANGE:
        b2 = 0b11000011
    elif flags == END:
        b2 = 0b11110011
    return bytes(bytearray([b0, b1, b2]))


def _group_by_color(stitches):
    """Agrupa las puntadas por código de color respetando el primer orden.

    Los movimientos con aguja arriba (JUMP_SENTINEL / TRIM_SENTINEL) no son
    puntadas reales y se descartan: solo interesan para la secuencia de
    registros.

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
    """Divide un movimiento largo en segmentos de <= MAX_STITCH_DISTANCE.

    Reparte de forma exacta (la suma de los segmentos es (dx, dy)) usando
    pasos enteros progresivos.
    """
    dist = max(abs(dx), abs(dy))
    if dist <= MAX_STITCH_DISTANCE:
        return [(dx, dy)]
    steps = -(-dist // MAX_STITCH_DISTANCE)  # ceil(dist/MAX) >= 2
    return [
        (
            dx * i // steps - dx * (i - 1) // steps,
            dy * i // steps - dy * (i - 1) // steps,
        )
        for i in range(1, steps + 1)
    ]


def export_dst(stitches, name="NeithLoom", out_path=None):
    """Escribe un archivo DST y devuelve la ruta generada.

    `stitches` es la lista de tuplas (x_mm, y_mm, código) EN ORDEN. El flujo
    se recorre tal cual: los marcadores de aguja arriba (JUMP_SENTINEL y
    TRIM_SENTINEL, indistinguibles en DST) se
    convierten en registros JUMP en su posición exacta, se insertan cambios
    de color (stop codes) al cambiar el código y saltos para movimientos
    largos.

    Antes de escribir, el flujo se reagrupa con `group_stitches_by_color`
    (mismo criterio de primer orden de aparición que las secciones CSewSeg
    del PES): cada código de hilo queda en un único bloque contiguo, de modo
    que la secuencia de stop codes del DST coincide SIEMPRE con el orden de
    secciones por color del PES, venga el flujo agrupado o no.
    """
    stitches = group_stitches_by_color(stitches)
    groups = _group_by_color(stitches)
    if out_path is None:
        out_path = Path.cwd() / "NeithLoom.dst"
    out_path = Path(out_path)

    records = []
    color_changes = 0
    stitches_count = 0

    def add_record(dx, dy, flags):
        records.append(encode_record(dx, dy, flags))

    px = 0
    py = 0
    current_code = None
    for x_mm, y_mm, code in stitches:
        x = int(round(x_mm * 10.0))
        y = int(round(y_mm * 10.0))
        if code in MOVE_SENTINELS:
            dx = x - px
            dy = y - py
            for sx, sy in _split_move(dx, dy):
                px += sx
                py += sy
                add_record(sx, sy, JUMP)
            continue
        if code != current_code:
            if current_code is not None:
                add_record(0, 0, COLOR_CHANGE)
                color_changes += 1
            current_code = code
        dx = x - px
        dy = y - py
        if max(abs(dx), abs(dy)) > MAX_STITCH_DISTANCE:
            for sx, sy in _split_move(dx, dy):
                px += sx
                py += sy
                add_record(sx, sy, JUMP)
            add_record(0, 0, STITCH)  # puntada de agarre al llegar
        else:
            add_record(dx, dy, STITCH)
            px = x
            py = y
        stitches_count += 1
    if records:
        add_record(0, 0, STITCH)  # puntada de cierre
        stitches_count += 1
    add_record(0, 0, END)

    # Límites en décimas de mm (espacio "decodificado", como en pyembroidery)
    if groups:
        xs = [p[0] for _, pts in groups for p in pts]
        ys = [p[1] for _, pts in groups for p in pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    else:
        minx = miny = maxx = maxy = 0

    last_pt = None
    if groups:
        last_pt = groups[-1][1][-1]
    last_x = last_pt[0] if last_pt else 0
    last_y = -last_pt[1] if last_pt else 0  # DST guarda la Y negada

    chunks = [
        b"LA:%-16s\r" % str(name)[:16].encode("ascii", "ignore"),
        b"ST:%7d\r" % stitches_count,
        b"CO:%3d\r" % color_changes,
        b"+X:%5d\r" % abs(maxx),
        b"-X:%5d\r" % abs(minx),
        b"+Y:%5d\r" % abs(maxy),
        b"-Y:%5d\r" % abs(miny),
        (b"AX:+%5d\r" % last_x) if last_x >= 0 else (b"AX:-%5d\r" % abs(last_x)),
        (b"AY:+%5d\r" % last_y) if last_y >= 0 else (b"AY:-%5d\r" % abs(last_y)),
        b"MX:+%5d\r" % 0,
        b"MY:+%5d\r" % 0,
        b"PD:%6s\r" % b"******",
        b"\x1a",
    ]
    header = b"".join(chunks)
    header = header.ljust(DST_HEADER_SIZE, b"\x20")

    with open(out_path, "wb") as fh:
        fh.write(header)
        for record in records:
            fh.write(record)

    return str(out_path)