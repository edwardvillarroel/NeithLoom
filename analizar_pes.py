#!/usr/bin/env python3
"""Analiza archivos .PES y construye un "perfil objetivo" de buena matriz.

Compara las matrices externas de `referencias/` (digitalizadores comerciales,
considered buenas) con las exportadas por esta aplicacion en `generadas/`,
y devuelve un perfil objetivo con promedios y rangos de las referencias.

Solo libreria estandar (struct / pathlib / csv / statistics / argparse).

Uso:
    python analizar_pes.py
    python analizar_pes.py --referencias R --generadas G --salida S
    python analizar_pes.py --csv            # ademas vuelca CSV

La decodificacion del bloque PEC replica exactamente la que usan
`core/exporters/pes_exporter.py` y `scripts/verify_pes.py`, de modo que los
conteos de este script son comparables con los de aquel. No depende de
pyembroidery (si se importa, solo sirve como verificacion cruzada opcional).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Flags del bloque PEC (identicos a los de pyembroidery/PecReader.py).
JUMP_CODE = 0x10
TRIM_CODE = 0x20
FLAG_LONG = 0x80

# Cabeceras v4/v5+: material para el nombre interno del diseno.
_METADATA_VERSIONS = {"#PES0040", "#PES0050", "#PES0055", "#PES0056",
                     "#PES0060", "#PES0070", "#PES0080", "#PES0090",
                     "#PES0100"}


class ErrorPes(Exception):
    """Archivo ilegible, corrupto o que no es un PES."""


# --------------------------------------------------------------------------
# Lector de bytes con cursor
# --------------------------------------------------------------------------
class Lector:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0

    def i8(self):
        if self.p >= len(self.d):
            return None
        v = self.d[self.p]
        self.p += 1
        return v

    def i16(self):
        if self.p + 2 > len(self.d):
            return None
        v = struct.unpack_from("<H", self.d, self.p)[0]
        self.p += 2
        return v

    def i24(self):
        if self.p + 3 > len(self.d):
            return None
        v = self.d[self.p] | (self.d[self.p + 1] << 8) | (self.d[self.p + 2] << 16)
        self.p += 3
        return v

    def i32(self):
        if self.p + 4 > len(self.d):
            return None
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def raw(self, n: int) -> bytes:
        v = self.d[self.p:self.p + n]
        self.p += n
        return v

    def skip(self, n: int) -> None:
        self.p = min(self.p + n, len(self.d))

    def texto(self, n: int):
        b = self.raw(n)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def pascal(self):
        """String con byte de longitud inicial (formato PES v4+)."""
        n = self.i8()
        if not n:
            return None
        return self.texto(n)


def _signed7(b: int) -> int:
    return -128 + b if b > 63 else b


def _signed12(v: int) -> int:
    v &= 0xFFF
    return -0x1000 + v if v > 0x7FF else v


# --------------------------------------------------------------------------
# Analisis de un .PES
# --------------------------------------------------------------------------
def analizar_pes(ruta: Path) -> dict:
    """Devuelve el diccionario de metricas de un .PES, o lanza ErrorPes."""
    try:
        data = ruta.read_bytes()
    except OSError as exc:
        raise ErrorPes(f"no se pudo leer: {exc}") from exc

    if len(data) < 32:
        raise ErrorPes("archivo demasiado corto para ser un PES")

    l = Lector(data)
    firma = l.texto(8)
    if firma is None:
        raise ErrorPes("la firma inicial no es texto legible: no parece un PES")
    if not firma.startswith("#PES"):
        raise ErrorPes(f"firma invalida: {firma!r}")

    m = {
        "archivo": ruta.name,
        "ruta": str(ruta),
        "bytes": len(data),
        "version": firma,
        "nombre": None,
        "puntadas": 0,
        "saltos": 0,
        "saltos_mov": 0,
        "trims": 0,
        "cambios_color": 0,
        # pyembroidery/verify_pes.py emiten un `move` ADEMAS de cada `trim`, asi
        # que alli JUMP = saltos crudos + trims. Se reporta para poder comparar
        # directamente con scripts/verify_pes.py y los benchmark_*.py.
        "saltos_pe": 0,
        "colores": None,
        "ancho_mm": None,
        "alto_mm": None,
        "declarado_ancho_mm": None,
        "declarado_alto_mm": None,
        "trims_por_1000": None,
        "saltos_por_1000": None,
        "salto_mov_por_1000": None,
        "largo_puntada_mm": None,
        "largo_salto_mm": None,
        "notas": [],
    }

    # --- Posicion del bloque PEC -------------------------------------------
    if firma == "#PEC0001":
        pec_pos = l.p
    else:
        pec_pos = l.i32()
        if pec_pos is None:
            raise ErrorPes("sin puntero al bloque PEC")
        if pec_pos >= len(data):
            m["notas"].append("puntero PEC fuera del archivo; se asume inicio")
            pec_pos = 8
        # Nombre interno desde la cabecera v4/v5+ (metadatos, no obligatorio).
        if firma in _METADATA_VERSIONS:
            nombre = _leer_nombre_metadatos(l)
            if nombre:
                m["nombre"] = nombre

    # --- Bloque PEC ---------------------------------------------------------
    # OJO: el bloque PEC NO empieza con un magic "#PEC0001"; arranca directamente
    # en "LA:" y el puntero de la cabecera PES apunta ahi. (Lo mismo asume
    # pyembroidery: PesReader hace seek(pec_block_position) y PecReader.read_pec
    # empieza en seek(3) sin haber leido ninguna firma.)
    l.p = pec_pos
    if l.raw(3) != b"LA:":
        m["notas"].append("el bloque PEC no empieza con 'LA:'")

    etiqueta = l.texto(16)        # nombre interno (16 bytes, sin prefijo)
    if etiqueta:
        etiqueta = etiqueta.strip().strip("\x00").strip()
    if etiqueta:
        m["nombre"] = m["nombre"] or etiqueta

    l.skip(0x0F)
    stride = l.i8()               # ancho del bitmap (bytes por fila)
    alto_icono = l.i8()
    if stride is None or alto_icono is None:
        raise ErrorPes("cabecera PEC truncada")
    m["stride"] = stride
    m["icono_alto"] = alto_icono

    l.skip(0x0C)
    cambios_pec = l.i8()
    if cambios_pec is None:
        raise ErrorPes("cabecera PEC truncada (cuenta de colores)")
    m["colores"] = cambios_pec + 1
    l.raw(m["colores"])           # indices de color (no se usan aqui)

    l.skip(0x1D0 - cambios_pec)
    largo_bloque = l.i24()
    if largo_bloque is None:
        raise ErrorPes("sin longitud de bloque de puntadas")

    inicio = l.p - 5              # inicio real del bloque (2 + 3 bytes)
    fin_puntadas = inicio + largo_bloque

    # 3 bytes magicos + 4 int16; los dos primeros son ancho/alto en decimas.
    l.p = inicio + 5
    l.skip(3)
    ancho10 = l.i16()
    alto10 = l.i16()
    l.p = inicio + 16             # magia(3) + 4 shorts(8) = 11 desde inicio+5

    # Campo declarado en la cabecera del bloque PEC (decimas de mm). Se guarda
    # aparte del tamano real medido sobre las puntadas, que es el dato fiable.
    if ancho10 and alto10:
        m["declarado_ancho_mm"] = ancho10 / 10.0
        m["declarado_alto_mm"] = alto10 / 10.0
    else:
        m["notas"].append("cabecera PEC con ancho/alto 0: se usa el tamano medido")

    # --- Flujo de comandos --------------------------------------------------
    puntadas = saltos = trims = cambios = 0
    saltos_mov = 0
    suma_puntada = 0.0
    suma_salto = 0.0
    centinela = False
    ax = ay = 0
    xmin = xmax = ymin = ymax = 0

    while l.p < fin_puntadas:
        val1 = l.i8()
        val2 = l.i8()
        if val1 is None or val2 is None:
            break
        if val1 == 0xFF and val2 == 0x00:
            centinela = True
            break
        if val1 == 0xFE and val2 == 0xB0:
            l.skip(1)
            cambios += 1
            continue

        jump = trim = False
        if val1 & FLAG_LONG:
            if val1 & TRIM_CODE:
                trim = True
            if val1 & JUMP_CODE:
                jump = True
            x = _signed12((val1 << 8) | val2)
            val2 = l.i8()
            if val2 is None:
                break
        else:
            x = _signed7(val1)

        if val2 & FLAG_LONG:
            if val2 & TRIM_CODE:
                trim = True
            if val2 & JUMP_CODE:
                jump = True
            val3 = l.i8()
            if val3 is None:
                break
            y = _signed12((val2 << 8) | val3)
        else:
            y = _signed7(val2)

        # Prioridad identica a pyembroidery: si hay flag de salto, el registro
        # es un salto aunque tambien lleve el de corte.
        ax += x
        ay += y
        xmin = min(xmin, ax)
        xmax = max(xmax, ax)
        ymin = min(ymin, ay)
        ymax = max(ymax, ay)
        if jump:
            saltos += 1
            if x or y:
                saltos_mov += 1
                suma_salto += math.hypot(x, y) / 10.0
        elif trim:
            trims += 1
        else:
            puntadas += 1
            suma_puntada += math.hypot(x, y) / 10.0

    if not centinela:
        m["notas"].append("sin centinela 0xFF00: conteo acotado por la longitud del bloque")

    if fin_puntadas > len(data):
        m["notas"].append("el bloque de puntadas declara mas bytes de los que hay")

    m["puntadas"] = puntadas
    m["saltos"] = saltos
    m["saltos_mov"] = saltos_mov
    m["trims"] = trims
    m["saltos_pe"] = saltos + trims
    m["cambios_color"] = cambios
    if not m["nombre"]:
        m["nombre"] = "(sin nombre)"
    if not m["colores"]:
        m["colores"] = None

    # Tamano real medido sobre las puntadas: siempre disponible si hubo al
    # menos una, y no depende de que la cabecera este bien escrita.
    if puntadas or saltos:
        m["ancho_mm"] = (xmax - xmin) / 10.0
        m["alto_mm"] = (ymax - ymin) / 10.0
        for k_decl, k_real in (("declarado_ancho_mm", "ancho_mm"),
                               ("declarado_alto_mm", "alto_mm")):
            if m[k_decl] and abs(m[k_decl] - m[k_real]) > 0.5:
                m["notas"].append(
                    f"{k_decl.replace('declarado_', '').replace('_mm', '')} "
                    f"declarado {m[k_decl]:.1f} != medido {m[k_real]:.1f}")
    else:
        m["notas"].append("sin puntadas: no se puede medir el tamano del diseno")

    if puntadas:
        m["trims_por_1000"] = trims * 1000.0 / puntadas
        m["saltos_por_1000"] = saltos * 1000.0 / puntadas
        m["salto_mov_por_1000"] = saltos_mov * 1000.0 / puntadas
        m["largo_puntada_mm"] = suma_puntada / puntadas
    if saltos_mov:
        m["largo_salto_mm"] = suma_salto / saltos_mov

    m["sospechoso"] = _es_sospechoso(m)
    return m


def _leer_nombre_metadatos(l: Lector):
    """Primer string de metadatos de una cabecera v4/v5+ (nombre del diseno).

    Replica `read_pes_header_version_4/5` de pyembroidery. Los offsets son
    heuristicos del formato; si algo no cuadra se devuelve None en vez de
    inventar un valor.
    """
    backup = l.p
    try:
        l.skip(4)
        nombre = l.pascal()
        return nombre.strip() if nombre else None
    except Exception:
        l.p = backup
        return None


# --------------------------------------------------------------------------
# Perfil objetivo
# --------------------------------------------------------------------------
_METRICAS_PERFIL = [
    ("puntadas", "puntadas"),
    ("trims", "trims"),
    ("saltos", "saltos"),
    ("saltos_mov", "saltos con movimiento"),
    ("cambios_color", "cambios de color"),
    ("trims_por_1000", "trims por 1000 puntadas"),
    ("saltos_por_1000", "saltos por 1000 puntadas"),
    ("largo_puntada_mm", "largo medio de puntada (mm)"),
]


# Avisos que invalidan la lectura: un archivo con alguno de estos NO debe
# entrar en el perfil objetivo porque sus numeros no son fiables.
_NOTAS_GRAVES = (
    "puntero PEC fuera",
    "el bloque PEC no empieza",
    "declara mas bytes",
    "sin centinela",
    "cabecera PEC truncada",
    "sin longitud",
    "sin puntadas",
    "firma PEC inesperada",
)


def _es_sospechoso(m: dict) -> bool:
    return any(nota.startswith(_NOTAS_GRAVES) for nota in m.get("notas", ()))


def construir_perfil(filas: list[dict]) -> dict:
    """Promedios y rangos sobre las filas NO sospechosas.

    Se excluyen los archivos corruptos o truncados: un solo byte de mas les
    descuadra los conteos y arruinaria el perfil objetivo.
    """
    filas = [f for f in filas if not _es_sospechoso(f)]
    perfil = {}
    for clave, etiqueta in _METRICAS_PERFIL:
        vals = [f[clave] for f in filas if f.get(clave) is not None]
        if not vals:
            perfil[clave] = {"etiqueta": etiqueta, "n": 0}
            continue
        perfil[clave] = {
            "etiqueta": etiqueta,
            "n": len(vals),
            "media": statistics.fmean(vals),
            "min": min(vals),
            "max": max(vals),
            "mediana": statistics.median(vals),
        }
    return perfil


def _fmt(v, dec=2):
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{dec}f}"


def _referencia_mas_parecida(generada: dict, refs: list[dict]):
    """Referencia sana cuyo nº de puntadas es el mas proximo al generado."""
    if not refs:
        return None
    sanas = [f for f in refs
             if not _es_sospechoso(f) and f.get("puntadas") and generada.get("puntadas")]
    if not sanas:
        return None
    return min(sanas,
               key=lambda f: (abs(f["puntadas"] - generada["puntadas"]), f["archivo"]))


def _desviacion(valor, estad):
    """Clasifica un valor de generadas contra el rango de las referencias."""
    if valor is None or not estad or not estad.get("n"):
        return "sin dato", "-"
    lo, hi, media = estad["min"], estad["max"], estad["media"]
    if lo <= valor <= hi:
        return "DENTRO", f"{valor:.2f} (rango {lo:.2f}-{hi:.2f}, media {media:.2f})"
    if valor > hi:
        return "POR ENCIMA", f"{valor:.2f} > max {hi:.2f} (+{valor - hi:.2f}, media {media:.2f})"
    return "POR DEBAJO", f"{valor:.2f} < min {lo:.2f} ({valor - lo:+.2f}, media {media:.2f})"


# --------------------------------------------------------------------------
# Reportes
# --------------------------------------------------------------------------
def escribir_reporte_completo(salida: Path, refs, gens, fallidos) -> Path:
    L = []
    add = L.append
    add("=" * 78)
    add("REPORTE DE ANALISIS DE MATRICES .PES")
    add("=" * 78)
    add(f"Referencias : {refs and len(refs) or 0} archivo(s) analizado(s)")
    add(f"Generadas   : {gens and len(gens) or 0} archivo(s) analizado(s)")
    add(f"Fallidos    : {len(fallidos)}")
    add("")

    add("-" * 78)
    add("1. METRICAS POR ARCHIVO")
    add("-" * 78)

    def tabla(titulo, filas):
        add("")
        add(f"[{titulo}]")
        if not filas:
            add("  (ninguno)")
            return
        cab = (f"  {'archivo':<34}{'ver':<10}{'bytes':>9}{'punt':>8}{'jump':>7}"
           f"{'trim':>7}{'cc':>5}{'col':>5}{'jumpPE':>8}")
        add(cab)
        add("  " + "-" * (len(cab) - 2))
        for f in filas:
            marca = " *" if f.get("sospechoso") else ""
            add(f"  {f['archivo']:<34}{f['version']:<10}{f['bytes']:>9}"
                f"{f['puntadas']:>8}{f['saltos']:>7}{f['trims']:>7}"
                f"{f['cambios_color']:>5}{str(f['colores'] or '-'):>5}{f['saltos_pe']:>8}{marca}")
        add("  nota: 'jumpPE' = saltos + trims, que es lo que cuentan JUMP")
        add("        scripts/verify_pes.py y los benchmark_*.py via pyembroidery.")
        add("        * = lectura sospechosa: EXCLUIDO del perfil objetivo (ver avisos).")

    tabla("REFERENCIAS (externas, perfil objetivo)", refs)
    tabla("GENERADAS (exportadas por la aplicacion)", gens)

    add("")
    add("-" * 78)
    add("2. METRICAS NORMALIZADAS (por cada 1000 puntadas)")
    add("-" * 78)
    add(f"  {'archivo':<26}{'trim/1k':>10}{'jump/1k':>10}{'saltos mov/1k':>16}{'colores':>9}")
    add("  " + "-" * 71)
    for grupo, titulo in ((refs, "REFERENCIAS"), (gens, "GENERADAS")):
        add(f"  -- {titulo}")
        for f in grupo:
            add(f"  {f['archivo']:<26}{_fmt(f['trims_por_1000']):>10}"
                f"{_fmt(f['saltos_por_1000']):>10}{_fmt(f['salto_mov_por_1000']):>16}"
                f"{str(f['colores'] or '-'):>9}")

    add("")
    add("-" * 78)
    add("3. TAMANO DE DISENO Y NOMBRE INTERNO")
    add("-" * 78)
    for grupo, titulo in ((refs, "REFERENCIAS"), (gens, "GENERADAS")):
        add(f"  -- {titulo}")
        for f in grupo:
            tam = (f"{f['ancho_mm']:.1f} x {f['alto_mm']:.1f} mm"
                   if f["ancho_mm"] is not None else "no legible")
            add(f"  {f['archivo']:<34}{tam:<20}{f['nombre']}")

    add("")
    add("-" * 78)
    add("4. PERFIL OBJETIVO (derivado SOLO de las referencias)")
    add("-" * 78)
    perfil = construir_perfil(refs) if refs else {}
    excluidas = [f["archivo"] for f in refs if _es_sospechoso(f)]
    if excluidas:
        add(f"  [excluidas {len(excluidas)} referencia(s) por lectura sospechosa: "
            f"{', '.join(excluidas)}]")
    if not any(p.get("n") for p in perfil.values()):
        add("  Sin referencias utilizables: no se puede construir el perfil objetivo.")
    else:
        add(f"  {'metrica':<30}{'n':>4}{'media':>12}{'mediana':>12}{'min':>12}{'max':>12}")
        add("  " + "-" * 82)
        for clave, e in perfil.items():
            if not e.get("n"):
                add(f"  {e['etiqueta']:<30}{'0':>4}{'-':>12}{'-':>12}{'-':>12}{'-':>12}")
                continue
            dec = 0 if clave in ("puntadas", "trims", "saltos", "saltos_mov",
                                 "cambios_color") else 2
            add(f"  {e['etiqueta']:<30}{e['n']:>4}"
                f"{_fmt(e['media'], dec):>12}{_fmt(e['mediana'], dec):>12}"
                f"{_fmt(e['min'], dec):>12}{_fmt(e['max'], dec):>12}")

    add("")
    add("-" * 78)
    add("5. COMPARACION: GENERADAS vs PERFIL OBJETIVO")
    add("-" * 78)
    if not gens:
        add("  No hay archivos en la carpeta de generadas.")
    elif not any(p.get("n") for p in perfil.values()):
        add("  Sin perfil objetivo: no se puede comparar.")
    else:
        for f in gens:
            add("")
            add(f"  * {f['archivo']}  ({f['puntadas']} puntadas, "
                f"{f['colores'] or '?'} colores)")
            if f.get("sospechoso"):
                add("      [lectura sospechosa: comparacion poco fiable, ver avisos]")
            for clave in ("puntadas", "trims", "saltos", "cambios_color",
                          "trims_por_1000", "saltos_por_1000", "largo_puntada_mm"):
                e = perfil.get(clave, {})
                veredicto, detalle = _desviacion(f.get(clave), e)
                add(f"      {e.get('etiqueta', clave):<30}{veredicto:<14}{detalle}")

            # Trims/1000 y saltos/1000 dependen mucho del diseno (una letra de
            # un color no tiene zonas separadas; un logo de varios colores si).
            # Comparar contra el perfil completo mezcla disenos: la referencia
            # de CONTEO DE PUNTADAS MAS PARECIDO es la comparacion justa.
            ref = _referencia_mas_parecida(f, refs)
            if ref:
                add(f"      -- referencia mas parecida en nº de puntadas: {ref['archivo']}")
                add(f"         {ref['puntadas']} puntadas, {ref['colores'] or '?'} colores")
                for clave, etq in (("trims_por_1000", "trims/1k"),
                                   ("saltos_por_1000", "saltos/1k"),
                                   ("largo_puntada_mm", "largo puntada mm")):
                    a, b = f.get(clave), ref.get(clave)
                    if a is None or b is None:
                        continue
                    # Si la referencia es 0 el ratio es infinito: decirlo, no
                    # inventar un numero.
                    if b == 0:
                        extra = ("sin ratio (referencia = 0)" if a > 0
                                 else "ambas = 0")
                    else:
                        extra = f"x{a / b:.1f}"
                    add(f"         {etq:<18}esta={_fmt(a):<9}referencia={_fmt(b):<9}"
                        f"{extra}")

    add("")
    add("-" * 78)
    add("6. RESUMEN DE DESVIACIONES (trims y jumps son las metricas clave)")
    add("-" * 78)
    add("  trims/1000 y saltos/1000 objetivos: "
        f"{_fmt(perfil.get('trims_por_1000', {}).get('media'))} "
        f"(rango {_fmt(perfil.get('trims_por_1000', {}).get('min'))}-"
        f"{_fmt(perfil.get('trims_por_1000', {}).get('max'))})  /  "
        f"{_fmt(perfil.get('saltos_por_1000', {}).get('media'))} "
        f"(rango {_fmt(perfil.get('saltos_por_1000', {}).get('min'))}-"
        f"{_fmt(perfil.get('saltos_por_1000', {}).get('max'))})")
    for f in gens:
        vt, _ = _desviacion(f.get("trims_por_1000"), perfil.get("trims_por_1000", {}))
        vs, _ = _desviacion(f.get("saltos_por_1000"), perfil.get("saltos_por_1000", {}))
        add(f"    {f['archivo']:<26} trims/1k={_fmt(f['trims_por_1000']):<8}{vt:<14}"
            f" jumps/1k={_fmt(f['saltos_por_1000']):<8}{vs}")

    if fallidos:
        add("")
        add("-" * 78)
        add("7. ARCHIVOS NO ANALIZADOS")
        add("-" * 78)
        for nombre, motivo in fallidos:
            add(f"  {nombre:<28}{motivo}")

    add("")
    add("-" * 78)
    add("AVISOS POR ARCHIVO")
    add("-" * 78)
    con_notas = [f for f in refs + gens if f.get("notas")]
    if not con_notas:
        add("  Ninguno.")
    for f in con_notas:
        for nota in f["notas"]:
            add(f"  {f['archivo']:<28}{nota}")

    ruta = salida / "reporte_completo.txt"
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")
    return ruta


def escribir_perfil_objetivo(salida: Path, refs, gens) -> Path:
    perfil = construir_perfil(refs) if refs else {}
    L = []
    add = L.append
    add("PERFIL OBJETIVO DE BUENA MATRIZ .PES")
    add("=" * 60)
    add(f"Derivado de {len(refs)} referencia(s) externa(s) en referencias/.")
    add("Fuente: archivos de digitalizadores comerciales (no generados por esta app).")
    excluidas = [f["archivo"] for f in refs if _es_sospechoso(f)]
    if excluidas:
        add(f"Excluidas por lectura sospechosa: {', '.join(excluidas)}")
    add("")
    if not any(p.get("n") for p in perfil.values()):
        add("SIN DATOS: no hay referencias analizables.")
    else:
        add("METRICA                          n     media    min      max")
        add("-" * 60)
        for clave, e in perfil.items():
            if not e.get("n"):
                add(f"{e['etiqueta']:<32}{0:>3}{'-':>10}{'-':>10}{'-':>10}")
                continue
            dec = 0 if clave in ("puntadas", "trims", "saltos", "saltos_mov",
                                 "cambios_color") else 2
            add(f"{e['etiqueta']:<32}{e['n']:>3}"
                f"{_fmt(e['media'], dec):>10}{_fmt(e['min'], dec):>10}{_fmt(e['max'], dec):>10}")

    colores = [f["colores"] for f in refs if f.get("colores")]
    if colores:
        add("")
        add(f"Colores por diseno (referencias): min={min(colores)} max={max(colores)} "
            f"media={statistics.fmean(colores):.1f}")
    tamanos = [(f["ancho_mm"], f["alto_mm"]) for f in refs
               if f["ancho_mm"] is not None]
    if tamanos:
        add(f"Tamano de diseno (referencias): "
            f"ancho {min(t[0] for t in tamanos):.0f}-{max(t[0] for t in tamanos):.0f} mm, "
            f"alto {min(t[1] for t in tamanos):.0f}-{max(t[1] for t in tamanos):.0f} mm")

    if gens:
        add("")
        add("CONTRASTE DE LAS GENERADAS POR ESTA APP")
        add("-" * 60)
        for f in gens:
            vt, _ = _desviacion(f.get("trims_por_1000"), perfil.get("trims_por_1000", {}))
            vs, _ = _desviacion(f.get("saltos_por_1000"), perfil.get("saltos_por_1000", {}))
            vp, _ = _desviacion(f.get("puntadas"), perfil.get("puntadas", {}))
            add(f"  {f['archivo']}")
            add(f"    puntadas={f['puntadas']} ({vp})")
            add(f"    trims/1000={_fmt(f['trims_por_1000'])} ({vt})")
            add(f"    jumps/1000={_fmt(f['saltos_por_1000'])} ({vs})")

    add("")
    add("Nota: trims por debajo del objetivo = se deja hilo colgando entre zonas;")
    add("      jumps por encima = demasiados reposicionamientos (hilo y tiempo).")
    add("      El largo medio de puntada es la palanca principal de calidad/velocidad.")

    ruta = salida / "perfil_objetivo.txt"
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")
    return ruta


def escribir_csv(salida: Path, filas: list[dict]) -> Path:
    cols = ["carpeta", "archivo", "version", "bytes", "nombre", "puntadas",
            "saltos", "saltos_pe", "saltos_mov", "trims", "cambios_color", "colores",
            "ancho_mm", "alto_mm", "declarado_ancho_mm", "declarado_alto_mm",
            "trims_por_1000", "saltos_por_1000",
            "salto_mov_por_1000", "largo_puntada_mm", "largo_salto_mm", "notas"]
    ruta = salida / "reporte_completo.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for carpeta, grupo in filas:
            for f in grupo:
                fila = {c: f.get(c) for c in cols}
                fila["carpeta"] = carpeta
                fila["notas"] = " | ".join(f.get("notas", []))
                w.writerow(fila)
    return ruta


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def listar_pes(carpeta: Path):
    if not carpeta.is_dir():
        return None
    return sorted(
        (p for p in carpeta.iterdir()
         if p.is_file() and p.suffix.lower() == ".pes"),
        key=lambda p: p.name.lower(),
    )


def asegurar_carpetas(refs: Path, gens: Path, salida: Path) -> None:
    for carpeta, que in ((refs, "matrices .PES externas de buena calidad "
                               "(digitalizadores comerciales). No son salidas de esta app."),
                         (gens, "matrices .PES exportadas por esta aplicacion "
                                "(una por diseno generado)."),
                         (salida, "reportes que produce este script (se puede borrar).")):
        if not carpeta.exists():
            carpeta.mkdir(parents=True, exist_ok=True)
            print(f"[carpeta creada] {carpeta}")
            print(f"    Qué poner aquí: {que}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Perfil objetivo de matrices .PES y comparacion con las generadas.")
    p.add_argument("--referencias", type=Path, default=BASE_DIR / "referencias",
                   help="carpeta con .PES externos buenos (por defecto ./referencias)")
    p.add_argument("--generadas", type=Path, default=BASE_DIR / "generadas",
                   help="carpeta con .PES generados por la app (por defecto ./generadas)")
    p.add_argument("--salida", type=Path, default=BASE_DIR / "salida",
                   help="carpeta de reportes (por defecto ./salida)")
    p.add_argument("--csv", action="store_true",
                   help="escribe tambien reporte_completo.csv")
    args = p.parse_args(argv)

    asegurar_carpetas(args.referencias, args.generadas, args.salida)

    def analizar_carpeta(carpeta: Path, etiqueta: str):
        archivos = listar_pes(carpeta)
        if archivos is None:
            print(f"[aviso] {etiqueta}: {carpeta} no existe o no es una carpeta")
            return [], []
        if not archivos:
            print(f"[aviso] {etiqueta}: {carpeta} no contiene archivos .pes")
            return [], []
        filas, fallidos = [], []
        for ruta in archivos:
            try:
                filas.append(analizar_pes(ruta))
            except ErrorPes as exc:
                fallidos.append((ruta.name, str(exc)))
                print(f"[aviso] {ruta.name}: {exc}")
            except Exception as exc:  # nunca tumbar el lote
                fallido = (ruta.name, f"error inesperado: {exc!r}")
                fallidos.append(fallido)
                print(f"[aviso] {ruta.name}: {fallido[1]}")
        print(f"[ok] {etiqueta}: {len(filas)} analizado(s), {len(fallidos)} con problemas")
        return filas, fallidos

    refs, fallidos_refs = analizar_carpeta(args.referencias, "referencias")
    gens, fallidos_gens = analizar_carpeta(args.generadas, "generadas")
    fallidos = fallidos_refs + fallidos_gens

    r1 = escribir_reporte_completo(args.salida, refs, gens, fallidos)
    r2 = escribir_perfil_objetivo(args.salida, refs, gens)
    print(f"\n[escrito] {r1}")
    print(f"[escrito] {r2}")
    if args.csv:
        r3 = escribir_csv(args.salida, [("referencias", refs), ("generadas", gens)])
        print(f"[escrito] {r3}")

    if not refs:
        print("\n[perfil objetivo] No hay referencias analizables: no se pudo construir.")
        print("  Copia .PES de un digitalizador comercial en:", args.referencias)
    return 0


if __name__ == "__main__":
    sys.exit(main())