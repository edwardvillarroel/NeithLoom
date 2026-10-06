"""Catálogo de bastidores (hoops) compatibles.

Cada bastidor lleva su marca, su nombre para mostrar y sus dimensiones
máximas de bordado en milímetros. La estructura tiene el campo "brand"
preparado para que en el futuro se puedan listar bastidores de otras marcas
(Janome, etc.) sin tocar la lógica de generación; por ahora solo se incluyen
los bastidores Brother más comunes y el de 13x18 cm se usa por defecto.
"""

HOOPS = (
    {
        "brand": "Brother",
        "name": "10 × 10 cm (4x4\")",
        "width_mm": 100,
        "height_mm": 100,
    },
    {
        "brand": "Brother",
        "name": "13 × 18 cm (5x7\")",
        "width_mm": 130,
        "height_mm": 180,
    },
    {
        "brand": "Brother",
        "name": "16 × 26 cm (6x10\")",
        "width_mm": 160,
        "height_mm": 260,
    },
    {
        "brand": "Brother",
        "name": "20 × 20 cm",
        "width_mm": 200,
        "height_mm": 200,
    },
    {
        "brand": "Brother",
        "name": "20 × 30 cm",
        "width_mm": 200,
        "height_mm": 300,
    },
    {
        "brand": "Brother",
        "name": "25 × 25 cm",
        "width_mm": 250,
        "height_mm": 250,
    },
)

DEFAULT_HOOP_INDEX = 1  # 13 × 18 cm (5x7")

# Rango del bastidor Personalizado, derivado del propio catálogo: del lado
# más corto (10 cm del 10 × 10) al más largo (30 cm del 20 × 30). El motor no
# impone techo: `stitch_generator.MAX_SIZE_MM` es solo el valor por defecto
# cuando no se pasan medidas, y el 20 × 30 ya las pasa.
MIN_CUSTOM_CM = min(min(h["width_mm"], h["height_mm"]) for h in HOOPS) / 10
MAX_CUSTOM_CM = max(max(h["width_mm"], h["height_mm"]) for h in HOOPS) / 10


def custom_hoop_label(width_mm: float, height_mm: float) -> str:
    """Nombre que se muestra de un bastidor de medidas libres, en cm."""
    return f"Personalizado {width_mm / 10:g} x {height_mm / 10:g} cm"


def set_custom_size(hoop: dict, width_mm: float, height_mm: float) -> None:
    """Ajusta en el sitio las medidas de un bastidor personalizado."""
    hoop["width_mm"] = float(width_mm)
    hoop["height_mm"] = float(height_mm)
    hoop["name"] = custom_hoop_label(width_mm, height_mm)


def custom_hoop(width_mm: float, height_mm: float) -> dict:
    """Bastidor de ancho y alto elegidos por el usuario.

    No forma parte de `HOOPS`: el catálogo sigue siendo solo bastidores de
    Brother, y `hoops_for_brand` no debe devolver un tamaño que la máquina no
    vende.
    """
    hoop = {
        "brand": "Personalizado",
        "name": "",
        "custom": True,
        "width_mm": 0.0,
        "height_mm": 0.0,
    }
    set_custom_size(hoop, width_mm, height_mm)
    return hoop


def hoops_for_brand(brand: str = "Brother"):
    """Devuelve los bastidores de una marca (por defecto, Brother)."""
    return [h for h in HOOPS if h["brand"] == brand]