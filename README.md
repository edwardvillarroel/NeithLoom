# NeithLoom

Convierte una imagen en una matriz de bordado lista para máquina (`.PES`, `.DST`), pensado para personas sin experiencia técnica: eliges una imagen y un bastidor, y el programa se encarga del resto.

## Características

- **Flujo guiado en 4 pasos**: Elegir imagen → Ajustar → Generar bordado → Guardar. No hace falta tocar un solo parámetro técnico.
- **Detección automática de colores de hilo**: el programa reduce la paleta de la imagen y la traduce a hilos reales del catálogo de Brother.
- **Selección de tamaño de bastidor**: bastidores predefinidos de Brother (10 × 10, 13 × 18, 16 × 26, 20 × 20, 20 × 30 y 25 × 25 cm) o un tamaño personalizado.
- **Exportación a `.PES` y `.DST`**: los formatos que leen la mayoría de máquinas de bordar modernas y los programas de paso a máquina.
- **Ajustes opcionales pero sencillos**: densidad del relleno (Baja / Media / Alta), cambiar el color de fondo, eliminar zonas o recolorear directamente sobre la imagen.

## Instalación

### Usuario final

1. Entra en la sección de **Releases** de este repositorio.
2. Descarga el instalador o el archivo `.exe` de la última versión (Windows).
3. Ejecuta el instalador y abre NeithLoom. No requiere ningún paso adicional.

### Desarrolladores

NeithLoom corre desde código fuente con Python. Requisitos reales del proyecto:

- **Python 3.13** (probado con 3.13.5)
- Dependencias runtime (fijadas en [requirements.txt](requirements.txt)):

  | Librería                 | Uso en el programa |
  | ------------------------ | ------------------ |
  | `PySide6`                | Interfaz gráfica (Qt) |
  | `pillow`                 | Carga y lectura de imágenes |
  | `numpy`                  | Procesamiento de píxeles |
  | `scipy`                  | Morfología y análisis de regiones |
  | `opencv-python-headless` | Operaciones de imagen (usa la variante *headless*, sin UI) |

```powershell
git clone https://github.com/<tu-usuario>/neithloom.git
cd neithloom

python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

.\venv\Scripts\python.exe main.py
```

> `psutil` y `pyembroidery` siguen instalados en el venv de desarrollo pero no se importan en runtime, por eso no aparecen en `requirements.txt`.

## Uso

El flujo completo son 4 pasos, uno por pantalla:

1. **Elegir imagen**: selecciona una imagen JPG o PNG desde tu equipo o desde la galería integrada.

   `[screenshot: paso 1 "Elegir imagen" aquí]`

2. **Ajustar**: elige el máximo de colores de hilo, el color de fondo, elimina zonas o recolorea con el cuentagotas. Comprueba la vista previa de los colores de hilo elegidos.

   `[screenshot: paso 2 "Ajustar" aquí]`

3. **Generar bordado**: elige la densidad (Baja / Media / Alta) y el tamaño de tu bastidor (predefinido o personalizado), y pulsa "Generar bordado". Verás una simulación del bordado dentro del bastidor elegido.

4. **Guardar**: exporta la matriz a `.PES` y/o `.DST`, elige la carpeta de destino y pasa el archivo a tu máquina de bordar.

El programa ajusta la imagen solo para que quepa en el bastidor elegido: en el archivo de bordado no hace falta escribir a mano ningún parámetro de máquina.

## Requisitos mínimos

- **Sistema operativo**: Windows 10 u 11 (64 bits). Es el único sistema que se prueba; el empaquetado para publicar (`.exe`) y las funciones nativas (icono en la barra de tareas, detección de pendrives USB) son específicas de Windows.
- **RAM**: cualquier equipo moderno sirve. El pico de memoria en las pruebas de rendimiento fue de ~170 MB durante el procesamiento más pesado, así que 1 GB libre es más que suficiente.
- **CPU**: no hay un requisito real de CPU. Las mediciones de las pruebas registraron un uso máximo de ~165% de un núcleo (procesamiento en paralelo leve), y la imagen se reduce antes de calcular la paleta y las puntadas. Funciona bien en gama baja.

## Contribuir

- **Reportar un bug**: abre un *issue* describiendo qué esperabas, qué pasó y, si es posible, adjunta la imagen y el archivo `.PES` generado.
- **Proponer un cambio**: abre un *pull request*. El código sigue una lógica de flujo por pasos ya implementada (elegir imagen, ajustar, generar, guardar) y un motor separado en `core/` (`image_loader`, `color_processor`, `stitch_generator`, `exporters`); mantén esa división al tocar algo.

## Apoyo

NeithLoom es y seguirá siendo **100% gratuito y sin publicidad**. Si te resulta útil, el programa incluye un apartado opcional de donación (QR de Mercado Pago) que aparece solo de vez en cuando después de un guardado exitoso; es completamente opcional y puedes pedir que no vuelva a mostrarse.

## Licencia

Este proyecto está publicado bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE).