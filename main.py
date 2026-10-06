import ctypes
import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow, find_app_icon

# Identificador del proceso para Windows. Sin él, la barra de tareas agrupa y
# etiqueta NeithLoom como "python" y muestra el icono genérico de Python en vez
# del de la aplicación. Se fija antes de crear la ventana.
APP_USER_MODEL_ID = "NeithLoom.bordado"


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )
    except Exception:  # noqa: BLE001 - sin AUMID solo se pierde el icono
        pass


def main() -> int:
    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    icon = find_app_icon()
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)
    window = MainWindow()
    window.show()
    if icon is not None and not icon.isNull():
        # WM_SETICON de la ventana nativa: en Windows es lo que usa la barra
        # de tareas y el conmutador Alt+Tab, no el icono de QApplication.
        handle = window.windowHandle()
        if handle is not None:
            handle.setIcon(icon)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())