#!/usr/bin/env python3
"""
Control de Acceso - Aplicación Principal

Aplicación de escritorio para control de puerta con cámara AXIS Q7401,
detección de personas, reconocimiento facial y gestos con YOLO.

Uso:
    python run.py
"""

import sys
from pathlib import Path

# Agregar el directorio padre al path para importar src
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ttkbootstrap as ttk
from src.utils.config_manager import load_config
from src.core.door_controller import DoorController
from src.core.camera_controller import CameraController
from src.core.automation_engine import AutomationEngine
from src.gui.main_window import MainWindow


def main():
    """Punto de entrada principal de la aplicación."""
    
    # Cargar configuración
    cfg = load_config()
    
    # Crear ventana principal
    root = ttk.Window(
        title="Control de Acceso",
        themename="darkly"
    )
    
    root.geometry("1500x900")
    root.minsize(1100, 700)
    
    # Inicializar controladores
    door_ctl = DoorController(cfg, logger=lambda msg: print(f"[DOOR] {msg}"))
    
    camera_ctl = CameraController(
        cfg,
        on_frame=None,  # Se asignará después
        log=lambda msg: print(f"[CAM] {msg}")
    )
    
    # Inicializar motor de automatización (opcional)
    engine = None
    try:
        engine = AutomationEngine(
            cfg,
            open_door_callback=lambda source: None,  # Se asignará después
            logger=lambda msg: print(f"[AUTO] {msg}")
        )
        print("[AUTO] Motor de automatización inicializado.")
    except Exception as exc:
        print(f"[AUTO] Error inicializando automatización: {exc}")
    
    # Crear interfaz principal
    app = MainWindow(root, cfg, door_ctl, camera_ctl, engine)
    
    # Conectar callbacks
    camera_ctl.on_frame = app.on_camera_frame
    
    if engine:
        engine.open_door = app.open_door
    
    # Manejar cierre de ventana
    def on_close():
        try:
            camera_ctl.stop()
        except Exception:
            pass
        
        if engine:
            try:
                engine.close()
            except Exception:
                pass
        
        try:
            from src.utils.config_manager import save_config
            save_config(cfg)
        except Exception:
            pass
        
        try:
            root.destroy()
        except Exception:
            pass
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # Iniciar aplicación
    root.mainloop()


if __name__ == "__main__":
    main()
