# Control de Acceso - Aplicación Reestructurada

Aplicación de escritorio en Python con Tkinter moderno (`ttkbootstrap`) para control de puerta con cámara AXIS Q7401.

## Estructura del Proyecto

```
/workspace/
├── run.py                      # Punto de entrada principal
├── src/
│   ├── __init__.py            # Paquete principal
│   ├── core/                  # Lógica de negocio
│   │   ├── __init__.py
│   │   ├── door_controller.py      # Controlador de puerta
│   │   ├── camera_controller.py    # Controlador de cámara
│   │   └── automation_engine.py    # Motor de automatización
│   ├── gui/                   # Interfaz gráfica
│   │   ├── __init__.py
│   │   └── main_window.py          # Ventana principal
│   └── utils/                 # Utilidades
│       ├── __init__.py
│       └── config_manager.py       # Gestión de configuración
├── config.json                # Configuración (generado automáticamente)
├── data/                      # Datos de la aplicación
│   └── faces/                 # Rostros registrados
└── requirements.txt           # Dependencias
```

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python run.py
```

## Módulos

### Core (`src/core/`)

- **DoorController**: Controla la apertura de puerta mediante HTTP GET/POST
- **CameraController**: Gestiona captura de video RTSP (AXIS) y cámaras locales
- **AutomationEngine**: Motor de automatización con detección de gestos YOLO Pose

### GUI (`src/gui/`)

- **MainWindow**: Interfaz gráfica principal con controles de cámara, puerta y automatización

### Utils (`src/utils/`)

- **config_manager**: Carga y guardado de configuración JSON

## Uso

```bash
python run.py
```

La aplicación abrirá una ventana con:
- Visualización de video en tiempo real
- Controles para conectar/desconectar cámara AXIS o local
- Botón de apertura manual de puerta
- Panel de automatización con detección de gestos
- Logs del sistema

## Configuración

La configuración se guarda en `config.json` e incluye:

- **camera**: IP, usuario, contraseña, puerto RTSP, path
- **door**: IP, puerto, endpoint, método HTTP, timeout
- **automation**: Lógica (ANY/ALL), gestos, thresholds, cooldown

## Notas Importantes

- Los endpoints reales de la puerta dependen del controlador/relé instalado
- Para producción: añadir HTTPS, autenticación y permisos
- Requiere modelos YOLO: `yolo11n.pt` y `yolo11n-pose.pt`
