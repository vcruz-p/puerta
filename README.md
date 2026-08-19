# Control de Puerta + AXIS Q7401

Aplicación de escritorio en Python con Tkinter moderno (`ttkbootstrap`).

## Funciones

- Visualización RTSP de una AXIS Q7401.
- Configuración de IP, usuario y contraseña de la cámara.
- Configuración del puerto RTSP.
- Configuración de IP, puerto, usuario y contraseña del controlador de puerta.
- Endpoint independiente para ABRIR.
- Endpoint independiente para CERRAR.
- HTTP GET o POST.
- Registro de eventos.
- Comunicación de cámara y puerta en hilos separados para no congelar la interfaz.

## Instalación en Windows

Abrir PowerShell en esta carpeta:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Si PowerShell bloquea la activación del entorno:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Luego:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

## AXIS Q7401

La aplicación usa RTSP. El valor inicial es:

`/axis-media/media.amp?videocodec=h264`

Si tu Q7401 utiliza otro perfil/stream, cambia `RTSP path` desde la interfaz.

## Puerta

Ejemplo:

IP: `192.168.1.50`

Puerto: `8080`

Abrir endpoint: `/api/door/open`

Cerrar endpoint: `/api/door/close`

Método: `POST`

La aplicación generará:

`http://192.168.1.50:8080/api/door/open`

y

`http://192.168.1.50:8080/api/door/close`

Si el controlador requiere autenticación HTTP Basic, completa usuario y contraseña.

## Importante

Los endpoints reales de la puerta dependen del controlador/relé que tengas instalado. No deben suponerse `/open` y `/close`: sustitúyelos por los endpoints reales del dispositivo.

Para una instalación de producción conviene añadir HTTPS, autenticación, permisos de operador/admin, registro de auditoría y confirmación antes de accionar el relé.
