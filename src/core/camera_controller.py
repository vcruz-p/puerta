"""
Controlador de cámaras para captura de video.

Soporta:
    - Cámara AXIS Q7401 mediante RTSP
    - Cámaras locales mediante OpenCV
"""

import cv2
import threading
import time
from urllib.parse import quote


class CameraController:
    """
    Controlador centralizado de cámaras.
    
    Gestiona la conexión, captura y desconexión de cámaras
    en un hilo separado para no bloquear la interfaz gráfica.
    """

    def __init__(
        self,
        config,
        on_frame=None,
        on_status=None,
        on_error=None,
        log=lambda x: None
    ):
        """
        Inicializa el controlador de cámara.
        
        Args:
            config (dict): Configuración completa del sistema.
            on_frame (callable, optional): Callback para cada frame capturado.
            on_status (callable, optional): Callback para cambios de estado.
            on_error (callable, optional): Callback para errores.
            log (callable, optional): Función de logging.
        """
        self.config = config

        self.on_frame = on_frame
        self.on_status = on_status
        self.on_error = on_error
        self.log = log

        self.running = False
        self.connected = False

        self.cap = None
        self.thread = None

        self.source = "AXIS"

        self.lock = threading.RLock()

        self.reconnect_delay = 2.0
        self.read_error_limit = 5

        self.target_fps = 25
        self.last_frame_time = 0.0

    def update_config(self, config):
        """
        Actualiza la configuración de cámara.
        
        Si la cámara está conectada, se desconecta para aplicar
        la nueva configuración en la próxima conexión.
        
        Args:
            config (dict): Nueva configuración completa.
        """
        self.config = config

        if self.running:
            self.log("Configuración de cámara actualizada.")
            self.stop()

    def build_rtsp_url(self):
        """
        Construye la URL RTSP para cámara AXIS.
        
        Returns:
            str: URL RTSP completa con autenticación.
            
        Raises:
            ValueError: Si no se ha configurado la IP de la cámara.
        """
        camera = self.config.get("camera", {})

        ip = str(camera.get("ip", "")).strip()
        port = int(camera.get("rtsp_port", 554))

        username = str(camera.get("username", ""))
        password = str(camera.get("password", ""))

        path = str(
            camera.get("rtsp_path", "/axis-media/media.amp?videocodec=h264")
        ).strip()

        if not ip:
            raise ValueError("No se ha configurado la IP de la cámara.")

        if not path.startswith("/"):
            path = "/" + path

        # Escapar correctamente usuario y contraseña.
        user = quote(username, safe="")
        pwd = quote(password, safe="")

        auth = ""
        if user:
            auth = user
            if pwd:
                auth += ":" + pwd
            auth += "@"

        return f"rtsp://{auth}{ip}:{port}{path}"

    def url(self):
        """Alias para build_rtsp_url()."""
        return self.build_rtsp_url()

    def start(self, source="AXIS"):
        """
        Inicia la captura de video.
        
        Args:
            source (str|int): "AXIS" para cámara de red, o índice numérico
                             para cámara local (0, 1, 2...).
                             
        Returns:
            bool: True si se inició correctamente.
        """
        with self.lock:
            if self.running:
                self.log("La cámara ya está ejecutándose.")
                return False

            self.source = source
            self.running = True
            self.connected = False

            self.thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="CameraController"
            )

            self.thread.start()

        self.log(f"Iniciando cámara: {source}")
        return True

    def _worker(self):
        """Hilo de trabajo para captura de video."""
        consecutive_errors = 0

        while self.running:
            cap = None

            try:
                # Seleccionar fuente
                if self.source == "AXIS":
                    source = self.build_rtsp_url()
                    self.log(
                        f"RTSP: {self.config['camera']['ip']}:"
                        f"{self.config['camera']['rtsp_port']}"
                    )

                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

                    # Para AXIS es preferible TCP.
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                else:
                    source = int(self.source)
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                # Comprobar conexión
                if not cap or not cap.isOpened():
                    self.connected = False
                    self._status("ERROR")
                    self._error("No se pudo abrir la cámara.")

                    if cap:
                        cap.release()
                    cap = None

                    if self.running:
                        self.log("Reintentando conexión de cámara...")
                        self._sleep(self.reconnect_delay)

                    continue

                # Cámara conectada
                with self.lock:
                    self.cap = cap

                self.connected = True
                consecutive_errors = 0
                self._status("CONECTADA")
                self.log(f"Cámara conectada: {self.source}")

                # Lectura de frames
                while self.running:
                    ok, frame = cap.read()

                    if not ok or frame is None:
                        consecutive_errors += 1

                        if consecutive_errors >= self.read_error_limit:
                            self.log("Se perdió la señal de cámara.")
                            break

                        self._sleep(0.05)
                        continue

                    consecutive_errors = 0

                    # Control FPS
                    now = time.monotonic()
                    frame_interval = 1.0 / self.target_fps

                    if now - self.last_frame_time < frame_interval:
                        continue

                    self.last_frame_time = now

                    # Callback
                    if self.on_frame:
                        try:
                            self.on_frame(frame)
                        except Exception as exc:
                            self.log(f"Error procesando frame: {exc}")

                # Liberar captura
                self.connected = False
                self._status("DESCONECTADA")

                try:
                    cap.release()
                except Exception:
                    pass

                with self.lock:
                    if self.cap is cap:
                        self.cap = None

                cap = None

                # Reconectar
                if self.running:
                    self.log("Reconectando cámara...")
                    self._sleep(self.reconnect_delay)

            except Exception as exc:
                self.connected = False
                self.log(f"Error cámara: {exc}")
                self._error(str(exc))
                self._status("ERROR")

                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass

                with self.lock:
                    if self.cap is cap:
                        self.cap = None

                if self.running:
                    self._sleep(self.reconnect_delay)

        # FIN
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        self.connected = False
        self._status("DESCONECTADA")
        self.log("Hilo de cámara detenido.")

    def _status(self, status):
        """Callback de estado."""
        if not self.on_status:
            return
        try:
            self.on_status(status)
        except Exception as exc:
            self.log(f"Error callback estado: {exc}")

    def _error(self, message):
        """Callback de error."""
        if not self.on_error:
            return
        try:
            self.on_error(message)
        except Exception as exc:
            self.log(f"Error callback cámara: {exc}")

    def _sleep(self, seconds):
        """Sleep interrumpible."""
        end = time.monotonic() + seconds
        while self.running and time.monotonic() < end:
            time.sleep(0.05)

    def stop(self):
        """Detiene la captura de video."""
        with self.lock:
            if not self.running:
                return

            self.running = False
            cap = self.cap
            self.cap = None

        if cap:
            try:
                cap.release()
            except Exception:
                pass

        self.connected = False
        self.log("Deteniendo cámara...")

    def is_running(self):
        """Verifica si la cámara está en ejecución."""
        return self.running

    def is_connected(self):
        """Verifica si la cámara está conectada."""
        return self.connected
