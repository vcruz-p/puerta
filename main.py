import tkinter as tk
import threading
import time

import cv2
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import messagebox

from config import load_config, save_config
from camera import CameraController
from door import DoorController

try:
    from automation import AutomationEngine
except ImportError:
    AutomationEngine = None


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()

        # ---------------------------------------------------------
        # CONFIGURACIÓN
        # ---------------------------------------------------------
        self.cfg.setdefault("camera", {})
        self.cfg.setdefault("door", {})
        self.cfg.setdefault("automation", {})

        camera_cfg = self.cfg["camera"]
        door_cfg = self.cfg["door"]
        automation_cfg = self.cfg["automation"]

        camera_cfg.setdefault("ip", "")
        camera_cfg.setdefault("username", "")
        camera_cfg.setdefault("password", "")
        camera_cfg.setdefault("rtsp_port", 554)
        camera_cfg.setdefault(
            "rtsp_path",
            "/axis-media/media.amp?videocodec=h264"
        )
        camera_cfg.setdefault("transport", "tcp")

        door_cfg.setdefault("ip", "")
        door_cfg.setdefault("port", 80)
        door_cfg.setdefault("open_endpoint", "/open")
        door_cfg.setdefault("method", "GET")
        door_cfg.setdefault("timeout", 5)

        automation_cfg.setdefault("enabled", False)
        automation_cfg.setdefault("logic", "ANY")
        automation_cfg.setdefault("standing_enabled", True)
        automation_cfg.setdefault("standing_seconds", 4)
        automation_cfg.setdefault("face_enabled", False)
        automation_cfg.setdefault("face_threshold", 0.48)
        automation_cfg.setdefault("gesture_enabled", False)
        automation_cfg.setdefault("gesture", "open_hand")
        automation_cfg.setdefault("yolo_confidence", 0.50)
        automation_cfg.setdefault("yolo_model", "yolo11n.pt")
        automation_cfg.setdefault("cooldown_seconds", 8)

        # ---------------------------------------------------------
        # VARIABLES
        # ---------------------------------------------------------
        self.frame = None
        self.camera_running = False
        self.camera_source = "AXIS"
        self.door_busy = False

        self.status = ttk.StringVar(value="SISTEMA LISTO")
        self.cam = ttk.StringVar(value="DESCONECTADA")
        self.door = ttk.StringVar(value="LISTA")
        self.http_status = ttk.StringVar(value="HTTP: ---")
        self.auto = ttk.StringVar(value="AUTOMATIZACIÓN: OFF")
        self.last_action = ttk.StringVar(
            value="Última operación: ---"
        )
        self.connection_info = ttk.StringVar(value="")

        self.auto_var = tk.BooleanVar(
            value=automation_cfg.get("enabled", False)
        )

        self.gesture_var = tk.BooleanVar(
            value=automation_cfg.get(
                "gesture_enabled",
                False
            )
        )

        # ---------------------------------------------------------
        # INTERFAZ
        # ---------------------------------------------------------
        self.build()

        # ---------------------------------------------------------
        # CONTROLADOR DE PUERTA
        # ---------------------------------------------------------
        self.door_ctl = DoorController(
            self.cfg,
            self.log
        )

        # ---------------------------------------------------------
        # CÁMARA
        # ---------------------------------------------------------
        self.camera_ctl = CameraController(
            self.cfg,
            on_frame=self.on_camera_frame,
            log=self.log
        )

        # ---------------------------------------------------------
        # AUTOMATIZACIÓN
        # ---------------------------------------------------------
        self.engine = None

        if AutomationEngine:
            try:
                self.engine = AutomationEngine(
                    self.cfg,
                    self.open_door,
                    self.log
                )

                self.log(
                    "Motor de automatización inicializado."
                )

            except Exception as exc:
                self.log(
                    f"Error inicializando automatización: {exc}"
                )

        else:
            self.log(
                "AutomationEngine no disponible."
            )

        self.update_automation_status()
        self.update_connection_info()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    # =============================================================
    # INTERFAZ PRINCIPAL
    # =============================================================

    def build(self):

        main = ttk.Frame(
            self.root,
            padding=12
        )

        main.pack(
            fill=BOTH,
            expand=True
        )

        # ---------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------

        header = ttk.Frame(main)
        header.pack(
            fill=X,
            pady=(0, 10)
        )

        title = ttk.Frame(header)
        title.pack(
            side=LEFT
        )

        ttk.Label(
            title,
            text="CONTROL DE ACCESO",
            font=(
                "Segoe UI",
                24,
                "bold"
            )
        ).pack(
            anchor=W
        )

        ttk.Label(
            title,
            text="AXIS Q7401 • PERSONA • ROSTRO • SEÑAS",
            bootstyle="secondary"
        ).pack(
            anchor=W
        )

        ttk.Button(
            header,
            text="⚙ CONFIGURACIÓN",
            bootstyle="secondary",
            command=self.open_configuration
        ).pack(
            side=RIGHT
        )

        ttk.Label(
            header,
            textvariable=self.status,
            bootstyle="info",
            font=(
                "Segoe UI",
                11,
                "bold"
            )
        ).pack(
            side=RIGHT,
            padx=12
        )

        # ---------------------------------------------------------
        # BODY
        # ---------------------------------------------------------

        body = ttk.Panedwindow(
            main,
            orient=HORIZONTAL
        )

        body.pack(
            fill=BOTH,
            expand=True
        )

        left = ttk.Frame(
            body,
            padding=(0, 0, 8, 0)
        )

        right = ttk.Frame(
            body,
            padding=(8, 0, 0, 0)
        )

        body.add(
            left,
            weight=5
        )

        body.add(
            right,
            weight=3
        )

        # =========================================================
        # VIDEO
        # =========================================================

        vp = ttk.Labelframe(
            left,
            text=" MONITOREO EN VIVO ",
            bootstyle="primary",
            padding=5
        )

        vp.pack(
            fill=BOTH,
            expand=True
        )

        self.video = ttk.Label(
            vp,
            text=(
                "CÁMARA DESCONECTADA\n\n"
                "Seleccione AXIS o CÁMARA LOCAL"
            ),
            anchor=CENTER,
            justify=CENTER,
            font=(
                "Segoe UI",
                18,
                "bold"
            )
        )

        self.video.pack(
            fill=BOTH,
            expand=True
        )

        # ---------------------------------------------------------
        # BARRA CÁMARA
        # ---------------------------------------------------------

        camera_bar = ttk.Frame(left)

        camera_bar.pack(
            fill=X,
            pady=7
        )

        self.connect_btn = ttk.Button(
            camera_bar,
            text="▶ AXIS",
            bootstyle="success",
            command=self.start_camera
        )

        self.connect_btn.pack(
            side=LEFT
        )

        self.local_btn = ttk.Button(
            camera_bar,
            text="💻 CÁMARA LOCAL",
            bootstyle="info",
            command=self.start_local_camera
        )

        self.local_btn.pack(
            side=LEFT,
            padx=5
        )

        ttk.Button(
            camera_bar,
            text="■ DESCONECTAR",
            bootstyle="secondary",
            command=self.stop_camera
        ).pack(
            side=LEFT
        )

        ttk.Label(
            camera_bar,
            textvariable=self.cam,
            bootstyle="info",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            side=RIGHT
        )

        # =========================================================
        # ESTADO
        # =========================================================

        sp = ttk.Labelframe(
            right,
            text=" ESTADO DEL SISTEMA ",
            bootstyle="info",
            padding=10
        )

        sp.pack(
            fill=X,
            pady=(0, 8)
        )

        self.row(
            sp,
            "CÁMARA",
            self.cam
        )

        self.row(
            sp,
            "PUERTA",
            self.door
        )

        self.row(
            sp,
            "HTTP",
            self.http_status
        )

        self.row(
            sp,
            "AUTOMÁTICO",
            self.auto
        )

        ttk.Separator(sp).pack(
            fill=X,
            pady=6
        )

        ttk.Label(
            sp,
            textvariable=self.connection_info,
            bootstyle="secondary",
            wraplength=380,
            justify=LEFT
        ).pack(
            fill=X
        )

        # =========================================================
        # CONTROL PUERTA
        # =========================================================

        dp = ttk.Labelframe(
            right,
            text=" CONTROL DE PUERTA ",
            bootstyle="warning",
            padding=12
        )

        dp.pack(
            fill=X,
            pady=(0, 8)
        )

        self.openbtn = ttk.Button(
            dp,
            text="🔓\n\nABRIR PUERTA",
            bootstyle="success",
            command=self.open_door
        )

        self.openbtn.pack(
            fill=X,
            ipady=22
        )

        ttk.Label(
            dp,
            text="Apertura mediante HTTP",
            bootstyle="secondary"
        ).pack(
            pady=(6, 0)
        )

        # =========================================================
        # AUTOMATIZACIÓN
        # =========================================================

        ap = ttk.Labelframe(
            right,
            text=" AUTOMATIZACIÓN ",
            bootstyle="primary",
            padding=10
        )

        ap.pack(
            fill=X,
            pady=(0, 8)
        )

        ttk.Checkbutton(
            ap,
            text="Activar automatización",
            variable=self.auto_var,
            command=self.save_auto,
            bootstyle="success"
        ).pack(
            anchor=W,
            pady=2
        )

        ttk.Label(
            ap,
            text="Lógica de señales:"
        ).pack(
            anchor=W,
            pady=(5, 2)
        )

        self.logic_combo = ttk.Combobox(
            ap,
            values=[
                "ANY",
                "ALL"
            ],
            state="readonly"
        )

        self.logic_combo.set(
            self.cfg["automation"].get(
                "logic",
                "ANY"
            )
        )

        self.logic_combo.pack(
            fill=X
        )

        # ---------------------------------------------------------
        # SEÑAS CON YOLO POSE
        # ---------------------------------------------------------

        ttk.Label(
            ap,
            text="👋 Reconocimiento de Señas (YOLO Pose)",
            bootstyle="primary"
        ).pack(
            anchor=W,
            pady=(5, 3)
        )

        ttk.Checkbutton(
            ap,
            text="✋ Activar reconocimiento de seña",
            variable=self.gesture_var,
            command=self.save_auto,
            bootstyle="success"
        ).pack(
            anchor=W,
            pady=2
        )

        # Selector de gesto
        ttk.Label(
            ap,
            text="Gesto para activar:"
        ).pack(
            anchor=W,
            pady=(8, 2)
        )

        self.gesture_combo = ttk.Combobox(
            ap,
            values=[
                "open_hand",
                "fist", 
                "two_fingers",
                "victory",
                "thumbs_up",
                "pointing"
            ],
            state="readonly"
        )

        self.gesture_combo.set(
            self.cfg["automation"].get(
                "gesture",
                "open_hand"
            )
        )

        self.gesture_combo.pack(
            fill=X
        )

        # Tiempo de estabilidad
        ttk.Label(
            ap,
            text="Tiempo de estabilidad (segundos):"
        ).pack(
            anchor=W,
            pady=(8, 2)
        )

        self.gesture_sec = ttk.Spinbox(
            ap,
            from_=0.1,
            to=2.0,
            increment=0.05
        )

        self.gesture_sec.set(
            self.cfg["automation"].get(
                "gesture_stable_seconds",
                0.25
            )
        )

        self.gesture_sec.pack(
            fill=X
        )

        ttk.Button(
            ap,
            text="💾 GUARDAR AUTOMATIZACIÓN",
            bootstyle="primary",
            command=self.save_auto
        ).pack(
            fill=X,
            pady=(10, 0)
        )

        # =========================================================
        # ÚLTIMA OPERACIÓN
        # =========================================================

        action = ttk.Labelframe(
            right,
            text=" ÚLTIMA OPERACIÓN ",
            bootstyle="secondary",
            padding=7
        )

        action.pack(
            fill=X,
            pady=(0, 8)
        )

        ttk.Label(
            action,
            textvariable=self.last_action,
            wraplength=390
        ).pack(
            fill=X
        )

        # =========================================================
        # LOGS
        # =========================================================

        lp = ttk.Labelframe(
            right,
            text=" LOGS DEL SISTEMA ",
            bootstyle="secondary",
            padding=5
        )

        lp.pack(
            fill=BOTH,
            expand=True
        )

        self.logbox = tk.Text(
            lp,
            height=8,
            wrap="word",
            font=(
                "Consolas",
                9
            )
        )

        self.logbox.pack(
            side=LEFT,
            fill=BOTH,
            expand=True
        )

        scroll = ttk.Scrollbar(
            lp,
            command=self.logbox.yview
        )

        scroll.pack(
            side=RIGHT,
            fill=Y
        )

        self.logbox.configure(
            yscrollcommand=scroll.set
        )

        self.log(
            "Interfaz iniciada."
        )

    # =============================================================
    # FILA DE ESTADO
    # =============================================================

    def row(
        self,
        parent,
        name,
        variable
    ):
        r = ttk.Frame(parent)

        r.pack(
            fill=X,
            pady=3
        )

        ttk.Label(
            r,
            text=name,
            font=(
                "Segoe UI",
                9,
                "bold"
            )
        ).pack(
            side=LEFT
        )

        ttk.Label(
            r,
            textvariable=variable,
            bootstyle="info",
            font=(
                "Segoe UI",
                9,
                "bold"
            )
        ).pack(
            side=RIGHT
        )

    # =============================================================
    # CONFIG ENTRY
    # =============================================================

    def config_entry(
        self,
        parent,
        label,
        value,
        row,
        password=False
    ):
        ttk.Label(
            parent,
            text=label
        ).grid(
            row=row,
            column=0,
            sticky=W,
            pady=4
        )

        entry = ttk.Entry(
            parent,
            show="*" if password else ""
        )

        entry.grid(
            row=row,
            column=1,
            sticky=EW,
            pady=4
        )

        entry.insert(
            0,
            str(value)
        )

        parent.columnconfigure(
            1,
            weight=1
        )

        return entry

    # =============================================================
    # CONFIGURACIÓN
    # =============================================================

    def open_configuration(self):

        win = ttk.Toplevel(
            self.root
        )

        win.title(
            "Configuración del sistema"
        )

        win.geometry(
            "760x820"
        )

        win.minsize(
            650,
            700
        )

        win.transient(
            self.root
        )

        win.grab_set()

        # ---------------------------------------------------------
        # SCROLL
        # ---------------------------------------------------------

        canvas = tk.Canvas(
            win,
            highlightthickness=0
        )

        scroll = ttk.Scrollbar(
            win,
            orient=VERTICAL,
            command=canvas.yview
        )

        content = ttk.Frame(
            canvas,
            padding=15
        )

        content.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas_window = canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw"
        )

        canvas.configure(
            yscrollcommand=scroll.set
        )

        canvas.pack(
            side=LEFT,
            fill=BOTH,
            expand=True
        )

        scroll.pack(
            side=RIGHT,
            fill=Y
        )

        # ---------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------

        ttk.Label(
            content,
            text="CONFIGURACIÓN DEL SISTEMA",
            font=(
                "Segoe UI",
                20,
                "bold"
            )
        ).pack(
            anchor=W,
            pady=(0, 12)
        )

        # =========================================================
        # CÁMARA
        # =========================================================

        camera = ttk.Labelframe(
            content,
            text=" CÁMARA AXIS Q7401 ",
            bootstyle="primary",
            padding=12
        )

        camera.pack(
            fill=X,
            pady=6
        )

        c = self.cfg["camera"]

        cam_ip = self.config_entry(
            camera,
            "IP:",
            c.get("ip", ""),
            0
        )

        cam_user = self.config_entry(
            camera,
            "Usuario:",
            c.get("username", ""),
            1
        )

        cam_pass = self.config_entry(
            camera,
            "Contraseña:",
            c.get("password", ""),
            2,
            True
        )

        cam_port = self.config_entry(
            camera,
            "Puerto RTSP:",
            c.get("rtsp_port", 554),
            3
        )

        cam_path = self.config_entry(
            camera,
            "RTSP Path:",
            c.get(
                "rtsp_path",
                "/axis-media/media.amp?videocodec=h264"
            ),
            4
        )

        ttk.Label(
            camera,
            text=(
                "Ejemplo:\n"
                "/axis-media/media.amp?videocodec=h264"
            ),
            bootstyle="secondary"
        ).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky=W,
            pady=5
        )

        # =========================================================
        # PUERTA
        # =========================================================

        door = ttk.Labelframe(
            content,
            text=" PUERTA / CONTROL HTTP ",
            bootstyle="warning",
            padding=12
        )

        door.pack(
            fill=X,
            pady=6
        )

        d = self.cfg["door"]

        door_ip = self.config_entry(
            door,
            "IP:",
            d.get("ip", ""),
            0
        )

        door_port = self.config_entry(
            door,
            "Puerto:",
            d.get("port", 80),
            1
        )

        endpoint = self.config_entry(
            door,
            "Endpoint abrir:",
            d.get(
                "open_endpoint",
                "/open"
            ),
            2
        )

        timeout = self.config_entry(
            door,
            "Timeout:",
            d.get(
                "timeout",
                5
            ),
            3
        )

        ttk.Label(
            door,
            text="La puerta no necesita usuario ni contraseña.",
            bootstyle="secondary"
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky=W,
            pady=5
        )

        # =========================================================
        # AUTOMATIZACIÓN
        # =========================================================

        auto = ttk.Labelframe(
            content,
            text=" AUTOMATIZACIÓN Y SEÑALES ",
            bootstyle="info",
            padding=12
        )

        auto.pack(
            fill=X,
            pady=6
        )

        a = self.cfg["automation"]

        # ---------------------------------------------------------
        # LÓGICA
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="Lógica de activación:"
        ).pack(
            anchor=W
        )

        logic = ttk.Combobox(
            auto,
            values=[
                "ANY",
                "ALL"
            ],
            state="readonly"
        )

        logic.set(
            a.get(
                "logic",
                "ANY"
            )
        )

        logic.pack(
            fill=X,
            pady=4
        )

        ttk.Label(
            auto,
            text=(
                "ANY = cualquier señal válida puede abrir.\n"
                "ALL = todas las señales activadas deben cumplirse."
            ),
            bootstyle="secondary"
        ).pack(
            anchor=W,
            pady=(0, 8)
        )

        # ---------------------------------------------------------
        # PERSONA
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="👤 PERSONA FRENTE A LA PUERTA",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor=W,
            pady=(5, 2)
        )

        ttk.Label(
            auto,
            text="Tiempo mínimo de presencia:"
        ).pack(
            anchor=W
        )

        standing = ttk.Spinbox(
            auto,
            from_=0.5,
            to=60,
            increment=0.5
        )

        standing.set(
            a.get(
                "standing_seconds",
                4
            )
        )

        standing.pack(
            fill=X,
            pady=3
        )

        # ---------------------------------------------------------
        # YOLO
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="Confianza YOLO:"
        ).pack(
            anchor=W,
            pady=(6, 2)
        )

        yolo_conf = ttk.Scale(
            auto,
            from_=0.10,
            to=0.95,
            value=float(
                a.get(
                    "yolo_confidence",
                    0.50
                )
            )
        )

        yolo_conf.pack(
            fill=X
        )

        yolo_conf_value = ttk.Label(
            auto,
            text=f"{float(yolo_conf.get()):.2f}"
        )

        yolo_conf_value.pack(
            anchor=E
        )

        def update_yolo_value(_event=None):
            yolo_conf_value.configure(
                text=f"{float(yolo_conf.get()):.2f}"
            )

        yolo_conf.configure(
            command=update_yolo_value
        )

        # ---------------------------------------------------------
        # MODELO YOLO
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="Modelo YOLO:"
        ).pack(
            anchor=W,
            pady=(6, 2)
        )

        yolo_model = ttk.Entry(
            auto
        )

        yolo_model.insert(
            0,
            a.get(
                "yolo_model",
                "yolo11n.pt"
            )
        )

        yolo_model.pack(
            fill=X
        )

        # ---------------------------------------------------------
        # ROSTRO
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="🙂 RECONOCIMIENTO FACIAL",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor=W,
            pady=(10, 2)
        )

        ttk.Label(
            auto,
            text="Umbral de reconocimiento:"
        ).pack(
            anchor=W
        )

        face_threshold = ttk.Scale(
            auto,
            from_=0.30,
            to=0.70,
            value=float(
                a.get(
                    "face_threshold",
                    0.48
                )
            )
        )

        face_threshold.pack(
            fill=X
        )

        face_value = ttk.Label(
            auto,
            text=f"{float(face_threshold.get()):.2f}"
        )

        face_value.pack(
            anchor=E
        )

        def update_face_value(_event=None):
            face_value.configure(
                text=f"{float(face_threshold.get()):.2f}"
            )

        face_threshold.configure(
            command=update_face_value
        )

        ttk.Label(
            auto,
            text=(
                "Un valor menor es más permisivo.\n"
                "Un valor mayor exige mayor similitud."
            ),
            bootstyle="secondary"
        ).pack(
            anchor=W
        )

        # ---------------------------------------------------------
        # SEÑAS
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="✋ RECONOCIMIENTO DE SEÑAS",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor=W,
            pady=(10, 2)
        )

        ttk.Label(
            auto,
            text="Seña autorizada:"
        ).pack(
            anchor=W
        )

        gesture = ttk.Combobox(
            auto,
            values=[
                "open_hand",
                "fist",
                "two_fingers"
            ],
            state="readonly"
        )

        gesture.set(
            a.get(
                "gesture",
                "open_hand"
            )
        )

        gesture.pack(
            fill=X,
            pady=3
        )

        ttk.Label(
            auto,
            text=(
                "open_hand = mano abierta\n"
                "fist = puño\n"
                "two_fingers = dos dedos"
            ),
            bootstyle="secondary",
            justify=LEFT
        ).pack(
            anchor=W
        )

        # ---------------------------------------------------------
        # COOLDOWN
        # ---------------------------------------------------------

        ttk.Label(
            auto,
            text="Cooldown de apertura:"
        ).pack(
            anchor=W,
            pady=(10, 2)
        )

        cooldown = ttk.Spinbox(
            auto,
            from_=1,
            to=120,
            increment=1
        )

        cooldown.set(
            a.get(
                "cooldown_seconds",
                8
            )
        )

        cooldown.pack(
            fill=X
        )

        # =========================================================
        # INFORMACIÓN
        # =========================================================

        info = ttk.Labelframe(
            content,
            text=" INFORMACIÓN ",
            bootstyle="secondary",
            padding=10
        )

        info.pack(
            fill=X,
            pady=6
        )

        ttk.Label(
            info,
            text=(
                "La apertura automática puede utilizar las señales:\n\n"
                "1. Persona frente a la puerta\n"
                "2. Rostro reconocido\n"
                "3. Seña autorizada\n\n"
                "La lógica ANY permite que cualquiera de las señales "
                "activadas autorice la apertura.\n\n"
                "La lógica ALL requiere que todas las señales "
                "activadas sean válidas."
            ),
            justify=LEFT,
            wraplength=650
        ).pack(
            fill=X
        )

        # =========================================================
        # GUARDAR
        # =========================================================

        buttons = ttk.Frame(
            content
        )

        buttons.pack(
            fill=X,
            pady=12
        )

        ttk.Button(
            buttons,
            text="CANCELAR",
            bootstyle="secondary",
            command=win.destroy
        ).pack(
            side=RIGHT,
            padx=5
        )

        def save():

            try:

                camera_ip = cam_ip.get().strip()
                camera_user = cam_user.get().strip()
                camera_password = cam_pass.get()

                camera_port = int(
                    cam_port.get()
                )

                camera_path = cam_path.get().strip()

                door_ip_value = door_ip.get().strip()

                door_port_value = int(
                    door_port.get()
                )

                endpoint_value = endpoint.get().strip()

                timeout_value = float(
                    timeout.get()
                )

                if not camera_ip:
                    raise ValueError(
                        "Indique la IP de la cámara."
                    )

                if not camera_path:
                    raise ValueError(
                        "Indique el RTSP Path."
                    )

                if not door_ip_value:
                    raise ValueError(
                        "Indique la IP de la puerta."
                    )

                if not endpoint_value:
                    raise ValueError(
                        "Indique el endpoint de apertura."
                    )

                # -------------------------------------------------
                # CÁMARA
                # -------------------------------------------------

                self.cfg["camera"] = {
                    "ip": camera_ip,
                    "username": camera_user,
                    "password": camera_password,
                    "rtsp_port": camera_port,
                    "rtsp_path": camera_path,
                    "transport": "tcp"
                }

                # -------------------------------------------------
                # PUERTA
                # -------------------------------------------------

                self.cfg["door"] = {
                    "ip": door_ip_value,
                    "port": door_port_value,
                    "open_endpoint": endpoint_value,
                    "method": "GET",
                    "timeout": timeout_value
                }

                # -------------------------------------------------
                # AUTOMATIZACIÓN
                # -------------------------------------------------

                self.cfg["automation"].update({

                    "logic": logic.get(),

                    "standing_seconds": float(
                        standing.get()
                    ),

                    "yolo_confidence": float(
                        yolo_conf.get()
                    ),

                    "face_threshold": float(
                        face_threshold.get()
                    ),

                    "gesture": gesture.get(),

                    "yolo_model": (
                        yolo_model.get().strip()
                        or "yolo11n.pt"
                    ),

                    "cooldown_seconds": float(
                        cooldown.get()
                    )
                })

                save_config(
                    self.cfg
                )

                # -------------------------------------------------
                # ACTUALIZAR CONTROLADORES
                # -------------------------------------------------

                try:
                    self.door_ctl.update_config(
                        self.cfg
                    )
                except Exception as exc:
                    self.log(
                        f"Error actualizando puerta: {exc}"
                    )

                try:
                    self.camera_ctl.config = self.cfg
                except Exception as exc:
                    self.log(
                        f"Error actualizando cámara: {exc}"
                    )

                if self.engine:

                    try:
                        self.engine.cfg = self.cfg
                    except Exception:
                        pass

                    try:
                        self.engine.reload_faces()
                    except Exception:
                        pass

                self.update_connection_info()

                self.log(
                    "Configuración guardada correctamente."
                )

                self.status.set(
                    "CONFIGURACIÓN GUARDADA"
                )

                messagebox.showinfo(
                    "Configuración",
                    "Configuración guardada correctamente.",
                    parent=win
                )

                win.destroy()

            except Exception as exc:

                messagebox.showerror(
                    "Error",
                    str(exc),
                    parent=win
                )

        ttk.Button(
            buttons,
            text="💾 GUARDAR CONFIGURACIÓN",
            bootstyle="success",
            command=save
        ).pack(
            side=RIGHT
        )

    # =============================================================
    # INFORMACIÓN DE CONEXIONES
    # =============================================================

    def update_connection_info(self):

        c = self.cfg.get(
            "camera",
            {}
        )

        d = self.cfg.get(
            "door",
            {}
        )

        self.connection_info.set(
            f"AXIS RTSP: "
            f"{c.get('ip', '')}:"
            f"{c.get('rtsp_port', 554)}\n"
            f"PUERTA HTTP: "
            f"{d.get('ip', '')}:"
            f"{d.get('port', 80)}\n"
            f"GET "
            f"{d.get('open_endpoint', '')}"
        )

    # =============================================================
    # CÁMARA AXIS
    # =============================================================

    def start_camera(self):

        if self.camera_running:
            return

        self.camera_source = "AXIS"

        self.log(
            "Conectando AXIS Q7401..."
        )

        self._prepare_camera_start(
            "AXIS"
        )

        try:

            self.camera_ctl.config = self.cfg

            self.camera_ctl.start()

        except Exception as exc:

            self.log(
                f"Error iniciando AXIS: {exc}"
            )

            self.camera_running = False

            self.cam.set(
                "ERROR"
            )

            self.connect_btn.configure(
                state="normal"
            )

            self.local_btn.configure(
                state="normal"
            )

    # =============================================================
    # CÁMARA LOCAL
    # =============================================================

    def start_local_camera(self):

        if self.camera_running:
            return

        self.log(
            "Iniciando cámara local..."
        )

        self.camera_source = "LOCAL"

        self._prepare_camera_start(
            "CÁMARA LOCAL"
        )

        threading.Thread(
            target=self.local_camera_worker,
            daemon=True
        ).start()

    # =============================================================
    # PREPARAR CÁMARA
    # =============================================================

    def _prepare_camera_start(
        self,
        label
    ):

        self.camera_running = True

        self.cam.set(
            "CONECTANDO..."
        )

        self.status.set(
            f"CONECTANDO {label}"
        )

        self.connect_btn.configure(
            state="disabled"
        )

        self.local_btn.configure(
            state="disabled"
        )

    # =============================================================
    # FRAME DE CAMERA CONTROLLER
    # =============================================================

    def on_camera_frame(
        self,
        frame
    ):

        if not self.camera_running:
            return

        self.frame = frame

        if self.camera_source != "AXIS":
            return

        # ---------------------------------------------
        # AUTOMATIZACIÓN
        # ---------------------------------------------

        if self.engine:

            try:
                self.engine.update(
                    frame
                )

            except Exception as exc:

                self.log(
                    f"Automatización: {exc}"
                )

        # ---------------------------------------------
        # UI
        # ---------------------------------------------

        try:
            self.root.after(
                0,
                self.show_frame
            )
        except Exception:
            pass

        if self.cam.get() == "CONECTANDO...":

            try:
                self.root.after(
                    0,
                    self.camera_connected
                )
            except Exception:
                pass

    # =============================================================
    # CÁMARA LOCAL
    # =============================================================

    def local_camera_worker(self):

        cap = None

        try:

            cap = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

            if not cap.isOpened():

                self.root.after(
                    0,
                    self.local_camera_error
                )

                return

            self.root.after(
                0,
                self.camera_connected
            )

            while self.camera_running:

                ok, frame = cap.read()

                if not ok:

                    self.log(
                        "No se recibió imagen de cámara local."
                    )

                    time.sleep(
                        0.3
                    )

                    continue

                self.frame = frame

                if self.engine:

                    try:

                        self.engine.update(
                            frame
                        )

                    except Exception as exc:

                        self.log(
                            f"Automatización: {exc}"
                        )

                try:

                    self.root.after(
                        0,
                        self.show_frame
                    )

                except Exception:
                    break

        except Exception as exc:

            self.log(
                f"Error cámara local: {exc}"
            )

            self.root.after(
                0,
                self.local_camera_error
            )

        finally:

            if cap:

                try:
                    cap.release()
                except Exception:
                    pass

    # =============================================================
    # CÁMARA CONECTADA
    # =============================================================

    def camera_connected(self):

        if self.camera_source == "AXIS":

            self.cam.set(
                "CONECTADA AXIS"
            )

            self.status.set(
                "SISTEMA OPERATIVO"
            )

            self.log(
                "AXIS Q7401 conectada."
            )

        else:

            self.cam.set(
                "CÁMARA LOCAL"
            )

            self.status.set(
                "SISTEMA OPERATIVO"
            )

            self.log(
                "Cámara local conectada."
            )

    # =============================================================
    # ERROR LOCAL
    # =============================================================

    def local_camera_error(self):

        self.camera_running = False

        self.cam.set(
            "ERROR"
        )

        self.status.set(
            "ERROR CÁMARA"
        )

        self.connect_btn.configure(
            state="normal"
        )

        self.local_btn.configure(
            state="normal"
        )

    # =============================================================
    # MOSTRAR VIDEO
    # =============================================================

    def show_frame(self):
        """Muestra el frame de la cámara con overlays de automatización."""

        if self.frame is None:
            return

        try:

            frame = cv2.cvtColor(
                self.frame.copy(),
                cv2.COLOR_BGR2RGB
            )

            # =====================================================
            # DIBUJAR OVERLAY DE SEÑA (YOLO POSE)
            # =====================================================

            if self.engine and self.gesture_var.get():
                status = self.engine.get_status()
                
                bbox = status.get("hand_bbox")
                gesture = status.get("current_gesture")
                confidence = status.get("gesture_confidence", 0.0)
                detected = status.get("gesture_detected", False)

                if bbox and gesture:
                    x1, y1, x2, y2 = bbox
                    
                    # Color según estado
                    if detected:
                        color = (0, 255, 0)  # Verde: gesto confirmado
                    else:
                        color = (0, 165, 255)  # Naranja: detectando

                    # Dibujar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Texto con información del gesto
                    label = f"{gesture}: {confidence:.2f}"
                    
                    # Fondo del texto
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_size[1] - 10),
                        (x1 + text_size[0], y1),
                        color,
                        -1
                    )

                    # Texto
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    # Mostrar fidelidad de la señal
                    fidelity_label = f"Fidelidad: {confidence*100:.1f}%"
                    cv2.putText(
                        frame,
                        fidelity_label,
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            image = Image.fromarray(
                frame
            )

            w = max(
                self.video.winfo_width() - 15,
                400
            )

            h = max(
                self.video.winfo_height() - 15,
                300
            )

            image.thumbnail(
                (
                    w,
                    h
                ),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                image
            )

            self.video.configure(
                image=photo,
                text=""
            )

            self.video.image = photo

        except Exception as exc:

            self.log(
                f"Error mostrando video: {exc}"
            )

    # =============================================================
    # DETENER CÁMARA
    # =============================================================

    def stop_camera(self):

        self.camera_running = False

        try:
            self.camera_ctl.stop()
        except Exception:
            pass

        self.cam.set(
            "DESCONECTADA"
        )

        self.status.set(
            "SISTEMA LISTO"
        )

        self.connect_btn.configure(
            state="normal"
        )

        self.local_btn.configure(
            state="normal"
        )

        self.frame = None

        self.video.configure(
            image="",
            text=(
                "CÁMARA DESCONECTADA\n\n"
                "Seleccione AXIS o CÁMARA LOCAL"
            )
        )

        self.video.image = None

        self.log(
            "Cámara desconectada."
        )

    # =============================================================
    # PUERTA
    # =============================================================

    def open_door(
        self,
        source="MANUAL"
    ):

        if self.door_busy:
            self.log(
                "Apertura ignorada: puerta ocupada."
            )
            return

        self.door_busy = True

        try:

            self.openbtn.configure(
                state="disabled",
                text="⏳\n\nABRIENDO..."
            )

        except Exception:
            pass

        self.door.set(
            "ABRIENDO..."
        )

        self.http_status.set(
            "HTTP: ESPERANDO"
        )

        self.status.set(
            f"ABRIENDO ({source})"
        )

        self.log(
            f"Solicitud apertura: {source}"
        )

        threading.Thread(
            target=self.open_door_worker,
            args=(source,),
            daemon=True
        ).start()

    # =============================================================
    # WORKER PUERTA
    # =============================================================

    def open_door_worker(
        self,
        source
    ):

        try:

            code, text = self.door_ctl.open()

            self.root.after(
                0,
                self.door_done,
                code,
                text,
                source
            )

        except Exception as exc:

            self.root.after(
                0,
                self.door_done,
                None,
                str(exc),
                source
            )

    # =============================================================
    # RESULTADO PUERTA
    # =============================================================

    def door_done(
        self,
        code,
        text,
        source
    ):

        self.door_busy = False

        self.openbtn.configure(
            state="normal",
            text="🔓\n\nABRIR PUERTA"
        )

        if code and 200 <= code < 300:

            self.door.set(
                "ABIERTA"
            )

            self.http_status.set(
                f"HTTP: {code} OK"
            )

            self.status.set(
                "PUERTA ABIERTA"
            )

            self.last_action.set(
                f"Última operación: APERTURA\n"
                f"HTTP {code}\n"
                f"Origen: {source}"
            )

            self.log(
                f"APERTURA CORRECTA | "
                f"HTTP {code} | {source}"
            )

        else:

            self.door.set(
                "ERROR"
            )

            self.http_status.set(
                f"HTTP: "
                f"{code if code else 'ERROR'}"
            )

            self.status.set(
                "ERROR PUERTA"
            )

            self.last_action.set(
                f"Última operación: ERROR\n"
                f"HTTP {code}\n"
                f"Origen: {source}"
            )

            self.log(
                f"ERROR APERTURA | "
                f"HTTP {code} | {source}"
            )

        if text:

            self.log(
                f"Respuesta: {str(text)[:300]}"
            )

    # =============================================================
    # GUARDAR AUTOMATIZACIÓN
    # =============================================================

    def save_auto(self):
        """Guarda la configuración de automatización."""

        try:

            automation_cfg = self.cfg.setdefault(
                "automation",
                {}
            )

            automation_cfg.update({

                "enabled":
                    self.auto_var.get(),

                "gesture_enabled":
                    self.gesture_var.get(),

                "gesture":
                    self.gesture_combo.get(),

                "gesture_stable_seconds":
                    float(self.gesture_sec.get()),

                "logic":
                    self.logic_combo.get()
            })

            save_config(
                self.cfg
            )

            if self.engine:

                try:
                    self.engine.cfg = self.cfg
                except Exception:
                    pass

            self.update_automation_status()

            self.log(
                "Automatización guardada."
            )

        except Exception as exc:

            self.log(
                f"Error automatización: {exc}"
            )

    # =============================================================
    # ESTADO AUTOMATIZACIÓN
    # =============================================================

    def update_automation_status(self):

        if self.auto_var.get():

            self.auto.set(
                "AUTOMATIZACIÓN: ON"
            )

        else:

            self.auto.set(
                "AUTOMATIZACIÓN: OFF"
            )

    # =============================================================
    # GESTIÓN DE CARAS
    # =============================================================

    def face_manager(self):

        if not self.engine:

            messagebox.showerror(
                "Reconocimiento facial",
                "AutomationEngine no está disponible."
            )

            return

        win = ttk.Toplevel(
            self.root
        )

        win.title(
            "Registro facial"
        )

        win.geometry(
            "700x700"
        )

        win.minsize(
            600,
            600
        )

        win.transient(
            self.root
        )

        win.grab_set()

        main = ttk.Frame(
            win,
            padding=12
        )

        main.pack(
            fill=BOTH,
            expand=True
        )

        ttk.Label(
            main,
            text="REGISTRO DE ROSTROS",
            font=(
                "Segoe UI",
                20,
                "bold"
            )
        ).pack(
            anchor=W
        )

        ttk.Label(
            main,
            text=(
                "1. Conecte AXIS o cámara local.\n"
                "2. Coloque a la persona frente a la cámara.\n"
                "3. Escriba el nombre.\n"
                "4. Capture el rostro."
            ),
            bootstyle="secondary",
            justify=LEFT
        ).pack(
            anchor=W,
            pady=8
        )

        # ---------------------------------------------------------
        # NOMBRE
        # ---------------------------------------------------------

        ttk.Label(
            main,
            text="Nombre de la persona:"
        ).pack(
            anchor=W
        )

        name = ttk.Entry(
            main
        )

        name.pack(
            fill=X,
            pady=5
        )

        # ---------------------------------------------------------
        # PREVIEW
        # ---------------------------------------------------------

        preview_frame = ttk.Labelframe(
            main,
            text=" CAPTURA ACTUAL ",
            bootstyle="primary",
            padding=5
        )

        preview_frame.pack(
            fill=BOTH,
            expand=True,
            pady=6
        )

        preview = ttk.Label(
            preview_frame,
            text="SIN IMAGEN",
            anchor=CENTER,
            font=(
                "Segoe UI",
                12,
                "bold"
            )
        )

        preview.pack(
            fill=BOTH,
            expand=True
        )

        # ---------------------------------------------------------
        # ACTUALIZAR PREVIEW
        # ---------------------------------------------------------

        def refresh():

            try:

                if not win.winfo_exists():
                    return

            except Exception:
                return

            if self.frame is not None:

                try:

                    rgb = cv2.cvtColor(
                        self.frame.copy(),
                        cv2.COLOR_BGR2RGB
                    )

                    image = Image.fromarray(
                        rgb
                    )

                    image.thumbnail(
                        (
                            620,
                            360
                        ),
                        Image.Resampling.LANCZOS
                    )

                    photo = ImageTk.PhotoImage(
                        image
                    )

                    preview.configure(
                        image=photo,
                        text=""
                    )

                    preview.image = photo

                except Exception:
                    pass

            try:

                win.after(
                    150,
                    refresh
                )

            except Exception:
                pass

        # ---------------------------------------------------------
        # CAPTURAR
        # ---------------------------------------------------------

        def capture():

            person = name.get().strip()

            if not person:

                messagebox.showwarning(
                    "Registro",
                    "Escriba el nombre de la persona.",
                    parent=win
                )

                return

            if self.frame is None:

                messagebox.showwarning(
                    "Registro",
                    "No existe una imagen de cámara.",
                    parent=win
                )

                return

            try:

                ok, msg = self.engine.register_face(
                    person,
                    self.frame.copy()
                )

                if ok:

                    self.log(
                        msg
                    )

                    messagebox.showinfo(
                        "Registro facial",
                        msg,
                        parent=win
                    )

                    refresh_list()

                else:

                    messagebox.showerror(
                        "Registro facial",
                        msg,
                        parent=win
                    )

            except Exception as exc:

                messagebox.showerror(
                    "Registro facial",
                    str(exc),
                    parent=win
                )

        ttk.Button(
            main,
            text="📷 CAPTURAR ROSTRO ACTUAL",
            bootstyle="success",
            command=capture
        ).pack(
            fill=X,
            pady=5
        )

        # ---------------------------------------------------------
        # LISTA
        # ---------------------------------------------------------

        ttk.Label(
            main,
            text="Personas registradas:",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor=W,
            pady=(8, 3)
        )

        listbox = tk.Listbox(
            main,
            height=6
        )

        listbox.pack(
            fill=X
        )

        # ---------------------------------------------------------
        # ACTUALIZAR LISTA
        # ---------------------------------------------------------

        def refresh_list():

            listbox.delete(
                0,
                tk.END
            )

            try:

                people = (
                    self.engine.list_registered_faces()
                )

                for person in people:

                    listbox.insert(
                        tk.END,
                        person
                    )

            except Exception as exc:

                self.log(
                    f"Error listando caras: {exc}"
                )

        # ---------------------------------------------------------
        # ELIMINAR
        # ---------------------------------------------------------

        def delete():

            selection = listbox.curselection()

            if not selection:
                return

            person = listbox.get(
                selection[0]
            )

            confirm = messagebox.askyesno(
                "Eliminar",
                f"¿Eliminar todos los rostros de {person}?",
                parent=win
            )

            if not confirm:
                return

            try:

                result = self.engine.delete_face(
                    person
                )

                if result:

                    self.log(
                        f"Rostro eliminado: {person}"
                    )

                    refresh_list()

                    messagebox.showinfo(
                        "Reconocimiento facial",
                        f"Se eliminó {person}.",
                        parent=win
                    )

                else:

                    messagebox.showerror(
                        "Reconocimiento facial",
                        "No se pudo eliminar la persona.",
                        parent=win
                    )

            except Exception as exc:

                messagebox.showerror(
                    "Reconocimiento facial",
                    str(exc),
                    parent=win
                )

        ttk.Button(
            main,
            text="🗑 ELIMINAR PERSONA",
            bootstyle="danger",
            command=delete
        ).pack(
            fill=X,
            pady=5
        )

        refresh_list()

        refresh()

    # =============================================================
    # LOG
    # =============================================================

    def log(
        self,
        message
    ):

        def append():

            try:

                ts = time.strftime(
                    "%H:%M:%S"
                )

                self.logbox.insert(
                    "end",
                    f"[{ts}] {message}\n"
                )

                self.logbox.see(
                    "end"
                )

            except Exception:
                pass

        try:

            self.root.after(
                0,
                append
            )

        except Exception:
            pass

    # =============================================================
    # CERRAR
    # =============================================================

    def on_close(self):

        self.camera_running = False

        try:
            self.camera_ctl.stop()
        except Exception:
            pass

        try:
            save_config(
                self.cfg
            )
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass


# =================================================================
# MAIN
# =================================================================

def main():

    root = ttk.Window(
        title="Control de Acceso",
        themename="darkly"
    )

    root.geometry(
        "1500x900"
    )

    root.minsize(
        1100,
        700
    )

    App(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()