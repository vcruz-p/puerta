"""
Interfaz gráfica principal de Control de Acceso.

Aplicación de escritorio en Python con Tkinter moderno (ttkbootstrap).
"""

import tkinter as tk
import threading
import time
import cv2
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import messagebox


class MainWindow:
    """
    Ventana principal de la aplicación.
    
    Gestiona la interfaz gráfica, controles de cámara, puerta
    y automatización.
    """

    def __init__(self, root, cfg, door_controller, camera_controller, automation_engine=None):
        """
        Inicializa la ventana principal.
        
        Args:
            root: Instancia de ttkbootstrap.Window.
            cfg (dict): Configuración del sistema.
            door_controller: Instancia de DoorController.
            camera_controller: Instancia de CameraController.
            automation_engine: Instancia opcional de AutomationEngine.
        """
        self.root = root
        self.cfg = cfg
        self.door_ctl = door_controller
        self.camera_ctl = camera_controller
        self.engine = automation_engine

        # Variables de estado
        self.frame = None
        self.camera_running = False
        self.camera_source = "AXIS"
        self.door_busy = False

        self.status = ttk.StringVar(value="SISTEMA LISTO")
        self.cam = ttk.StringVar(value="DESCONECTADA")
        self.door = ttk.StringVar(value="LISTA")
        self.http_status = ttk.StringVar(value="HTTP: ---")
        self.auto = ttk.StringVar(value="AUTOMATIZACIÓN: OFF")
        self.last_action = ttk.StringVar(value="Última operación: ---")
        self.connection_info = ttk.StringVar(value="")

        # Variables de automatización
        auto_cfg = cfg.get("automation", {})
        self.auto_var = tk.BooleanVar(value=auto_cfg.get("enabled", False))
        self.gesture_var = tk.BooleanVar(value=auto_cfg.get("gesture_enabled", False))

        # Construir interfaz
        self.build()

    def build(self):
        """Construye la interfaz gráfica principal."""
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=BOTH, expand=True)

        # HEADER
        header = ttk.Frame(main)
        header.pack(fill=X, pady=(0, 10))

        title = ttk.Frame(header)
        title.pack(side=LEFT)

        ttk.Label(
            title,
            text="CONTROL DE ACCESO",
            font=("Segoe UI", 24, "bold")
        ).pack(anchor=W)

        ttk.Label(
            title,
            text="AXIS Q7401 • PERSONA • ROSTRO • SEÑAS",
            bootstyle="secondary"
        ).pack(anchor=W)

        ttk.Button(
            header,
            text="⚙ CONFIGURACIÓN",
            bootstyle="secondary",
            command=self.open_configuration
        ).pack(side=RIGHT)

        ttk.Label(
            header,
            textvariable=self.status,
            bootstyle="info",
            font=("Segoe UI", 11, "bold")
        ).pack(side=RIGHT, padx=12)

        # BODY
        body = ttk.Panedwindow(main, orient=HORIZONTAL)
        body.pack(fill=BOTH, expand=True)

        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        right = ttk.Frame(body, padding=(8, 0, 0, 0))

        body.add(left, weight=5)
        body.add(right, weight=3)

        # VIDEO
        vp = ttk.Labelframe(left, text=" MONITOREO EN VIVO ", bootstyle="primary", padding=5)
        vp.pack(fill=BOTH, expand=True)

        self.video = ttk.Label(
            vp,
            text="CÁMARA DESCONECTADA\n\nSeleccione AXIS o CÁMARA LOCAL",
            anchor=CENTER,
            justify=CENTER,
            font=("Segoe UI", 18, "bold")
        )
        self.video.pack(fill=BOTH, expand=True)

        # BARRA CÁMARA
        camera_bar = ttk.Frame(left)
        camera_bar.pack(fill=X, pady=7)

        self.connect_btn = ttk.Button(
            camera_bar,
            text="▶ AXIS",
            bootstyle="success",
            command=self.start_camera
        )
        self.connect_btn.pack(side=LEFT)

        self.local_btn = ttk.Button(
            camera_bar,
            text="💻 CÁMARA LOCAL",
            bootstyle="info",
            command=self.start_local_camera
        )
        self.local_btn.pack(side=LEFT, padx=5)

        ttk.Button(
            camera_bar,
            text="■ DESCONECTAR",
            bootstyle="secondary",
            command=self.stop_camera
        ).pack(side=LEFT)

        ttk.Label(
            camera_bar,
            textvariable=self.cam,
            bootstyle="info",
            font=("Segoe UI", 10, "bold")
        ).pack(side=RIGHT)

        # ESTADO
        sp = ttk.Labelframe(right, text=" ESTADO DEL SISTEMA ", bootstyle="info", padding=10)
        sp.pack(fill=X, pady=(0, 8))

        self._row(sp, "CÁMARA", self.cam)
        self._row(sp, "PUERTA", self.door)
        self._row(sp, "HTTP", self.http_status)
        self._row(sp, "AUTOMÁTICO", self.auto)

        ttk.Separator(sp).pack(fill=X, pady=6)

        ttk.Label(
            sp,
            textvariable=self.connection_info,
            bootstyle="secondary",
            wraplength=380,
            justify=LEFT
        ).pack(fill=X)

        # CONTROL PUERTA
        dp = ttk.Labelframe(right, text=" CONTROL DE PUERTA ", bootstyle="warning", padding=12)
        dp.pack(fill=X, pady=(0, 8))

        self.openbtn = ttk.Button(
            dp,
            text="🔓\n\nABRIR PUERTA",
            bootstyle="success",
            command=self.open_door
        )
        self.openbtn.pack(fill=X, ipady=22)

        ttk.Label(dp, text="Apertura mediante HTTP", bootstyle="secondary").pack(pady=(6, 0))

        # AUTOMATIZACIÓN
        ap = ttk.Labelframe(right, text=" AUTOMATIZACIÓN ", bootstyle="primary", padding=10)
        ap.pack(fill=X, pady=(0, 8))

        ttk.Checkbutton(
            ap,
            text="Activar automatización",
            variable=self.auto_var,
            command=self.save_auto,
            bootstyle="success"
        ).pack(anchor=W, pady=2)

        ttk.Label(ap, text="Lógica de señales:").pack(anchor=W, pady=(5, 2))

        self.logic_combo = ttk.Combobox(ap, values=["ANY", "ALL"], state="readonly")
        self.logic_combo.set(self.cfg.get("automation", {}).get("logic", "ANY"))
        self.logic_combo.pack(fill=X)

        # SEÑAS CON YOLO POSE
        ttk.Label(
            ap,
            text="👋 Reconocimiento de Señas (YOLO Pose)",
            bootstyle="primary"
        ).pack(anchor=W, pady=(5, 3))

        ttk.Checkbutton(
            ap,
            text="✋ Activar reconocimiento de seña",
            variable=self.gesture_var,
            command=self.save_auto,
            bootstyle="success"
        ).pack(anchor=W, pady=2)

        ttk.Label(ap, text="Gesto para activar:").pack(anchor=W, pady=(8, 2))

        self.gesture_combo = ttk.Combobox(
            ap,
            values=["open_hand", "fist", "two_fingers", "victory", "thumbs_up", "pointing"],
            state="readonly"
        )
        self.gesture_combo.set(self.cfg.get("automation", {}).get("gesture", "open_hand"))
        self.gesture_combo.pack(fill=X)

        ttk.Label(ap, text="Tiempo de estabilidad (segundos):").pack(anchor=W, pady=(8, 2))

        self.gesture_sec = ttk.Spinbox(ap, from_=0.1, to=2.0, increment=0.05)
        self.gesture_sec.set(self.cfg.get("automation", {}).get("gesture_stable_seconds", 0.25))
        self.gesture_sec.pack(fill=X)

        ttk.Button(
            ap,
            text="💾 GUARDAR AUTOMATIZACIÓN",
            bootstyle="primary",
            command=self.save_auto
        ).pack(fill=X, pady=(10, 0))

        # ÚLTIMA OPERACIÓN
        action = ttk.Labelframe(right, text=" ÚLTIMA OPERACIÓN ", bootstyle="secondary", padding=7)
        action.pack(fill=X, pady=(0, 8))

        ttk.Label(action, textvariable=self.last_action, wraplength=390).pack(fill=X)

        # LOGS
        lp = ttk.Labelframe(right, text=" LOGS DEL SISTEMA ", bootstyle="secondary", padding=5)
        lp.pack(fill=BOTH, expand=True)

        self.logbox = tk.Text(lp, height=8, wrap="word", font=("Consolas", 9))
        self.logbox.pack(side=LEFT, fill=BOTH, expand=True)

        scroll = ttk.Scrollbar(lp, command=self.logbox.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.logbox.configure(yscrollcommand=scroll.set)

        self.log("Interfaz iniciada.")

    def _row(self, parent, name, variable):
        """Crea una fila de estado."""
        r = ttk.Frame(parent)
        r.pack(fill=X, pady=3)

        ttk.Label(r, text=f"{name}:", width=10, anchor=W).pack(side=LEFT)
        ttk.Label(r, textvariable=variable, bootstyle="info").pack(side=LEFT)

    def start_camera(self):
        """Inicia la cámara AXIS."""
        self.camera_source = "AXIS"
        self.camera_ctl.start("AXIS")

    def start_local_camera(self):
        """Inicia la cámara local."""
        self.camera_source = 0
        self.camera_ctl.start(0)

    def stop_camera(self):
        """Detiene la cámara."""
        self.camera_ctl.stop()

    def on_camera_frame(self, frame):
        """Callback para cada frame de cámara."""
        self.frame = frame

        try:
            rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((900, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            self.video.configure(image=photo, text="")
            self.video.image = photo

        except Exception as exc:
            self.log(f"Error mostrando frame: {exc}")

        # Automatización
        if self.engine and self.auto_var.get():
            try:
                detected, gesture_name, confidence, bbox = self.engine.update(frame)

                if detected:
                    self.open_door(source=f"AUTOMÁTICO: {gesture_name}")

            except Exception as exc:
                self.log(f"Error en automatización: {exc}")

    def open_door(self, source="MANUAL"):
        """Abre la puerta."""
        if self.door_busy:
            return

        self.door_busy = True
        self.door.set("OCUPADA")
        self.status.set("ABRIENDO PUERTA...")

        def worker():
            try:
                code, text = self.door_ctl.open()

                if code == 200:
                    self.root.after(0, lambda: self._on_open_success(code, text, source))
                else:
                    self.root.after(0, lambda: self._on_open_error(code, text, source))

            except Exception as exc:
                self.root.after(0, lambda: self._on_open_error("ERROR", str(exc), source))

            finally:
                self.root.after(0, self._reset_door_state)

        threading.Thread(target=worker, daemon=True).start()

    def _on_open_success(self, code, text, source):
        self.door.set("LISTA")
        self.http_status.set(f"HTTP: {code}")
        self.status.set("PUERTA ABIERTA")
        self.last_action.set(f"Última operación: ÉXITO\nHTTP {code}\nOrigen: {source}")
        self.log(f"APERTURA CORRECTA | HTTP {code} | {source}")

        if text:
            self.log(f"Respuesta: {str(text)[:300]}")

    def _on_open_error(self, code, text, source):
        self.door.set("ERROR")
        self.http_status.set(f"HTTP: {code if code else 'ERROR'}")
        self.status.set("ERROR PUERTA")
        self.last_action.set(f"Última operación: ERROR\nHTTP {code}\nOrigen: {source}")
        self.log(f"ERROR APERTURA | HTTP {code} | {source}")

        if text:
            self.log(f"Respuesta: {str(text)[:300]}")

    def _reset_door_state(self):
        self.door_busy = False
        self.door.set("LISTA")

    def save_auto(self):
        """Guarda la configuración de automatización."""
        try:
            automation_cfg = self.cfg.setdefault("automation", {})
            automation_cfg.update({
                "enabled": self.auto_var.get(),
                "gesture_enabled": self.gesture_var.get(),
                "gesture": self.gesture_combo.get(),
                "gesture_stable_seconds": float(self.gesture_sec.get()),
                "logic": self.logic_combo.get()
            })

            from src.utils.config_manager import save_config
            save_config(self.cfg)

            if self.engine:
                try:
                    self.engine.cfg = self.cfg
                except Exception:
                    pass

            self.update_automation_status()
            self.log("Automatización guardada.")

        except Exception as exc:
            self.log(f"Error automatización: {exc}")

    def update_automation_status(self):
        """Actualiza el estado de automatización."""
        if self.auto_var.get():
            self.auto.set("AUTOMATIZACIÓN: ON")
        else:
            self.auto.set("AUTOMATIZACIÓN: OFF")

    def open_configuration(self):
        """Abre la ventana de configuración."""
        messagebox.showinfo("Configuración", "Funcionalidad pendiente de implementar", parent=self.root)

    def log(self, message):
        """Agrega un mensaje al log."""
        def append():
            try:
                ts = time.strftime("%H:%M:%S")
                self.logbox.insert("end", f"[{ts}] {message}\n")
                self.logbox.see("end")
            except Exception:
                pass

        try:
            self.root.after(0, append)
        except Exception:
            pass
