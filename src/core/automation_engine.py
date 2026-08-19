"""
Motor de automatización para control de acceso.

Soporta:
    - Detección de gestos con YOLO Pose
    - Reconocimiento facial (opcional)
    - Lógica de señales ANY/ALL
"""

import time
import cv2
import numpy as np
from pathlib import Path


class AutomationEngine:
    """
    Motor de automatización para control de acceso.

    Señales disponibles:
        SEÑA (YOLO Pose)
            Usa YOLO11n-pose para detectar manos y gestos:
                - open_hand (mano abierta)
                - fist (puño)
                - two_fingers (dos dedos)
                - victory (signo V)
                - thumbs_up (pulgar arriba)
                - pointing (dedo índice señalando)
    """

    def __init__(self, cfg, open_door_callback, logger=None):
        """
        Inicializa el motor de automatización.
        
        Args:
            cfg (dict): Configuración completa del sistema.
            open_door_callback (callable): Función para abrir puerta.
            logger (callable, optional): Función de logging.
        """
        self.cfg = cfg or {}
        self.open_door = open_door_callback
        self.log = logger or (lambda msg: None)

        # ESTADO SEÑAS
        self.current_gesture = None
        self.last_gesture = None
        self.gesture_stable_since = None
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.hand_bbox = None

        # AUTOMATIZACIÓN
        self.last_trigger = 0.0
        self.last_trigger_source = None

        # MODELO YOLO POSE
        self.yolo_pose = None
        self.model_path = None

        # CONTROL DE PROCESAMIENTO
        self.frame_counter = 0
        self.last_gesture_process = 0.0
        self.gesture_interval = 0.08

        # CARGAR MODELO
        self._load_yolo_pose()

    def _automation_cfg(self):
        """Devuelve el bloque de configuración de automatización."""
        if not isinstance(self.cfg, dict):
            self.cfg = {}

        automation = self.cfg.get("automation")
        if not isinstance(automation, dict):
            automation = {}
            self.cfg["automation"] = automation

        return automation

    def _load_yolo_pose(self):
        """Carga el modelo YOLO Pose."""
        self.yolo_pose = None

        try:
            from ultralytics import YOLO

            automation = self._automation_cfg()
            self.model_path = automation.get("yolo_pose_model", "yolo11n-pose.pt")

            self.yolo_pose = YOLO(self.model_path)
            self.yolo_pose.to("cpu")

            self.log(f"YOLO Pose cargado: {self.model_path}")

        except Exception as exc:
            self.log(f"YOLO Pose no disponible: {exc}")
            self.yolo_pose = None

    def _gesture_detected(self, frame):
        """
        Detecta gestos de mano usando YOLO Pose.
        
        Returns:
            tuple: (detectado, nombre_gesto, confianza, bbox)
        """
        if self.yolo_pose is None or frame is None:
            return False, None, 0.0, None

        try:
            results = self.yolo_pose(frame, verbose=False, conf=0.4, iou=0.45)

            if not results or len(results) == 0:
                self.gesture_detected = False
                self.current_gesture = None
                self.gesture_confidence = 0.0
                self.hand_bbox = None
                return False, None, 0.0, None

            result = results[0]

            if not hasattr(result, 'keypoints') or result.keypoints is None:
                self.gesture_detected = False
                self.current_gesture = None
                self.gesture_confidence = 0.0
                self.hand_bbox = None
                return False, None, 0.0, None

            for i in range(len(result.keypoints)):
                kp = result.keypoints[i]

                if kp.xy is None or len(kp.xy[0]) < 21:
                    continue

                points = kp.xy[0].cpu().numpy()
                confidences = kp.conf[0].cpu().numpy() if hasattr(kp, 'conf') else None

                if confidences is not None:
                    avg_conf = float(np.mean(confidences[:5]))
                else:
                    avg_conf = 0.8

                gesture, confidence = self._classify_gesture(points, avg_conf)

                if gesture:
                    x_min = np.min(points[:, 0])
                    y_min = np.min(points[:, 1])
                    x_max = np.max(points[:, 0])
                    y_max = np.max(points[:, 1])

                    padding = 20
                    bbox = [
                        max(0, int(x_min - padding)),
                        max(0, int(y_min - padding)),
                        min(frame.shape[1], int(x_max + padding)),
                        min(frame.shape[0], int(y_max + padding))
                    ]

                    now = time.monotonic()

                    if self.current_gesture != gesture:
                        self.current_gesture = gesture
                        self.gesture_stable_since = now
                        self.gesture_detected = False
                        self.gesture_confidence = confidence
                        self.hand_bbox = bbox
                        return False, gesture, confidence, bbox

                    if self.gesture_stable_since is None:
                        self.gesture_stable_since = now
                        self.gesture_detected = False
                        self.gesture_confidence = confidence
                        self.hand_bbox = bbox
                        return False, gesture, confidence, bbox

                    stable_seconds = float(
                        self._automation_cfg().get("gesture_stable_seconds", 0.25)
                    )

                    elapsed = now - self.gesture_stable_since

                    if elapsed >= stable_seconds:
                        self.last_gesture = gesture
                        self.gesture_detected = True
                        self.gesture_confidence = confidence
                        self.hand_bbox = bbox
                        return True, gesture, confidence, bbox

            self.gesture_detected = False
            self.current_gesture = None
            self.gesture_confidence = 0.0
            self.hand_bbox = None
            return False, None, 0.0, None

        except Exception as exc:
            self.log(f"Error detectando gesto YOLO: {exc}")
            return False, None, 0.0, None

    def _classify_gesture(self, points, base_confidence):
        """Clasifica el gesto basado en los 21 keypoints de la mano."""
        if len(points) < 21:
            return None, 0.0

        wrist = points[0]
        thumb_tip = points[4]
        thumb_ip = points[3]
        index_tip = points[8]
        index_pip = points[6]
        middle_tip = points[12]
        middle_pip = points[10]
        ring_tip = points[16]
        ring_pip = points[14]
        pinky_tip = points[20]
        pinky_pip = points[18]

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def is_finger_extended(tip, pip, mcp, wrist_ref=None):
            dist_tip_wrist = distance(tip, wrist)
            dist_pip_wrist = distance(pip, wrist)
            return dist_tip_wrist > dist_pip_wrist * 1.1

        index_ext = is_finger_extended(index_tip, index_pip, points[5])
        middle_ext = is_finger_extended(middle_tip, middle_pip, points[9])
        ring_ext = distance(ring_tip, wrist) > distance(ring_pip, wrist) * 1.1
        pinky_ext = is_finger_extended(pinky_tip, pinky_pip, points[17])

        thumb_ext = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0])

        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        if thumb_ext:
            extended_count += 1

        gesture = None
        confidence = base_confidence

        if extended_count == 0:
            gesture = "fist"
            confidence = min(1.0, base_confidence + 0.2)
        elif extended_count >= 5:
            gesture = "open_hand"
            confidence = min(1.0, base_confidence + 0.15)
        elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
            gesture = "pointing"
            confidence = min(1.0, base_confidence + 0.12)
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            finger_gap = abs(index_tip[0] - middle_tip[0])
            hand_width = max(0.01, abs(np.max(points[:, 0]) - np.min(points[:, 0])))
            if finger_gap > hand_width * 0.15:
                gesture = "victory"
            else:
                gesture = "two_fingers"
            confidence = min(1.0, base_confidence + 0.1)
        elif thumb_ext and not index_ext and not middle_ext and extended_count <= 2:
            gesture = "thumbs_up"
            confidence = min(1.0, base_confidence + 0.15)
        elif index_ext and middle_ext and ring_ext and not pinky_ext:
            gesture = "three_fingers"
            confidence = min(1.0, base_confidence + 0.08)
        elif extended_count == 4:
            if not thumb_ext:
                gesture = "four_fingers"
                confidence = min(1.0, base_confidence + 0.08)
            else:
                gesture = "open_hand"
                confidence = min(1.0, base_confidence + 0.1)

        return gesture, confidence

    def _cooldown_ok(self):
        """Verifica si el cooldown ha terminado."""
        cooldown = float(self._automation_cfg().get("cooldown_seconds", 8))
        return time.monotonic() - self.last_trigger >= cooldown

    def _trigger(self, source):
        """Ejecuta la apertura automática."""
        if not self._cooldown_ok():
            return False

        self.last_trigger = time.monotonic()
        self.last_trigger_source = source

        self.log(f"AUTOMATIZACIÓN → APERTURA: {source}")

        try:
            self.open_door(source=f"AUTOMÁTICO: {source}")
            return True
        except Exception as exc:
            self.log(f"Error ejecutando apertura: {exc}")
            return False

    def update(self, frame):
        """
        Actualiza el estado de la automatización.
        
        Returns:
            tuple: (gesture_detected, gesture_name, confidence, bbox)
        """
        if frame is None:
            return False, None, 0.0, None

        automation = self._automation_cfg()

        if not automation.get("enabled", False):
            self.current_gesture = None
            self.gesture_stable_since = None
            self.gesture_detected = False
            self.gesture_confidence = 0.0
            self.hand_bbox = None
            return False, None, 0.0, None

        now = time.monotonic()
        self.frame_counter += 1

        if automation.get("gesture_enabled", False):
            if now - self.last_gesture_process >= self.gesture_interval:
                self.last_gesture_process = now

                detected, gesture_name, confidence, bbox = self._gesture_detected(frame)

                if detected and gesture_name:
                    wanted_gesture = str(automation.get("gesture", "open_hand"))

                    if gesture_name == wanted_gesture:
                        self.log(f"SEÑA DETECTADA: {gesture_name} ({confidence:.2f})")
                        return True, gesture_name, confidence, bbox

        return False, None, 0.0, None

    def get_status(self):
        """Retorna el estado actual del motor de automatización."""
        return {
            "gesture": self.last_gesture,
            "current_gesture": self.current_gesture,
            "gesture_detected": self.gesture_detected,
            "gesture_confidence": self.gesture_confidence,
            "hand_bbox": self.hand_bbox,
            "last_trigger": self.last_trigger,
            "last_trigger_source": self.last_trigger_source,
            "yolo_pose_available": self.yolo_pose is not None,
        }

    def update_config(self, cfg):
        """Actualiza la configuración y resetea estados."""
        self.cfg = cfg or {}
        self.log("Configuración de automatización actualizada.")

        self.current_gesture = None
        self.last_gesture = None
        self.gesture_stable_since = None
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.hand_bbox = None

    def close(self):
        """Libera recursos del motor de automatización."""
        try:
            if self.yolo_pose is not None:
                pass
        except Exception:
            pass

        self.yolo_pose = None
