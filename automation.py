import time
import cv2
import numpy as np


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

    Lógica:
        - Detección de puntos clave de la mano
        - Cálculo de geometría para identificar gestos
        - Visualización en tiempo real con bounding box y confianza

    No utiliza:
        - Detección de persona
        - Reconocimiento facial
        - MediaPipe
        - standing_seconds

    La apertura siempre se realiza mediante:

        open_door(source="...")
    """

    def __init__(
        self,
        cfg,
        open_door_callback,
        logger=None,
    ):

        self.cfg = cfg or {}

        self.open_door = open_door_callback

        self.log = logger or (lambda msg: None)

        # =========================================================
        # ESTADO SEÑAS
        # =========================================================

        self.current_gesture = None
        self.last_gesture = None
        self.gesture_stable_since = None
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.hand_bbox = None  # [x, y, w, h]

        # =========================================================
        # AUTOMATIZACIÓN
        # =========================================================

        self.last_trigger = 0.0
        self.last_trigger_source = None

        # =========================================================
        # MODELO YOLO POSE
        # =========================================================

        self.yolo_pose = None
        self.model_path = None

        # =========================================================
        # CONTROL DE PROCESAMIENTO
        # =========================================================

        self.frame_counter = 0
        self.last_gesture_process = 0.0
        self.gesture_interval = 0.08  # Procesar cada 80ms

        # =========================================================
        # CARGAR MODELO
        # =========================================================

        self._load_yolo_pose()

    # =============================================================
    # CONFIGURACIÓN
    # =============================================================

    def _automation_cfg(self):
        """
        Devuelve siempre el bloque automation.
        """

        if not isinstance(self.cfg, dict):
            self.cfg = {}

        automation = self.cfg.get("automation")

        if not isinstance(automation, dict):
            automation = {}
            self.cfg["automation"] = automation

        return automation

    # =============================================================
    # CARGAR YOLO POSE
    # =============================================================

    def _load_yolo_pose(self):

        self.yolo_pose = None

        try:
            from ultralytics import YOLO

            # Usar modelo pose por defecto o el configurado
            automation = self._automation_cfg()
            self.model_path = automation.get(
                "yolo_pose_model",
                "yolo11n-pose.pt"
            )

            self.yolo_pose = YOLO(self.model_path)
            self.yolo_pose.to("cpu")  # Usar CPU

            self.log(
                f"YOLO Pose cargado: {self.model_path}"
            )

        except Exception as exc:
            self.log(
                f"YOLO Pose no disponible: {exc}"
            )
            self.yolo_pose = None

    # =============================================================
    # DETECTAR GESTO CON YOLO POSE
    # =============================================================

    def _gesture_detected(self, frame):
        """
        Detecta gestos de mano usando YOLO Pose.
        Retorna: (detectado, nombre_gesto, confianza, bbox)
        """

        if self.yolo_pose is None or frame is None:
            return False, None, 0.0, None

        try:
            # Ejecutar inferencia YOLO Pose
            results = self.yolo_pose(
                frame,
                verbose=False,
                conf=0.4,
                iou=0.45
            )

            if not results or len(results) == 0:
                self.gesture_detected = False
                self.current_gesture = None
                self.gesture_confidence = 0.0
                self.hand_bbox = None
                return False, None, 0.0, None

            result = results[0]
            
            # Verificar si hay keypoints (manos)
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                self.gesture_detected = False
                self.current_gesture = None
                self.gesture_confidence = 0.0
                self.hand_bbox = None
                return False, None, 0.0, None

                # Procesar cada mano detectada
                for i in range(len(result.keypoints)):
                    kp = result.keypoints[i]
                    
                    # Obtener coordenadas de keypoints
                    if kp.xy is None or len(kp.xy[0]) < 21:
                        continue

                    points = kp.xy[0].cpu().numpy()  # 21 puntos de la mano
                    confidences = kp.conf[0].cpu().numpy() if hasattr(kp, 'conf') else None
                    
                    # Calcular confianza promedio de puntos clave
                    if confidences is not None:
                        avg_conf = float(np.mean(confidences[:5]))  # Primeros 5 puntos
                    else:
                        avg_conf = 0.8

                    # Identificar gesto basado en geometría de puntos
                    gesture, confidence = self._classify_gesture(points, avg_conf)
                    
                    if gesture:
                        # Calcular bounding box alrededor de la mano [x1, y1, x2, y2]
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

                        # Verificar cambio de gesto
                        if self.current_gesture != gesture:
                            self.current_gesture = gesture
                            self.gesture_stable_since = now
                            self.gesture_detected = False
                            self.gesture_confidence = confidence
                            self.hand_bbox = bbox
                            return False, gesture, confidence, bbox

                        # Iniciar temporizador si es nuevo
                        if self.gesture_stable_since is None:
                            self.gesture_stable_since = now
                            self.gesture_detected = False
                            self.gesture_confidence = confidence
                            self.hand_bbox = bbox
                            return False, gesture, confidence, bbox

                        # Verificar estabilidad temporal
                        stable_seconds = float(
                            self._automation_cfg().get(
                                "gesture_stable_seconds",
                                0.25
                            )
                        )

                        elapsed = now - self.gesture_stable_since

                        if elapsed >= stable_seconds:
                            self.last_gesture = gesture
                            self.gesture_detected = True
                            self.gesture_confidence = confidence
                            self.hand_bbox = bbox
                            return True, gesture, confidence, bbox

            # Sin manos válidas
            self.gesture_detected = False
            self.current_gesture = None
            self.gesture_confidence = 0.0
            self.hand_bbox = None
            return False, None, 0.0, None

        except Exception as exc:
            self.log(f"Error detectando gesto YOLO: {exc}")
            return False, None, 0.0, None

    def _classify_gesture(self, points, base_confidence):
        """
        Clasifica el gesto basado en los 21 keypoints de la mano.
        
        Puntos clave (0-20):
        0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 
        13-16: ring, 17-20: pinky
        
        Retorna: (nombre_gesto, confianza)
        """
        
        if len(points) < 21:
            return None, 0.0

        # Extraer puntos clave
        wrist = points[0]
        thumb_tip = points[4]
        thumb_ip = points[3]
        index_tip = points[8]
        index_pip = points[6]
        index_mcp = points[5]
        middle_tip = points[12]
        middle_pip = points[10]
        middle_mcp = points[9]
        ring_tip = points[16]
        ring_pip = points[14]
        pinky_tip = points[20]
        pinky_pip = points[18]

        # Calcular distancias para determinar dedos extendidos
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def is_finger_extended(tip, pip, mcp, wrist_ref=None):
            """Verifica si un dedo está extendido comparando con la muñeca"""
            # El dedo está extendido si la punta está más lejos de la muñeca que el PIP
            dist_tip_wrist = distance(tip, wrist)
            dist_pip_wrist = distance(pip, wrist)
            return dist_tip_wrist > dist_pip_wrist * 1.1

        # Verificar cada dedo (excepto pulgar)
        index_ext = is_finger_extended(index_tip, index_pip, index_mcp)
        middle_ext = is_finger_extended(middle_tip, middle_pip, middle_mcp)
        ring_ext = is_finger_extended(ring_tip, ring_pip, ring_mcp) if 'ring_mcp' in dir() else distance(ring_tip, wrist) > distance(ring_pip, wrist) * 1.1
        pinky_ext = is_finger_extended(pinky_tip, pinky_pip, points[17])
        
        # Verificar pulgar (comparar posición horizontal)
        thumb_ext = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0])

        # Contar dedos extendidos
        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        if thumb_ext:
            extended_count += 1

        # Clasificar gestos con prioridades claras
        gesture = None
        confidence = base_confidence

        # PUÑO (ningún dedo extendido)
        if extended_count == 0:
            gesture = "fist"
            confidence = min(1.0, base_confidence + 0.2)

        # MANO ABIERTA (todos o casi todos los dedos extendidos)
        elif extended_count >= 5:
            gesture = "open_hand"
            confidence = min(1.0, base_confidence + 0.15)

        # DEDO SEÑALANDO (solo índice)
        elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
            gesture = "pointing"
            confidence = min(1.0, base_confidence + 0.12)

        # DOS DEDOS (índice y medio, anular y meñique cerrados)
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            # Verificar separación entre índice y medio para distinguir V
            finger_gap = abs(index_tip[0] - middle_tip[0])
            hand_width = max(0.01, abs(np.max(points[:, 0]) - np.min(points[:, 0])))
            
            if finger_gap > hand_width * 0.15:  # Separación amplia = V
                gesture = "victory"
            else:
                gesture = "two_fingers"
            confidence = min(1.0, base_confidence + 0.1)

        # PULGAR ARRIBA (pulgar extendido, otros cerrados)
        elif thumb_ext and not index_ext and not middle_ext and extended_count <= 2:
            gesture = "thumbs_up"
            confidence = min(1.0, base_confidence + 0.15)

        # TRES DEDOS (índice, medio, anular)
        elif index_ext and middle_ext and ring_ext and not pinky_ext:
            gesture = "three_fingers"
            confidence = min(1.0, base_confidence + 0.08)

        # CUATRO DEDOS (todos menos pulgar o meñique)
        elif extended_count == 4:
            if not thumb_ext:
                gesture = "four_fingers"
                confidence = min(1.0, base_confidence + 0.08)
            else:
                gesture = "open_hand"
                confidence = min(1.0, base_confidence + 0.1)

        return gesture, confidence

    # =============================================================
    # COOLDOWN
    # =============================================================

    def _cooldown_ok(self):

        cooldown = float(
            self._automation_cfg().get(
                "cooldown_seconds",
                8,
            )
        )

        return (
            time.monotonic()
            - self.last_trigger
            >= cooldown
        )

    # =============================================================
    # APERTURA
    # =============================================================

    def _trigger(self, source):

        if not self._cooldown_ok():

            return False

        self.last_trigger = (
            time.monotonic()
        )

        self.last_trigger_source = source

        self.log(
            "AUTOMATIZACIÓN → "
            f"APERTURA: {source}"
        )

        try:

            self.open_door(
                source=(
                    f"AUTOMÁTICO: {source}"
                )
            )

            return True

        except Exception as exc:

            self.log(
                "Error ejecutando "
                f"apertura: {exc}"
            )

            return False

    # =============================================================
    # ACTUALIZAR
    # =============================================================

    def update(self, frame):
        """
        Actualiza el estado de la automatización.
        Retorna: (gesture_detected, gesture_name, confidence, bbox)
        """

        if frame is None:
            return False, None, 0.0, None

        automation = self._automation_cfg()

        # =========================================================
        # AUTOMATIZACIÓN DESACTIVADA
        # =========================================================

        if not automation.get("enabled", False):
            self.current_gesture = None
            self.gesture_stable_since = None
            self.gesture_detected = False
            self.gesture_confidence = 0.0
            self.hand_bbox = None
            return False, None, 0.0, None

        now = time.monotonic()
        self.frame_counter += 1

        # =========================================================
        # DETECTAR SEÑA CON YOLO POSE
        # =========================================================

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

    # =============================================================
    # ESTADO
    # =============================================================

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

    # =============================================================
    # ACTUALIZAR CONFIGURACIÓN
    # =============================================================

    def update_config(self, cfg):
        """Actualiza la configuración y resetea estados."""

        self.cfg = cfg or {}

        self.log(
            "Configuración de "
            "automatización actualizada."
        )

        # =========================================================
        # RESET DE ESTADOS
        # =========================================================

        self.current_gesture = None
        self.last_gesture = None
        self.gesture_stable_since = None
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.hand_bbox = None

    # =============================================================
    # CERRAR
    # =============================================================

    def close(self):
        """Libera recursos del motor de automatización."""

        try:
            if self.yolo_pose is not None:
                # YOLO no requiere close explícito
                pass

        except Exception:
            pass

        self.yolo_pose = None

        self.mp = None