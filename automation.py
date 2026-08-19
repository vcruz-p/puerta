import time


class AutomationEngine:
    """
    Motor de automatización para control de acceso.

    Señales disponibles:

        PERSONA
            YOLO detecta una persona dentro de la zona configurada
            y exige una permanencia mínima.

        SEÑA
            MediaPipe Hands reconoce:
                - open_hand
                - fist
                - two_fingers

    Lógica:

        ANY
            Cualquiera de las señales habilitadas puede abrir.

        ALL
            Todas las señales habilitadas deben cumplirse.

    No utiliza:
        - face_recognition
        - dlib
        - reconocimiento facial

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
        # ESTADO PERSONA
        # =========================================================

        self.person_since = None

        self.person_detected = False

        self.person_confidence = 0.0

        self.person_box = None

        self.last_person_time = 0.0

        # =========================================================
        # ESTADO SEÑAS
        # =========================================================

        self.current_gesture = None

        self.last_gesture = None

        self.gesture_stable_since = None

        self.gesture_detected = False

        # =========================================================
        # AUTOMATIZACIÓN
        # =========================================================

        self.last_trigger = 0.0

        self.last_trigger_source = None

        # =========================================================
        # MODELOS
        # =========================================================

        self.yolo = None

        self.mp = None

        self.mp_hands = None

        self.hands = None

        # =========================================================
        # CONTROL DE PROCESAMIENTO
        # =========================================================

        self.frame_counter = 0

        self.last_yolo_process = 0.0

        self.last_gesture_process = 0.0

        self.yolo_interval = 0.12

        self.gesture_interval = 0.10

        # =========================================================
        # CARGAR MODELOS
        # =========================================================

        self._load_models()

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
    # CARGAR MODELOS
    # =============================================================

    def _load_models(self):

        self._load_yolo()

        self._load_mediapipe()

    # =============================================================
    # YOLO
    # =============================================================

    def _load_yolo(self):

        self.yolo = None

        try:

            from ultralytics import YOLO

            automation = self._automation_cfg()

            model_path = automation.get(
                "yolo_model",
                "yolo11n.pt",
            )

            self.log(
                f"Cargando YOLO: {model_path}"
            )

            self.yolo = YOLO(model_path)

            self.log(
                "YOLO cargado correctamente."
            )

        except Exception as exc:

            self.yolo = None

            self.log(
                f"YOLO no disponible: {exc}"
            )

    # =============================================================
    # MEDIAPIPE HANDS
    # =============================================================

    def _load_mediapipe(self):

        self.mp = None

        self.mp_hands = None

        self.hands = None

        try:

            import mediapipe as mp

            self.mp = mp

            self.mp_hands = (
                mp.solutions.hands
            )

            self.hands = (
                self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
            )

            self.log(
                "MediaPipe Hands cargado correctamente."
            )

        except Exception as exc:

            self.log(
                f"MediaPipe no disponible: {exc}"
            )

            self.mp = None
            self.mp_hands = None
            self.hands = None

    # =============================================================
    # DETECTAR PERSONA
    # =============================================================

    def _person_detected(self, frame):

        if self.yolo is None:

            self.person_detected = False
            self.person_confidence = 0.0
            self.person_box = None

            return False

        if frame is None:

            return False

        try:

            automation = self._automation_cfg()

            confidence = float(
                automation.get(
                    "yolo_confidence",
                    0.50,
                )
            )

            results = self.yolo.predict(
                source=frame,
                verbose=False,
                conf=confidence,
                classes=[0],
                imgsz=640,
            )

            height, width = frame.shape[:2]

            best_confidence = 0.0

            best_box = None

            # =====================================================
            # ZONA DE DETECCIÓN
            # =====================================================

            zone_x_min = float(
                automation.get(
                    "zone_x_min",
                    0.15,
                )
            )

            zone_x_max = float(
                automation.get(
                    "zone_x_max",
                    0.85,
                )
            )

            zone_y_min = float(
                automation.get(
                    "zone_y_min",
                    0.10,
                )
            )

            zone_y_max = float(
                automation.get(
                    "zone_y_max",
                    0.95,
                )
            )

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    try:

                        cls = int(
                            box.cls[0].item()
                        )

                        conf = float(
                            box.conf[0].item()
                        )

                        # Clase 0 = persona
                        if cls != 0:
                            continue

                        coords = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            coords,
                        )

                        # -----------------------------------------
                        # CENTRO DEL OBJETO
                        # -----------------------------------------

                        cx = (
                            x1 + x2
                        ) / 2.0

                        cy = (
                            y1 + y2
                        ) / 2.0

                        normalized_x = (
                            cx / max(width, 1)
                        )

                        normalized_y = (
                            cy / max(height, 1)
                        )

                        # -----------------------------------------
                        # COMPROBAR ZONA
                        # -----------------------------------------

                        inside_zone = (
                            zone_x_min
                            <= normalized_x
                            <= zone_x_max
                            and
                            zone_y_min
                            <= normalized_y
                            <= zone_y_max
                        )

                        if not inside_zone:
                            continue

                        if conf > best_confidence:

                            best_confidence = conf

                            best_box = (
                                x1,
                                y1,
                                x2,
                                y2,
                            )

                    except Exception:

                        continue

            # =====================================================
            # PERSONA ENCONTRADA
            # =====================================================

            if best_box is not None:

                self.person_detected = True

                self.person_confidence = (
                    best_confidence
                )

                self.person_box = best_box

                return True

        except Exception as exc:

            self.log(
                f"YOLO persona: {exc}"
            )

        self.person_detected = False

        self.person_confidence = 0.0

        self.person_box = None

        return False

    # =============================================================
    # RECONOCIMIENTO DE SEÑAS
    # =============================================================

    def _gesture_detected(self, frame):

        if self.hands is None:

            self.gesture_detected = False

            self.current_gesture = None

            return False, None

        if frame is None:

            return False, None

        try:

            import cv2

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            result = self.hands.process(rgb)

            if not result.multi_hand_landmarks:

                self.gesture_detected = False

                self.current_gesture = None

                self.gesture_stable_since = None

                return False, None

            detected_gestures = []

            # =====================================================
            # ANALIZAR CADA MANO
            # =====================================================

            for hand in result.multi_hand_landmarks:

                lm = hand.landmark

                fingers = 0

                # ---------------------------------------------
                # ÍNDICE
                # ---------------------------------------------

                if lm[8].y < lm[6].y:
                    fingers += 1

                # ---------------------------------------------
                # MEDIO
                # ---------------------------------------------

                if lm[12].y < lm[10].y:
                    fingers += 1

                # ---------------------------------------------
                # ANULAR
                # ---------------------------------------------

                if lm[16].y < lm[14].y:
                    fingers += 1

                # ---------------------------------------------
                # MEÑIQUE
                # ---------------------------------------------

                if lm[20].y < lm[18].y:
                    fingers += 1

                # ---------------------------------------------
                # PULGAR
                # ---------------------------------------------

                thumb_extended = (
                    abs(
                        lm[4].x - lm[3].x
                    ) > 0.035
                )

                # =================================================
                # CLASIFICACIÓN
                # =================================================

                gesture = None

                # MANO ABIERTA

                if (
                    fingers >= 4
                    and thumb_extended
                ):

                    gesture = "open_hand"

                # PUÑO

                elif fingers == 0:

                    gesture = "fist"

                # DOS DEDOS

                elif fingers == 2:

                    gesture = "two_fingers"

                if gesture:

                    detected_gestures.append(
                        gesture
                    )

            # =====================================================
            # SIN SEÑA
            # =====================================================

            if not detected_gestures:

                self.gesture_detected = False

                self.current_gesture = None

                self.gesture_stable_since = None

                return False, None

            # =====================================================
            # TOMAR PRIMERA SEÑA
            # =====================================================

            gesture = detected_gestures[0]

            now = time.monotonic()

            # =====================================================
            # CAMBIO DE SEÑA
            # =====================================================

            if self.current_gesture != gesture:

                self.current_gesture = gesture

                self.gesture_stable_since = now

                self.gesture_detected = False

                return False, gesture

            # =====================================================
            # INICIAR TEMPORIZADOR
            # =====================================================

            if self.gesture_stable_since is None:

                self.gesture_stable_since = now

                self.gesture_detected = False

                return False, gesture

            # =====================================================
            # TIEMPO DE ESTABILIDAD
            # =====================================================

            stable_seconds = float(
                self._automation_cfg().get(
                    "gesture_stable_seconds",
                    0.30,
                )
            )

            elapsed = (
                now
                - self.gesture_stable_since
            )

            # =====================================================
            # SEÑA CONFIRMADA
            # =====================================================

            if elapsed >= stable_seconds:

                self.last_gesture = gesture

                self.gesture_detected = True

                return True, gesture

        except Exception as exc:

            self.log(
                f"Reconocimiento de seña: {exc}"
            )

        return False, None

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

        if frame is None:

            return

        automation = (
            self._automation_cfg()
        )

        # =========================================================
        # AUTOMATIZACIÓN DESACTIVADA
        # =========================================================

        if not automation.get(
            "enabled",
            False,
        ):

            self.person_since = None

            self.current_gesture = None

            self.gesture_stable_since = None

            return

        now = time.monotonic()

        self.frame_counter += 1

        # =========================================================
        # SEÑALES
        # =========================================================

        person_signal = False

        gesture_signal = False

        gesture_name = None

        # =========================================================
        # PERSONA
        # =========================================================

        if automation.get(
            "standing_enabled",
            True,
        ):

            if (
                now
                - self.last_yolo_process
                >= self.yolo_interval
            ):

                self.last_yolo_process = now

                detected = (
                    self._person_detected(
                        frame
                    )
                )

                if detected:

                    if self.person_since is None:

                        self.person_since = now

                        self.log(
                            "PERSONA DETECTADA "
                            "FRENTE A LA PUERTA"
                        )

                    self.last_person_time = now

                    required_seconds = float(
                        automation.get(
                            "standing_seconds",
                            4,
                        )
                    )

                    elapsed = (
                        now
                        - self.person_since
                    )

                    if (
                        elapsed
                        >= required_seconds
                    ):

                        person_signal = True

                else:

                    if self.person_since is not None:

                        self.log(
                            "PERSONA SALIÓ "
                            "DE LA ZONA"
                        )

                    self.person_since = None

        else:

            self.person_since = None

        # =========================================================
        # SEÑA
        # =========================================================

        if automation.get(
            "gesture_enabled",
            False,
        ):

            if (
                now
                - self.last_gesture_process
                >= self.gesture_interval
            ):

                self.last_gesture_process = now

                (
                    gesture_signal,
                    gesture_name,
                ) = self._gesture_detected(
                    frame
                )

        # =========================================================
        # CONSTRUIR SEÑALES
        # =========================================================

        signals = []

        # ---------------------------------------------------------
        # PERSONA
        # ---------------------------------------------------------

        if person_signal:

            signals.append(
                "PERSONA"
            )

        # ---------------------------------------------------------
        # SEÑA
        # ---------------------------------------------------------

        wanted_gesture = str(
            automation.get(
                "gesture",
                "open_hand",
            )
        )

        if (
            gesture_signal
            and gesture_name
            and gesture_name
            == wanted_gesture
        ):

            signals.append(
                f"SEÑA:{gesture_name}"
            )

        # =========================================================
        # NO HAY SEÑALES
        # =========================================================

        if not signals:

            return

        # =========================================================
        # LÓGICA
        # =========================================================

        logic = str(
            automation.get(
                "logic",
                "ANY",
            )
        ).upper()

        # =========================================================
        # ANY
        # =========================================================

        if logic == "ANY":

            self._trigger(
                " + ".join(signals)
            )

            return

        # =========================================================
        # ALL
        # =========================================================

        required = []

        if automation.get(
            "standing_enabled",
            True,
        ):

            required.append(
                "PERSONA"
            )

        if automation.get(
            "gesture_enabled",
            False,
        ):

            required.append(
                "SEÑA"
            )

        # =========================================================
        # SI NO HAY SEÑALES CONFIGURADAS
        # =========================================================

        if not required:

            return

        # =========================================================
        # COMPROBAR TODAS
        # =========================================================

        for requirement in required:

            found = any(
                signal.startswith(
                    requirement
                )
                for signal in signals
            )

            if not found:

                return

        # =========================================================
        # APERTURA
        # =========================================================

        self._trigger(
            " + ".join(signals)
        )

    # =============================================================
    # ESTADO
    # =============================================================

    def get_status(self):

        standing_seconds = 0.0

        if self.person_since is not None:

            standing_seconds = (
                time.monotonic()
                - self.person_since
            )

        return {

            "person_detected":
                self.person_detected,

            "person_confidence":
                self.person_confidence,

            "person_box":
                self.person_box,

            "standing_seconds":
                standing_seconds,

            "gesture":
                self.last_gesture,

            "current_gesture":
                self.current_gesture,

            "gesture_detected":
                self.gesture_detected,

            "last_trigger":
                self.last_trigger,

            "last_trigger_source":
                self.last_trigger_source,

            "yolo_available":
                self.yolo is not None,

            "mediapipe_available":
                self.hands is not None,
        }

    # =============================================================
    # ACTUALIZAR CONFIGURACIÓN
    # =============================================================

    def update_config(self, cfg):

        self.cfg = cfg or {}

        self.log(
            "Configuración de "
            "automatización actualizada."
        )

        # =========================================================
        # RECARGAR YOLO
        # =========================================================

        try:

            automation = (
                self._automation_cfg()
            )

            model_path = automation.get(
                "yolo_model",
                "yolo11n.pt",
            )

            from ultralytics import YOLO

            self.yolo = YOLO(
                model_path
            )

            self.log(
                f"YOLO actualizado: "
                f"{model_path}"
            )

        except Exception as exc:

            self.log(
                f"No se pudo actualizar "
                f"YOLO: {exc}"
            )

        # =========================================================
        # RESET DE ESTADOS
        # =========================================================

        self.person_since = None

        self.person_detected = False

        self.person_confidence = 0.0

        self.person_box = None

        self.current_gesture = None

        self.last_gesture = None

        self.gesture_stable_since = None

        self.gesture_detected = False

    # =============================================================
    # CERRAR
    # =============================================================

    def close(self):

        try:

            if self.hands is not None:

                self.hands.close()

        except Exception:

            pass

        self.hands = None

        self.mp_hands = None

        self.mp = None

        self.yolo = None