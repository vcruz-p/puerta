import time


class AutomationEngine:
    """
    Motor de automatización para control de acceso.

    Señales disponibles:

        ROSTRO
            Reconocimiento facial mediante face_recognition/dlib.
            Compara rostros detectados con la base de datos registrada.

        SEÑA
            MediaPipe Hands reconoce:
                - open_hand
                - fist
                - two_fingers
                - victory
                - thumbs_up

    Lógica:

        ANY
            Cualquiera de las señales habilitadas puede abrir.

        ALL
            Todas las señales habilitadas deben cumplirse.

    No utiliza:
        - YOLO
        - Detección de persona
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
        # ESTADO ROSTRO
        # =========================================================

        self.face_detected = False

        self.face_match_id = None

        self.face_confidence = 0.0

        self.face_encoding = None

        self.last_face_time = 0.0

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

        self.face_model = None

        self.known_faces = {}

        self.mp = None

        self.mp_hands = None

        self.hands = None

        # =========================================================
        # CONTROL DE PROCESAMIENTO
        # =========================================================

        self.frame_counter = 0

        self.last_face_process = 0.0

        self.last_gesture_process = 0.0

        self.face_interval = 0.20

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

        self._load_face_recognition()

        self._load_mediapipe()

    # =============================================================
    # FACE RECOGNITION
    # =============================================================

    def _load_face_recognition(self):

        self.face_model = None

        self.known_faces = {}

        try:

            import face_recognition

            self.face_model = face_recognition

            automation = self._automation_cfg()

            faces_db = automation.get(
                "faces_db",
                "faces_db"
            )

            self.log(
                f"Cargando base de datos facial: {faces_db}"
            )

            self._load_known_faces(faces_db)

            self.log(
                "Reconocimiento facial cargado correctamente."
            )

        except Exception as exc:

            self.face_model = None

            self.log(
                f"Face recognition no disponible: {exc}"
            )

    def _load_known_faces(self, db_path):

        import os
        import pickle

        if not self.face_model:
            return

        db_file = os.path.join(db_path, "known_faces.pkl")

        if not os.path.exists(db_file):
            self.log("Base de datos facial vacía.")
            return

        try:

            with open(db_file, "rb") as f:
                self.known_faces = pickle.load(f)

            self.log(
                f"{len(self.known_faces)} rostros cargados."
            )

        except Exception as exc:

            self.log(f"Error cargando rostros: {exc}")
            self.known_faces = {}

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
    # DETECTAR ROSTRO
    # =============================================================

    def _face_detected(self, frame):

        if self.face_model is None:

            self.face_detected = False
            self.face_match_id = None
            self.face_confidence = 0.0

            return False

        if frame is None:

            return False

        try:

            automation = self._automation_cfg()

            threshold = float(
                automation.get(
                    "face_threshold",
                    0.48,
                )
            )

            rgb = frame[:, :, ::-1]

            face_locations = self.face_model.face_locations(rgb)

            if not face_locations:

                self.face_detected = False
                self.face_match_id = None
                self.face_confidence = 0.0

                return False

            face_encodings = self.face_model.face_encodings(
                rgb,
                face_locations
            )

            if not face_encodings:

                self.face_detected = False
                self.face_match_id = None
                self.face_confidence = 0.0

                return False

            face_encoding = face_encodings[0]

            best_match = None
            best_distance = float('inf')

            for name, known_encoding in self.known_faces.items():

                distance = self.face_model.face_distance(
                    [known_encoding],
                    face_encoding
                )[0]

                if distance < best_distance:

                    best_distance = distance
                    best_match = name

            confidence = 1.0 - best_distance

            if confidence >= (1.0 - threshold):

                self.face_detected = True
                self.face_match_id = best_match
                self.face_confidence = confidence
                self.face_encoding = face_encoding
                self.last_face_time = time.monotonic()

                return True

        except Exception as exc:

            self.log(
                f"Face recognition: {exc}"
            )

        self.face_detected = False
        self.face_match_id = None
        self.face_confidence = 0.0

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

            self.face_detected = False
            self.face_match_id = None
            self.face_confidence = 0.0

            self.current_gesture = None

            self.gesture_stable_since = None

            return

        now = time.monotonic()

        self.frame_counter += 1

        # =========================================================
        # SEÑALES
        # =========================================================

        face_signal = False
        face_name = None

        gesture_signal = False

        gesture_name = None

        # =========================================================
        # ROSTRO
        # =========================================================

        if automation.get(
            "face_enabled",
            False,
        ):

            if (
                now
                - self.last_face_process
                >= self.face_interval
            ):

                self.last_face_process = now

                detected = (
                    self._face_detected(
                        frame
                    )
                )

                if detected:

                    face_signal = True
                    face_name = self.face_match_id

                    self.log(
                        f"ROSTRO RECONOCIDO: {face_name} "
                        f"({self.face_confidence:.2f})"
                    )

                else:

                    if self.face_detected:
                        self.log(
                            "ROSTRO NO RECONOCIDO"
                        )

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
        # ROSTRO
        # ---------------------------------------------------------

        if face_signal and face_name:

            signals.append(
                f"ROSTRO:{face_name}"
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
            "face_enabled",
            False,
        ):

            required.append(
                "ROSTRO"
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

        return {

            "face_detected":
                self.face_detected,

            "face_match_id":
                self.face_match_id,

            "face_confidence":
                self.face_confidence,

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

            "face_recognition_available":
                self.face_model is not None,

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
        # RECARGAR FACE RECOGNITION
        # =========================================================

        try:

            automation = (
                self._automation_cfg()
            )

            faces_db = automation.get(
                "faces_db",
                "faces_db",
            )

            self._load_known_faces(faces_db)

            self.log(
                f"Base de datos facial actualizada: "
                f"{len(self.known_faces)} rostros."
            )

        except Exception as exc:

            self.log(
                f"No se pudo actualizar "
                f"face recognition: {exc}"
            )

        # =========================================================
        # RESET DE ESTADOS
        # =========================================================

        self.face_detected = False
        self.face_match_id = None
        self.face_confidence = 0.0
        self.face_encoding = None

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

        self.face_model = None

        self.known_faces = {}