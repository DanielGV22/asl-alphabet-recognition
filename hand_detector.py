import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr):
        frame_rgb = frame_bgr[:, :, ::-1]
        res = self.hands.process(frame_rgb)

        if not res.multi_hand_landmarks:
            return None

        hand_landmarks = res.multi_hand_landmarks[0]

        pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)

        x_min = float(np.min(pts[:, 0])); x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1])); y_max = float(np.max(pts[:, 1]))

        conf = 1.0
        if res.multi_handedness and res.multi_handedness[0].classification:
            conf = float(res.multi_handedness[0].classification[0].score)

        return {
            "landmarks": pts,                 # (21,3) normalized
            "bbox_norm": (x_min, y_min, x_max, y_max),
            "hand_conf": conf
        }
