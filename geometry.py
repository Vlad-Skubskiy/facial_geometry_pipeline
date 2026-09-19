import cv2
import numpy as np
import mediapipe as mp

class FacialGeometryPipeline:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.LEFT_EYE_CORNER = 33
        self.RIGHT_EYE_CORNER = 263
        self.NOSE_TIP = 4
        self.CHIN = 152
        self.FOREHEAD = 10
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.JAW_LEFT = 234
        self.JAW_RIGHT = 454

    def _euclidean_distance(self, pt1, pt2):
        return np.linalg.norm(pt1 - pt2)

    def process_image(self, image_np: np.ndarray) -> dict:
        if image_np is None or len(image_np.shape) != 3 or image_np.shape[2] != 3:
            return {"error": "Invalid image shape or format"}

        h, w, _ = image_np.shape
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        ipd = self._euclidean_distance(coords[self.LEFT_EYE_CORNER], coords[self.RIGHT_EYE_CORNER])
        if ipd == 0:
            return {"error": "Invalid IPD calculation"}

        mouth_width = self._euclidean_distance(coords[self.MOUTH_LEFT], coords[self.MOUTH_RIGHT])
        face_height = self._euclidean_distance(coords[self.FOREHEAD], coords[self.CHIN])
        jaw_width = self._euclidean_distance(coords[self.JAW_LEFT], coords[self.JAW_RIGHT])
        nose_length = self._euclidean_distance(coords[self.FOREHEAD], coords[self.NOSE_TIP])

        return {
            "ipd_pixels": round(float(ipd), 2),
            "ratios": {
                "mouth_width_ratio": round(float(mouth_width / ipd), 4),
                "face_height_ratio": round(float(face_height / ipd), 4),
                "jaw_width_ratio": round(float(jaw_width / ipd), 4),
                "nose_length_ratio": round(float(nose_length / ipd), 4)
            }
        }