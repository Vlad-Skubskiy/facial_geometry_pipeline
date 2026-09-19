import cv2
import numpy as np
import mediapipe as mp

class FacialGeometryPipeline:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
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