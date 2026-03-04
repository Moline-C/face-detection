"""
Face detection and recognition module.
Uses MediaPipe for face detection/landmarks and face_recognition for embeddings.
"""
import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from typing import List, Tuple, Optional


class FaceDetector:
    """Handles face detection, landmark detection, and face encoding."""

    def __init__(self):
        # TODO: Get MediaPipe solutions.face_mesh module
        self.mp_face_mesh = mp.solutions.face_mesh

        # TODO: Create FaceMesh object with these parameters:
        # - static_image_mode=False (for video)
        # - max_num_faces=10
        # - min_detection_confidence=0.5
        # - min_tracking_confidence=0.5
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode= False,
            max_num_faces= 10,
            min_detection_confidence= 0.5,
            min_tracking_confidence= 0.5
        )

        # TODO: Get MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_faces(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces and extract facial landmarks.

        Args:
            image: BGR image from OpenCV

        Returns:
            Tuple of (face_locations, face_landmarks)
        """
        # TODO: Convert BGR to RGB (required by face detection libraries)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TODO: Get face locations using face_recognition.face_locations()
        # Use model="hog" for speed (or "cnn" for accuracy)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")

        # TODO: Get facial landmarks using MediaPipe face_mesh.process()
        results = self.face_mesh.process(rgb_image)

        return face_locations, results

    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """
        Generate 128-dimensional face encoding.

        Args:
            image: BGR image from OpenCV
            face_location: (top, right, bottom, left) tuple

        Returns:
            np.ndarray: 128D encoding, or None if failed
        """
        # TODO: Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TODO: Generate encoding using face_recognition.face_encodings()
        # Pass the rgb_image and a list containing the face_location
        encodings = face_recognition.face_encodings(rgb_image, [face_location])

        # Return first encoding if successful
        if encodings:
            return encodings[0]
        return None