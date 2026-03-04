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
        # Get MediaPipe solutions.face_mesh module
        self.mp_face_mesh = mp.solutions.face_mesh

        # Create FaceMesh object with these parameters:
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

        # Get MediaPipe drawing utilities
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
        # Convert BGR to RGB (required by face detection libraries)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get face locations using face_recognition.face_locations()
        # Use model="hog" for speed (or "cnn" for accuracy)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")

        # Get facial landmarks using MediaPipe face_mesh.process()
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
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate encoding using face_recognition.face_encodings()
        # Pass the rgb_image and a list containing the face_location
        encodings = face_recognition.face_encodings(rgb_image, [face_location])

        # Return first encoding if successful
        if encodings:
            return encodings[0]
        return None

    def draw_face_boxes(self, image: np.ndarray, face_locations: List) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.

        Args:
            image: BGR image
            face_locations: List of (top, right, bottom, left) tuples

        Returns:
            np.ndarray: Image with drawn boxes
        """
        # Create a copy to avoid modifying original
        output = image.copy()

        # Draw rectangle for each face
        for (top, right, bottom, left) in face_locations:
            # Draw rectangle using cv2.rectangle()
            # Parameters: image, (x1, y1), (x2, y2), color_BGR, thickness
            # Use green color (0, 255, 0) and thickness 2
            cv2.rectangle(output, (left, top), (right, bottom),
                        (0, 255, 0), 2)

        return output


    def draw_face_landmarks(self, image: np.ndarray, face_mesh_results) -> np.ndarray:
        """
        Draw MediaPipe facial landmarks (468 points).

        Args:
            image: BGR image
            face_mesh_results: MediaPipe face mesh results

        Returns:
            np.ndarray: Image with drawn landmarks
        """
        # Create a copy
        output = image.copy()

        # Check if faces were detected
        if face_mesh_results.multi_face_landmarks:
            # Process each detected face
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Draw tessellation (full mesh)
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Draw contours (emphasize features)
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        return output

    def crop_face(self, image: np.ndarray, face_location: Tuple, padding: int = 20) -> np.ndarray:
        """
        Crop face region with padding.

        Args:
            image: BGR image
            face_location: (top, right, bottom, left) tuple
            padding: Extra pixels around face

        Returns:
            np.ndarray: Cropped face image
        """
        top, right, bottom, left = face_location

        # Add padding but stay within image bounds
        height, width = image.shape[:2]

        # Add padding to top, ensuring >= 0
        top = max(0, top - padding)

        # Add padding to bottom, ensuring <= height
        bottom = min(height, bottom + padding)

        # Add padding to left, ensuring >= 0
        left = max(0, left - padding)

        # Add padding to right, ensuring <= width
        right = min(width, right + padding)

        # Crop using NumPy array slicing [y1:y2, x1:x2]
        return image[top:bottom, left:right]


    def cleanup(self):
        """Release MediaPipe resources."""
        # Close the face_mesh object
        self.face_mesh.close()

