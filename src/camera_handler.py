"""
Camera handler for live video capture.
"""
import cv2
import numpy as np
from typing import Optional


class CameraHandler:
    """Manages webcam video capture."""

    def __init__(self, camera_index: int = 0):
        # Store the camera_index parameter as an instance variable
        self.camera_index = camera_index

        # Initialize capture to None (will be set when camera starts)
        self.capture = None

        # Initialize running state flag to False
        self.is_running = False

    def start(self) -> bool:
        """
        Start the camera capture.

        Returns:
            bool: True if camera started successfully, False if failed
        """
        # If camera is already running, return success
        if self.capture is not None:
            return True

        # Initialize OpenCV VideoCapture with the camera_index
        self.capture = cv2.VideoCapture(self.camera_index)

        # Check if camera opened successfully using .isOpened() method
        if not self.capture.isOpened():
            # Failed - cleanup and return False
            self.capture = None
            return False

        # Set the is_running flag to True
        self.is_running = True
        return True

    def stop(self):
            """
            Stop the camera capture and release resources.
            """
            # Set is_running flag to False
            self.is_running = False

            # Release camera resources if active
            if self.capture is not None:
                # Call the release() method to free the camera
                self.capture.release()
                self.capture = None

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.

        Returns:
            np.ndarray: BGR image (H x W x 3), or None if failed
        """
        # Check if camera is initialized and running
        if self.capture is None or not self.is_running:
            return None

        # Read frame using capture.read() method
        # This returns (success_flag, frame_array)
        ret, frame = self.capture.read()

        # Return frame if successful, otherwise None
        if ret:
            return frame
        return None

    def get_frame_size(self) -> tuple:
        """
        Get current frame dimensions.

        Returns:
            tuple: (width, height) in pixels
        """
        # Return default if not initialized
        if self.capture is None:
            return (640, 480)

        # Query camera properties using cv2.CAP_PROP_FRAME_WIDTH
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Query camera properties using cv2.CAP_PROP_FRAME_HEIGHT
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return (width, height)



















































































































































































































































































        ing = False