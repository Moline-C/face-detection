"""
Main GUI application for face recognition.
Built with PySide6 (Qt for Python).
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# TODO: Import Qt widgets - fill in the missing widget names
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QScrollArea, QFileDialog,
    QStatusBar, QToolBar, QMessageBox, QInputDialog, QFrame
)

# TODO: Import Qt core classes
from PySide6.QtCore import Qt, QTimer, Signal, QSize

# TODO: Import Qt GUI classes
from PySide6.QtGui import QImage, QPixmap, QAction

# TODO: Import our custom modules
from face_detector import FaceDetector
from face_database import FaceDatabase
from camera_handler import CameraHandler


class FaceItemWidget(QFrame):
    """Widget representing a single saved face in the list."""

    # TODO: Define a signal that emits two strings (old_name, new_name)
    rename_requested = Signal(str, str)
    delete_requested = Signal(str)
    def __init__(self, name: str, display_name: str, face_path: str):
        super().__init__()

        # TODO: Store parameters as instance variables
        self.name = name
        self.display_name = display_name
        self.face_path = face_path

        # Create horizontal layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # TODO: Create thumbnail label with fixed size 64x64
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(64, 64)
        self.thumbnail.setScaledContents(True)
        self.load_thumbnail()
        layout.addWidget(self.thumbnail)

        # TODO: Create name label with display_name
        self.name_label = QLabel(display_name)
        self.name_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.name_label, 1)  # Stretch factor

        # TODO: Create rename button with emoji pencil "✏️"
        rename_btn = QPushButton("✏️")
        rename_btn.setFixedSize(30, 30)
        rename_btn.clicked.connect(self.on_rename_clicked)

        layout.addWidget(rename_btn)
        delete_btn = QPushButton("🗑️")
        delete_btn.setFixedSize(30, 30)
        delete_btn.clicked.connect(self.on_delete_clicked)
        layout.addWidget(delete_btn)

        self.setLayout(layout)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)

    def load_thumbnail(self):
        """Load and display face thumbnail."""
        # TODO: Check if face_path exists using Path().exists()
        if Path(self.face_path).exists():
            # TODO: Load image using cv2.imread()
            image = cv2.imread(self.face_path)

            if image is not None:
                # TODO: Resize to 64x64 using cv2.resize()
                image = cv2.resize(image, (64, 64))

                # TODO: Convert BGR to RGB for Qt
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                # TODO: Create QImage from numpy array data
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line,
                                   QImage.Format_RGB888)

                # TODO: Set pixmap on thumbnail label
                self.thumbnail.setPixmap(QPixmap.fromImage(qt_image))

    def on_rename_clicked(self):
        """Handle rename button click."""
        # TODO: Show input dialog using QInputDialog.getText()
        # Parameters: parent, title, label, text=default_value
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Face",
            "Enter new name:",
            text=self.display_name
        )

        if ok and new_name and new_name != self.display_name:

            self.rename_requested.emit(self.name, new_name)

            # Update label
            self.name_label.setText(new_name)
            self.display_name = new_name

    def on_delete_clicked(self):
        """Handles delete button click"""
        self.delete_requested.emit(self.name)

class FaceRecognitionApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()


        self.detector = FaceDetector()
        self.database = FaceDatabase("data")
        self.camera = CameraHandler()

        self.current_mode = None  # 'live', 'image', or None
        self.current_image = None
        self.current_face_locations = []
        self.current_face_landmarks = None


        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_feed)

        # Setup UI and load faces
        self.setup_ui()
        self.load_saved_faces()

    def setup_ui(self):
        """Initialize the user interface."""

        self.setWindowTitle("Face Recognition App")


        self.setGeometry(100, 100, 1200, 800)


        central_widget = QWidget()
        self.setCentralWidget(central_widget)


        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)


        left_layout = QVBoxLayout()

        # Create toolbar
        toolbar = QToolBar()



        # Create Live Feed button with emoji "📹"
        self.live_btn = QPushButton("📹 Live Feed")
        self.live_btn.clicked.connect(self.start_live_feed)
        toolbar.addWidget(self.live_btn)

        # Create Upload button with emoji "📁"
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        toolbar.addWidget(self.upload_btn)

        # Create Save Face button with emoji "💾"
        self.save_face_btn = QPushButton("💾 Save Face")
        self.save_face_btn.clicked.connect(self.save_current_face)
        self.save_face_btn.setEnabled(False)  # Disabled initially
        toolbar.addWidget(self.save_face_btn)

        left_layout.addWidget(toolbar)

        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.display_label.setText("Click 'Live Feed' or 'Upload Image' to start")
        left_layout.addWidget(self.display_label)


        main_layout.addLayout(left_layout, 3)


        right_layout = QVBoxLayout()

        # Search label
        search_label = QLabel("Search Faces:")
        right_layout.addWidget(search_label)


        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search...")
        self.search_input.textChanged.connect(self.on_search_changed)
        right_layout.addWidget(self.search_input)


        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(300)

        self.faces_container = QWidget()
        self.faces_layout = QVBoxLayout()
        self.faces_layout.setAlignment(Qt.AlignTop)
        self.faces_container.setLayout(self.faces_layout)

        self.scroll_area.setWidget(self.faces_container)
        right_layout.addWidget(self.scroll_area)


        main_layout.addLayout(right_layout, 1)


        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def start_live_feed(self):
        """Start or stop live camera feed."""

        if self.current_mode == 'live':
            # Stop camera
            self.timer.stop()
            self.camera.stop()
            self.current_mode = None
            self.live_btn.setText("📹 Live Feed")
            self.display_label.setText("Live feed stopped")
            self.save_face_btn.setEnabled(False)
            self.status_bar.showMessage("Live feed stopped")
            return


        if self.camera.start():
            self.current_mode = 'live'

            self.timer.start(30)

            self.live_btn.setText("⏹️ Stop Feed")
            self.status_bar.showMessage("Live feed started")
        else:
            QMessageBox.warning(self, "Camera Error", "Failed to start camera")

    def update_live_feed(self):
        """Update live feed frame (called by timer)."""

        frame = self.camera.read_frame()
        if frame is None:
            return

        self.current_image = frame.copy()

        self.current_face_locations, self.current_face_landmarks = \
            self.detector.detect_faces(frame)

        if self.current_face_locations:
            frame = self.detector.draw_face_boxes(frame, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
        else:
            self.save_face_btn.setEnabled(False)

        if self.current_face_landmarks:
            frame = self.detector.draw_face_landmarks(frame, self.current_face_landmarks)


        if self.current_face_locations:
            first_face = self.current_face_locations[0]


            encoding = self.detector.get_face_encoding(self.current_image, first_face)

            if encoding is not None:
                match = self.database.find_closest_match(encoding)

                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")

        self.display_frame(frame)

    def upload_image(self):
        """Upload and process a static image."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return  # User cancelled

        # Stop live feed if running
        if self.current_mode == 'live':
            self.timer.stop()
            self.camera.stop()
            self.live_btn.setText("📹 Live Feed")

        self.current_mode = 'image'

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image")
            return

        self.current_image = image.copy()
        self.current_face_locations, self.current_face_landmarks = \
            self.detector.detect_faces(image)


        if self.current_face_locations:
            image = self.detector.draw_face_boxes(image, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
        else:
            self.save_face_btn.setEnabled(False)
            QMessageBox.information(self, "No Faces", "No faces detected in image")

        if self.current_face_landmarks:
            image = self.detector.draw_landmarks(image, self.current_face_landmarks)

        if self.current_face_locations:
            first_face = self.current_face_locations[0]
            encoding = self.detector.get_face_encoding(self.current_image, first_face)

            if encoding is not None:
                match = self.database.find_closest_match(encoding)
                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")

        # Display image
        self.display_frame(image)

    def save_current_face(self):
        """Save the first detected face to the database."""

        if not self.current_face_locations or self.current_image is None:
            QMessageBox.warning(self, "No Face", "No face detected to save")
            return

        name, ok = QInputDialog.getText(self, "Save Face", "Enter person's name:")
        if not ok or not name:
            return

        # Get first face
        first_face = self.current_face_locations[0]

        face_image = self.detector.crop_face(self.current_image, first_face)


        encoding = self.detector.get_face_encoding(self.current_image, first_face)
        if encoding is None:
            QMessageBox.warning(self, "Error", "Failed to generate face encoding")
            return

        if self.database.save_face(name, face_image, encoding):
            QMessageBox.information(self, "Success", f"Face saved as '{name}'")
            self.load_saved_faces()  # Refresh list
        else:
            QMessageBox.warning(self, "Error", "Failed to save face")

    def load_saved_faces(self, search_query: str = ""):
        """Load and display saved faces in the sidebar."""


        for i in reversed(range(self.faces_layout.count())):
            widget = self.faces_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()


        if search_query:
            faces = self.database.search_faces(search_query)
        else:
            faces = self.database.get_all_faces()

        for face in faces:
            widget = FaceItemWidget(
                face['name'],
                face['display_name'],
                face['face_path']
            )

            widget.rename_requested.connect(self.on_face_renamed)
            widget.delete_requested.connect(self.on_face_deleted)
            self.faces_layout.addWidget(widget)


    def on_search_changed(self, text: str):
        """Handle search input changes."""
        self.load_saved_faces(text)


    def on_face_renamed(self, old_name: str, new_name: str):
        """Handle face rename request."""

        if self.database.rename_face(old_name, new_name):
            self.status_bar.showMessage(f"Renamed to '{new_name}'")

    def on_face_deleted(self, name: str):
        confirm = QMessageBox.question(
            self,
            "Delete Face",
            f"Delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            if self.database.delete_face(name):
                self.load_saved_faces()
                self.status_bar.showMessage(f"Deleted '{name}'")
            else:
                QMessageBox.warning(self, "Error", "Failed to delete face")

    def display_frame(self, frame: np.ndarray):
        """Display a frame in the GUI."""


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate resize to fit display
        h, w, ch = rgb_frame.shape
        display_width = self.display_label.width()
        display_height = self.display_label.height()

        # Calculate aspect ratio
        aspect = w / h
        if display_width / display_height > aspect:
            new_height = display_height
            new_width = int(new_height * aspect)
        else:
            new_width = display_width
            new_height = int(new_width / aspect)

        resized = cv2.resize(rgb_frame, (new_width, new_height))
        h, w, ch = resized.shape

        bytes_per_line = ch * w
        qt_image = QImage(resized.data, w, h, bytes_per_line,
                           QImage.Format_RGB888)

        self.display_label.setPixmap(QPixmap.fromImage(qt_image))


    def closeEvent(self, event):
        """Handle window close event - cleanup resources."""
        self.timer.stop()

        self.camera.stop()

        self.detector.cleanup()

        event.accept()

def main():
    """Main entry point for the application."""

    app = QApplication(sys.argv)

    window = FaceRecognitionApp()

    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()