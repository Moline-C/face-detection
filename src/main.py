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

        # TODO: Check if user clicked OK and name is different
        if ok and new_name and new_name != self.display_name:
            # TODO: Emit the rename_requested signal
            self.rename_requested.emit(self.name, new_name)

            # Update label
            self.name_label.setText(new_name)
            self.display_name = new_name


class FaceRecognitionApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # TODO: Initialize core components
        self.detector = FaceDetector()
        self.database = FaceDatabase("data")
        self.camera = CameraHandler()
        # TODO: Initialize state variables
        self.current_mode = None  # 'live', 'image', or None
        self.current_image = None
        self.current_face_locations = []
        self.current_face_landmarks = None

        # TODO: Create QTimer for live feed updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_feed)

        # Setup UI and load faces
        self.setup_ui()
        self.load_saved_faces()

    def setup_ui(self):
        """Initialize the user interface."""

        # TODO: Set window title
        self.setWindowTitle("Face Recognition App")

        # TODO: Set window geometry (x, y, width, height)
        self.setGeometry(100, 100, 1200, 800)

        # TODO: Create and set central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # TODO: Create main horizontal layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # TODO: Create left vertical layout for display area
        left_layout = QVBoxLayout()

        # Create toolbar
        toolbar = QToolBar()



        # TODO: Create Live Feed button with emoji "📹"
        self.live_btn = QPushButton("📹 Live Feed")
        self.live_btn.clicked.connect(self.start_live_feed)
        toolbar.addWidget(self.live_btn)

        # TODO: Create Upload button with emoji "📁"
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        toolbar.addWidget(self.upload_btn)

        # TODO: Create Save Face button with emoji "💾"
        self.save_face_btn = QPushButton("💾 Save Face")
        self.save_face_btn.clicked.connect(self.save_current_face)
        self.save_face_btn.setEnabled(False)  # Disabled initially
        toolbar.addWidget(self.save_face_btn)

        left_layout.addWidget(toolbar)

        # TODO: Create display label for video/image
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.display_label.setText("Click 'Live Feed' or 'Upload Image' to start")
        left_layout.addWidget(self.display_label)

        # TODO: Add left_layout to main_layout with stretch factor 3
        main_layout.addLayout(left_layout, 3)

        # TODO: Create right vertical layout
        right_layout = QVBoxLayout()

        # Search label
        search_label = QLabel("Search Faces:")
        right_layout.addWidget(search_label)

        # TODO: Create search input field
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search...")
        self.search_input.textChanged.connect(self.on_search_changed)
        right_layout.addWidget(self.search_input)

        # TODO: Create scroll area for faces list
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(300)

        # TODO: Create container widget for faces
        self.faces_container = QWidget()
        self.faces_layout = QVBoxLayout()
        self.faces_layout.setAlignment(Qt.AlignTop)
        self.faces_container.setLayout(self.faces_layout)

        self.scroll_area.setWidget(self.faces_container)
        right_layout.addWidget(self.scroll_area)

        # TODO: Add right_layout to main_layout with stretch factor 1
        main_layout.addLayout(right_layout, 1)

        # TODO: Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")