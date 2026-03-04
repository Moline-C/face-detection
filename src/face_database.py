"""
Face database manager.
Handles saving, loading, and searching saved faces with embeddings.
"""
import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FaceDatabase:
    """Manages saved faces and their embeddings on the filesystem."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the face database.

        Args:
            data_dir: Root directory for storing face data
        """
        # Create Path object for data directory
        self.data_dir = Path(data_dir)

        # Create path for faces subdirectory
        self.faces_dir = self.data_dir / "faces"

        # Create path for index.json file
        self.index_file = self.data_dir / "index.json"

        # Create directories if they don't exist
        # Use mkdir with parents=True and exist_ok=True
        self.faces_dir.mkdir(parents=True, exist_ok=True)


        self.index = {}

        # Load existing index from disk
        self._load_index()