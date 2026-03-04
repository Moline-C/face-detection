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


    def _load_index(self):
        """Load the index.json file into memory."""
        # TODO: Check if index_file exists using .exists() method
        if self.index_file.exists():
            # Load existing index
            # TODO: Open file in read mode and load JSON
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            # Create new empty index
            self.index = {}
            self._save_index()


    def _save_index(self):
        """Save the in-memory index to index.json."""
        # TODO: Open file in write mode
        with open(self.index_file, 'w') as f:
            # TODO: Write index as JSON with indentation for readability
            json.dump(self.index, f, fp=f)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Convert display name to filesystem-safe name.

        Examples: "John Doe" -> "john_doe"
        """
        # TODO: Replace non-alphanumeric characters with underscores
        # Use a list comprehension: for each character c, keep if alphanumeric, else use "_"
        safe = "".join(c if c.isalnum() else "_" for c in name)

        # Remove consecutive underscores
        while "__" in safe:
            safe = safe.replace("__", "_")

        # TODO: Strip underscores from edges and convert to lowercase
        return safe.strip("_").lower()