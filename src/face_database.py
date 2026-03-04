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

    def save_face(self, name: str, face_image: np.ndarray, embedding: np.ndarray) -> bool:
        """
        Save a face image and its embedding.

        Args:
            name: Person's name (will be sanitized)
            face_image: Cropped face image (BGR)
            embedding: 128D face encoding

        Returns:
            bool: True if saved successfully
        """
        #Sanitize name for filesystem
        safe_name = self._sanitize_name(name)

        #Create directory path for this person
        person_dir = self.faces_dir / safe_name

        #Create directory
        person_dir.mkdir(parents = True, exist_ok= True)

        #Save face image as JPEG using cv2.imwrite()
        face_path = person_dir / "face.jpg"
        cv2.imwrite(str(face_path), face_image)

        #Save embedding as numpy binary file using np.save()
        embedding_path = person_dir / "embedding.npy"
        np.save(str(embedding_path), embedding)

        #Update index with metadata dictionary
        self.index[safe_name] = {
            "name": name,  # Original name
            "face_path": str(face_path),
            "embedding_path": str(embedding_path)
        }

        # Persist index to disk
        self._save_index()
        return True

    def get_all_faces(self) -> List[Dict]:
        """
        Get all saved faces with their information.

        Returns:
            List[Dict]: List of face dictionaries
        """
        faces = []

        # TODO: Iterate over index items (use .items() method)
        for safe_name, data in self.index.items():
            faces.append({
                "name": safe_name,
                # TODO: Get display_name from data, default to safe_name if missing
                "display_name": data.get("name", safe_name),
                "face_path": data["face_path"],
                "embedding_path": data["embedding_path"]
            })

        return faces

    def load_embedding(self, name: str) -> Optional[np.ndarray]:
        """
        Load embedding for a specific person.

        Args:
            name: Person's safe name

        Returns:
            np.ndarray: 128D embedding, or None if not found
        """
        # TODO: Check if name exists in index
        if name in self.index:
            embedding_path = self.index[name]["embedding_path"]

            # TODO: Check if file exists using os.path.exists()
            if os.path.exists(embedding_path):
                # TODO: Load numpy array using np.load()
                return np.load(embedding_path)

        return None

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load all saved embeddings into memory.

        Returns:
            Dict mapping safe_name to 128D embedding
        """
        embeddings = {}

        # TODO: Iterate over index keys
        for safe_name in self.index.keys():
            embedding = self.load_embedding(safe_name)
            if embedding is not None:
                embeddings[safe_name] = embedding

        return embeddings

    def find_closest_match(self, query_embedding: np.ndarray,
                           threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find the closest matching face in the database.

        Uses Euclidean distance. Typical values:
        - Same person: 0.0 - 0.4
        - Different people: 0.6 - 1.2

        Args:
            query_embedding: 128D encoding to match
            threshold: Maximum distance to consider a match (default: 0.6)

        Returns:
            Tuple of (display_name, distance) if match found, else None
        """
        # Load all embeddings
        embeddings = self.load_all_embeddings()

        if not embeddings:
            return None

        # Initialize tracking variables
        best_match = None
        best_distance = float('inf')  # Start with infinity




        # TODO: Compare query against each saved face
        for name, saved_embedding in embeddings.items():
            # TODO: Calculate Euclidean distance using np.linalg.norm()
            # Distance = norm(query_embedding - saved_embedding)
            distance = np.linalg.norm(query_embedding - saved_embedding)

            # TODO: Update best match if this distance is lower
            if distance < best_distance:
                best_distance = distance
                best_match = name

        # TODO: Only return match if within threshold
        if best_distance <= threshold:
            # Get display name
            display_name = self.index[best_match].get("display_name", best_match)
            return (display_name, best_distance)

        return None

    def rename_face(self, old_name: str, new_name: str) -> bool:
        """
        Rename a saved face (display name only).

        Args:
            old_name: Current safe name
            new_name: New display name

        Returns:
            bool: True if successful
        """
        # TODO: Check if old_name exists in index
        if old_name not in self.index:
            return False

        # TODO: Update display_name in index
        self.index[old_name]["display_name"] = new_name

        self._save_index()
        return True

    def delete_face(self, name: str) -> bool:
        """
        Delete a saved face and its data.

        Args:
            name: Safe name of person to delete

        Returns:
            bool: True if successful
        """
        # TODO: Check if name exists
        if name not in self.index:
            return False

        # Remove from filesystem
        person_dir = self.faces_dir / name
        if person_dir.exists():
            import shutil
            # TODO: Recursively delete directory using shutil.rmtree()
            shutil.rmtree(person_dir)

        # TODO: Remove from index using del keyword
        del self.index[name]

        self._save_index()
        return True

    def search_faces(self, query: str) -> List[Dict]:
        """
        Search for faces by name (case-insensitive substring match).

        Args:
            query: Search string

        Returns:
            List[Dict]: Matching face dictionaries
        """
        # TODO: Convert query to lowercase
        query_lower = query.lower()
        results = []

        # Search through all faces
        for face in self.get_all_faces():
            # TODO: Check if query is substring of display_name (case-insensitive)
            if query_lower in face["display_name"].lower():
                results.append(face)

        return results