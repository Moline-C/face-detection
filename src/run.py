#!/usr/bin/env python3
"""
Launcher script for Face Recognition App.
Run this from the project root directory.
"""
import sys
import os

# TODO: Add src directory to Python path
# Join the directory containing this file with 'src'
sys.path.insert(0, os.path.dirname(__file__))

# TODO: Import main function from src.main module
from main import main

if __name__ == "__main__":
    # Display startup message
    print("Starting Face Recognition App...")
    print("Press Ctrl+C to exit")

    # TODO: Launch the main application
    main()