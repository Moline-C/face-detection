#!/usr/bin/env python3
"""
Launcher script for Face Recognition App.
Run this from the project root directory.
"""
import sys
import os

# Join the directory containing this file with 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


from main import main

if __name__ == "__main__":
    main()