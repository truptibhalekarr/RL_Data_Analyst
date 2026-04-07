"""
server/app.py — Required entry point for OpenEnv multi-mode deployment.
Imports and re-exports the FastAPI app from inference.py
"""

import sys
import os

# Add parent directory to path so we can import inference.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import app

__all__ = ["app"]