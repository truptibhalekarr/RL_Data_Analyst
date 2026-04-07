"""
server/app.py — Required entry point for OpenEnv multi-mode deployment.
Entry point: server.app:main
"""

import sys
import os
import uvicorn

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import app


def main():
    """Main entry point required by OpenEnv validator."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()