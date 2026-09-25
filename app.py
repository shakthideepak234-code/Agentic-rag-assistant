"""
Hugging Face Spaces & Streamlit Application Entrypoint
"""
import runpy
import sys
from pathlib import Path

# Run streamlit_app.py as the primary application
if __name__ == "__main__" or "streamlit" in sys.modules:
    streamlit_app_path = Path(__file__).parent / "streamlit_app.py"
    runpy.run_path(str(streamlit_app_path), run_name="__main__")