#!/usr/bin/env bash
# ── Enhancer — one-time setup (macOS / Linux) ───────────────────────────────
# From the project folder run:   bash setup.sh
set -e

# Pick a python3 launcher
if command -v python3 >/dev/null 2>&1; then PYBOOT=python3; else PYBOOT=python; fi

echo "Creating virtual environment (venv)..."
"$PYBOOT" -m venv venv
if [ ! -x "venv/bin/python" ]; then
    echo "ERROR: venv was not created. Is Python 3.10/3.11 installed?" >&2
    exit 1
fi

# Call the venv interpreter DIRECTLY so packages always land in the venv.
PY="venv/bin/python"

"$PY" -m pip install --upgrade pip

read -r -p "Install the full WEB APP (plates/faces/full-frame)? [Y/n] " app
if [[ -z "$app" || "$app" =~ ^[Yy] ]]; then
    echo "Installing web-app requirements (this is large)..."
    "$PY" -m pip install -r requirements-app.txt

    echo "Pinning OpenCV to 4.x (easyocr/ultralytics/paddle pull a broken v5)..."
    "$PY" -m pip uninstall -y opencv-python opencv-python-headless || true
    "$PY" -m pip install "opencv-python==4.10.0.84"
else
    echo "Installing core (CLI) requirements..."
    "$PY" -m pip install -r requirements.txt
fi

echo
"$PY" -c "import cv2; print('OpenCV', cv2.__version__, '| Cascade:', hasattr(cv2,'CascadeClassifier'))"
echo
echo "Setup complete."
echo "Activate the venv in each new terminal, then run:"
echo "  source venv/bin/activate"
echo "  Web app :  streamlit run app.py"
echo "  CLI     :  python -m enhancer -i input.jpg -o out/out.png --stages superres"
echo "See QUICKSTART.md and COMMANDS.txt for more."
