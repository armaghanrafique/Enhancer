# Enhancer — Quick Start

Image & video enhancement with three tools in one local web app:
**License-plate reader (ANPR)**, **face enhancement**, and **full-frame enhancement**.

## 1. Requirements
- Python **3.10 or 3.11** (64-bit) — from [python.org](https://www.python.org/downloads/)
- No system ffmpeg needed (bundled). Model weights download automatically on first use.

## 2. Install
```powershell
cd path\to\Enhancer
.\setup.ps1            # creates venv, installs deps, fixes OpenCV
```
If PowerShell blocks the script:
```powershell
Set-ExecutionPolicy -Scope Process -Bypass
.\setup.ps1
```

Manual alternative:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install "opencv-python==4.10.0.84"
```

## 3. Run the web app
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py          # opens http://localhost:8501
```
Pick a **Mode**, import an image or video from the sidebar:

| Mode | What it does |
|------|--------------|
| 🚗 **License Plates** | detect car → plate → de-skew → enhance → OCR (PaddleOCR); reads stream into the sidebar, download as JSON |
| 🙂 **Face Enhancement** | detect + restore faces (GFPGAN); shows original vs enhanced; dedups to one entry per person |
| 🖼️ **Full-Frame Enhance** | upscale/clean the whole image or video (Real-ESRGAN); download the enhanced result |

## 4. Command line (optional)
```powershell
# upscale an image 4x
python -m enhancer -i input.jpg -o out\out.png --stages denoise superres --sr-model realesr-general-x4v3 --outscale 4 --tile 0

# read plates in a video
python -m enhancer -i traffic.mp4 -o out\plates.mp4 --stages lowlight denoise plate --plate-mode scan --scan-outscale 4 --ocr --draw
```
See **COMMANDS.txt** for the full command reference.

## Notes
- First run downloads model weights to `models\` and the HuggingFace cache.
- CPU-only is supported (default). For an NVIDIA GPU, reinstall CUDA PyTorch and it will be used automatically.
- Whole-frame video SR is heavy on CPU — keep the upscale at 2× and use frame-skip.
