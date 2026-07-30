# Enhancer — Image & Video Enhancement + ANPR + Face Restoration

A CPU-friendly toolkit that enhances the quality of **any** image or video, plus a
local web app with three modes:

- 🚗 **License Plates (ANPR)** — detect car → detect plate → de-skew → enhance → read (OCR)
- 🙂 **Face Enhancement** — detect + restore faces, with unique-face de-duplication
- 🖼️ **Full-Frame Enhance** — upscale/clean the whole image or video

Everything runs on CPU (no GPU required). Model weights download automatically on
first use. `ffmpeg` is bundled — no system install needed.

---

## Table of contents
1. [Models used](#models-used)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Run the web app](#run-the-web-app)
5. [Web app — modes & parameters](#web-app--modes--parameters)
6. [Command-line interface](#command-line-interface)
7. [CLI parameter reference](#cli-parameter-reference)
8. [Per-model commands (image + video)](#per-model-commands-image--video)
9. [Troubleshooting](#troubleshooting)
10. [Project layout](#project-layout)

---

## Models used

| Task | Model | Notes |
|------|-------|-------|
| Super-resolution | **Real-ESRGAN** (RRDBNet `x4plus`/`x2plus`, SRVGG `general-x4v3`, `animevideov3`) | auto-downloads; `general-x4v3` fastest, `x4plus` sharpest |
| Super-resolution (SOTA) | **HAT / DAT / DRCT / SwinIR** via `spandrel` | supply a `.pth`, load with `--sr-model spandrel` |
| Super-resolution (diffusion) | **Stable Diffusion x4 Upscaler** | `--sr-model diffusion-x4`; best quality, very slow |
| Denoise / deblur | **NAFNet** (deep) / OpenCV (classical) | classical fallback is default on CPU |
| Low-light | **Zero-DCE** (deep) / CLAHE+gamma (classical) | classical fallback by default |
| Vehicle detection | **YOLOv8** (`yolov8n`) | COCO car/motorcycle/bus/truck |
| Plate detection | **YOLOv11 license-plate** (`morsetechlab/...`) | auto-downloads from HuggingFace |
| Plate OCR | **PaddleOCR** (default) / **EasyOCR** | PaddleOCR is more accurate |
| Face restoration | **GFPGAN v1.4** | detects (RetinaFace) + restores faces |
| Face recognition (dedup) | **ArcFace** (via facexlib) | one entry per unique identity |

---

## Requirements

- **Python 3.10 or 3.11** (64-bit) — <https://www.python.org/downloads/>
- ~5 GB free disk for model weights (more if you use `diffusion-x4`)
- Windows / macOS / Linux (commands below are Windows PowerShell)

---

## Installation

### Option A — one command
```powershell
# Windows (PowerShell)
cd path\to\Enhancer
.\setup.ps1
```
```bash
# macOS / Linux
cd path/to/Enhancer
bash setup.sh
```
The setup script creates a venv, installs dependencies, and fixes the OpenCV conflict.
If PowerShell blocks `setup.ps1`: `Set-ExecutionPolicy -Scope Process -Bypass` then re-run.

### Option B — manual (full web app)
```powershell
cd path\to\Enhancer
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install -r requirements-app.txt

# REQUIRED: easyocr/ultralytics/paddle pull a broken OpenCV 5 — pin it to 4.x:
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install "opencv-python==4.10.0.84"

# verify (should print: OpenCV 4.10.0  Cascade: True)
python -c "import cv2; print('OpenCV', cv2.__version__, 'Cascade:', hasattr(cv2,'CascadeClassifier'))"
```

### Option C — CLI only (lighter, no web app)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### GPU (optional)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# then pass --device cuda on the CLI; the app auto-uses CPU
```

---

## Run the web app
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py            # opens http://localhost:8501
```
Import an image or video from the sidebar, pick a mode, and results stream in.

---

## Web app — modes & parameters

A shared control appears in every mode:

| Setting | Meaning | Default |
|---------|---------|---------|
| **Process every Nth video frame** | frame-skip for video (1 = every frame) | 5 |

### 🚗 License Plates

| Setting | Meaning | Default |
|---------|---------|---------|
| **Plate SR model** | model that sharpens the plate crop (`realesrgan-x4plus`, `realesr-general-x4v3`, `realesrgan-x2plus`, `diffusion-x4`) | x4plus |
| **Plate upscale** | how much to enlarge the plate before OCR (2–8×) | 4 |
| **Plate detector** | YOLOv11 size: `n` fastest, `s` balanced, `m` most accurate | m |
| **Car confidence** | min YOLO score to accept a car (0.1–0.9) | 0.30 |
| **Plate confidence** | min score to accept a plate (0.05–0.9); lower catches faint/far plates | 0.20 |
| **Min OCR confidence** | reject reads below this (filters garbage like "F") | 0.35 |
| **Min plate characters** | drop reads shorter than this (removes single-char noise) | 5 |
| **OCR engine** | `paddle` (PaddleOCR, more accurate) or `easyocr` | paddle |
| **Enhance plate region** | run low-light + denoise + SR on the crop | on |
| **De-skew tilted plates** | perspective-correct angled plates before OCR | on |
| **Custom plate weights** | optional path to your own plate `.pt` | — |

Output: car + plate boxes overlaid, original vs enhanced plate + predicted text, a
live **unique-plate** sidebar, and a **Download plate results (JSON)** button.

### 🙂 Face Enhancement

| Setting | Meaning | Default |
|---------|---------|---------|
| **Face output upscale** | enlarge the whole output frame (1× or 2×) | 1 |
| **Crop margin** | context around the face in the crop (0 = tight, higher = bigger window) | 0.35 |
| **Same-face threshold** | ArcFace similarity to treat two faces as the same person (higher = stricter) | 0.45 |
| **Only main face** | restore just the largest/most-prominent face | off |

Output: face boxes overlaid, **original vs enhanced** face (bigger window), and a
**unique-faces** sidebar (one best crop per identity).

### 🖼️ Full-Frame Enhance

| Setting | Meaning | Default |
|---------|---------|---------|
| **SR model** | `realesrgan-x4plus` (sharpest), `realesr-general-x4v3` (fast), `realesrgan-x2plus`, `diffusion-x4` (best, slow, images) | x4plus |
| **Upscale factor** | whole-frame magnification (1–4×) | 2 |
| **Tile size** | `0` = none (fastest); `512`/`256`/`128` bound memory on large frames | 0 |
| **Denoise** | clean grain before upscaling | on |
| **Low-light boost** | brighten dark footage | off |

Output: original vs enhanced (images), live preview (video), and a **Download
enhanced image / video** button.

---

## Command-line interface

```powershell
python -m enhancer -i <input> -o <output> [options]
```
`<input>` is an image, a video, or a folder (batch). See the full flag list below.

```powershell
# 4x upscale an image (fast model)
python -m enhancer -i input.jpg -o out\out.png --stages denoise superres --sr-model realesr-general-x4v3 --outscale 4 --tile 0

# enhance a whole video 2x
python -m enhancer -i input.mp4 -o out\out.mp4 --stages denoise superres --sr-model realesr-general-x4v3 --outscale 2 --tile 0 --smooth 0.2

# read number plates in a video (scan mode is best for small/far plates)
python -m enhancer -i traffic.mp4 -o out\plates.mp4 --stages lowlight denoise plate --plate-mode scan --scan-outscale 4 --ocr --draw --min-plate-len 5
```

---

## CLI parameter reference

**Core**

| Flag | Meaning |
|------|---------|
| `-i, --input` | input image, video, or folder |
| `-o, --output` | output file (or folder for a folder input) |
| `--stages` | any of `lowlight denoise superres face plate` (run in that fixed order), or `all` |
| `--target` | preset that picks stages: `general` / `face` / `plate` / `all` (overrides `--stages`) |
| `--device` | `auto` / `cpu` / `cuda` |
| `--no-deep` | disable deep models; classical fallbacks only (fast preview) |

**Super-resolution**

| Flag | Meaning |
|------|---------|
| `--sr-model` | `realesr-general-x4v3`, `realesrgan-x4plus`, `realesrgan-x2plus`, `realesr-animevideov3`, `diffusion-x4`, `spandrel` |
| `--sr-weights` | path to a custom `.pth` (use with `--sr-model spandrel` for HAT/DAT/DRCT/SwinIR) |
| `--outscale` | final zoom factor (2, 4, 1.5, …); default = model native scale |
| `--tile` | tile size; `0` = none (fastest, ≤720p); `128`/`96`/`64` = less memory |

**Denoise / low-light** (only when those stages run)

| Flag | Meaning | Default |
|------|---------|---------|
| `--denoise-strength` | classical denoise amount | 1.0 |
| `--sharpen` | unsharp amount after denoise | 0.6 |
| `--median` | salt-and-pepper filter: `auto`, `0`, or odd int (`3`/`5`) | auto |
| `--gamma` | low-light gamma; `0` = adaptive | 0 |

**Number-plate / OCR**

| Flag | Meaning | Default |
|------|---------|---------|
| `--ocr` | turn on OCR text reading (needs easyocr/paddle) | off |
| `--plate-mode` | `crop` (detect then enhance) or `scan` (whole-frame SR then OCR-scan; best for far plates) | crop |
| `--scan-outscale` | whole-frame upscale in scan mode (2–8) | 2 |
| `--crop-outscale` | upscale each detected plate crop (crop mode) | 4 |
| `--min-plate-len` | min characters to count text as a plate | 4 |
| `--detector` | `haar` (built-in) or `yolo` | haar |
| `--detector-weights` | YOLO `.pt` for `--detector yolo` | — |
| `--plate-sr-model` | SR model used on plate crops | realesr-general-x4v3 |
| `--ocr-engine` | `easyocr` or `tesseract` (CLI); web app also has `paddle` | easyocr |
| `--ocr-langs` | OCR languages | en |
| `--draw` | draw boxes + text on the output | off |
| `--save-crops` | folder to save each enhanced plate crop | — |

**Diffusion** (only with `--sr-model diffusion-x4`)

| Flag | Meaning | Default |
|------|---------|---------|
| `--steps` | denoising steps (60+ best, 15–20 faster) | 30 |
| `--guidance` | text guidance scale (0 = fastest) | 0 |
| `--prompt` | optional text guidance | "" |
| `--noise-level` | conditioning noise | 20 |

**Video**

| Flag | Meaning | Default |
|------|---------|---------|
| `--smooth` | temporal flicker reduction 0–1 (0.2 recommended) | 0 |
| `--no-audio` | do not re-mux the source audio | off |

---

## Per-model commands (image + video)

**realesr-general-x4v3** — fast, real-world (recommended on CPU)
```powershell
python -m enhancer -i in.jpg -o out\gen.png  --stages denoise superres --sr-model realesr-general-x4v3 --outscale 4 --tile 0
python -m enhancer -i in.mp4 -o out\gen.mp4  --stages denoise superres --sr-model realesr-general-x4v3 --outscale 2 --tile 0 --smooth 0.2
```

**realesrgan-x4plus** — maximum detail (slower)
```powershell
python -m enhancer -i in.jpg -o out\x4.png  --stages superres --sr-model realesrgan-x4plus --outscale 4 --tile 256
python -m enhancer -i in.mp4 -o out\x4.mp4  --stages superres --sr-model realesrgan-x4plus --outscale 2 --tile 128 --smooth 0.2
```

**realesr-animevideov3** — anime / cartoon
```powershell
python -m enhancer -i anime.png -o out\a.png --stages superres --sr-model realesr-animevideov3 --outscale 4 --tile 0
python -m enhancer -i anime.mp4 -o out\a.mp4 --stages superres --sr-model realesr-animevideov3 --outscale 4 --tile 0 --smooth 0.3
```

**diffusion-x4** — Stable Diffusion upscaler (best quality, very slow)
```powershell
python -m enhancer -i in.jpg -o out\d.png --stages superres --sr-model diffusion-x4 --tile 128 --steps 20
```

**spandrel** — HAT / DAT / DRCT / SwinIR (`pip install spandrel`, supply a .pth)
```powershell
python -m enhancer -i in.jpg -o out\hat.png --stages superres --sr-model spandrel --sr-weights models\HAT_SRx4.pth --tile 128
```

**Plate reading (image + video)**
```powershell
python -m enhancer -i car.jpg     -o out\car.png     --stages lowlight denoise plate --plate-mode scan --scan-outscale 4 --ocr --draw --min-plate-len 5
python -m enhancer -i traffic.mp4 -o out\traffic.mp4 --stages lowlight denoise plate --plate-mode scan --scan-outscale 4 --ocr --draw --min-plate-len 5 --smooth 0.2
```

**Face restoration (image + video)**
```powershell
python -m enhancer -i portrait.jpg -o out\face.png --target face --outscale 2 --tile 0
python -m enhancer -i people.mp4   -o out\face.mp4 --target face --outscale 2 --tile 0 --smooth 0.2
```

**Batch a folder**
```powershell
python -m enhancer -i .\inputs -o .\outputs --stages denoise superres --sr-model realesr-general-x4v3 --outscale 2 --tile 0
```

See **COMMANDS.txt** for a best-quality command sheet.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cv2 has no attribute CascadeClassifier` / OpenCV 5 pulled in | `pip uninstall -y opencv-python opencv-python-headless` then `pip install "opencv-python==4.10.0.84"` |
| PaddleOCR crash: `ConvertPirAttribute2RuntimeAttribute` | already handled in code (`enable_mkldnn=False`) — reinstall paddleocr if it persists |
| Killed with exit code 137 (out of memory) | lower `--tile` (128 → 96 → 64), lower `--outscale`, or reduce video resolution/fps |
| Video super-res very slow | use `realesr-general-x4v3`, `--outscale 2`, `--tile 0`, and raise the frame-skip |
| Plate detected but not read | source too dark/blurred/overexposed — no model can recover missing pixels; use a clearer frame |
| `gfpgan` / `basicsr` import error | the code installs a torchvision shim; if install fails, `pip install gfpgan` again |

---

## Project layout
```
Enhancer/
  app.py                 Streamlit web app (3 modes)
  enhancer/
    __main__.py          `python -m enhancer` entry point
    archs/               model architectures (RRDBNet, SRVGGNet, NAFNet, Zero-DCE)
    stages/              pipeline stages (lowlight, denoise, superres, face, plate)
    utils/               I/O, tiling, download, detection, OCR, de-skew, video
    anpr.py              vehicle→plate→enhance→OCR pipeline
    faces.py             GFPGAN face restoration + ArcFace dedup
    pipeline.py          stage orchestration (Enhancer)
    config.py            model registry + download URLs
    cli.py               command-line interface
  models/                model weights (auto-download on first use) + README
  examples/              small sample image/video to test with
  .streamlit/            web-app theme + UI config
  requirements.txt       core (CLI)
  requirements-app.txt   web app (adds ANPR / faces / OCR)
  setup.ps1              one-command setup (Windows)
  setup.sh               one-command setup (macOS / Linux)
  README.md              this file
  QUICKSTART.md          short getting-started guide
  COMMANDS.txt           best-quality command sheet
```

> **First run downloads model weights** (~0.6 GB for the defaults) to `models/`
> and the HuggingFace cache. No weights ship inside the zip — they fetch on demand.
