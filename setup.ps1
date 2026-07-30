# ── Enhancer — one-time setup (Windows PowerShell) ──────────────────────────
# From the project folder run:   .\setup.ps1
# If PowerShell blocks it:        Set-ExecutionPolicy -Scope Process -Bypass  (then re-run)

$ErrorActionPreference = "Stop"

# Pick a Python launcher (prefer the 'py' launcher, fall back to 'python')
$launcher = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }

Write-Host "Creating virtual environment (venv)..." -ForegroundColor Cyan
& $launcher -m venv venv
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "ERROR: venv was not created. Is Python 3.10/3.11 installed and on PATH?" -ForegroundColor Red
    exit 1
}

# Call the venv interpreter DIRECTLY so packages always land in the venv,
# even if PowerShell blocks Activate.ps1 (running python.exe is not blocked).
$py = ".\venv\Scripts\python.exe"

& $py -m pip install --upgrade pip

# Choose CLI-only or the full web app
$app = Read-Host "Install the full WEB APP (plates/faces/full-frame)? [Y/n]"
if ($app -eq "" -or $app -match "^[Yy]") {
    Write-Host "Installing web-app requirements (this is large)..." -ForegroundColor Cyan
    & $py -m pip install -r requirements-app.txt

    Write-Host "Pinning OpenCV to 4.x (easyocr/ultralytics/paddle pull a broken v5)..." -ForegroundColor Cyan
    & $py -m pip uninstall -y opencv-python opencv-python-headless
    & $py -m pip install "opencv-python==4.10.0.84"
} else {
    Write-Host "Installing core (CLI) requirements..." -ForegroundColor Cyan
    & $py -m pip install -r requirements.txt
}

Write-Host ""
& $py -c "import cv2; print('OpenCV', cv2.__version__, '| Cascade:', hasattr(cv2,'CascadeClassifier'))"
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the venv in each new terminal, then run:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  Web app :  streamlit run app.py" -ForegroundColor Yellow
Write-Host "  CLI     :  python -m enhancer -i input.jpg -o out\out.png --stages superres" -ForegroundColor Yellow
Write-Host "See QUICKSTART.md and COMMANDS.txt for more." -ForegroundColor Green
