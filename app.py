"""Enhancer web app — license-plate reader, face enhancer, full-frame enhancer.

Run:
    streamlit run app.py
"""
import difflib
import gc
import logging
import os
import tempfile
import time
import warnings
from pathlib import Path

# ---- native-runtime safety (MUST be set before torch/paddle import) --------
# PaddleOCR and PyTorch each bundle their own Intel OpenMP (libiomp5md.dll).
# With both in one process OpenMP can abort with "OMP: Error #15 ... already
# initialized", killing the app with no Python traceback. This lets the two
# copies coexist. Harmless if only one runtime ends up loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- quiet noisy third-party startup chatter ------------------------------
# (HF "unauthenticated request" notices, paddle's ccache/model-creation spam,
#  etc.) so the terminal shows our own step-by-step pipeline logs clearly.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GLOG_minloglevel", "3")        # silence paddle C++ logs
warnings.filterwarnings("ignore")
for _noisy in ("ppocr", "paddleocr", "paddle", "paddlex", "ultralytics",
               "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import cv2
import numpy as np
import streamlit as st
import imageio.v2 as iio

from enhancer.anpr import ANPR
from enhancer.faces import FaceEnhancer
from enhancer.pipeline import Enhancer
from enhancer.utils import video as vmod

st.set_page_config(page_title="CLARITYX | OPTIFRAME", layout="wide")


# ---- theme / styling ------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
      :root {
        --bg:#0C1611; --panel:#122019; --panel-2:#16281E;
        --line:#1F3A2B; --accent:#2FA968; --accent-soft:#245C40;
        --text:#DEE9E3; --muted:#8AA394;
      }
      .stApp { background:var(--bg); }
      #MainMenu, footer, [data-testid="stStatusWidget"] { visibility:hidden; }
      header[data-testid="stHeader"] { background:transparent; height:0; }
      .block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1400px; }

      /* header bar */
      .app-header { display:flex; align-items:baseline; gap:.9rem;
        border-bottom:1px solid var(--line); padding-bottom:.85rem;
        margin-bottom:1.1rem; }
      .app-title { font-size:1.55rem; font-weight:800; letter-spacing:.14em;
        color:var(--text); }
      .app-title span { color:var(--accent); }
      .app-tag { font-size:.8rem; letter-spacing:.05em; color:var(--muted); }

      /* section + small labels */
      .section { font-size:.78rem; font-weight:700; letter-spacing:.16em;
        text-transform:uppercase; color:var(--muted); margin:1.4rem 0 .6rem; }
      .col-label { font-size:.72rem; font-weight:700; letter-spacing:.12em;
        text-transform:uppercase; color:var(--muted); margin-bottom:.35rem; }

      /* plate read-out */
      .plate-read { font-size:2.1rem; font-weight:800; letter-spacing:.08em;
        color:var(--text); line-height:1.15; }
      .plate-read.dim { color:var(--muted); font-weight:600; font-size:1.5rem; }
      .plate-meta { font-size:.82rem; color:var(--muted); margin-top:.3rem; }
      .pk { color:var(--accent); font-weight:700; }

      /* empty state */
      .empty { border:1px dashed var(--line); border-radius:14px;
        background:var(--panel); color:var(--muted); text-align:center;
        padding:4.5rem 1rem; font-size:.95rem; letter-spacing:.03em; }

      /* images */
      [data-testid="stImage"] img { border-radius:10px; border:1px solid var(--line); }

      /* buttons */
      .stButton>button, .stDownloadButton>button {
        background:var(--accent); color:#04120A; border:0; border-radius:9px;
        font-weight:700; letter-spacing:.02em; padding:.55rem 1.1rem;
        transition:filter .15s ease; width:100%; }
      .stButton>button:hover, .stDownloadButton>button:hover { filter:brightness(1.08); }

      /* sidebar */
      section[data-testid="stSidebar"] { background:var(--panel);
        border-right:1px solid var(--line); }
      section[data-testid="stSidebar"] .block-container { padding-top:1.4rem; }

      /* dropdowns (selectbox) — clearly outlined so they read as dropdowns */
      [data-baseweb="select"] > div { background:var(--panel-2);
        border:1.5px solid var(--accent-soft); border-radius:9px;
        transition:border-color .15s ease; }
      [data-baseweb="select"] > div:hover,
      [data-baseweb="select"] > div:focus-within { border-color:var(--accent); }
      [data-baseweb="select"] svg { color:var(--accent); fill:var(--accent); }
      [data-baseweb="popover"] [role="listbox"] { background:var(--panel-2);
        border:1px solid var(--accent-soft); border-radius:9px; }
      [data-baseweb="popover"] [role="option"] { color:var(--text); }
      /* text input — same outlined look */
      [data-testid="stTextInput"] input { background:var(--panel-2);
        border:1.5px solid var(--accent-soft); border-radius:9px; color:var(--text); }
      [data-testid="stTextInput"] input:focus { border-color:var(--accent); }

      /* file uploader */
      [data-testid="stFileUploaderDropzone"] { background:var(--panel-2);
        border:1px dashed var(--accent-soft); border-radius:12px; }
      [data-testid="stFileUploaderDropzone"] button { background:transparent;
        border:1px solid var(--accent); color:var(--accent); font-weight:700; }

      /* segmented control (mode) */
      [data-testid="stSegmentedControl"] { justify-content:flex-start; }

      /* expander */
      [data-testid="stExpander"] { border:1px solid var(--line);
        border-radius:12px; background:var(--panel); }
      [data-testid="stExpander"] summary { font-weight:700; color:var(--text); }

      /* metric */
      [data-testid="stMetricValue"] { color:var(--accent); }

      /* result rows in sidebar */
      .plate-row { border:1px solid var(--line); border-radius:10px;
        background:var(--panel-2); padding:.55rem .7rem; margin-bottom:.55rem; }
      .plate-row .id { font-size:1.05rem; font-weight:800; letter-spacing:.06em;
        color:var(--text); }
      .plate-row .c { font-size:.75rem; color:var(--muted); }

      /* progress */
      [data-testid="stProgress"] div[role="progressbar"] > div { background:var(--accent); }

      /* ---- welcome / splash screen (Pakistan patriot: dark green + white) ---- */
      .splash-overlay { position:fixed; inset:0; z-index:99990;
        background:radial-gradient(circle at 50% 34%, #0c4a2b 0%, #063d22 45%,
                   #04220f 78%, #020f08 100%);
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; text-align:center; overflow:hidden;
        animation:splashfade .6s ease; }
      @keyframes splashfade { from{opacity:0} to{opacity:1} }
      .splash-flag { position:absolute; top:0; left:0; right:0; height:7px;
        background:linear-gradient(90deg, #ffffff 0 24%, #01552b 24% 100%); }
      .splash-star { font-size:3.4rem; line-height:1; color:#eafff4;
        margin-bottom:1.2rem; text-shadow:0 0 30px rgba(63,220,140,.55); }
      .splash-title { font-size:3.6rem; font-weight:900; letter-spacing:.13em;
        color:#ffffff; margin:0; text-shadow:0 2px 30px rgba(0,0,0,.5); }
      .splash-title .x { color:#3ddb8a; }
      .splash-sub { margin-top:1rem; font-size:.92rem; letter-spacing:.34em;
        text-transform:uppercase; color:#9fdabc; }
      .splash-rule { width:120px; height:2px; margin:1.7rem auto 0;
        background:linear-gradient(90deg, transparent, #2FA968, transparent); }
      .splash-timer { margin-top:2.6rem; font-size:.86rem; letter-spacing:.05em;
        color:#8fc7ac; }
      .splash-timer b { color:#3ddb8a; font-size:1.05rem; }
      .splash-bar-wrap { width:240px; height:4px; margin:.9rem auto 0;
        background:rgba(255,255,255,.1); border-radius:99px; overflow:hidden; }
      .splash-bar { height:100%; background:linear-gradient(90deg,#2FA968,#3ddb8a);
        border-radius:99px; transition:width .9s linear; }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ---- welcome screen -------------------------------------------------------
# Shown once per session: a themed splash for CLARITYX | OPTIFRAME, then a
# 5-second countdown before the app redirects to the main workspace.
def _splash_html(remaining):
    pct = int((6 - remaining) / 5 * 100)     # 20% .. 100% as it counts down
    plural = "" if remaining == 1 else "s"
    return f"""
    <div class="splash-flag"></div>
    <div class="splash-overlay">
      <div class="splash-star">&#9770;</div>
      <h1 class="splash-title">CLARITY<span class="x">X</span>&nbsp;&nbsp;OPTI<span class="x">FRAME</span></h1>
      <div class="splash-sub">Plate Clarity &amp; Recognition</div>
      <div class="splash-rule"></div>
      <div class="splash-timer">Redirecting to the workspace in
        <b>{remaining}</b> second{plural}&hellip;</div>
      <div class="splash-bar-wrap"><div class="splash-bar"
        style="width:{pct}%"></div></div>
    </div>
    """


if not st.session_state.get("entered"):
    holder = st.empty()
    for remaining in range(5, 0, -1):
        holder.markdown(_splash_html(remaining), unsafe_allow_html=True)
        time.sleep(1)
    holder.empty()
    st.session_state["entered"] = True
    st.rerun()


# ---- cached model loaders -------------------------------------------------
# max_entries=1: keep only the most recent build of each pipeline. Without this,
# every mode switch / setting change leaves the previous heavy models resident
# for the whole session (e.g. GFPGAN x1 AND x2), which grows RAM until the
# process is OOM-killed with no traceback.
@st.cache_resource(show_spinner=False, max_entries=1)
def load_anpr(sr_model, plate_outscale, vehicle_conf, plate_conf, plate_size,
              enhance, deskew, ocr_engine, plate_weights):
    return ANPR(device="cpu", sr_model=sr_model, plate_outscale=plate_outscale,
                vehicle_conf=vehicle_conf, plate_conf=plate_conf,
                plate_size=plate_size, enhance=enhance, deskew=deskew,
                ocr_engine=ocr_engine, plate_weights=plate_weights or None)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_faces(upscale, only_center_face, crop_pad):
    return FaceEnhancer(device="cpu", upscale=upscale,
                        only_center_face=only_center_face, crop_pad=crop_pad)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_full(sr_model, outscale, stages, tile):
    opts = {"superres": {"model": sr_model, "tile": tile, "outscale": outscale},
            "denoise": {}, "lowlight": {}}
    return Enhancer(stages=list(stages), device="cpu", deep=True, stage_opts=opts)


def thumb(img_rgb, width=220):
    if img_rgb is None or img_rgb.size == 0:
        return np.zeros((40, width, 3), np.uint8)
    h, w = img_rgb.shape[:2]
    if w == 0:
        return img_rgb
    return cv2.resize(img_rgb, (width, max(1, int(h * width / w))),
                      interpolation=cv2.INTER_AREA if w > width else cv2.INTER_CUBIC)


def png_bytes(img_rgb):
    return cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def label(text):
    st.markdown(f"<div class='col-label'>{text}</div>", unsafe_allow_html=True)


# ---- header + mode --------------------------------------------------------
st.markdown(
    "<div class='app-header'>"
    "<div class='app-title'>CLARITY<span>X</span> | OPTI<span>FRAME</span></div>"
    "</div>",
    unsafe_allow_html=True)

mode = st.segmented_control(
    "Mode", ["License Plates", "Face Enhancement", "Full-Frame Enhance"],
    default="License Plates", label_visibility="collapsed")
mode = mode or "License Plates"
do_plates = "License" in mode
do_faces = "Face" in mode
do_full = "Full" in mode

# defaults so every name exists regardless of which section is shown
sr_model, plate_outscale, plate_size = "realesrgan-x4plus", 4, "m"
vehicle_conf, plate_conf = 0.3, 0.2
min_ocr_conf, min_plate_chars = 0.35, 5
ocr_engine, enhance, deskew, plate_weights = "paddle", True, True, ""
reconstruct = True
roi_on = False
roi_x1, roi_y1, roi_x2, roi_y2 = 0.20, 0.30, 0.80, 0.90
face_upscale, only_center_face, crop_pad, face_sim = 1, False, 0.35, 0.45
full_model, full_outscale, full_tile = "realesrgan-x4plus", 2, 0
full_denoise, full_lowlight = True, False

# Full-form option labels with a performance tier (Best / Medium / Advanced),
# shown in the dropdowns while the underlying value keys stay unchanged.
SR_MODEL_LABELS = {
    "realesrgan-x4plus":    "Real-ESRGAN ×4 Plus — Best (sharp general 4× upscaler)",
    "realesr-general-x4v3": "Real-ESRGAN General ×4 v3 — Medium (lighter & faster)",
    "realesrgan-x2plus":    "Real-ESRGAN ×2 Plus — Medium (quick 2× upscaler)",
    "diffusion-x4":         "Diffusion ×4 Upscaler — Advanced (slowest, highest quality)",
}
PLATE_DET_LABELS = {
    "n": "YOLOv11 Nano — Medium (lightest & fastest)",
    "s": "YOLOv11 Small — Best (balanced speed & accuracy)",
    "m": "YOLOv11 Medium — Advanced (most accurate, slower)",
}
OCR_ENGINE_LABELS = {
    "paddle":  "PaddleOCR — Best (most accurate, recommended)",
    "easyocr": "EasyOCR — Medium (lighter fallback engine)",
}
FACE_UPSCALE_LABELS = {
    1: "×1 — Best (native size, fastest)",
    2: "×2 — Advanced (2× larger output, slower)",
}
TILE_LABELS = {
    0:   "Off — Best quality (no tiling, needs most RAM)",
    512: "512 px — Medium (large tiles, moderate RAM)",
    256: "256 px — Advanced (low-RAM, slower)",
    128: "128 px — Advanced (lowest RAM, slowest)",
}

with st.expander("Settings", expanded=False):
    frame_skip = st.slider("Process every Nth video frame", 1, 30, 5, 1)

    if do_plates:
        c1, c2, c3 = st.columns(3)
        sr_model = c1.selectbox("Plate super-resolution model",
                                list(SR_MODEL_LABELS), index=0,
                                format_func=lambda v: SR_MODEL_LABELS[v])
        plate_outscale = c2.slider("Plate upscale", 2, 8, 4, 1)
        plate_size = c3.selectbox("Plate detector model", list(PLATE_DET_LABELS),
                                  index=2, format_func=lambda v: PLATE_DET_LABELS[v])
        c5, c6, c7, c8 = st.columns(4)
        vehicle_conf = c5.slider("Car confidence", 0.1, 0.9, 0.3, 0.05)
        plate_conf = c6.slider("Plate confidence", 0.05, 0.9, 0.20, 0.05)
        min_ocr_conf = c7.slider("Min OCR confidence", 0.0, 0.9, 0.35, 0.05)
        min_plate_chars = c8.slider("Min plate characters", 2, 9, 5, 1)
        d1, d2, d3, d4 = st.columns(4)
        ocr_engine = d1.selectbox("OCR engine", list(OCR_ENGINE_LABELS), index=0,
                                  format_func=lambda v: OCR_ENGINE_LABELS[v])
        enhance = d2.checkbox("Enhance plate region", value=True)
        deskew = d3.checkbox("De-skew tilted plates", value=True)
        reconstruct = d4.checkbox("Rebuild digits", value=True,
                                  help="Redraw the recognized number as crisp "
                                       "glyphs under the enhanced crop")
        plate_weights = st.text_input("Custom plate-detector weights (.pt)", "")

        roi_on = st.checkbox("Detection area", value=False)
        e1, e2, e3, e4 = st.columns(4)
        roi_x1 = e1.slider("Left (X1)", 0.0, 1.0, 0.20, 0.01, disabled=not roi_on)
        roi_y1 = e2.slider("Top (Y1)", 0.0, 1.0, 0.30, 0.01, disabled=not roi_on)
        roi_x2 = e3.slider("Right (X2)", 0.0, 1.0, 0.80, 0.01, disabled=not roi_on)
        roi_y2 = e4.slider("Bottom (Y2)", 0.0, 1.0, 0.90, 0.01, disabled=not roi_on)

    if do_faces:
        f1, f2, f3, f4 = st.columns(4)
        face_upscale = f1.selectbox("Face output upscale", list(FACE_UPSCALE_LABELS),
                                    index=0, format_func=lambda v: FACE_UPSCALE_LABELS[v])
        crop_pad = f2.slider("Crop margin", 0.0, 1.0, 0.35, 0.05)
        face_sim = f3.slider("Same-face threshold", 0.20, 0.90, 0.45, 0.05)
        only_center_face = f4.checkbox("Only main face", value=False)

    if do_full:
        g1, g2, g3 = st.columns(3)
        full_model = g1.selectbox("Super-resolution model", list(SR_MODEL_LABELS),
                                  index=0, format_func=lambda v: SR_MODEL_LABELS[v])
        full_outscale = g2.slider("Upscale factor", 1, 4, 2, 1)
        full_tile = g3.selectbox("Tile size (memory vs. speed)", list(TILE_LABELS),
                                 index=0, format_func=lambda v: TILE_LABELS[v])
        h1, h2 = st.columns(2)
        full_denoise = h1.checkbox("Denoise", value=True)
        full_lowlight = h2.checkbox("Low-light boost", value=False)

# free the OTHER features' models before loading the active one, so only one
# feature's models are ever resident. Switching features auto-releases the rest.
_LOADERS = {"plates": load_anpr, "faces": load_faces, "full": load_full}
active_mode = "plates" if do_plates else "faces" if do_faces else "full"
if st.session_state.get("active_mode") != active_mode:
    for _name, _loader in _LOADERS.items():
        if _name != active_mode:
            _loader.clear()                 # drop the cached model for that feature
    gc.collect()                            # actually reclaim the RAM
    st.session_state["active_mode"] = active_mode

# load only what the mode needs
anpr = load_anpr(sr_model, plate_outscale, vehicle_conf, plate_conf, plate_size,
                 enhance, deskew, ocr_engine, plate_weights) if do_plates else None
if anpr is not None:
    # set on the cached object so moving the sliders never reloads the models
    anpr.roi = (roi_x1, roi_y1, roi_x2, roi_y2) if roi_on else None
    anpr.reconstruct = reconstruct
face_enh = load_faces(face_upscale, only_center_face, crop_pad) if do_faces else None
if do_full:
    fstages = (["lowlight"] if full_lowlight else []) + \
              (["denoise"] if full_denoise else []) + ["superres"]
    full_enh = load_full(full_model, full_outscale, tuple(fstages), full_tile)
else:
    full_enh = None

if do_plates and anpr.ocr is None:
    st.error("OCR engine unavailable.")
if do_faces and face_enh.restorer is None:
    st.error("Face model unavailable.")

# ---- sidebar --------------------------------------------------------------
st.sidebar.markdown("<div class='section' style='margin-top:0'>Import</div>",
                    unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Image or video",
                                    type=["jpg", "jpeg", "png", "bmp", "webp",
                                          "mp4", "mov", "avi", "mkv", "webm"],
                                    label_visibility="collapsed")
st.sidebar.divider()
sidebar_ph = st.sidebar.empty()


def render_sidebar(best_plate, faces):
    with sidebar_ph.container():
        if do_full:
            return
        if do_plates:
            st.markdown("<div class='section' style='margin-top:0'>Final detection</div>",
                        unsafe_allow_html=True)
            if best_plate and best_plate["text"]:
                a, b = st.columns([1, 1])
                a.image(thumb(best_plate["plate_enh"], 120))
                with b:
                    pk = " <span class='pk'>PK</span>" if best_plate["valid"] else ""
                    st.markdown(
                        f"<div class='plate-row'><div class='id'>{best_plate['pretty']}</div>"
                        f"<div class='c'>conf {best_plate['conf']:.2f}{pk}</div></div>",
                        unsafe_allow_html=True)
        if do_faces:
            st.markdown("<div class='section' style='margin-top:0'>Detected faces</div>",
                        unsafe_allow_html=True)
            if faces:
                st.metric("Total", len(faces))
                for f in faces:
                    a, b = st.columns(2)
                    a.image(thumb(f["face_orig"], 100))
                    b.image(thumb(f["face_enh"], 100))


def plate_score(det, min_conf=0.0, min_chars=1):
    """Rank a plate read for choosing the single best across the whole clip:
    a valid PK pattern dominates confidence, which dominates length; reads below
    the confidence/length gates are rejected outright."""
    if not det["text"] or len(det["text"]) < min_chars or det["conf"] < min_conf:
        return -1.0
    return (1.0 if det["valid"] else 0.0) + det["conf"] + min(len(det["text"]), 8) * 0.01


def add_face(faces, det, thresh):
    emb = det.get("embedding")
    if emb is None:
        faces.append(det)
        return True
    for i, u in enumerate(faces):
        ue = u.get("embedding")
        if ue is not None and float(np.dot(emb, ue)) >= thresh:
            if det["quality"] > u["quality"]:
                faces[i] = det
                return True
            return False
    faces.append(det)
    return True


def run_frame(frame_rgb):
    if do_plates:
        return anpr.process_frame(frame_rgb)
    if do_faces:
        return face_enh.process_frame(frame_rgb)
    return frame_rgb, []


# ---- main -----------------------------------------------------------------
if uploaded is None:
    render_sidebar(None, [])
    st.markdown("<div class='empty'>Import an image or video from the panel on the left.</div>",
                unsafe_allow_html=True)
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
is_video = suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
frame_ph = st.empty()

# ===== FULL-FRAME ENHANCEMENT =============================================
if do_full:
    render_sidebar(None, [])
    if not is_video:
        data = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with st.spinner("Enhancing"):
            out = full_enh.enhance_frame(frame)
        a, b = st.columns(2)
        with a:
            label(f"Original  {frame.shape[1]}x{frame.shape[0]}")
            st.image(frame, width='stretch')
        with b:
            label(f"Enhanced  {out.shape[1]}x{out.shape[0]}")
            st.image(out, width='stretch')
        st.download_button("Download enhanced image", png_bytes(out),
                           "enhanced.png", "image/png")
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.close()
        reader = iio.get_reader(tmp.name, "ffmpeg")
        meta = reader.get_meta_data()
        src_fps = meta.get("fps", 30.0)
        total = meta.get("nframes")
        total = int(total) if total and total != float("inf") else None
        out_path = str(Path(tempfile.gettempdir()) / "enhanced_output.mp4")
        writer = vmod.VideoWriter(out_path, fps=max(1.0, src_fps / frame_skip))
        progress = st.progress(0.0)
        n = 0
        try:
            for i, frame in enumerate(reader):
                if i % frame_skip != 0:
                    continue
                out = full_enh.enhance_frame(frame[:, :, :3])
                writer.append(out)
                if n % 3 == 0:
                    frame_ph.image(out, width='stretch')
                n += 1
                if total:
                    progress.progress(min(1.0, i / total))
        finally:
            writer.close()
            reader.close()
        progress.progress(1.0)
        with open(out_path, "rb") as f:
            st.download_button("Download enhanced video", f.read(),
                               "enhanced.mp4", "video/mp4")
    gc.collect()                            # release per-frame buffers post-task
    st.stop()

# ===== DETECTION MODES (plates / faces) ===================================
# `state["best"]` is the single highest-scoring read (shown big at the top);
# `plates` accumulates EVERY distinct plate seen across the clip (deduped),
# so a multi-car video lists one row per car — see add_plate / handle below.
state = {"best": None, "score": -1e9}
plates = []
faces = []
render_sidebar(None, faces)

plate_slot = st.container()
face_slot = st.container()
if do_plates:
    with plate_slot:
        st.markdown("<div class='section'>Current plate</div>", unsafe_allow_html=True)
        pp1, pp2, pp3 = st.columns(3)
        with pp1:
            label("Original")
            orig_ph = st.empty()
        with pp2:
            label("Enhanced")
            enh_ph = st.empty()
        with pp3:
            label("Prediction")
            pred_ph = st.empty()
        pred_ph.markdown("<div class='plate-read dim'>Awaiting detection</div>",
                         unsafe_allow_html=True)
        st.markdown("<div class='section'>Confident detections &middot; &gt;90%</div>",
                    unsafe_allow_html=True)
        plates_ph = st.empty()
        plates_ph.markdown("<div class='plate-read dim'>None yet</div>",
                           unsafe_allow_html=True)
if do_faces:
    with face_slot:
        st.markdown("<div class='section'>Latest face</div>", unsafe_allow_html=True)
        fp1, fp2 = st.columns(2)
        with fp1:
            label("Original")
            forig_ph = st.empty()
        with fp2:
            label("Enhanced")
            fenh_ph = st.empty()


def show_plate(det):
    orig_ph.image(thumb(det["plate_orig"], 280))
    enh_ph.image(thumb(det["plate_enh"], 280))
    with pred_ph.container():
        if det["text"]:
            st.markdown(f"<div class='plate-read'>{det['pretty']}</div>",
                        unsafe_allow_html=True)
            pk = " &middot; <span class='pk'>PK</span>" if det["valid"] else ""
            st.markdown(f"<div class='plate-meta'>conf {det['conf']:.2f}{pk}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='plate-read dim'>Not readable</div>",
                        unsafe_allow_html=True)


def show_face(det):
    forig_ph.image(thumb(det["face_orig"], 300))
    fenh_ph.image(thumb(det["face_enh"], 300))


def _same_plate(a, b):
    """True if two OCR strings are the same physical plate. Exact match, or a
    near-match (OCR jitter like LEE5360 vs LEE536O / one dropped char), so the
    same car across many frames collapses to one row instead of many."""
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82


# Only reads the system is highly confident about make the final list.
CONFIDENT_THRESHOLD = 0.90


def add_plate(det):
    """Add a read to the running list of distinct plates, keeping the single
    best read per physical plate. Only *confident* reads (>= 90%) qualify, so
    the final list holds detections the system is sure about. Returns True if
    the list changed."""
    if det["conf"] < CONFIDENT_THRESHOLD:       # not confident enough — ignore
        return False
    s = plate_score(det, min_ocr_conf, min_plate_chars)
    if s < 0:                                   # below conf/length gate — ignore
        return False
    for i, p in enumerate(plates):
        if _same_plate(det["text"], p["text"]):
            if s > p["_score"]:                 # better read of a known plate
                det["_score"] = s
                plates[i] = det
                return True
            return False
    det["_score"] = s                           # a brand-new plate
    plates.append(det)
    return True


def render_plate_list():
    with plates_ph.container():
        if not plates:
            st.markdown("<div class='plate-read dim'>None yet</div>",
                        unsafe_allow_html=True)
            return
        for p in sorted(plates, key=lambda d: d["_score"], reverse=True):
            a, b = st.columns([1, 2])
            a.image(thumb(p["plate_enh"], 150))
            with b:
                pk = " <span class='pk'>PK</span>" if p["valid"] else ""
                st.markdown(
                    f"<div class='plate-row'><div class='id'>{p['pretty']}</div>"
                    f"<div class='c'>conf {p['conf']:.2f}{pk}</div></div>",
                    unsafe_allow_html=True)


def handle(dets_p, dets_f):
    for d in dets_p:
        # the top Original/Enhanced/Prediction panel always follows the CURRENT
        # plate being processed (live), not a frozen best
        show_plate(d)
        # track the single best read for the sidebar ...
        s = plate_score(d, min_ocr_conf, min_plate_chars)
        if s >= 0 and s > state["score"]:
            state["score"] = s
            state["best"] = d
            render_sidebar(state["best"], faces)
        # ... and accumulate every distinct plate for the full list
        if add_plate(d):
            render_plate_list()
    for d in dets_f:
        show_face(d)
        add_face(faces, d, face_sim)


if not is_video:
    data = np.frombuffer(uploaded.read(), np.uint8)
    frame = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with st.spinner("Processing"):
        annotated, dets = run_frame(frame)
    pdets = dets if do_plates else []
    fdets = dets if do_faces else []
    frame_ph.image(annotated, width='stretch')
    handle(pdets, fdets)
    render_sidebar(state["best"], faces)
else:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    reader = iio.get_reader(tmp.name, "ffmpeg")
    meta = reader.get_meta_data()
    total = meta.get("nframes")
    total = int(total) if total and total != float("inf") else None
    progress = st.progress(0.0)
    processed = 0
    for i, frame in enumerate(reader):
        if i % frame_skip != 0:
            continue
        frame = frame[:, :, :3]
        try:
            annotated, dets = run_frame(frame)
        except Exception as e:  # noqa: BLE001 — skip a bad frame, keep the run alive
            print(f"[app] frame {i} failed ({e}); skipping")
            continue
        pdets = dets if do_plates else []
        fdets = dets if do_faces else []
        frame_ph.image(annotated, width='stretch')
        handle(pdets, fdets)
        processed += 1
        if processed % 15 == 0:        # keep memory flat over long clips
            gc.collect()
        if total:
            progress.progress(min(1.0, i / total))
    reader.close()
    progress.progress(1.0)
    render_plate_list()
    render_sidebar(state["best"], faces)

gc.collect()                                # release per-frame buffers post-task

if do_plates and plates:
    import json
    rows = [{"text": p["text"], "plate": p["pretty"],
             "confidence": p["conf"], "valid_pk": p["valid"]}
            for p in sorted(plates, key=lambda d: d["_score"], reverse=True)]
    st.download_button(f"Download {len(rows)} plate(s) (JSON)",
                       json.dumps(rows, indent=2), "plates.json", "application/json")
