"""ANPR — Automatic Number-Plate Recognition pipeline.

Flow per frame:
  1. detect vehicles (YOLOv8, COCO car/motorcycle/bus/truck)
  2. inside each vehicle box, locate the number-plate region
     - via a plate-detection YOLO model if supplied, else
     - via EasyOCR's text detector on an upscaled car crop
  3. crop that plate region from the ORIGINAL frame and enhance ONLY it,
     OCR-guided: several enhancement recipes (plain Lanczos upscale, CLAHE +
     unsharp, Real-ESRGAN) are each read with OCR and the one the OCR reads
     best is kept. A plain upscale is always in the pool, so enhancement can
     only ever *help* the read — it can never smear the plate the way a blind
     denoise->SR chain did.
  4. use that winning read -> predicted text (Pakistan-plate normalized)
  5. return structured detections + the original & enhanced plate crops +
     an annotated frame (vehicle boxes + plate boxes overlaid)

Everything is CPU-friendly and degrades gracefully if optional packages
(ultralytics / easyocr) are missing.
"""
import os
import re

import cv2
import numpy as np

from .config import get_device
from .stages.superres import SuperResStage
from .utils.ocr import build_ocr, is_plate_like
from .utils.deskew import deskew_plate

# COCO vehicle class ids
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Dedicated license-plate detector (YOLOv11), auto-downloaded from HuggingFace.
PLATE_HF_REPO = "morsetechlab/yolov11-license-plate-detection"
PLATE_HF_FILES = {
    "n": "license-plate-finetune-v1n.pt",   # fastest
    "s": "license-plate-finetune-v1s.pt",   # balanced (default)
    "m": "license-plate-finetune-v1m.pt",   # most accurate of the light set
}


def download_plate_model(size="s"):
    """Download the plate-detection weights from HuggingFace, return local path."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(PLATE_HF_REPO, PLATE_HF_FILES.get(size, PLATE_HF_FILES["s"]))

# Pakistan plate patterns. A plate is letters followed by digits (e.g. "LEB5700",
# "LEA123", "AB12") or the older digits-then-letters form. Crucially there is a
# hard *upper* bound — at most 3 letters and at most 4 digits — but fewer of
# either is allowed. Anything longer is not a valid plate token.
PK_MAX_LETTERS = 3
PK_MAX_DIGITS = 4
_PK_PATTERNS = [
    re.compile(r"^[A-Z]{1,3}[0-9]{1,4}$"),   # LEB5700 / LEA123 / AB12
    re.compile(r"^[0-9]{1,4}[A-Z]{1,3}$"),   # 1234ABC (older)
]


def normalize_pk_plate(text):
    """Clean OCR text to a Pakistan-plate-like token and flag if it matches."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    valid = any(p.match(t) for p in _PK_PATTERNS)
    # pretty form: letters-digits (e.g. LEB-5700)
    m = re.match(r"^([A-Z]+)([0-9]+)$", t)
    pretty = f"{m.group(1)}-{m.group(2)}" if m else t
    return t, pretty, valid


class ANPR:
    def __init__(self, device="auto", vehicle_weights="yolov8n.pt",
                 plate_weights=None, plate_size="s", plate_conf=0.20,
                 sr_model="realesr-general-x4v3", ocr_langs=("en",),
                 ocr_engine="paddle", plate_outscale=4, vehicle_conf=0.30,
                 enhance=True, deskew=True, verbose=True, roi=None,
                 reconstruct=True):
        self.device = get_device(device)
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.enhance = enhance
        self.deskew = deskew
        # render the recognized number as crisp glyphs under the enhanced crop
        self.reconstruct = reconstruct
        self.verbose = verbose      # print step-by-step pipeline logs
        # detection area as fractions of the frame (x1, y1, x2, y2) in 0..1, or
        # None for the whole frame. Only plates whose centre falls inside it are
        # enhanced + read. Mutable so the UI can change it without reloading models.
        self.roi = roi
        self._frame_idx = -1

        # -- vehicle detector (YOLO) --------------------------------------
        self.vehicle_model = None
        try:
            from ultralytics import YOLO
            self.vehicle_model = YOLO(vehicle_weights)
        except Exception as e:  # noqa: BLE001
            print(f"[anpr] vehicle detector unavailable ({e}); "
                  "install ultralytics for car detection")

        # -- dedicated plate detector (auto-downloads by default) ----------
        self.plate_model = None
        try:
            from ultralytics import YOLO
            weights = plate_weights or download_plate_model(plate_size)
            self.plate_model = YOLO(weights)
            print(f"[anpr] plate detector loaded ({os.path.basename(str(weights))})")
        except Exception as e:  # noqa: BLE001
            print(f"[anpr] dedicated plate detector unavailable ({e}); "
                  "falling back to OCR-based plate localization")

        # -- plate super-resolver -----------------------------------------
        # Real-ESRGAN is used as ONE OCR-guided enhancement candidate (see
        # `_enhance_and_read`), not as a blind fixed chain. The old
        # LowLight->Denoise->SR chain destroyed plates: the classical denoiser's
        # median blur smears the small digits of a (bright) plate, and the GAN
        # then hallucinates texture over the smear. We now let OCR pick the
        # enhancement that actually reads best.
        self._sr = None
        if enhance:
            self._sr = SuperResStage(self.device, deep=True, model=sr_model,
                                     tile=0, outscale=plate_outscale)

        # -- OCR ----------------------------------------------------------
        self.ocr = build_ocr(ocr_engine, langs=tuple(ocr_langs))

    def _log(self, msg):
        if self.verbose:
            print(f"[anpr] {msg}")

    # ----------------------------------------------------------------- API
    def process_frame(self, frame_rgb, draw=True):
        """Return (annotated_frame, [detections]).

        Cars are detected for context/overlay; plates are detected directly on
        the whole frame with the dedicated plate model (most reliable), each
        plate is enhanced + read, and associated to the car that contains it.

        Each detection: dict(car_box, plate_box, plate_orig, plate_enh, text,
        pretty, conf, valid, cls, det_conf).
        """
        annotated = frame_rgb.copy()
        detections = []
        h_img, w_img = frame_rgb.shape[:2]
        self._frame_idx += 1

        # detection area (region of interest); None -> whole frame
        roi = self._roi_rect(w_img, h_img)
        if draw and roi is not None:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
            cv2.putText(annotated, "DETECTION AREA", (rx1, max(16, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        self._log(f"frame {self._frame_idx}: detecting vehicles + plates "
                  f"({w_img}x{h_img})" + (f"  roi={roi}" if roi else ""))
        cars = self._detect_vehicles(frame_rgb)
        if draw:
            for (cx, cy, cw, ch, cls, cconf) in cars:
                cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch),
                              (255, 180, 0), 2)
                cv2.putText(annotated, f"{cls} {cconf:.2f}", (cx, max(14, cy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        # detect plates (whole frame with the dedicated model; else OCR fallback)
        plates = self._detect_plates(frame_rgb)
        if plates is None:
            self._log(f"  {len(cars)} vehicle(s); no plate model -> "
                      "locating plates via OCR inside each vehicle")
            plates = []
            for (cx, cy, cw, ch, _cls, _cc) in cars:
                pb = self._locate_plate(frame_rgb[cy:cy + ch, cx:cx + cw])
                if pb:
                    px, py, pw, ph = pb
                    plates.append((cx + px, cy + py, pw, ph, 0.0))
        else:
            self._log(f"  {len(cars)} vehicle(s), {len(plates)} plate(s) detected")

        for i, (px, py, pw, ph, pconf) in enumerate(plates):
            px, py = max(0, px), max(0, py)
            pw, ph = min(pw, w_img - px), min(ph, h_img - py)
            plate_orig = frame_rgb[py:py + ph, px:px + pw]
            if plate_orig.size == 0:
                continue

            # only enhance + read a plate once it has entered the detection area
            if not self._in_roi(px, py, pw, ph, roi):
                if draw:
                    cv2.rectangle(annotated, (px, py), (px + pw, py + ph),
                                  (120, 120, 120), 1)
                self._log(f"  plate #{i}: box=({px},{py},{pw}x{ph}) "
                          "outside detection area -> skipped")
                continue

            self._log(f"  plate #{i}: box=({px},{py},{pw}x{ph}) det={pconf:.2f}")
            # flatten a tilted plate, then let OCR choose the enhancement that
            # reads best (the shown crop is exactly the one that was read)
            plate_geo = deskew_plate(plate_orig) if self.deskew else plate_orig
            if self.deskew:
                tag = "warped flat" if plate_geo.shape != plate_orig.shape else "no skew found"
                self._log(f"    deskew: {tag}")
            plate_enh, text, pretty, conf, valid = self._enhance_and_read(plate_geo)
            cls = self._car_of(px + pw // 2, py + ph // 2, cars)
            self._log(f"    -> read: {pretty or '(not readable)'}"
                      f"  conf={conf:.2f}  PK={'yes' if valid else 'no'}")

            if draw:
                cv2.rectangle(annotated, (px, py), (px + pw, py + ph),
                              (0, 230, 0), 2)
                cv2.putText(annotated, pretty or "plate", (px, max(28, py - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)

            detections.append({
                "car_box": None, "cls": cls,
                "plate_box": [px, py, pw, ph],
                "plate_orig": plate_orig, "plate_enh": plate_enh,
                "text": text, "pretty": pretty, "conf": round(conf, 3),
                "valid": valid, "det_conf": round(pconf, 3),
            })
        return annotated, detections

    def _roi_rect(self, w_img, h_img):
        """Convert the fractional ROI (x1,y1,x2,y2 in 0..1) to a pixel rectangle
        clamped to the frame, or None for the whole frame."""
        if not self.roi:
            return None
        x1f, y1f, x2f, y2f = self.roi
        x1, x2 = sorted((int(x1f * w_img), int(x2f * w_img)))
        y1, y2 = sorted((int(y1f * h_img), int(y2f * h_img)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _in_roi(px, py, pw, ph, roi):
        """True if the plate's centre falls inside the detection area."""
        if roi is None:
            return True
        cx, cy = px + pw // 2, py + ph // 2
        x1, y1, x2, y2 = roi
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _detect_plates(self, frame_rgb):
        """Detect plate boxes on the whole frame. Returns [(x,y,w,h,conf)] or
        None if there is no dedicated plate model (caller uses OCR fallback)."""
        if self.plate_model is None:
            return None
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # YOLO expects BGR
        res = self.plate_model.predict(bgr, conf=self.plate_conf, verbose=False)
        out = []
        for r in res:
            if r.boxes is None:
                continue
            for b, cf in zip(r.boxes.xyxy.cpu().numpy(),
                             r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = b[:4]
                out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1),
                            float(cf)))
        return out

    def _car_of(self, cx, cy, cars):
        for (x, y, w, h, cls, _conf) in cars:
            if x <= cx <= x + w and y <= cy <= y + h:
                return cls
        return "vehicle"

    # ------------------------------------------------------------ internals
    def _detect_vehicles(self, frame_rgb):
        if self.vehicle_model is None:
            # no detector -> treat whole frame as one "region"
            h, w = frame_rgb.shape[:2]
            return [(0, 0, w, h, "frame", 1.0)]
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # YOLO expects BGR
        res = self.vehicle_model.predict(bgr, conf=self.vehicle_conf,
                                         classes=list(VEHICLE_CLASSES),
                                         verbose=False)
        out = []
        for r in res:
            if r.boxes is None:
                continue
            for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy(),
                                r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = b[:4]
                out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1),
                            VEHICLE_CLASSES.get(int(c), "vehicle"), float(cf)))
        return out

    def _locate_plate(self, car_crop):
        """Return plate box (x, y, w, h) in car-crop coords, or None."""
        # 1) dedicated plate YOLO
        if self.plate_model is not None:
            res = self.plate_model.predict(cv2.cvtColor(car_crop, cv2.COLOR_RGB2BGR),
                                           conf=self.plate_conf, verbose=False)
            best, best_area = None, 0
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b[:4]
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if best:
                return best

        # 2) fallback: EasyOCR text detector on an upscaled car crop
        if self.ocr is None:
            return None
        scale = 2
        up = cv2.resize(car_crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
        candidates = self.ocr.detect_read(up)
        # prefer plate-like tokens; else the largest text box in lower half
        plate_like = [c for c in candidates if is_plate_like(c[1], 4)]
        pool = plate_like or candidates
        if not pool:
            return None
        # pick highest-confidence candidate
        (x, y, w, h), _text, _conf = max(pool, key=lambda c: c[2])
        # map back to car-crop coords, pad a little
        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
        pad_x, pad_y = int(0.08 * w), int(0.3 * h)
        return (max(0, x - pad_x), max(0, y - pad_y), w + 2 * pad_x, h + 2 * pad_y)

    # ---- OCR-guided plate enhancement -----------------------------------
    # Height (px) at which small plate text reads comfortably. Crops shorter
    # than this are upscaled up to it; taller crops are left as-is.
    _OCR_TARGET_H = 96

    def _upscale(self, img):
        """Text-friendly Lanczos upscale to a comfortable OCR height. This is the
        non-destructive baseline candidate — it can only ever help OCR, never
        smear the digits the way median/NLM denoising did."""
        h, w = img.shape[:2]
        if h <= 0:
            return img
        scale = max(1.0, min(4.0, self._OCR_TARGET_H / h))
        if scale <= 1.0 + 1e-3:
            return img
        return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_LANCZOS4)

    def _contrast_sharpen(self, img):
        """Gentle local-contrast (CLAHE) + unsharp on the upscaled crop. Recovers
        faint / low-light text WITHOUT the blur that erased plate digits."""
        up = self._upscale(img)
        lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        up = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(up, (0, 0), 1.0)
        return np.clip(cv2.addWeighted(up, 1.5, blur, -0.5, 0),
                       0, 255).astype(np.uint8)

    # ---- display enhancement + digit reconstruction ---------------------
    # Minimum height (px) of the sharpened photo shown in the "Enhanced" panel.
    _DISPLAY_H = 180

    def _display_enhance(self, img):
        """A *strong*, always-visible sharpening pass for the crop shown in the
        Enhanced panel. Decoupled from the OCR-guided read: OCR still picks the
        text from the candidate pool, but the displayed image is deliberately
        pushed harder (bigger, denoised, local-contrast, unsharp) so the panel
        never looks identical to the raw crop."""
        up = self._upscale(img)
        h, w = up.shape[:2]
        if h < self._DISPLAY_H:                    # blow small crops up for display
            f = self._DISPLAY_H / max(1, h)
            up = cv2.resize(up, (max(1, int(round(w * f))), self._DISPLAY_H),
                            interpolation=cv2.INTER_LANCZOS4)
        den = cv2.bilateralFilter(up, 7, 55, 55)   # edge-preserving denoise
        lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
        den = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(den, (0, 0), 1.2)  # strong unsharp mask
        sharp = cv2.addWeighted(den, 1.9, blur, -0.9, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    @staticmethod
    def _fit_scale(text, font, max_w, max_h, thickness):
        """Largest cv2 font scale that fits `text` inside max_w x max_h."""
        (tw, th), base = cv2.getTextSize(text, font, 1.0, thickness)
        if tw <= 0 or th <= 0:
            return 1.0
        return min(max_w / tw, max_h / (th + base))

    def _digit_band(self, width, pretty, valid):
        """Render the recognized plate number as crisp, clean glyphs — literally
        'building the numbers' so the Enhanced panel shows unambiguous digits
        even when the source photo is blurry. Returns an RGB strip `width` wide."""
        band_h = max(64, int(round(width * 0.36)))
        band = np.full((band_h, width, 3), 245, np.uint8)   # clean off-white plate
        # thin top separator so it reads as a distinct reconstructed strip
        band[:3, :] = (0, 90, 40) if valid else (70, 70, 70)

        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = max(2, int(round(band_h / 22)))
        pad_w, pad_h = int(width * 0.06), int(band_h * 0.24)
        scale = self._fit_scale(pretty, font, width - 2 * pad_w,
                                band_h - 2 * pad_h, thickness)
        (tw, th), base = cv2.getTextSize(pretty, font, scale, thickness)
        org = ((width - tw) // 2, (band_h + th) // 2)
        colour = (15, 15, 15)                                # crisp near-black text
        cv2.putText(band, pretty, org, font, scale, colour, thickness,
                    cv2.LINE_AA)
        return band

    def _reconstruct_plate(self, best_img, pretty, valid):
        """Compose the Enhanced-panel image: a strongly sharpened photo of the
        plate with, when a number was read, a crisp reconstructed-digit band
        stacked beneath it. Falls back to just the sharpened photo when nothing
        was read."""
        photo = self._display_enhance(best_img)
        if not self.reconstruct or not pretty:
            return photo
        band = self._digit_band(photo.shape[1], pretty, valid)
        return np.vstack([photo, band])

    def _plate_candidates(self, plate_rgb):
        """Yield (name, image) enhancement candidates, cheapest first. A plain
        Lanczos upscale is always first, so no recipe can beat the honest
        baseline: enhancement can only *help* the read, never worsen it."""
        yield "upscale", self._upscale(plate_rgb)
        if self.enhance:
            yield "contrast", self._contrast_sharpen(plate_rgb)
            if self._sr is not None:
                try:
                    yield "superres", self._sr.process(plate_rgb)
                except Exception as e:  # noqa: BLE001
                    print(f"[anpr] super-resolution failed ({e}); skipping")

    @staticmethod
    def _score_read(text, conf, valid):
        """Rank an OCR result: a Pakistan-plate-pattern match outweighs
        confidence, which outweighs a small length tie-break; no read < 0."""
        if not text:
            return -1.0
        return (1.0 if valid else 0.0) + conf + min(len(text), 8) * 0.01

    def _enhance_and_read(self, plate_rgb):
        """OCR-guided enhancement. Read each candidate and keep the image the OCR
        reads best; return (best_image, text, pretty, conf, valid).

        Stops as soon as a candidate yields a valid, confident plate, so the
        expensive Real-ESRGAN pass is skipped whenever a cheap recipe already
        wins."""
        best = None  # (score, img, text, pretty, conf, valid)
        for name, img in self._plate_candidates(plate_rgb):
            text, pretty, conf, valid = self._read_plate(img)
            score = self._score_read(text, conf, valid)
            is_best = best is None or score > best[0]
            self._log(f"    candidate {name:8s}: "
                      f"{(pretty or '(no text)'):12s} conf={conf:.2f} "
                      f"valid={'yes' if valid else 'no '}"
                      f"{'   <== best so far' if is_best else ''}")
            if is_best:
                best = (score, img, text, pretty, conf, valid)
            if valid and conf >= 0.60:      # good enough — don't pay for more
                self._log(f"    (stopping early: '{name}' is a confident plate)")
                break
        _score, img, text, pretty, conf, valid = best
        # Build the Enhanced-panel image from the best-reading crop: a strongly
        # sharpened photo plus a crisp reconstructed-digit band, so the panel
        # shows clearly better digits instead of looking like the raw input.
        display = self._reconstruct_plate(img, pretty, valid)
        return display, text, pretty, conf, valid

    def _read_plate(self, plate_rgb):
        if self.ocr is None:
            return "", "", 0.0, False
        # detect_read gives every text box so a two-line plate (e.g. "LEB" over
        # "5700", with a "PUNJAB"/"06" tag) can be reassembled into one number
        dets = self.ocr.detect_read(plate_rgb)    # [((x,y,w,h), clean_text, conf)]
        if not dets:
            return "", "", 0.0, False
        best_text, best_conf = self._assemble_pk(dets)
        text, pretty, valid = normalize_pk_plate(best_text)
        return text, pretty, best_conf, valid

    @staticmethod
    def _assemble_pk(dets):
        """Build the most plausible Pakistan plate number from OCR text boxes.

        Handles two-line plates: it fuses the letter group (<=3 letters, e.g.
        "LEB") with the main number group (the 3-4 digit block, e.g. "5700"),
        ignoring a province name ("PUNJAB") or a small year tag ("06")."""
        toks = [(t, c) for (_b, t, c) in dets if t]
        if not toks:
            return "", 0.0

        # 1) a single token that is already a full plate wins outright
        full = [(t, c) for (t, c) in toks
                if any(p.match(t) for p in _PK_PATTERNS)
                and any(ch.isalpha() for ch in t)
                and any(ch.isdigit() for ch in t)]
        if full:
            return max(full, key=lambda x: x[1])

        # 2) otherwise fuse the best letter group with the best number group
        letters = [(t, c) for (t, c) in toks
                   if t.isalpha() and len(t) <= PK_MAX_LETTERS]
        digits = [(t, c) for (t, c) in toks
                  if t.isdigit() and len(t) <= PK_MAX_DIGITS]
        lt = max(letters, key=lambda x: x[1]) if letters else None
        # prefer the 3-4 digit registration block over a 1-2 digit year tag
        dt = (max(digits, key=lambda x: (3 <= len(x[0]) <= 4, x[1]))
              if digits else None)
        if lt and dt:
            return lt[0] + dt[0], min(lt[1], dt[1])

        # 3) fall back to the single highest-confidence token
        return max(toks, key=lambda x: x[1])
