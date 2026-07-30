"""OCR readers for number-plate text.

Primary engine: EasyOCR (deep learning, strong accuracy, CPU-capable). It is an
optional dependency — install with `pip install easyocr`. Optional fallback:
Tesseract via pytesseract (needs the Tesseract binary installed).

`read(img_rgb)` returns a list of (text, confidence) tuples, best-effort.
"""
import re

import numpy as np


def _clean_plate(text):
    """Normalize OCR text to a plate-like token: upper-case alphanumerics."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_plate_like(text, min_len=4, max_len=10):
    """Heuristic filter for number-plate strings: short alphanumeric with a digit."""
    if not (min_len <= len(text) <= max_len):
        return False
    return any(c.isdigit() for c in text)


class EasyOCRReader:
    def __init__(self, langs=("en",)):
        import easyocr
        # gpu=False -> CPU; downloads its detection/recognition models on first use
        self.reader = easyocr.Reader(list(langs), gpu=False, verbose=False)

    def read(self, img_rgb):
        out = []
        for _box, text, conf in self.reader.readtext(img_rgb):
            cleaned = _clean_plate(text)
            if cleaned:
                out.append((cleaned, float(conf)))
        return out

    def detect_read(self, img_rgb):
        """Like read() but also returns each text region's bounding box.

        Returns [(box_xywh, text, conf)] where box is (x, y, w, h) pixels.
        """
        out = []
        for box, text, conf in self.reader.readtext(img_rgb):
            cleaned = _clean_plate(text)
            if not cleaned:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))
            out.append(((x0, y0, x1 - x0, y1 - y0), cleaned, float(conf)))
        return out


class PaddleOCRReader:
    """PaddleOCR (PP-OCR) — generally more accurate than EasyOCR on text/plates.

    Uses PaddleOCR 3.x. MKLDNN is disabled because its oneDNN backend crashes on
    some CPUs (AMD Ryzen) under the new PIR runtime.
    """
    def __init__(self, langs=("en",)):
        from paddleocr import PaddleOCR
        lang = "en"
        for l in langs:
            lang = "en" if l in ("en", "eng") else l
            break
        self.ocr = PaddleOCR(use_textline_orientation=False,
                             enable_mkldnn=False, lang=lang)

    def _predict(self, img_rgb):
        import cv2
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # paddle expects BGR
        res = self.ocr.predict(bgr)
        if not res:
            return [], [], []
        r = res[0]
        texts = r.get("rec_texts") or []
        scores = r.get("rec_scores") or []
        polys = r.get("rec_polys")
        if polys is None:
            polys = r.get("dt_polys") or []
        return texts, scores, polys

    def read(self, img_rgb):
        texts, scores, _ = self._predict(img_rgb)
        out = []
        for t, s in zip(texts, scores):
            c = _clean_plate(t)
            if c:
                out.append((c, float(s)))
        return out

    def detect_read(self, img_rgb):
        texts, scores, polys = self._predict(img_rgb)
        out = []
        for t, s, poly in zip(texts, scores, polys):
            c = _clean_plate(t)
            if not c:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
            out.append(((x0, y0, x1 - x0, y1 - y0), c, float(s)))
        return out


class TesseractReader:
    def __init__(self, langs=("eng",)):
        import pytesseract  # noqa: F401
        self._pt = pytesseract
        self.lang = "+".join("eng" if l == "en" else l for l in langs)

    def read(self, img_rgb):
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = self._pt.image_to_string(gray, lang=self.lang, config=cfg)
        cleaned = _clean_plate(text)
        return [(cleaned, 0.0)] if cleaned else []


def build_ocr(engine="easyocr", langs=("en",)):
    """Return an OCR reader, or None if the engine can't be loaded."""
    try:
        if engine in ("paddle", "paddleocr"):
            return PaddleOCRReader(langs)
        if engine == "easyocr":
            return EasyOCRReader(langs)
        if engine == "tesseract":
            return TesseractReader(langs)
        print(f"  ! [ocr] unknown engine '{engine}'")
    except ImportError:
        hints = {"easyocr": "pip install easyocr",
                 "paddle": "pip install paddlepaddle paddleocr",
                 "paddleocr": "pip install paddlepaddle paddleocr",
                 "tesseract": "pip install pytesseract"}
        print(f"  ! [ocr] {engine} not installed — run: "
              f"{hints.get(engine, engine)}  (continuing without OCR)")
    except Exception as e:  # noqa: BLE001
        print(f"  ! [ocr] could not init {engine} ({e}); continuing without OCR")
    return None
