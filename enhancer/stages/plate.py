"""License-plate enhancement + OCR stage.

For each detected plate the stage:
  1. crops the region (with padding),
  2. super-resolves the crop (Real-ESRGAN by default) for readable detail,
  3. optionally reads the text with OCR (EasyOCR),
  4. pastes the sharpened region back into the frame,
  5. optionally draws the box + text and/or saves the enhanced crop,
  6. records the reads in `self.log` (dumped to a JSON sidecar by the pipeline).

Works whether or not OCR is installed — without OCR it still detects and
sharpens plates.
"""
import os

import cv2

from .base import Stage
from .superres import SuperResStage
from ..utils.detect import build_detector
from ..utils.ocr import build_ocr, is_plate_like
from ..utils.imageio_utils import write_image


class PlateStage(Stage):
    name = "plate"

    def __init__(self, device, deep=True, **opts):
        super().__init__(device, deep=deep, **opts)
        self.log = []            # [{frame, plates:[{box,text,conf,all}]}]
        self.last_results = []
        self._frame_idx = -1
        self._detector = None
        self._ocr = None
        self._sr = None

    def _load_model(self):
        self.mode = self.opts.get("mode", "crop")   # "crop" | "scan"
        self._detector = build_detector(
            kind="plate",
            backend=self.opts.get("detector", "haar"),
            weights=self.opts.get("detector_weights"),
            conf=float(self.opts.get("det_conf", 0.25)),
        )
        # in "scan" mode we super-resolve the whole frame (smaller factor);
        # in "crop" mode we super-resolve each detected plate crop (bigger factor)
        sr_scale = (float(self.opts.get("scan_outscale", 2)) if self.mode == "scan"
                    else float(self.opts.get("crop_outscale", 4)))
        self._sr = SuperResStage(
            self.device, deep=self.deep,
            model=self.opts.get("sr_model", "realesr-general-x4v3"),
            tile=int(self.opts.get("tile", 0)), outscale=sr_scale,
        )
        if self.opts.get("ocr"):
            self._ocr = build_ocr(
                self.opts.get("ocr_engine", "easyocr"),
                langs=tuple(self.opts.get("ocr_langs", ("en",))))
        return self._detector  # non-None marks the stage "ready"

    def process(self, img):
        # the detector must load regardless of deep/classical (deep only affects
        # the super-resolver), so bypass the base deep-gating here.
        if self._detector is None:
            self._load_model()
            print(f"  - [plate] mode={self.mode}"
                  f"  detector={type(self._detector).__name__}"
                  f"  ocr={'on' if self._ocr else 'off'}  SR={self._sr.model_key}")
        if self.mode == "scan":
            return self._run_scan(img)
        return self._run(img)

    # unused (process is overridden) but kept for interface completeness
    def _process_deep(self, img):
        return self._run(img)

    def _process_classical(self, img):
        return self._run(img)

    def _run(self, img):
        self._frame_idx += 1
        out = img.copy()
        h_img, w_img = img.shape[:2]
        pad = int(self.opts.get("pad", 4))
        draw = self.opts.get("draw", False)
        crops_dir = self.opts.get("save_crops")

        frame_res = []
        for i, (x, y, w, h) in enumerate(self._detector.detect(img)):
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            enhanced = self._sr.process(crop)          # larger, sharper crop
            texts = self._ocr.read(enhanced) if self._ocr else []
            best_text, best_conf = (texts[0] if texts else ("", 0.0))
            for t, c in texts:                          # keep the highest-conf read
                if c > best_conf:
                    best_text, best_conf = t, c

            # paste the sharpened region back at its original box size
            out[y0:y1, x0:x1] = cv2.resize(
                enhanced, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)

            if crops_dir:
                os.makedirs(crops_dir, exist_ok=True)
                tag = f"_{best_text}" if best_text else ""
                write_image(os.path.join(
                    crops_dir, f"plate_f{self._frame_idx:05d}_{i}{tag}.png"), enhanced)

            if draw:
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if best_text:
                    cv2.putText(out, best_text, (x0, max(12, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_res.append({
                "box": [x0, y0, x1 - x0, y1 - y0],
                "text": best_text, "conf": round(best_conf, 3),
                "all": [{"text": t, "conf": round(c, 3)} for t, c in texts],
            })

        self.last_results = frame_res
        if frame_res:
            self.log.append({"frame": self._frame_idx, "plates": frame_res})
        return out

    def _run_scan(self, img):
        """Distant-plate mode: super-resolve the WHOLE frame, then OCR-scan it
        for plate-like text. Best when plates are small/far (traffic cameras)."""
        self._frame_idx += 1
        big = self._sr.process(img)          # upscaled, clearer frame
        out = big.copy()
        draw = self.opts.get("draw", False)
        min_len = int(self.opts.get("min_len", 4))
        crops_dir = self.opts.get("save_crops")

        frame_res = []
        if not self._ocr:
            print("  ! [plate] scan mode needs OCR — run: pip install easyocr")
        else:
            for i, ((x, y, w, h), text, conf) in enumerate(self._ocr.detect_read(big)):
                if not is_plate_like(text, min_len):
                    continue
                if draw:
                    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(out, text, (x, max(14, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if crops_dir:
                    os.makedirs(crops_dir, exist_ok=True)
                    write_image(os.path.join(
                        crops_dir, f"plate_f{self._frame_idx:05d}_{i}_{text}.png"),
                        big[max(0, y):y + h, max(0, x):x + w])
                frame_res.append({"box": [x, y, w, h], "text": text,
                                  "conf": round(conf, 3)})

        self.last_results = frame_res
        if frame_res:
            self.log.append({"frame": self._frame_idx, "plates": frame_res})
        return out

    def reset_log(self):
        self.log = []
        self.last_results = []
        self._frame_idx = -1
