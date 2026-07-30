"""Enhancement pipeline: builds the requested stages and runs them over images
or videos, in the fixed order  low-light -> denoise -> super-res -> face.
"""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import get_device
from .stages.lowlight import LowLightStage
from .stages.denoise import DenoiseStage
from .stages.superres import SuperResStage
from .stages.face import FaceStage
from .stages.plate import PlateStage
from .utils.imageio_utils import (read_image, write_image, IMG_EXTS, VIDEO_EXTS)
from .utils import video as videomod

# fixed processing order — correcting exposure and noise before upscaling, then
# refining objects (faces, plates) at the upscaled resolution, is cleanest.
STAGE_ORDER = ["lowlight", "denoise", "superres", "face", "plate"]
STAGE_CLASSES = {
    "lowlight": LowLightStage,
    "denoise": DenoiseStage,
    "superres": SuperResStage,
    "face": FaceStage,
    "plate": PlateStage,
}


class Enhancer:
    def __init__(self, stages, device="auto", deep=True, stage_opts=None):
        """`stages` is an iterable of stage names to enable."""
        self.device = get_device(device)
        stage_opts = stage_opts or {}
        ordered = [s for s in STAGE_ORDER if s in stages]
        self.stages = [
            STAGE_CLASSES[name](self.device, deep=deep, **stage_opts.get(name, {}))
            for name in ordered
        ]
        self.stage_names = ordered
        self.plate_stage = next((s for s in self.stages if s.name == "plate"), None)
        print(f"Device: {self.device}  |  Stages: {' -> '.join(ordered) or '(none)'}")

    def enhance_frame(self, img):
        for stage in self.stages:
            img = stage.process(img)
        return img

    def _dump_plate_log(self, dst):
        """Write detected/OCR'd plates to a JSON sidecar and print a summary."""
        ps = self.plate_stage
        if not ps or not ps.log:
            return
        import json
        path = str(dst) + ".plates.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ps.log, f, indent=2, ensure_ascii=False)
        reads = [pl["text"] for e in ps.log for pl in e["plates"] if pl["text"]]
        uniq = sorted(set(reads))
        print(f"Plates: {len(reads)} reads on {len(ps.log)} frame(s) -> {path}")
        if uniq:
            print("  detected texts: " + ", ".join(uniq[:20]) +
                  (" ..." if len(uniq) > 20 else ""))
        ps.reset_log()

    # -- images -------------------------------------------------------------
    def enhance_image(self, src, dst):
        img = read_image(src)
        out = self.enhance_frame(img)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        write_image(dst, out)
        print(f"Saved: {dst}  ({img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]})")
        self._dump_plate_log(dst)

    # -- video --------------------------------------------------------------
    def enhance_video(self, src, dst, smooth=0.0, keep_audio=True):
        fps, n = videomod.probe_video(src)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        writer = videomod.VideoWriter(dst, fps=fps)

        prev = None
        t0 = time.time()
        try:
            for i, frame in enumerate(tqdm(videomod.iter_frames(src), total=n,
                                           unit="f", desc="frames")):
                out = self.enhance_frame(frame)
                # temporal smoothing: blend with previous output to damp flicker
                if smooth > 0 and prev is not None and prev.shape == out.shape:
                    out = (smooth * prev.astype(np.float32)
                           + (1 - smooth) * out.astype(np.float32))
                    out = np.clip(out, 0, 255).astype(np.uint8)
                prev = out
                writer.append(out)
                # release per-frame buffers so memory doesn't creep up and OOM
                del frame
                if i % 8 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
        finally:
            writer.close()

        if keep_audio:
            videomod.remux_audio(dst, src)
        print(f"Saved: {dst}  ({time.time() - t0:.1f}s, {fps:.2f} fps source)")
        self._dump_plate_log(dst)

    # -- dispatch -----------------------------------------------------------
    def run(self, src, dst, **video_kw):
        src = Path(src)
        ext = src.suffix.lower()
        if ext in VIDEO_EXTS:
            self.enhance_video(src, dst, **video_kw)
        elif ext in IMG_EXTS:
            self.enhance_image(src, dst)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
