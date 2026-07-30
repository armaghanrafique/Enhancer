"""Face restoration stage (GFPGAN).

deep:      GFPGAN v1.4 — detects faces, restores them, pastes them back.
           Uses the optional `gfpgan` package (install with: pip install gfpgan).
classical: no reliable classical equivalent, so this stage becomes a no-op and
           logs a hint if the package/weights aren't available.

A compatibility shim registers `torchvision.transforms.functional_tensor` (which
newer torchvision removed) so GFPGAN's basicsr dependency imports cleanly.
"""
import sys
import types

import numpy as np

from .base import Stage
from ..config import FACE_MODELS, DEFAULT_FACE
from ..utils.download import resolve_weights


def _install_torchvision_shim():
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as F
        mod = types.ModuleType("torchvision.transforms.functional_tensor")
        mod.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = mod


class FaceStage(Stage):
    name = "face"

    def _load_model(self):
        _install_torchvision_shim()
        try:
            from gfpgan import GFPGANer
        except ImportError:
            print("  ! [face] `gfpgan` not installed — run: pip install gfpgan")
            return None

        cfg = FACE_MODELS.get(self.opts.get("model", DEFAULT_FACE))
        weights = resolve_weights(cfg["file"], cfg["url"], self.opts.get("weights"))
        if weights is None:
            return None

        # upscale=1: only restore faces; the SR stage handles overall upscaling
        return GFPGANer(model_path=str(weights), upscale=1, arch="clean",
                        channel_multiplier=2, bg_upsampler=None)

    def _process_deep(self, img):
        # GFPGANer works in BGR
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, _, restored = self._model.enhance(
            bgr, has_aligned=False, only_center_face=False, paste_back=True)
        if restored is None:
            return img
        return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

    def _process_classical(self, img):
        # nothing safe to do without the model; pass through unchanged
        return img
