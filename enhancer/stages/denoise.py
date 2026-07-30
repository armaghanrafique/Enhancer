"""Denoise + deblur stage.

deep:      NAFNet (needs a NAFNet checkpoint in models/; heavy on CPU)
classical: Non-local-means / bilateral denoise followed by an unsharp-mask
           sharpen to counter blur. Fast and effective on CPU.
"""
import cv2
import numpy as np
import torch

from .base import Stage
from ..config import DENOISE_MODELS
from ..utils.download import resolve_weights
from ..utils.imageio_utils import to_tensor, to_image
from ..utils.tiling import tiled_forward


class DenoiseStage(Stage):
    name = "denoise"

    def _load_model(self):
        key = self.opts.get("model", "nafnet-sidd")
        cfg = DENOISE_MODELS[key]
        weights = resolve_weights(cfg["file"], cfg["url"], self.opts.get("weights"))
        if weights is None:
            return None
        from ..archs.nafnet import NAFNet
        model = NAFNet(**cfg["arch"]).to(self.device).eval()
        state = torch.load(weights, map_location=self.device)
        state = state.get("params", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=True)
        return model

    @torch.no_grad()
    def _process_deep(self, img):
        x = to_tensor(img, self.device)
        out = tiled_forward(self._model, x, scale=1,
                            tile=int(self.opts.get("tile", 256)), overlap=16)
        return to_image(out)

    @staticmethod
    def _median_ksize(img):
        """Auto-pick a median kernel from the fraction of impulse pixels.

        Counts near-black / near-white pixels (the signature of salt-and-pepper
        noise) and chooses a stronger median when there are more of them.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        frac = ((gray <= 12) | (gray >= 243)).mean()
        if frac > 0.12:
            return 5
        if frac > 0.04:
            return 3
        return 0

    def _process_classical(self, img):
        strength = float(self.opts.get("strength", 1.0))

        # median pre-filter removes salt-and-pepper / impulse noise that
        # Gaussian-oriented denoisers leave behind. ksize 0 = off; auto = detect.
        median = self.opts.get("median", "auto")
        ksize = self._median_ksize(img) if median == "auto" else int(median)
        if ksize >= 3:
            img = cv2.medianBlur(img, ksize if ksize % 2 else ksize + 1)

        # edge-preserving denoise
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h=3 * strength, hColor=3 * strength,
            templateWindowSize=7, searchWindowSize=21)

        # unsharp mask to restore edge crispness lost to blur/denoise
        amount = float(self.opts.get("sharpen", 0.6))
        if amount > 0:
            blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
            denoised = cv2.addWeighted(denoised, 1 + amount, blur, -amount, 0)
        return np.clip(denoised, 0, 255).astype(np.uint8)
