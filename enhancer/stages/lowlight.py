"""Low-light enhancement stage.

deep:      Zero-DCE curve estimation (needs models/zerodce.pth)
classical: CLAHE on the L channel + adaptive gamma — robust and instant on CPU.
"""
import cv2
import numpy as np
import torch

from .base import Stage
from ..config import LOWLIGHT_MODELS
from ..utils.download import resolve_weights
from ..utils.imageio_utils import to_tensor, to_image


class LowLightStage(Stage):
    name = "lowlight"

    def _load_model(self):
        cfg = LOWLIGHT_MODELS["zerodce"]
        weights = resolve_weights(cfg["file"], cfg["url"], self.opts.get("weights"))
        if weights is None:
            return None
        from ..archs.zerodce import ZeroDCE
        model = ZeroDCE().to(self.device).eval()
        state = torch.load(weights, map_location=self.device)
        state = state.get("params", state) if isinstance(state, dict) else state
        # strip a possible "module." prefix from DataParallel training
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        return model

    @torch.no_grad()
    def _process_deep(self, img):
        x = to_tensor(img, self.device)
        out = self._model(x)
        return to_image(out)

    def _process_classical(self, img):
        clip = float(self.opts.get("clahe_clip", 2.0))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # adaptive gamma: darker frames get a stronger lift
        mean = l.mean() / 255.0
        gamma = float(self.opts.get("gamma", 0.0)) or np.clip(0.6 + mean, 0.5, 1.0)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        l = cv2.LUT(l, lut)

        out = cv2.merge([l, a, b])
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
