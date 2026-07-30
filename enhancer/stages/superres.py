"""Super-resolution / upscaling stage.

deep:      Real-ESRGAN (RRDBNet) with tiled inference — downloads automatically.
classical: high-quality Lanczos upscaling — instant fallback.

`outscale` controls the final magnification. The model produces a native scale
(x2 or x4); if outscale differs, the result is resized to match.
"""
import cv2
import torch

from .base import Stage
from ..config import SR_MODELS, DEFAULT_SR
from ..utils.download import resolve_weights
from ..utils.imageio_utils import to_tensor, to_image
from ..utils.tiling import tiled_forward


class SuperResStage(Stage):
    name = "superres"

    def __init__(self, device, deep=True, **opts):
        super().__init__(device, deep=deep, **opts)
        self.model_key = opts.get("model", DEFAULT_SR)
        self.native_scale = SR_MODELS[self.model_key]["scale"]
        self._outscale_explicit = "outscale" in opts
        self.outscale = float(opts.get("outscale", self.native_scale))

    def _load_model(self):
        cfg = SR_MODELS[self.model_key]
        arch_type = cfg.get("arch_type", "rrdbnet")

        # diffusion weights are managed by the HuggingFace cache, not models/
        if arch_type == "diffusion":
            return self._load_diffusion(cfg)

        weights = resolve_weights(cfg["file"], cfg["url"], self.opts.get("weights"))
        if weights is None:
            return None

        if arch_type == "spandrel":
            return self._load_spandrel(weights)

        if arch_type == "srvgg":
            from ..archs.srvggnet import SRVGGNetCompact
            model = SRVGGNetCompact(**cfg["arch"]).to(self.device).eval()
        else:
            from ..archs.rrdbnet import RRDBNet
            model = RRDBNet(**cfg["arch"]).to(self.device).eval()
        state = torch.load(weights, map_location=self.device)
        # Real-ESRGAN checkpoints store weights under "params_ema" or "params"
        if isinstance(state, dict):
            state = state.get("params_ema") or state.get("params") or state
        model.load_state_dict(state, strict=True)
        return model

    def _load_spandrel(self, weights):
        """Load any HAT/DAT/DRCT/SwinIR/... checkpoint via spandrel auto-detection."""
        try:
            from spandrel import ModelLoader, ImageModelDescriptor
        except ImportError:
            print("  ! [superres] spandrel not installed — run: pip install spandrel")
            return None
        descriptor = ModelLoader(device=self.device).load_from_file(str(weights))
        if not isinstance(descriptor, ImageModelDescriptor):
            print("  ! [superres] not a single-image SR model; skipping")
            return None
        descriptor.to(self.device).eval()
        # use the model's real scale (overrides the registry placeholder)
        self.native_scale = descriptor.scale
        if not self._outscale_explicit:
            self.outscale = float(descriptor.scale)
        arch_name = getattr(descriptor, "architecture", None)
        arch_name = getattr(arch_name, "name", arch_name)
        print(f"    spandrel loaded: {arch_name}  (x{descriptor.scale})")
        return descriptor

    def _load_diffusion(self, cfg):
        """Load the Stable Diffusion x4 Upscaler as a tile-friendly callable."""
        try:
            from diffusers import StableDiffusionUpscalePipeline
        except ImportError:
            print("  ! [superres] diffusers not installed — run: "
                  "pip install diffusers transformers accelerate safetensors")
            return None

        model_id = cfg["hf_id"]
        print(f"  > loading diffusion pipeline: {model_id}")
        print("    (first run downloads ~3.5 GB to the HuggingFace cache)")
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float32)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for fn in ("enable_attention_slicing", "enable_vae_tiling"):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass

        self._diffusion = True
        self.native_scale = 4
        if not self._outscale_explicit:
            self.outscale = 4.0
        return _DiffusionUpscaler(
            pipe,
            steps=int(self.opts.get("steps", 30)),
            guidance=float(self.opts.get("guidance", 0.0)),
            prompt=self.opts.get("prompt", ""),
            noise_level=int(self.opts.get("noise_level", 20)),
        )

    @torch.no_grad()
    def _process_deep(self, img):
        x = to_tensor(img, self.device)
        tile = int(self.opts.get("tile", 192))
        # the diffusion upscaler is trained on ~128px inputs; keep tiles small
        if getattr(self, "_diffusion", False):
            tile = 128 if (tile <= 0 or tile > 128) else tile
        out = tiled_forward(self._model, x, scale=self.native_scale,
                            tile=tile, overlap=16)
        result = to_image(out)
        return self._fit_outscale(img, result, self.native_scale)

    def _process_classical(self, img):
        h, w = img.shape[:2]
        tw, th = int(round(w * self.outscale)), int(round(h * self.outscale))
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    def _fit_outscale(self, src, result, produced_scale):
        if abs(self.outscale - produced_scale) < 1e-3:
            return result
        h, w = src.shape[:2]
        tw, th = int(round(w * self.outscale)), int(round(h * self.outscale))
        interp = cv2.INTER_AREA if self.outscale < produced_scale else cv2.INTER_LANCZOS4
        return cv2.resize(result, (tw, th), interpolation=interp)


class _DiffusionUpscaler:
    """Adapts the diffusers x4 upscale pipeline to a `model(tensor) -> tensor`
    callable so it drops straight into the tiling loop. Tensors are 1x3xHxW in
    [0,1]; internally the pipeline works on PIL images.
    """
    scale = 4

    def __init__(self, pipe, steps, guidance, prompt, noise_level):
        self.pipe = pipe
        self.steps = steps
        self.guidance = guidance
        self.prompt = prompt
        self.noise_level = noise_level

    @torch.no_grad()
    def __call__(self, x):
        from PIL import Image
        import numpy as np

        arr = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255)
        pil = Image.fromarray(arr.astype(np.uint8))
        out = self.pipe(
            prompt=self.prompt, image=pil,
            num_inference_steps=self.steps, guidance_scale=self.guidance,
            noise_level=self.noise_level).images[0]
        out_arr = np.asarray(out).astype(np.float32) / 255.0
        return torch.from_numpy(out_arr).permute(2, 0, 1).unsqueeze(0).to(x.device)
