"""Runtime config: device selection and the pretrained-model registry.

Each registry entry names the checkpoint file, an optional download URL, and the
architecture constructor arguments needed to load it. URLs that point at stable
GitHub release assets download automatically; weights hosted only on Google
Drive must be placed in models/ by hand (see README).
"""
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(prefer="auto"):
    """Pick a torch device. 'auto' uses CUDA if available, else CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Super-resolution (Real-ESRGAN / RRDBNet)
# ---------------------------------------------------------------------------
SR_MODELS = {
    # --- RRDBNet family (ESRGAN generator) --------------------------------
    "realesrgan-x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4, "arch_type": "rrdbnet",
        "arch": dict(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
    },
    "realesrgan-x2plus": {
        "file": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2, "arch_type": "rrdbnet",
        "arch": dict(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32),
    },
    "realesrgan-x4plus-anime": {
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4, "arch_type": "rrdbnet",
        "arch": dict(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32),
    },
    # --- SRVGGNetCompact family (newer, faster, real-world tuned) ---------
    "realesr-general-x4v3": {  # recommended: newest general real-world model
        "file": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "scale": 4, "arch_type": "srvgg",
        "arch": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
    },
    "realesr-animevideov3": {  # newest anime/cartoon video model
        "file": "realesr-animevideov3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4, "arch_type": "srvgg",
        "arch": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"),
    },
    # --- SOTA transformers, loaded via spandrel (slow on CPU, but runnable) -
    # spandrel auto-detects the architecture from the checkpoint, so any
    # HAT / DAT / DRCT / SwinIR .pth "just works".  Requires: pip install spandrel
    "swinir-real-l-x4": {  # real-world SwinIR-Large GAN — auto-downloads
        "file": "SwinIR-L_realSR_x4_GAN.pth",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
               "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "scale": 4, "arch_type": "spandrel", "arch": {},
    },
    "spandrel": {  # generic: point --sr-weights at any HAT/DAT/DRCT/SwinIR .pth
        "file": "spandrel_model.pth", "url": None,
        "scale": 4, "arch_type": "spandrel", "arch": {},
    },
    # --- Diffusion SR (the only diffusion model practical on CPU) ----------
    # Stable Diffusion x4 Upscaler via diffusers. Weights are managed by the
    # HuggingFace cache (~3.5 GB on first use). Requires:
    #   pip install diffusers transformers accelerate safetensors
    "diffusion-x4": {
        "file": None, "url": None,
        "hf_id": "stabilityai/stable-diffusion-x4-upscaler",
        "scale": 4, "arch_type": "diffusion", "arch": {},
    },
}
DEFAULT_SR = "realesrgan-x4plus"

# ---------------------------------------------------------------------------
# Low-light (Zero-DCE)
# ---------------------------------------------------------------------------
LOWLIGHT_MODELS = {
    # Official weights are distributed in the Zero-DCE repo / Google Drive.
    # Drop the snapshot file in models/ as zerodce.pth to enable the deep path.
    "zerodce": {"file": "zerodce.pth", "url": None},
}

# ---------------------------------------------------------------------------
# Denoise / deblur (NAFNet)
# ---------------------------------------------------------------------------
DENOISE_MODELS = {
    "nafnet-sidd": {  # real-image denoising
        "file": "NAFNet-SIDD-width64.pth",
        "url": None,  # Google-Drive hosted; place manually to enable
        "arch": dict(width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
    },
    "nafnet-reds": {  # motion deblur
        "file": "NAFNet-REDS-width64.pth",
        "url": None,
        "arch": dict(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]),
    },
}

# ---------------------------------------------------------------------------
# Face restoration (GFPGAN)
# ---------------------------------------------------------------------------
FACE_MODELS = {
    "gfpgan-1.4": {
        "file": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    },
}
DEFAULT_FACE = "gfpgan-1.4"
