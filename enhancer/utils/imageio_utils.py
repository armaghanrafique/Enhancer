"""Image read/write and tensor conversion helpers.

Images are handled as HxWx3 uint8 RGB numpy arrays everywhere in the app, and
converted to/from NCHW float tensors in [0,1] only at the model boundary.
"""
import cv2
import numpy as np
import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


def read_image(path):
    """Read an image file as RGB uint8 (HxWx3)."""
    data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path, img_rgb):
    """Write an RGB uint8 array to disk (unicode-safe)."""
    ext = "." + str(path).rsplit(".", 1)[-1]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise IOError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def to_tensor(img_rgb, device):
    """HxWx3 uint8 RGB -> 1x3xHxW float tensor in [0,1]."""
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def to_image(tensor):
    """1x3xHxW (or 3xHxW) float tensor in [0,1] -> HxWx3 uint8 RGB."""
    t = tensor.detach().float().cpu().clamp_(0, 1)
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr
