"""Face detection + enhancement pipeline (GFPGAN) with ArcFace identity dedup.

Per frame: detect faces, restore/sharpen each with GFPGAN, crop a padded window
(a bit larger than the tight face) from both the original and enhanced frame,
and compute an ArcFace embedding so the same person can be de-duplicated into a
single best detection across frames.
"""
import cv2
import numpy as np
import torch

from .config import get_device, FACE_MODELS, DEFAULT_FACE
from .utils.download import resolve_weights
from .stages.face import _install_torchvision_shim


def _load_recognition():
    """Load facexlib ArcFace on CPU, working around its CUDA-forcing bug
    (it ignores device='cpu' and torch.loads CUDA storages)."""
    try:
        from facexlib.recognition import init_recognition_model
    except Exception:
        return None
    _to, _cuda, _load = torch.nn.Module.to, torch.nn.Module.cuda, torch.load

    def safe_to(self, *a, **k):
        a = tuple('cpu' if (isinstance(x, str) and x.startswith('cuda')) else x for x in a)
        if isinstance(k.get('device'), str) and k['device'].startswith('cuda'):
            k['device'] = 'cpu'
        return _to(self, *a, **k)

    def safe_load(*a, **k):
        k.setdefault('map_location', 'cpu')
        return _load(*a, **k)

    torch.nn.Module.to = safe_to
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.load = safe_load
    try:
        return init_recognition_model('arcface', device='cpu').eval()
    except Exception as e:  # noqa: BLE001
        print(f"[faces] recognition model unavailable ({e})")
        return None
    finally:
        torch.nn.Module.to, torch.nn.Module.cuda, torch.load = _to, _cuda, _load


class FaceEnhancer:
    def __init__(self, device="auto", model=DEFAULT_FACE, upscale=1,
                 only_center_face=False, crop_pad=0.35, recognize=True,
                 max_side=1280):
        self.device = get_device(device)
        self.upscale = upscale
        self.only_center_face = only_center_face
        self.crop_pad = crop_pad
        # cap the working resolution: GFPGAN on full 1080p/4K frames allocates
        # huge tensors and can OOM-kill the process, especially once faces are
        # found (each triggers a 512px restoration pass + paste-back buffers).
        self.max_side = int(max_side)
        self.restorer = None
        self.recog = None
        _install_torchvision_shim()
        try:
            from gfpgan import GFPGANer
        except ImportError:
            print("[faces] gfpgan not installed — run: pip install gfpgan")
            return
        cfg = FACE_MODELS.get(model, FACE_MODELS[DEFAULT_FACE])
        weights = resolve_weights(cfg["file"], cfg["url"])
        if weights is None:
            print("[faces] GFPGAN weights unavailable")
            return
        self.restorer = GFPGANer(model_path=str(weights), upscale=upscale,
                                 arch="clean", channel_multiplier=2,
                                 bg_upsampler=None)
        print(f"[faces] GFPGAN loaded ({cfg['file']}, upscale x{upscale})")
        if recognize:
            self.recog = _load_recognition()
            if self.recog is not None:
                print("[faces] ArcFace loaded — unique-face dedup enabled")

    def embed(self, aligned_bgr):
        """512-d L2-normalized ArcFace embedding for an aligned face crop."""
        if self.recog is None:
            return None
        f = cv2.resize(aligned_bgr, (112, 112))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32)
        f = (f / 255.0 - 0.5) / 0.5
        t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            e = self.recog(t).cpu().numpy().flatten()
        n = np.linalg.norm(e)
        return (e / n).astype(np.float32) if n > 0 else e

    def _limit_size(self, img_rgb):
        """Downscale so the longer side is <= max_side (keeps CPU/RAM bounded).
        All downstream coords stay in this reduced space, so crops stay valid."""
        h, w = img_rgb.shape[:2]
        m = max(h, w)
        if m <= self.max_side:
            return img_rgb
        scale = self.max_side / m
        return cv2.resize(img_rgb, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def process_frame(self, img_rgb, draw=True):
        """Return (annotated_frame, [detections]).

        ONLY the face regions are enhanced: GFPGAN restores each detected face to
        a clean 512x512 aligned crop (what's shown in the list). The costly
        whole-frame paste-back is skipped, so the returned frame is the original
        with face boxes drawn for the live preview.

        Each detection: dict(box, face_orig, face_enh, embedding, quality) where
        face_orig is the aligned original face and face_enh the restored face.
        """
        if self.restorer is None:
            return img_rgb, []
        img_rgb = self._limit_size(img_rgb)          # bound memory before GFPGAN
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        with torch.no_grad():                        # no autograd graph retention
            # NOTE: paste_back=False SEGFAULTS in this facexlib/gfpgan build, so
            # keep the stable paste_back=True path and simply discard the pasted
            # full frame — we only want the restored face crops for the list.
            cropped, restored, _ = self.restorer.enhance(
                bgr, has_aligned=False, only_center_face=self.only_center_face,
                paste_back=True)

        out = img_rgb.copy()                         # preview = original + boxes
        boxes = getattr(self.restorer.face_helper, "det_faces", None)
        if boxes is None or len(boxes) == 0:
            boxes = np.empty((0, 5))
        else:
            boxes = np.asarray(boxes, dtype=float)
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)

        dets = []
        for i in range(min(len(cropped), len(restored))):
            # the enhanced face region straight from GFPGAN (aligned 512x512)
            face_orig = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2RGB)
            face_enh = cv2.cvtColor(restored[i], cv2.COLOR_BGR2RGB)

            if i < len(boxes):
                b = boxes[i]
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                bw, bh = x2 - x1, y2 - y1
                box, quality = [x1, y1, bw, bh], float(bw * bh)
                if draw:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(out, "face", (x1, max(14, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                box = [0, 0, 0, 0]
                quality = float(face_enh.shape[0] * face_enh.shape[1])

            dets.append({
                "box": box, "face_orig": face_orig, "face_enh": face_enh,
                "embedding": self.embed(cropped[i]),
                "quality": quality,          # larger/closer face = better crop
            })
        return out, dets
