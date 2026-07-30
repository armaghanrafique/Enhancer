"""Tile-based inference.

Splitting a frame into overlapping tiles keeps peak memory bounded — essential
on CPU and for large (1080p/4K) frames. Tiles are blended back with a feathered
overlap so seams don't show.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def tiled_forward(model, x, scale=1, tile=256, overlap=16):
    """Run `model` over `x` (1x3xHxW) tile-by-tile, returning the full output.

    `scale` is the model's spatial magnification (1 for restoration, 4 for SR).
    If `tile` is 0 or the image is small, runs in a single pass.
    """
    b, c, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return model(x)

    stride = tile - overlap
    out_h, out_w = h * scale, w * scale
    output = x.new_zeros(b, c, out_h, out_w)
    weight = x.new_zeros(b, 1, out_h, out_w)

    # feather mask for smooth blending across overlaps
    ramp = torch.linspace(0, 1, overlap * scale, device=x.device) if overlap > 0 else None

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))
    for y in ys:
        for xx in xs:
            y2 = min(y + tile, h)
            x2 = min(xx + tile, w)
            y1 = max(y2 - tile, 0)
            x1 = max(x2 - tile, 0)

            patch = x[:, :, y1:y2, x1:x2]
            out_patch = model(patch)

            oy1, oy2 = y1 * scale, y2 * scale
            ox1, ox2 = x1 * scale, x2 * scale

            mask = x.new_ones(1, 1, out_patch.shape[2], out_patch.shape[3])
            if ramp is not None:
                ph, pw = mask.shape[2], mask.shape[3]
                k = ramp.numel()
                if y1 > 0:
                    mask[:, :, :k, :] *= ramp.view(1, 1, -1, 1)
                if x1 > 0:
                    mask[:, :, :, :k] *= ramp.view(1, 1, 1, -1)
                if y2 < h:
                    mask[:, :, ph - k:, :] *= ramp.flip(0).view(1, 1, -1, 1)
                if x2 < w:
                    mask[:, :, :, pw - k:] *= ramp.flip(0).view(1, 1, 1, -1)

            output[:, :, oy1:oy2, ox1:ox2] += out_patch * mask
            weight[:, :, oy1:oy2, ox1:ox2] += mask

    return output / weight.clamp_min(1e-8)
