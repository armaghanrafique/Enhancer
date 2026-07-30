"""SRVGGNetCompact — the architecture behind the newer `realesr-general-x4v3`
and `realesr-animevideov3` Real-ESRGAN models.

Lighter and faster than RRDBNet (a plain VGG-style stack + pixel-shuffle), which
makes it a strong practical choice on CPU. Parameter names match the official
checkpoints, so the public weights load directly.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale

        def activation():
            if act_type == "relu":
                return nn.ReLU(inplace=True)
            if act_type == "prelu":
                return nn.PReLU(num_parameters=num_feat)
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(activation())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # learned residual over a nearest-neighbour upscale of the input
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base
