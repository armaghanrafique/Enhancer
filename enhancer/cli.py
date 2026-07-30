"""Command-line interface for the Enhancer pipeline.

Examples
--------
  # Upscale a single image 4x with Real-ESRGAN (auto-downloads weights)
  python -m enhancer -i input.jpg -o out.png --stages superres

  # Full pipeline on a traffic video, CPU-friendly defaults
  python -m enhancer -i traffic.mp4 -o traffic_4k.mp4 \
      --stages lowlight denoise superres --outscale 2 --smooth 0.2

  # Batch a whole folder of images
  python -m enhancer -i ./frames -o ./frames_out --stages denoise superres

  # Force classical (no deep models, fastest) — good for a quick preview
  python -m enhancer -i in.jpg -o out.png --stages superres --no-deep
"""
import argparse
from pathlib import Path

from .pipeline import Enhancer, STAGE_ORDER
from .config import SR_MODELS, DEFAULT_SR
from .utils.imageio_utils import IMG_EXTS, VIDEO_EXTS


def build_parser():
    p = argparse.ArgumentParser(
        prog="enhancer",
        description="Generalized image & video quality enhancer (CPU-friendly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("-i", "--input", required=True,
                   help="input image, video, or folder of images")
    p.add_argument("-o", "--output", required=True,
                   help="output file or folder")
    p.add_argument("--stages", nargs="+", default=["superres"],
                   choices=STAGE_ORDER + ["all"],
                   help="enhancement stages to apply (default: superres)")
    p.add_argument("--target", default=None,
                   choices=["general", "face", "plate", "all"],
                   help="preset that picks stages by input type (overrides --stages): "
                        "general=denoise+superres, face=superres+face, "
                        "plate=plate(+ocr), all=everything")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="compute device (default: auto)")
    p.add_argument("--no-deep", action="store_true",
                   help="disable deep models; use fast classical fallbacks only")

    # super-resolution
    p.add_argument("--sr-model", default=DEFAULT_SR, choices=list(SR_MODELS),
                   help="super-resolution model")
    p.add_argument("--sr-weights", default=None,
                   help="path to a custom SR checkpoint; use with "
                        "--sr-model spandrel to run any HAT/DAT/DRCT/SwinIR .pth")
    p.add_argument("--outscale", type=float, default=None,
                   help="final upscale factor (default: model native scale)")
    p.add_argument("--tile", type=int, default=192,
                   help="tile size for SR inference; lower = less memory (default: 192)")

    # denoise / low-light
    p.add_argument("--denoise-strength", type=float, default=1.0,
                   help="classical denoise strength (default: 1.0)")
    p.add_argument("--sharpen", type=float, default=0.6,
                   help="unsharp amount after denoise (default: 0.6)")
    p.add_argument("--median", default="auto",
                   help="median filter size for salt-and-pepper noise: "
                        "'auto' (detect), 0 (off), or an odd int like 3/5")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="low-light gamma; 0 = adaptive (default: 0)")

    # diffusion (only used by --sr-model diffusion-x4)
    p.add_argument("--steps", type=int, default=30,
                   help="diffusion denoising steps; fewer = faster (default: 30)")
    p.add_argument("--guidance", type=float, default=0.0,
                   help="diffusion guidance scale; 0 = fastest, no text guidance")
    p.add_argument("--prompt", default="",
                   help="optional text prompt for the diffusion upscaler")
    p.add_argument("--noise-level", type=int, default=20,
                   help="diffusion conditioning noise level (default: 20)")

    # object detection / OCR (plate + face targeting)
    p.add_argument("--ocr", action="store_true",
                   help="read number-plate text with OCR (needs: pip install easyocr)")
    p.add_argument("--ocr-langs", nargs="+", default=["en"],
                   help="OCR languages (default: en)")
    p.add_argument("--ocr-engine", default="easyocr",
                   choices=["easyocr", "tesseract"],
                   help="OCR engine (default: easyocr)")
    p.add_argument("--plate-mode", default="crop", choices=["crop", "scan"],
                   help="crop: detect plate then super-resolve it (close-up plates); "
                        "scan: super-resolve whole frame then OCR-scan it "
                        "(best for small/far traffic plates)")
    p.add_argument("--scan-outscale", type=float, default=2.0,
                   help="whole-frame upscale for --plate-mode scan (default: 2)")
    p.add_argument("--min-plate-len", type=int, default=4,
                   help="minimum characters to count text as a plate (default: 4)")
    p.add_argument("--detector", default="haar", choices=["haar", "yolo"],
                   help="plate detector backend for crop mode (default: haar)")
    p.add_argument("--detector-weights", default=None,
                   help="YOLO weights (.pt) when --detector yolo")
    p.add_argument("--plate-sr-model", default="realesr-general-x4v3",
                   choices=list(SR_MODELS),
                   help="SR model used to sharpen detected plate crops")
    p.add_argument("--crop-outscale", type=float, default=4.0,
                   help="upscale factor applied to each detected crop (default: 4)")
    p.add_argument("--draw", action="store_true",
                   help="draw detection boxes + OCR text on the output")
    p.add_argument("--save-crops", default=None,
                   help="folder to save each enhanced plate crop")

    # video
    p.add_argument("--smooth", type=float, default=0.0,
                   help="temporal smoothing 0..1 to reduce flicker (default: 0)")
    p.add_argument("--no-audio", action="store_true",
                   help="do not re-mux the original audio into video output")
    return p


TARGET_PRESETS = {
    "general": ["denoise", "superres"],
    "face": ["superres", "face"],
    "plate": ["plate"],
    "all": list(STAGE_ORDER),
}


def resolve_stages(stages, target=None):
    if target:
        return list(TARGET_PRESETS[target])
    if "all" in stages:
        return list(STAGE_ORDER)
    return stages


def build_stage_opts(args):
    return {
        "lowlight": {"gamma": args.gamma},
        "denoise": {"strength": args.denoise_strength, "sharpen": args.sharpen,
                    "median": args.median, "tile": args.tile},
        "superres": {"model": args.sr_model, "tile": args.tile,
                     "steps": args.steps, "guidance": args.guidance,
                     "prompt": args.prompt, "noise_level": args.noise_level,
                     **({"weights": args.sr_weights} if args.sr_weights else {}),
                     **({"outscale": args.outscale} if args.outscale else {})},
        "face": {},
        "plate": {"mode": args.plate_mode,
                  "detector": args.detector,
                  "detector_weights": args.detector_weights,
                  "ocr": args.ocr, "ocr_engine": args.ocr_engine,
                  "ocr_langs": tuple(args.ocr_langs),
                  "sr_model": args.plate_sr_model,
                  "crop_outscale": args.crop_outscale,
                  "scan_outscale": args.scan_outscale,
                  "min_len": args.min_plate_len,
                  "draw": args.draw, "save_crops": args.save_crops},
    }


def iter_jobs(inp, out):
    """Yield (src, dst) pairs, expanding a folder input into per-file jobs."""
    inp, out = Path(inp), Path(out)
    if inp.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        media = sorted(f for f in inp.iterdir()
                       if f.suffix.lower() in IMG_EXTS | VIDEO_EXTS)
        if not media:
            raise SystemExit(f"No images/videos found in {inp}")
        for f in media:
            suffix = ".png" if f.suffix.lower() in IMG_EXTS else f.suffix
            yield f, out / (f.stem + "_enhanced" + suffix)
    else:
        yield inp, out


def main(argv=None):
    args = build_parser().parse_args(argv)
    stages = resolve_stages(args.stages, args.target)

    enhancer = Enhancer(
        stages=stages,
        device=args.device,
        deep=not args.no_deep,
        stage_opts=build_stage_opts(args),
    )

    video_kw = dict(smooth=args.smooth, keep_audio=not args.no_audio)
    for src, dst in iter_jobs(args.input, args.output):
        print(f"\n=== {src.name} -> {dst.name} ===")
        enhancer.run(src, dst, **video_kw)
    print("\nDone.")


if __name__ == "__main__":
    main()
