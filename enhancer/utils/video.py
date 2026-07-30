"""Video read/write built on imageio + the bundled ffmpeg.

No system ffmpeg required: imageio-ffmpeg ships a static binary. Frames are
yielded as RGB uint8 arrays. After writing the enhanced (silent) video, the
original audio track is re-muxed back in with a single ffmpeg call.
"""
import subprocess
from pathlib import Path

import imageio.v2 as imageio


def probe_video(path):
    """Return (fps, n_frames or None) for a video file."""
    reader = imageio.get_reader(str(path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30.0)
    n = meta.get("nframes")
    if n in (None, float("inf")):
        n = None
    reader.close()
    return fps, (int(n) if n else None)


def iter_frames(path):
    """Yield RGB uint8 frames from a video."""
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        for frame in reader:
            # ensure 3-channel RGB
            if frame.ndim == 2:
                frame = frame[:, :, None].repeat(3, axis=2)
            yield frame[:, :, :3]
    finally:
        reader.close()


class VideoWriter:
    def __init__(self, path, fps, quality=8):
        self.path = str(path)
        self.writer = imageio.get_writer(
            self.path, "ffmpeg", fps=fps, quality=quality,
            macro_block_size=None, codec="libx264", pixelformat="yuv420p")

    def append(self, frame_rgb):
        self.writer.append_data(frame_rgb)

    def close(self):
        self.writer.close()


def remux_audio(enhanced_path, source_path):
    """Copy the audio track from `source_path` onto `enhanced_path` in place.

    Silently no-ops if the source has no audio or ffmpeg errors.
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return

    enhanced = Path(enhanced_path)
    tmp = enhanced.with_name(enhanced.stem + "_muxed" + enhanced.suffix)
    cmd = [
        ffmpeg, "-y", "-i", str(enhanced), "-i", str(source_path),
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
        "-c:a", "aac", "-shortest", str(tmp),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        if res.returncode == 0 and tmp.exists():
            tmp.replace(enhanced)
        elif tmp.exists():
            tmp.unlink()
    except Exception:
        if tmp.exists():
            tmp.unlink()
