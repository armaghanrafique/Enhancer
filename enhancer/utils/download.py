"""Model-weight resolution and download.

Weights live in the project `models/` directory. A weight is resolved by:
  1. an explicit path passed on the CLI, else
  2. a file already present in models/, else
  3. downloading from the registered URL (if any).
Stages that can't get weights fall back to their classical path.
"""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


_last_pct = [-1]


def _reporthook(count, block_size, total_size):
    done = count * block_size
    if total_size > 0:
        pct = min(100, done * 100 // total_size)
        if pct == _last_pct[0]:  # only redraw when the percentage changes
            return
        _last_pct[0] = pct
        mb = done / 1024 / 1024
        print(f"\r    downloading {pct:3d}%  ({mb:6.1f} MB)", end="", flush=True)


def resolve_weights(filename, url=None, explicit=None):
    """Return a local path to the weight file, downloading if needed.

    Returns None if the weight can't be found and there's no URL to fetch it.
    """
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        print(f"  ! weight not found at {explicit}")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local = MODELS_DIR / filename
    if local.is_file():
        return local

    if not url:
        return None

    print(f"  > fetching {filename}")
    try:
        tmp = local.with_suffix(local.suffix + ".part")
        urllib.request.urlretrieve(url, tmp, _reporthook)
        os.replace(tmp, local)
        print()  # newline after progress
        return local
    except Exception as e:  # noqa: BLE001 — network failures shouldn't crash the run
        print(f"\n  ! download failed: {e}")
        return None
