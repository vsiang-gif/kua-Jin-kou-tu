from __future__ import annotations

import os
import sys
from pathlib import Path


def app_dir() -> Path:
    """Folder that contains PrintCut.exe (frozen) or the repo root (source run)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _search_roots() -> list[Path]:
    roots = [app_dir(), app_dir() / "vendor"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    return roots


def models_dir() -> Path:
    for root in _search_roots():
        candidate = root / "models"
        if candidate.is_dir():
            return candidate
    target = app_dir() / "models"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PrintCut" / "models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def realesrgan_exe() -> Path | None:
    name = "realesrgan-ncnn-vulkan.exe" if sys.platform == "win32" else "realesrgan-ncnn-vulkan"
    for root in _search_roots():
        candidate = root / "bin" / "realesrgan" / name
        if candidate.is_file():
            return candidate
    return None
