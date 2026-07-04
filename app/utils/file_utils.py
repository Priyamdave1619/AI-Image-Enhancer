from __future__ import annotations

import uuid
from pathlib import Path


def safe_filename(original_name: str) -> str:
    suffix = Path(original_name).suffix.lower()
    stem = Path(original_name).stem.replace(" ", "_")
    token = uuid.uuid4().hex[:12]
    return f"{stem}_{token}{suffix}"


def allowed_image_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
