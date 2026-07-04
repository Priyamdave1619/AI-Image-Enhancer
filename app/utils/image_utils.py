from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image


def save_image(image: Image.Image, output_path: Path, quality: int = 95) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=quality, optimize=True)
    elif ext == ".png":
        image.save(output_path, format="PNG", optimize=True)
    elif ext == ".webp":
        image.save(output_path, format="WEBP", quality=quality, method=6)
    else:
        image.save(output_path)


def resize_long_edge(image: Image.Image, target_long_edge: int) -> Image.Image:
    width, height = image.size
    current_long_edge = max(width, height)

    if current_long_edge <= 0:
        return image

    if current_long_edge == target_long_edge:
        return image

    scale = target_long_edge / float(current_long_edge)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_size(image: Image.Image) -> Tuple[int, int]:
    return image.size
