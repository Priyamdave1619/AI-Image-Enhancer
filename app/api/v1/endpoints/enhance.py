from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.image import EnhanceResponse, ErrorResponse, UpscalePreset
from app.services.upscaler import ImageUpscaler
from app.utils.file_utils import allowed_image_extension, safe_filename

logger = logging.getLogger(__name__)
router = APIRouter()
upscaler = ImageUpscaler()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
    }


@router.post(
    "/enhance",
    response_model=EnhanceResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def enhance_image(
    file: UploadFile = File(...),
    preset: UpscalePreset = Form(UpscalePreset.k4),
    preserve_aspect: bool = Form(True),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A valid file name is required.")

    if not allowed_image_extension(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG, JPEG, PNG, WEBP, BMP, or TIFF.")

    upload_name = safe_filename(file.filename)
    upload_path = settings.upload_dir / upload_name
    output_name = f"enhanced_{Path(upload_name).stem}.png"
    output_path = settings.output_dir / output_name

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is {settings.max_upload_mb} MB.")

    upload_path.write_bytes(content)

    try:
        result = upscaler.enhance(
            input_path=upload_path,
            output_path=output_path,
            preset=preset,
            preserve_aspect=preserve_aspect,
        )
    except Exception as exc:
        logger.exception("Enhancement failed")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}") from exc

    return EnhanceResponse(
        success=True,
        message="Image enhanced successfully",
        input_filename=upload_path.name,
        output_filename=output_path.name,
        preset=result.preset,
        target_long_edge=result.target_long_edge,
        input_width=result.input_width,
        input_height=result.input_height,
        output_width=result.output_width,
        output_height=result.output_height,
        download_url=f"/api/v1/download/{output_path.name}",
    )


@router.get("/download/{filename}")
def download_file(filename: str):
    file_path = settings.output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
