# app/schemas/image.py
from enum import Enum
from pydantic import BaseModel, Field

class UpscalePreset(str, Enum):
    hd = "hd"
    fhd = "fhd"
    k2 = "2k"
    k3 = "3k"
    k4 = "4k"
    k8 = "8k"
    k16 = "16k"  # Added 16K support

class EnhanceResponse(BaseModel):
    success: bool = True
    message: str = "Image enhanced successfully"
    input_filename: str
    output_filename: str
    preset: UpscalePreset
    target_long_edge: int = Field(..., description="Requested output long-edge in pixels")
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    download_url: str

class ErrorResponse(BaseModel):
    success: bool = False
    message: str