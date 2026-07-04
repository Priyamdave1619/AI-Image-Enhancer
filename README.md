---
title: Image Enhancer
emoji: 📉
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

# AI Image Enhancer / Upscaler Backend

A production-style FastAPI backend for image enhancement and upscaling with support for:

- HD
- Full HD
- 2K
- 3K
- 4K
- 8K

The project is designed to use **Real-ESRGAN** when available, and falls back to a high-quality Pillow resize pipeline when the model or dependencies are not installed.

## Features

- Upload low-quality images
- Enhance/upscale images via API
- Preset-based output sizes
- Download the processed file
- Health check endpoint
- Clean layered project structure
- Optional Real-ESRGAN integration

## Project Structure

```text
app/
  api/v1/endpoints/   # API routes
  core/               # settings and logging
  schemas/            # Pydantic request/response models
  services/           # image enhancement logic
  utils/              # file/image helpers
storage/
  uploads/            # original uploads
  outputs/            # enhanced files
weights/              # model weights folder
```

## Presets

| Preset | Long edge |
|---|---:|
| hd | 1280 |
| fhd | 1920 |
| 2k | 2048 |
| 3k | 3072 |
| 4k | 3840 |
| 8k | 7680 |

The backend first enhances the image, then resizes it to the requested long-edge target.

## Quick Start

### 1) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate    # Windows
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the API
```bash
uvicorn app.main:app --reload
```

### 4) Open docs
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API

### Health
`GET /health`

### Enhance image
`POST /api/v1/enhance`

Form-data:
- `file`: image file
- `preset`: one of `hd`, `fhd`, `2k`, `3k`, `4k`, `8k`
- `preserve_aspect`: `true` / `false`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/enhance"   -F "file=@input.jpg"   -F "preset=4k"   -F "preserve_aspect=true"
```

### Download output
`GET /api/v1/download/{filename}`

## Real-ESRGAN Weights

To use Real-ESRGAN properly, place model weights in:

```text
weights/
```

The code is written to work even without weights. In that case, it will use the fallback enhancer.

## Notes

- True 8K detail cannot be guaranteed from a very low-quality source image.
- For best results, use the highest-quality source image available.
- If you want GPU acceleration later, this structure is ready for it.