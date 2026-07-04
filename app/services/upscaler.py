# app/services/upscaler.py
from __future__ import annotations

import logging
import ssl  # <-- Added for SSL Fix
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

from app.core.config import settings
from app.schemas.image import UpscalePreset
from app.utils.image_utils import open_image, resize_long_edge, save_image

# --- MAC OS SSL FIX ---
# This prevents the "[SSL: CERTIFICATE_VERIFY_FAILED]" error when facexlib/gfpgan
# tries to download the face detection model weights in the background.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ----------------------

logger = logging.getLogger(__name__)

PRESET_TARGETS = {
    UpscalePreset.hd: 1280,
    UpscalePreset.fhd: 1920,
    UpscalePreset.k2: 2048,
    UpscalePreset.k3: 3072,
    UpscalePreset.k4: 3840,
    UpscalePreset.k8: 7680,
    UpscalePreset.k16: 15360,
}


@dataclass
class EnhancementResult:
    input_path: Path
    output_path: Path
    preset: UpscalePreset
    target_long_edge: int
    input_width: int
    input_height: int
    output_width: int
    output_height: int


class RealESRGANBackend:
    """
    Features: 
    1. 4x-UltraSharp (Industry standard for REALISTIC whole-image textures)
    2. GFPGANv1.4 (Photorealistic Face Restoration)
    3. Multi-Pass up to 16K sizes.
    """

    BASE_MODEL_URL = "https://huggingface.co/lokcx/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
    BASE_MODEL_NAME = "4x-UltraSharp.pth"

    GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    GFPGAN_NAME = "GFPGANv1.4.pth"

    def __init__(self) -> None:
        self._upsampler = None
        self._face_enhancer = None
        self._available = False
        self._load()

    def _download_file(self, url: str, model_path: Path, desc: str) -> bool:
        """Helper to download model weights."""
        logger.info(f"Downloading {desc} model...")
        try:
            # Tell requests to ignore SSL verification if system certs are missing
            response = requests.get(url, stream=True, timeout=120, verify=False)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f, tqdm(
                desc=f"Downloading {desc}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"✅ {desc} downloaded successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ {desc} download failed: {e}")
            return False

    def _get_model_path(self, url: str, filename: str, desc: str) -> Optional[Path]:
        model_path = settings.weights_dir / filename
        
        # Download if missing
        if not model_path.exists() or model_path.stat().st_size < 50_000_000:
            success = self._download_file(url, model_path, desc)
            if not success or not model_path.exists():
                return None

        # --- BULLETPROOF AUTO-TRANSLATOR FOR COMMUNITY MODELS ---
        try:
            state = torch.load(model_path, map_location='cpu')
            
            if isinstance(state, dict):
                state_dict = state.get('params_ema', state.get('params', state))
            else:
                state_dict = state

            needs_save = False

            if 'model.0.weight' in state_dict or 'model.1.sub.0.RDB1.conv1.0.weight' in state_dict:
                logger.info(f"Old architecture detected in {filename}. Translating network keys safely...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k == 'model.0.weight': new_state_dict['conv_first.weight'] = v
                    elif k == 'model.0.bias': new_state_dict['conv_first.bias'] = v
                    elif k == 'model.1.weight': new_state_dict['conv_body.weight'] = v
                    elif k == 'model.1.bias': new_state_dict['conv_body.bias'] = v
                    elif k == 'model.3.weight': new_state_dict['conv_up1.weight'] = v
                    elif k == 'model.3.bias': new_state_dict['conv_up1.bias'] = v
                    elif k == 'model.6.weight': new_state_dict['conv_up2.weight'] = v
                    elif k == 'model.6.bias': new_state_dict['conv_up2.bias'] = v
                    elif k == 'model.8.weight': new_state_dict['conv_hr.weight'] = v
                    elif k == 'model.8.bias': new_state_dict['conv_hr.bias'] = v
                    elif k == 'model.10.weight': new_state_dict['conv_last.weight'] = v
                    elif k == 'model.10.bias': new_state_dict['conv_last.bias'] = v
                    elif k.startswith('model.1.sub.'):
                        parts = k.split('.')
                        if len(parts) == 5 and parts[4] in['weight', 'bias']:
                            new_state_dict[f'conv_body.{parts[4]}'] = v
                        elif len(parts) >= 6:
                            body_idx = parts[3]
                            rdb_idx = parts[4].lower()
                            conv_idx = parts[5]
                            weight_type = parts[-1]
                            new_key = f'body.{body_idx}.{rdb_idx}.{conv_idx}.{weight_type}'
                            new_state_dict[new_key] = v
                        else:
                            new_state_dict[k] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
                needs_save = True

            if isinstance(state, dict) and 'params_ema' not in state and 'params' not in state:
                needs_save = True

            if needs_save:
                logger.info(f"Saving permanently corrected version of {filename}...")
                torch.save({'params': state_dict}, model_path)
                logger.info("✅ Translation & Repackaging complete!")

        except Exception as e:
            logger.warning(f"Failed to check or repackage model weights: {e}")
        # ------------------------------------------------------

        return model_path

    def _load(self) -> None:
        try:
            # --- MONKEY PATCH FOR NEWER TORCHVISION VERSIONS ---
            import sys
            import torchvision.transforms.functional as TF
            sys.modules["torchvision.transforms.functional_tensor"] = TF
            # ---------------------------------------------------

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {device.upper()}")

            # 1. Load 4x-UltraSharp
            ultrasharp_path = self._get_model_path(self.BASE_MODEL_URL, self.BASE_MODEL_NAME, "4x-UltraSharp (Realism Engine)")
            if not ultrasharp_path:
                logger.warning("Base model missing. Using Pillow fallback.")
                return

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._upsampler = RealESRGANer(
                scale=4, model_path=str(ultrasharp_path), model=model,
                tile=400, tile_pad=10, pre_pad=0, half=False, device=device
            )

            # 2. Load GFPGAN
            try:
                from gfpgan import GFPGANer
                gfpgan_path = self._get_model_path(self.GFPGAN_URL, self.GFPGAN_NAME, "GFPGAN (Face Engine)")
                
                if gfpgan_path:
                    self._face_enhancer = GFPGANer(
                        model_path=str(gfpgan_path),
                        upscale=4,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self._upsampler, 
                        device=device
                    )
                    logger.info("✅ GFPGAN Photorealistic Face Engine Loaded!")
            except ImportError:
                logger.warning("GFPGAN not installed. Photos will be upscaled but faces won't be reconstructed.")
                self._face_enhancer = None

            self._available = True
            logger.info(f"✅ AI Engine Ready for Whole-Image Photorealism up to 16K.")

        except Exception as exc:
            logger.warning(f"AI Engine failed to initialize: {exc}. Using fallback.")

    def enhance(self, image: Image.Image, target_scale: float = 4.0) -> Image.Image:
        if not self._available or self._upsampler is None:
            return image

        try:
            rgb = image.convert("RGB")
            np_img = np.array(rgb)
            current_img = np_img[:, :, ::-1]  # RGB to BGR

            current_scale = 1.0
            pass_count = 0
            max_passes = 2  

            while current_scale < target_scale and pass_count < max_passes:
                logger.info(f"Running AI Enhance Pass {pass_count + 1}...")

                if pass_count == 0 and self._face_enhancer:
                    _, _, current_img = self._face_enhancer.enhance(
                        current_img, 
                        has_aligned=False, 
                        only_center_face=False, 
                        paste_back=True
                    )
                else:
                    current_img, _ = self._upsampler.enhance(current_img, outscale=4)
                
                current_scale *= 4.0
                pass_count += 1

            if isinstance(current_img, np.ndarray):
                output_rgb = current_img[:, :, ::-1]
                output_rgb = np.clip(output_rgb, 0, 255).astype(np.uint8)
                return Image.fromarray(output_rgb)

            return image
        except Exception as exc:
            logger.warning(f"Enhancement error: {exc}. Returning original.")
            return image


class ImageUpscaler:
    def __init__(self) -> None:
        self.backend = RealESRGANBackend()

    def enhance(
        self,
        input_path: Path,
        output_path: Path,
        preset: UpscalePreset,
        preserve_aspect: bool = True,
    ) -> EnhancementResult:
        image = open_image(input_path)
        input_width, input_height = image.size
        input_long_edge = max(input_width, input_height)
        
        target_long_edge = PRESET_TARGETS[preset]
        target_scale = target_long_edge / float(max(1, input_long_edge))

        # 1. AI Enhancement
        enhanced = self.backend.enhance(image, target_scale=target_scale)

        # 2. Perfect Pixel Resizing
        if preserve_aspect:
            enhanced = resize_long_edge(enhanced, target_long_edge)
        else:
            enhanced = enhanced.resize((target_long_edge, target_long_edge), Image.Resampling.LANCZOS)

        # 3. Add Real Camera Crispness
        sharpener = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpener.enhance(1.3)

        save_image(enhanced, output_path)

        output_width, output_height = enhanced.size

        return EnhancementResult(
            input_path=input_path,
            output_path=output_path,
            preset=preset,
            target_long_edge=target_long_edge,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
        )

# Tell requests/urllib3 to stop showing the InsecureRequestWarning when verifying=False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)