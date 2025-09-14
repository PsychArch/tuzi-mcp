"""
Seedream (Doubao) image generation client for the Tuzi MCP Server.

Uses the OpenAI-compatible images generation/edit endpoints for reliability and multi-image support.
"""

import base64
import os
import httpx
from datetime import datetime
from typing import List, Optional

from fastmcp.exceptions import ToolError

from .image_utils import (
    download_image_from_url,
    save_image_to_file,
    validate_image_file,
    get_image_mime_type,
)
from .task_manager import ImageTask, task_manager


class SeedreamImageClient:
    """Handles Seedream (doubao-seedream-4-0-250828) image generation"""

    def __init__(self):
        self.api_key = os.getenv("TUZI_API_KEY")
        self.base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")

        if not self.api_key:
            raise ToolError("TUZI_API_KEY environment variable is required")

        # Fixed model per API.md (do not use env override)
        self.model = "doubao-seedream-4-0-250828"

    async def images_generate(self, prompt: str, size: str = "1024x1024", quality: str = "high", n: int = 1) -> List[str]:
        """Call images generation and return list of image URLs (OpenAI-compatible)."""
        api_url = f"{self.base_url}/v1/images/generations"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
        }

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(api_url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                items = data.get("data", [])
                if not items:
                    raise ToolError("No images returned from Seedream images generation")

                urls: List[str] = []
                for item in items:
                    url = item.get("url")
                    if url:
                        urls.append(url)

                if not urls:
                    raise ToolError("Images generation returned no URLs")

                return urls

        except httpx.TimeoutException as e:
            raise ToolError(f"Seedream images request timeout: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise ToolError(f"Seedream images HTTP {e.response.status_code} error: {str(e)}")
        except Exception as e:
            raise ToolError(f"Unexpected Seedream images error: {str(e)}")

    async def images_edit(self, image_path: str, prompt: str, size: str = "1024x1024", n: int = 1) -> List[str]:
        """Call images edits and return list of image URLs."""
        api_url = f"{self.base_url}/v1/images/edits"

        # Validate and load file
        if not image_path:
            raise ToolError("edit_image_path is required for image edits")
        
        await validate_image_file(image_path)

        mime_type = get_image_mime_type(image_path)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Multipart form fields
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": str(max(1, int(n))),
            "size": size,
        }

        try:
            with open(image_path, "rb") as f:
                files = {
                    "image": (os.path.basename(image_path), f, mime_type)
                }
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(api_url, headers=headers, data=data, files=files)
                    response.raise_for_status()

                    data_json = response.json()
                    items = data_json.get("data", [])
                    if not items:
                        raise ToolError("No images returned from Seedream image edit")

                    urls: List[str] = []
                    for item in items:
                        url = item.get("url")
                        if url:
                            urls.append(url)

                    if not urls:
                        raise ToolError("Image edit returned no URLs")

                    return urls

        except httpx.TimeoutException as e:
            raise ToolError(f"Seedream image edit timeout: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise ToolError(f"Seedream image edit HTTP {e.response.status_code} error: {str(e)}")
        except Exception as e:
            raise ToolError(f"Unexpected Seedream image edit error: {str(e)}")

    async def generate_task(
        self,
        task: ImageTask,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "high",
        n: int = 1,
        edit_image_path: Optional[str] = None,
    ) -> None:
        """Execute a Seedream image generation or edit task via images endpoints (OpenAI-compatible)."""
        try:
            task.status = "running"
            start_time = datetime.now()

            # Choose endpoint
            if edit_image_path:
                outputs = await self.images_edit(image_path=edit_image_path, prompt=prompt, size=size, n=max(1, int(n)))
            else:
                outputs = await self.images_generate(prompt=prompt, size=size, quality=quality, n=max(1, int(n)))

            # Download and save images (support n>1 by suffixing)
            warnings: List[str] = []
            images_saved = 0
            
            if len(outputs) == 1:
                img_bytes = await download_image_from_url(outputs[0])
                b64_image = base64.b64encode(img_bytes).decode("utf-8")
                
                # Detect actual image format and adjust extension if needed
                actual_path, format_warning = self._adjust_path_for_format(task.output_path, img_bytes)
                if format_warning:
                    warnings.append(format_warning)
                
                _, warning = await save_image_to_file(b64_image, actual_path)
                if warning:
                    warnings.append(warning)
                images_saved = 1
            else:
                # Multiple outputs: insert index suffix before extension
                base_path = task.output_path
                dot = base_path.rfind('.')
                if dot == -1:
                    stem, ext = base_path, ".jpg"  # default to jpg
                else:
                    stem, ext = base_path[:dot], base_path[dot:]

                for idx, url in enumerate(outputs, start=1):
                    img_bytes = await download_image_from_url(url)
                    b64_image = base64.b64encode(img_bytes).decode("utf-8")
                    save_path = f"{stem}_{idx}{ext}"
                    
                    # Detect actual image format and adjust extension if needed
                    actual_path, format_warning = self._adjust_path_for_format(save_path, img_bytes)
                    if format_warning:
                        warnings.append(format_warning)
                    
                    _, warning = await save_image_to_file(b64_image, actual_path)
                    if warning:
                        warnings.append(warning)
                    images_saved += 1
            
            # Add image count info to warnings when user requested multiple images
            if n > 1:
                warnings.append(f"Generated {images_saved} images (requested {n})")

            # Store one combined warning if needed
            if warnings:
                task.result = {"warning": " | ".join(warnings)}
            else:
                task.result = None

            task.status = "completed"

            # Record completion
            elapsed = (datetime.now() - start_time).total_seconds()
            task_manager.record_completion_time(elapsed)

        except Exception as e:
            task.error = f"Seedream task error: {str(e)}"
            task.status = "failed"
    
    def _adjust_path_for_format(self, requested_path: str, image_bytes: bytes) -> tuple[str, Optional[str]]:
        """Adjust file path extension based on actual image format and return warning if changed.
        
        Returns:
            Tuple of (actual_path, warning_message)
        """
        # Detect image format by checking magic bytes
        is_jpeg = image_bytes.startswith(b'\xff\xd8\xff')
        is_png = image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
        is_webp = image_bytes[8:12] == b'WEBP' if len(image_bytes) > 12 else False
        
        # Determine actual format
        if is_jpeg:
            actual_format = 'jpg'
        elif is_png:
            actual_format = 'png'
        elif is_webp:
            actual_format = 'webp'
        else:
            # Unknown format, keep original extension
            return requested_path, None
        
        # Check if extension needs to be changed
        dot_pos = requested_path.rfind('.')
        if dot_pos == -1:
            # No extension, add the correct one
            actual_path = f"{requested_path}.{actual_format}"
            return actual_path, f"File saved as: {actual_path}"
        
        requested_ext = requested_path[dot_pos+1:].lower()
        if requested_ext != actual_format:
            # Extension mismatch, change it
            actual_path = requested_path[:dot_pos] + f".{actual_format}"
            warning = f"File saved as: {actual_path}"
            return actual_path, warning
        
        # Extension matches, no change needed
        return requested_path, None


# Global instance
seedream_client = SeedreamImageClient()
