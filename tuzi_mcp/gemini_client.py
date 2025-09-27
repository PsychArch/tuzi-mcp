"""
Gemini image generation client for the Tuzi MCP Server.

Handles Gemini-2.5-flash-image generation with direct streaming.
"""

import base64
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import httpx

from fastmcp.exceptions import ToolError

from .image_utils import (
    load_and_encode_images,
    prepare_multimodal_content,
    download_image_from_url,
    save_image_to_file,
    adjust_path_for_image_bytes,
    derive_indexed_output_path,
)
from .task_manager import ImageTask, task_manager


@dataclass
class SavedImage:
    """Result of persisting one Gemini-generated image."""

    index: int
    path: Optional[str]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.path is not None


@dataclass
class ImageResolution:
    """Represents resolving raw Gemini output into base64 data."""

    base64_data: Optional[str]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class GeminiImageClient:
    """Handles Gemini image generation"""
    
    def __init__(self):
        self.api_key = os.getenv("TUZI_API_KEY")
        self.base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")
        
        if not self.api_key:
            raise ToolError("TUZI_API_KEY environment variable is required")
    
    def extract_images(self, response_content: str) -> List[str]:
        """Extract all base64 image data or URLs from Gemini streaming response"""
        images: List[str] = []

        # Primary pattern: base64 data (majority of responses)
        images.extend(
            re.findall(
                r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)',
                response_content,
            )
        )

        # Secondary pattern: HTTPS URL with image extension
        images.extend(
            re.findall(
                r'https://[^\s<>")\]]+\.(?:jpeg|jpg|png|gif|webp|bmp)',
                response_content,
                re.IGNORECASE,
            )
        )

        return images

    async def stream_api(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        model: str = "gemini-2.5-flash-image"
    ) -> str:
        """
        Call Gemini streaming API and return complete response content

        Args:
            prompt: Text prompt for image generation (can include requests for multiple images)
            reference_image_paths: Optional list of paths to reference images
            model: Gemini model to use (gemini-2.5-flash-image or gemini-2.5-flash-image-hd)

        Returns:
            Complete response content from streaming API
        """
        
        api_url = f"{self.base_url}/v1/chat/completions"
        
        # Handle reference images if provided
        image_data_urls = None
        if reference_image_paths:
            try:
                image_data_urls = await load_and_encode_images(reference_image_paths)
            except ToolError:
                raise
            except Exception as e:
                raise ToolError(f"Failed to process reference images: {str(e)}")
        
        # Prepare content for API request - use prompt as-is since it may already contain multiple image requests
        content = prepare_multimodal_content(prompt, image_data_urls)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": True
        }
        
        try:
            # Use 120s timeout as recommended in docs
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    # Read complete streaming response
                    response_content = ""
                    async for chunk in response.aiter_text():
                        response_content += chunk
                    
                    return response_content
                    
        except httpx.TimeoutException as e:
            raise ToolError(f"Gemini API request timeout: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise ToolError(f"Gemini API HTTP {e.response.status_code} error: {str(e)}")
        except Exception as e:
            raise ToolError(f"Unexpected Gemini API error: {str(e)}")
    
    @staticmethod
    def _format_warning(actual_path: str, previous_ext: Optional[str], change_type: str) -> str:
        if change_type == "changed" and previous_ext:
            return f"Saved as {actual_path} (changed from .{previous_ext})"
        return f"File saved as: {actual_path}"

    async def generate_task(self, task: ImageTask, prompt: str, model: str = "gemini-2.5-flash-image", reference_image_paths: Optional[List[str]] = None) -> None:
        """Execute a Gemini image generation task using direct streaming"""
        short_id = task.task_id[:8] + "..."
        
        try:
            task.status = "running"
            start_time = datetime.now()

            # Stream Gemini API call
            response_content = await self.stream_api(
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                model=model
            )

            # Extract all image data from response
            image_data_list = self.extract_images(response_content)

            if not image_data_list:
                response_clip = response_content[:1000] + "..." if len(response_content) > 1000 else response_content
                error_msg = f"No image data found in Gemini response ({len(response_content)} chars): {response_clip}"
                task.error = error_msg
                task.status = "failed"
                return

            # Persist images and collect outcomes
            results: List[SavedImage] = []
            total_images = len(image_data_list)
            for idx, raw_image in enumerate(image_data_list, start=1):
                results.append(
                    await self._persist_image(
                        raw_image,
                        task.output_path,
                        index=idx,
                        total=total_images,
                    )
                )

            successful = [result for result in results if result.success]
            failed = [result for result in results if not result.success]

            if not successful:
                error_messages = [result.error for result in failed if result.error]
                task.error = error_messages[0] if error_messages else "Failed to save Gemini images"
                task.status = "failed"
                return

            warnings: List[str] = []
            for result in results:
                warnings.extend(result.warnings)
                if result.error:
                    warnings.append(result.error)

            if total_images > 1:
                warnings.append(f"Generated {len(successful)} images")

            if warnings:
                joined_warning = "; ".join(warnings)
                task.result = {
                    "warning": joined_warning,
                    "warnings": warnings,
                    "images": [result.path for result in successful if result.path],
                    "failed_images": [result.index for result in failed if result.error],
                }
            else:
                task.result = {
                    "images": [result.path for result in successful if result.path],
                }

            task.status = "completed"

            # Record completion time
            elapsed = (datetime.now() - start_time).total_seconds()
            task_manager.record_completion_time(elapsed)
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            error_msg = f"Gemini task error: {str(e)}"
            task.error = error_msg
            task.status = "failed"

    async def _resolve_base64(self, raw_image: str, index: int) -> ImageResolution:
        """Convert raw Gemini output into base64 data, downloading URLs when required."""
        warnings: List[str] = []

        if raw_image.startswith('http'):
            try:
                downloaded = await download_image_from_url(raw_image)
            except ToolError as e:
                return ImageResolution(base64_data=None, warnings=warnings, error=f"Image {index}: {str(e)}")
            except Exception as e:
                return ImageResolution(base64_data=None, warnings=warnings, error=f"Image {index}: Failed to download image URL ({str(e)})")

            return ImageResolution(
                base64_data=base64.b64encode(downloaded).decode('utf-8'),
                warnings=warnings,
                error=None,
            )

        return ImageResolution(base64_data=raw_image, warnings=warnings, error=None)

    async def _persist_image(self, raw_image: str, base_path: str, index: int, total: int) -> SavedImage:
        """Resolve, adjust, and persist a single image from the Gemini response."""
        resolution = await self._resolve_base64(raw_image, index)
        warnings = list(resolution.warnings)

        if resolution.error:
            return SavedImage(index=index, path=None, warnings=warnings, error=resolution.error)

        b64_image = resolution.base64_data
        if not b64_image:
            return SavedImage(index=index, path=None, warnings=warnings, error=f"Image {index}: Empty image data")

        save_path = derive_indexed_output_path(base_path, index, total)

        try:
            img_bytes = base64.b64decode(b64_image)
        except Exception as e:
            return SavedImage(index=index, path=None, warnings=warnings, error=f"Image {index}: Invalid base64 data ({str(e)})")

        actual_path, format_warning = adjust_path_for_image_bytes(
            save_path,
            img_bytes,
            warning_factory=self._format_warning,
        )
        if format_warning:
            warnings.append(format_warning)

        try:
            saved_path, warning = await save_image_to_file(b64_image, actual_path)
        except ToolError as e:
            return SavedImage(index=index, path=None, warnings=warnings, error=f"Image {index}: {str(e)}")
        except Exception as e:
            return SavedImage(index=index, path=None, warnings=warnings, error=f"Image {index}: Failed to save ({str(e)})")

        if warning:
            warnings.append(warning)

        return SavedImage(index=index, path=saved_path, warnings=warnings)


# Global instance
gemini_client = GeminiImageClient()
