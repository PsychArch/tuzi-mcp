"""
Gemini image generation client for the Tuzi MCP Server.

Handles Gemini-2.5-flash-image generation with direct streaming.
"""

import base64
import os
import re
import httpx
from datetime import datetime
from typing import List, Optional

from fastmcp.exceptions import ToolError

from .image_utils import load_and_encode_images, prepare_multimodal_content, download_image_from_url, save_image_to_file
from .task_manager import ImageTask, task_manager


class GeminiImageClient:
    """Handles Gemini image generation"""
    
    def __init__(self):
        self.api_key = os.getenv("TUZI_API_KEY")
        self.base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")
        
        if not self.api_key:
            raise ToolError("TUZI_API_KEY environment variable is required")
    
    async def extract_image(self, response_content: str) -> Optional[str]:
        """Extract base64 image data or URL from Gemini streaming response"""
        
        # Primary pattern: base64 data (90% of responses)
        base64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', response_content)
        if base64_match:
            return base64_match.group(1)
        
        # Secondary pattern: any HTTPS URL with image file extension
        url_match = re.search(r'https://[^\s<>")\]]+\.(jpeg|jpg|png|gif|webp|bmp)', response_content, re.IGNORECASE)
        if url_match:
            return url_match.group(0)
        
        return None
    
    async def stream_api(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        model: str = "gemini-2.5-flash-image"
    ) -> str:
        """
        Call Gemini streaming API and return complete response content
        
        Args:
            prompt: Text prompt for image generation
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
        
        # Prepare content for API request
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
            
            # Extract image data from response
            image_data = await self.extract_image(response_content)
            
            if not image_data:
                response_clip = response_content[:1000] + "..." if len(response_content) > 1000 else response_content
                error_msg = f"No image data found in Gemini response ({len(response_content)} chars): {response_clip}"
                task.error = error_msg
                task.status = "failed"
                return
            
            # Handle both base64 data and URLs
            if image_data.startswith('http'):
                # URL case: download the image
                try:
                    downloaded_data = await download_image_from_url(image_data)
                    b64_image = base64.b64encode(downloaded_data).decode('utf-8')
                    final_url = image_data
                except Exception as e:
                    error_msg = f"Failed to download image from URL: {str(e)}"
                    task.error = error_msg
                    task.status = "failed"
                    return
            else:
                # Base64 case: use directly
                b64_image = image_data
                final_url = None
            
            # Save image to file
            actual_path, warning = await save_image_to_file(b64_image, task.output_path)
            
            # Only store warning if present (no need for full result object)
            if warning:
                task.result = {"warning": warning}
            else:
                task.result = None
            task.status = "completed"
            
            # Record completion time
            elapsed = (datetime.now() - start_time).total_seconds()
            task_manager.record_completion_time(elapsed)
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            error_msg = f"Gemini task error: {str(e)}"
            task.error = error_msg
            task.status = "failed"


# Global instance
gemini_client = GeminiImageClient()