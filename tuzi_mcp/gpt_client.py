"""
GPT image generation client for the Tuzi MCP Server.

Handles GPT-4o async image generation with coordinated polling.
"""

import asyncio
import os
import re
import httpx
from datetime import datetime
from typing import List, Optional, Tuple

from fastmcp.exceptions import ToolError

from .image_utils import load_and_encode_images, prepare_multimodal_content
from .task_manager import ImageTask, polling_coordinator, task_manager


class GPTImageClient:
    """Handles GPT-4o async image generation"""
    
    def __init__(self):
        self.api_key = os.getenv("TUZI_API_KEY")
        self.base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")
        
        if not self.api_key:
            raise ToolError("TUZI_API_KEY environment variable is required")
    
    async def extract_async_urls(self, response_content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract preview and source URLs from async response content"""
        
        preview_match = re.search(r'\[preview\]\(([^)]+)\)', response_content)
        source_match = re.search(r'\[source\]\(([^)]+)\)', response_content)
        
        preview_url = preview_match.group(1) if preview_match else None
        source_url = source_match.group(1) if source_match else None
        
        
        return preview_url, source_url
    
    async def submit_async_request(
        self, 
        prompt: str, 
        model: str = "gpt-4o-image-async",
        aspect_ratio: str = "1:1",
        reference_image_paths: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Quickly submit async request and return preview/source URLs (no waiting/polling)
        
        Args:
            prompt: Text prompt for image generation
            model: Model to use for generation
            aspect_ratio: Aspect ratio of the generated image
            reference_image_paths: Optional list of paths to reference images for multimodal input
        
        Returns:
            tuple: (preview_url, source_url)
        """
        
        api_url = f"{self.base_url}/v1/chat/completions"
        
        # Use prompt as-is since it includes aspect ratio information
        enhanced_prompt = prompt
        
        # Handle reference images if provided
        image_data_urls = None
        if reference_image_paths:
            try:
                image_data_urls = await load_and_encode_images(reference_image_paths)
            except ToolError:
                raise  # Re-raise ToolError as-is
            except Exception as e:
                error_msg = f"Failed to process reference images: {str(e)}"
                raise ToolError(error_msg)
        
        # Prepare content for API request
        content = prepare_multimodal_content(enhanced_prompt, image_data_urls)
        
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
            # Increased timeout for initial API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    # Read streaming response chunks
                    response_content = ""
                    chunk_count = 0
                    max_chunks = 20  # Safety limit
                    
                    async for chunk in response.aiter_text():
                        chunk_count += 1
                        response_content += chunk
                        
                        if chunk_count >= max_chunks:
                            break
                    
                    # Extract URLs from complete response
                    preview_url, source_url = await self.extract_async_urls(response_content)
                    
                    
                    if not source_url:
                        error_msg = f"Failed to extract source URL from async response ({len(response_content)} chars): {response_content[:500]}..."
                        raise ToolError(error_msg)
            
            return preview_url, source_url
            
        except httpx.TimeoutException as e:
            error_msg = f"API request timeout: {str(e)}"
            raise ToolError(error_msg)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"API HTTP {e.response.status_code} error: {str(e)}"
            raise ToolError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected API error: {str(e)}"
            raise ToolError(error_msg)
    
    async def generate_task(self, task: ImageTask, prompt: str, model: str, reference_image_paths: Optional[List[str]] = None) -> None:
        """Execute an image generation task using coordinated polling"""
        short_id = task.task_id[:8] + "..."
        
        try:
            task.status = "running"
            start_time = datetime.now()
            
            # Phase 1: Quickly get source URL
            preview_url, source_url = await self.submit_async_request(
                prompt=prompt,
                model=model,
                aspect_ratio="1:1",
                reference_image_paths=reference_image_paths
            )
            
            phase1_elapsed = (datetime.now() - start_time).total_seconds()
            
            # Phase 2: Register with polling coordinator  
            await polling_coordinator.add_task_for_polling(task.task_id, source_url, task, preview_url)
            
            # Phase 3: Wait for coordinator to complete the task
            max_wait_time = 600  # Increased to 10 minutes (coordinator may take up to ~9.5 minutes now)
            wait_interval = 5    # Check every 5 seconds
            waited = 0
            
            while waited < max_wait_time:
                if task.status in ["completed", "failed"]:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    break
                
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                
            # If still running after max wait time, mark as failed
            if task.status == "running":
                elapsed = (datetime.now() - start_time).total_seconds()
                error_msg = f"Task timed out after {elapsed:.1f}s (max: {max_wait_time}s)"
                task.error = error_msg
                task.status = "failed"
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            task.error = error_msg
            task.status = "failed"


# Global instance
gpt_client = GPTImageClient()