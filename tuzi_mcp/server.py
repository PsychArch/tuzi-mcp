#!/usr/bin/env python3
"""
Tuzi MCP Server - GPT Image Generation with Async Task Management

Provides tools for submitting GPT image generation requests and waiting for completion.
"""

import asyncio
import base64
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Union
from typing import Annotated
from pydantic import Field, BaseModel
import httpx

# Setup logging (disabled by default)
if os.getenv("TUZI_ENABLE_LOGGING"):
    log_file = os.path.join(os.getcwd(), "tuzi_mcp_debug.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr)
        ]
    )
logger = logging.getLogger(__name__)
from mimetypes import guess_type

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.exceptions import ToolError
from mcp.types import TextContent



# Task tracking
class ImageTask:
    """Represents an image generation task"""
    
    def __init__(self, task_id: str, output_path: str):
        self.task_id = task_id
        self.output_path = output_path
        self.status = "pending"
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.future: Optional[asyncio.Task] = None
        self.start_time = datetime.now()


class TaskManager:
    """Manages async image generation tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, ImageTask] = {}
        self.active_tasks: List[asyncio.Task] = []
        self.task_counter: int = 0
        self.completion_times: List[float] = []  # Rolling list of completion times in seconds
        self.max_history: int = 5  # Keep last 5 completions
    
    def create_task(self, output_path: str) -> str:
        """Create a new task and return its ID"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter:04d}"
        task = ImageTask(task_id, output_path)
        self.tasks[task_id] = task
        return task_id
    
    def get_task(self, task_id: str) -> Optional[ImageTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[ImageTask]:
        """Get all pending tasks"""
        return [task for task in self.tasks.values() if task.status == "pending"]
    
    def get_active_tasks(self) -> List[ImageTask]:
        """Get all running tasks"""
        return [task for task in self.tasks.values() if task.status == "running"]
    
    async def wait_all_tasks(self, timeout_seconds: int = 600, auto_cleanup: bool = True) -> Dict[str, Any]:
        """Wait for all active tasks to complete with detailed result processing"""
        if not self.active_tasks:
            return {
                "message": "No active tasks", 
                "completed_count": 0,
                "completed_tasks": [],
                "failed_tasks": [],
                "still_running": []
            }
        
        active_count = len(self.active_tasks)
        start_time = datetime.now()
        elapsed = 0.0  # Initialize elapsed time
        
        # Wait for all tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.active_tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
            elapsed = (datetime.now() - start_time).total_seconds()
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
        
        # Collect detailed results
        completed_tasks = []
        failed_tasks = []
        still_running = []
        
        for task in self.tasks.values():
            # Calculate individual task elapsed time
            task_elapsed = (datetime.now() - task.start_time).total_seconds()
            
            if task.status == "completed":
                task_info = {
                    "task_id": task.task_id,
                    "status": task.status,
                    "elapsed_time": task_elapsed
                }
                
                completed_tasks.append(task_info)
                
            elif task.status == "failed":
                failed_tasks.append({
                    "task_id": task.task_id,
                    "error": task.error,
                    "status": task.status,
                    "elapsed_time": task_elapsed
                })
            elif task.status in ["pending", "running"]:
                still_running.append({
                    "task_id": task.task_id,
                    "status": task.status,
                    "elapsed_time": task_elapsed
                })
        
        # Clear completed active tasks
        self.active_tasks = [task for task in self.active_tasks if not task.done()]
        
        # Auto-cleanup finished tasks if requested
        if auto_cleanup:
            completed_task_ids = [task["task_id"] for task in completed_tasks]
            failed_task_ids = [task["task_id"] for task in failed_tasks]
            all_finished_task_ids = completed_task_ids + failed_task_ids
            
            for task_id in all_finished_task_ids:
                if task_id in self.tasks:
                    del self.tasks[task_id]
        
        return {
            "message": f"All tasks completed: {len(completed_tasks)} successful, {len(failed_tasks)} failed",
            "completed_count": len(completed_tasks),
            "total_completed": len(completed_tasks),
            "total_failed": len(failed_tasks),
            "still_running_count": len(still_running),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "still_running": still_running,
            "elapsed_time": elapsed
        }
    
    def record_completion_time(self, completion_time_seconds: float) -> None:
        """Record a task completion time for adaptive wait calculation"""
        self.completion_times.append(completion_time_seconds)
        if len(self.completion_times) > self.max_history:
            self.completion_times.pop(0)  # Remove oldest
    
    def get_adaptive_wait_time(self) -> int:
        """Calculate adaptive initial wait time based on completion history"""
        if len(self.completion_times) < 3:  # Need at least 3 samples
            return 90  # Default fallback
        
        avg_completion = sum(self.completion_times) / len(self.completion_times)
        adaptive_wait = int(avg_completion * 0.8)  # 80% of average
        # Cap between 30-120 seconds for safety
        capped_wait = max(30, min(120, adaptive_wait))
        
        return capped_wait


# Polling coordinator for efficient multi-task handling
class PollingCoordinator:
    """Coordinates polling of multiple source URLs efficiently"""
    
    def __init__(self):
        self.polling_tasks: Dict[str, Dict] = {}  # task_id -> {source_url, task, result}
        self.polling_active = False
        self.polling_lock = asyncio.Lock()
        
    async def add_task_for_polling(self, task_id: str, source_url: str, task: 'ImageTask', preview_url: str = None):
        """Add a task to the polling coordinator"""
        async with self.polling_lock:
            self.polling_tasks[task_id] = {
                'source_url': source_url,
                'preview_url': preview_url,
                'task': task,
                'result': None,
                'completed': False
            }
            
            # Start polling if not already active
            if not self.polling_active:
                asyncio.create_task(self._poll_all_sources())
    
    async def _poll_all_sources(self):
        """Main polling loop that handles all source URLs concurrently"""
        start_time = datetime.now()
        async with self.polling_lock:
            if self.polling_active:
                return  # Already polling
            self.polling_active = True
        
        
        # Use adaptive wait time based on completion history
        adaptive_wait = task_manager.get_adaptive_wait_time()
        await asyncio.sleep(adaptive_wait)
        
        # Increased max attempts and timeout handling
        max_attempts = 40  # Increased from 30 to 40 to handle tasks completing around 420-450s
        poll_timeout = 45.0  # Increased from 30.0 to 45.0 seconds
        
        for attempt in range(max_attempts):
            if not self.polling_tasks:  # All tasks completed
                break
                
            remaining_count = len([t for t in self.polling_tasks.values() if not t['completed']])
            
            # Poll all incomplete tasks in parallel with increased timeout
            try:
                async with httpx.AsyncClient(timeout=poll_timeout) as client:
                    poll_tasks = []
                    incomplete_task_ids = [
                        task_id for task_id, info in self.polling_tasks.items() 
                        if not info['completed']
                    ]
                    
                    
                    for task_id in incomplete_task_ids:
                        poll_tasks.append(
                            self._check_single_source(client, task_id, self.polling_tasks[task_id])
                        )
                    
                    if poll_tasks:
                        # Add timeout protection for the gather operation
                        try:
                            results = await asyncio.wait_for(
                                asyncio.gather(*poll_tasks, return_exceptions=True),
                                timeout=poll_timeout + 10.0  # Give extra 10s buffer
                            )
                            
                            # Log any exceptions from individual polling tasks
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    task_id = incomplete_task_ids[i] if i < len(incomplete_task_ids) else "unknown"
                                    
                        except asyncio.TimeoutError:
                            pass
            except Exception as e:
                pass
            
            # Remove completed tasks and log progress
            completed_tasks = [
                task_id for task_id, info in self.polling_tasks.items()
                if info['completed']
            ]
            
            for task_id in completed_tasks:
                del self.polling_tasks[task_id]
            
            if not self.polling_tasks:  # All done
                elapsed = (datetime.now() - start_time).total_seconds()
                break
                
            # Wait before next polling round
            await asyncio.sleep(10)
        
        # Handle any remaining incomplete tasks
        if self.polling_tasks:
            elapsed = (datetime.now() - start_time).total_seconds()
            timeout_duration = adaptive_wait + max_attempts * 10
            
            for task_id, info in self.polling_tasks.items():
                if not info['completed']:
                    error_msg = f"Image generation timed out after {elapsed:.1f}s (max: ~{timeout_duration}s)"
                    info['task'].error = error_msg
                    info['task'].status = "failed"
        
        async with self.polling_lock:
            self.polling_active = False
            self.polling_tasks.clear()
        
        elapsed = (datetime.now() - start_time).total_seconds()
    
    async def _check_single_source(self, client: httpx.AsyncClient, task_id: str, task_info: Dict):
        """Check a single source URL and handle completion"""
        source_url = task_info['source_url']
        short_id = task_id[:8] + "..."
        
        # Calculate elapsed time from task creation
        task_start_time = task_info['task'].start_time if hasattr(task_info['task'], 'start_time') else datetime.now()
        elapsed = (datetime.now() - task_start_time).total_seconds()
        
        try:
            response = await client.get(source_url)
            response.raise_for_status()
            
            # Parse JSON response and extract content field
            try:
                json_data = response.json()
                content = json_data.get('content', '')
            except json.JSONDecodeError:
                # Fallback to raw text if not JSON
                content = response.text
            
            
            # Look for final image URL - prioritize download URLs (language-independent)
            download_url_match = re.search(r'https://filesystem\.site/cdn/download/[^\s\)\]]+\.(?:png|jpg|jpeg|webp)', content)
            if download_url_match:
                final_url = download_url_match.group(0)
            else:
                # Fallback to display image URL from filesystem.site
                display_url_match = re.search(r'https://filesystem\.site/cdn/[^\s\)\]]+\.(?:png|jpg|jpeg|webp)', content)
                if display_url_match:
                    final_url = display_url_match.group(0)
                else:
                    # Still no image URL found, check if still processing by looking for specific domain patterns
                    if "asyncdata.net" in content and not re.search(r'https://filesystem\.site/cdn/', content):
                        return  # Still processing
                    else:
                        return  # No image URL found
            
            
            # Download the image
            try:
                image_data = await download_image_from_url(final_url)
                b64_image = base64.b64encode(image_data).decode('utf-8')
                
                
                # Prepare result
                result = {
                    "data": [{"b64_json": b64_image, "url": final_url}],
                    "preview_url": task_info.get('preview_url'),
                    "source_url": task_info['source_url'],
                    "final_url": final_url
                }
                
                # Save image to file
                task = task_info['task']
                if result and "data" in result and len(result["data"]) > 0:
                    first_image = result["data"][0]
                    if "b64_json" in first_image and first_image["b64_json"]:
                        await save_image_to_file(first_image["b64_json"], task.output_path)
                        result["saved_to"] = task.output_path
                
                # Update task
                task.result = result
                task.status = "completed"
                task_info['completed'] = True
                task_info['result'] = result
                
                # Record completion time for adaptive wait calculation
                
                # Record completion time for adaptive wait calculation
                task_manager.record_completion_time(elapsed)
                    
            except Exception as e:
                error_msg = f"Failed to download image: {str(e)}"
                task_info['task'].error = error_msg
                task_info['task'].status = "failed"
                task_info['completed'] = True
            else:
                # No image URL found yet, but no error either - keep polling
                pass
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error"
            # Don't mark as completed on HTTP errors, will retry
            pass
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {str(e)}"
            # Don't mark as completed on timeout errors, will retry
            pass
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            # Don't mark as completed on polling errors, will retry
            pass


# Global instances
polling_coordinator = PollingCoordinator()
task_manager = TaskManager()

# Initialize FastMCP server
mcp = FastMCP("tuzi-mcp-server")


# Image processing helper functions
async def validate_image_file(image_path: str) -> None:
    """Validate image file exists and is supported format"""
    if not image_path:
        raise ToolError("Image path cannot be empty")
    
    path = Path(image_path)
    
    # Check if file exists
    if not path.exists():
        raise ToolError(f"Image file not found: {image_path}")
    
    # Check if it's a file (not directory)
    if not path.is_file():
        raise ToolError(f"Path is not a file: {image_path}")
    
    # Check file extension
    supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}
    if path.suffix.lower() not in supported_extensions:
        raise ToolError(f"Unsupported image format: {path.suffix}. Supported: {', '.join(supported_extensions)}")
    
    # Check file size (limit to 20MB)
    max_size = 20 * 1024 * 1024  # 20MB
    file_size = path.stat().st_size
    if file_size > max_size:
        raise ToolError(f"Image file too large: {file_size / (1024*1024):.1f}MB (max: 20MB)")


async def validate_image_files(image_paths: List[str]) -> None:
    """Validate multiple image files"""
    if not image_paths:
        return
    
    for i, image_path in enumerate(image_paths):
        try:
            await validate_image_file(image_path)
        except ToolError as e:
            raise ToolError(f"Reference image {i+1}: {str(e)}")
    


async def load_and_encode_image(image_path: str) -> str:
    """Load image file and convert to base64 data URL"""
    await validate_image_file(image_path)
    
    # Guess MIME type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        # Fallback based on extension
        ext = Path(image_path).suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_map.get(ext, 'image/jpeg')
    
    try:
        # Read and encode image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
        
        # Construct data URL
        data_url = f"data:{mime_type};base64,{base64_encoded}"
        
        return data_url
        
    except Exception as e:
        error_msg = f"Failed to load/encode image {image_path}: {str(e)}"
        raise ToolError(error_msg)


async def load_and_encode_images(image_paths: List[str]) -> List[str]:
    """Load multiple image files and convert to base64 data URLs"""
    if not image_paths:
        return []
    
    await validate_image_files(image_paths)
    
    data_urls = []
    for image_path in image_paths:
        try:
            data_url = await load_and_encode_image(image_path)
            data_urls.append(data_url)
        except ToolError:
            raise  # Re-raise ToolError as-is
        except Exception as e:
            error_msg = f"Failed to process reference image {image_path}: {str(e)}"
            raise ToolError(error_msg)
    
    return data_urls


def prepare_multimodal_content(prompt: str, image_data_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Prepare content array for multimodal API request"""
    content = [{"type": "text", "text": prompt}]
    
    # Handle multiple images
    if image_data_urls:
        for data_url in image_data_urls:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "high"  # Use high detail for better quality
                }
            })
    
    return content



async def extract_async_urls(response_content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract preview and source URLs from async response content"""
    
    preview_match = re.search(r'\[preview\]\(([^)]+)\)', response_content)
    source_match = re.search(r'\[source\]\(([^)]+)\)', response_content)
    
    preview_url = preview_match.group(1) if preview_match else None
    source_url = source_match.group(1) if source_match else None
    
    
    return preview_url, source_url



async def download_image_from_url(image_url: str) -> bytes:
    """Download image data from URL"""
    
    try:
        # Increased timeout for image downloads
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
            return image_data
            
    except httpx.TimeoutException as e:
        error_msg = f"Image download timeout after 120s: {str(e)}"
        raise ToolError(error_msg)
        
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error downloading image: {str(e)}"
        raise ToolError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error downloading image: {str(e)}"
        raise ToolError(error_msg)


async def get_source_url_fast(
    prompt: str, 
    model: str = "gpt-4o-image-async", 
    aspect_ratio: str = "1:1",
    reference_image_paths: Optional[List[str]] = None
) -> tuple[str, str]:
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
    
    api_key = os.getenv("TUZI_API_KEY")
    if not api_key:
        error_msg = "TUZI_API_KEY environment variable is required"
        raise ToolError(error_msg)
    
    # Get base URL (with default)
    base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")
    api_url = f"{base_url}/v1/chat/completions"
    
    
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
        "Authorization": f"Bearer {api_key}",
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
                preview_url, source_url = await extract_async_urls(response_content)
                
                
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




async def save_image_to_file(b64_image: str, output_path: str) -> None:
    """Save base64 image data to file"""
    # Decode base64 image
    image_data = base64.b64decode(b64_image)
    
    # Create parent directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write image to file
    with open(output_path, 'wb') as f:
        f.write(image_data)


async def generate_image_task(task: ImageTask, prompt: str, model: str, reference_image_paths: Optional[List[str]] = None) -> None:
    """Execute an image generation task using coordinated polling"""
    short_id = task.task_id[:8] + "..."
    
    try:
        task.status = "running"
        start_time = datetime.now()
        
        # Phase 1: Quickly get source URL
        preview_url, source_url = await get_source_url_fast(
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


@mcp.tool
async def submit_gpt_image(
    prompt: Annotated[str, "The text prompt describing the image to generate. Must include aspect ratio (1:1, 3:2, or 2:3) in it"],
    output_path: Annotated[str, "Absolute path to save the generated image"],
    model: Annotated[
        Literal["gpt-4o-image-async", "gpt-4o-image-vip-async"], 
        "The GPT image model to use -- only use gpt-4o-image-vip-async when failure rate is too high"
    ] = "gpt-4o-image-async",
    reference_image_paths: Annotated[
        Optional[str],
        "Optional comma-separated paths (e.g., '/path/to/img1.png,/path/to/img2.png'). Supports PNG, JPEG, WebP, GIF, BMP."
    ] = None,
) -> ToolResult:
    """
    Submit an async GPT image generation task.
    
    Use wait_tasks() to wait for all submitted tasks to complete.
    """
    try:
        # Parse comma-separated reference image paths
        parsed_image_paths = None
        if reference_image_paths:
            # Split by comma and strip whitespace
            parsed_image_paths = [path.strip() for path in reference_image_paths.split(',') if path.strip()]
        
        # Create task
        task_id = task_manager.create_task(output_path)
        task = task_manager.get_task(task_id)
        
        # Start async execution
        future = asyncio.create_task(generate_image_task(task, prompt, model, parsed_image_paths))
        task.future = future
        task_manager.active_tasks.append(future)
        
        result_data = {
            "task_id": task_id,
            "status": "submitted"
        }
        
        return ToolResult(
            content=[TextContent(type="text", text=f"{task_id} submitted.")],
            structured_content=result_data
        )
        
    except Exception as e:
        raise ToolError(f"Failed to submit task: {str(e)}")


@mcp.tool
async def wait_tasks(
    timeout_seconds: Annotated[
        int, 
        Field(ge=30, le=1200, description="Maximum time to wait for tasks (30-1200 seconds)")
    ] = 600
) -> ToolResult:
    """
    Wait for all previously submitted image generation tasks to complete.
    """
    try:
        # Delegate core logic to TaskManager
        result = await task_manager.wait_all_tasks(timeout_seconds=timeout_seconds, auto_cleanup=True)
        
        # Format message for MCP response
        completed_tasks = result["completed_tasks"]
        failed_tasks = result["failed_tasks"]
        still_running = result["still_running"]
        
        status_message = ""

        # Show task status for each task
        if completed_tasks:
            task_list = ", ".join([f"{task['task_id']} ({task['duration']:.1f}s)" if task.get('duration') else task['task_id'] for task in completed_tasks])
            status_message += f"\ncompleted_tasks({len(completed_tasks)}): {task_list}"
        
        if failed_tasks:
            status_message += f"\nfailed_tasks({len(failed_tasks)}):"
            for task in failed_tasks:
                task_id = task['task_id']
                error = task.get('error', 'Unknown error')
                status_message += f"\n- {task_id}: {error}"
        
        if still_running:
            task_list = ", ".join([task['task_id'] for task in still_running])
            status_message += f"\nrunning_tasks({len(still_running)}): {task_list}"
        
        # Prepare summary for structured content
        summary = {
            "total_completed": result["total_completed"],
            "total_failed": result["total_failed"],
            "still_running": result["still_running"]
        }
        
        return ToolResult(
            content=[TextContent(type="text", text=status_message)],
            structured_content=summary
        )
        
    except Exception as e:
        raise ToolError(f"Failed to wait for tasks: {str(e)}")


@mcp.tool
async def list_tasks(
    status_filter: Annotated[
        Optional[Literal["pending", "running", "completed", "failed"]], 
        "Filter tasks by status"
    ] = None
) -> ToolResult:
    """
    List all image generation tasks with their current status.
    
    Args:
        status_filter: Optional filter to show only tasks with specific status
    
    Returns:
        List of tasks with their details and status
    """
    try:
        all_tasks = list(task_manager.tasks.values())
        
        if status_filter:
            filtered_tasks = [task for task in all_tasks if task.status == status_filter]
        else:
            filtered_tasks = all_tasks
        
        tasks_info = []
        for task in filtered_tasks:
            task_info = {
                "task_id": task.task_id,
                "status": task.status
            }
            
            if task.error:
                task_info["error"] = task.error
            
            tasks_info.append(task_info)
        
        summary = {
            "total_tasks": len(all_tasks),
            "filtered_tasks": len(filtered_tasks),
            "filter": status_filter,
            "tasks": tasks_info
        }
        
        message = f"Found {len(filtered_tasks)} tasks"
        if status_filter:
            message += f" with status '{status_filter}'"
        
        # Add error details for failed tasks
        failed_task_count = len([task for task in filtered_tasks if task.status == "failed"])
        if failed_task_count > 0:
            message += f"\n\nFailed tasks ({failed_task_count}):"
            for task in filtered_tasks:
                if task.status == "failed":
                    error_msg = task.error or "Unknown error"
                    message += f"\n- {task.task_id}: {error_msg}"
        
        return ToolResult(
            content=[TextContent(type="text", text=message)],
            structured_content=summary
        )
        
    except Exception as e:
        raise ToolError(f"Failed to list tasks: {str(e)}")


def main():
    """Main entry point for the MCP server."""
    # Check for API key
    if not os.getenv("TUZI_API_KEY"):
        print("TUZI_API_KEY environment variable not set", file=sys.stderr)
        print("Please set your Tu-zi API key: export TUZI_API_KEY='your-api-key'", file=sys.stderr)
    
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
