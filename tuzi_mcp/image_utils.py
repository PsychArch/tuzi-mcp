"""
Image processing utilities for the Tuzi MCP Server.

Provides functions for image validation, encoding, downloading, and file operations.
"""

import base64
import os
import httpx
import tempfile
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastmcp.exceptions import ToolError


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


async def save_image_to_file(b64_image: str, output_path: str) -> Tuple[str, Optional[str]]:
    """Save base64 image data to file with fail-safe temp directory fallback
    
    Returns:
        Tuple of (actual_save_path, warning_message)
        - actual_save_path: Where the file was actually saved
        - warning_message: None if saved to intended path, warning if fallback used
    """
    # Decode base64 image
    image_data = base64.b64decode(b64_image)
    
    # Try to save to the intended path first
    try:
        # Create parent directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write image to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return output_path, None
        
    except (OSError, PermissionError, FileNotFoundError) as e:
        # Fallback to temp directory with timestamp-based filename
        original_path = Path(output_path)
        original_extension = original_path.suffix.lower() if original_path.suffix else '.png'
        
        # Always use timestamp format to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_filename = f"tuzi_image_{timestamp}{original_extension}"
        
        temp_dir = tempfile.gettempdir()
        fallback_path = os.path.join(temp_dir, fallback_filename)
        
        try:
            with open(fallback_path, 'wb') as f:
                f.write(image_data)
            
            warning_msg = f"Could not save to '{output_path}' ({str(e)}), saved to '{fallback_path}' instead"
            return fallback_path, warning_msg
            
        except Exception as fallback_error:
            # If even temp directory fails, raise the original error
            raise ToolError(f"Failed to save image to '{output_path}': {str(e)}. Temp directory fallback also failed: {str(fallback_error)}")