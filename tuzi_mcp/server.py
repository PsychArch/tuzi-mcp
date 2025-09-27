#!/usr/bin/env python3
"""
Tuzi MCP Server - Image Generation with Async Task Management

Provides tools for submitting image generation requests and waiting for completion.
"""

import asyncio
import os
import sys
from typing import Literal, Optional
from typing import Annotated
from pydantic import Field

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.exceptions import ToolError
from mcp.types import TextContent

# Import our modular components
from .task_manager import task_manager, ImageTask
from .gpt_client import gpt_client
from .gemini_client import gemini_client
from .seedream_client import seedream_client
from .image_utils import validate_image_file

# Initialize FastMCP server
mcp = FastMCP("tuzi-mcp-server")


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
        
        # Start async execution using GPT client
        future = asyncio.create_task(gpt_client.generate_task(task, prompt, model, parsed_image_paths))
        task.future = future
        task_manager.active_tasks.append(future)
        
        return ToolResult(
            content=[TextContent(type="text", text=f"{task_id} submitted.")]
        )
        
    except Exception as e:
        raise ToolError(f"Failed to submit task: {str(e)}")


@mcp.tool
async def submit_gemini_image(
    prompt: Annotated[str, "The text prompt describing the image to generate. Must include aspect ratio (1:1, 3:2, 2:3, 16:9, 9:16, 4:5) in it. For generating multiple images, explicitly prompt like \"Generate 4 different images:\" "],
    output_path: Annotated[str, "Absolute path to save the generated image(s). When generating multiple images, files saved with _1, _2, _3, _4 suffixes"],
    reference_image_paths: Annotated[
        Optional[str],
        "Optional comma-separated paths (e.g., '/path/to/img1.png,/path/to/img2.png'). Supports PNG, JPEG, WebP, GIF, BMP."
    ] = None,
    hd: Annotated[
        bool,
        "HD quality. Only enable it when user explicitly requests. Only support .webp output under HD quality."
    ] = False,
    vip: Annotated[
        bool,
        "VIP model. Only use when normal model fails or user explicitly requests it."
    ] = False,
) -> ToolResult:
    """
    Submit a Gemini image generation task.

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
        
        # Select model based on VIP and HD parameters (VIP takes priority)
        if vip:
            model = "gemini-2.5-flash-image-vip"
        elif hd:
            model = "gemini-2.5-flash-image-hd"
        else:
            model = "gemini-2.5-flash-image"
        
        # Start async execution using Gemini client
        future = asyncio.create_task(gemini_client.generate_task(task, prompt, model, parsed_image_paths))
        task.future = future
        task_manager.active_tasks.append(future)
        
        return ToolResult(
            content=[TextContent(type="text", text=f"{task_id} submitted.")]
        )
        
    except Exception as e:
        raise ToolError(f"Failed to submit Gemini task: {str(e)}")


@mcp.tool
async def submit_seedream_image(
    prompt: Annotated[str, "Prompt for image generation or editing. No aspect ratio or image size needed."],
    output_path: Annotated[str, "Absolute path to save the image(s). Only support JPEG format."],
    size: Annotated[
        Literal[
            "1024x1024",
            "2048x2048",
            "4096x4096",
            "2560x1440",  # 16:9
            "1440x2560",  # 9:16
            "2304x1728",  # 4:3
            "1728x2304",  # 3:4
            "2496x1664",  # 3:2
            "1664x2496",  # 2:3
            "3024x1296",  # 21:9
        ],
        "Image size（WxH)"
    ] = "1024x1024",
    quality: Annotated[
        Literal["standard", "high"],
        "Image quality for generations. Default 'high'."
    ] = "high",
    n: Annotated[
        int,
        Field(ge=1, le=8, description="Number of images to generate.")
    ] = 1,
    reference_image_paths: Annotated[
        Optional[str],
        "Optional comma-separated reference image paths."
    ] = None,
) -> ToolResult:
    """
    Submit a Seedream (即梦) image generation/editing task. Suitable for Chinese-context tasks.

    For generating multiple images while keeping same style (n>1), use the prompt format:
    '[General style description] Create separate images: Image 1: [description], Image 2: [description], Image 3: [description], Image 4: [description]'
    
    Output files are saved with _1, _2, _3, _4 suffixes when n>1.
    
    Use wait_tasks() to wait for completion.
    """
    try:
        # Create task
        task_id = task_manager.create_task(output_path)
        task = task_manager.get_task(task_id)

        # Parse reference paths (only first is used for edits)
        edit_image_path = None
        if reference_image_paths:
            paths = [p.strip() for p in reference_image_paths.split(',') if p.strip()]
            if paths:
                edit_image_path = paths[0]

        # Start async execution using Seedream images client (generation or edit)
        future = asyncio.create_task(
            seedream_client.generate_task(
                task,
                prompt,
                size=size,
                quality=quality,
                n=n,
                edit_image_path=edit_image_path,
            )
        )
        task.future = future
        task_manager.active_tasks.append(future)

        return ToolResult(
            content=[TextContent(type="text", text=f"{task_id} submitted.")]
        )

    except Exception as e:
        raise ToolError(f"Failed to submit Seedream task: {str(e)}")


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
            # Check if any tasks have warnings
            tasks_with_warnings = [task for task in completed_tasks if task.get('warning')]
            
            if tasks_with_warnings:
                # Show detailed format with warnings
                status_message += f"\ncompleted_tasks({len(completed_tasks)}):"
                for task in completed_tasks:
                    duration_str = f" ({task['elapsed_time']:.1f}s)" if task.get('elapsed_time') else ""
                    if task.get('warning'):
                        status_message += f"\n- {task['task_id']}{duration_str}: {task['warning']}"
                    else:
                        status_message += f"\n- {task['task_id']}{duration_str}"
            else:
                # Use concise format when no warnings
                task_list = ", ".join([f"{task['task_id']} ({task['elapsed_time']:.1f}s)" if task.get('elapsed_time') else task['task_id'] for task in completed_tasks])
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
        
        return ToolResult(
            content=[TextContent(type="text", text=status_message)]
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
        
        if not filtered_tasks:
            message = "No tasks found"
            if status_filter:
                message += f" with status '{status_filter}'"
        else:
            # Show detailed task list for LLM decision making
            message = f"Tasks ({len(filtered_tasks)} found"
            if status_filter:
                message += f", filtered by '{status_filter}'"
            message += "):\n"
            
            for task in filtered_tasks:
                status_display = task.status.upper()
                message += f"- {task.task_id}: {status_display}"
                if task.status == "failed" and task.error:
                    message += f" - Error: {task.error}"
                message += "\n"
            
            # Remove trailing newline
            message = message.rstrip()
        
        return ToolResult(
            content=[TextContent(type="text", text=message)]
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
