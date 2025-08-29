#!/usr/bin/env python3
"""
Tuzi MCP Server - GPT/Gemini Image Generation with Async Task Management

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
async def submit_gemini_image(
    prompt: Annotated[str, "The text prompt describing the image to generate. Must include aspect ratio (1:1, 3:2, 2:3, 16:9, 9:16, 4:5) in it"],
    output_path: Annotated[str, "Absolute path to save the generated image"],
    reference_image_paths: Annotated[
        Optional[str],
        "Optional comma-separated paths (e.g., '/path/to/img1.png,/path/to/img2.png'). Supports PNG, JPEG, WebP, GIF, BMP."
    ] = None,
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
        
        # Start async execution using Gemini client
        future = asyncio.create_task(gemini_client.generate_task(task, prompt, "gemini-2.5-flash-image", parsed_image_paths))
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
        raise ToolError(f"Failed to submit Gemini task: {str(e)}")


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
        
        if filtered_tasks:
            message += "\n\nTasks:"
            for task in filtered_tasks:
                status_display = task.status.upper()
                message += f"\n- {task.task_id}: {status_display}"
                if task.status == "failed" and task.error:
                    message += f" ({task.error})"
        
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
