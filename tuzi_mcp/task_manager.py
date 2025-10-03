"""
Task management and polling coordination for the Tuzi MCP Server.

Handles image generation task lifecycle and efficient polling of multiple source URLs.
"""

import asyncio
import base64
import json
import re
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional

from .image_utils import adjust_path_for_image_bytes, download_image_from_url, save_image_to_file


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
        self.warnings: List[str] = []  # Unified warning collection

    def add_warning(self, warning: str) -> None:
        """Add a warning message to the task"""
        if warning:
            self.warnings.append(warning)

    def get_warnings_string(self) -> Optional[str]:
        """Get all warnings as a single string, or None if no warnings"""
        if not self.warnings:
            return None
        return " | ".join(self.warnings)


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

                # Include unified warnings if available
                warning_str = task.get_warnings_string()
                if warning_str:
                    task_info["warning"] = warning_str

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
        
        # Collect polling errors from coordinator
        polling_errors = polling_coordinator.polling_errors.copy() if polling_coordinator.polling_errors else []

        return {
            "message": f"All tasks completed: {len(completed_tasks)} successful, {len(failed_tasks)} failed",
            "completed_count": len(completed_tasks),
            "total_completed": len(completed_tasks),
            "total_failed": len(failed_tasks),
            "still_running_count": len(still_running),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "still_running": still_running,
            "elapsed_time": elapsed,
            "polling_errors": polling_errors
        }
    
    def record_completion_time(self, completion_time_seconds: float) -> None:
        """Record a task completion time for adaptive wait calculation"""
        self.completion_times.append(completion_time_seconds)
        if len(self.completion_times) > self.max_history:
            self.completion_times.pop(0)  # Remove oldest
    
    def get_adaptive_wait_time(self) -> int:
        """Calculate adaptive initial wait time based on completion history"""
        if len(self.completion_times) < 3:  # Need at least 3 samples
            return 10  # Default fallback - start polling quickly

        avg_completion = sum(self.completion_times) / len(self.completion_times)
        adaptive_wait = int(avg_completion * 0.1)  # 10% of average (conservative early start)
        # Cap between 5-15 seconds - don't wait too long before first poll
        capped_wait = max(5, min(15, adaptive_wait))

        return capped_wait


class PollingCoordinator:
    """Coordinates polling of multiple source URLs efficiently"""

    def __init__(self):
        self.polling_tasks: Dict[str, Dict] = {}  # task_id -> {source_url, task, result}
        self.polling_active = False
        self.polling_lock = asyncio.Lock()
        self.polling_errors: List[str] = []  # Collect errors during polling for reporting
        
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
            self.polling_errors.clear()  # Clear previous errors

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
                            
                            # Collect any exceptions from individual polling tasks
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    task_id = incomplete_task_ids[i] if i < len(incomplete_task_ids) else "unknown"
                                    error_msg = f"Polling error for {task_id}: {type(result).__name__}: {str(result)}"
                                    self.polling_errors.append(error_msg)

                        except asyncio.TimeoutError as e:
                            error_msg = f"Polling round timeout after {poll_timeout + 10.0}s"
                            self.polling_errors.append(error_msg)
            except Exception as e:
                error_msg = f"HTTP client error: {type(e).__name__}: {str(e)}"
                self.polling_errors.append(error_msg)
            
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

            # Find all URLs with image suffixes and take the last one (most recent)
            all_image_urls = re.findall(r'https://[^\s\)\]]+\.(?:png|jpg|jpeg|webp)[^\s\)\]]*', content)

            if not all_image_urls:
                # No image URLs found yet, continue polling
                return

            final_url = all_image_urls[-1]  # Take the last (most recent) URL
            
            
            # Download the image
            try:
                image_data = await download_image_from_url(final_url)
                b64_image = base64.b64encode(image_data).decode('utf-8')

                task = task_info['task']

                actual_path, format_warning = adjust_path_for_image_bytes(task.output_path, image_data)
                if format_warning:
                    task.add_warning(format_warning)

                saved_path, save_warning = await save_image_to_file(b64_image, actual_path)
                if save_warning:
                    task.add_warning(save_warning)

                task.output_path = saved_path
                task.status = "completed"
                task_info['completed'] = True

                task_manager.record_completion_time(elapsed)
                    
            except Exception as e:
                error_msg = f"Failed to download image: {str(e)}"
                task_info['task'].error = error_msg
                task_info['task'].status = "failed"
                task_info['completed'] = True

        except httpx.HTTPStatusError as e:
            # Don't mark as completed on HTTP errors, will retry
            pass
        except httpx.TimeoutException as e:
            # Don't mark as completed on timeout errors, will retry
            pass
        except Exception as e:
            # Don't mark as completed on unexpected errors, will retry
            pass


# Global instances
polling_coordinator = PollingCoordinator()
task_manager = TaskManager()

