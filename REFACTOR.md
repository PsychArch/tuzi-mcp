# Tuzi MCP Server Refactoring Guide

## Overview
This document outlines the refactoring plan to transform the monolithic 1116-line `server.py` into a clean, modular architecture with clear separation of concerns.

## Current State
- **File**: `tuzi_mcp/server.py` (1116 lines)
- **Issues**: 
  - Mixed responsibilities (task management, API calls, image processing, MCP tools)
  - Difficult to test individual components
  - Hard to add new image providers
  - Complex polling logic intertwined with business logic

## Target Architecture

### File Structure
```
tuzi_mcp/
├── server.py           # MCP server entry & tool definitions (~200 lines)
├── task_manager.py     # Task tracking & polling coordination (~400 lines)
├── gpt_client.py       # GPT-specific image generation (~200 lines)
├── gemini_client.py    # Gemini-specific image generation (~150 lines)
└── image_utils.py      # Shared image utilities (~150 lines)
```

### Module Responsibilities

1. **server.py**: MCP tool definitions and server initialization
2. **task_manager.py**: Task lifecycle management and polling coordination
3. **gpt_client.py**: GPT-4o async image generation logic
4. **gemini_client.py**: Gemini image generation logic  
5. **image_utils.py**: Shared utilities (validation, encoding, file I/O)

## Refactoring Tasks

### Phase 1: Extract Shared Utilities
**Goal**: Create `image_utils.py` with reusable functions

#### Task 1.1: Create image_utils.py
Extract from `server.py` (lines to move):
- Lines 405-430: `validate_image_file()`
- Lines 432-442: `validate_image_files()`
- Lines 445-478: `load_and_encode_image()`
- Lines 480-498: `load_and_encode_images()`
- Lines 501-516: `prepare_multimodal_content()`
- Lines 767-777: `save_image_to_file()`
- Lines 550-572: `download_image_from_url()`

**Dependencies**: 
- Keep all imports needed by these functions
- Add proper module docstring

### Phase 2: Extract Task Management
**Goal**: Create `task_manager.py` with task tracking and polling

#### Task 2.1: Create task_manager.py
Extract from `server.py` (lines to move):
- Lines 45-56: `ImageTask` class
- Lines 58-187: `TaskManager` class
- Lines 190-393: `PollingCoordinator` class
- Lines 397-398: Global instances creation

**Key Considerations**:
- Import `image_utils` for `download_image_from_url` and `save_image_to_file`
- Keep the `_check_single_source` method's URL extraction logic
- Maintain global instances at module level

#### Task 2.2: Update PollingCoordinator dependencies
- Update imports to use `image_utils.download_image_from_url`
- Update imports to use `image_utils.save_image_to_file`
- Keep URL extraction patterns (lines 325-338)

### Phase 3: Extract GPT Client
**Goal**: Create `gpt_client.py` for GPT-specific logic

#### Task 3.1: Create gpt_client.py
Extract from `server.py` (lines to move):
- Lines 520-530: `extract_async_urls()` (GPT-specific)
- Lines 575-680: `get_source_url_fast()`
- Lines 780-827: `generate_image_task()`

**Class Structure**:
```python
class GPTImageClient:
    def __init__(self):
        self.api_key = os.getenv("TUZI_API_KEY")
        self.base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")
    
    async def submit_async_request(...)
    async def extract_urls(...)
    async def generate_task(...)
```

**Dependencies**:
- Import from `image_utils`: validation, encoding functions
- Import from `task_manager`: ImageTask, polling_coordinator

### Phase 4: Extract Gemini Client
**Goal**: Create `gemini_client.py` for Gemini-specific logic

#### Task 4.1: Create gemini_client.py
Extract from `server.py` (lines to move):
- Lines 533-546: `extract_gemini_image()` (Gemini-specific)
- Lines 683-762: `stream_gemini_api()`
- Lines 829-892: `generate_gemini_image_task()`

**Class Structure**:
```python
class GeminiImageClient:
    def __init__(self):
        self.api_key = os.getenv("TUZI_API_KEY")
        self.base_url = os.getenv("TUZI_URL_BASE", "https://api.tu-zi.com")
    
    async def stream_api(...)
    async def extract_image(...)
    async def generate_task(...)
```

**Dependencies**:
- Import from `image_utils`: validation, encoding, download functions
- Import from `task_manager`: ImageTask, task_manager

### Phase 5: Refactor Main Server
**Goal**: Slim down `server.py` to just MCP tools and coordination

#### Task 5.1: Update imports
Replace internal functions with module imports:
```python
from .task_manager import task_manager, ImageTask
from .gpt_client import GPTImageClient
from .gemini_client import GeminiImageClient
from .image_utils import validate_image_file
```

#### Task 5.2: Simplify MCP tools
Update each `@mcp.tool` function to:
1. Parse and validate inputs using imported utilities
2. Create task via task_manager
3. Delegate to appropriate client (GPT or Gemini)
4. Return standardized ToolResult

#### Task 5.3: Clean up server.py
Remove all extracted code, leaving only:
- Lines 1-22: Headers and imports (updated)
- Lines 401: MCP initialization
- Lines 895-941: `submit_gpt_image` tool (simplified)
- Lines 943-984: `submit_gemini_image` tool (simplified)
- Lines 987-1037: `wait_tasks` tool (simplified)
- Lines 1040-1101: `list_tasks` tool (simplified)
- Lines 1104-1115: `main()` entry point

### Phase 6: Testing & Validation

#### Task 6.1: Verify imports
- Ensure all circular dependencies are resolved
- Check that all modules can be imported independently

#### Task 6.2: Integration testing
- Test each MCP tool with sample inputs
- Verify task management still works
- Confirm both GPT and Gemini generation work

#### Task 6.3: Clean up
- Remove unused imports
- Add proper type hints where missing
- Ensure consistent error handling

## Implementation Order

1. **Day 1**: Phase 1 (Extract utilities) - Low risk, no breaking changes
2. **Day 2**: Phase 2 (Extract task management) - Medium risk, test thoroughly
3. **Day 3**: Phase 3 & 4 (Extract clients) - Can be done in parallel
4. **Day 4**: Phase 5 (Refactor server) - High risk, careful testing needed
5. **Day 5**: Phase 6 (Testing & validation) - Critical for stability

## Success Metrics

- [ ] `server.py` reduced from 1116 to ~200 lines
- [ ] Each module under 400 lines
- [ ] All existing functionality preserved
- [ ] No circular dependencies
- [ ] Each module can be unit tested independently
- [ ] Clear separation between GPT and Gemini logic
- [ ] Shared utilities properly reused

## Rollback Plan

If issues arise during refactoring:
1. Keep original `server.py` as `server_backup.py`
2. Test each phase independently before moving to next
3. Use git branches for each phase
4. Maintain backward compatibility during transition

## Future Enhancements (Post-Refactor)

Once refactoring is complete, the modular structure enables:
- Easy addition of new image providers (DALL-E 3, Midjourney, etc.)
- Independent scaling of task management
- Provider-specific optimizations
- Better unit test coverage
- Easier debugging and maintenance