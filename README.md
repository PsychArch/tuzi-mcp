# Tuzi MCP - GPT/Gemini/Seedream Image Generation Server

[中文文档](README_CN.md)

An MCP (Model Context Protocol) server for asynchronous image generation using the Tu-zi API, supporting GPT-4o, Gemini-2.5-flash, and Seedream image generation.

## MCP Configuration

```json
{
  "mcpServers": {
    "tuzi-mcp": {
      "command": "uvx",
      "args": [ "tuzi-mcp"],
      "env": {
        "TUZI_API_KEY": "your tuzi key",
        "TUZI_URL_BASE": "https://api.tu-zi.com"
      }
    }
  }
}
```

### Environment Variables

- `TUZI_API_KEY` (required): Your Tu-zi API key
- `TUZI_URL_BASE` (optional): API base URL (default: `https://api.tu-zi.com`, alternatives: `apius.tu-zi.com`, `apicdn.tu-zi.com`, `api.sydney-ai.com`)

## MCP Tools

#### `submit_gpt_image`
Submit async GPT-4o image generation task.
- `prompt` (string): Image description with aspect ratio (1:1, 3:2, or 2:3)
- `output_path` (string): Absolute save path
- `model` (string, optional): `gpt-4o-image-async` or `gpt-4o-image-vip-async`, defaults to `gpt-4o-image-async`
- `reference_image_paths` (string, optional): Comma-separated reference image paths (supports PNG, JPEG, WebP, GIF, BMP)

#### `submit_gemini_image`
Submit Gemini image generation task.
- `prompt` (string): Image description with aspect ratio (1:1, 3:2, 2:3, 16:9, 9:16, 4:5)
- `output_path` (string): Absolute save path
- `reference_image_paths` (string, optional): Comma-separated reference image paths (supports PNG, JPEG, WebP, GIF, BMP)
- `hd` (boolean, optional): HD quality, only enable when user explicitly requests, HD mode only supports .webp output
- `vip` (boolean, optional): VIP model, only use when normal model fails or user explicitly requests it

#### `submit_seedream_image`
Submit Seedream image generation/editing task. Suitable for Chinese-context tasks.
- `prompt` (string): Prompt for image generation or editing
- `output_path` (string): Absolute save path
- `size` (string, optional): Image size, supports 1024x1024, 2048x2048, 4096x4096, 2560x1440 (16:9), 1440x2560 (9:16), 2304x1728 (4:3), 1728x2304 (3:4), 2496x1664 (3:2), 1664x2496 (2:3), 3024x1296 (21:9), defaults to 1024x1024
- `quality` (string, optional): Image quality `standard` or `high`, defaults to `high`
- `n` (integer, optional): Number of images to generate (1-8), defaults to 1
- `reference_image_paths` (string, optional): Comma-separated reference image paths.

#### `wait_tasks`
Wait for all submitted tasks to complete.
- `timeout_seconds` (integer, optional): Max wait time (30-1200 seconds), defaults to 600

#### `list_tasks`
List all tasks with status.
- `status_filter` (string, optional): Filter by `pending`/`running`/`completed`/`failed`
