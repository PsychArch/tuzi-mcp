# Tuzi MCP - GPT Image Generation Server

[中文文档](README_CN.md)

An MCP (Model Context Protocol) server for asynchronous image generation using the Tu-zi API.

## MCP Configuration

```json
{
  "mcpServers": {
    "tuzi-mcp": {
      "command": "uvx",
      "args": [ "tuzi-mcp"],
      "env": {"TUZI_API_KEY": "your tuzi key"}
    }
  }
}
```

## MCP Tools

#### `submit_gpt_image`
Submit async image generation task.
- `prompt` (string): Image description with aspect ratio (1:1, 3:2, or 2:3)
- `model` (string): `gpt-4o-image-async` or `gpt-4o-image-vip-async`
- `output_path` (string): Absolute save path
- `reference_image_path` (string, optional): Reference image path

#### `wait_tasks`
Wait for all submitted tasks to complete.
- `timeout_seconds` (integer): Max wait time (30-1200)

#### `list_tasks`
List all tasks with status.
- `status_filter` (string, optional): Filter by `pending`/`running`/`completed`/`failed`
