# Tuzi MCP - GPT 图像生成服务器

[English Documentation](README.md)

基于 Tu-zi API 的异步图像生成 MCP（模型上下文协议）服务器。

## MCP 配置

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

## MCP 工具

#### `submit_gpt_image`
提交异步图像生成任务。
- `prompt` (字符串): 图像描述，需包含宽高比 (1:1, 3:2, 或 2:3)
- `model` (字符串): `gpt-4o-image-async` 或 `gpt-4o-image-vip-async`
- `output_path` (字符串): 绝对保存路径
- `reference_image_path` (字符串，可选): 参考图像路径

#### `wait_tasks`
等待所有已提交任务完成。
- `timeout_seconds` (整数): 最大等待时间 (30-1200)

#### `list_tasks`
列出所有任务及状态。
- `status_filter` (字符串，可选): 按状态筛选 `pending`/`running`/`completed`/`failed`