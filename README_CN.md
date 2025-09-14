# Tuzi MCP - GPT/Gemini/即梦 图像生成服务器

[English Documentation](README.md)

基于 Tu-zi API 的异步图像生成 MCP（模型上下文协议）服务器，支持 GPT-4o、Gemini-2.5-flash 和即梦（Seedream）图像生成。

## MCP 配置

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

### 环境变量

- `TUZI_API_KEY` (必需): 您的 Tu-zi API 密钥
- `TUZI_URL_BASE` (可选): API 基础 URL (默认: `https://api.tu-zi.com`，其他: `apius.tu-zi.com`, `apicdn.tu-zi.com`, `api.sydney-ai.com`)

## MCP 工具

#### `submit_gpt_image`
提交异步 GPT-4o 图像生成任务。
- `prompt` (字符串): 图像描述，需包含宽高比 (1:1, 3:2, 或 2:3)
- `output_path` (字符串): 绝对保存路径
- `model` (字符串，可选): `gpt-4o-image-async` 或 `gpt-4o-image-vip-async`，默认 `gpt-4o-image-async`
- `reference_image_paths` (字符串，可选): 逗号分隔的参考图像路径 (支持 PNG, JPEG, WebP, GIF, BMP)

#### `submit_gemini_image`
提交 Gemini 图像生成任务。
- `prompt` (字符串): 图像描述，需包含宽高比 (1:1, 3:2, 2:3, 16:9, 9:16, 4:5)
- `output_path` (字符串): 绝对保存路径
- `reference_image_paths` (字符串，可选): 逗号分隔的参考图像路径 (支持 PNG, JPEG, WebP, GIF, BMP)
- `hd` (布尔值，可选): HD 质量，仅在用户明确要求时启用，HD 模式仅支持 .webp 输出

#### `submit_seedream_image`
提交即梦（Seedream）图像生成/编辑任务，适合中文语境任务。
- `prompt` (字符串): 图像生成或编辑提示
- `output_path` (字符串): 绝对保存路径
- `size` (字符串，可选): 图像尺寸，支持 1024x1024, 2048x2048, 4096x4096, 2560x1440 (16:9), 1440x2560 (9:16), 2304x1728 (4:3), 1728x2304 (3:4), 2496x1664 (3:2), 1664x2496 (2:3), 3024x1296 (21:9)，默认 1024x1024
- `quality` (字符串，可选): 图像质量 `standard` 或 `high`，默认 `high`
- `n` (整数，可选): 生成图像数量 (1-8)，默认 1
- `reference_image_paths` (字符串，可选): 逗号分隔的参考图像路径

#### `wait_tasks`
等待所有已提交任务完成。
- `timeout_seconds` (整数，可选): 最大等待时间 (30-1200 秒)，默认 600

#### `list_tasks`
列出所有任务及状态。
- `status_filter` (字符串，可选): 按状态筛选 `pending`/`running`/`completed`/`failed`