# OpenAI API Proxy - Copilot Instructions

## Architecture Overview

This is a **FastAPI proxy** that sits between clients and upstream OpenAI-compatible APIs (OpenRouter, OpenAI, etc.). It operates in two modes:

1. **Pass-through**: Any model name → forwarded directly to upstream
2. **Augmented**: Models prefixed with `augmented/` → enhanced with tools, MCP servers, and skills

```
Client → app.py → [augmented?] → proxy.py (tool loop) → client.py → Upstream API
                       ↓
              mcp_client.py (MCP tools)
              handlers.py (built-in tools)
              skills.py (system prompt injection)
```

## Key Files & Responsibilities

| File | Purpose |
|------|---------|
| `app.py` | FastAPI routes, lifespan management, pass-through vs augmented routing |
| `proxy.py` | Core tool-calling loop, merges tools, handles streaming with status events |
| `mcp_client.py` | MCP server connections (stdio/sse/streamablehttp transports) |
| `handlers.py` | Built-in tool implementations (subclass `ToolHandler`, register with `register_handler()`) |
| `models.py` | `ModelConfig` pydantic model, `ModelRegistry` for loading YAML configs |
| `config.py` | Settings via pydantic-settings, env vars + optional `.env` file |
| `skills.py` | Loads markdown skills, injects into system prompts |

## Development Commands

```bash
# Run with hot reload (debug mode)
uv run openai-proxy --debug --config ./conf

# Run production
uv run openai-proxy --config ~/.config/openai_proxy

# Format & lint
uv run ruff format . && uv run ruff check .
```

## Conventions

### Adding a New Built-in Tool

1. Create handler in `handlers.py`:
```python
class MyToolHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "my_tool"  # Must match tool name in model config
    
    async def execute(self, arguments: dict[str, Any]) -> str:
        return json.dumps({"result": "..."})

register_handler(MyToolHandler())
```

2. Add tool definition to model YAML config under `tools:`

### MCP Tools Naming

MCP tools are automatically prefixed: `mcp_{server_name}_{tool_name}`
- Filter patterns in model configs apply to the **original** tool name (before prefix)

### Configuration Priority

1. CLI flags (`--config`, `--host`, `--port`, `--debug`)
2. Environment variables (`OPENAI_PROXY_CONFIG_DIR`, `BASE_URL`, `API_KEY`)
3. `.env` file in current working directory (only if exists)
4. Defaults (`~/.config/openai_proxy`)

### Streaming Status Events

During tool execution, the proxy emits `proxy_status` events:
```json
{"proxy_status": {"type": "tool_execution", "status": "executing_tool", "tool_name": "..."}}
```

## Testing Locally

```bash
# Start proxy
uv run openai-proxy --debug --config ./conf

# Test pass-through
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}'

# Test augmented model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "augmented/assistant", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Config Directory Structure

```
~/.config/openai_proxy/   # or --config path
├── models/               # YAML files defining augmented models
├── mcp/servers.yaml      # MCP server connections
├── skills/               # Markdown files injected into system prompts
└── logs/                 # Chat history (JSONL) and debug logs
```

See `examples/configuration/` for documented example configs.
