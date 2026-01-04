# OpenAI API Proxy

A Python proxy server that augments OpenAI-compatible APIs with additional tools, MCP server integration, and Claude Code-style skills. The proxy acts as a transparent pass-through for all upstream models while providing augmented models with custom capabilities.

## Features

- **Pass-Through Mode**: All upstream models are available directly - just use their original name
- **Augmented Models**: Custom models (prefixed with `augmented/`) with tools, MCP servers, and skills
- **Tool Augmentation**: Add custom tools to augmented model configurations
- **MCP Server Integration**: Connect to Model Context Protocol servers (stdio, SSE, Streamable HTTP)
- **Per-Server Tool Filtering**: Whitelist/blacklist tools from each MCP server
- **System Prompt Templates**: Call tools in system prompts using `{{tool_name(args)}}` syntax
- **Skills Support**: Claude Code-style skills that inject capabilities into system prompts
- **Streaming Tool Status**: Real-time status updates during tool execution (similar to "thinking")
- **Automatic Tool Handling**: Proxy executes its own tools transparently
- **Client Tool Passthrough**: Tools from clients are preserved and passed through
- **Chat History Logging**: Full conversation logs saved to `logs/` folder
- **Hot Reload**: Reload configurations without restarting

## Installation

```bash
# Clone the repository
cd openai-proxy

# Install dependencies with uv
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASE_URL` | Upstream OpenAI-compatible API URL | `https://openrouter.ai/api/v1` |
| `API_KEY` | API key for the upstream API | (required) |
| `HOST` | Host to bind the proxy server | `0.0.0.0` |
| `PORT` | Port to bind the proxy server | `8000` |
| `MODELS_CONFIG_DIR` | Directory containing model configs | `conf/models` |
| `REQUEST_TIMEOUT` | Timeout for upstream requests (seconds) | `300` |

### Model Configuration

Model configurations are YAML files in the `conf/models/` directory. Each file defines an augmented model exposed by the proxy.

```yaml
# conf/models/homeassistant.yml

# Required: The name to expose this model as (use augmented/ prefix)
name: augmented/homeassistant

# Optional: The actual upstream model to use (defaults to 'name')
upstream_model: google/gemini-3-flash-preview

# Optional: System prompt prepended to all conversations
system_prompt: |
  You are a helpful assistant for controlling Home Assistant.

# Optional: Additional tools to add to requests
tools:
  - type: function
    function:
      name: get_current_time
      description: Get the current date and time
      parameters:
        type: object
        properties:
          timezone:
            type: string
            description: The timezone (e.g., 'UTC', 'America/New_York')
        required: []

# MCP servers - simple list (all tools) or with filtering
mcp_servers:
  homeassistant:
    blacklist:
      - "HassDebug*"
  music_assistant:
    whitelist:
      - "search_*"
      - "play_*"

# Or simple list format (all tools enabled):
# mcp_servers:
#   - homeassistant
#   - music_assistant

# Optional: Skills to include (names from conf/skills/)
skills:
  - python-expert
  - code-review

# Optional: Whether to include global skills (default: true)
include_global_skills: true

# Optional: Model metadata
description: "Home Assistant control assistant"
owned_by: openai-proxy
```

### MCP Server Configuration

MCP (Model Context Protocol) servers are configured in `conf/mcp/servers.yaml`:

```yaml
servers:
  # Stdio-based MCP server (runs a command)
  perplexity:
    type: stdio
    command: npx
    args:
      - "-y"
      - "perplexity-mcp"
    env:
      PERPLEXITY_API_KEY: "${PERPLEXITY_API_KEY}"
    description: "Perplexity search"

  # Streamable HTTP server (used by Home Assistant)
  homeassistant:
    type: streamablehttp
    url: "https://your-ha-instance.com/api/mcp"
    headers:
      Authorization: "Bearer ${HA_TOKEN}"
    description: "Home Assistant MCP"

  # SSE-based MCP server
  remote-tools:
    type: sse
    url: "http://localhost:3001/sse"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    description: "Remote tool server"
```

**Server Types:**
- `stdio`: Runs a local command, communicates via stdin/stdout
- `sse`: Server-Sent Events over HTTP
- `streamablehttp`: Streamable HTTP (MCP 2025-06 spec, used by Home Assistant)

MCP tools are automatically prefixed with `mcp_{server_name}_` to avoid conflicts.

### Tool Filtering

Filter tools per MCP server using whitelist/blacklist with glob patterns:

```yaml
mcp_servers:
  homeassistant:
    # Only allow these tools
    whitelist:
      - "HassControl*"
      - "HassGet*"
    # Exclude these (applied after whitelist)
    blacklist:
      - "HassGetDebug"
  
  music_assistant:
    # Exclude all debug tools
    blacklist:
      - "*debug*"
      - "*test*"
```

### System Prompt Templates

You can call tools directly in the system prompt using template syntax. Templates are processed at request time and replaced with the tool's result:

```yaml
system_prompt: |
  You are a helpful assistant.
  
  The current time is: {{get_current_time(timezone="UTC")}}
  
  Available devices:
  {{mcp_homeassistant_list_entities(domain="light")}}
```

**Template Syntax:**
- `{{tool_name()}}` - Call a tool with no arguments
- `{{tool_name(arg="value")}}` - Call with string argument
- `{{tool_name(count=10)}}` - Call with integer argument
- `{{tool_name(enabled=true)}}` - Call with boolean argument

**Supported Value Types:**
- Strings: `"value"` or `'value'`
- Integers: `123`
- Floats: `12.5`
- Booleans: `true` or `false`
- Null: `null`

**Examples:**

```yaml
# Built-in tool
system_prompt: |
  Current time: {{get_current_time()}}

# MCP tool (prefixed with mcp_{server}_)
system_prompt: |
  Weather forecast: {{mcp_weather_get_forecast(city="Berlin")}}

# Multiple templates
system_prompt: |
  Time: {{get_current_time(timezone="Europe/Berlin")}}
  Calculator test: {{calculator(expression="2+2")}}
```

Templates are processed concurrently for better performance.

### Skills Configuration

Skills are markdown or YAML files in `conf/skills/` that provide instructions and capabilities to models.

```
conf/skills/
├── code-review.md       # Global skill available to all models
├── python-expert.md     # Global skill
└── augmented/           # Model-specific skills
    └── homeassistant/
        └── custom-skill.md
```

## Usage

### Starting the Server

```bash
# Run with uvicorn
uv run uvicorn openai_proxy.app:app --host 0.0.0.0 --port 8000

# With auto-reload for development
uv run uvicorn openai_proxy.app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List all models (upstream + augmented) |
| `/v1/models/{model_id}` | GET | Get a specific model |
| `/v1/chat/completions` | POST | Create a chat completion |
| `/admin/reload` | POST | Reload all configurations |
| `/health` | GET | Health check |

### Two Operating Modes

**1. Pass-Through Mode** - Any model not in `conf/models/`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**2. Augmented Mode** - Models defined in `conf/models/`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "augmented/homeassistant",
    "messages": [{"role": "user", "content": "Turn on the living room lights"}]
  }'
```

### Streaming with Tool Status

When streaming with augmented models, the proxy sends status updates during tool execution:

```json
{
  "proxy_status": {
    "type": "tool_execution",
    "status": "executing_tool",
    "tool_name": "mcp_homeassistant_HassLightSet"
  }
}
```

Status values:
- `calling_llm` - Sending request to LLM
- `executing_tool` - Running a tool
- `tool_completed` - Tool finished
- `processing_results` - Processing tool results

### Example Client Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Proxy uses its own API key
)

# List all available models
models = client.models.list()
print(f"Total models: {len(models.data)}")

# Use upstream model directly (pass-through)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Use augmented model with tools
response = client.chat.completions.create(
    model="augmented/homeassistant",
    messages=[{"role": "user", "content": "What's the temperature in the bedroom?"}],
)
```

## Adding Custom Tools

### 1. Define the Tool in Model Config

```yaml
tools:
  - type: function
    function:
      name: my_custom_tool
      description: Description of what the tool does
      parameters:
        type: object
        properties:
          param1:
            type: string
            description: Description of param1
        required:
          - param1
```

### 2. Implement the Tool Handler

Create a handler in `src/openai_proxy/handlers.py`:

```python
class MyCustomToolHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "my_custom_tool"

    async def execute(self, arguments: dict[str, Any]) -> str:
        param1 = arguments.get("param1", "")
        result = {"status": "success", "data": param1}
        return json.dumps(result)

# Register the handler
register_handler(MyCustomToolHandler())
```

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Client    │────▶│   Proxy Server  │────▶│ Upstream API │
│             │◀────│                 │◀────│ (OpenRouter) │
└─────────────┘     └─────────────────┘     └──────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   ┌───────────┐    ┌─────────────┐    ┌────────────┐
   │   Tools   │    │ MCP Servers │    │   Skills   │
   │ (Handler) │    │(stdio/http) │    │ (Markdown) │
   └───────────┘    └─────────────┘    └────────────┘
```

**Flow:**
1. Client sends chat completion request
2. If model starts with `augmented/`:
   - Load model config, tools, MCP servers, skills
   - Merge tools, inject system prompt
   - Handle tool calls automatically
3. Otherwise: pass-through directly to upstream
4. Return response to client

## Project Structure

```
openai-proxy/
├── conf/
│   ├── models/              # Augmented model configurations
│   │   ├── homeassistant.yml
│   │   └── super-assistant.yaml
│   ├── mcp/                 # MCP server configurations
│   │   └── servers.yaml
│   └── skills/              # Skills (Claude Code-style)
│       ├── code-review.md
│       └── python-expert.md
├── logs/                    # Chat history logs (JSONL)
├── src/openai_proxy/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── app.py               # FastAPI application
│   ├── chat_logger.py       # Chat history logging
│   ├── client.py            # Upstream API client
│   ├── config.py            # Settings
│   ├── handlers.py          # Built-in tool handlers
│   ├── mcp_client.py        # MCP client integration
│   ├── models.py            # Model config loader
│   ├── proxy.py             # Core proxy logic
│   ├── schemas.py           # OpenAI-compatible schemas
│   └── skills.py            # Skills loader
├── .env.example
├── pyproject.toml
└── README.md
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `get_current_time` | Returns the current date/time in any timezone |
| `calculator` | Evaluates mathematical expressions safely |

## Chat Logging

All conversations are logged to `logs/` in JSONL format:
- Filename: `{model}_{conversation_id}.jsonl`
- Includes: requests, responses, tool calls with results
- Useful for debugging and analysis

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## License

MIT
