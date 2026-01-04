# Example Configuration

This directory contains example configurations for the OpenAI API Proxy.

## Directory Structure

```
configuration/
├── models/           # Augmented model configurations
│   ├── assistant.yaml
│   └── researcher.yaml
├── mcp/              # MCP server configurations
│   └── servers.yaml
└── skills/           # Skills (injected into system prompts)
    ├── code-review.md
    └── writing-style.md
```

## Usage

Copy this directory to `~/.config/openai_proxy` or use the `--config` flag:

```bash
# Use default location
cp -r examples/configuration/* ~/.config/openai_proxy/

# Or specify a custom path
openai-proxy --config ./examples/configuration
```

## Configuration Files

### Models (`models/*.yaml`)

Define augmented models that wrap upstream models with additional capabilities:
- Custom system prompts
- MCP server tools
- Skills injection
- Built-in tools

See [models/assistant.yaml](models/assistant.yaml) for a full example.

### MCP Servers (`mcp/servers.yaml`)

Configure Model Context Protocol servers that provide tools:
- **stdio**: Local commands (npx, python scripts)
- **sse**: Server-Sent Events endpoints
- **streamablehttp**: HTTP streaming (Home Assistant, etc.)

See [mcp/servers.yaml](mcp/servers.yaml) for examples of each type.

### Skills (`skills/*.md`)

Markdown files that inject instructions into the system prompt:
- Global skills: Available to all models
- Model-specific: Place in `skills/{model-name}/`

See [skills/](skills/) for examples.
