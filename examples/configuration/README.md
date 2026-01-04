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
├── skills/           # Skills (injected into system prompts)
│   ├── code-review.md
│   └── writing-style.md
├── templates/        # Jinja2 templates for system prompts
│   ├── base-assistant.j2
│   ├── common/
│   └── context/
└── variables.yaml    # Variables for use in templates
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

### Templates (`templates/*.j2`)

Jinja2 templates for reusable system prompt components:
- Include templates with `{% include "template.j2" %}`
- Call tools with `{{ tool("tool_name", arg="value") }}`
- Use variables with `{{ variable_name }}`
- Full Jinja2 features: conditionals, loops, filters, etc.

See [templates/README.md](templates/README.md) for detailed documentation.

### Variables (`variables.yaml`)

YAML file containing variables available in all templates:

```yaml
app_name: "My Assistant"
version: "1.0"
settings:
  max_tokens: 4096
  debug_mode: false
```

Use in templates: `{{ app_name }}` or `{{ settings.max_tokens }}`
