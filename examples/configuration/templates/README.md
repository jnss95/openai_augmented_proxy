# Templates Directory

This directory contains reusable Jinja2 templates that can be included in model system prompts.

## Structure

```
templates/
├── base-assistant.j2        # Base template for assistant prompts
├── common/                  # Common reusable snippets
│   └── tool-instructions.j2 # Tool usage instructions
└── context/                 # Context injection templates
    └── time.j2              # Time context template
```

## Testing Templates

Use the `/admin/template/eval` endpoint to test templates before deploying:

```bash
# Simple template with variable
http POST localhost:8000/admin/template/eval \
  template='Hello {{ name }}!' \
  variables:='{"name": "World"}'

# Template with tool call
http POST localhost:8000/admin/template/eval \
  template='The time is: {{ tool("get_current_time", timezone="UTC") }}'

# Template using variables.yaml
http POST localhost:8000/admin/template/eval \
  template='Welcome to {{ app_name }} v{{ version }}'

# Template with conditionals
http POST localhost:8000/admin/template/eval \
  template='{% if debug %}Debug mode{% else %}Production{% endif %}' \
  variables:='{"debug": true}'

# View all available variables
http GET localhost:8000/admin/template/variables
```

## Usage

### Including Templates

In your model's `system_prompt`, use Jinja2's include directive:

```yaml
system_prompt: |
  {% include "base-assistant.j2" %}
  
  Your specific instructions here...
```

### Calling Tools

Use the `tool()` function to call tools directly in templates:

```jinja
The current time is: {{ tool("get_current_time", timezone="UTC") }}
```

### Using Variables

Variables from `variables.yaml` are available directly:

```jinja
Welcome to {{ app_name }}!
Max tokens: {{ settings.max_tokens }}
```

### Conditionals and Loops

Full Jinja2 syntax is supported:

```jinja
{% if features.debug_mode %}
Debug mode is enabled.
{% endif %}

{% for item in items %}
- {{ item }}
{% endfor %}
```

### Filters

Standard Jinja2 filters work:

```jinja
{{ app_name | upper }}
{{ description | default("No description") }}
```

## Creating New Templates

1. Create a `.j2` file in the appropriate subdirectory
2. Use standard Jinja2 syntax
3. Reference it with `{% include "path/to/template.j2" %}`

## Best Practices

- Keep templates focused on a single purpose
- Use meaningful names and organize in subdirectories
- Add comments with `{# comment #}` syntax
- Use `| default("fallback")` for optional variables
- Test templates before deploying
