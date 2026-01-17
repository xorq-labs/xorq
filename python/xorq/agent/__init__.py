"""Agent-native utilities for the xorq CLI."""

from .prime import (
    agent_prime_command,
    render_prime_context,
)
from .templates import (
    get_template,
    iter_templates,
    list_template_names,
    scaffold_template,
)


__all__ = [
    "agent_prime_command",
    "get_template",
    "iter_templates",
    "list_template_names",
    "render_prime_context",
    "scaffold_template",
]
