"""Agent-native utilities for the xorq CLI."""

from .prime import (
    agent_prime_command,
    render_prime_context,
)
from .prompts import (
    CORE_PROMPT_TIER,
    PROMPT_TIER_DESCRIPTIONS,
    iter_prompt_specs,
    list_prompt_names,
    load_prompt_text,
)
from .templates import (
    get_template,
    iter_templates,
    list_template_names,
    scaffold_template,
)


__all__ = [
    "CORE_PROMPT_TIER",
    "PROMPT_TIER_DESCRIPTIONS",
    "agent_prime_command",
    "iter_prompt_specs",
    "list_prompt_names",
    "load_prompt_text",
    "get_template",
    "iter_templates",
    "list_template_names",
    "render_prime_context",
    "scaffold_template",
]
