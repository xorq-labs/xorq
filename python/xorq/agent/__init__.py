"""Agent-native utilities for the xorq CLI."""

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
    "iter_prompt_specs",
    "list_prompt_names",
    "load_prompt_text",
    "get_template",
    "iter_templates",
    "list_template_names",
    "scaffold_template",
]
