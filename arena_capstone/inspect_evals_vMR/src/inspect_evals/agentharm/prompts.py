# prompts.py
from __future__ import annotations

from typing import Optional

from .prompt_utils import (
    get_system_prompt_template,
    get_prompt_technique_text,
    get_jailbreak_template_text,
    get_active_language,  # read active language if caller omits one
)

def get_system_prompt(
    system_prompt_template: str,
    prompt_technique: str,
    language: Optional[str] = None,
) -> str:
    """
    Compose the system prompt: system template + technique snippet.
    - system_prompt_template: e.g., "default", "llama-3.1"
    - prompt_technique: one of {"standard","cot","react","refusal"}
    - language: optional ("EN-GB", "DE" or short "en","de"); if None, uses active language.
    """
    lang = language or get_active_language()
    system_prompt = get_system_prompt_template(system_prompt_template, lang)
    technique_text = get_prompt_technique_text(prompt_technique, lang)
    return system_prompt + technique_text

def get_jailbreak_template(
    prompt_template_name: Optional[str],
    language: Optional[str] = None,
) -> str:
    """
    Build the final user prompt template.

    Rules:
    - If prompt_template_name is None / "empty" / "plain" / "none" -> return "{prompt}" (no JSON lookup).
    - Otherwise, load the named template from jailbreak_templates.json (localized) and prepend it.
    """
    if not prompt_template_name or str(prompt_template_name).lower() in {"empty", "plain", "none"}:
        return "{prompt}"

    lang = language or get_active_language()
    prompt_template = get_jailbreak_template_text(prompt_template_name, lang)
    return (prompt_template + "\n\n{prompt}").strip()
