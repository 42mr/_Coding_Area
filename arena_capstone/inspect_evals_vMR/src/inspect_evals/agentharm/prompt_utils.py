# prompt_utils.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Tuple

# ---- Language map & active language -----------------------------------------

LANGUAGE_MAP: Dict[str, str] = {
    "EN-GB": "en",  # English
    "ES": "es",     # Spanish
    "FR": "fr",     # French
    "DE": "de",     # German
    "BG": "bg",     # Bulgarian
    "ZH": "zh",     # Chinese (simplified)
    "ID": "id",     # Indonesian
    "FI": "fi",     # Finnish
    "LT": "lt",     # Lithuanian
}

_DEFAULT_LANG_LONG = "EN-GB"
_DEFAULT_LANG_SHORT = LANGUAGE_MAP[_DEFAULT_LANG_LONG]

# Global, process-local “active language” (long form) used when callers omit `language`
_ACTIVE_LANGUAGE_LONG = _DEFAULT_LANG_LONG

def set_active_language(lang: str) -> None:
    """
    Set global active language (long key like 'EN-GB').
    Accepts either long ('EN-GB') or short ('en') codes.
    """
    global _ACTIVE_LANGUAGE_LONG
    long_k, _ = _normalize_lang_key(lang)
    _ACTIVE_LANGUAGE_LONG = long_k

def get_active_language() -> str:
    """Return the active language long key (e.g., 'EN-GB')."""
    return _ACTIVE_LANGUAGE_LONG

def _normalize_lang_key(lang: str) -> Tuple[str, str]:
    """
    Return (long, short) pair for a provided language token.
    Accepts either 'EN-GB' or 'en', etc. Raises on unsupported.
    """
    if not lang:
        return _DEFAULT_LANG_LONG, _DEFAULT_LANG_SHORT
    lang = lang.strip()
    if lang in LANGUAGE_MAP:
        return lang, LANGUAGE_MAP[lang]
    # maybe a short code was provided
    for long_k, short_k in LANGUAGE_MAP.items():
        if lang.lower() == short_k.lower():
            return long_k, short_k
    raise ValueError(f"Unsupported language: {lang}")

# ---- File locations ----------------------------------------------------------

PROMPT_DATA_DIR = os.path.expanduser(
    "~/arena_capstone/arena_capstone/agentharm-project/prompt_data"
)

SYSTEM_PROMPTS_FILE = os.path.join(PROMPT_DATA_DIR, "system_prompts.json")
PROMPT_TECHNIQUES_FILE = os.path.join(PROMPT_DATA_DIR, "prompt_techniques.json")
JAILBREAK_TEMPLATES_FILE = os.path.join(PROMPT_DATA_DIR, "jailbreak_templates.json")

# ---- Helpers ----------------------------------------------------------------

def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pick_lang(obj: Dict[str, Any], lang_long: str, lang_short: str, file_label: str) -> Any:
    """
    Given an object like { "EN-GB": "...", "de": "..." } choose the best match.
    Fallbacks: EN-GB, then 'en'.
    """
    if lang_long in obj:
        return obj[lang_long]
    if lang_short in obj:
        return obj[lang_short]
    if _DEFAULT_LANG_LONG in obj:
        return obj[_DEFAULT_LANG_LONG]
    if _DEFAULT_LANG_SHORT in obj:
        return obj[_DEFAULT_LANG_SHORT]
    raise KeyError(f"No content for {lang_long}/{lang_short} and no EN fallback in {file_label}")

# ---- Cached payload loaders --------------------------------------------------

@lru_cache(maxsize=1)
def _payload_system_prompts() -> Dict[str, Dict[str, str]]:
    data = _load_json(SYSTEM_PROMPTS_FILE)
    if not isinstance(data, dict):
        raise ValueError("system_prompts.json must be {template: {lang: text}}")
    return data

@lru_cache(maxsize=1)
def _payload_prompt_techniques() -> Dict[str, Dict[str, str]]:
    data = _load_json(PROMPT_TECHNIQUES_FILE)
    if not isinstance(data, dict):
        raise ValueError("prompt_techniques.json must be {lang: {tech: text}}")
    return data

@lru_cache(maxsize=1)
def _payload_jailbreak_templates() -> Dict[str, Dict[str, str]]:
    data = _load_json(JAILBREAK_TEMPLATES_FILE)
    if not isinstance(data, dict):
        raise ValueError("jailbreak_templates.json must be {name: {lang: text}}")
    return data

# ---- Public getters ----------------------------------------------------------

def get_system_prompt_template(template_name: str, language: str) -> str:
    """
    Look up a system prompt by name, localize by language (long/short accepted).
    """
    lang_long, lang_short = _normalize_lang_key(language)
    payload = _payload_system_prompts()
    if template_name not in payload:
        raise ValueError(f"Unknown system prompt template: {template_name}")
    text = _pick_lang(payload[template_name], lang_long, lang_short, "system_prompts.json")
    if not isinstance(text, str):
        raise TypeError("System prompt must be a string")
    return text

def get_prompt_technique_text(technique_key: str, language: str) -> str:
    """
    Look up a technique snippet by key (standard/cot/react/refusal) localized.
    """
    lang_long, lang_short = _normalize_lang_key(language)
    payload = _payload_prompt_techniques()
    lang_block = _pick_lang(payload, lang_long, lang_short, "prompt_techniques.json")
    if not isinstance(lang_block, dict):
        raise TypeError("Language block must be an object")
    if technique_key not in lang_block:
        raise ValueError(f"Technique '{technique_key}' not found for {lang_long}")
    text = lang_block[technique_key]
    if not isinstance(text, str):
        raise TypeError("Technique text must be a string")
    return text

def get_jailbreak_template_text(template_name: str, language: str) -> str:
    """
    Look up a jailbreak template by name, localized.
    """
    lang_long, lang_short = _normalize_lang_key(language)
    payload = _payload_jailbreak_templates()
    if template_name not in payload:
        raise ValueError(f"Unknown jailbreak template: {template_name}")
    text = _pick_lang(payload[template_name], lang_long, lang_short, "jailbreak_templates.json")
    if not isinstance(text, str):
        raise TypeError("Jailbreak template must be a string")
    return text
