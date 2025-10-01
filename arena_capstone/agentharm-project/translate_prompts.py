# translate_prompts.py

"""
Translates structured JSON behavior files to multiple languages using the DeepL API.

Outputs a separate JSON file for each language.

Usage:
    python translate_prompts.py -i prompt_data/validation_prompts_en.json -o output.json --langs DE FR
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
import deepl
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple

# --- FILE PATH CONSTANTS ---

PROMPT_DIR = Path('prompt_data')

# Full list of languages (including EN for source copy)
LANGUAGE_MAP = {
    'EN-GB': 'en',  # English (for DeepL source/target, outputs as 'en')
    'ES': 'es',     # Spanish
    'FR': 'fr',     # French
    'DE': 'de',     # German
    'BG': 'bg',     # Bulgarian
    'ZH': 'zh',     # Chinese (simplified)
    'ID': 'id',     # Indonesian
    'FI': 'fi',     # Finnish
    'LT': 'lt',     # Lithuanian
}

LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'bg': 'Bulgarian',
    'zh': 'Chinese',
    'id': 'Indonesian',
    'fi': 'Finnish',
    'lt': 'Lithuanian',
}

CODE_SPAN_RE = re.compile(r"`([^`]+)`")  # markdown inline code

# --- ARGPARSE SETUP ---
def parse_args():
    parser = argparse.ArgumentParser(description="Translate JSON prompt files using DeepL.")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file (English source).")
    parser.add_argument("-o", "--output", help="Path to output JSON file. If not given, defaults to <input>_<lang>.json")
    parser.add_argument("--langs", nargs="+", required=True, help="List of target languages (e.g. DE FR ES).")
    return parser.parse_args()

def main():
    args = parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load your DeepL API key
    auth_key = os.getenv("DEEPL_AUTH_KEY")
    if not auth_key:
        raise RuntimeError("DEEPL_AUTH_KEY environment variable is not set.")

    translator = deepl.Translator(auth_key)

    for lang in args.langs:
        lang = lang.upper()
        if lang not in LANGUAGE_MAP:
            raise ValueError(f"Unsupported language code: {lang}")

        # Deep copy so we don’t overwrite original
        translated_data = deepcopy(data)

        # TODO: implement your translation logic here
        # Example: translated_data["prompt"] = translator.translate_text(...)

        # Build output filename
        if args.output:
            output_file = Path(args.output)
            if len(args.langs) > 1:
                # If multiple langs and single -o given, append lang code
                output_file = output_file.with_stem(f"{output_file.stem}_{LANGUAGE_MAP[lang]}")
        else:
            output_file = input_file.with_stem(f"{input_file.stem}_{LANGUAGE_MAP[lang]}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {LANGUAGE_NAMES[LANGUAGE_MAP[lang]]} translation to {output_file}")

if __name__ == "__main__":
    main()
