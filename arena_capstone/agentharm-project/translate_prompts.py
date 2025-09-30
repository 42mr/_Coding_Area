# translate_prompts.py
"""
Translates structured JSON behavior files (like benign_behaviours_validation.json)
to multiple languages using the DeepL API. Outputs a separate JSON file for each language.

Usage: python translate_prompts.py -i prompt_data/benign_behaviors_validation.json --langs DE
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
# Define the directory where the source and output files reside relative to the script
PROMPT_DIR = Path('prompt_data')
SOURCE_FILE_NAME = 'benign_behaviors_validation.json' # Correct file name

# Construct the full paths (these are in the GLOBAL scope, resolving the NameError)
SOURCE_FILE = PROMPT_DIR / SOURCE_FILE_NAME
# ---------------------------

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

# Build a tuple of retryable exceptions that exist in this deepl version.
_retryables = []
for name in ("TooManyRequestsException", "ConnectionException", "NetworkException"):
    exc = getattr(deepl.exceptions, name, None)
    if exc:
        _retryables.append(exc)
RETRYABLE_EXC = tuple(_retryables) or (deepl.exceptions.DeepLException,)

# --- UTILITY FUNCTIONS ---

def get_translator() -> deepl.Translator:
    key = "1d33ba23-4eab-4245-ba14-d72f5f7f8d82:fx"
    if not key:
        # NOTE: For production, use os.getenv("DEEPL_API_KEY")
        raise SystemExit("DEEPL_API_KEY is not set.")
    return deepl.Translator(key)

def load_prompts(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Load the list of behaviors from the nested structure of the input JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}")
    
    # 1. Extract the 'behaviors' list and 'canary_guid'
    behaviors = data.get("behaviors")
    canary_guid = data.get("canary_guid", "")

    if not isinstance(behaviors, list):
        raise SystemExit(f"Input file {path} must contain a 'behaviors' key with a list value.")
    
    # 2. Integrity check on the extracted list
    for i, row in enumerate(behaviors):
        if not isinstance(row, dict) or "id" not in row or "prompt" not in row:
            # The row is malformed or missing required keys, raising the "Row X missing 'id' or 'prompt'" error
            raise SystemExit(f"Row {i} in 'behaviors' list is malformed (missing 'id' or 'prompt').")
            
    # Return the clean list and the canary ID
    return behaviors, canary_guid

def md_to_xml(text: str) -> str:
    # Wrap markdown inline code in <code>…</code> so DeepL can ignore it
    return CODE_SPAN_RE.sub(r"<code>\1</code>", text)

def xml_to_md(text: str) -> str:
    # Restore markdown after translation
    return (text
            .replace("<code>", "`")
            .replace("</code>", "`"))

def translate_batch(translator: deepl.Translator, texts: list[str], target: str,
                    formality: str | None = "default",
                    source: str | None = "EN",
                    max_retries: int = 5,
                    glossary: deepl.GlossaryInfo | None = None) -> list[str]:
    attempt = 0
    while True:
        try:
            res = translator.translate_text(
                texts,
                target_lang=target,
                source_lang=source,
                formality=formality,
                preserve_formatting=True,
                split_sentences="nonewlines",
                tag_handling="xml",
                ignore_tags=["code"],
                glossary=glossary,
            )
            return [r.text for r in res]
        # Catch retryable exceptions
        except RETRYABLE_EXC as e:
            if attempt >= max_retries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"   Retryable error ({e.__class__.__name__}). Retrying in {wait}s…")
            time.sleep(wait)
            attempt += 1
        except deepl.exceptions.DeepLException:
            # Non-retryable DeepL error
            raise

def maybe_make_glossary(translator: deepl.Translator, target_code: str, terms: set):
    """
    Optional: pin tool names to themselves. Creates a temporary glossary when supported.
    """
    if not terms:
        return None
    try:
        entries = {t: t for t in terms}
        return translator.create_glossary(
            name=f"tool-names-en-{target_code.lower()}",
            source_lang="EN",
            target_lang=target_code,
            entries=entries,
        )
    except deepl.exceptions.DeepLException:
        return None

# --- MAIN LOGIC ---

def main():
    ap = argparse.ArgumentParser(description="Translate structured JSON behavior prompts to multiple languages.")
    # Use the global SOURCE_FILE path as the default, allowing command-line override
    ap.add_argument("-i", "--input", default=SOURCE_FILE, type=Path,
                    help=f"Input JSON file path (e.g., {PROMPT_DIR}/benign_behaviors_validation.json).")
    ap.add_argument("-o", "--outdir", default=PROMPT_DIR, type=Path,
                    help="Output directory for translated files.")
    ap.add_argument("--langs", help="Comma-separated DeepL codes to translate to (e.g. ES,FR,DE,ZH). If omitted, all supported languages will be used.")
    args = ap.parse_args()

    # Create output directory if it doesn't exist
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Check if the input file exists in the specified location
    input_file_path = args.input
    if not input_file_path.exists():
        print(f"Error: Input file not found at {input_file_path}")
        return

    translator = get_translator()
    
    # Load rows and the global canary ID
    rows, canary_guid = load_prompts(input_file_path)
    print(f"Loaded {len(rows)} behaviors from {input_file_path}.")

    # --- Preprocessing for Translation ---
    
    texts_to_translate = []
    text_map = [] 
    
    for row_index, r in enumerate(rows):
        # 1a. Prompt text (critical)
        texts_to_translate.append(md_to_xml(r["prompt"]))
        text_map.append((row_index, "prompt"))
        
        # 1b. Name text (critical for the target files)
        if "name" in r:
            texts_to_translate.append(r["name"])
            text_map.append((row_index, "name"))

    # 2. Extract unique tool names for the glossary
    tool_names = set()
    for r in rows:
        tool_names.update(r.get("target_functions", []))
        tool_names.update(CODE_SPAN_RE.findall(r["prompt"]))

    # 3. Determine target languages
    if args.langs:
        user_codes = {c.strip().upper() for c in args.langs.split(",") if c.strip()}
        languages_map = {
            deepl_code: our_code 
            for deepl_code, our_code in LANGUAGE_MAP.items() 
            if deepl_code in user_codes or our_code == 'en'
        }
    else:
        languages_map = LANGUAGE_MAP

    # --- Translation Loop ---

    base_file_stem = input_file_path.stem # e.g., benign_behaviors_validation

    for deepl_code, our_code in languages_map.items():
        lang_name = LANGUAGE_NAMES.get(our_code, our_code)
        
        # Determine output file name: benign_behaviors_validation_es.json
        outfile_name = f"{base_file_stem}_{our_code}.json"
        outfile_path = outdir / outfile_name

        print(f"\n[{our_code.upper()}] Processing {lang_name}...")
        
        is_source_lang = (our_code == 'en')
        
        # 1. Translate / Copy Texts
        if is_source_lang:
            translated_texts_xml = texts_to_translate
        else:
            glossary = maybe_make_glossary(translator, deepl_code, tool_names)
            if glossary:
                print(" Using glossary to preserve tool names.")
            
            try:
                translated_texts_xml = translate_batch(
                    translator, texts_to_translate, target=deepl_code, glossary=glossary
                )
            finally:
                if glossary:
                    try:
                        translator.delete_glossary(glossary.glossary_id)
                    except deepl.exceptions.DeepLException:
                        pass
        
        # 2. Post-process and Assemble new JSON
        translated_rows = deepcopy(rows)
        translated_texts = [xml_to_md(t) for t in translated_texts_xml]
        
        # Distribute translated texts back into the data structure
        for text, (row_index, field_key) in zip(translated_texts, text_map):
            translated_rows[row_index][field_key] = text
        
        # Add metadata fields to each row
        final_rows = []
        for row in translated_rows:
            new_row = {**row}
            new_row["target_language"] = lang_name
            new_row["target_language_code"] = our_code
            new_row["source_language_code"] = "EN"
            final_rows.append(new_row)
        
        # 3. Write Output File
        output_data = {
            "canary_guid": canary_guid,
            "behaviors": final_rows,
            "metadata": {
                "source_file": input_file_path.name,
                "translation_target": lang_name,
                "translation_service": "Manual Copy" if is_source_lang else "DeepL API"
            }
        }
        
        # Write to disk
        outfile_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" Saved {len(final_rows)} behaviors to {outfile_name}")

    # --- Usage Check ---
    try:
        usage = translator.get_usage()
        if usage.any_limit_reached:
            print("\nTranslation limit reached.")
        if usage.character.valid:
            print(f" Character usage: {usage.character.count} of {usage.character.limit}")
    except deepl.exceptions.DeepLException as e:
        print(f" Could not fetch usage: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()