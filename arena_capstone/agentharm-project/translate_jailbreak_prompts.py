# translate_jailbreak_prompts.py
"""
Translate the English jailbreak prompt components to 8 other languages using DeepL API.
Outputs a single JSON file with all translations, storing the source and output 
in the '../prompt_data/' directory.
"""

import json
import os
from pathlib import Path
import deepl
from typing import Dict, Any

# --- FILE PATH CONSTANTS ---
# Define the directory where the source and output files reside relative to the script
PROMPT_DIR = Path('prompt_data')
SOURCE_FILE_NAME = 'jail_break_prompt_source.json'
OUTPUT_FILE_NAME = 'jail_break_prompt_all.json'

# Construct the full paths (assuming script is run from project root)
SOURCE_FILE = PROMPT_DIR / SOURCE_FILE_NAME
OUTPUT_FILE = PROMPT_DIR / OUTPUT_FILE_NAME
# ---------------------------

# Language mapping: DeepL codes to our codes
LANGUAGE_MAP = {
    'ES': 'es',  # Spanish
    'FR': 'fr',  # French
    'DE': 'de',  # German
    'BG': 'bg',  # Bulgarian
    'ZH': 'zh',  # Chinese (simplified)
    'ID': 'id',  # Indonesian
    'FI': 'fi',  # Finnish
    'LT': 'lt',  # Lithuanian
}

LANGUAGE_NAMES = {
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'bg': 'Bulgarian',
    'zh': 'Chinese',
    'id': 'Indonesian',
    'fi': 'Finnish',
    'lt': 'Lithuanian',
    'en': 'English'
}

# 1. CORRECTED: Load both components from the source JSON
def load_source_prompts(filepath: Path) -> Dict[str, str]:
    """
    Load the English text from the nested structure of the source JSON.
    Expected source structure is: {component_key: {'en': {'text': '...'}}}
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Source file not found at: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Define the keys we expect to find at the top level
    PROMPT_KEYS = ['system_message_addendum', 'user_message_addendum']
    
    source_texts = {}
    for key in PROMPT_KEYS:
        # Navigate the nested structure: data[key]['en']['text']
        try:
            # Check if the key exists and has the correct nested structure
            english_text = data[key]['en']['text']
            source_texts[key] = english_text
        except KeyError:
            # This handles missing top-level keys or missing nested 'en':'text' keys
            source_texts[key] = "" 
            print(f"Warning: Could not find '{key}' or its nested 'en':'text' key in the source file. Using empty string.")

    return source_texts

# 2. Translate both components
def translate_jailbreak_prompts(source_prompts: Dict[str, str], output_filepath: Path):
    """Translate all jailbreak prompt components to all target languages using DeepL."""
    
    # Initialize DeepL translator (Use your actual API key)
    auth_key = "1d33ba23-4eab-4245-ba14-d72f5f7f8d82:fx"
    if not auth_key:
        raise SystemExit("DeepL API Key is not set in environment or provided.")
    
    translator = deepl.Translator(auth_key)
    
    translated_data: Dict[str, Any] = {}
    
    print(f"Translating {len(source_prompts)} prompt components to {len(LANGUAGE_MAP)} languages...")
    
    # Iterate over each prompt component ('system_message_addendum', 'user_message_addendum')
    for prompt_key, source_text in source_prompts.items():
        print("-" * 50)
        print(f"Processing component: {prompt_key} ({len(source_text):,} chars)")
        
        # Initialize translation dictionary for this component
        translated_data[prompt_key] = {}
        
        # Add English source text
        translated_data[prompt_key]['en'] = {
            'text': source_text,
            'language_name': 'English',
            'source': True
        }
        
        # Translate to each target language
        for deepl_code, our_code in LANGUAGE_MAP.items():
            lang_name = LANGUAGE_NAMES[our_code]
            print(f"  -> Translating to {lang_name} ({our_code})...", end=' ')
            
            # Skip translation if the source text is empty (avoids DeepL ValueError)
            if not source_text:
                translated_data[prompt_key][our_code] = {
                    'text': "", 
                    'language_name': lang_name, 
                    'error': "Source text was empty"
                }
                print("SKIPPED (Empty source)")
                continue

            try:
                result = translator.translate_text(
                    source_text,
                    target_lang=deepl_code,
                    formality='default'
                )
                
                translated_data[prompt_key][our_code] = {
                    'text': result.text,
                    'language_name': lang_name,
                    'detected_source_lang': result.detected_source_lang
                }
                
                print("✓")
                
            except deepl.DeepLException as e:
                print(f"✗ Error: {e}")
                continue
    
    # 3. Final output structure
    output_data = {
        'metadata': {
            'description': 'Jailbreak prompt components for AgentHarm multilingual evaluation',
            'source_language': 'en',
            'translation_service': 'DeepL API',
            'total_languages': len(LANGUAGE_MAP) + 1,
            'total_components': len(source_prompts)
        }
    }
    output_data.update(translated_data) 
    
    # Ensure the output directory exists before saving
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print(f"All translations saved to {output_filepath}")
    
    # Check usage
    usage = translator.get_usage()
    if usage.character.valid:
        print(f"DeepL character usage: {usage.character.count:,} of {usage.character.limit:,}")
    
    return output_data

# 4. Preview the nested structure
def preview_translations(filepath: Path):
    """Preview translations from the output file."""
    if not filepath.exists():
        print(f"Error: Output file not found at: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "=" * 70)
    print("JAILBREAK PROMPT TRANSLATIONS PREVIEW")
    print("=" * 70)
    
    # Filter out metadata
    prompt_components = {
        k: v for k, v in data.items() if k != 'metadata'
    }
    
    for component_key, lang_data in prompt_components.items():
        print(f"\n--- Component: {component_key} ---")
        
        # Preview English source
        en_data = lang_data.get('en', {})
        if en_data:
             # Use replace to handle newlines elegantly for preview
             preview_en = en_data.get('text', '')[:100].replace('\n', ' ') + "..."
             print(f"[EN] English (Source): {preview_en}")
        
        # Find the first translated language for comparison
        first_lang_code = next(iter(k for k in lang_data.keys() if k != 'en'), None)
        if first_lang_code:
            prompt_data = lang_data[first_lang_code]
            lang_name = prompt_data['language_name']
            text = prompt_data['text']
            preview = text[:100].replace('\n', ' ') + "..."
            print(f"[{first_lang_code.upper()}] {lang_name}: {preview}")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Translate jailbreak prompt components to multiple languages via DeepL")
    ap.add_argument("--source", default=SOURCE_FILE, type=Path,
                    help=f"Source JSON file with English prompt components (default: {SOURCE_FILE})")
    ap.add_argument("--output", default=OUTPUT_FILE, type=Path,
                    help=f"Output JSON file for all translations (default: {OUTPUT_FILE})")
    ap.add_argument("--preview", action="store_true",
                    help="Preview existing translations without re-translating")
    args = ap.parse_args()
    
    if args.preview:
        if not args.output.exists():
            print(f"Error: {args.output} not found. Run without --preview first.")
        else:
            preview_translations(args.output)
    else:
        # Load source
        try:
            source_prompts = load_source_prompts(args.source)
        except FileNotFoundError as e:
            print(e)
            exit()
        
        print(f"Loaded {len(source_prompts)} source prompt components.")
        for key, text in source_prompts.items():
             print(f"  - {key}: {len(text):,} characters")

        # Translate
        translate_jailbreak_prompts(source_prompts, args.output)
        
        # Preview
        preview_translations(args.output)