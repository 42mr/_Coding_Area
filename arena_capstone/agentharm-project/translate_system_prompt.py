# translate_system_prompt_variants.py
"""
Translate the English system prompts (including variants) to 8 other languages using DeepL API.
Outputs a single JSON file with all translations.
"""

import json
import os
from pathlib import Path
import deepl
from typing import Dict, Any

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

# 1. ADJUSTED: Load all prompts from the source JSON structure
def load_source_prompts(filepath: str = 'system_prompt_source.json') -> Dict[str, str]:
    """Load all English system prompt variants from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all prompt keys that contain 'en' as the language
    source_prompts = {
        key: value['en']
        for key, value in data.items()
        if isinstance(value, dict) and 'en' in value
    }
    return source_prompts

# 2. ADJUSTED: Iterate over all source prompts and translate each one
def translate_system_prompts(source_prompts: Dict[str, str], output_file: str = 'system_prompts_all.json'):
    """Translate all system prompt variants to all target languages using DeepL."""
    
    # Initialize DeepL translator
    auth_key = "1d33ba23-4eab-4245-ba14-d72f5f7f8d82:fx"
    if not auth_key:
        # In a real script, this should be os.getenv('DEEPL_API_KEY')
        raise SystemExit("DeepL API Key is not available.") 
    
    translator = deepl.Translator(auth_key)
    
    # Initialize the structure for all prompts and all languages
    translated_data: Dict[str, Any] = {}
    
    print(f"Translating {len(source_prompts)} prompt variants to {len(LANGUAGE_MAP)} languages...")
    
    # 2.1. Iterate over each prompt variant (e.g., 'system_prompt', 'system_prompt_ReAct')
    for prompt_key, source_text in source_prompts.items():
        print("-" * 50)
        print(f"Processing variant: {prompt_key}")
        
        # Initialize dictionary for this prompt variant's translations
        translated_data[prompt_key] = {}
        
        # Add English source text
        translated_data[prompt_key]['en'] = {
            'text': source_text,
            'language_name': 'English',
            'source': True
        }
        
        # 2.2. Translate to each target language
        for deepl_code, our_code in LANGUAGE_MAP.items():
            lang_name = LANGUAGE_NAMES[our_code]
            print(f"  -> Translating to {lang_name} ({our_code})...", end=' ')
            
            try:
                # Assuming the DeepL call is correct here
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
    
    # 2.3. Final output structure (Metadata + all prompts)
    output_data = {
        'metadata': {
            'description': 'System prompts for AgentHarm multilingual evaluation',
            'source_language': 'en',
            'translation_service': 'DeepL API',
            'total_languages': len(LANGUAGE_MAP) + 1,
            'total_prompt_variants': len(source_prompts)
        }
    }
    # Merge the prompt data into the final output
    output_data.update(translated_data) 
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print(f"All translations saved to {output_file}")
    
    # Check usage
    usage = translator.get_usage()
    if usage.character.valid:
        print(f"DeepL character usage: {usage.character.count:,} of {usage.character.limit:,}")
    
    return output_data

# 3. ADJUSTED: Preview the nested structure
def preview_translations(filepath: str = 'system_prompts_all.json'):
    """Preview all translations from the output file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "=" * 70)
    print("SYSTEM PROMPT TRANSLATIONS PREVIEW")
    print("=" * 70)
    
    # Filter out metadata
    prompt_variants = {
        k: v for k, v in data.items() if k != 'metadata'
    }
    
    for prompt_key, lang_data in prompt_variants.items():
        print(f"\n--- Variant: {prompt_key} ---")
        # Just preview English and one other language for brevity
        
        # Find English for source preview
        en_data = lang_data.get('en', {})
        if en_data:
             preview_en = en_data['text'][:100].replace('\n', ' ') + "..."
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
    
    ap = argparse.ArgumentParser(description="Translate system prompt variants to multiple languages via DeepL")
    ap.add_argument("--source", default="system_prompt_source.json", 
                    help="Source JSON file with English prompts (e.g., system_prompt, system_prompt_ReAct)")
    ap.add_argument("--output", default="system_prompts_all.json",
                    help="Output JSON file for all translations")
    ap.add_argument("--preview", action="store_true",
                    help="Preview existing translations without re-translating")
    args = ap.parse_args()
    
    if args.preview:
        if not Path(args.output).exists():
            print(f"Error: {args.output} not found. Run without --preview first.")
        else:
            preview_translations(args.output)
    else:
        # Load source
        source_prompts = load_source_prompts(args.source)
        
        print(f"Loaded {len(source_prompts)} source prompt variants.")
        for key, text in source_prompts.items():
             print(f"  - {key}: {len(text):,} characters")

        # Translate
        translate_system_prompts(source_prompts, args.output)
        
        # Preview
        preview_translations(args.output)