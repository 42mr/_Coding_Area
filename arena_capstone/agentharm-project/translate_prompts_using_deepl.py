# translate_prompts_using_deepl.py
import json
import os
import deepl

# Initialize DeepL translator
auth_key = os.getenv("DEEPL_API_KEY")  # Set this in your .env
translator = deepl.Translator(auth_key)

# DeepL language codes
languages = {
    'ES': 'Spanish',
    'FR': 'French', 
    'DE': 'German',
    'ZH': 'Chinese (simplified)'
}

with open('validation_prompts_en.json', 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# Translate each prompt to each language
for lang_code, lang_name in languages.items():
    print(f"\nTranslating to {lang_name}...")
    translated = []
    
    for i, prompt_data in enumerate(prompts):
        print(f"  {i+1}/{len(prompts)}: {prompt_data['id']}")
        
        try:
            # DeepL translation with formality setting where applicable
            result = translator.translate_text(
                prompt_data['prompt'],
                target_lang=lang_code,
                formality='default'  # or 'less'/'more' for some languages
            )
            
            translated.append({
                **prompt_data,
                'original_prompt': prompt_data['prompt'],
                'translated_prompt': result.text,
                'target_language': lang_name,
                'detected_source_lang': result.detected_source_lang
            })
            
        except deepl.DeepLException as e:
            print(f"    Error translating {prompt_data['id']}: {e}")
            continue
    
    # Save
    with open(f'validation_prompts_{lang_code.lower()}.json', 'w', encoding='utf-8') as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(translated)} translations to validation_prompts_{lang_code.lower()}.json")

# Check usage
usage = translator.get_usage()
if usage.any_limit_reached:
    print('\nTranslation limit reached.')
if usage.character.valid:
    print(f"\nCharacter usage: {usage.character.count} of {usage.character.limit}")

print("\nTranslation complete!")