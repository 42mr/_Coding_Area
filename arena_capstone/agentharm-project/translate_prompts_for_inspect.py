# translate_prompts_for_inspect.py
# keeps the structure of the original inspect evals JSON, only translates the "prompt" field
# 
""" 
command line example:
python translate_prompts_for_inspect.py -i ./prompt_data/harmful_behaviors_validation.json -o ./prompt_data/harmful_behaviors_validation_de.json -l DE 
"""

import json
import deepl
import argparse

def translate_prompts(input_file, output_file, target_lang="DE"):
    auth_key = "1d33ba23-4eab-4245-ba14-d72f5f7f8d82:fx"
    translator = deepl.Translator(auth_key)
    
    # Load the original JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check if data has "behaviors" key (English format)
    if "behaviors" in data:
        behaviors = data["behaviors"]
    else:
        # If it's already a list, use it directly
        behaviors = data
    
    # Translate only the "prompt" field in each behavior
    for item in behaviors:
        if "prompt" in item and item["prompt"].strip():
            result = translator.translate_text(
                item["prompt"],
                target_lang=target_lang
            )
            item["prompt"] = result.text
            print(f"Translated: {item.get('id', 'unknown')}")
    
    # Save with the SAME structure as input
    output_data = {"behaviors": behaviors} if "behaviors" in data else behaviors
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Translation complete! Saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate only 'prompt' fields in JSON using DeepL")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-l", "--lang", default="DE", help="Target language (default: DE for German)")
    args = parser.parse_args()
    
    translate_prompts(args.input, args.output, args.lang)
    
