# translate_prompts_v2.py
# example prompt: python translate_prompts_v2.py -i ./prompt_data/validation_prompts_en.json -o ./prompt_data/validation_prompts_de_new.json 


import json
import deepl
import argparse

def translate_prompts(input_file, output_file):
    auth_key = "1d33ba23-4eab-4245-ba14-d72f5f7f8d82:fx"  # Replace with your actual DeepL API key
    translator = deepl.Translator(auth_key)

    # Load the original JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Translate only the "prompt" field
    for item in data:
        if "prompt" in item and item["prompt"].strip():
            result = translator.translate_text(
                item["prompt"],
                target_lang="DE"   # German
            )
            item["prompt"] = result.text

    # Save new JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Translation complete! Saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate only 'prompt' fields in JSON using DeepL")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file (English)")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file (German)")
    args = parser.parse_args()

    translate_prompts(args.input, args.output)
