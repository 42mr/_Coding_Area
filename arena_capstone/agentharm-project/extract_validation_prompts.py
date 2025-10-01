# extract_validation_prompts.py
from inspect_ai.log import read_eval_log, list_eval_logs
import json
import sys

logs = list_eval_logs()
if not logs:
    print("No evaluation logs found.")
    sys.exit(1)

log = read_eval_log(logs[0])
prompts_data = []

for sample in log.samples:
    user_messages = [msg for msg in sample.messages if msg.role == "user"]
    if user_messages:
        prompts_data.append({
            'id': sample.metadata.get('id'),
            'id_original': sample.metadata.get('id_original'),
            'detailed_prompt': sample.metadata.get('detailed_prompt'),
            'hint_included': sample.metadata.get('hint_included'),
            'name': sample.metadata.get('name'),
            'category': sample.metadata.get('category'),
            'prompt': user_messages[0].content,
            'target_functions': sample.metadata.get('target_functions', []),
            'grading_function': sample.metadata.get('grading_function'),
            'semantic_judge': sample.metadata.get('semantic_judge'),
        })

out_file = 'validation_prompts_en.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(prompts_data, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(prompts_data)} prompts to {out_file}")