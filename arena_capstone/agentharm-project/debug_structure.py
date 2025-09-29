# debug_structure.py
from inspect_ai.log import read_eval_log, list_eval_logs

logs = list_eval_logs()
log = read_eval_log(logs[0])

# Deep dive into the first sample's score structure
sample = log.samples[0]
combined_score = sample.scores['combined_scorer']

print("Type of combined_scorer:", type(combined_score))
print("\nAttributes:")
print(dir(combined_score))

print("\n=== Value ===")
print(combined_score.value)
print(f"Type: {type(combined_score.value)}")

if hasattr(combined_score, 'metadata'):
    print("\n=== Metadata ===")
    print(combined_score.metadata)

# Check the actual score structure
print("\n=== Full sample info ===")
print(f"Sample ID: {sample.id}")
print(f"Scores keys: {sample.scores.keys()}")