# analyze_results.py
from inspect_ai.log import read_eval_log, list_eval_logs

# Get the most recent log file
logs = list_eval_logs()
if not logs:
    print("No log files found!")
    exit(1)

latest_log = logs[0]
log = read_eval_log(latest_log)

print(f"Model: {log.eval.model}")
print(f"Total samples: {len(log.samples)}")

# Debug: Check what scores are actually present
if log.samples:
    print("\nFirst sample scores available:")
    print(log.samples[0].scores.keys())
    print("\nFirst sample metadata:")
    print(log.samples[0].metadata)

# Calculate metrics - using the correct score names
harm_scores = []
refusals = []

for sample in log.samples:
    # Check all possible score keys
    score_keys = list(sample.scores.keys())
    
    # The actual key might be different
    for key in score_keys:
        if 'harm' in key.lower():
            harm_scores.append(sample.scores[key].value)
        if 'refusal' in key.lower() or 'refuse' in key.lower():
            refusals.append(sample.scores[key].value)

print(f"\nFound {len(harm_scores)} harm scores")
print(f"Found {len(refusals)} refusal scores")

if harm_scores:
    avg_harm = sum(harm_scores) / len(harm_scores)
    print(f"Average Harm Score: {avg_harm:.2%}")

if refusals:
    refusal_rate = sum(refusals) / len(refusals)
    print(f"Refusal Rate: {refusal_rate:.2%}")
    
    # Non-refusal harm score
    non_refused_scores = [h for h, r in zip(harm_scores, refusals) if not r]
    if non_refused_scores:
        non_refusal_harm = sum(non_refused_scores) / len(non_refused_scores)
        print(f"Non-refusal Harm Score: {non_refusal_harm:.2%}")