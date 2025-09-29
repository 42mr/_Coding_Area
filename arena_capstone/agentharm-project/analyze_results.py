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

# Extract metrics from combined_scorer
harm_scores = []
refusals = []

for sample in log.samples:
    if 'combined_scorer' in sample.scores:
        score_dict = sample.scores['combined_scorer'].value
        if 'score' in score_dict:
            harm_scores.append(score_dict['score'])
        if 'refusal' in score_dict:
            refusals.append(score_dict['refusal'])

print(f"\nFound {len(harm_scores)} harm scores")
print(f"Found {len(refusals)} refusal scores")

# Calculate metrics
avg_harm = sum(harm_scores) / len(harm_scores) if harm_scores else 0
refusal_rate = sum(refusals) / len(refusals) if refusals else 0

print(f"\nAverage Harm Score: {avg_harm:.2%}")
print(f"Refusal Rate: {refusal_rate:.2%}")

# Non-refusal harm score
non_refused_scores = [h for h, r in zip(harm_scores, refusals) if r == 0]
non_refusal_harm = sum(non_refused_scores) / len(non_refused_scores) if non_refused_scores else 0
print(f"Non-refusal Harm Score: {non_refusal_harm:.2%}")

# Category breakdown
categories = {}
for sample in log.samples:
    category = sample.metadata.get('category', 'Unknown')
    if category not in categories:
        categories[category] = {'scores': [], 'refusals': []}
    
    if 'combined_scorer' in sample.scores:
        score_dict = sample.scores['combined_scorer'].value
        categories[category]['scores'].append(score_dict.get('score', 0))
        categories[category]['refusals'].append(score_dict.get('refusal', 0))

print("\nPer-category results:")
for cat, data in sorted(categories.items()):
    if data['scores']:
        avg_score = sum(data['scores']) / len(data['scores'])
        avg_refusal = sum(data['refusals']) / len(data['refusals'])
        print(f"  {cat}: Harm={avg_score:.2%}, Refusal={avg_refusal:.2%}, Count={len(data['scores'])}")
        
print("\nAnalysis complete.")
