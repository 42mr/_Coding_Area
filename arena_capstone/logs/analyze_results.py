from inspect_ai.log import read_eval_log

# Read the log file
log = read_eval_log('2025-10-01T17-37-37+01-00_agentharm_ZVTcTS5T2NDrNcE3tVs8HB.eval')

print("=" * 60)
print("AGENTHARM EVALUATION SUMMARY")
print("=" * 60)
print(f"Model: {log.eval.model}")
print(f"Dataset: {log.eval.dataset.name}")
print(f"Started: {log.stats.started_at}")
print(f"Completed: {log.stats.completed_at}")

# Token usage
for model_name, usage in log.stats.model_usage.items():
    print(f"\nToken Usage ({model_name}):")
    print(f"  Input tokens: {usage.input_tokens:,}")
    print(f"  Output tokens: {usage.output_tokens:,}")
    print(f"  Total tokens: {usage.total_tokens:,}")
    print(f"  Cached tokens: {usage.input_tokens_cache_read:,}")

print("\n" + "=" * 60)
print("OVERALL SCORES")
print("=" * 60)
for score in log.results.scores:
    print(f"\nScorer: {score.name}")
    print(f"Scored samples: {score.scored_samples}")
    print(f"Unscored samples: {score.unscored_samples}")
    
    print("\nKey Metrics:")
    metrics = score.metrics
    print(f"  Average Score: {metrics['avg_score'].value:.2%}")
    print(f"  Average Full Score: {metrics['avg_full_score'].value:.2%}")
    print(f"  Average Refusals: {metrics['avg_refusals'].value:.2%}")
    print(f"  Avg Score (non-refusals): {metrics['avg_score_non_refusals'].value:.2%}")

print("\n" + "=" * 60)
print("SCORES BY CATEGORY")
print("=" * 60)
for score in log.results.scores:
    metrics = score.metrics
    categories = ['Fraud', 'Drugs', 'Harassment', 'Copyright', 'Cybercrime', 
                  'Sexual', 'Disinformation', 'Hate']
    
    for cat in categories:
        score_key = f'{cat}_avg_scores'
        refusal_key = f'{cat}_avg_refusals'
        if score_key in metrics:
            print(f"{cat:15s}: Score={metrics[score_key].value:.2%}, Refusal={metrics[refusal_key].value:.2%}")

print("\n" + "=" * 60)
print(f"Total samples evaluated: {len(log.samples)}")
print("=" * 60)

