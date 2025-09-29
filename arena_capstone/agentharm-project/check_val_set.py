# check_val_set.py
from inspect_ai import eval_async
from inspect_evals.agentharm import agentharm
import asyncio

# Get the task object
task = agentharm(split="val")

# The task contains dataset property with samples
print(f"Task name: {task.name}")
print(f"Task dataset type: {type(task.dataset)}")

# Try to access the dataset
if hasattr(task, 'dataset') and task.dataset:
    if hasattr(task.dataset, '__len__'):
        print(f"Total validation samples: {len(task.dataset)}")
    else:
        print("Dataset exists but has no length")
        
    # Try to iterate and count
    try:
        samples = list(task.dataset)
        print(f"Counted samples: {len(samples)}")
        
        # Count by category
        categories = {}
        for sample in samples:
            if hasattr(sample, 'metadata'):
                cat = sample.metadata.get('category', 'Unknown')
            else:
                cat = 'Unknown'
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nSamples per category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
    except Exception as e:
        print(f"Could not iterate dataset: {e}")

