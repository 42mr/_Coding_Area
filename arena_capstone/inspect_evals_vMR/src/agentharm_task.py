# agentharm_task.py
# Expose the two @task-decorated callables directly in this module.
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign

__all__ = ["agentharm", "agentharm_benign"]