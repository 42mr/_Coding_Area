import os
print(os.getenv("OPENAI_API_KEY"))

# test_setup.py
import inspect_ai
from inspect_ai import eval
from inspect_ai.log import list_eval_logs

print(f"Inspect AI version: {inspect_ai.__version__}")
print("Setup successful!")

# Check if AgentHarm is available
try:
    from inspect_evals.agentharm import agentharm
    print("AgentHarm benchmark found!")
except ImportError as e:
    print(f"AgentHarm import issue: {e}")


import sys
print(sys.executable)
print(sys.path)
