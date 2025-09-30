# download_agentharm_datasets.py
"""
Download AgentHarm behavior sets (and optional chat sets) from Hugging Face.
Dataset page: https://huggingface.co/datasets/ai-safety-institute/AgentHarm

Files are stored under the 'benchmark/' subfolder of the dataset repo.
This script is compatible with multiple huggingface_hub versions.
"""

import json
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download

# Try to import a hub HTTP error class in a version-agnostic way.
try:
    # Newer path
    from huggingface_hub.utils import HfHubHTTPError as _HubError
except Exception:
    try:
        # Some versions export it at the top-level (rare)
        from huggingface_hub import HfHubHTTPError as _HubError  # type: ignore
    except Exception:
        # Fallback: just use Exception as a catch-all
        _HubError = Exception  # type: ignore

REPO_ID = "ai-safety-institute/AgentHarm"

# Behavior file mappings (live under benchmark/)
DATASETS: Dict[str, str] = {
    "harmful_public_test":  "benchmark/harmful_behaviors_test_public.json",
    "benign_public_test":   "benchmark/benign_behaviors_test_public.json",
    "harmful_validation":   "benchmark/harmful_behaviors_validation.json",
    "benign_validation":    "benchmark/benign_behaviors_validation.json",
}

# Optional chat-only prompt sets (also under benchmark/)
CHAT_DATASETS: Dict[str, str] = {
    "chat_public_test":     "benchmark/chat_public_test.json",
    "chat_validation":      "benchmark/chat_validation.json",
}

def _count_items(payload):
    # Behavior files: object with "behaviors": [...]
    if isinstance(payload, dict) and isinstance(payload.get("behaviors"), list):
        return "behaviors", len(payload["behaviors"])
    # Chat files: usually a list
    if isinstance(payload, list):
        return "items", len(payload)
    # Fallback
    try:
        return "keys", len(payload)
    except Exception:
        return "items", 0

def download_agentharm_datasets(output_dir=".", include_chat=False, revision="main"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING AGENTHARM DATASETS")
    print("=" * 70)
    print(f"Repository:      {REPO_ID}")
    print(f"Revision/branch: {revision}")
    print(f"Output dir:      {out.resolve()}\n")

    targets = dict(DATASETS)
    if include_chat:
        targets.update(CHAT_DATASETS)

    downloaded = {}

    for name, hub_path in targets.items():
        print(f"Downloading {name:>22s}  ({hub_path}) ... ", end="")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=hub_path,           # includes 'benchmark/' subdir
                repo_type="dataset",
                revision=revision,           # default 'main'
            )
        except _HubError as e:
            print(f"✗ hub error: {e}")
            continue
        except Exception as e:
            print(f"✗ error: {e}")
            continue

        # Load and re-save in output dir (pretty-printed)
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"✗ JSON read error: {e}")
            continue

        dest_file = out / Path(hub_path).name  # save as basename
        try:
            with open(dest_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"✗ write error: {e}")
            continue

        kind, count = _count_items(payload)
        downloaded[name] = {"hub_path": hub_path, "filename": dest_file.name, "count": count}
        print(f"✓ ({count} {kind})")

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    def _line(key, label):
        if key in downloaded:
            info = downloaded[key]
            print(f"{label:<28} {info['count']:>5}  ({info['filename']})")
    _line("harmful_public_test", "Public test (harmful):")
    _line("benign_public_test",  "Public test (benign):")
    _line("harmful_validation",  "Validation (harmful):")
    _line("benign_validation",   "Validation (benign):")
    if include_chat:
        _line("chat_public_test", "Public test (chat):")
        _line("chat_validation",  "Validation (chat):")
    print("=" * 70)
    return downloaded

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download AgentHarm datasets from Hugging Face")
    ap.add_argument("--output-dir", default=".", help="Directory to save JSON files")
    ap.add_argument("--include-chat", action="store_true", help="Also download chat_* JSON files")
    ap.add_argument("--revision", default="main", help="Repo revision/branch/tag (default: main)")
    args = ap.parse_args()

    # Dependency presence check
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise SystemExit(1)

    download_agentharm_datasets(args.output_dir, include_chat=args.include_chat, revision=args.revision)
