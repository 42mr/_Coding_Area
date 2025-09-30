# translate_prompts_using_deepl.py
import argparse
import json
import os
import re
import time
from pathlib import Path
import deepl

DEFAULT_LANGS = {
#    "ES": "Spanish",
#    "FR": "French",
#    "DE": "German",
#    "BG": "Bulgarian",
#    "ZH": "Chinese (simplified)",
#    "ID": "Indonesian",
#    "FI": "Finnish",
#    "LT": "Lithuanian",
}

CODE_SPAN_RE = re.compile(r"`([^`]+)`")  # markdown inline code

# Build a tuple of retryable exceptions that exist in this deepl version.
_retryables = []
for name in ("TooManyRequestsException", "ConnectionException", "NetworkException"):
    exc = getattr(deepl.exceptions, name, None)
    if exc:
        _retryables.append(exc)
RETRYABLE_EXC = tuple(_retryables) or (deepl.exceptions.DeepLException,)

def get_translator() -> deepl.Translator:
    key = "1d33ba23-4eab-4245-ba14-d72f5f7f8d82:fx"
    if not key:
        raise SystemExit("DEEPL_API_KEY is not set.")
    return deepl.Translator(key)

def load_prompts(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}")
    for i, row in enumerate(data):
        if "id" not in row or "prompt" not in row:
            raise SystemExit(f"Row {i} missing 'id' or 'prompt'.")
    return data

def md_to_xml(text: str) -> str:
    # Wrap markdown inline code in <code>…</code> so DeepL can ignore it
    return CODE_SPAN_RE.sub(r"<code>\1</code>", text)

def xml_to_md(text: str) -> str:
    # Restore markdown after translation
    return (text
            .replace("<code>", "`")
            .replace("</code>", "`"))

def translate_batch(translator: deepl.Translator, texts: list[str], target: str,
                    formality: str | None = "default",
                    source: str | None = "EN",
                    max_retries: int = 5,
                    glossary: deepl.GlossaryInfo | None = None) -> list[str]:
    attempt = 0
    while True:
        try:
            res = translator.translate_text(
                texts,
                target_lang=target,
                source_lang=source,
                formality=formality,
                preserve_formatting=True,
                split_sentences="nonewlines",
                tag_handling="xml",
                ignore_tags=["code"],
                glossary=glossary,
            )
            return [r.text for r in res]
        # Catch retryable exceptions
        except RETRYABLE_EXC as e:
            if attempt >= max_retries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"   Retryable error ({e.__class__.__name__}). Retrying in {wait}s…")
            time.sleep(wait)
            attempt += 1
        except deepl.exceptions.DeepLException:
            # Non-retryable DeepL error (e.g., auth/quota/invalid request)
            raise
        

def maybe_make_glossary(translator: deepl.Translator, target_code: str, terms: list[str]):
    """
    Optional: pin tool names to themselves. Creates a temporary glossary when supported.
    If glossary creation is unsupported for the language pair, returns None.
    """
    if not terms:
        return None
    try:
        entries = {t: t for t in terms}
        return translator.create_glossary(
            name=f"tool-names-en-{target_code.lower()}",
            source_lang="EN",
            target_lang=target_code,
            entries=entries,
        )
    except deepl.exceptions.DeepLException:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="validation_prompts_en.json")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--langs", help="Comma-separated codes, e.g. ES,FR,DE,ZH")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    translator = get_translator()
    rows = load_prompts(Path(args.input))

    # Preprocess: convert markdown code spans to <code>…</code>
    texts_xml = [md_to_xml(r["prompt"]) for r in rows]

    # Extract unique tool names from both target_functions and backticked spans
    tool_names = set()
    for r in rows:
        tool_names.update(r.get("target_functions", []))
        tool_names.update(CODE_SPAN_RE.findall(r["prompt"]))

    if args.langs:
        codes = [c.strip().upper() for c in args.langs.split(",") if c.strip()]
        languages = {c: DEFAULT_LANGS.get(c, c) for c in codes}
    else:
        languages = DEFAULT_LANGS

    for code, name in languages.items():
        print(f"\nTranslating to {name} ({code})…")

        glossary = maybe_make_glossary(translator, code, sorted(tool_names))
        if glossary:
            print(" Using glossary to preserve tool names.")

        try:
            translated_xml = translate_batch(
                translator, texts_xml, target=code, glossary=glossary
            )
        finally:
            # Clean up any ephemeral glossary to avoid clutter (ignore failures)
            if glossary:
                try:
                    translator.delete_glossary(glossary.glossary_id)
                except deepl.exceptions.DeepLException:
                    pass

        translated_texts = [xml_to_md(t) for t in translated_xml]

        out_rows = []
        for base, t in zip(rows, translated_texts):
            new_row = {**base}
            new_row["prompt"] = t  # replace prompt with translated text
            new_row["target_language"] = name
            new_row["target_language_code"] = code
            new_row["source_language_code"] = "EN"
            # Ensure machine-facing fields are unchanged
            # (target_functions stays exactly as in source)
            out_rows.append(new_row)

        outfile = outdir / f"validation_prompts_{code.lower()}.json"
        outfile.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" Saved {len(out_rows)} → {outfile}")

    # Usage check
    try:
        usage = translator.get_usage()
        if usage.any_limit_reached:
            print("\nTranslation limit reached.")
        if usage.character.valid:
            print(f" Character usage: {usage.character.count} of {usage.character.limit}")
    except deepl.exceptions.DeepLException as e:
        print(f" Could not fetch usage: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
