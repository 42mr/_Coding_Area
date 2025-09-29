# mmlu_prompt_sweep.py
from inspect_ai import task
from inspect_ai.dataset import hf_dataset
from inspect_ai.model import user_prompt, system_message
from inspect_ai.solver import generate, multiple_choice, prompt_template
from inspect_ai.scorer import choice, accuracy

# -- Define atomic switches you will sweep --
def build_messages(q, opts, subject, variant):
    # variant is a dict of flags/strings
    role = variant.get("role", "")
    subject_tag = f"[Subject: {subject}]\n" if variant.get("subject_tag", True) else ""
    layout = variant.get("layout", "plain")  # "plain"|"bulleted"|"fenced"
    answer_fmt = variant.get("answer_fmt", "letter_only")  # "letter_only"|"final_answer"|"json"|"fenced"
    ordering = variant.get("ordering", "Q_first")          # "Q_first"|"Opts_first"
    constraint_placement = variant.get("constraint_pos", "end")  # "start"|"end"
    brevity = variant.get("brevity", "default")            # "short"|"verbose"
    internal_deliberation = variant.get("internal_delib", False)

    # Build instruction lines
    role_line = f"You are a domain expert in {subject}.\n" if role == "expert" else ""
    think_hint = "Consider carefully before answering.\n" if internal_deliberation else ""
    constraint = {
        "letter_only": "Answer with the single capital letter (A, B, C, or D) only.",
        "final_answer": "Provide your final answer as a single capital letter.",
        "json": "Respond as JSON: {\"answer\": \"A|B|C|D\"}.",
        "fenced": "Return only the letter inside a fenced code block."
    }[answer_fmt]

    constraint_block = constraint + (" Do not include explanations." if not variant.get("allow_explain", False) else "")

    instr_start = f"{subject_tag}{role_line}{think_hint}"
    instr_end = constraint_block

    # Lay out question/options
    if layout == "bulleted":
        opts_rendered = "\n".join([f"- {k}) {v}" for k, v in zip(list('ABCD'), opts)])
    else:
        opts_rendered = "\n".join([f"{k}) {v}" for k, v in zip(list('ABCD'), opts)])

    q_block = f"Question:\n{q}\nOptions:\n{opts_rendered}"
    if ordering == "Opts_first":
        q_block = f"Options:\n{opts_rendered}\nQuestion:\n{q}"

    # Constraint placement
    if constraint_placement == "start":
        text = f"{instr_start}{constraint_block}\n{q_block}"
    else:
        text = f"{instr_start}{q_block}\n{instr_end}"

    # Optional fencing
    if layout == "fenced":
        text = f"```text\n{text}\n```"

    # Brevity dial
    if brevity == "short":
        text = text.replace("Consider carefully before answering.\n", "")
    elif brevity == "verbose":
        text = "Please follow the instructions below.\n" + text

    return [
        system_message(variant.get("system_msg", "You are a helpful and careful assistant.")),
        user_prompt(text)
    ]

@task
def mmlu_prompt_sweep(model="openai/gpt-4o", variant=None, limit=None, seed=0):
    ds = hf_dataset("cais/mmlu", split="validation")  # or "test" if available locally
    # Narrow to 2000 stratified samples upstream if needed for runtime.

    def template(sample):
        q = sample["question"]
        options = [sample["choices"][k] for k in range(4)]
        subject = sample.get("subject", "general")
        msgs = build_messages(q, options, subject, variant or {})
        return prompt_template(messages=msgs)

    return {
        "dataset": ds,
        "solver": multiple_choice(generate(template=template), choices=list("ABCD")),
        "scorer": accuracy(choice(target="answer")),  # dataset has gold "answer" letter
        "config": {"limit": limit, "sample_shuffle": seed}
    }
