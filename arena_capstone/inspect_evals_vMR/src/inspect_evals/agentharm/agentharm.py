# agentharm.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model

from inspect_evals.agentharm.agents.agent import get_agent
from inspect_evals.agentharm.benchmark import (
    benign_grading_functions,
    harmful_grading_functions,
)
from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer
from inspect_evals.agentharm.utils import (
    filter_dataset,
    get_benchmark_tools,
    load_dataset,
    setup_tools_from_metadata,
)
from inspect_evals.agentharm.prompt_utils import set_active_language  # ensure prompts localize

PromptTechnique = Literal["standard", "cot", "react", "refusal"]
ToolChoice = Literal["forced_first", "auto", "none"]


@task
def agentharm_benign(
    agent: str = "default",
    behavior_ids: list[str] = [],
    split: Literal["val", "test_private", "test_public"] = "test_public",
    language: str = "en",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
    refusal_judge: str = "openai/gpt-4o-2024-08-06",
    semantic_judge: str = "openai/gpt-4o-2024-08-06",

    # First-class flags (no JSON blobs)
    system_prompt_template: Optional[str] = "default",
    prompt_technique: PromptTechnique = "standard",
    # None => plain "{prompt}" (no jailbreak template)
    user_prompt_template: Optional[str] = None,
    tool_choice: ToolChoice = "auto",

    # Back-compat: will be merged; explicit flags here win on conflict
    agent_kwargs: Optional[Dict[str, Any]] = None,
) -> Task:
    """
    AgentHarm benign benchmark task.
    """
    return run_agent_benchmark(
        agent=agent,
        behavior_ids=behavior_ids,
        task_name="benign",
        split=split,
        language=language,
        detailed_behaviors=detailed_behaviors,
        hint_included=hint_included,
        n_irrelevant_tools=n_irrelevant_tools,
        refusal_judge=refusal_judge,
        semantic_judge=semantic_judge,
        system_prompt_template=system_prompt_template,
        prompt_technique=prompt_technique,
        user_prompt_template=user_prompt_template,
        tool_choice=tool_choice,
        agent_kwargs=agent_kwargs,
    )


@task
def agentharm(
    agent: str = "default",
    behavior_ids: list[str] = [],
    split: Literal["val", "test_private", "test_public"] = "test_public",
    language: str = "en",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
    refusal_judge: str = "openai/gpt-4o-2024-08-06",
    semantic_judge: str = "openai/gpt-4o-2024-08-06",

    # First-class flags (no JSON blobs)
    system_prompt_template: Optional[str] = "default",
    prompt_technique: PromptTechnique = "standard",
    # None => plain "{prompt}" (no jailbreak template)
    user_prompt_template: Optional[str] = None,
    tool_choice: ToolChoice = "auto",

    # Back-compat: will be merged; explicit flags here win on conflict
    agent_kwargs: Optional[Dict[str, Any]] = None,
    chat_dataset: Optional[bool] = False,
) -> Task:
    """
    AgentHarm harmful (or chat) benchmark task.
    """
    return run_agent_benchmark(
        agent=agent,
        behavior_ids=behavior_ids,
        task_name="harmful" if not chat_dataset else "chat",
        split=split,
        language=language,
        detailed_behaviors=detailed_behaviors,
        hint_included=hint_included,
        n_irrelevant_tools=n_irrelevant_tools,
        refusal_judge=refusal_judge,
        semantic_judge=semantic_judge,
        system_prompt_template=system_prompt_template,
        prompt_technique=prompt_technique,
        user_prompt_template=user_prompt_template,
        tool_choice=tool_choice,
        agent_kwargs=agent_kwargs,
    )


def run_agent_benchmark(
    agent: str,
    behavior_ids: list[str] = [],
    task_name: Literal["harmful", "benign", "chat"] = "harmful",
    split: Literal["val", "test_private", "test_public"] = "test_public",
    language: str = "en",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
    refusal_judge: str = "openai/gpt-4o-2024-08-06",
    semantic_judge: str = "openai/gpt-4o-2024-08-06",

    # First-class flags (no JSON blobs)
    system_prompt_template: Optional[str] = "default",
    prompt_technique: PromptTechnique = "standard",
    # None => plain "{prompt}" (no jailbreak template)
    user_prompt_template: Optional[str] = None,
    tool_choice: ToolChoice = "auto",

    # Back-compat
    agent_kwargs: Optional[Dict[str, Any]] = None,
) -> Task:
    """
    Runs the AgentHarm benchmark task (harmful, benign, or chat).
    """
    # Load dataset (already localized by `language`)
    dataset = load_dataset(task_name, split, language)
    dataset = filter_dataset(dataset, behavior_ids, detailed_behaviors, hint_included)
    all_tool_names = get_benchmark_tools(task_name) if task_name != "chat" else []

    # Set active language for prompts (so agents don't need a 'language' arg)
    set_active_language(language)

    # Merge kwargs: start with any provided agent_kwargs, then overlay our flags (flags win)
    merged_kwargs: Dict[str, Any] = {}
    if agent_kwargs:
        merged_kwargs.update(agent_kwargs)

    merged_kwargs.update({
        # Do NOT pass 'language' into the agent; many agents don't accept it.
        "system_prompt_template": system_prompt_template,
        "prompt_technique": prompt_technique,
        "user_prompt_template": user_prompt_template,  # may be None => plain {prompt}
        "tool_choice": tool_choice,
    })

    agent_solver = get_agent(agent=agent, **merged_kwargs)

    plan = [
        setup_tools_from_metadata(
            task_name,
            all_tool_names,
            n_irrelevant_tools,
        ),
        agent_solver,
    ]

    refusal_judge_model = RefusalJudgeLLM(judge_model=refusal_judge)
    semantic_judge_model = get_model(semantic_judge)
    grading_module = (
        harmful_grading_functions
        if task_name == "harmful"
        else benign_grading_functions
        if task_name == "benign"
        else None
    )

    return Task(
        dataset=dataset,
        solver=plan,
        scorer=combined_scorer(
            grading_module, refusal_judge_model, semantic_judge_model
        ),
        max_messages=20,
        config=GenerateConfig(
            max_retries=3,
            max_tokens=4096,
            temperature=0.0,
            max_connections=100,
            seed=0,
        ),
    )
