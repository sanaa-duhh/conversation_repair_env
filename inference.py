#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# OpenEnv hackathon inference runner: local ConversationRepairEnvironment + HF router LLM.
# Stdout must contain only [START], [STEP], and [END] lines (no other stdout).

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# Package lives next to this script: conversation_repair_env/conversation_repair/
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI

from conversation_repair.models import (
    ConversationRepairAction,
    ConversationRepairActionType,
    ConversationRepairObservation,
)
from conversation_repair.server.conversation_repair_environment import (
    ConversationRepairEnvironment,
)
from conversation_repair.tasks import list_task_ids

MAX_STEPS = 6
ENV_NAME = "conversation_repair"

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _format_reward_list(rewards: list[float]) -> str:
    return ",".join(f"{r:.2f}" for r in rewards)


def _normalize_score(sum_rewards: float, steps_taken: int) -> float:
    """
    Map cumulative step reward to [0, 1].
    Assumes per-step rewards are roughly in [-1, 1] and at most MAX_STEPS steps.
    """
    if steps_taken <= 0:
        return 0.0
    w = float(MAX_STEPS)
    return max(0.0, min(1.0, (sum_rewards + w) / (2.0 * w)))


def _single_line_compact(s: str) -> str:
    """Single-line token for action= (no newlines / control whitespace)."""
    t = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(t.split())


def _compact_action_log(action: ConversationRepairAction) -> str:
    t = action.action_type.value
    if t == "ASK":
        out = f"ASK:len={len(action.content.strip())}"
    elif t == "EXTRACT":
        out = f"EXTRACT:facts_count={len(action.extracted_facts)}"
    elif t == "ALIGN":
        out = f"ALIGN:conflict_alignments_count={len(action.conflict_alignments)}"
    elif t == "RESOLVE":
        has = bool((action.resolution_summary or action.content or "").strip())
        out = f"RESOLVE:has_summary={str(has).lower()}"
    else:
        out = f"{t}:default"
    return _single_line_compact(out)


def _messages_to_json(messages: list[Any]) -> str:
    out = []
    for m in messages:
        if hasattr(m, "model_dump"):
            out.append(m.model_dump(mode="json"))
        else:
            out.append(str(m))
    return json.dumps(out, ensure_ascii=False)


def _build_policy_prompt(
    task_id: str,
    step_n: int,
    obs: ConversationRepairObservation,
    state_thread_json: str,
    active_conflicts_json: str,
    previous_actions: list[str],
) -> tuple[str, str]:
    system = (
        "You are an agent in the Conversation Repair benchmark. You read a messy multi-user thread "
        "and emit exactly one structured action per turn as JSON (no markdown, no prose outside JSON).\n\n"
        "REQUIRED MULTI-STEP REASONING (follow this order across turns):\n"
        "Step 1 — EXTRACT: Pull key, thread-grounded facts into structured key→value pairs.\n"
        "Step 2 — ALIGN: Reconcile conflicting claims, using the facts you extracted; reference each "
        "active conflict_id and both sides in conflict_alignments.\n"
        "Step 3 — RESOLVE: Produce a final resolution that is consistent with extracted facts and "
        "only after conflicts are handled.\n\n"
        "HARD CONSTRAINTS:\n"
        "- You MUST start with EXTRACT until you have committed substantive structured facts from "
        "this thread. Do not open with RESOLVE. If you have not yet successfully EXTRACTed non-empty "
        "extracted_facts in a prior turn, your default next action is EXTRACT: populate extracted_facts "
        "with several clear facts supported by the thread.\n"
        "- You MUST use ALIGN only after at least one EXTRACT turn with non-empty extracted_facts. "
        "Do not ALIGN on an empty fact base.\n"
        "- ALIGN OUTPUT REQUIREMENT: You MUST provide a non-empty conflict_alignments mapping. "
        "For every active conflict_id, include one reconciliation statement that explicitly combines "
        "both sides (claim_a and claim_b) into a coherent merged explanation.\n"
        "- If you choose ALIGN and conflict_alignments is empty, your answer is invalid.\n"
        "- You MUST use RESOLVE only after: (1) listed conflicts are cleared or you have just ALIGNed "
        "them away in a prior turn, and (2) you have already extracted enough facts to cover the dispute. "
        "Never RESOLVE while meaningful active conflicts remain unaddressed.\n\n"
        "GUARDS (apply using the user message’s active_conflicts JSON and your prior_action_summaries):\n"
        "- If your planned extracted_facts for this turn would be empty and the thread still contains "
        "usable content, strongly bias toward EXTRACT and fill extracted_facts; do not skip straight "
        "to ALIGN or RESOLVE.\n"
        "- If active_conflicts is non-empty, prefer ALIGN over RESOLVE for this turn.\n"
        "- Read active_conflicts carefully. For each conflict: understand claim_a and claim_b, then "
        "write a merged explanation that resolves both. Always populate conflict_alignments with ALL "
        "active conflict_ids.\n"
        "- If active_conflicts is empty and prior turns show you already EXTRACTed facts, prefer RESOLVE "
        "when a closing summary is appropriate.\n\n"
        "ASK is optional when critical information is missing from the thread; otherwise follow EXTRACT "
        "→ ALIGN → RESOLVE. Do not invent facts not supported by the thread.\n\n"
        "JSON schema for your reply:\n"
        "{\n"
        '  "action_type": "ASK" | "EXTRACT" | "ALIGN" | "RESOLVE",\n'
        '  "content": "string (thread-visible line; question for ASK, optional note for EXTRACT)",\n'
        '  "target_message_id": null or integer,\n'
        '  "extracted_facts": { "fact_key": "fact_value" },\n'
        '  "alignment_summary": "string or null",\n'
        '  "resolution_summary": "string or null",\n'
        '  "conflict_alignments": { "conflict_id": "reconciliation text for ALIGN" }\n'
        "}\n"
        "Example ALIGN payload snippet:\n"
        '{ "conflict_alignments": { "cf_1": "Both users are observing different symptoms of the same issue and these can be true at once..." } }\n'
    )
    user = (
        f"task_id: {task_id}\n"
        f"step_number: {step_n} (max {MAX_STEPS})\n"
        f"system_feedback: {obs.system_feedback!r}\n"
        f"latest_messages (delta): {_messages_to_json(obs.latest_messages)}\n"
        f"full_thread_history: {state_thread_json}\n"
        f"active_conflicts: {active_conflicts_json}\n"
        f"previous_action_summaries: {json.dumps(previous_actions, ensure_ascii=False)}\n"
        "Output the JSON action now."
    )
    return system, user


def _parse_json_object(raw: str) -> dict[str, Any]:
    """
    Extract exactly one JSON object: strip fences, find first '{', then raw_decode
    so nested braces inside strings do not break parsing.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object start '{' in model output")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text[i:])
    if not isinstance(obj, dict):
        raise ValueError("JSON value must be an object at top level")
    return obj


def _compact_action_prefix(compact: str) -> str | None:
    if not compact or compact == "none":
        return None
    for name in ("EXTRACT", "ALIGN", "RESOLVE", "ASK"):
        if compact.startswith(name):
            return name
    return None


def _count_action_prefix(previous_actions: list[str], prefix: str) -> int:
    return sum(1 for a in previous_actions if a.startswith(prefix))


def _forced_action_type(
    previous_actions: list[str],
    conflicting_claims: list[Any],
) -> ConversationRepairActionType:
    """
    Deterministic controller: EXTRACT → ALIGN (while conflicts or pre-RESOLVE shim) → RESOLVE.
    Does not depend on the LLM's stated action_type.
    """
    has_conflicts = len(conflicting_claims) > 0
    n_extract = _count_action_prefix(previous_actions, "EXTRACT")
    n_align = _count_action_prefix(previous_actions, "ALIGN")

    if len(previous_actions) >= 2:
        p_last = _compact_action_prefix(previous_actions[-1])
        p_prev = _compact_action_prefix(previous_actions[-2])
        if p_last and p_last == p_prev:
            if p_last == "EXTRACT":
                return (
                    ConversationRepairActionType.ALIGN
                    if has_conflicts
                    else ConversationRepairActionType.RESOLVE
                )
            if p_last == "ALIGN":
                return ConversationRepairActionType.RESOLVE
            if p_last == "RESOLVE":
                return ConversationRepairActionType.RESOLVE
            if p_last == "ASK":
                return ConversationRepairActionType.EXTRACT

    if n_extract == 0:
        return ConversationRepairActionType.EXTRACT

    if has_conflicts and n_extract >= 1:
        return ConversationRepairActionType.ALIGN

    if not has_conflicts and n_extract >= 1 and n_align >= 1:
        if n_extract >= 2:
            return ConversationRepairActionType.RESOLVE
        return ConversationRepairActionType.EXTRACT

    if not has_conflicts and n_extract >= 1 and n_align == 0:
        return ConversationRepairActionType.ALIGN

    return ConversationRepairActionType.EXTRACT


def _llm_decide_action(
    client: OpenAI,
    model_name: str,
    task_id: str,
    step_n: int,
    obs: ConversationRepairObservation,
    env: ConversationRepairEnvironment,
    previous_actions: list[str],
) -> ConversationRepairAction:
    state = env.state
    forced = _forced_action_type(previous_actions, list(state.conflicting_claims))
    thread_j = _messages_to_json(list(state.thread_history))
    conflicts_j = json.dumps(
        [c.model_dump(mode="json") for c in state.conflicting_claims],
        ensure_ascii=False,
    )
    sys_p, usr_p = _build_policy_prompt(
        task_id, step_n, obs, thread_j, conflicts_j, previous_actions
    )
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_p},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
    )
    text = (resp.choices[0].message.content or "").strip()
    data = _parse_json_object(text)
    return ConversationRepairAction(
        action_type=forced,
        content=str(data.get("content") or ""),
        target_message_id=data.get("target_message_id"),
        extracted_facts=dict(data.get("extracted_facts") or {}),
        alignment_summary=data.get("alignment_summary"),
        resolution_summary=data.get("resolution_summary"),
        conflict_alignments={
            str(k): str(v) for k, v in dict(data.get("conflict_alignments") or {}).items()
        },
    )


def _print_step(
    step_n: int,
    action_str: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    action_str = _single_line_compact(action_str)
    if error is None:
        err = "null"
    else:
        err = _single_line_compact(error)
        if len(err) > 240:
            err = err[:237] + "..."
    rw = max(-1e9, min(1e9, float(reward)))
    print(
        f"[STEP] step={step_n} action={action_str} reward={rw:.2f} "
        f"done={str(done).lower()} error={err}"
    )


def _print_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={ENV_NAME} model={model_name}")


def _print_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    sc = max(0.0, min(1.0, float(score)))
    print(
        f"[END] success={str(success).lower()} steps={steps} score={sc:.2f} rewards={_format_reward_list(rewards)}"
    )


def run_episode(
    env: ConversationRepairEnvironment,
    client: OpenAI,
    model_name: str,
    task_id: str,
) -> tuple[bool, int, list[float], str | None]:
    """
    Runs up to MAX_STEPS. Returns (success, steps_taken, rewards, fatal_error).
    """
    rewards: list[float] = []
    prev_summaries: list[str] = []
    fatal: str | None = None
    success = False
    steps_taken = 0

    try:
        obs = env.reset(task_id=task_id)
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
        return False, 0, [], f"reset_failed:{e!s}"

    for step_n in range(1, MAX_STEPS + 1):
        if obs.done:
            break
        action_str = "none"
        step_reward = 0.0
        step_done = obs.done
        step_err: str | None = None
        llm_action: ConversationRepairAction | None = None
        try:
            try:
                llm_action = _llm_decide_action(
                    client, model_name, task_id, step_n, obs, env, prev_summaries
                )
                step_err = None
            except Exception as e:
                llm_action = None
                step_err = str(e)

            if llm_action is None:
                rewards.append(0.0)
                steps_taken = step_n
                _print_step(step_n, "none", step_reward, False, step_err)
                break

            action = llm_action
            action_str = _compact_action_log(action)
            obs = env.step(action)
            step_reward = float(obs.reward) if obs.reward is not None else 0.0
            step_done = bool(obs.done)
            rewards.append(step_reward)
            steps_taken = step_n
            prev_summaries.append(action_str)
            if obs.is_resolved:
                success = True
        except Exception as e:
            step_err = f"{type(e).__name__}:{e!s}"
            rewards.append(0.0)
            steps_taken = step_n
            fatal = traceback.format_exc()
            _print_step(step_n, action_str, step_reward, False, step_err)
            break

        _print_step(step_n, action_str, step_reward, step_done, None)

        if step_done:
            break

    return success, steps_taken, rewards, fatal


def main() -> int:
    try:
        api_base = API_BASE_URL
        hf_token = HF_TOKEN
        model_name = MODEL_NAME
        if not hf_token:
            raise RuntimeError("Missing required environment variable: HF_TOKEN")
    except Exception as e:
        print("[END] success=false steps=0 score=0.00 rewards=")
        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
        return 1

    client = OpenAI(base_url=api_base, api_key=hf_token)

    env = ConversationRepairEnvironment()

    for task_id in list_task_ids():
        _print_start(task_name=task_id, model_name=model_name)
        success, steps_taken, rewards, fatal_err = run_episode(
            env, client, model_name, task_id
        )
        total = sum(rewards)
        score = _normalize_score(total, steps_taken)
        _print_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        if fatal_err:
            sys.stderr.write(fatal_err)
            if not fatal_err.endswith("\n"):
                sys.stderr.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
