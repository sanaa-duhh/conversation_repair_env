"""
`conversation_repair` environment: multi-step repair with deterministic semantic rubrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ConversationRepairAction,
        ConversationRepairActionType,
        ConversationRepairObservation,
        ConversationRepairState,
        ConflictRecord,
        FactEvaluationResult,
        TaskGroundTruth,
        ThreadMessage,
    )
    from ..semantic_rubric import (
        T_FACT_CORRECT,
        T_FACT_PARTIAL,
        T_RESOLUTION_OK,
        T_RES_FACT_MEAN,
        alignment_semantic_eval,
        content_similarity,
        evaluate_fact_matrix,
        fact_scores_to_evaluation_lists,  # ✅ IMPORTANT FIX
        forbidden_cooccurrence_hits,
        normalize_text as _normalize_text,
        pairwise_extracted_contradictions,
        resolution_contradicts_extracted,
        resolution_semantic_eval,
        similarity_dict,
    )
    from ..tasks import DEFAULT_TASK_ID, get_task

except ImportError:
    from models import (  # type: ignore[no-redef]
        ConversationRepairAction,
        ConversationRepairActionType,
        ConversationRepairObservation,
        ConversationRepairState,
        ConflictRecord,
        FactEvaluationResult,
        TaskGroundTruth,
        ThreadMessage,
    )
    from semantic_rubric import (  # type: ignore[no-redef]
        T_FACT_CORRECT,
        T_FACT_PARTIAL,
        T_RESOLUTION_OK,
        T_RES_FACT_MEAN,
        alignment_semantic_eval,
        content_similarity,
        evaluate_fact_matrix,
        fact_scores_to_evaluation_lists,  # ✅ IMPORTANT FIX
        forbidden_cooccurrence_hits,
        normalize_text as _normalize_text,
        pairwise_extracted_contradictions,
        resolution_contradicts_extracted,
        resolution_semantic_eval,
        similarity_dict,
    )
    from tasks import DEFAULT_TASK_ID, get_task  # type: ignore[no-redef]
# --- Episode limits ---
_MAX_EPISODE_STEPS = 25

# --- Reward coefficients (step total clamped to [-1, 1]) ---
_BASE_STEP_COST = -0.015

# Information gain when a rubric fact crosses into a higher semantic band vs ground truth.
_INFO_GAIN_FULL = 0.12
_INFO_GAIN_PARTIAL = 0.055

_PENALTY_WRONG_OR_SPURIOUS_KEY = 0.08
_PENALTY_INCORRECT_FACT = 0.06

_PENALTY_REDUNDANT_ACTION = 0.07
_PENALTY_REDUNDANT_FACT_PAIR = 0.05

_REWARD_CONFLICT_FULL = 0.22
_REWARD_CONFLICT_PARTIAL = 0.09
_PENALTY_ALIGN_WEAK = 0.06
_PENALTY_UNKNOWN_CONFLICT_ID = 0.04

_REWARD_SUCCESSFUL_RESOLUTION = 1.0
_PENALTY_FAILED_RESOLUTION = 0.12

# Contradictions between extracted rubric facts (negation heuristic + optional forbidden pairs).
_PENALTY_CONTRADICTION_UNIT = 0.06
_MAX_CONTRADICTION_PENALTY = 0.15


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _action_fingerprint(action: ConversationRepairAction) -> str:
    """Stable hash of what the agent did this turn (for redundancy detection)."""
    ef = {k: _normalize_text(v) for k, v in sorted(action.extracted_facts.items())}
    ca = {k: _normalize_text(v) for k, v in sorted(action.conflict_alignments.items())}
    payload = (
        action.action_type.value,
        _normalize_text(action.content),
        action.target_message_id,
        json.dumps(ef, sort_keys=True),
        json.dumps(ca, sort_keys=True),
        _normalize_text(action.alignment_summary or ""),
        _normalize_text(action.resolution_summary or ""),
    )
    return json.dumps(payload, sort_keys=True)


def _conflict_by_id(conflicts: list[ConflictRecord], cid: str) -> ConflictRecord | None:
    for c in conflicts:
        if c.conflict_id == cid:
            return c
    return None


class ConversationRepairEnvironment(Environment):
    """Multi-turn conversation repair with deterministic semantic scoring."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state: ConversationRepairState = self._initial_state()
        self._last_observed_len: int = 0
        self._ground_truth: TaskGroundTruth = get_task(DEFAULT_TASK_ID).ground_truth
        self._last_action_fingerprint: str | None = None
        self._active_task_id: str = DEFAULT_TASK_ID

    def _initial_state(self) -> ConversationRepairState:
        return ConversationRepairState(
            episode_id=str(uuid4()),
            step_count=0,
            turn_count=0,
            thread_history=[],
            extracted_context={},
            unresolved_ambiguities=[],
            conflicting_claims=[],
            last_public_agent_text="",
            episode_flags={},
            last_fact_evaluation=None,
            last_reward_components={},
            last_semantic_trace={},
        )

    def _apply_task_bundle(self, task_id: str) -> None:
        bundle = get_task(task_id)
        self._ground_truth = bundle.ground_truth
        self._active_task_id = bundle.ground_truth.task_id
        s = self._state
        s.thread_history = list(bundle.thread_messages)
        s.unresolved_ambiguities = list(bundle.unresolved_ambiguities)
        s.conflicting_claims = list(bundle.conflicting_claims)

    def _fact_eval_from_context(self, ctx: dict[str, str]) -> tuple[FactEvaluationResult, dict]:
        scores, breakdown = evaluate_fact_matrix(ctx, self._ground_truth)
        lists = fact_scores_to_evaluation_lists(scores, self._ground_truth)
        ev = FactEvaluationResult(
            correct_keys=lists["correct_keys"],
            partial_value_keys=lists["partial_value_keys"],
            missing_keys=lists["missing_keys"],
            wrong_value_keys=lists["wrong_value_keys"],
            spurious_keys=lists["spurious_keys"],
            semantic_similarity_by_key=similarity_dict(scores, self._ground_truth),
        )
        breakdown["classification"] = {
            "correct_keys": ev.correct_keys,
            "partial_value_keys": ev.partial_value_keys,
            "wrong_value_keys": ev.wrong_value_keys,
            "missing_keys": ev.missing_keys,
            "spurious_keys": ev.spurious_keys,
        }
        return ev, breakdown

    def _apply_contradiction_penalties(self, ctx: dict[str, str]) -> tuple[float, dict]:
        """Negation-style clashes + task-level forbidden co-occurrence (deterministic)."""
        rubric_keys = frozenset(self._ground_truth.facts.keys())
        neg_pairs = pairwise_extracted_contradictions(ctx, rubric_keys)
        forbid = forbidden_cooccurrence_hits(
            ctx, self._ground_truth.forbidden_fact_cooccurrence
        )
        n = len(neg_pairs) + len(forbid)
        raw = _PENALTY_CONTRADICTION_UNIT * n
        pen = min(_MAX_CONTRADICTION_PENALTY, raw)
        detail = {
            "negation_style_pairs": [list(p) for p in neg_pairs],
            "forbidden_cooccurrence_hits": [list(p) for p in forbid],
            "penalty_applied": pen,
        }
        return -pen, detail

    def _resolution_semantically_valid(self, action: ConversationRepairAction) -> bool:
        if action.action_type != ConversationRepairActionType.RESOLVE:
            return False
        if self._state.conflicting_claims:
            return False
        ev, _ = self._fact_eval_from_context(self._state.extracted_context)
        if ev.spurious_keys or ev.missing_keys or ev.wrong_value_keys:
            return False
        res_text = (action.resolution_summary or action.content or "").strip()
        if not res_text:
            return False
        rs = resolution_semantic_eval(res_text, self._ground_truth, self._state.extracted_context)
        if rs.resolution_vs_expected < T_RESOLUTION_OK:
            return False
        if rs.mean_resolution_vs_gt_facts < T_RES_FACT_MEAN:
            return False
        if not rs.all_rubric_keys_present_partial_or_better:
            return False
        if resolution_contradicts_extracted(
            res_text, self._state.extracted_context, self._ground_truth
        ):
            return False
        return True

    def _merge_extracted(self, incoming: dict[str, str]) -> None:
        for k, v in incoming.items():
            key = k.strip()
            if not key:
                continue
            self._state.extracted_context[key] = v.strip()

    def _apply_step_reward_logic(
        self, action: ConversationRepairAction
    ) -> tuple[float, str, str]:
        components: dict[str, float] = {"base_step": _BASE_STEP_COST}
        reward = _BASE_STEP_COST
        feedback_parts: list[str] = []
        semantic_trace: dict = {"step_type": action.action_type.value}

        fp = _action_fingerprint(action)
        if self._last_action_fingerprint is not None and fp == self._last_action_fingerprint:
            components["redundant_action"] = -_PENALTY_REDUNDANT_ACTION
            reward -= _PENALTY_REDUNDANT_ACTION
            feedback_parts.append("Redundant: identical action fingerprint as prior step.")
        self._last_action_fingerprint = fp

        if action.action_type == ConversationRepairActionType.ASK:
            self._append_agent_message(action.content)
            feedback_parts.append("Clarification recorded (user simulator not wired yet).")

        elif action.action_type == ConversationRepairActionType.EXTRACT:
            ctx_before = dict(self._state.extracted_context)
            incoming = {
                k.strip(): v.strip()
                for k, v in action.extracted_facts.items()
                if k.strip()
            }

            incorrect_batch = 0
            redundant_pairs = 0
            info_full = 0
            info_partial = 0
            fact_batch_trace: list[dict] = []

            for key, raw_val in incoming.items():
                prior = ctx_before.get(key)
                prior_norm = prior.strip() if prior else ""

                # Semantic redundancy: near-duplicate paraphrase of existing value.
                if prior_norm and content_similarity(raw_val, prior_norm) >= 0.91:
                    redundant_pairs += 1
                    components["redundant_fact_pair"] = components.get(
                        "redundant_fact_pair", 0.0
                    ) - _PENALTY_REDUNDANT_FACT_PAIR
                    reward -= _PENALTY_REDUNDANT_FACT_PAIR
                    continue

                if key in self._ground_truth.facts:
                    ref = self._ground_truth.facts[key]
                    sim_new = content_similarity(raw_val, ref)
                    sim_old = content_similarity(prior_norm, ref) if prior_norm else 0.0
                    fact_batch_trace.append(
                        {
                            "key": key,
                            "similarity_new": round(sim_new, 4),
                            "similarity_old": round(sim_old, 4),
                        }
                    )

                    if sim_new < T_FACT_PARTIAL:
                        incorrect_batch += 1
                        components["incorrect_extraction"] = components.get(
                            "incorrect_extraction", 0.0
                        ) - _PENALTY_INCORRECT_FACT
                        reward -= _PENALTY_INCORRECT_FACT
                    else:
                        crossed_full = sim_new >= T_FACT_CORRECT and sim_old < T_FACT_CORRECT
                        crossed_partial = sim_new >= T_FACT_PARTIAL and sim_old < T_FACT_PARTIAL
                        if crossed_full:
                            info_full += 1
                            components["information_gain_full"] = (
                                components.get("information_gain_full", 0.0) + _INFO_GAIN_FULL
                            )
                            reward += _INFO_GAIN_FULL
                        elif crossed_partial and sim_new < T_FACT_CORRECT:
                            info_partial += 1
                            components["information_gain_partial"] = (
                                components.get("information_gain_partial", 0.0)
                                + _INFO_GAIN_PARTIAL
                            )
                            reward += _INFO_GAIN_PARTIAL
                else:
                    incorrect_batch += 1
                    components["spurious_extraction"] = components.get(
                        "spurious_extraction", 0.0
                    ) - _PENALTY_WRONG_OR_SPURIOUS_KEY
                    reward -= _PENALTY_WRONG_OR_SPURIOUS_KEY

            self._merge_extracted(action.extracted_facts)
            self._append_agent_message(action.content)

            cpen, cdetail = self._apply_contradiction_penalties(self._state.extracted_context)
            if cpen < 0:
                components["contradiction_penalty"] = cpen
                reward += cpen
                feedback_parts.append("Contradiction penalty on extracted facts.")
            semantic_trace["extract_batch"] = fact_batch_trace
            semantic_trace["contradictions"] = cdetail

            if info_full:
                feedback_parts.append(
                    f"Semantic information gain (strong): {info_full} fact(s) crossed full-correct band."
                )
            if info_partial:
                feedback_parts.append(
                    f"Semantic information gain (partial): {info_partial} fact(s) entered partial band."
                )
            if incorrect_batch:
                feedback_parts.append(
                    f"Extraction issues: {incorrect_batch} weak or unsupported key(s) in batch."
                )
            if redundant_pairs:
                feedback_parts.append(f"{redundant_pairs} near-duplicate fact(s) re-submitted.")

        elif action.action_type == ConversationRepairActionType.ALIGN:
            active_ids = {c.conflict_id for c in self._state.conflicting_claims}
            align_traces: dict[str, object] = {}

            for cid, stmt in action.conflict_alignments.items():
                norm_stmt = _normalize_text(stmt)
                if not norm_stmt:
                    continue
                if cid not in active_ids:
                    components[f"unknown_conflict_{cid}"] = -_PENALTY_UNKNOWN_CONFLICT_ID
                    reward -= _PENALTY_UNKNOWN_CONFLICT_ID
                    feedback_parts.append(f"No active conflict '{cid}' to align.")
                    continue

                ref_align = self._ground_truth.conflict_alignments.get(cid)
                if ref_align is None:
                    components[f"ungraded_conflict_{cid}"] = -_PENALTY_ALIGN_WEAK
                    reward -= _PENALTY_ALIGN_WEAK
                    feedback_parts.append(f"Conflict '{cid}' has no rubric entry.")
                    continue

                record = _conflict_by_id(self._state.conflicting_claims, cid)
                assert record is not None
                ae = alignment_semantic_eval(stmt, record, ref_align)
                align_traces[cid] = ae.model_dump()

                if ae.verdict == "full":
                    self._state.conflicting_claims = [
                        c for c in self._state.conflicting_claims if c.conflict_id != cid
                    ]
                    active_ids.discard(cid)
                    components["conflict_understanding_full"] = (
                        components.get("conflict_understanding_full", 0.0) + _REWARD_CONFLICT_FULL
                    )
                    reward += _REWARD_CONFLICT_FULL
                    feedback_parts.append(
                        f"Conflict '{cid}' reconciled (both sides + rubric similarity)."
                    )
                elif ae.verdict == "partial":
                    components["conflict_understanding_partial"] = (
                        components.get("conflict_understanding_partial", 0.0)
                        + _REWARD_CONFLICT_PARTIAL
                    )
                    reward += _REWARD_CONFLICT_PARTIAL
                    feedback_parts.append(
                        f"Conflict '{cid}' partially addressed (see semantic_trace)."
                    )
                else:
                    components["align_weak"] = components.get("align_weak", 0.0) - _PENALTY_ALIGN_WEAK
                    reward -= _PENALTY_ALIGN_WEAK
                    feedback_parts.append(
                        f"Alignment for '{cid}' too weak: does not cover both positions adequately."
                    )

            semantic_trace["alignments"] = align_traces

            first_stmt = ""
            if action.conflict_alignments:
                first_stmt = next(iter(action.conflict_alignments.values()), "")
            public_line = (
                action.content
                or (action.alignment_summary or "").strip()
                or first_stmt.strip()
            )
            self._append_agent_message(public_line)
            if not action.conflict_alignments:
                feedback_parts.append(
                    "No structured conflict_alignments provided; state unchanged."
                )

        elif action.action_type == ConversationRepairActionType.RESOLVE:
            res_text = action.content or action.resolution_summary or ""
            self._append_agent_message(res_text)
            rs = resolution_semantic_eval(
                res_text, self._ground_truth, self._state.extracted_context
            )
            semantic_trace["resolution_eval"] = rs.model_dump()
            semantic_trace["resolution_contradicts_extracted"] = (
                resolution_contradicts_extracted(
                    res_text, self._state.extracted_context, self._ground_truth
                )
            )

            if self._resolution_semantically_valid(action):
                components["resolution_quality"] = _REWARD_SUCCESSFUL_RESOLUTION
                reward += _REWARD_SUCCESSFUL_RESOLUTION
                feedback_parts.append(
                    "Episode resolved: semantic match to rubric, facts coherent, conflicts cleared."
                )
            else:
                components["resolution_quality"] = -_PENALTY_FAILED_RESOLUTION
                reward -= _PENALTY_FAILED_RESOLUTION
                ev, _ = self._fact_eval_from_context(self._state.extracted_context)
                reasons: list[str] = []
                if self._state.conflicting_claims:
                    reasons.append("conflicts remain")
                if ev.missing_keys:
                    reasons.append(f"missing facts: {ev.missing_keys}")
                if ev.wrong_value_keys:
                    reasons.append(f"incorrect fact semantics: {ev.wrong_value_keys}")
                if ev.spurious_keys:
                    reasons.append(f"spurious facts: {ev.spurious_keys}")
                if rs.resolution_vs_expected < T_RESOLUTION_OK:
                    reasons.append(
                        f"resolution too far from expected (sim={rs.resolution_vs_expected})"
                    )
                if rs.mean_resolution_vs_gt_facts < T_RES_FACT_MEAN:
                    reasons.append("resolution does not reference rubric facts strongly enough")
                if not rs.all_rubric_keys_present_partial_or_better:
                    reasons.append("rubric facts not all in partial-or-better band")
                if semantic_trace["resolution_contradicts_extracted"]:
                    reasons.append("resolution semantically disconnected from extracted facts")
                feedback_parts.append("Resolution rejected: " + "; ".join(reasons) + ".")

        else:
            feedback_parts.append("Unknown action type.")

        self._state.last_reward_components = components
        self._state.last_semantic_trace = semantic_trace
        clamped = max(-1.0, min(1.0, reward))
        return clamped, " ".join(feedback_parts).strip(), "step"

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: object,
    ) -> ConversationRepairObservation:
        _ = seed
        self._state = self._initial_state()
        if episode_id is not None:
            self._state.episode_id = episode_id
        self._last_action_fingerprint = None

        tid_kw = kwargs.get("task_id", task_id)
        tid = DEFAULT_TASK_ID if tid_kw is None or tid_kw == "" else str(tid_kw)
        self._apply_task_bundle(tid)
        canonical_task_id = self._active_task_id

        self._last_observed_len = 0

        ev, fb = self._fact_eval_from_context(self._state.extracted_context)
        self._state.last_fact_evaluation = ev
        self._state.last_semantic_trace = {
            "phase": "post_reset",
            "facts": fb,
        }

        latest = list(self._state.thread_history[self._last_observed_len :])
        self._last_observed_len = len(self._state.thread_history)

        return ConversationRepairObservation(
            latest_messages=latest,
            system_feedback=(
                f"Episode started (task_id={canonical_task_id}). Facts and resolution are graded "
                "with deterministic semantic similarity (see metadata.semantic_trace)."
            ),
            is_resolved=False,
            done=False,
            reward=0.0,
            metadata={
                "phase": "post_reset",
                "task_id": canonical_task_id,
                "semantic_trace": self._state.last_semantic_trace,
            },
        )

    def _next_message_id(self) -> int:
        if not self._state.thread_history:
            return 1
        return max(m.message_id for m in self._state.thread_history) + 1

    def _append_agent_message(self, text: str) -> None:
        if not text.strip():
            return
        msg = ThreadMessage(
            message_id=self._next_message_id(),
            sender="agent",
            timestamp_iso=_utc_now_iso(),
            content=text.strip(),
        )
        self._state.thread_history.append(msg)
        self._state.last_public_agent_text = msg.content

    def step(
        self,
        action: ConversationRepairAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> ConversationRepairObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._state.turn_count = self._state.step_count

        reward, system_feedback, info_phase = self._apply_step_reward_logic(action)

        ev, fact_breakdown = self._fact_eval_from_context(self._state.extracted_context)
        self._state.last_fact_evaluation = ev

        is_resolved = (
            action.action_type == ConversationRepairActionType.RESOLVE
            and self._resolution_semantically_valid(action)
        )
        if is_resolved:
            info_phase = "resolved"

        done = is_resolved or self._state.step_count >= _MAX_EPISODE_STEPS

        latest = list(self._state.thread_history[self._last_observed_len :])
        self._last_observed_len = len(self._state.thread_history)

        merged_trace = {
            "global_fact_breakdown": fact_breakdown,
            "step": self._state.last_semantic_trace,
        }

        meta = {
            "phase": info_phase,
            "step": self._state.step_count,
            "fact_evaluation": self._state.last_fact_evaluation.model_dump(),
            "reward_components": self._state.last_reward_components,
            "semantic_trace": merged_trace,
        }

        return ConversationRepairObservation(
            latest_messages=latest,
            system_feedback=system_feedback,
            is_resolved=is_resolved,
            done=done,
            reward=reward,
            metadata=meta,
        )

    @property
    def state(self) -> ConversationRepairState:  # type: ignore[override]
        return self._state
