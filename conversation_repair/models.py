# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models for the Context-Aware Conversation Repair System.

Design goals (hackathon / blueprint):
- Represent ambiguity, conflicting claims, and extracted facts explicitly.
- Structured action space: ASK, EXTRACT, ALIGN, RESOLVE (multi-step reasoning).
- Observations expose deltas (`latest_messages`) plus hidden `system_feedback`.
- State is the source of truth for grading later; avoid encoding task logic in
  free-text fields alone—use typed structures for facts and conflicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ConversationRepairActionType(str, Enum):
    """High-level repair strategy for one turn."""

    ASK = "ASK"
    EXTRACT = "EXTRACT"
    ALIGN = "ALIGN"
    RESOLVE = "RESOLVE"


class ThreadMessage(BaseModel):
    """One message in the messy thread (immutable once stored)."""

    model_config = {"extra": "forbid"}

    message_id: int = Field(..., description="Stable id within the episode thread")
    sender: str = Field(..., description="Logical speaker label (user, agent, system, …)")
    timestamp_iso: str = Field(
        ...,
        description="ISO-8601 timestamp string for ordering and auditability",
    )
    content: str = Field(..., description="Raw message text")


class AmbiguityItem(BaseModel):
    """An identified gap or underspecified aspect of the thread."""

    model_config = {"extra": "forbid"}

    ambiguity_id: str = Field(..., description="Stable id for this ambiguity")
    description: str = Field(..., description="What is unclear or missing")
    related_message_ids: list[int] = Field(
        default_factory=list,
        description="Thread messages that contribute to this ambiguity",
    )


class ConflictRecord(BaseModel):
    """Structured representation of incompatible claims (not free-text matching)."""

    model_config = {"extra": "forbid"}

    conflict_id: str = Field(..., description="Stable id for this conflict")
    topic: str = Field(..., description="What the parties disagree about")
    party_a: str = Field(..., description="Label for first position")
    claim_a: str = Field(..., description="First incompatible claim")
    party_b: str = Field(..., description="Label for second position")
    claim_b: str = Field(..., description="Second incompatible claim")
    related_message_ids: list[int] = Field(
        default_factory=list,
        description="Messages that support or state each side",
    )


class ConversationRepairAction(Action):
    """
    Structured agent action.

    - ASK: ask a targeted question (`content`); optional `target_message_id`.
    - EXTRACT: commit structured facts in `extracted_facts` (merged by key in the env).
    - ALIGN: `conflict_alignments` scored for both-sides coverage vs `ConflictRecord` claims/parties
      plus similarity to reference alignment text (deterministic semantic rubric).
    - RESOLVE: graded with semantic similarity to expected resolution and overlap with rubric facts.
    """

    action_type: ConversationRepairActionType = Field(
        ...,
        description="Which repair operation this turn performs",
    )
    content: str = Field(
        default="",
        description="Text posted to the thread (question, summary, or resolution)",
    )
    target_message_id: int | None = Field(
        default=None,
        description="Optional reply anchor in the thread",
    )
    extracted_facts: dict[str, str] = Field(
        default_factory=dict,
        description="For EXTRACT: canonical fact_key -> normalized value strings",
    )
    alignment_summary: str | None = Field(
        default=None,
        description="For ALIGN: concise reconciliation of conflicting views",
    )
    resolution_summary: str | None = Field(
        default=None,
        description="For RESOLVE: structured final resolution text",
    )
    conflict_alignments: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "For ALIGN: maps active conflict_id -> canonical reconciliation statement; "
            "compared to task ground truth (normalized equality), not keyword search."
        ),
    )


class TaskGroundTruth(BaseModel):
    """
    Hidden task specification for deterministic grading.

    Not exposed on observations; the environment holds this and compares structured
    agent outputs using deterministic semantic rubrics (see semantic_rubric.py).
    """

    model_config = {"extra": "forbid"}

    task_id: str = Field(..., description="Stable scenario identifier")
    facts: dict[str, str] = Field(
        ...,
        description="Required fact keys and canonical values the agent must extract",
    )
    conflict_alignments: dict[str, str] = Field(
        ...,
        description="For each resolvable conflict_id, reference alignment text",
    )
    expected_resolution: str = Field(
        ...,
        description="Canonical final resolution the agent must emit on RESOLVE",
    )
    forbidden_fact_cooccurrence: list[tuple[str, str]] = Field(
        default_factory=list,
        description=(
            "Optional key pairs: if both appear in extracted_context, apply a contradiction "
            "penalty (task-specific mutual exclusion)."
        ),
    )


class FactEvaluationResult(BaseModel):
    """
    Deterministic comparison of extracted_context against task ground-truth facts.

    - ``partial_value_keys``: rubric keys present with moderate semantic similarity.
    - ``wrong_value_keys``: rubric keys present but semantically too far from reference.
    - ``spurious_keys``: keys not in the rubric (unsupported extractions).
    """

    model_config = {"extra": "forbid"}

    correct_keys: list[str] = Field(default_factory=list)
    partial_value_keys: list[str] = Field(default_factory=list)
    missing_keys: list[str] = Field(default_factory=list)
    wrong_value_keys: list[str] = Field(default_factory=list)
    spurious_keys: list[str] = Field(default_factory=list)
    semantic_similarity_by_key: dict[str, float] = Field(
        default_factory=dict,
        description="content_similarity score vs rubric for each required key (0 if missing)",
    )


class ConversationRepairObservation(Observation):
    """
    What the agent sees after reset or step.

    `latest_messages` is the delta since the last observation (reset returns the
    initial slice). `system_feedback` is simulator-only context (drops, latency, etc.).
    """

    latest_messages: list[ThreadMessage] = Field(
        default_factory=list,
        description="New thread messages since the previous step",
    )
    system_feedback: str = Field(
        default="",
        description="Hidden environment feedback (not part of the user thread)",
    )
    is_resolved: bool = Field(
        default=False,
        description="Whether the episode objective is satisfied",
    )


class ConversationRepairState(State):
    """
    Full episodic state for debugging, logging, and deterministic grading.

    Inherits OpenEnv `episode_id` and `step_count`. `turn_count` mirrors
    `step_count` for blueprint readability.
    """

    turn_count: int = Field(
        default=0,
        ge=0,
        description="Number of completed env steps (mirrors step_count)",
    )
    thread_history: list[ThreadMessage] = Field(
        default_factory=list,
        description="Full ordered thread",
    )
    extracted_context: dict[str, str] = Field(
        default_factory=dict,
        description="Merged structured facts extracted so far",
    )
    unresolved_ambiguities: list[AmbiguityItem] = Field(
        default_factory=list,
        description="Ambiguities not yet cleared",
    )
    conflicting_claims: list[ConflictRecord] = Field(
        default_factory=list,
        description="Active conflicts between participants",
    )
    last_public_agent_text: str = Field(
        default="",
        description="Last agent-visible line added to the thread",
    )
    episode_flags: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured progress flags and episode-local counters for grading",
    )
    last_fact_evaluation: FactEvaluationResult | None = Field(
        default=None,
        description="Latest deterministic fact audit vs ground truth (updated each step)",
    )
    last_reward_components: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of the most recent step reward (for debugging / rubrics)",
    )
    last_semantic_trace: dict[str, Any] = Field(
        default_factory=dict,
        description="Latest semantic scoring breakdown (facts, alignment, resolution slices)",
    )
