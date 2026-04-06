"""
Conversation Repair Environment Client.

Serializes structured actions and parses structured observations/state.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ConversationRepairAction,
    ConversationRepairObservation,
    ConversationRepairState,
    ThreadMessage,
)


class ConversationRepairEnv(
    EnvClient[
        ConversationRepairAction,
        ConversationRepairObservation,
        ConversationRepairState,
    ]
):
    def _step_payload(self, action: ConversationRepairAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[ConversationRepairObservation]:
        obs_data = payload.get("observation", {})
        raw_messages = obs_data.get("latest_messages") or []
        messages = [ThreadMessage.model_validate(m) for m in raw_messages]

        observation = ConversationRepairObservation(
            latest_messages=messages,
            system_feedback=obs_data.get("system_feedback", ""),
            is_resolved=obs_data.get("is_resolved", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ConversationRepairState:
        return ConversationRepairState.model_validate(payload)
