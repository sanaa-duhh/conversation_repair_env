# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Structured evaluation scenarios: messy threads + hidden rubrics. Timestamps are
# fixed strings so episodes are reproducible without RNG.

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, Field, model_validator

from .models import (
    AmbiguityItem,
    ConflictRecord,
    TaskGroundTruth,
    ThreadMessage,
)

# Default when reset() is called without task_id (backward compatible).
DEFAULT_TASK_ID = "ui_backend_latency"


class RepairTaskScenario(BaseModel):
    """
    Full initial episode specification: what the agent sees in state versus what
    the grader expects in TaskGroundTruth.
    """

    model_config = {"extra": "forbid"}

    task_id: str = Field(..., description="Identifier passed to reset(task_id=...).")
    thread_messages: list[ThreadMessage] = Field(
        ...,
        description="Initial thread, ordered; may include noise and contradictions.",
    )
    unresolved_ambiguities: list[AmbiguityItem] = Field(default_factory=list)
    conflicting_claims: list[ConflictRecord] = Field(default_factory=list)
    ground_truth: TaskGroundTruth = Field(..., description="Hidden rubric for this thread.")

    @model_validator(mode="after")
    def _task_ids_match(self) -> RepairTaskScenario:
        if self.ground_truth.task_id != self.task_id:
            raise ValueError(
                f"ground_truth.task_id ({self.ground_truth.task_id!r}) != task_id ({self.task_id!r})"
            )
        return self


class TaskBundle(NamedTuple):
    """Return type for get_task: rubric plus everything needed to populate state."""

    ground_truth: TaskGroundTruth
    thread_messages: list[ThreadMessage]
    unresolved_ambiguities: list[AmbiguityItem]
    conflicting_claims: list[ConflictRecord]


# Deterministic ISO timestamps (ordering by message_id; no wall-clock calls).
def _ts(i: int) -> str:
    return f"2026-04-05T12:{i:02d}:00+00:00"


def _scenario_ui_backend_latency() -> RepairTaskScenario:
    """Two engineers attribute the same outage to different layers; urgency is real."""
    return RepairTaskScenario(
        task_id="ui_backend_latency",
        thread_messages=[
            ThreadMessage(
                message_id=1,
                sender="user_a",
                timestamp_iso=_ts(0),
                content=(
                    "The app is unusable — everything freezes when I open the dashboard. "
                    "Pretty sure it's our front-end bundle."
                ),
            ),
            ThreadMessage(
                message_id=2,
                sender="user_b",
                timestamp_iso=_ts(1),
                content=(
                    "No, the problem is the backend — requests take 30s before anything renders. "
                    "I already told you that yesterday."
                ),
            ),
            ThreadMessage(
                message_id=3,
                sender="user_a",
                timestamp_iso=_ts(2),
                content="We need this fixed today; it's blocking the whole team.",
            ),
            ThreadMessage(
                message_id=4,
                sender="intern_bot",
                timestamp_iso=_ts(3),
                content=(
                    "Unrelated but the snack channel says the espresso machine is broken again."
                ),
            ),
        ],
        unresolved_ambiguities=[
            AmbiguityItem(
                ambiguity_id="amb_failure_signature",
                description="Is the primary symptom client-side freeze, server latency, or both?",
                related_message_ids=[1, 2],
            ),
            AmbiguityItem(
                ambiguity_id="amb_blame_narrative",
                description="Participants give incompatible locus-of-failure stories.",
                related_message_ids=[1, 2, 3],
            ),
        ],
        conflicting_claims=[
            ConflictRecord(
                conflict_id="cf_ui_vs_api",
                topic="Primary failure locus",
                party_a="user_a",
                claim_a="Front-end / UI freeze when loading dashboard.",
                party_b="user_b",
                claim_b="Backend requests ~30s before UI updates.",
                related_message_ids=[1, 2],
            )
        ],
        ground_truth=TaskGroundTruth(
            task_id="ui_backend_latency",
            facts={
                "user_a_reported_symptom": (
                    "client-side ui freeze when opening the dashboard"
                ),
                "user_b_reported_symptom": (
                    "server-side latency of about 30 seconds before the ui updates"
                ),
                "business_constraint": (
                    "same-day fix required; issue blocks the entire team"
                ),
            },
            conflict_alignments={
                "cf_ui_vs_api": (
                    "the dashboard freeze and slow server responses are consistent: "
                    "long backend response times can present as a frozen user interface"
                ),
            },
            expected_resolution=(
                "treat as an end-to-end latency problem: validate backend request timing "
                "first while correlating with client-reported dashboard freezes; "
                "prioritize a same-day mitigation because the team is blocked"
            ),
        ),
    )


def _scenario_deploy_database_dispute() -> RepairTaskScenario:
    """On-call thread: vague 'database error', DNS red herring, DBA gives concrete signal."""
    return RepairTaskScenario(
        task_id="deploy_database_dispute",
        thread_messages=[
            ThreadMessage(
                message_id=1,
                sender="release_captain",
                timestamp_iso=_ts(10),
                content=(
                    "Red deploy failed with a database error — whole release train is stopped. "
                    "Need someone to own root cause in the next hour."
                ),
            ),
            ThreadMessage(
                message_id=2,
                sender="infra_lisa",
                timestamp_iso=_ts(11),
                content=(
                    "Seeing flaky resolver alerts in that AZ — I'd bet it's DNS propagation, "
                    "not the app. Re-run after cache TTL."
                ),
            ),
            ThreadMessage(
                message_id=3,
                sender="dba_marc",
                timestamp_iso=_ts(12),
                content=(
                    "It's not DNS. Payments postgres replica is throwing 53300 — connection "
                    "pool exhausted. Deploy hammered new connections."
                ),
            ),
            ThreadMessage(
                message_id=4,
                sender="pm_alex",
                timestamp_iso=_ts(13),
                content="Leadership wants a go/no-go in two hours for the marketing launch.",
            ),
            ThreadMessage(
                message_id=5,
                sender="vendor_sales",
                timestamp_iso=_ts(14),
                content=(
                    "Hi team — reminder our premium support package includes quarterly "
                    "architecture reviews. Reply YES to schedule."
                ),
            ),
        ],
        unresolved_ambiguities=[
            AmbiguityItem(
                ambiguity_id="amb_which_database",
                description="Which database system and tenant are implicated?",
                related_message_ids=[1, 3],
            ),
            AmbiguityItem(
                ambiguity_id="amb_error_class",
                description="Is the failure environmental (DNS/network) or data-tier saturation?",
                related_message_ids=[1, 2, 3],
            ),
        ],
        conflicting_claims=[
            ConflictRecord(
                conflict_id="cf_root_cause_dns_vs_postgres",
                topic="Root cause of deploy failure",
                party_a="infra_lisa",
                claim_a="DNS / resolver flakiness as primary hypothesis.",
                party_b="dba_marc",
                claim_b="Postgres replica pool exhausted (error 53300) from connection spike.",
                related_message_ids=[2, 3],
            ),
        ],
        ground_truth=TaskGroundTruth(
            task_id="deploy_database_dispute",
            facts={
                "incident_summary": (
                    "production deploy failed with a database-related error blocking the release train"
                ),
                "authoritative_db_diagnosis": (
                    "payments postgres replica hit connection pool exhaustion with sqlstate 53300"
                ),
                "misleading_hypothesis": (
                    "dns or resolver flakiness was suggested but does not match the database error"
                ),
                "decision_constraint": (
                    "executive go/no-go decision required within two hours for marketing launch"
                ),
            },
            conflict_alignments={
                "cf_root_cause_dns_vs_postgres": (
                    "resolver noise is not supported by the database error code; "
                    "the blocking failure is postgres connection pool exhaustion on the payments replica"
                ),
            },
            expected_resolution=(
                "prioritize remediating payments postgres replica pool exhaustion before "
                "re-running deploy; treat dns as secondary unless new evidence appears; "
                "report pool scaling or connection throttling status for the two-hour go/no-go"
            ),
        ),
    )


def _scenario_refund_policy_crossfire() -> RepairTaskScenario:
    """Customer refund: support vs finance cite incompatible policies; unrelated IT noise."""
    return RepairTaskScenario(
        task_id="refund_policy_crossfire",
        thread_messages=[
            ThreadMessage(
                message_id=1,
                sender="customer_jordan",
                timestamp_iso=_ts(20),
                content=(
                    "You charged my subscription twice this month. I want a full refund "
                    "credited today — this is unacceptable."
                ),
            ),
            ThreadMessage(
                message_id=2,
                sender="cs_sam",
                timestamp_iso=_ts(21),
                content=(
                    "Hi Jordan — per our standard policy you have a 30-day money-back window "
                    "and we'll take care of you."
                ),
            ),
            ThreadMessage(
                message_id=3,
                sender="finance_ria",
                timestamp_iso=_ts(22),
                content=(
                    "Hold on — Jordan is on enterprise tier. Appendix C caps refunds at 14 days "
                    "unless legal approves an exception."
                ),
            ),
            ThreadMessage(
                message_id=4,
                sender="it_helpdesk",
                timestamp_iso=_ts(23),
                content="Cafeteria guest wifi is down again. Not ticket-related.",
            ),
            ThreadMessage(
                message_id=5,
                sender="customer_jordan",
                timestamp_iso=_ts(24),
                content=(
                    "Nobody told me about appendix anything. Your chatbot guaranteed premium support."
                ),
            ),
        ],
        unresolved_ambiguities=[
            AmbiguityItem(
                ambiguity_id="amb_applicable_policy",
                description="Which refund window applies given tier and appendix references?",
                related_message_ids=[1, 2, 3, 5],
            ),
            AmbiguityItem(
                ambiguity_id="amb_customer_expectation",
                description="Customer cites chatbot promise vs written contract terms.",
                related_message_ids=[2, 3, 5],
            ),
        ],
        conflicting_claims=[
            ConflictRecord(
                conflict_id="cf_refund_window_cs_vs_finance",
                topic="Applicable refund period",
                party_a="cs_sam",
                claim_a="30-day money-back policy applies.",
                party_b="finance_ria",
                claim_b="Enterprise appendix limits refunds to 14 days without legal exception.",
                related_message_ids=[2, 3],
            ),
        ],
        ground_truth=TaskGroundTruth(
            task_id="refund_policy_crossfire",
            facts={
                "billing_issue": (
                    "duplicate subscription charge; customer requests immediate full refund"
                ),
                "support_cited_rule": (
                    "customer support stated a thirty-day money-back policy for standard handling"
                ),
                "finance_cited_rule": (
                    "finance stated enterprise appendix caps refunds at fourteen days without legal exception"
                ),
                "customer_pushback": (
                    "customer disputes appendix visibility and cites premium support expectations"
                ),
            },
            conflict_alignments={
                "cf_refund_window_cs_vs_finance": (
                    "the thirty-day statement reflects general consumer policy while the "
                    "fourteen-day cap reflects enterprise contract terms; tier and appendix "
                    "determine which rule governs this account"
                ),
            },
            expected_resolution=(
                "verify enterprise tier and appendix c applicability before quoting a refund window; "
                "if enterprise terms apply use the fourteen-day rule and escalate legal exceptions "
                "if warranted; otherwise apply thirty days; communicate one consistent refund "
                "determination to the customer"
            ),
        ),
    )


def _scenario_api_launch_triad() -> RepairTaskScenario:
    """Three-way API story: mobile vs platform vs auth; vendor SDK noise."""
    return RepairTaskScenario(
        task_id="api_launch_triad",
        thread_messages=[
            ThreadMessage(
                message_id=1,
                sender="mobile_taylor",
                timestamp_iso=_ts(30),
                content=(
                    "We're shipping GraphQL v3 to production this Friday — backend signed off "
                    "last week. Marketing already published the timeline."
                ),
            ),
            ThreadMessage(
                message_id=2,
                sender="platform_umar",
                timestamp_iso=_ts(31),
                content=(
                    "Public REST stays on v2 until Q3. GraphQL v3 is internal beta only — "
                    "please do not tell customers it's GA."
                ),
            ),
            ThreadMessage(
                message_id=3,
                sender="pm_kerry",
                timestamp_iso=_ts(32),
                content="Friday launch is non-negotiable. I need one external story everyone repeats.",
            ),
            ThreadMessage(
                message_id=4,
                sender="legacy_vendor_bot",
                timestamp_iso=_ts(33),
                content=(
                    "Action required: migrate to SDK 1.9 to unlock the new blue theme pack."
                ),
            ),
            ThreadMessage(
                message_id=5,
                sender="backend_nina",
                timestamp_iso=_ts(34),
                content=(
                    "Auth middleware is still enforcing v2 OAuth scopes. v3 routes break login "
                    "unless feature flag oauth_v3_compat is enabled in prod."
                ),
            ),
            ThreadMessage(
                message_id=6,
                sender="mobile_taylor",
                timestamp_iso=_ts(35),
                content="We're not rolling back — find a way to make v3 true externally.",
            ),
        ],
        unresolved_ambiguities=[
            AmbiguityItem(
                ambiguity_id="amb_public_api_truth",
                description="Is GraphQL v3 externally supported on Friday or internal-only?",
                related_message_ids=[1, 2, 6],
            ),
            AmbiguityItem(
                ambiguity_id="amb_auth_readiness",
                description="Are OAuth and middleware ready for external v3 traffic?",
                related_message_ids=[2, 5],
            ),
        ],
        conflicting_claims=[
            ConflictRecord(
                conflict_id="cf_public_graphql_narrative",
                topic="External availability of GraphQL v3",
                party_a="mobile_taylor",
                claim_a="v3 ships publicly Friday with leadership pressure.",
                party_b="platform_umar",
                claim_b="v3 is internal beta; public contract remains v2 until Q3.",
                related_message_ids=[1, 2, 3],
            ),
            ConflictRecord(
                conflict_id="cf_auth_gate_for_v3",
                topic="Auth readiness for v3",
                party_a="mobile_taylor",
                claim_a="v3 must be treated as externally launchable now.",
                party_b="backend_nina",
                claim_b="v3 breaks OAuth unless oauth_v3_compat flag is on.",
                related_message_ids=[5, 6],
            ),
        ],
        ground_truth=TaskGroundTruth(
            task_id="api_launch_triad",
            facts={
                "mobile_launch_position": (
                    "mobile commits to shipping graphql v3 publicly on friday with executive pressure"
                ),
                "platform_contract_position": (
                    "platform states public api remains v2 until third quarter and graphql v3 is internal beta"
                ),
                "backend_auth_constraint": (
                    "authentication middleware enforces v2 oauth scopes; v3 requires oauth_v3_compat enabled"
                ),
                "exec_comms_constraint": (
                    "product demands a single coordinated external narrative by friday"
                ),
            },
            conflict_alignments={
                "cf_public_graphql_narrative": (
                    "mobile and platform disagree on whether v3 is customer-facing; "
                    "external communications must follow platform's v2-until-q3 contract until "
                    "platform formally promotes v3"
                ),
                "cf_auth_gate_for_v3": (
                    "external v3 traffic is unsafe without oauth_v3_compat because middleware "
                    "oauth scopes are still v2-shaped; enable the flag and verify auth before "
                    "any public v3 claim"
                ),
            },
            expected_resolution=(
                "align externally on v2 as the supported public surface until q3 unless platform "
                "promotes v3; any friday mobile deliverable must be behind beta or flags until "
                "oauth_v3_compat is production-verified; publish one comms story that matches "
                "platform contract and auth readiness"
            ),
        ),
    )


_TASK_REGISTRY: dict[str, RepairTaskScenario] = {
    s.task_id: s
    for s in (
        _scenario_ui_backend_latency(),
        _scenario_deploy_database_dispute(),
        _scenario_refund_policy_crossfire(),
        _scenario_api_launch_triad(),
    )
}

# Older code may still reference the pre-tasks.py identifier.
_TASK_ALIASES: dict[str, str] = {
    "builtin_ui_vs_backend": "ui_backend_latency",
}


def list_task_ids() -> list[str]:
    """Stable order for benchmarks and CI."""
    return sorted(_TASK_REGISTRY.keys())


def get_task(task_id: str) -> TaskBundle:
    """
    Resolve a scenario by id. Pure data lookup — no randomness.

    Returns ground truth (for grading) and initial thread / ambiguity / conflict state.
    """
    resolved_id = _TASK_ALIASES.get(task_id, task_id)
    scenario = _TASK_REGISTRY.get(resolved_id)
    if scenario is None:
        known = ", ".join(list_task_ids())
        raise ValueError(f"Unknown task_id={task_id!r}. Known tasks: {known}")

    return TaskBundle(
        ground_truth=scenario.ground_truth,
        thread_messages=list(scenario.thread_messages),
        unresolved_ambiguities=list(scenario.unresolved_ambiguities),
        conflicting_claims=list(scenario.conflicting_claims),
    )


def get_task_scenario(task_id: str) -> RepairTaskScenario:
    """Full scenario object (for introspection, tests, or dataset export)."""
    resolved_id = _TASK_ALIASES.get(task_id, task_id)
    scenario = _TASK_REGISTRY.get(resolved_id)
    if scenario is None:
        known = ", ".join(list_task_ids())
        raise ValueError(f"Unknown task_id={task_id!r}. Known tasks: {known}")
    return scenario
