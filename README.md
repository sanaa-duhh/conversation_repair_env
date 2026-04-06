# Context-Aware Conversation Repair System

An OpenEnv benchmark for evaluating whether agents can repair ambiguous, conflicting, multi-user conversations through structured multi-step reasoning.

This project reframes conversation understanding as a **structured reasoning process**, rather than a one-shot generation task, requiring agents to build, align, and resolve context over multiple steps.


## 1. Overview

**Context-Aware Conversation Repair System** is an OpenEnv benchmark that evaluates whether an agent can extract, align, and resolve messy multi-user conversations using structured state and deterministic semantic rewards.

## 2. Problem

Real conversations are noisy: users disagree, key details are missing, and irrelevant chatter appears in the same thread.

This environment targets that regime directly:
- Conversations contain ambiguity (`unresolved_ambiguities`) and explicit conflicts (`conflicting_claims`).
- The agent must build structured understanding over turns, not just generate fluent text.
- Naive single-shot LLM behavior fails because it tends to guess a resolution without building consistent intermediate state.

## 3. Key Idea

The core idea is to force repair as a **process**, not a one-line answer:
- `ASK`: request missing information.
- `EXTRACT`: commit structured facts to state.
- `ALIGN`: reconcile conflicting claims with explicit per-conflict mappings.
- `RESOLVE`: propose final resolution only after the state supports it.

The environment grades this trajectory with semantic rubrics and contradiction checks, so reward reflects reasoning quality, not just wording style.

## 4. System Architecture

### Data Models (`models.py`)

- `ThreadMessage`: immutable conversation entries (`message_id`, `sender`, `timestamp_iso`, `content`).
- `AmbiguityItem`: underspecified points linked to message IDs.
- `ConflictRecord`: structured incompatible claims (`party_a/claim_a` vs `party_b/claim_b`).
- `TaskGroundTruth`: hidden rubric per task (`facts`, `conflict_alignments`, `expected_resolution`, optional `forbidden_fact_cooccurrence`).
- `ConversationRepairState`: full episodic state (`thread_history`, `extracted_context`, `conflicting_claims`, reward traces, semantic traces).

### Environment (`server/conversation_repair_environment.py`)

`ConversationRepairEnvironment` (OpenEnv `Environment`) does:
- `reset(task_id=...)`: loads deterministic task bundle (thread + ambiguities + conflicts + hidden ground truth).
- `step(action)`: routes by action type, updates state, computes semantic reward components, returns observation.
- `state`: exposes full structured state for debugging/evaluation.

State transitions are explicit:
- `EXTRACT` updates `extracted_context`.
- `ALIGN` can remove conflicts when reconciliation is semantically strong.
- `RESOLVE` succeeds only if semantic and logical checks pass.

## 5. Action Space

- `ASK`
  - Purpose: clarification when thread evidence is insufficient.
  - Effect: appends agent message, minimal shaping only.

- `EXTRACT`
  - Purpose: translate unstructured conversation into typed facts.
  - Input fields: `extracted_facts`, optional `content`.
  - Effect: merges facts into `extracted_context`; semantic gain/penalty computed per fact.

- `ALIGN`
  - Purpose: reconcile active conflicts.
  - Input fields: `conflict_alignments` (`conflict_id -> reconciliation text`), optional summaries.
  - Effect: per-conflict semantic evaluation; full-quality alignment removes corresponding conflict record.

- `RESOLVE`
  - Purpose: produce final repair decision.
  - Input fields: `resolution_summary` and/or `content`.
  - Effect: terminal success only when facts, conflicts, and resolution semantics are all consistent.

## 6. Tasks + Ground Truth

`tasks.py` defines deterministic `RepairTaskScenario`s:
- `ui_backend_latency`
- `deploy_database_dispute`
- `refund_policy_crossfire`
- `api_launch_triad`

Each scenario includes:
- `thread_messages`: intentionally messy conversation (noise, competing narratives, urgency constraints).
- `unresolved_ambiguities`: explicit open questions.
- `conflicting_claims`: structured disagreements to be reconciled.
- `ground_truth`:
  - `facts`: required structured facts.
  - `conflict_alignments`: reference reconciliations per `conflict_id`.
  - `expected_resolution`: canonical final resolution target.

Task construction is intentionally non-trivial: red herrings (`intern_bot`, `vendor_sales`, `legacy_vendor_bot`), policy disputes, operational constraints, and multi-conflict cases.

## 7. Reward Design (Critical)

The reward system is designed to approximate semantic understanding without relying on external LLM judges or keyword matching.

### Semantic core (`semantic_rubric.py`)

- Similarity metric:
  - Stopword-filtered token Jaccard (`word_jaccard`)
  - Character-bigram Jaccard (`char_bigram_jaccard`)
  - Weighted blend: `content_similarity = 0.58 * word + 0.42 * char`
- Fact thresholds:
  - `T_FACT_CORRECT = 0.68`
  - `T_FACT_PARTIAL = 0.38`
- Resolution thresholds:
  - `T_RESOLUTION_OK = 0.58`
  - `T_RES_FACT_MEAN = 0.18`

### Fact evaluation

`evaluate_fact_matrix` classifies each key into:
- `correct`
- `partial`
- `incorrect`
- `missing`
- `spurious`

Reward in `step()` uses this structure:
- Positive information gain when a fact crosses into stronger semantic bands.
- Penalties for incorrect/spurious facts.
- Penalties for semantic redundancy (near-duplicate re-submissions).

### Contradiction penalties

Two deterministic contradiction checks:
- Negation-style clashes across extracted rubric facts (`pairwise_extracted_contradictions`).
- Task-defined mutually exclusive key pairs (`forbidden_fact_cooccurrence`).

Both contribute capped negative reward.

### Alignment scoring logic

`alignment_semantic_eval` scores an alignment statement against:
- `claim_a` / `party_a` coverage
- `claim_b` / `party_b` coverage
- similarity to reference alignment text

Verdicts:
- `full`: captures both sides with sufficient rubric similarity -> conflict removed.
- `partial`: partially acceptable -> partial reward.
- `weak`: insufficient reconciliation -> penalty.

### Resolution scoring logic

`resolution_semantic_eval` checks:
- similarity to `expected_resolution`,
- mean overlap with rubric facts,
- coherence with extracted rubric facts.

A resolution is accepted only if:
- no active conflicts remain,
- no missing/wrong/spurious facts remain (partial facts can pass depending on thresholds),
- semantic resolution thresholds are met,
- resolution does not contradict extracted context.

This makes keyword hacks and one-step guessing ineffective.

## 8. Inference Pipeline

`inference.py` runs local episodes with OpenAI-compatible API calls:
- Client: `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`
- Model: `MODEL_NAME`
- Iterates over all tasks from `list_task_ids()`
- Strict stdout format:
  - `[START] ...`
  - `[STEP] ...`
  - `[END] ...`

Policy behavior:
- LLM outputs JSON actions.
- A deterministic controller forces stage progression to reduce unstable loops:
  - start with `EXTRACT`,
  - prioritize `ALIGN` while conflicts remain,
  - allow `RESOLVE` only after prior extraction/alignment conditions.

Controller is necessary because unconstrained LLM policies frequently repeat weak actions or attempt premature resolution.

## 9. Example Run

Example trace from this implementation:

```text
[START] task=ui_backend_latency env=conversation_repair model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=EXTRACT:facts_count=3 reward=-0.26 done=false error=null
[STEP] step=2 action=ALIGN:conflict_alignments_count=1 reward=-0.07 done=false error=null
[STEP] step=3 action=ALIGN:conflict_alignments_count=1 reward=-0.07 done=false error=null
[STEP] step=4 action=RESOLVE:has_summary=false reward=-0.14 done=false error=null
[END] success=false steps=6 score=0.45 rewards=-0.26,-0.07,-0.07,-0.14,-0.01,-0.07
```

What this shows:
- The agent follows staged actions.
- Poor/empty alignment quality and weak resolution still get penalized.
- The environment provides dense feedback instead of a single terminal label.

## 10. Why This Is Strong

Compared to standard LLM evaluation setups, this system introduces:
- Eliminates keyword-based reward shortcuts.
- Prevents single-step resolution shortcuts via enforced multi-step reasoning.
- Structured intermediate representations (`facts`, `conflicts`, `ambiguities`) are first-class.
- Deterministic semantic scoring is reproducible and inspectable.
- Same evaluation logic transfers across multiple task families.

## 11. How to Run

From project root (`conversation_repair_env`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./conversation_repair
```

Set inference variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="<your_hf_token>"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
```

Run:

```bash
python inference.py
```

## 12. Future Work

- Add richer user simulators so `ASK` has stronger interactive utility.
- Expand task library with longer threads, temporal updates, and multi-conflict cascades.
- Train policy agents directly on this environment (instead of prompt-only control) and compare with supervised baselines.

## 13. Impact

This framework provides a foundation for evaluating agents in real-world scenarios such as customer support, incident triage, and collaborative debugging—where ambiguity and conflicting information are the norm rather than the exception.

