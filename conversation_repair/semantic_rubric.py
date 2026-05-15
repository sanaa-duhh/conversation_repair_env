# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Deterministic semantic-style scoring without external APIs: combines stopword-stripped
# word-token Jaccard with character bigram Jaccard (paraphrase-friendly, reproducible).

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import ConflictRecord, TaskGroundTruth

# --- Thresholds (fixed; no randomness) ---
# Fact value vs ground-truth reference text.
T_FACT_CORRECT = 0.68
T_FACT_PARTIAL = 0.38

# Resolution vs expected_resolution (combined similarity).
T_RESOLUTION_OK = 0.58
T_RESOLUTION_STRONG = 0.72

# Alignment must touch both sides of the structured conflict record.
T_SIDE_COVER = 0.24
# How close the statement should be to the reference alignment (paraphrase band).
T_ALIGN_RUBRIC = 0.42
T_ALIGN_RUBRIC_LOOSE = 0.32

# Resolution should echo rubric facts (mean coverage).
T_RES_FACT_MEAN = 0.18
# Resolution should not ignore extracted rubric facts entirely.
T_RES_FACT_COHERENCE = 0.12

# Weight: word-level vs character bigram channel.
_W_WORD = 0.42
_W_CHAR = 0.58

_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if in on at to for of as by with from than then so not no
    is are was were be been being it its this that these those we you they he she
    about into over after before also just only very more most some any all each
    both either neither do does did doing done can could should would will may might
    must shall need per via etc
    """.split()
)


def normalize_text(s: str) -> str:
    return " ".join(s.casefold().split())


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(s))


def content_tokens(s: str) -> frozenset[str]:
    """Alphanumeric tokens minus a small English stopword list (deterministic)."""
    return frozenset(t for t in _tokens(s) if t not in _STOPWORDS and len(t) > 1)


def word_jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def char_bigram_jaccard(a: str, b: str) -> float:
    """Character bigrams over whitespace-stripped text (captures morphology/order)."""
    sa = "".join(normalize_text(a).split())
    sb = "".join(normalize_text(b).split())
    if len(sa) < 2 and len(sb) < 2:
        return 1.0 if sa == sb else 0.0
    if len(sa) < 2 or len(sb) < 2:
        return 0.0
    ba = {sa[i : i + 2] for i in range(len(sa) - 1)}
    bb = {sb[i : i + 2] for i in range(len(sb) - 1)}
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def content_similarity(a: str, b: str) -> float:
    """
    Single scalar in [0, 1]: blends token Jaccard and char-bigram Jaccard.
    Paraphrases often share content words or substrings without exact equality.
    """
    return _W_WORD * word_jaccard(a, b) + _W_CHAR * char_bigram_jaccard(a, b)


FactLabel = Literal["correct", "partial", "incorrect", "missing", "spurious"]


def classify_fact_similarity(sim: float) -> Literal["correct", "partial", "incorrect"]:
    if sim >= T_FACT_CORRECT:
        return "correct"
    if sim >= T_FACT_PARTIAL:
        return "partial"
    return "incorrect"


class FactSemanticScore(BaseModel):
    model_config = {"extra": "forbid"}

    key: str
    similarity: float = Field(ge=0.0, le=1.0)
    word_jaccard: float = Field(ge=0.0, le=1.0)
    char_bigram_jaccard: float = Field(ge=0.0, le=1.0)
    label: FactLabel


class AlignmentSemanticScore(BaseModel):
    model_config = {"extra": "forbid"}

    conflict_id: str
    side_a_score: float
    side_b_score: float
    rubric_similarity: float
    captures_both_sides: bool
    captures_one_side: bool
    verdict: Literal["full", "partial", "weak"]


class ResolutionSemanticScore(BaseModel):
    model_config = {"extra": "forbid"}

    resolution_vs_expected: float
    mean_resolution_vs_gt_facts: float
    resolution_vs_extracted_rubric: float
    all_rubric_keys_present_partial_or_better: bool
    min_rubric_fact_similarity: float


def evaluate_fact_matrix(
    extracted: dict[str, str], gt: TaskGroundTruth
) -> tuple[list[FactSemanticScore], dict[str, Any]]:
    """Per-key semantic scores + aggregate reasoning breakdown."""
    scores: list[FactSemanticScore] = []
    by_key: dict[str, dict[str, Any]] = {}

    for key in sorted(gt.facts.keys()):
        ref = gt.facts[key]
        if key not in extracted or not str(extracted[key]).strip():
            fs = FactSemanticScore(
                key=key,
                similarity=0.0,
                word_jaccard=0.0,
                char_bigram_jaccard=0.0,
                label="missing",
            )
            scores.append(fs)
            by_key[key] = fs.model_dump()
            continue
        val = extracted[key]
        wj = word_jaccard(val, ref)
        cj = char_bigram_jaccard(val, ref)
        sim = content_similarity(val, ref)
        lbl: FactLabel = classify_fact_similarity(sim)  # type: ignore[assignment]
        fs = FactSemanticScore(
            key=key,
            similarity=round(sim, 4),
            word_jaccard=round(wj, 4),
            char_bigram_jaccard=round(cj, 4),
            label=lbl,  # type: ignore[arg-type]
        )
        scores.append(fs)
        by_key[key] = fs.model_dump()

    spurious = sorted(k for k in extracted if k not in gt.facts)
    for sk in spurious:
        fs = FactSemanticScore(
            key=sk,
            similarity=0.0,
            word_jaccard=0.0,
            char_bigram_jaccard=0.0,
            label="spurious",
        )
        scores.append(fs)
        by_key[sk] = fs.model_dump()

    breakdown: dict[str, Any] = {
        "facts_by_key": by_key,
        "spurious_keys": spurious,
        "mean_rubric_similarity": round(
            sum(s.similarity for s in scores if s.key in gt.facts)
            / max(1, len(gt.facts)),
            4,
        ),
    }
    return scores, breakdown


def fact_scores_to_evaluation_lists(
    scores: list[FactSemanticScore], gt: TaskGroundTruth
) -> dict[str, list[str]]:
    """Maps semantic labels to FactEvaluationResult-style key lists."""
    correct: list[str] = []
    partial: list[str] = []
    wrong: list[str] = []
    missing: list[str] = []
    spurious: list[str] = []
    for s in scores:
        if s.key not in gt.facts:
            if s.label == "spurious":
                spurious.append(s.key)
            continue
        if s.label == "correct":
            correct.append(s.key)
        elif s.label == "partial":
            partial.append(s.key)
        elif s.label == "missing":
            missing.append(s.key)
        else:
            wrong.append(s.key)
    return {
        "correct_keys": sorted(correct),
        "partial_value_keys": sorted(partial),
        "wrong_value_keys": sorted(wrong),
        "missing_keys": sorted(missing),
        "spurious_keys": sorted(spurious),
    }


def similarity_dict(scores: list[FactSemanticScore], gt: TaskGroundTruth) -> dict[str, float]:
    return {s.key: s.similarity for s in scores if s.key in gt.facts}


def alignment_semantic_eval(
    stmt: str, conflict: ConflictRecord, reference_alignment: str
) -> AlignmentSemanticScore:
    """
    Both-sides check: overlap with each claim and with party labels (short strings).
    """
    sa = max(content_similarity(stmt, conflict.claim_a), content_similarity(stmt, conflict.party_a))
    sb = max(content_similarity(stmt, conflict.claim_b), content_similarity(stmt, conflict.party_b))
    rub = content_similarity(stmt, reference_alignment)
    both = sa >= T_SIDE_COVER and sb >= T_SIDE_COVER
    one = (sa >= T_SIDE_COVER) ^ (sb >= T_SIDE_COVER)
    if both and rub >= T_ALIGN_RUBRIC:
        verdict: Literal["full", "partial", "weak"] = "full"
    elif both:
        verdict = "partial"
    elif one and rub >= T_ALIGN_RUBRIC_LOOSE:
        verdict = "partial"
    else:
        verdict = "weak"
    return AlignmentSemanticScore(
        conflict_id=conflict.conflict_id,
        side_a_score=round(sa, 4),
        side_b_score=round(sb, 4),
        rubric_similarity=round(rub, 4),
        captures_both_sides=both,
        captures_one_side=one,
        verdict=verdict,
    )


def resolution_semantic_eval(
    resolution: str, gt: TaskGroundTruth, extracted: dict[str, str]
) -> ResolutionSemanticScore:
    exp = gt.expected_resolution
    rv = content_similarity(resolution, exp)

    fact_sims: list[float] = []
    rubric_vals: list[str] = []
    for k, ref in gt.facts.items():
        rubric_vals.append(ref)
        if k in extracted and str(extracted[k]).strip():
            fact_sims.append(content_similarity(resolution, extracted[k]))
        else:
            fact_sims.append(0.0)
    mean_rf = sum(fact_sims) / max(1, len(fact_sims))

    # Coherence: resolution vs concatenated extracted rubric facts (what the agent committed to).
    concat = " ".join(str(extracted.get(k, "")) for k in sorted(gt.facts.keys()))
    rcoh = content_similarity(resolution, concat)

    min_sim = 1.0
    all_ok = True
    for k, ref in gt.facts.items():
        if k not in extracted or not str(extracted[k]).strip():
            min_sim = 0.0
            all_ok = False
            continue
        sim = content_similarity(extracted[k], ref)
        min_sim = min(min_sim, sim)
        if sim < T_FACT_PARTIAL:
            all_ok = False

    return ResolutionSemanticScore(
        resolution_vs_expected=round(rv, 4),
        mean_resolution_vs_gt_facts=round(mean_rf, 4),
        resolution_vs_extracted_rubric=round(rcoh, 4),
        all_rubric_keys_present_partial_or_better=all_ok and bool(gt.facts),
        min_rubric_fact_similarity=round(min_sim, 4),
    )


def negation_style_contradiction(a: str, b: str) -> bool:
    """
    Lightweight deterministic cue: one text negates while sharing substantive tokens
    with the other (not substring fuzzy matching on the whole thread).
    """
    ta, tb = content_tokens(a), content_tokens(b)
    if len(ta) < 2 or len(tb) < 2:
        return False
    shared = ta & tb
    if len(shared) < 2:
        return False
    neg = frozenset(
        "not no never false deny denied rejecting reject isnt isn't wasnt wasn't cannot can't wont won't".split()
    )
    raw_a = set(_tokens(a))
    raw_b = set(_tokens(b))
    a_neg = bool(raw_a & neg)
    b_neg = bool(raw_b & neg)
    return a_neg ^ b_neg


def pairwise_extracted_contradictions(
    extracted: dict[str, str], rubric_keys: frozenset[str]
) -> list[tuple[str, str]]:
    """Undirected pairs of rubric-key values that trigger negation-style clash."""
    keys = sorted(k for k in extracted if k in rubric_keys and str(extracted[k]).strip())
    out: list[tuple[str, str]] = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            if negation_style_contradiction(extracted[k1], extracted[k2]):
                out.append((k1, k2))
    return out


def forbidden_cooccurrence_hits(
    extracted: dict[str, str], pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Task-specified key pairs that should not both be asserted."""
    hits: list[tuple[str, str]] = []
    for a, b in pairs:
        if (
            a in extracted
            and b in extracted
            and str(extracted[a]).strip()
            and str(extracted[b]).strip()
        ):
            hits.append((a, b))
    return hits


def resolution_contradicts_extracted(
    resolution: str, extracted: dict[str, str], gt: TaskGroundTruth
) -> bool:
    """
    Resolution largely ignores the agent's own extracted rubric facts (semantic gap),
    which is a reasoning failure when those facts are non-empty.
    """
    parts = [str(extracted[k]).strip() for k in sorted(gt.facts.keys()) if k in extracted]
    if len(parts) < 2:
        return False
    blob = " ".join(parts)
    if len(blob) < 12:
        return False
    return content_similarity(resolution, blob) < T_RES_FACT_COHERENCE
