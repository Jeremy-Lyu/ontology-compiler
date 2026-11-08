"""Decision contract — ported from NanoJev (NanoJev-main/NANOJEV_CORE_REUSE.md).

Covers the reusable core WITHOUT requiring CUDA/MPS/torch/checkpoint:
  - validate_request: top-level {"states": [...]} strict validation
  - three primitives: boolean / choice(2-255) / score(2-10)
  - answer_from_probabilities: sum(p)=1+-1e-5, derive value/choice/score/level
  - decision_score: probability-weighted level index
  - brier / nll / ece: calibration metrics matching results/composed_*.json headers

Source semantics:
  NanoJev scripts/predict_toy_decisions.py::validate_request,
  ::answer_from_probabilities; docs/TYPESAFE_CONTRACT.md; docs/ATOMIC_PLANNING.md.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, field_validator

QuestionType = Literal["boolean", "choice", "score"]

MAX_STATES = 32
MAX_QUESTIONS = 96
MAX_PATHS = 256
MAX_BODY_BYTES = 2_000_000


class ChoiceQ(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: Dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(v, dict) or not 2 <= len(v) <= 255:
            raise ValueError("Choice criteria must contain 2-255 items")
        for k, desc in v.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("Choice candidate ID must be a non-empty string")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError("Choice candidate description must be a non-empty string")
        return v


class BooleanQ(BaseModel):
    type: Literal["boolean"] = "boolean"
    instructions: str = Field(min_length=1)
    criteria: Dict[str, str] | None = None

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v: Dict[str, str] | None) -> Dict[str, str] | None:
        if v is None:
            return v
        if not isinstance(v, dict) or set(v) - {"false", "true"}:
            raise ValueError("Boolean criteria may only contain false/true keys")
        for desc in v.values():
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError("Boolean criterion must be a non-empty string")
        return v


class ScoreQ(BaseModel):
    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    criteria: List[str] = Field(min_length=2, max_length=10)

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v: List[str]) -> List[str]:
        for desc in v:
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError("Score level description must be a non-empty string")
        return v


Question = Union[ChoiceQ, BooleanQ, ScoreQ]


class StateItem(BaseModel):
    id: str = Field(min_length=1)
    state: Any
    questions: Dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _check_questions(cls, v: Dict[str, Question]) -> Dict[str, Question]:
        if not isinstance(v, dict) or not v:
            raise ValueError("questions must be a non-empty object")
        for qid in v:
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("question ID must be a non-empty string")
        return v

    model_config = {"extra": "forbid"}


class DecisionPayload(BaseModel):
    states: List[StateItem] = Field(min_length=1, max_length=MAX_STATES)

    model_config = {"extra": "forbid"}


def validate_request(payload: Dict[str, Any]) -> DecisionPayload:
    """Strict validation. Extra keys rejected; truncation never applied."""
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError('Input must be exactly {"states": [...]}')
    parsed = DecisionPayload.model_validate(payload)
    n_questions = sum(len(s.questions) for s in parsed.states)
    n_paths = 0
    for s in parsed.states:
        for q in s.questions.values():
            if isinstance(q, BooleanQ):
                n_paths += 1
            elif isinstance(q, ChoiceQ):
                n_paths += len(q.criteria)
            else:
                n_paths += len(q.criteria)
    if n_questions > MAX_QUESTIONS:
        raise ValueError(f"Local demo limit: {MAX_QUESTIONS} questions per request")
    if n_paths > MAX_PATHS:
        raise ValueError(f"Local demo limit: {MAX_PATHS} candidate paths per request")
    ids = [s.id for s in parsed.states]
    if len(set(ids)) != len(ids):
        raise ValueError("state id must be unique non-empty strings")
    for s in parsed.states:
        if isinstance(s.state, str):
            if not s.state.strip():
                raise ValueError("state content must not be empty")
        elif s.state is None or s.state == {} or s.state == []:
            raise ValueError("state content must not be empty")
    return parsed


def candidate_ids(question: Question) -> List[str]:
    if isinstance(question, BooleanQ):
        return ["false", "true"]
    if isinstance(question, ChoiceQ):
        return list(question.criteria.keys())
    return [str(i) for i in range(len(question.criteria))]  # type: ignore[union-attr]


def decision_score(probabilities: List[float]) -> float:
    """Probability-weighted level index: score = sum(i * p_i)."""
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("Probabilities must sum to 1 within 1e-5")
    return math.fsum(i * p for i, p in enumerate(probabilities))


def answer_from_probabilities(question: Question, probabilities: List[float]) -> Dict[str, Any]:
    """Derive decision values from a full distribution. No partial returns."""
    ids = candidate_ids(question)
    if len(probabilities) != len(ids):
        raise ValueError("Probability vector length mismatch")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("Model produced invalid probabilities")
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("Probability sum must be 1")
    best = max(range(len(ids)), key=probabilities.__getitem__)
    result: Dict[str, Any] = {"type": question.type, "probabilities": dict(zip(ids, probabilities))}
    if isinstance(question, BooleanQ):
        result.update(p_true=probabilities[1], value=bool(best))
    elif isinstance(question, ChoiceQ):
        result.update(choice=ids[best], value=ids[best])
    else:
        score = decision_score(probabilities)
        result.update(score=score, level=best, value=score)
    return result


def scalar_brier(p_true: float, gold: bool) -> float:
    """Scalar Bernoulli Brier: (p - y)^2. Two-class summed Brier is 2x this."""
    return (p_true - (1.0 if gold else 0.0)) ** 2


def multiclass_nll(probabilities: List[float], gold_index: int) -> float:
    return -math.log(max(probabilities[gold_index], 1e-30))


def expected_calibration_error(
    p_trues: List[float], golds: List[bool], n_bins: int = 10
) -> float:
    """Standard ECE over confidence bins. Empty bins contribute 0."""
    if len(p_trues) != len(golds) or not p_trues:
        raise ValueError("p_trues and golds must be non-empty and aligned")
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for i, p in enumerate(p_trues):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append(i)
    ece = 0.0
    n = len(p_trues)
    for b in bins:
        if not b:
            continue
        acc = sum(1 for i in b if golds[i]) / len(b)
        conf = sum(p_trues[i] for i in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
    return ece
