"""Typed, per-question Tier-1 signals; these are not student-level findings."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class ProbeAnchor(BaseModel):
    """An actually delivered patient probe, required to screen attention/omission."""

    turn_idx: int = Field(ge=0, strict=True)
    text: str = Field(min_length=1)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability

    @model_validator(mode="after")
    def validate_distribution(self):
        if not self.probabilities or not math.isclose(
            sum(self.probabilities.values()), 1.0, abs_tol=0.001
        ):
            raise ValueError("Probabilities must sum to one")
        if self.choice not in self.probabilities:
            raise ValueError("Choice must be a declared option")
        if self.probabilities[self.choice] < max(self.probabilities.values()):
            raise ValueError("Choice must be a most probable option")
        return self


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    score: float = Field(strict=True, ge=0.0, allow_inf_nan=False)
    probabilities: dict[str, Probability]
    legend: dict[str, str]
    confidence: Probability

    @model_validator(mode="after")
    def validate_distribution(self):
        levels = {str(i) for i in range(len(self.legend))}
        if len(levels) < 2 or set(self.legend) != levels:
            raise ValueError("Score levels must be consecutive starting at zero")
        if set(self.probabilities) != levels or not math.isclose(
            sum(self.probabilities.values()), 1.0, abs_tol=0.001
        ):
            raise ValueError("Every score level needs a normalized probability")
        expected = sum(int(k) * p for k, p in self.probabilities.items())
        if self.score > len(levels) - 1 or not math.isclose(
            self.score, expected, abs_tol=0.01
        ):
            raise ValueError("Score must be the probability-weighted level index")
        return self


DecisionAnswer = Annotated[ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


ReviewReason = Literal[
    "low_confidence",  # provider confidence below Config.CONFIDENCE_THRESHOLD
    "uncertain",  # the model chose the "uncertain" option
    "flagged_choice",  # a confident choice that requests review (e.g. "biased")
    "score_tail",  # an outer score level is at least as likely as the middle one
    "transport_error",  # timeout, connection failure or HTTP error status
    "invalid_response",  # malformed or incomplete answers from Kev
    "invalid_config",  # local misconfiguration (e.g. KEV_TIMEOUT_SECONDS); not sent
    "input_too_long",  # transcript text over KEV_MAX_INPUT_CHARS; not sent
    # a state value that cannot be sent as text (set, bytes, NaN/inf, object,
    # non-string key, keys colliding after flattening, nesting over 8); not sent
    "invalid_input",
]
_REQUEST_REASONS = {
    "transport_error",
    "invalid_response",
    "invalid_config",
    "input_too_long",
    "invalid_input",
}


class ReviewTrigger(BaseModel):
    """Why Tier-1 requested review: one question (or the request) and one reason."""

    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(default=None, min_length=1)
    reason: ReviewReason

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.reason in _REQUEST_REASONS) != (self.question is None):
            raise ValueError(
                "Request-level reasons have no question; answer-level reasons need one"
            )
        return self


class Tier1Result(BaseModel):
    provider: Literal["kev"] = "kev"
    status: Literal["ok", "unavailable"]
    model: str
    checkpoint: str | None = None
    prompt_hash: str
    latency_ms: int = Field(ge=0)
    answers: dict[str, DecisionAnswer] = Field(default_factory=dict)
    error: str | None = None
    # Every reason review was requested, so escalation rates and causes can be
    # measured. Empty means Tier-1 did not request review.
    review_triggers: list[ReviewTrigger] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_triggers(self):
        for trigger in self.review_triggers:
            if trigger.question is not None and trigger.question not in self.answers:
                raise ValueError("A review trigger must name an answered question")
        # A failed request is never "ok". The converse is not enforced, so legacy
        # rows stored as unavailable with an error and no triggers still load.
        if self.status == "ok" and (
            self.error is not None
            or any(t.reason in _REQUEST_REASONS for t in self.review_triggers)
        ):
            raise ValueError("An ok result cannot carry an error or request failure")
        return self
