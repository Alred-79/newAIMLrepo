"""Async client for a self-hosted jaredpalmer/kev System One server."""

import asyncio
import hashlib
import json
import math
import time

import httpx
from pydantic import TypeAdapter, ValidationError
from shared.config import Config
from shared.models import BiasLabel, EvaluationInput, EvaluationResult
from shared.tier1_models import ChoiceAnswer, DecisionAnswer, Tier1Result

from bias_eval.kev_questions import build_questions
from bias_eval.keywords import scan_for_stigma_anchors
from bias_eval.routing import compute_routing_decision

_ANSWER = TypeAdapter(DecisionAnswer)

_REVIEW_CHOICES = {
    "tone": {"biased"},
    "attribution": {"stereotype_linked"},
    "epistemic_stance": {"doubts", "overrides"},
    "affective_stance": {"cold", "hostile"},
}


def _needs_review(name: str, answer) -> bool:
    if answer.confidence < Config.CONFIDENCE_THRESHOLD:
        return True
    if isinstance(answer, ChoiceAnswer):
        return answer.choice == "uncertain" or answer.choice in _REVIEW_CHOICES.get(
            name, set()
        )
    # Preserve both tails: an expected score of 1 can hide under/over attention.
    return (
        max(answer.probabilities["0"], answer.probabilities["2"])
        >= answer.probabilities["1"]
    )


def _parse_answers(body: dict, questions: dict) -> dict:
    if not isinstance(body, dict) or body.get("truncated", False) is not False:
        raise ValueError("Incomplete input was evaluated")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("The response must answer every requested question")
    parsed = {}
    for name, question in questions.items():
        answer = _ANSWER.validate_python(answers[name])
        if answer.type != question["type"]:
            raise ValueError("Unexpected answer type")
        if isinstance(answer, ChoiceAnswer):
            if set(answer.probabilities) != set(question["criteria"]):
                raise ValueError("Unexpected choice options")
        else:
            legend = {str(i): text for i, text in enumerate(question["criteria"])}
            if answer.legend != legend:
                raise ValueError("Unexpected score levels")
        parsed[name] = answer
    return parsed


async def evaluate_with_kev(
    input: EvaluationInput, *, client: httpx.AsyncClient | None = None
) -> EvaluationResult:
    """Classify one turn; transport/schema failures abstain without changing providers.

    An injected client is caller-owned and can be pooled by a service. Otherwise this
    call creates and closes its own client, including for the synchronous wrapper.
    """
    started = time.monotonic()
    questions = build_questions(input)
    prompt_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    # The detector must not be told the bias axis, the scenario category (including
    # "control") or the bias direction. bias_direction is used only locally, in
    # build_questions, to decide which questions to ask.
    payload = {
        "model": Config.KEV_MODEL,
        "state": {
            "student_message": input.student_message,
            "recent_context": [
                m.model_dump(exclude_none=True) for m in input.recent_context
            ],
            "chart_context": input.chart_context,
            "probe": input.probe.model_dump() if input.probe else None,
        },
        "questions": questions,
    }
    error = None
    answers = {}
    try:
        timeout = Config.KEV_TIMEOUT_SECONDS
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        headers = {"Accept": "application/json"}
        if Config.KEV_API_KEY:
            headers["Authorization"] = f"Bearer {Config.KEV_API_KEY}"
        # Bound the entire operation, not just individual socket reads. No retries
        # in the synchronous encounter lane and no silent fallback to another model.
        async with asyncio.timeout(timeout):
            if client is None:
                async with httpx.AsyncClient(trust_env=False) as owned_client:
                    response = await owned_client.post(
                        Config.KEV_BASE_URL.rstrip("/") + "/v1/systemone",
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )
            else:
                response = await client.post(
                    Config.KEV_BASE_URL.rstrip("/") + "/v1/systemone",
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or body.get("model") != Config.KEV_MODEL:
                raise ValueError("Unexpected model")
            answers = _parse_answers(body, questions)
    except (TimeoutError, httpx.TimeoutException):
        error = "Kev request timed out"
    except httpx.HTTPStatusError as exc:
        error = f"Kev returned HTTP {exc.response.status_code}"
    except (httpx.HTTPError, httpx.InvalidURL):
        error = "Kev connection failed"
    except (ValueError, ValidationError, TypeError):
        # Do not echo exception strings: they can contain transcript text or keys.
        error = "Kev configuration or response was invalid"

    tier1 = Tier1Result(
        status="unavailable" if error else "ok",
        model=Config.KEV_MODEL,
        checkpoint=Config.KEV_CHECKPOINT or None,
        prompt_hash=prompt_hash,
        latency_ms=int((time.monotonic() - started) * 1000),
        answers=answers,
        error=error,
    )
    tone = answers.get("tone")
    label = BiasLabel(tone.choice) if tone else BiasLabel.UNCERTAIN
    confidence = tone.confidence if tone else 0.0
    should_escalate = (
        error is not None
        or any(_needs_review(name, answer) for name, answer in answers.items())
        or compute_routing_decision(label, confidence)
    )
    return EvaluationResult(
        tone_label=label,
        tone_confidence=confidence,
        stigma_anchors_found=scan_for_stigma_anchors(input.student_message),
        brief_reason=error
        or "Kev Tier-1 screening; see independent question probabilities.",
        should_escalate=should_escalate,
        tier1=tier1,
    )
