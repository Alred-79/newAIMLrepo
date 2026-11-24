"""Async client for a self-hosted jaredpalmer/kev System One server."""

import asyncio
import contextlib
import hashlib
import json
import re
import time

import httpx
from pydantic import TypeAdapter, ValidationError
from shared.config import (
    Config,
    kev_base_url_is_allowed,
    kev_max_input_chars_is_valid,
    kev_max_response_bytes_is_valid,
    kev_timeout_is_valid,
)
from shared.models import BiasLabel, EvaluationInput, EvaluationResult
from shared.tier1_models import (
    ChoiceAnswer,
    DecisionAnswer,
    ReviewTrigger,
    Tier1Result,
)

from bias_eval.kev_questions import build_questions
from bias_eval.keywords import scan_for_stigma_anchors
from bias_eval.routing import compute_routing_decision

_ANSWER = TypeAdapter(DecisionAnswer)


class _InvalidConfig(Exception):
    """A local setting makes the request impossible; nothing is sent to Kev."""


class _InputTooLong(Exception):
    """The transcript text exceeds KEV_MAX_INPUT_CHARS; nothing is sent to Kev."""


class _EncodedResponse(Exception):
    """Kev sent a Content-Encoding other than identity; the body is never read."""


def _is_identity_encoded(response: httpx.Response) -> bool:
    # Any coding (gzip, deflate, br, stacked "gzip, gzip", or an empty value)
    # is refused: httpx would inflate it before the byte cap could count it.
    codings = [
        coding.strip().lower()
        for value in response.headers.get_list("content-encoding")
        for coding in value.split(",")
    ]
    return all(coding == "identity" for coding in codings)


# Kev renders the state as labelled lines, so a line break inside a message
# could fake another field (e.g. a "recent_context:" block). Any run containing
# Every control character or line/paragraph separator becomes a space, then each
# whitespace run becomes one space and the ends are trimmed. Same rule for every
# student; only the copy sent to Kev is changed, never the stored transcript.
# Two single-class passes keep this linear in the text length (an earlier
# combined pattern backtracked quadratically on long whitespace runs).
_CONTROL_CHAR = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_WHITESPACE_RUN = re.compile(r"\s+")


def _flatten(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", _CONTROL_CHAR.sub(" ", text)).strip()


def _input_chars(state: dict) -> int:
    chars = len(state["student_message"])
    chars += sum(len(m.get("content", "")) for m in state["recent_context"])
    if state["probe"]:
        chars += len(state["probe"]["text"])
    if state["chart_context"] is not None:
        chars += len(json.dumps(state["chart_context"], ensure_ascii=False))
    return chars


# patient_bias_response has no flagged choices: a confident answer (including
# "not_applicable", the expected answer when the patient expressed no prejudice)
# requests review only through low confidence or "uncertain".
_REVIEW_CHOICES = {
    "tone": {"biased"},
    "attribution": {"stereotype_linked"},
    "epistemic_stance": {"doubts", "overrides"},
    "affective_stance": {"cold", "hostile"},
}


def _review_reasons(name: str, answer) -> list[str]:
    """All reasons this answer requests review; empty means no review."""
    reasons = []
    if answer.confidence < Config.CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if isinstance(answer, ChoiceAnswer):
        if answer.choice == "uncertain":
            reasons.append("uncertain")
        elif answer.choice in _REVIEW_CHOICES.get(name, set()):
            reasons.append("flagged_choice")
    # Preserve both tails: an expected score of 1 can hide under/over attention.
    elif (
        max(answer.probabilities["0"], answer.probabilities["2"])
        >= answer.probabilities["1"]
    ):
        reasons.append("score_tail")
    return reasons


def _needs_review(name: str, answer) -> bool:
    return bool(_review_reasons(name, answer))


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
    Redirects are refused per request even if an injected client was built with
    follow_redirects=True; callers should still build it with redirects disabled.
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
    # "control") or the bias direction, neither in the state nor through which
    # questions are asked: build_questions does not read either field.
    payload = {
        "model": Config.KEV_MODEL,
        "state": {
            # Raw here; flattened in place below, after the length budget passes.
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
    error_reason = None
    answers = {}
    received = False  # whether a response arrived, to attribute unexpected errors
    try:
        timeout = Config.KEV_TIMEOUT_SECONDS
        if not kev_timeout_is_valid(timeout):
            # shared.config already warned once at import
            raise _InvalidConfig(
                "KEV_TIMEOUT_SECONDS is not a number between 0.05 and 60 seconds"
            )
        if not kev_base_url_is_allowed(Config.KEV_BASE_URL):
            raise _InvalidConfig(
                "KEV_BASE_URL must be https (plain http only to a loopback host)"
            )
        max_bytes = Config.KEV_MAX_RESPONSE_BYTES
        if not kev_max_response_bytes_is_valid(max_bytes):
            raise _InvalidConfig("KEV_MAX_RESPONSE_BYTES is not a valid byte count")
        if not kev_max_input_chars_is_valid(Config.KEV_MAX_INPUT_CHARS):
            raise _InvalidConfig("KEV_MAX_INPUT_CHARS is not a valid character count")
        # Budget on the raw text first, so oversized input costs no regex work.
        state = payload["state"]
        if _input_chars(state) > Config.KEV_MAX_INPUT_CHARS:
            raise _InputTooLong
        state["student_message"] = _flatten(state["student_message"])
        for message in state["recent_context"]:  # fresh dicts from model_dump
            message["content"] = _flatten(message["content"])
        try:
            url = httpx.URL(Config.KEV_BASE_URL.strip().rstrip("/") + "/v1/systemone")
        except (httpx.InvalidURL, ValueError, TypeError):
            raise _InvalidConfig("KEV_BASE_URL is not a valid URL") from None
        # Ask for an uncompressed body; a compressed one is refused unread below.
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if Config.KEV_API_KEY:
            headers["Authorization"] = f"Bearer {Config.KEV_API_KEY}"
        # Bound the entire operation, not just individual socket reads. No retries
        # in the synchronous encounter lane and no silent fallback to another model.
        # Redirects are never followed (also on an injected client), so the
        # transcript cannot be re-sent to another host; a 3xx is a transport error.
        async with asyncio.timeout(timeout):
            async with contextlib.AsyncExitStack() as stack:
                if client is None:
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(trust_env=False, follow_redirects=False)
                    )
                response = await stack.enter_async_context(
                    client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                        follow_redirects=False,
                    )
                )
                response.raise_for_status()
                received = True
                # Decided from the headers alone, before any body byte is read.
                if not _is_identity_encoded(response):
                    raise _EncodedResponse
                # Identity-only, so the streamed bytes are the raw wire bytes: the
                # cap rejects an oversized body without buffering it in full.
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("Kev response exceeds KEV_MAX_RESPONSE_BYTES")
                    chunks.append(chunk)
            body = json.loads(b"".join(chunks))
            if not isinstance(body, dict) or body.get("model") != Config.KEV_MODEL:
                raise ValueError("Unexpected model")
            answers = _parse_answers(body, questions)
    except (TimeoutError, httpx.TimeoutException):
        error, error_reason = "Kev request timed out", "transport_error"
    except httpx.HTTPStatusError as exc:
        error = f"Kev returned HTTP {exc.response.status_code}"
        error_reason = "transport_error"
    except (httpx.HTTPError, httpx.InvalidURL):
        error, error_reason = "Kev connection failed", "transport_error"
    except _InputTooLong:
        error = "Kev input exceeds KEV_MAX_INPUT_CHARS; not sent"
        error_reason = "input_too_long"
    except _EncodedResponse:
        error = "Kev response was compressed; refused unread"
        error_reason = "invalid_response"
    except _InvalidConfig as exc:
        # Our own fixed messages; they name a setting, never its value.
        error, error_reason = str(exc), "invalid_config"
    except (ValueError, ValidationError, TypeError):
        # Do not echo exception strings: they can contain transcript text or keys.
        error = "Kev response was invalid"
        error_reason = "invalid_response"
    except Exception:  # noqa: BLE001 - fail safe is the intent
        # Fail safe on anything else (e.g. RecursionError from deeply nested JSON,
        # or a transport raising RuntimeError/OSError): abstain and escalate.
        # Never record str(exc): it can contain transcript text.
        if received:
            error, error_reason = "Kev response was invalid", "invalid_response"
        else:
            error, error_reason = "Kev request failed", "transport_error"

    if error_reason:
        triggers = [ReviewTrigger(reason=error_reason)]
    else:
        triggers = [
            ReviewTrigger(question=name, reason=reason)
            for name, answer in answers.items()
            for reason in _review_reasons(name, answer)
        ]

    tier1 = Tier1Result(
        status="unavailable" if error else "ok",
        model=Config.KEV_MODEL,
        checkpoint=Config.KEV_CHECKPOINT or None,
        prompt_hash=prompt_hash,
        latency_ms=int((time.monotonic() - started) * 1000),
        answers=answers,
        error=error,
        review_triggers=triggers,
    )
    tone = answers.get("tone")
    label = BiasLabel(tone.choice) if tone else BiasLabel.UNCERTAIN
    confidence = tone.confidence if tone else 0.0
    should_escalate = bool(triggers) or compute_routing_decision(label, confidence)
    return EvaluationResult(
        tone_label=label,
        tone_confidence=confidence,
        stigma_anchors_found=scan_for_stigma_anchors(input.student_message),
        brief_reason=error
        or "Kev Tier-1 screening; see independent question probabilities.",
        should_escalate=should_escalate,
        tier1=tier1,
    )
