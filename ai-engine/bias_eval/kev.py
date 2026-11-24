"""Async client for a self-hosted jaredpalmer/kev System One server."""

import asyncio
import contextlib
import hashlib
import json
import math
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


class _InvalidInput(Exception):
    """A state value cannot be sent as plain text; nothing is sent to Kev."""


# Kev renders the state as labelled lines, so a line break inside any string
# could fake another field (e.g. a "student_message:" line). In every string
# sent (dict keys, roles, timestamps, probe and chart text included), every
# control character and line/paragraph separator becomes a space, each
# whitespace run becomes one space, and the ends are trimmed. Only the
# invisible characters used to hide or reorder text are deleted: zero-width
# space U+200B, bidi embeddings/overrides U+202A-202E and isolates
# U+2066-2069, word joiner and invisible operators U+2060-2064, BOM U+FEFF,
# interlinear annotation U+FFF9-FFFB, and the tag characters U+E0001 and
# U+E0020-E007F. A well-formed subdivision flag (U+1F3F4, 3-7 tag letters or
# digits, then cancel tag U+E007F) is kept whole. Every other format character
# is kept, including ZWNJ/ZWJ (U+200C/U+200D: Persian, Indic and emoji
# spelling), LRM/RLM, Arabic number signs and the soft hyphen, because
# deleting them changes real words. Same rule for every student; only the copy
# sent to Kev changes, never the stored transcript. Each pass is one
# alternation without nested repetition (the flag branch is bounded), so this
# stays linear in the text length (an earlier combined pattern backtracked
# quadratically).
_FORMAT_CHAR = re.compile(
    "(\U0001f3f4[\U000e0030-\U000e0039\U000e0061-\U000e007a]{3,7}\U000e007f)"
    "|[\u200b\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff"
    "\ufff9-\ufffb\U000e0001\U000e0020-\U000e007f]"
)
_CONTROL_CHAR = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_WHITESPACE_RUN = re.compile(r"\s+")
_MAX_STATE_DEPTH = 8  # container nesting levels allowed inside a state field


def _flatten(text: str) -> str:
    text = _CONTROL_CHAR.sub(" ", _FORMAT_CHAR.sub(r"\1", text))
    return _WHITESPACE_RUN.sub(" ", text).strip()


# Kev labels the state's top-level fields with these names; a chart_context
# key equal to one (after flattening, ignoring case) could pass for a field.
_STATE_LABELS = frozenset(
    {"student_message", "recent_context", "chart_context", "probe"}
)
_CONTAINERS = (dict, list, tuple)


def _json_str_chars(text: str, total: int, budget: int) -> int:
    # Escaping only lengthens a string, so check its raw length (plus quotes)
    # first: an oversized string is rejected without being encoded.
    if total + len(text) + 2 > budget:
        raise _InputTooLong
    return len(json.dumps(text, ensure_ascii=False))


def _scalar_chars(value) -> int:
    """Length of a non-container value as JSON text; raise if it is not text-safe.

    Exact types only: a subclass (str, int, float, enum, ...) could change its
    own length, text or JSON form, so it is refused rather than trusted.
    """
    if value is None or value is True:
        return 4
    if value is False:
        return 5
    if type(value) is int:
        try:
            return len(str(value))
        except ValueError:  # beyond Python's int-to-str digit limit
            raise _InvalidInput from None
    if type(value) is float and math.isfinite(value):
        return len(repr(value))
    raise _InvalidInput  # set, bytes, NaN/inf, subclasses, arbitrary objects, ...


def _raw_state_chars(state: dict, budget: int) -> int:
    """Size of the state as compact JSON, checked before any regex work.

    The count is exactly len(json.dumps(state, ensure_ascii=False,
    separators=(",", ":"))), the form httpx sends, measured on the state
    before flattening (flattening only shortens it). The walk stops as soon
    as the budget is exceeded, so its cost is bounded by the budget. Values
    that are not exactly str/int/float/bool/None/dict/list/tuple, non-str
    keys and nesting deeper than _MAX_STATE_DEPTH raise _InvalidInput.
    """
    total = 0
    stack = [(state, 0)]
    while stack:
        value, depth = stack.pop()
        kind = type(value)
        if kind is str:
            total += _json_str_chars(value, total, budget)
        elif kind in _CONTAINERS:
            if depth > _MAX_STATE_DEPTH:
                raise _InvalidInput
            # Brackets, commas between items, and a colon per dict entry.
            total += 2 + max(len(value) - 1, 0)
            if kind is dict:
                total += len(value)
            if total > budget:
                raise _InputTooLong
            if kind is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        raise _InvalidInput
                    total += _json_str_chars(key, total, budget)
                    if total > budget:
                        raise _InputTooLong
                    stack.append((item, depth + 1))
            else:
                stack.extend((item, depth + 1) for item in value)
        else:
            total += _scalar_chars(value)
        if total > budget:
            raise _InputTooLong
    return total


def _flattened(value, reserved: frozenset = frozenset()):
    """Copy with every string and dict key flattened; structure otherwise kept."""
    if type(value) is str:
        return _flatten(value)
    if type(value) is dict:
        copy = {}
        for key, item in value.items():
            flat_key = _flatten(key)
            if flat_key in copy:  # e.g. "a\nb" and "a b": never drop a value
                raise _InvalidInput
            if flat_key.casefold() in reserved:
                raise _InvalidInput
            copy[flat_key] = _flattened(item, reserved)
        return copy
    if type(value) in (list, tuple):
        return [_flattened(item, reserved) for item in value]
    return value


def _state_for_kev(state: dict, budget: int) -> dict:
    """The only path from transcript data to the outgoing state (see above).

    chart_context keys, at any depth, may not equal a state label.
    """
    _raw_state_chars(state, budget)
    return {
        name: _flattened(
            value, _STATE_LABELS if name == "chart_context" else frozenset()
        )
        for name, value in state.items()
    }


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
            # Raw here; budgeted, validated and flattened by _state_for_kev below.
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
        payload["state"] = _state_for_kev(payload["state"], Config.KEV_MAX_INPUT_CHARS)
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
                # aiter_raw never decodes, so the cap counts wire bytes and an
                # oversized body is rejected without buffering it in full, even
                # if the identity check above were ever bypassed.
                chunks, size = [], 0
                async for chunk in response.aiter_raw():
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
    except _InvalidInput:
        error = "Kev input has a value that cannot be sent as text; not sent"
        error_reason = "invalid_input"
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
