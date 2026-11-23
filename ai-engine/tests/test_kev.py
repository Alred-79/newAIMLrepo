"""Offline protocol/routing tests, not clinical accuracy or calibration evidence."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from bias_eval.kev import evaluate_with_kev
from bias_eval.kev_questions import build_questions
from shared.config import Config
from shared.models import (
    BiasDirection,
    BiasLabel,
    EvaluationInput,
    Message,
    ScenarioCategory,
)
from shared.tier1_models import ProbeAnchor


@pytest.fixture(autouse=True)
def kev_config(monkeypatch):
    monkeypatch.setattr(Config, "KEV_BASE_URL", "http://kev.test:8009/")
    monkeypatch.setattr(Config, "KEV_MODEL", "kev-latest")
    monkeypatch.setattr(Config, "KEV_CHECKPOINT", "test-checkpoint@revision")
    monkeypatch.setattr(Config, "KEV_API_KEY", "")
    monkeypatch.setattr(Config, "KEV_TIMEOUT_SECONDS", 1.5)


@pytest.fixture
def turn():
    return EvaluationInput(
        student_message="Can you tell me more about the pain?",
        recent_context=[Message(role="patient", content="I have chest pain.")],
        scenario_category=ScenarioCategory.WEIGHT,
        chart_context={"symptom_duration": "two weeks"},
    )


def response_for(questions, choices=None):
    """Synthetic wire responses for contract tests; never gold labels."""
    choices = {"tone": "unbiased", **(choices or {})}
    answers = {}
    for name, question in questions.items():
        criteria = question["criteria"]
        if question["type"] == "score":
            answers[name] = {
                "type": "score",
                "score": 1.0,
                "confidence": 0.925,
                "legend": {str(i): label for i, label in enumerate(criteria)},
                "probabilities": {"0": 0.025, "1": 0.95, "2": 0.025},
            }
        else:
            selected = choices.get(name, next(iter(criteria)))
            count = len(criteria)
            answers[name] = {
                "type": "choice",
                "choice": selected,
                "confidence": (0.95 - 1 / count) / (1 - 1 / count),
                "probabilities": {
                    label: 0.95 if label == selected else 0.05 / (count - 1)
                    for label in criteria
                },
            }
    return {"model": "kev-latest", "answers": answers}


async def classify(turn, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await evaluate_with_kev(turn, client=client)
        assert not client.is_closed  # Caller retains ownership of a pooled client.
        return result


@pytest.mark.asyncio
async def test_request_and_independent_probabilities(turn, monkeypatch):
    monkeypatch.setattr(Config, "KEV_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "http://kev.test:8009/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["state"]["student_message"] == turn.student_message
        assert payload["state"]["recent_context"][0]["role"] == "patient"
        assert payload["state"]["chart_context"] == turn.chart_context
        assert payload["model"] == "kev-latest"
        return httpx.Response(
            200,
            json=response_for(
                payload["questions"],
                {
                    "attribution": "stereotype_linked",
                    "affective_stance": "cold",
                },
            ),
        )

    result = await classify(turn, handler)
    assert len(requests) == 1
    assert result.tone_label == BiasLabel.UNBIASED
    assert result.should_escalate  # Flagged axes request review without changing tone.
    assert result.tier1.answers["attribution"].choice == "stereotype_linked"
    assert result.tier1.answers["affective_stance"].choice == "cold"
    assert result.tier1.status == "ok"
    assert result.tier1.checkpoint == "test-checkpoint@revision"
    assert result.tier1.prompt_hash.startswith("sha256:")
    assert result.tier1.latency_ms >= 0
    assert "test-key" not in result.model_dump_json()
    assert "clinical_action" not in result.tier1.answers


@pytest.mark.asyncio
async def test_patient_bias_response_is_separate_from_student_tone(turn):
    turn.bias_direction = BiasDirection.PATIENT_TO_STUDENT
    body = response_for(
        build_questions(turn), {"patient_bias_response": "name_and_limit"}
    )
    result = await classify(turn, lambda _: httpx.Response(200, json=body))
    assert result.tone_label == BiasLabel.UNBIASED
    assert result.tier1.answers["patient_bias_response"].choice == "name_and_limit"
    turn.bias_direction = BiasDirection.STUDENT_TO_PATIENT
    assert "patient_bias_response" not in build_questions(turn)


@pytest.mark.asyncio
async def test_attention_requires_anchor_and_preserves_both_tails(turn):
    assert "probe_attention" not in build_questions(turn)
    turn.probe = ProbeAnchor(turn_idx=3, text="I worry you will blame my weight.")
    body = response_for(build_questions(turn))
    body["answers"]["probe_attention"].update(
        score=1.0, confidence=0.0, probabilities={"0": 0.5, "1": 0.0, "2": 0.5}
    )

    def handler(request):
        assert json.loads(request.content)["state"]["probe"] == turn.probe.model_dump()
        return httpx.Response(200, json=body)

    result = await classify(turn, handler)
    attention = result.tier1.answers["probe_attention"]
    assert attention.probabilities == {"0": 0.5, "1": 0.0, "2": 0.5}
    assert result.should_escalate  # A mean of 1 does not imply appropriate attention.


@pytest.mark.parametrize("signal", ["tone", "attribution", "epistemic_stance"])
@pytest.mark.asyncio
async def test_uncertainty_on_any_question_escalates(turn, signal):
    body = response_for(build_questions(turn), {signal: "uncertain"})
    result = await classify(turn, lambda _: httpx.Response(200, json=body))
    assert result.should_escalate


@pytest.mark.asyncio
async def test_low_axis_confidence_escalates_even_with_confident_tone(turn):
    body = response_for(build_questions(turn))
    body["answers"]["affective_stance"]["confidence"] = 0.1
    result = await classify(turn, lambda _: httpx.Response(200, json=body))
    assert result.tone_confidence > Config.CONFIDENCE_THRESHOLD
    assert result.should_escalate


@pytest.mark.parametrize(
    "name,label",
    [
        ("tone", "biased"),
        ("attribution", "stereotype_linked"),
        ("epistemic_stance", "doubts"),
        ("epistemic_stance", "overrides"),
        ("affective_stance", "cold"),
        ("affective_stance", "hostile"),
    ],
)
@pytest.mark.asyncio
async def test_confident_flags_request_review(turn, name, label):
    body = response_for(build_questions(turn), {name: label})
    result = await classify(turn, lambda _: httpx.Response(200, json=body))
    assert result.should_escalate
    assert result.tier1.status == "ok"


@pytest.mark.asyncio
async def test_confident_neutral_signals_do_not_escalate(turn):
    body = response_for(build_questions(turn))
    result = await classify(turn, lambda _: httpx.Response(200, json=body))
    assert not result.should_escalate
    assert result.tone_label == BiasLabel.UNBIASED


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
@pytest.mark.asyncio
async def test_invalid_timeout_abstains_before_sending(turn, monkeypatch, timeout):
    monkeypatch.setattr(Config, "KEV_TIMEOUT_SECONDS", timeout)

    def handler(request):
        pytest.fail("An invalid timeout must not send a request")

    result = await classify(turn, handler)
    assert result.tier1.status == "unavailable"
    assert result.should_escalate


def test_sync_dispatch_closes_owned_http_client(turn, monkeypatch):
    from bias_eval import kev, lightweight

    clients = []
    client_class = httpx.AsyncClient

    def factory(**kwargs):
        client = client_class(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=response_for(build_questions(turn)))
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(Config, "LIGHTWEIGHT_BACKEND", "kev")
    monkeypatch.setattr(kev.httpx, "AsyncClient", factory)
    for _ in range(2):
        result = lightweight.evaluate_tone_sync(turn)
        assert result.tier1.status == "ok"
    assert len(clients) == 2
    assert all(client.is_closed for client in clients)


def corrupt(body, case):
    tone = body["answers"]["tone"]
    attention = body["answers"]["probe_attention"]
    if case == "missing_answer":
        del body["answers"]["attribution"]
    elif case == "extra_answer":
        body["answers"]["extra"] = tone
    elif case == "unknown_option":
        tone["probabilities"]["unexpected"] = tone["probabilities"].pop("biased")
    elif case == "unknown_choice":
        tone["choice"] = "unexpected"
    elif case == "not_argmax":
        tone["choice"] = "biased"
    elif case == "not_normalized":
        tone["probabilities"]["unbiased"] = 0.1
    elif case == "boolean_probability":
        tone["probabilities"]["unbiased"] = True
    elif case == "negative_probability":
        tone["probabilities"]["biased"] = -0.1
    elif case == "confidence_range":
        tone["confidence"] = 1.1
    elif case == "missing_confidence":
        del tone["confidence"]
    elif case == "wrong_type":
        tone["type"] = "noul"
    elif case == "wrong_model":
        body["model"] = "unexpected"
    elif case == "truncated":
        body["truncated"] = True
    elif case == "score_range":
        attention["score"] = 3.0
    elif case == "score_mean":
        attention["score"] = 0.0
    elif case == "score_legend":
        attention["legend"]["0"] = "Unexpected meaning"
    elif case == "score_missing_level":
        del attention["probabilities"]["2"]
    return body


@pytest.mark.parametrize(
    "case",
    [
        "missing_answer",
        "extra_answer",
        "unknown_option",
        "unknown_choice",
        "not_argmax",
        "not_normalized",
        "boolean_probability",
        "negative_probability",
        "confidence_range",
        "missing_confidence",
        "wrong_type",
        "wrong_model",
        "truncated",
        "score_range",
        "score_mean",
        "score_legend",
        "score_missing_level",
    ],
)
@pytest.mark.asyncio
async def test_malformed_responses_abstain(turn, case):
    turn.probe = ProbeAnchor(turn_idx=3, text="I am worried.")
    body = corrupt(response_for(build_questions(turn)), case)
    result = await classify(turn, lambda _: httpx.Response(200, json=body))
    assert result.tone_label == BiasLabel.UNCERTAIN
    assert result.tone_confidence == 0.0
    assert result.should_escalate
    assert result.tier1.status == "unavailable"
    assert result.tier1.answers == {}


@pytest.mark.parametrize(
    "content", [b"not json", b"null", b"[]", b'{"model":"kev-latest","answers":null}']
)
@pytest.mark.asyncio
async def test_invalid_json_and_shapes_abstain(turn, content):
    result = await classify(turn, lambda _: httpx.Response(200, content=content))
    assert result.tone_label == BiasLabel.UNCERTAIN
    assert result.should_escalate


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.asyncio
async def test_nonfinite_probabilities_abstain(turn, value):
    body = response_for(build_questions(turn))
    body["answers"]["tone"]["confidence"] = value
    result = await classify(
        turn, lambda _: httpx.Response(200, content=json.dumps(body))
    )
    assert result.tier1.status == "unavailable"


@pytest.mark.parametrize("status", [401, 422, 429, 500])
@pytest.mark.asyncio
async def test_http_errors_abstain_without_retries_or_echoing_body(turn, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private transcript and secret")

    result = await classify(turn, handler)
    assert len(calls) == 1
    assert result.tone_label == BiasLabel.UNCERTAIN
    assert result.should_escalate
    assert str(status) in result.tier1.error
    assert "private transcript" not in result.model_dump_json()


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
@pytest.mark.asyncio
async def test_transport_failures_abstain(turn, error):
    def handler(request):
        raise error("secret", request=request)

    result = await classify(turn, handler)
    assert result.should_escalate
    assert result.tier1.status == "unavailable"
    assert "secret" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_deadline_cancels_slow_request(turn, monkeypatch):
    monkeypatch.setattr(Config, "KEV_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()

    async def handler(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    result = await asyncio.wait_for(classify(turn, handler), timeout=1)
    assert cancelled.is_set()
    assert result.tier1.error == "Kev request timed out"


@pytest.mark.asyncio
async def test_external_cancellation_propagates(turn):
    def handler(request):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await classify(turn, handler)


@pytest.mark.asyncio
async def test_hash_tracks_question_bank_not_transcript(turn):
    def handler(request):
        assert "Authorization" not in request.headers
        return httpx.Response(
            200, json=response_for(json.loads(request.content)["questions"])
        )

    first = await classify(turn, handler)
    turn.student_message = "Different text."
    second = await classify(turn, handler)
    assert first.tier1.prompt_hash == second.tier1.prompt_hash
    turn.probe = ProbeAnchor(turn_idx=4, text="A concern.")
    third = await classify(turn, handler)
    assert first.tier1.prompt_hash != third.tier1.prompt_hash


@pytest.mark.asyncio
async def test_dispatch_and_simulator_respect_axis_escalation(turn, monkeypatch):
    from bias_eval import kev, lightweight
    from patient_sim import simulator

    body = response_for(
        build_questions(turn), {"tone": "biased", "attribution": "uncertain"}
    )
    expected = await classify(turn, lambda _: httpx.Response(200, json=body))
    monkeypatch.setattr(Config, "LIGHTWEIGHT_BACKEND", "kev")
    mocked = AsyncMock(return_value=expected)
    monkeypatch.setattr(kev, "evaluate_with_kev", mocked)
    legacy = AsyncMock(side_effect=AssertionError("Kev must not invoke a chat model"))
    monkeypatch.setattr(lightweight, "llm_completion", legacy)
    assert await lightweight.evaluate_tone(turn) == expected
    sim_input = SimpleNamespace(
        student_message=turn.student_message,
        state=SimpleNamespace(conversation_history=turn.recent_context),
        scenario=SimpleNamespace(
            category=turn.scenario_category, bias_direction=turn.bias_direction
        ),
    )
    messages, escalate = await simulator._run_lightweight(sim_input)
    assert messages == []  # Raw Kev flags do not trigger student-facing feedback.
    assert escalate
    legacy.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_backend_and_unknown_backend(turn, monkeypatch):
    from bias_eval import lightweight

    legacy = AsyncMock(
        return_value='{"bias_label":"unbiased","confidence":0.9,"reasoning":"Neutral"}'
    )
    monkeypatch.setattr(lightweight, "llm_completion", legacy)
    monkeypatch.setattr(Config, "LIGHTWEIGHT_BACKEND", "ollama")
    result = await lightweight.evaluate_tone(turn)
    assert result.tone_label == BiasLabel.UNBIASED
    assert result.tier1 is None
    legacy.assert_awaited_once()
    monkeypatch.setattr(Config, "LIGHTWEIGHT_BACKEND", "typo")
    result = await lightweight.evaluate_tone(turn)
    assert result.tone_label == BiasLabel.UNCERTAIN
    assert result.should_escalate
    legacy.assert_awaited_once()
