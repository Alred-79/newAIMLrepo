"""Exception text can carry student words; it must not reach output or stored fields.

Each case raises an exception whose message is a canary standing in for a
student's words, then checks stdout, stderr, every log record (DEBUG and up,
including tracebacks) and the returned or stored fields. The exception class
name is still recorded so failures stay diagnosable.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.config import Config
from shared.models import (
    BiasDirection,
    EvaluationInput,
    Message,
    ScenarioCategory,
)

CANARY = "student said canary qx81"


def _assert_no_canary(capsys, caplog, *stored):
    out, err = capsys.readouterr()
    assert CANARY not in out
    assert CANARY not in err
    for record in caplog.records:
        assert CANARY not in record.getMessage()
        assert CANARY not in (record.exc_text or "")
        assert record.exc_info is None
    for value in stored:
        assert CANARY not in str(value)


def _history():
    return [Message(role="student", content="hello")]


def _scenario(direction=BiasDirection.STUDENT_TO_PATIENT):
    return SimpleNamespace(
        title="t",
        category=ScenarioCategory.WEIGHT,
        bias_direction=direction,
        patient_profile=SimpleNamespace(name="p"),
        content_nodes=[],
    )


@pytest.mark.asyncio
async def test_simulator_heavyweight_error_is_logged_by_class_only(
    monkeypatch, capsys, caplog
):
    from patient_sim import simulator

    monkeypatch.setattr(Config, "HEAVYWEIGHT_ENABLED", True)
    monkeypatch.setattr(
        simulator, "evaluate_realtime", AsyncMock(side_effect=RuntimeError(CANARY))
    )
    sim_input = SimpleNamespace(
        student_message="hello",
        state=SimpleNamespace(conversation_history=_history()),
        scenario=_scenario(),
    )
    with caplog.at_level("DEBUG"):
        messages = await simulator._run_heavyweight(sim_input)

    assert messages == []
    _assert_no_canary(capsys, caplog)
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


def _tone_input():
    return EvaluationInput(
        student_message="hello",
        recent_context=_history(),
        scenario_category=ScenarioCategory.WEIGHT,
    )


@pytest.mark.asyncio
async def test_lightweight_llm_error_is_not_stored(monkeypatch, capsys, caplog):
    from bias_eval import lightweight

    monkeypatch.setattr(Config, "LIGHTWEIGHT_BACKEND", "ollama")
    monkeypatch.setattr(
        lightweight, "llm_completion", AsyncMock(side_effect=RuntimeError(CANARY))
    )
    with caplog.at_level("DEBUG"):
        result = await lightweight.evaluate_tone(_tone_input())

    _assert_no_canary(capsys, caplog, result.model_dump())
    assert result.should_escalate is True
    assert "RuntimeError" in result.brief_reason


@pytest.mark.asyncio
async def test_lightweight_parse_error_is_not_stored_in_brief_reason(
    monkeypatch, capsys, caplog
):
    from bias_eval import lightweight

    monkeypatch.setattr(Config, "LIGHTWEIGHT_BACKEND", "ollama")
    # Model output the parser cannot use (it once quoted it in the error).
    monkeypatch.setattr(
        lightweight, "llm_completion", AsyncMock(return_value=f"no json {CANARY}")
    )
    with caplog.at_level("DEBUG"):
        result = await lightweight.evaluate_tone(_tone_input())

    # raw_llm_response keeps the model's text by existing design; only the
    # status field is checked here.
    _assert_no_canary(capsys, caplog, result.brief_reason)
    assert result.should_escalate is True
    assert "ValueError" in result.brief_reason


@pytest.mark.asyncio
async def test_heavyweight_realtime_error_is_not_stored(monkeypatch, capsys, caplog):
    from bias_eval import heavyweight

    monkeypatch.setattr(
        heavyweight, "llm_completion", AsyncMock(side_effect=RuntimeError(CANARY))
    )
    with caplog.at_level("DEBUG"):
        result = await heavyweight.evaluate_realtime(
            student_message="hello",
            conversation_history=_history(),
            scenario=_scenario(),
        )

    _assert_no_canary(capsys, caplog, result.model_dump())
    assert result.confidence == 0.0
    assert "RuntimeError" in result.reasoning


@pytest.mark.asyncio
async def test_heavyweight_control_profile_error_is_not_stored(
    monkeypatch, capsys, caplog
):
    from bias_eval import heavyweight

    monkeypatch.setattr(
        heavyweight, "llm_completion", AsyncMock(side_effect=RuntimeError(CANARY))
    )
    with caplog.at_level("DEBUG"):
        profile = await heavyweight.generate_control_profile(_history())

    _assert_no_canary(capsys, caplog, profile)
    assert profile["total_questions"] == 0
    assert "RuntimeError" in profile["error"]


@pytest.mark.asyncio
async def test_heavyweight_delta_error_is_not_stored(monkeypatch, capsys, caplog):
    from bias_eval import heavyweight

    monkeypatch.setattr(
        heavyweight, "llm_completion", AsyncMock(side_effect=RuntimeError(CANARY))
    )
    with caplog.at_level("DEBUG"):
        delta = await heavyweight.generate_delta_report(
            control_history=_history(),
            experimental_history=_history(),
            control_scenario=_scenario(BiasDirection.NONE),
            experimental_scenario=_scenario(),
        )

    _assert_no_canary(capsys, caplog, delta.model_dump())
    assert "RuntimeError" in delta.overall_summary


@pytest.mark.parametrize("control", [True, False], ids=["control", "delta"])
@pytest.mark.asyncio
async def test_session_end_errors_are_not_stored(monkeypatch, capsys, caplog, control):
    from bias_eval import heavyweight
    from shared.session import SessionManager

    failing = AsyncMock(side_effect=RuntimeError(CANARY))
    if control:
        monkeypatch.setattr(heavyweight, "generate_control_profile", failing)
        session = SessionManager(_scenario(BiasDirection.NONE))
        key = "control_profile_error"
    else:
        monkeypatch.setattr(heavyweight, "generate_delta_report", failing)
        session = SessionManager(
            _scenario(),
            control_history=_history(),
            control_scenario=_scenario(BiasDirection.NONE),
        )
        key = "delta_analysis_error"
    with caplog.at_level("DEBUG"):
        result = await session.end_encounter()

    failing.assert_awaited_once()
    _assert_no_canary(capsys, caplog, result)
    assert "RuntimeError" in result["metadata"][key]


def _load_demo_app():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "app.py"
    spec = importlib.util.spec_from_file_location("clineq_demo_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_websocket_processing_error_sends_only_a_fixed_message_and_id(
    monkeypatch, capsys, caplog
):
    import re

    from fastapi.testclient import TestClient

    demo = _load_demo_app()

    class FailingSession:
        def __init__(self, *args, **kwargs):
            pass

        async def process_student_message(self, content):
            raise RuntimeError(f"{CANARY} {content}")

    monkeypatch.setitem(
        demo.SCENARIOS, "s1", SimpleNamespace(patient_profile=SimpleNamespace(name="p"))
    )
    monkeypatch.setattr(demo, "SessionManager", FailingSession)
    with (
        caplog.at_level("DEBUG"),
        TestClient(demo.app).websocket_connect("/ws/chat/s1") as ws,
    ):
        ws.receive_json()  # welcome message
        ws.send_json({"type": "message", "content": CANARY})
        payload = ws.receive_json()
    assert payload["type"] == "error"
    assert CANARY not in str(payload)
    assert re.fullmatch(r"[0-9a-f]{12}", payload["error_id"])
    assert payload["message"] == (
        f"Processing error (reference {payload['error_id']}). Please try again."
    )
    logged = caplog.text
    assert payload["error_id"] in logged
    assert "RuntimeError" in logged
    _assert_no_canary(capsys, caplog, payload)


def test_extract_json_error_does_not_quote_model_text():
    from shared.utils import extract_json

    with pytest.raises(ValueError) as info:
        extract_json(f"no json here: {CANARY}")
    assert CANARY not in str(info.value)
    assert CANARY not in repr(info.value)
