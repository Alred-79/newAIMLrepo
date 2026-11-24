"""Shared fixtures for the round-6 exception-text and Kev hardening tests."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def flag(s: str) -> str:
    """A black-flag emoji followed by tag characters spelling ``s`` and a cancel tag."""
    return "\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in s) + "\U000E007F"


def _load_demo_app():
    path = Path(__file__).resolve().parents[2] / "app.py"
    spec = importlib.util.spec_from_file_location("clineq_demo_app_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client:
    """A websocket client whose session fails when the encounter ends."""

    def __init__(self, demo):
        self._demo = demo
        self._http = TestClient(demo.app)

    def send_end_and_receive(self) -> dict:
        with self._http.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()  # welcome message
            ws.send_json({"type": "end_encounter"})
            return ws.receive_json()

    def connect_unknown(self, scenario_id: str) -> dict:
        with self._http.websocket_connect(f"/ws/chat/{scenario_id}") as ws:
            return ws.receive_json()


@pytest.fixture
def client_with_failing_end(monkeypatch):
    demo = _load_demo_app()

    class FailingSession:
        def __init__(self, *args, **kwargs):
            pass

        async def end_encounter(self):
            raise RuntimeError("CANARY end: patient said X")

    monkeypatch.setitem(
        demo.SCENARIOS, "s1", SimpleNamespace(patient_profile=SimpleNamespace(name="p"))
    )
    monkeypatch.setattr(demo, "SessionManager", FailingSession)
    return _Client(demo)


@pytest.fixture
def client():
    demo = _load_demo_app()
    return _Client(demo)


@pytest.fixture
def failing_startup(monkeypatch, tmp_path):
    """Run the app startup with a scenario load and a model warmup that both raise."""
    demo = _load_demo_app()
    (tmp_path / "broken.json").write_text("{}")
    monkeypatch.setattr(demo, "SCENARIOS_DIR", tmp_path)

    async def boom_load(path):
        raise RuntimeError("CANARY load: scenario text")

    monkeypatch.setattr(demo, "load_and_seed_scenario", boom_load)

    import shared.llm as llm

    async def boom_llm(*args, **kwargs):
        raise RuntimeError("CANARY warmup: model said Y")

    monkeypatch.setattr(llm, "llm_completion", boom_llm)
    asyncio.run(demo.startup())
    return demo
