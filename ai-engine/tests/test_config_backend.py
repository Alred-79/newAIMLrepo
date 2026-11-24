"""Config is read once at import, so each case imports it in a fresh interpreter.

load_dotenv is replaced with a no-op so a developer's .env cannot affect results;
no network, model or database is touched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
sys.path.insert(0, sys.argv[1])
from shared.config import Config
print(json.dumps({
    "backend": Config.LIGHTWEIGHT_BACKEND,
    "timeout": repr(Config.KEV_TIMEOUT_SECONDS),
}))
"""


def load_config(**overrides):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("KEV_", "LIGHTWEIGHT_"))
    }
    env.update(overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(ENGINE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # the assert below reports stderr on failure
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "overrides",
    [{}, {"LIGHTWEIGHT_BACKEND": ""}, {"LIGHTWEIGHT_BACKEND": "  "}],
    ids=["unset", "empty", "blank"],
)
def test_unset_or_empty_backend_defaults_to_ollama(overrides):
    assert load_config(**overrides)["backend"] == "ollama"


@pytest.mark.parametrize("value", ["kev", " KEV ", "Ollama"])
def test_backend_is_normalized(value):
    assert load_config(LIGHTWEIGHT_BACKEND=value)["backend"] == value.strip().lower()


def test_unsupported_backend_is_kept_so_tier1_abstains():
    assert load_config(LIGHTWEIGHT_BACKEND="kevv")["backend"] == "kevv"


def test_env_template_matches_code_default():
    template = (ENGINE.parent / ".env.template").read_text()
    active = [
        line.split("=", 1)
        for line in template.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    settings = {k.strip(): v.strip() for k, v in active}
    assert settings.get("LIGHTWEIGHT_BACKEND") == "ollama"
    assert not any(k.startswith("KEV_") for k in settings)  # Kev is opt-in.


@pytest.mark.parametrize("value", ["abc", "1.5s"])
def test_unparseable_kev_timeout_does_not_crash_import(value):
    # Config is imported by every engine entry point, including the default
    # ollama backend, so a bad Kev-only setting must not stop the engine.
    config = load_config(LIGHTWEIGHT_BACKEND="ollama", KEV_TIMEOUT_SECONDS=value)
    assert config["backend"] == "ollama"
    assert config["timeout"] == "nan"  # evaluate_with_kev abstains on non-finite.


def test_valid_kev_timeout_is_parsed():
    assert load_config(KEV_TIMEOUT_SECONDS="4")["timeout"] == "4.0"
    assert load_config()["timeout"] == "1.5"
    assert load_config(KEV_TIMEOUT_SECONDS=" ")["timeout"] == "1.5"  # empty = unset
