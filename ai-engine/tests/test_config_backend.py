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
import json, logging, sys
import dotenv
logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s")
dotenv.load_dotenv = lambda *a, **k: False
sys.path.insert(0, sys.argv[1])
from shared.config import Config
print(json.dumps({
    "backend": Config.LIGHTWEIGHT_BACKEND,
    "timeout": repr(Config.KEV_TIMEOUT_SECONDS),
    "max_bytes": repr(Config.KEV_MAX_RESPONSE_BYTES),
}))
"""


def run_probe(_script=_PROBE, **overrides):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("KEV_", "LIGHTWEIGHT_"))
    }
    env.update(overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _script, str(ENGINE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # the assert below reports stderr on failure
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def load_config(**overrides):
    return run_probe(**overrides)[0]


def warnings_of(stderr):
    return [line for line in stderr.splitlines() if line.startswith("WARNING:")]


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


# A distinctive value lets the tests prove it never reaches the log.
_SENTINEL = "zq7-not-logged"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"LIGHTWEIGHT_BACKEND": "kev"},
        {"LIGHTWEIGHT_BACKEND": " Ollama "},
        {"KEV_TIMEOUT_SECONDS": "4"},
        {"KEV_TIMEOUT_SECONDS": " "},
    ],
    ids=["defaults", "kev", "ollama", "timeout", "blank-timeout"],
)
def test_valid_settings_log_no_warning(overrides):
    assert warnings_of(run_probe(**overrides)[1]) == []


def test_unsupported_backend_logs_one_warning_naming_only_the_setting():
    lines = warnings_of(run_probe(LIGHTWEIGHT_BACKEND=_SENTINEL)[1])
    assert len(lines) == 1
    assert lines[0].startswith("WARNING:shared.config:LIGHTWEIGHT_BACKEND ")
    assert _SENTINEL not in lines[0]


@pytest.mark.parametrize("value", [_SENTINEL, "nan", "inf", "-1", "0"])
def test_invalid_kev_timeout_logs_one_warning_naming_only_the_setting(value):
    stderr = run_probe(LIGHTWEIGHT_BACKEND="kev", KEV_TIMEOUT_SECONDS=value)[1]
    lines = warnings_of(stderr)
    assert len(lines) == 1
    assert lines[0].startswith("WARNING:shared.config:KEV_TIMEOUT_SECONDS ")
    assert value not in lines[0]


def test_invalid_settings_warn_once_per_process_not_per_import():
    # Config is evaluated once per process; importing it again (as every engine
    # module does) must not repeat the warnings.
    script = _PROBE.replace(
        "from shared.config import Config",
        "from shared.config import Config\n"
        "import shared.config\n"
        "from shared.config import Config as _again",
    )
    stderr = run_probe(
        script, LIGHTWEIGHT_BACKEND=_SENTINEL, KEV_TIMEOUT_SECONDS=_SENTINEL
    )[1]
    lines = warnings_of(stderr)
    assert len(lines) == 2
    assert not any(_SENTINEL in line for line in lines)


def test_response_cap_is_parsed_strictly():
    assert load_config()["max_bytes"] == "1048576"
    assert load_config(KEV_MAX_RESPONSE_BYTES=" 2048 ")["max_bytes"] == "2048"
    for value in [
        "1_048_576",
        "1e6",
        "0x400",
        "\u0661\u0660\u0662\u0664",  # Arabic-Indic digits
        "-5",
        "0",
        "10",
        "2048.0",
        str(10**9),
    ]:
        assert load_config(KEV_MAX_RESPONSE_BYTES=value)["max_bytes"] == "nan", value


@pytest.mark.parametrize(
    "name,value",
    [
        ("KEV_BASE_URL", "http://" + _SENTINEL + ".example"),
        ("KEV_BASE_URL", _SENTINEL),
        ("KEV_MAX_RESPONSE_BYTES", _SENTINEL),
    ],
)
def test_invalid_kev_transport_setting_warns_once_only_for_kev(name, value):
    lines = warnings_of(run_probe(LIGHTWEIGHT_BACKEND="kev", **{name: value})[1])
    assert len(lines) == 1
    assert lines[0].startswith(f"WARNING:shared.config:{name} ")
    assert _SENTINEL not in lines[0]
    assert warnings_of(run_probe(**{name: value})[1]) == []  # ollama: not used


@pytest.mark.parametrize(
    "url", ["https://kev.example", "http://127.0.0.1:8009", "http://localhost"]
)
def test_allowed_kev_base_url_logs_no_warning(url):
    assert warnings_of(run_probe(LIGHTWEIGHT_BACKEND="kev", KEV_BASE_URL=url)[1]) == []
