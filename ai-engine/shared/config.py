"""
ClinEQ Shared Configuration
All tunable parameters in one place. Never hardcode these values elsewhere.
"""

import ipaddress
import logging
import math
import os
import re
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# Plain ASCII decimal or scientific notation only. float() alone would also
# accept underscores ("1_0"), non-ASCII digits, "nan" and "inf".
_ASCII_DECIMAL = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", re.ASCII
)


def _env_float(name: str, default: float, low: float, high: float) -> float:
    """Strict finite decimal within [low, high]; empty means unset.

    Anything else becomes NaN instead of raising at import, so a bad optional
    setting cannot stop the whole engine; callers reject it at call time
    (evaluate_with_kev abstains with invalid_config).
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    if not _ASCII_DECIMAL.fullmatch(raw):
        return float("nan")
    value = float(raw)
    if not (math.isfinite(value) and low <= value <= high):
        return float("nan")
    return value


_ASCII_INT = re.compile(r"\+?[0-9]+")


def _env_int(name: str, default: int, low: int, high: int) -> float:
    """Strict ASCII decimal integer within [low, high]; empty means unset.

    Anything else (underscores, signs other than +, non-ASCII digits, hex,
    exponents, out of range) becomes NaN so the setting is reported as
    invalid_config instead of being silently reinterpreted.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    if not _ASCII_INT.fullmatch(raw):
        return float("nan")
    value = int(raw)
    return value if low <= value <= high else float("nan")


class Config:
    # API keys (not needed for local mode)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TOGETHERAI_API_KEY: str = os.getenv("TOGETHERAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:dev@localhost:5433/clineq"
    )

    # LLM models (all via LiteLLM → Ollama)
    # Qwen 2.5 is significantly better than Llama 3.1 at:
    #   - Structured JSON output (near-perfect with schema enforcement)
    #   - Classification tasks (fewer false positives on bias detection)
    #   - Instruction following (cleaner wrappers, better length control)
    LIGHTWEIGHT_MODEL: str = "ollama_chat/qwen2.5:14b"
    HEAVYWEIGHT_MODEL: str = "ollama_chat/qwen2.5:14b"   # Post-MVP1
    WRAPPER_MODEL: str = "ollama_chat/qwen2.5:14b"
    FALLBACK_CLASSIFIER_MODEL: str = "ollama_chat/qwen2.5:14b"
    EMBEDDING_MODEL: str = "ollama/mxbai-embed-large"

    # Lightweight evaluator
    # Unset or empty selects the legacy Ollama evaluator; any other unsupported
    # value makes Tier-1 abstain and escalate (see bias_eval.lightweight).
    LIGHTWEIGHT_BACKEND: str = (
        (os.getenv("LIGHTWEIGHT_BACKEND") or "").strip().lower() or "ollama"
    )
    KEV_BASE_URL: str = os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009")
    KEV_API_KEY: str = os.getenv("KEV_API_KEY", "")
    KEV_MODEL: str = os.getenv("KEV_MODEL", "kev-latest")
    # Deployment-declared checkpoint/revision; the API model name is only an alias.
    KEV_CHECKPOINT: str = os.getenv("KEV_CHECKPOINT", "")
    # Whole-request deadline in seconds (connect, send and read included).
    KEV_TIMEOUT_SECONDS_RANGE = (0.05, 60.0)
    KEV_TIMEOUT_SECONDS: float = _env_float(
        "KEV_TIMEOUT_SECONDS", 1.5, *KEV_TIMEOUT_SECONDS_RANGE
    )
    # Largest (decoded) Kev response body read before abstaining as invalid.
    KEV_MAX_RESPONSE_BYTES_RANGE = (1024, 64 * 1024 * 1024)
    KEV_MAX_RESPONSE_BYTES: int | float = _env_int(
        "KEV_MAX_RESPONSE_BYTES", 1024 * 1024, *KEV_MAX_RESPONSE_BYTES_RANGE
    )
    # Combined characters of transcript text sent to Kev (student message, recent
    # context, probe text, serialized chart context). Longer input is not sent and
    # abstains with input_too_long. The default is a conservative guard well below
    # Kev's validated 8,192-token window; it is not a tokenizer-exact limit.
    KEV_MAX_INPUT_CHARS_RANGE = (200, 30000)
    KEV_MAX_INPUT_CHARS: int | float = _env_int(
        "KEV_MAX_INPUT_CHARS", 12000, *KEV_MAX_INPUT_CHARS_RANGE
    )
    SUPPORTED_LIGHTWEIGHT_BACKENDS = ("kev", "ollama")
    CONFIDENCE_THRESHOLD: float = 0.80   # RAISED from 0.75 — reduce false positive reflections
    LIGHTWEIGHT_TEMPERATURE: float = 0.0  # LOWERED from 0.1 — max determinism for classification
    LIGHTWEIGHT_MAX_TOKENS: int = 400     # was 500 (Qwen is less verbose than Llama)
    MAX_CONTEXT_EXCHANGES: int = 4

    # Patient simulator
    # Legacy cosine-only thresholds (kept for reference / fallback)
    NODE_MATCH_CONFIDENT: float = 0.60
    NODE_MATCH_AMBIGUOUS: float = 0.35
    RE_ASK_SIMILARITY_THRESHOLD: float = 0.50

    # Hybrid retrieval (RRF)
    # Reciprocal Rank Fusion merges semantic (pgvector) and BM25 ranked lists.
    # RRF scores are small numbers — typical range 0.008 to 0.033.
    # These thresholds MUST be calibrated with the diagnostic script.
    RRF_K: int = 60                           # Standard RRF constant (no tuning needed)
    RRF_CONFIDENT_THRESHOLD: float = 0.029    # Above this: confident match
    RRF_AMBIGUOUS_THRESHOLD: float = 0.012    # Below this: no match. Between: LLM fallback.
    RRF_REASK_THRESHOLD: float = 0.020        # RRF score for re-ask detection
    AMBIGUOUS_THRESHOLD_MULTIPLIER: float = 1.2  # Stricter threshold for ambiguous messages
    WRAPPER_TEMPERATURE: float = 0.5      # RAISED from 0.4 — more variety with no-repeat constraint
    # REACTION_TEMPERATURE — REMOVED (no more bias reactions)
    MAX_WRAPPER_TOKENS: int = 30
    FALLBACK_CLASSIFIER_TEMPERATURE: float = 0.0  # LOWERED — max determinism
    FALLBACK_CLASSIFIER_MAX_TOKENS: int = 100
    MAX_CONVERSATION_LENGTH: int = 40

    # Timed node delivery
    TIMED_NODE_FALLBACK_TURNS: int = 3    # NEW — fire timed node after this many turns of readiness

    # Embedding
    EMBEDDING_DIMENSIONS: int = 1024      # CHANGED from 1536 — mxbai-embed-large is 1024

    # MVP1 flags
    HEAVYWEIGHT_ENABLED: bool = True   # CHANGED from False — now active
    HEAVYWEIGHT_TEMPERATURE: float = 0.1
    HEAVYWEIGHT_MAX_TOKENS: int = 800
    HEAVYWEIGHT_CONFIDENCE_THRESHOLD: float = 0.75  # For real-time mode reflection triggers
    COMPOUND_MATCHING_ENABLED: bool = False  # MVP1: top match only
    PREREQUISITE_NODES_ENABLED: bool = False  # MVP1: unused


UrlRejectReason = Literal[
    "scheme", "query", "fragment", "userinfo", "host", "port", "malformed"
]


def kev_base_url_is_allowed(url: object) -> tuple[bool, UrlRejectReason | None]:
    """Return (True, None) when the URL is usable, else (False, reason).

    https anywhere; plain http only to a loopback host (localhost, 127/8, ::1).

    Rejected, so they are reported as a local configuration error rather than
    a Kev failure: empty, relative or malformed URLs; userinfo (user:pass@);
    any query or fragment ("?" or "#", which would swallow the /v1/systemone
    suffix); non-ASCII characters (in the host: "host"; elsewhere, such as
    the path: "malformed"), and IDNA (xn--) host labels; an empty, zero,
    non-numeric or out-of-range port; "%" in the host (IPv6 zone ids,
    percent-encoded hosts); and
    IPv4-mapped IPv6 addresses such as [::ffff:127.0.0.1] for plain http
    (Python 3.13 calls that loopback, 3.11/3.12 do not; rejected on all).
    The reason is a fixed category, never any part of the URL.
    """
    if not isinstance(url, str) or not url.strip():
        return False, "malformed"
    url = url.strip()
    if "?" in url:
        return False, "query"
    if "#" in url:
        return False, "fragment"
    try:
        parts = urlsplit(url)
    except ValueError:
        return False, "malformed"
    try:
        port = parts.port
        bad_port = False
    except ValueError:  # non-numeric or out-of-range (> 65535) port
        port = None
        bad_port = True
    host = parts.hostname
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return False, "scheme"
    if not host:
        return False, "malformed"
    # userinfo would be sent as Basic auth, replacing the Bearer key.
    if "@" in parts.netloc:
        return False, "userinfo"
    if not parts.netloc.isascii() or "%" in parts.netloc:
        return False, "host"
    if bad_port or port == 0 or (port is None and parts.netloc.endswith(":")):
        return False, "port"
    if not url.isascii():  # non-ASCII outside the host (the path)
        return False, "malformed"
    if any(label.startswith("xn--") for label in host.split(".")):
        return False, "host"
    if scheme == "https" or host == "localhost":
        return True, None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A named host that is not localhost: https returned above, so this is
        # plain http to a non-loopback host.
        return False, "scheme"
    if address.version == 6 and address.ipv4_mapped is not None:
        return False, "host"
    if not address.is_loopback:
        return False, "scheme"
    return True, None


def _int_in_range(value: object, bounds: tuple[int, int]) -> bool:
    return type(value) is int and bounds[0] <= value <= bounds[1]


def kev_timeout_is_valid(value: object) -> bool:
    low, high = Config.KEV_TIMEOUT_SECONDS_RANGE
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def kev_max_response_bytes_is_valid(value: object) -> bool:
    return _int_in_range(value, Config.KEV_MAX_RESPONSE_BYTES_RANGE)


def kev_max_input_chars_is_valid(value: object) -> bool:
    return _int_in_range(value, Config.KEV_MAX_INPUT_CHARS_RANGE)


def _warn_on_invalid_settings() -> None:
    """Warn once, at import, about settings that make Tier-1 abstain on every turn.

    Only the setting's name and a short reason are logged, never its value.
    """
    if Config.LIGHTWEIGHT_BACKEND not in Config.SUPPORTED_LIGHTWEIGHT_BACKENDS:
        logger.warning(
            "LIGHTWEIGHT_BACKEND is not supported (use kev or ollama); "
            "Tier-1 will abstain and escalate every turn"
        )
    if Config.LIGHTWEIGHT_BACKEND == "kev":
        ok, _ = kev_base_url_is_allowed(Config.KEV_BASE_URL)
        if not ok:
            logger.warning(
                "KEV_BASE_URL must be an absolute https URL (plain http only to "
                "a loopback host) with an ASCII host and no query or fragment; "
                "Kev requests will abstain with invalid_config"
            )
        if not kev_max_response_bytes_is_valid(Config.KEV_MAX_RESPONSE_BYTES):
            logger.warning(
                "KEV_MAX_RESPONSE_BYTES is not a plain integer in the allowed "
                "range; Kev requests will abstain with invalid_config"
            )
        if not kev_max_input_chars_is_valid(Config.KEV_MAX_INPUT_CHARS):
            logger.warning(
                "KEV_MAX_INPUT_CHARS is not a plain integer in the allowed "
                "range; Kev requests will abstain with invalid_config"
            )
        if not kev_timeout_is_valid(Config.KEV_TIMEOUT_SECONDS):
            # No digits in the message, so a rejected value can never appear.
            logger.warning(
                "KEV_TIMEOUT_SECONDS is not a plain decimal number of seconds in "
                "the allowed range; Kev requests will abstain with invalid_config"
            )


_warn_on_invalid_settings()
