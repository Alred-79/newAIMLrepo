"""
ClinEQ Shared Configuration
All tunable parameters in one place. Never hardcode these values elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()


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
