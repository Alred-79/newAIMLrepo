"""
ClinEQ — LLM Call Wrapper

Thin wrapper around LiteLLM that adds Ollama structured output support.
All LLM calls in the project should go through these functions.
"""

import json
from typing import Optional

import litellm
from shared.config import Config

# JSON schemas for structured output enforcement
BIAS_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "bias_label": {
            "type": "string",
            "enum": ["biased", "unbiased", "uncertain"]
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["bias_label", "confidence", "reasoning"],
}

HEAVYWEIGHT_REALTIME_SCHEMA = {
    "type": "object",
    "properties": {
        "bias_label": {
            "type": "string",
            "enum": ["biased", "unbiased", "uncertain"]
        },
        "confidence": {"type": "number"},
        "is_clinical_decision": {"type": "boolean"},
        "decision_description": {"type": "string"},
        "bias_pattern": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["bias_label", "confidence", "is_clinical_decision", "reasoning"],
}

FALLBACK_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "node_id": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["node_id", "reasoning"],
}


async def llm_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
    json_schema: Optional[dict] = None,
) -> str:
    """
    Make an LLM completion call via LiteLLM -> Ollama.

    If json_schema is provided AND the model is an Ollama model,
    uses Ollama's native structured output (format parameter) for
    guaranteed valid JSON. Otherwise falls back to standard completion.

    Args:
        model: LiteLLM model string (e.g., "ollama_chat/qwen2.5:14b")
        system_prompt: System message content.
        user_prompt: User message content.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        json_schema: Optional JSON schema dict for structured output.

    Returns:
        The raw string content of the model's response.
    """
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_base": Config.OLLAMA_BASE_URL,
    }

    # Use Ollama native structured output if available
    if json_schema and model.startswith("ollama"):
        kwargs["format"] = json_schema

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content.strip()


async def llm_embedding(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings via LiteLLM -> Ollama.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    response = await litellm.aembedding(
        model=Config.EMBEDDING_MODEL,
        input=texts,
        api_base=Config.OLLAMA_BASE_URL,
    )
    return [item["embedding"] for item in response.data]
