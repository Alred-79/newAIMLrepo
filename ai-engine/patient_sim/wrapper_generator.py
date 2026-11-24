"""
ClinEQ — Wrapper Generator

Generates brief neutral transition wrappers. Two modes:
1. Standard — generic content-free filler before answering a question
2. Timed — topic-change signal for patient-initiated content

Bias reactions are removed — the patient responds identically regardless
of whether the evaluator detected bias.
"""

from shared.config import Config
from shared.models import PatientProfile
from patient_sim.prompts import build_wrapper_standard, build_wrapper_timed

try:
    from shared.llm import llm_completion
except ImportError:
    # Fallback if shared/llm.py doesn't exist yet
    import litellm

    async def llm_completion(model, system_prompt, user_prompt, temperature=0, max_tokens=400, json_schema=None):
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=getattr(Config, "OLLAMA_BASE_URL", None),
        )
        return response.choices[0].message.content.strip()


def _clean_wrapper(wrapper: str, max_words: int = 10) -> str:
    """Strip quotes, ellipsis artifacts, and truncate if too long."""
    wrapper = wrapper.strip('"').strip("'").strip()
    if wrapper.endswith("..."):
        wrapper = wrapper[:-3].strip()
        if wrapper:
            wrapper += "."
    words = wrapper.split()
    if len(words) > max_words:
        wrapper = " ".join(words[:max_words - 2])
        if not wrapper.endswith((".", "!", "?", "—")):
            wrapper += "."
    return wrapper if wrapper else "Hmm."


async def generate_wrapper(
    patient_profile: PatientProfile,
    wrapper_history: list[str] | None = None,
) -> str:
    """
    Generate a standard neutral transition wrapper.

    The student's message is NOT passed to the wrapper prompt — the wrapper
    must be content-free and must not echo or restate the student's question.

    Args:
        patient_profile: Patient demographic info (name, age, gender).
        wrapper_history: Last 3 wrappers used (for no-repeat constraint).

    Returns:
        A brief transition remark (1-8 words).
    """
    system_prompt, user_prompt = build_wrapper_standard(
        patient_name=patient_profile.name,
        patient_age=patient_profile.age,
        patient_gender=patient_profile.gender,
        wrapper_history=wrapper_history,
    )

    try:
        wrapper = await llm_completion(
            model=Config.WRAPPER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=Config.WRAPPER_TEMPERATURE,
            max_tokens=Config.MAX_WRAPPER_TOKENS,
        )
        return _clean_wrapper(wrapper)
    except Exception:
        return "Hmm."  # safe fallback


async def generate_timed_wrapper(
    patient_profile: PatientProfile,
    wrapper_history: list[str] | None = None,
) -> str:
    """
    Generate a wrapper for timed (patient-initiated) node delivery.

    Signals a topic change rather than a generic transition.

    Args:
        patient_profile: Patient demographic info (name, age, gender).
        wrapper_history: Last 3 wrappers used (for no-repeat constraint).

    Returns:
        A topic-change transition remark (4-12 words).
    """
    system_prompt, user_prompt = build_wrapper_timed(
        patient_name=patient_profile.name,
        patient_age=patient_profile.age,
        patient_gender=patient_profile.gender,
        wrapper_history=wrapper_history,
    )

    try:
        wrapper = await llm_completion(
            model=Config.WRAPPER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=Config.WRAPPER_TEMPERATURE,
            max_tokens=Config.MAX_WRAPPER_TOKENS,
        )
        return _clean_wrapper(wrapper, max_words=14)
    except Exception:
        return "Oh, by the way —"  # safe fallback
