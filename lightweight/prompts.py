"""
Prompt utilities for the lightweight evaluator.

These prompts are:
- Scenario-aware
- Focused on professionalism, empathy, and bias/stigma detection
- Designed to force STRICT JSON output from the LLM
"""

from __future__ import annotations

from typing import Dict, List


SCENARIO_CONTEXT: Dict[str, str] = {
    "control": (
        "You are evaluating a medical student's message during a neutral adult primary "
        "care visit. No special bias is expected from the patient. "
        "Use this as a baseline for professional, empathic, and clear communication."
    ),
    "ageism": (
        "You are evaluating a medical student's message during an encounter with a 93-year-old "
        "patient who sometimes expresses racist or provocative comments. The student must remain "
        "professional, non-defensive, non-ageist, and avoid assumptions about cognitive ability, "
        "frailty, or worth based on age. Pay special attention to ageist stereotypes, "
        "patronizing tone, dismissiveness, or unprofessional responses to racist comments."
    ),
    "zionist": (
        "You are evaluating a medical student's message during an encounter with a young "
        "Orthodox Jewish college student distressed about geopolitical events and personal safety. "
        "The patient expresses fear tied to identity (e.g., Zionist, Israeli, Jewish) and may express "
        "bias toward others. You MUST NOT treat religious, ethnic, or political identity itself as a "
        "pathology. Focus instead on whether the student's response is professional, empathic, and "
        "avoids stereotyping or hostility toward any group."
    ),
}


def build_evaluation_messages(
    message: str,
    scenario: str,
    stigma_flags: List[str],
) -> List[Dict[str, str]]:
    scenario_key = scenario if scenario in SCENARIO_CONTEXT else "control"
    scenario_text = SCENARIO_CONTEXT[scenario_key]

    flags_text = ", ".join(stigma_flags) if stigma_flags else "none"

    system_prompt = f"""
You are a clinical communication evaluator for a medical education simulation.

{scenario_text}
Your job is to review ONE message written by a medical student to a patient and
decide whether the message is:
- "safe": professionally appropriate, non-stigmatizing, and reasonably empathic
- "concerning": minor lapses (e.g., mildly dismissive, limited empathy) but not clearly harmful
- "harmful": clearly unprofessional, stigmatizing, discriminatory, or likely to damage trust

You must:
- Consider professionalism, empathy, clarity, and clinical appropriateness
- Pay attention to bias / stigma related to age, race, religion, gender, politics, or identity
- Use the provided LWIC-style tags: professional, reassuring, dismissive, ageist, defensive,
  rude, unprofessional, political, accusatory

CRITICAL SCENARIO RULES:
- Ageism scenario: heavily penalize ageist statements (e.g., assuming cognitive decline,
  dismissing concerns as “just aging”, implying the patient is a burden, or talking down).
- Zionist scenario: DO NOT pathologize identity (e.g., being Jewish, Israeli, or Zionist).
  Only treat hostility or stereotyping toward any group as concerning or harmful.
- Control scenario: use as a baseline; avoid over-flagging minor imperfections.

OUTPUT FORMAT (STRICT):
You MUST respond with ONLY a single JSON object, no prose, no Markdown, no backticks.
The JSON schema is:
{{
  "label": "safe" | "concerning" | "harmful",
  "confidence": float,  # between 0.0 and 1.0
  "reasoning": "short, plain-language explanation"
}}
""".strip()

    user_prompt = f"""
STUDENT_MESSAGE:
\"\"\"{message.strip()}\"\"\"

DETECTED_STIGMA_FLAGS (heuristic, may be incomplete):
{flags_text}

Return ONLY the JSON object described above.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


__all__ = ["SCENARIO_CONTEXT", "build_evaluation_messages"]


