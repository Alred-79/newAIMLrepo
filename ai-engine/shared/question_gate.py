"""
ClinEQ — Question Detection Gate

Fast heuristic classifier. Runs on every student message BEFORE
any embedding, pgvector query, or LLM call. Determines whether
the message is a question (node matching should run) or a
non-question (skip to acknowledgment/closing/timed-node handling).

This is NOT a replacement for the LLM fallback classifier. It is
a fast pre-filter. Messages classified as non-questions here may
still be reclassified by the LLM fallback if needed.
"""

import re


# Phrases that strongly indicate acknowledgment / empathetic statement
_ACKNOWLEDGMENT_PREFIXES = [
    "i'm sorry", "im sorry", "i am sorry",
    "thank you", "thanks",
    "i understand", "i see",
    "that makes sense", "that's understandable",
    "i appreciate", "i hear you",
    "take your time", "no rush",
    "i'm here for you", "im here for you",
    "that must be", "that sounds",
    "okay", "ok", "got it", "understood",
    "alright", "right",
    "i know it can be", "i know that can be",
    "i hope", "good luck",
]

# Phrases that indicate the student wants clarification / repetition.
# These start with interrogative words ("what", "can you") but are NOT
# clinical questions — the student is asking the patient to rephrase,
# not requesting new medical information.  Must be checked BEFORE the
# interrogative-word heuristic.
_CLARIFICATION_PATTERNS = [
    "what do you mean",
    "what does that mean",
    "what does this mean",
    "what is that supposed to mean",
    "what are you talking about",
    "what are you saying",
    "what was that",
    "what did you say",
    "what did you just say",
    "can you explain that",
    "can you clarify",
    "could you explain",
    "could you clarify",
    "can you repeat that",
    "could you repeat that",
    "can you say that again",
    "could you say that again",
    "say that again",
    "come again",
    "sorry what",
    "sorry i didn't catch that",
    "sorry i didn't get that",
    "i don't understand",
    "i dont understand",
    "i didn't understand",
    "i didn't get that",
    "i don't follow",
    "i dont follow",
    "pardon",
    "excuse me what",
    "huh",
]

# Phrases that strongly indicate closing / wrap-up
_CLOSING_PREFIXES = [
    "here is a summary", "here's a summary", "here is what i",
    "i will go discuss", "i'll go discuss", "let me go discuss",
    "i will go talk", "i'll go talk", "let me go talk",
    "i'm going to go speak", "i will go speak",
    "that's everything", "that is everything",
    "great, i will", "great i will",
    "the doctor will be in", "he will be in", "she will be in",
    "thank you for your time", "thank you for your patience",
    "is there anything else before i go",
    "we're all done", "we are all done",
    "to summarize", "in summary",
]


def classify_message(text: str) -> str:
    """
    Classify a student message as 'question', 'acknowledgment',
    'closing', 'clarification', or 'ambiguous'.

    Args:
        text: The student's raw message text.

    Returns:
        One of: 'question', 'acknowledgment', 'closing',
        'clarification', 'ambiguous'.

    'question' → proceed to node matching (embedding + pgvector).
    'acknowledgment' → skip node matching. Check timed nodes, then ack response.
    'closing' → skip node matching. Deliver closing response.
    'clarification' → skip node matching. Patient asks student to rephrase.
    'ambiguous' → run node matching. If no match, LLM fallback decides.
    """
    cleaned = text.strip().lower()

    # Empty or very short messages
    if len(cleaned) < 2:
        return "ambiguous"

    # ── Check for closing first (more specific than acknowledgment) ──
    for prefix in _CLOSING_PREFIXES:
        if cleaned.startswith(prefix):
            return "closing"

    # ── Check for acknowledgment ──
    for prefix in _ACKNOWLEDGMENT_PREFIXES:
        if cleaned.startswith(prefix):
            # BUT if it also contains a question, it's a question with
            # an acknowledgment preamble. E.g., "I'm sorry to hear that,
            # do you have any allergies?"
            if "?" in cleaned:
                return "question"
            return "acknowledgment"

    # ── Check for clarification / repetition requests ──
    # Must come BEFORE the "?" and interrogative-word checks because
    # phrases like "what do you mean?" start with question words but
    # are NOT clinical questions.
    for pattern in _CLARIFICATION_PATTERNS:
        if cleaned.startswith(pattern) or cleaned == pattern:
            return "clarification"

    # ── Check for explicit question ──
    if "?" in cleaned:
        return "question"

    # ── Interrogative words at the start suggest a question ──
    question_starters = [
        "what ", "when ", "where ", "who ", "why ", "how ",
        "do you", "does ", "did ", "have you", "has ",
        "are you", "is ", "can you", "could you", "would you",
        "tell me", "describe", "explain",
        "any ", "have any",
    ]
    for starter in question_starters:
        if cleaned.startswith(starter):
            return "question"

    # ── Default: ambiguous ──
    # The LLM fallback classifier will make the final determination.
    # But crucially, re-ask embedding search will NOT run on ambiguous
    # messages — only on messages classified as 'question'.
    return "ambiguous"
