"""
ClinEQ — Keyword Re-Ranking for Content Node Matching

Runs AFTER pgvector returns top-N embedding matches. Re-ranks candidates
using exact keyword overlap between the student's message and each
candidate node's trigger phrases.

This breaks ties when embedding similarities are close, catching cases
where the semantically "closer" embedding is topically wrong.

Example:
  Student: "Do you smoke or drink?"
  Top embedding match: occupation (similarity 0.58)
  Second match: substance_use (similarity 0.55)

  Keyword overlap:
    occupation trigger phrases: "What do you do for work?" → overlap: 1 ("you")
    substance_use trigger phrases: "Do you smoke?" → overlap: 3 ("do", "you", "smoke")

  Re-ranked: substance_use wins because of higher keyword overlap.
"""

import re
from typing import Optional

from shared.models import ScenarioConfig, ContentNode


# Words too common to be useful as matching signals
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must",
    "i", "you", "your", "my", "me", "we", "they", "them", "it",
    "to", "of", "in", "for", "on", "at", "by", "with", "from",
    "and", "or", "but", "not", "no", "so", "if", "then",
    "that", "this", "what", "when", "where", "who", "how",
    "about", "any", "some", "all", "also", "just", "more",
    "there", "here", "very", "really", "tell", "know", "like",
    "up", "out", "been", "being", "get", "got", "take", "go",
    "going", "please", "thank", "thanks", "right", "okay", "ok",
    "let", "lets", "much", "many",
}


def _tokenize(text: str) -> set[str]:
    """Extract meaningful words from text, lowercased, stop words removed."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def compute_keyword_overlap(
    student_message: str,
    node: ContentNode,
) -> int:
    """
    Count how many meaningful words from the student's message appear
    in any of the node's trigger phrases.

    Returns the count of overlapping unique words.
    """
    student_words = _tokenize(student_message)
    if not student_words:
        return 0

    # Combine all trigger phrases into one word pool
    trigger_words = set()
    for phrase in node.trigger_phrases:
        trigger_words.update(_tokenize(phrase))

    # Also include words from the node's topic field
    trigger_words.update(_tokenize(node.topic))

    return len(student_words & trigger_words)


def rerank_matches(
    student_message: str,
    embedding_results: list[dict],
    scenario: ScenarioConfig,
    similarity_tie_range: float = 0.10,
    min_keyword_advantage: int = 2,
) -> list[dict]:
    """
    Re-rank pgvector embedding results using keyword overlap.

    Only re-ranks when the top two matches are within similarity_tie_range
    of each other (i.e., the embedding is uncertain). If a lower-ranked
    candidate has significantly more keyword overlap (min_keyword_advantage
    more matching words), it gets promoted to the top.

    Args:
        student_message: The student's raw message text.
        embedding_results: Top-N results from pgvector query_similar().
            Each dict has: node_id, trigger_phrase, similarity.
        scenario: The active scenario (used to look up ContentNode objects
            for trigger phrase access).
        similarity_tie_range: Max difference in similarity between top
            and candidate for re-ranking to apply. Default 0.10.
        min_keyword_advantage: Minimum keyword overlap advantage needed
            to promote a lower-ranked candidate. Default 2.

    Returns:
        The same list, potentially re-ordered. The top item is the
        recommended match.
    """
    if len(embedding_results) < 2:
        return embedding_results

    # Build node lookup
    node_map = {n.node_id: n for n in scenario.content_nodes}

    # Compute keyword overlap for each candidate
    scored = []
    for r in embedding_results:
        node = node_map.get(r["node_id"])
        kw_overlap = 0
        if node:
            kw_overlap = compute_keyword_overlap(student_message, node)
        scored.append({
            **r,
            "keyword_overlap": kw_overlap,
        })

    top = scored[0]
    top_sim = top["similarity"]
    top_kw = top["keyword_overlap"]

    # Check if any lower-ranked candidate should be promoted
    best_promotion = None
    for candidate in scored[1:]:
        sim_diff = top_sim - candidate["similarity"]
        kw_advantage = candidate["keyword_overlap"] - top_kw

        # Only re-rank if:
        # 1. Similarities are close (within tie range)
        # 2. Candidate has significantly more keyword overlap
        if sim_diff <= similarity_tie_range and kw_advantage >= min_keyword_advantage:
            if best_promotion is None or candidate["keyword_overlap"] > best_promotion["keyword_overlap"]:
                best_promotion = candidate

    if best_promotion:
        # Move the promoted candidate to the top
        scored.remove(best_promotion)
        scored.insert(0, best_promotion)

    # Return without the keyword_overlap field (clean output)
    return [
        {k: v for k, v in r.items() if k != "keyword_overlap"}
        for r in scored
    ]
