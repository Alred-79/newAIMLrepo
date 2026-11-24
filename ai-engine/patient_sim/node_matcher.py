"""
ClinEQ — Hybrid Content Node Matcher (PATCH G — BM25 + Semantic RRF)

Two parallel retrieval paths:
  Path A: pgvector cosine similarity (semantic embeddings)
  Path B: BM25 keyword matching (in-memory, zero-infra)

Fused via Reciprocal Rank Fusion (RRF), then:
  - Confident match → deliver node
  - Ambiguous → LLM fallback classifier
  - No match → LLM fallback (tangential, acknowledgment, closing, OOB)
  - Re-ask → hybrid re-ask detection against ALL nodes

Falls back gracefully to semantic-only if no BM25 index is available.
"""

import json
from typing import Optional

from shared.models import (
    ScenarioConfig,
    ScenarioState,
    ContentNode,
    MatchResult,
    Message,
)
from shared.config import Config
from shared.llm import llm_completion, llm_embedding, FALLBACK_CLASSIFIER_SCHEMA
from patient_sim.db import query_similar, query_similar_all, query_bm25
from patient_sim.prompts import build_fallback_classifier


async def embed_message(text: str) -> list[float]:
    embeddings = await llm_embedding([text])
    return embeddings[0]



def _deduplicate_semantic(results: list[dict]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for r in results:
        nid = r["node_id"]
        sim = r["similarity"]
        if nid not in best or sim > best[nid]:
            best[nid] = sim
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def reciprocal_rank_fusion(
    semantic_ranked: list[tuple[str, float]],
    bm25_ranked: list[tuple[str, float]],
    rrf_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Combine semantic and BM25 ranked lists using Reciprocal Rank Fusion.

    Each input is a list of (node_id, score) tuples sorted by score descending.
    Returns a list of (node_id, rrf_score) tuples sorted by RRF score descending.

    RRF score = sum over all lists of: 1 / (k + rank)
    where rank is 1-indexed. Score-scale agnostic — only uses rank positions.
    """
    k = rrf_k if rrf_k is not None else Config.RRF_K
    rrf_scores: dict[str, float] = {}

    for rank, (node_id, _) in enumerate(semantic_ranked, start=1):
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)

    for rank, (node_id, _) in enumerate(bm25_ranked, start=1):
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)



async def match_content_node(
    student_message: str,
    scenario: ScenarioConfig,
    state: ScenarioState,
    allow_re_ask: bool = True,
    scenario_index=None,
    message_type: str = "question",
) -> MatchResult:
    """
    Hybrid content node matching: semantic embeddings + BM25 with RRF fusion.

    Falls back to semantic-only if scenario_index is None.

    Flow:
    1. Embed student message.
    2. Path A: pgvector cosine similarity (undelivered nodes).
    3. Path B: BM25 keyword matching (undelivered nodes).
    4. Fuse with RRF.
    5. Three-way threshold branch on RRF score:
       > RRF_CONFIDENT → confident match, deliver node.
       RRF_AMBIGUOUS–RRF_CONFIDENT → ambiguous, LLM fallback.
       < RRF_AMBIGUOUS → no match, LLM fallback.
    6. Re-ask detection with hybrid retrieval (if applicable).
    """
    # Step 1: Embed
    query_embedding = await embed_message(student_message)

    # Step 2: Path A — Semantic (existing pgvector)
    semantic_results = query_similar(
        scenario_id=scenario.scenario_id,
        query_embedding=query_embedding,
        excluded_node_ids=state.delivered_node_ids,
        top_k=5,
    )
    semantic_ranked = _deduplicate_semantic(semantic_results)

    # Step 3: Path B — BM25 keyword matching
    bm25_ranked: list[tuple[str, float]] = []
    if scenario_index is not None:
        bm25_ranked = query_bm25(
            message=student_message,
            scenario_index=scenario_index,
            excluded_node_ids=state.delivered_node_ids,
            top_k=5,
        )

    # Step 4: Fuse with RRF
    if bm25_ranked:
        fused = reciprocal_rank_fusion(semantic_ranked, bm25_ranked)
    else:
        # No BM25 index — fall back to semantic-only ranking
        fused = semantic_ranked

    # Step 5: Three-way threshold branch
    if fused:
        top_node_id, top_score = fused[0]

        # Determine threshold: use RRF thresholds when BM25 contributed,
        # otherwise use legacy cosine thresholds for backward compatibility.
        # The ambiguous zone (between confident and no-match) is handled by
        # the LLM fallback classifier, so only the confident threshold is needed.
        if bm25_ranked:
            confident_threshold = Config.RRF_CONFIDENT_THRESHOLD
        else:
            confident_threshold = Config.NODE_MATCH_CONFIDENT

        # Stricter threshold for ambiguous messages — if the question gate
        # isn't sure it's a question, require higher confidence to auto-match.
        if message_type == "ambiguous":
            confident_threshold *= Config.AMBIGUOUS_THRESHOLD_MULTIPLIER

        # Confident match
        if top_score > confident_threshold:
            node = _find_node(scenario, top_node_id)
            if node:
                method = "hybrid_rrf" if bm25_ranked else "embedding"
                return MatchResult(
                    matched_node=node,
                    match_confidence=top_score,
                    match_method=method,
                )
    else:
        top_score = 0.0

    # Step 6: Re-ask detection — check ALL nodes (including delivered)
    if allow_re_ask and state.delivered_node_ids:
        reask_result = _check_reask(
            student_message=student_message,
            query_embedding=query_embedding,
            state=state,
            scenario=scenario,
            scenario_index=scenario_index,
        )
        if reask_result:
            return reask_result

    # Step 7: LLM fallback
    return await _llm_fallback(
        student_message=student_message,
        scenario=scenario,
        state=state,
    )



def _check_reask(
    student_message: str,
    query_embedding: list[float],
    state: ScenarioState,
    scenario: ScenarioConfig,
    scenario_index=None,
) -> Optional[MatchResult]:
    """Check if student is re-asking about an already delivered node."""
    # Semantic: query ALL nodes (including delivered)
    all_semantic = query_similar_all(
        scenario_id=scenario.scenario_id,
        query_embedding=query_embedding,
        top_k=5,
    )
    all_semantic_ranked = _deduplicate_semantic(all_semantic)

    # BM25: query all trigger phrases (don't filter delivered)
    all_bm25: list[tuple[str, float]] = []
    if scenario_index is not None:
        all_bm25 = query_bm25(
            message=student_message,
            scenario_index=scenario_index,
            excluded_node_ids=[],  # Include all
            top_k=5,
        )

    # Fuse
    if all_bm25:
        fused_all = reciprocal_rank_fusion(all_semantic_ranked, all_bm25)
        reask_threshold = Config.RRF_REASK_THRESHOLD
    else:
        fused_all = all_semantic_ranked
        reask_threshold = Config.RE_ASK_SIMILARITY_THRESHOLD

    if not fused_all:
        return None

    top_node_id, top_score = fused_all[0]

    # If the top match is a DELIVERED node and score is above re-ask threshold
    if (
        top_node_id in state.delivered_node_ids
        and top_score >= reask_threshold
    ):
        re_ask_node = _find_node(scenario, top_node_id)
        if re_ask_node:
            method = "hybrid_rrf_reask" if all_bm25 else "re_ask"
            return MatchResult(
                is_re_ask=True,
                re_ask_node=re_ask_node,
                match_confidence=top_score,
                match_method=method,
            )

    return None



async def _llm_fallback(
    student_message: str,
    scenario: ScenarioConfig,
    state: ScenarioState,
) -> MatchResult:
    """
    LLM fallback classifier for ambiguous or no-match cases.
    Handles 5 categories: node_id, tangential, acknowledgment, closing, out_of_bounds.
    """
    undelivered = [
        {"node_id": node.node_id, "topic": node.topic}
        for node in scenario.content_nodes
        if node.node_id not in state.delivered_node_ids
        and node.node_type.value == "triggered"
    ]

    if not undelivered:
        return MatchResult(is_tangential=True, match_method="no_nodes_remaining")

    # Get previous exchange for context
    prev_student = None
    prev_patient = None
    history = state.conversation_history
    if len(history) >= 2:
        for i in range(len(history) - 1, -1, -1):
            if history[i].role == "patient" and prev_patient is None:
                prev_patient = history[i].content
            elif history[i].role == "student" and prev_student is None:
                prev_student = history[i].content
            if prev_student and prev_patient:
                break

    system_prompt, user_prompt = build_fallback_classifier(
        student_message=student_message,
        undelivered_nodes=undelivered,
        previous_student_message=prev_student,
        previous_patient_response=prev_patient,
    )

    try:
        raw = await llm_completion(
            model=Config.FALLBACK_CLASSIFIER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=Config.FALLBACK_CLASSIFIER_TEMPERATURE,
            max_tokens=Config.FALLBACK_CLASSIFIER_MAX_TOKENS,
            json_schema=FALLBACK_CLASSIFIER_SCHEMA,
        )
        parsed = json.loads(raw)
        node_id = parsed.get("node_id", "tangential")
    except Exception:
        return MatchResult(is_tangential=True, match_method="llm_fallback_error")

    if node_id == "out_of_bounds":
        return MatchResult(is_out_of_bounds=True, match_method="llm_fallback")
    elif node_id == "acknowledgment":
        return MatchResult(is_acknowledgment=True, match_method="llm_fallback")
    elif node_id == "closing":
        return MatchResult(is_acknowledgment=True, match_method="closing")
    elif node_id == "tangential":
        return MatchResult(is_tangential=True, match_method="llm_fallback")
    else:
        node = _find_node(scenario, node_id)
        if node:
            return MatchResult(
                matched_node=node,
                match_confidence=0.5,
                match_method="llm_fallback",
            )
        return MatchResult(is_tangential=True, match_method="llm_fallback")


def _find_node(scenario: ScenarioConfig, node_id: str) -> Optional[ContentNode]:
    for node in scenario.content_nodes:
        if node.node_id == node_id:
            return node
    return None
