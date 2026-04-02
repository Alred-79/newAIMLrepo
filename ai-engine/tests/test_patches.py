"""Tests for keyword re-ranking, evaluator prompt changes, and hybrid retrieval."""

from shared.keyword_rerank import compute_keyword_overlap, rerank_matches, _tokenize
from shared.models import ContentNode, ScenarioConfig, PatientProfile
from patient_sim.scenario_loader import ScenarioIndex
from patient_sim.node_matcher import reciprocal_rank_fusion, _deduplicate_semantic
from patient_sim.db import query_bm25


def test_tokenize_removes_stop_words():
    words = _tokenize("Do you have any allergies to medications?")
    assert "allergies" in words
    assert "medications" in words
    assert "do" not in words  # stop word
    assert "you" not in words  # stop word
    assert "any" not in words  # stop word


def test_keyword_overlap_basic():
    node = ContentNode(
        node_id="substance_use",
        topic="Smoking and alcohol use",
        canonical_content="No, I don't smoke.",
        trigger_phrases=[
            "Do you smoke?",
            "Do you use tobacco?",
            "Do you drink alcohol?",
            "Any recreational drug use?",
        ],
    )
    # "smoke" and "drink" should match
    overlap = compute_keyword_overlap("Do you smoke or drink?", node)
    assert overlap >= 2  # "smoke" and "drink" at minimum


def test_keyword_overlap_no_match():
    node = ContentNode(
        node_id="occupation",
        topic="Work and employment",
        canonical_content="I am an accountant.",
        trigger_phrases=[
            "What do you do for work?",
            "What is your occupation?",
            "Where do you work?",
        ],
    )
    # "smoke" and "drink" should NOT match occupation triggers
    overlap = compute_keyword_overlap("Do you smoke or drink?", node)
    assert overlap == 0  # no meaningful word overlap


def test_rerank_promotes_keyword_match():
    """When embedding scores are close, keyword overlap should win."""
    # Simulate: embedding says occupation is top, substance_use is second
    embedding_results = [
        {"node_id": "occupation", "trigger_phrase": "What do you do?", "similarity": 0.58},
        {"node_id": "substance_use", "trigger_phrase": "Do you smoke?", "similarity": 0.55},
    ]

    # Build a minimal scenario with these nodes
    scenario = ScenarioConfig(
        scenario_id="test",
        title="Test",
        category="control",
        bias_direction="none",
        clinical_context="primary_care",
        patient_profile=PatientProfile(
            name="Test", age=50, gender="male", chief_complaint="test"
        ),
        opening_line="Hi.",
        content_nodes=[
            ContentNode(
                node_id="occupation", topic="Work", canonical_content="Accountant.",
                trigger_phrases=["What do you do for work?", "What is your occupation?"],
            ),
            ContentNode(
                node_id="substance_use", topic="Smoking and alcohol",
                canonical_content="No.",
                trigger_phrases=["Do you smoke?", "Do you drink alcohol?", "Any tobacco use?"],
            ),
        ],
    )

    reranked = rerank_matches(
        student_message="Do you smoke or drink?",
        embedding_results=embedding_results,
        scenario=scenario,
    )

    # substance_use should now be top
    assert reranked[0]["node_id"] == "substance_use"


def test_rerank_preserves_clear_winner():
    """When embedding scores are far apart, keyword overlap shouldn't override."""
    embedding_results = [
        {"node_id": "allergies", "trigger_phrase": "Any allergies?", "similarity": 0.75},
        {"node_id": "medications", "trigger_phrase": "Any medications?", "similarity": 0.45},
    ]

    scenario = ScenarioConfig(
        scenario_id="test", title="Test", category="control",
        bias_direction="none", clinical_context="primary_care",
        patient_profile=PatientProfile(
            name="Test", age=50, gender="male", chief_complaint="test"
        ),
        opening_line="Hi.",
        content_nodes=[
            ContentNode(
                node_id="allergies", topic="Allergies", canonical_content="Peanuts.",
                trigger_phrases=["Any allergies?", "Are you allergic to anything?"],
            ),
            ContentNode(
                node_id="medications", topic="Medications", canonical_content="None.",
                trigger_phrases=["Any medications?", "What pills do you take?"],
            ),
        ],
    )

    reranked = rerank_matches(
        student_message="Do you have any allergies?",
        embedding_results=embedding_results,
        scenario=scenario,
    )

    # allergies should stay on top — the embedding gap (0.30) is too large for re-ranking
    assert reranked[0]["node_id"] == "allergies"



def _make_test_scenario():
    """Build a minimal scenario for BM25/RRF tests."""
    return ScenarioConfig(
        scenario_id="test",
        title="Test",
        category="control",
        bias_direction="none",
        clinical_context="primary_care",
        patient_profile=PatientProfile(
            name="Test", age=50, gender="male", chief_complaint="test"
        ),
        opening_line="Hi.",
        content_nodes=[
            ContentNode(
                node_id="allergies", topic="Allergies",
                canonical_content="Peanuts.",
                trigger_phrases=["Any allergies?", "Are you allergic to anything?",
                                 "Do you have any drug allergies?"],
            ),
            ContentNode(
                node_id="medications", topic="Medications",
                canonical_content="Lisinopril.",
                trigger_phrases=["What medications are you on?", "Any medications?",
                                 "Are you taking lisinopril?"],
            ),
            ContentNode(
                node_id="substance_use", topic="Smoking and alcohol",
                canonical_content="No.",
                trigger_phrases=["Do you smoke?", "Do you drink alcohol?",
                                 "Any tobacco use?"],
            ),
        ],
    )


def test_scenario_index_builds_bm25():
    """ScenarioIndex.build_bm25 creates a working BM25 index."""
    scenario = _make_test_scenario()
    index = ScenarioIndex()
    index.build_bm25(scenario)

    assert index.bm25 is not None
    # 3 nodes × 3 phrases each = 9 phrases
    assert len(index.bm25_corpus) == 9
    assert len(index.bm25_phrase_to_node) == 9
    assert index.bm25_phrase_to_node["Do you smoke?"] == "substance_use"


def test_scenario_index_tokenize():
    """BM25 tokenizer preserves medical terms and removes stop words."""
    tokens = ScenarioIndex._tokenize("Are you taking lisinopril 10mg?")
    assert "lisinopril" in tokens
    assert "10mg" in tokens
    assert "taking" in tokens
    assert "are" not in tokens   # Stop word removed
    assert "you" not in tokens   # Stop word removed


def test_query_bm25_basic():
    """BM25 query returns correct node for exact keyword match."""
    scenario = _make_test_scenario()
    index = ScenarioIndex()
    index.build_bm25(scenario)

    results = query_bm25(
        message="Do you have any allergies?",
        scenario_index=index,
        excluded_node_ids=[],
        top_k=3,
    )

    # allergies should be top result
    assert len(results) > 0
    assert results[0][0] == "allergies"
    assert results[0][1] > 0  # positive BM25 score


def test_query_bm25_excludes_delivered():
    """BM25 query respects excluded_node_ids."""
    scenario = _make_test_scenario()
    index = ScenarioIndex()
    index.build_bm25(scenario)

    results = query_bm25(
        message="Do you have any allergies?",
        scenario_index=index,
        excluded_node_ids=["allergies"],  # exclude the best match
        top_k=3,
    )

    # allergies should NOT appear
    node_ids = [r[0] for r in results]
    assert "allergies" not in node_ids


def test_query_bm25_specific_drug_name():
    """BM25 matches specific drug names that embeddings may miss."""
    scenario = _make_test_scenario()
    index = ScenarioIndex()
    index.build_bm25(scenario)

    results = query_bm25(
        message="Are you taking lisinopril?",
        scenario_index=index,
        excluded_node_ids=[],
        top_k=3,
    )

    # medications node has "lisinopril" in a trigger phrase
    assert len(results) > 0
    assert results[0][0] == "medications"


def test_deduplicate_semantic():
    """Deduplication keeps best similarity per node_id."""
    raw = [
        {"node_id": "allergies", "trigger_phrase": "Any allergies?", "similarity": 0.75},
        {"node_id": "allergies", "trigger_phrase": "Drug allergies?", "similarity": 0.72},
        {"node_id": "medications", "trigger_phrase": "Any meds?", "similarity": 0.65},
    ]
    deduped = _deduplicate_semantic(raw)
    assert len(deduped) == 2
    assert deduped[0] == ("allergies", 0.75)  # best score kept
    assert deduped[1] == ("medications", 0.65)


def test_rrf_basic_fusion():
    """RRF combines two ranked lists correctly."""
    semantic = [("allergies", 0.75), ("medications", 0.65)]
    bm25 = [("allergies", 5.2), ("medications", 3.1)]

    fused = reciprocal_rank_fusion(semantic, bm25, rrf_k=60)

    # allergies is #1 in both lists → highest RRF score
    assert fused[0][0] == "allergies"
    assert fused[1][0] == "medications"
    # RRF score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    assert abs(fused[0][1] - 2.0 / 61) < 1e-6


def test_rrf_promotes_bm25_winner():
    """When BM25 disagrees with semantic across 3 items, RRF adjusts ranking."""
    # Semantic: medications #1, substance_use #2, allergies #3
    semantic = [("medications", 0.58), ("substance_use", 0.55), ("allergies", 0.50)]
    # BM25: substance_use #1 (keyword "smoke" matched), allergies #2, medications #3
    bm25 = [("substance_use", 6.0), ("allergies", 3.0), ("medications", 1.0)]

    fused = reciprocal_rank_fusion(semantic, bm25, rrf_k=60)

    # substance_use: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
    # medications:   1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03227
    # allergies:     1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200
    # substance_use should win
    assert fused[0][0] == "substance_use"


def test_rrf_with_disjoint_results():
    """RRF handles nodes that appear in only one ranking."""
    semantic = [("allergies", 0.70), ("medications", 0.50)]
    bm25 = [("substance_use", 5.0)]  # completely different

    fused = reciprocal_rank_fusion(semantic, bm25, rrf_k=60)

    # All three nodes should appear in fused results
    node_ids = [nid for nid, _ in fused]
    assert "allergies" in node_ids
    assert "medications" in node_ids
    assert "substance_use" in node_ids

    # allergies and substance_use both ranked #1 (in different lists)
    # allergies: 1/(60+1) = 0.01639
    # substance_use: 1/(60+1) = 0.01639
    # medications: 1/(60+2) = 0.01613
    # allergies and substance_use tie, medications is last
    top_two = {fused[0][0], fused[1][0]}
    assert top_two == {"allergies", "substance_use"}


def test_rrf_empty_bm25():
    """RRF with empty BM25 list returns semantic ranking."""
    semantic = [("allergies", 0.70), ("medications", 0.50)]

    fused = reciprocal_rank_fusion(semantic, [], rrf_k=60)

    assert fused[0][0] == "allergies"
    assert fused[1][0] == "medications"
