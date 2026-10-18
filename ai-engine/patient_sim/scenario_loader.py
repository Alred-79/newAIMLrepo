"""
ClinEQ — Scenario Loader

Loads scenario JSON → ScenarioConfig, embeds trigger phrases, seeds pgvector,
and builds the BM25 in-memory index for hybrid retrieval.
"""

import re
import json
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from shared.models import ScenarioConfig, NodeType
from shared.llm import llm_embedding
from patient_sim.db import seed_embeddings



class ScenarioIndex:
    """Holds retrieval indices for a loaded scenario."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_phrase_to_node: dict[str, str] = {}  # trigger phrase → node_id
        self.bm25_corpus: list[str] = []  # raw phrases (parallel to bm25 internal)

    def build_bm25(self, scenario: ScenarioConfig) -> None:
        """Build BM25 index from all trigger phrases in the scenario."""
        corpus_tokens = []
        phrase_list = []
        phrase_to_node = {}

        for node in scenario.content_nodes:
            if node.node_type == NodeType.TRIGGERED or (
                node.node_type == NodeType.TIMED and node.trigger_phrases
            ):
                for phrase in node.trigger_phrases:
                    tokens = self._tokenize(phrase)
                    corpus_tokens.append(tokens)
                    phrase_list.append(phrase)
                    phrase_to_node[phrase] = node.node_id

        if corpus_tokens:
            self.bm25 = BM25Okapi(corpus_tokens)
            self.bm25_corpus = phrase_list
            self.bm25_phrase_to_node = phrase_to_node

    # Common English function words that inflate BM25 scores without
    # discriminating between topics.  "do you smoke" vs "do you drink"
    # shared 2/3 tokens before this filter — now only content words
    # ("smoke" vs "drink") are compared, preventing false cross-topic
    # matches.  Applied to BOTH corpus and query for consistency.
    _STOP_WORDS: set[str] = {
        # determiners / articles
        "a", "an", "the", "this", "that", "these", "those",
        # pronouns
        "i", "me", "my", "you", "your", "we", "our", "us",
        "they", "them", "their", "he", "she", "him", "her", "it", "its",
        # be / have / do auxiliaries
        "is", "are", "was", "were", "am", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "having",
        # modals
        "can", "could", "will", "would", "shall", "should",
        "may", "might", "must",
        # prepositions
        "in", "on", "at", "to", "for", "of", "with", "from",
        "by", "about", "as", "into", "through",
        # conjunctions / negation
        "and", "or", "but", "not", "no", "nor",
        # interrogatives (discriminating info is in the rest of the phrase)
        "what", "which", "who", "whom", "when", "where", "why", "how",
        # quantifiers / misc
        "any", "some", "all", "each", "every",
        "so", "if", "just", "also", "very", "too",
        "up", "down", "out", "off", "over",
    }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase tokenization with stop word removal.

        Removes common function words so BM25 scores reflect content-word
        overlap only.  Do NOT stem or lemmatize — "lisinopril" should
        match "lisinopril" exactly.
        """
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return [t for t in tokens if t not in ScenarioIndex._STOP_WORDS]



def load_scenario(json_path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario JSON file into a ScenarioConfig model."""
    path = Path(json_path)
    with open(path) as f:
        data = json.load(f)
    return ScenarioConfig(**data)


async def embed_trigger_phrases(scenario: ScenarioConfig) -> list[dict]:
    """
    Generate embeddings for all trigger phrases in a scenario.

    Returns list of dicts ready for seed_embeddings():
        [{"node_id": ..., "trigger_phrase": ..., "embedding": [...]}, ...]
    """
    # Collect all trigger phrases with their node_ids
    phrases_to_embed = []
    for node in scenario.content_nodes:
        for phrase in node.trigger_phrases:
            phrases_to_embed.append({
                "node_id": node.node_id,
                "trigger_phrase": phrase,
            })

    if not phrases_to_embed:
        return []

    # Batch embed all phrases in one API call
    texts = [p["trigger_phrase"] for p in phrases_to_embed]
    embeddings = await llm_embedding(texts)

    # Attach embeddings to the phrase records
    results = []
    for i, item in enumerate(phrases_to_embed):
        item["embedding"] = embeddings[i]
        results.append(item)

    return results


async def load_and_seed_scenario(
    json_path: str | Path,
) -> tuple[ScenarioConfig, ScenarioIndex]:
    """
    Full pipeline: load JSON → validate → embed trigger phrases → seed pgvector
    → build BM25 index.

    Args:
        json_path: Path to the scenario JSON file.

    Returns:
        Tuple of (ScenarioConfig, ScenarioIndex).
    """
    scenario = load_scenario(json_path)
    embeddings = await embed_trigger_phrases(scenario)
    count = seed_embeddings(
        scenario_id=scenario.scenario_id,
        embeddings=embeddings,
        clear_existing=True,
    )
    print(f"Seeded {count} trigger phrase embeddings for {scenario.scenario_id}")

    # Build BM25 index
    index = ScenarioIndex()
    index.build_bm25(scenario)
    phrase_count = len(index.bm25_corpus)
    print(f"Built BM25 index with {phrase_count} trigger phrases for {scenario.scenario_id}")

    return scenario, index
