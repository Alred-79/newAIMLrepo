"""
ClinEQ — Database & Retrieval Operations

Four operations:
1. seed_embeddings(): Insert trigger phrase embeddings at scenario load time.
2. query_similar(): pgvector cosine similarity for undelivered nodes.
3. query_similar_all(): pgvector cosine similarity for ALL nodes (re-ask detection).
4. query_bm25(): In-memory BM25 keyword matching against trigger phrases.
"""

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from shared.config import Config


def get_connection():
    conn = psycopg2.connect(Config.DATABASE_URL)
    register_vector(conn)
    return conn


def seed_embeddings(
    scenario_id: str,
    embeddings: list[dict],
    clear_existing: bool = True,
) -> int:
    """
    Insert trigger phrase embeddings into sim.trigger_embeddings.

    Args:
        scenario_id: The scenario these embeddings belong to.
        embeddings: List of dicts with keys:
            - "node_id": str
            - "trigger_phrase": str
            - "embedding": list[float] (1536 dimensions)
        clear_existing: If True, delete existing embeddings for this
                        scenario before inserting. Default True.

    Returns:
        Number of embeddings inserted.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if clear_existing:
                cur.execute(
                    "DELETE FROM sim.trigger_embeddings WHERE scenario_id = %s",
                    (scenario_id,),
                )

            if not embeddings:
                conn.commit()
                return 0

            values = [
                (
                    scenario_id,
                    e["node_id"],
                    e["trigger_phrase"],
                    e["embedding"],
                )
                for e in embeddings
            ]

            execute_values(
                cur,
                """
                INSERT INTO sim.trigger_embeddings
                    (scenario_id, node_id, trigger_phrase, embedding)
                VALUES %s
                """,
                values,
                template="(%s, %s, %s, %s::vector)",
            )

            conn.commit()
            return len(values)
    finally:
        conn.close()


def query_similar(
    scenario_id: str,
    query_embedding: list[float],
    excluded_node_ids: list[str] | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Find the closest trigger phrase embeddings by cosine similarity.

    Args:
        scenario_id: Only search within this scenario's embeddings.
        query_embedding: The student message embedding (1536 dims).
        excluded_node_ids: Node IDs to exclude (already delivered).
        top_k: Number of top matches to return.

    Returns:
        List of dicts sorted by similarity (highest first):
            - "node_id": str
            - "trigger_phrase": str
            - "similarity": float (0.0 to 1.0, cosine similarity)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            excluded = excluded_node_ids or []

            if excluded:
                cur.execute(
                    """
                    SELECT node_id, trigger_phrase,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM sim.trigger_embeddings
                    WHERE scenario_id = %s
                      AND node_id != ALL(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        query_embedding,
                        scenario_id,
                        excluded,
                        query_embedding,
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT node_id, trigger_phrase,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM sim.trigger_embeddings
                    WHERE scenario_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, scenario_id, query_embedding, top_k),
                )

            rows = cur.fetchall()
            # Score filter: ignore very low similarity noise (threshold 0.3)
            return [
                {
                    "node_id": row[0],
                    "trigger_phrase": row[1],
                    "similarity": float(row[2]),
                }
                for row in rows
                if float(row[2]) >= 0.3
            ]
    finally:
        conn.close()


def query_similar_all(
    scenario_id: str,
    query_embedding: list[float],
    top_k: int = 1,
) -> list[dict]:
    """
    Find the closest trigger phrase embeddings across ALL nodes (including
    already-delivered ones). Used for re-ask detection.

    Same return format as query_similar():
        [{"node_id": str, "trigger_phrase": str, "similarity": float}, ...]
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT node_id, trigger_phrase,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM sim.trigger_embeddings
                WHERE scenario_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, scenario_id, query_embedding, top_k),
            )
            rows = cur.fetchall()
            # Score filter: ignore very low similarity noise (threshold 0.3)
            return [
                {
                    "node_id": row[0],
                    "trigger_phrase": row[1],
                    "similarity": float(row[2]),
                }
                for row in rows
                if float(row[2]) >= 0.3
            ]
    finally:
        conn.close()


def query_bm25(
    message: str,
    scenario_index,
    excluded_node_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Query BM25 index for keyword-based matches.

    Args:
        message: The student's raw message text.
        scenario_index: ScenarioIndex with a built BM25 index.
        excluded_node_ids: Node IDs to exclude (already delivered).
        top_k: Number of top results to return.

    Returns:
        List of (node_id, bm25_score) tuples, sorted by score descending.
        Only returns undelivered nodes (one entry per unique node_id).
    """
    if scenario_index.bm25 is None:
        return []

    from patient_sim.scenario_loader import ScenarioIndex
    tokens = ScenarioIndex._tokenize(message)
    if not tokens:
        return []

    scores = scenario_index.bm25.get_scores(tokens)

    excluded = set(excluded_node_ids) if excluded_node_ids else set()
    seen_nodes: set[str] = set()
    results: list[tuple[str, float]] = []

    # Sort by score descending
    scored_phrases = sorted(
        zip(scenario_index.bm25_corpus, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    for phrase, score in scored_phrases:
        if score <= 0:
            continue
        node_id = scenario_index.bm25_phrase_to_node[phrase]
        if node_id in excluded:
            continue
        if node_id in seen_nodes:
            continue  # Take best score per node
        seen_nodes.add(node_id)
        results.append((node_id, float(score)))
        if len(results) >= top_k:
            break

    return results
