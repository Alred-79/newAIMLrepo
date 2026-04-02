"""
ClinEQ — Evaluation Logger

Logs every evaluation result to eval.evaluation_logs in PostgreSQL.
Called after each turn, regardless of whether bias was detected.
"""

import json
import psycopg2
from shared.config import Config
from shared.models import EvaluationResult


def log_evaluation(
    session_id: str,
    message_index: int,
    student_message: str,
    result: EvaluationResult,
    reflection_triggered: bool = False,
    student_reflection_text: str | None = None,
    model_used: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """
    Write an evaluation result to the database.

    Args:
        session_id: The active session ID.
        message_index: Which student message this is (0-indexed).
        student_message: The student's message text.
        result: The EvaluationResult from the lightweight evaluator.
        reflection_triggered: Whether a System message was sent.
        student_reflection_text: Not used in MVP1 (no reflection input).
        model_used: Which LLM model was used.
        latency_ms: Evaluation latency in milliseconds.
    """
    conn = psycopg2.connect(Config.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval.evaluation_logs (
                    session_id, message_index, evaluator_tier,
                    student_message, label, confidence,
                    should_escalate,
                    stigma_anchors, brief_reason, internal_reasoning,
                    reflection_triggered, student_reflection_text,
                    latency_ms, model_used
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    session_id,
                    message_index,
                    "lightweight",
                    student_message,
                    result.tone_label.value,
                    result.tone_confidence,
                    result.should_escalate,
                    json.dumps(result.stigma_anchors_found),
                    result.brief_reason,
                    result.raw_llm_response,
                    reflection_triggered,
                    student_reflection_text,
                    latency_ms,
                    model_used or Config.LIGHTWEIGHT_MODEL,
                ),
            )
            conn.commit()
    finally:
        conn.close()
