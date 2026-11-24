-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS sim;
CREATE SCHEMA IF NOT EXISTS eval;
CREATE SCHEMA IF NOT EXISTS app;

-- Simulator schema

CREATE TABLE sim.trigger_embeddings (
    id SERIAL PRIMARY KEY,
    scenario_id VARCHAR(100) NOT NULL,
    node_id VARCHAR(100) NOT NULL,
    trigger_phrase TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON sim.trigger_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE INDEX idx_trigger_scenario ON sim.trigger_embeddings(scenario_id);

-- Evaluator schema

CREATE TABLE eval.evaluation_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    message_index INTEGER,
    evaluator_tier VARCHAR(20) NOT NULL,  -- 'lightweight' or 'heavyweight'
    student_message TEXT NOT NULL,
    label VARCHAR(20) NOT NULL,
    confidence FLOAT,
    is_decision_point BOOLEAN DEFAULT FALSE,
    should_escalate BOOLEAN DEFAULT FALSE,
    stigma_anchors JSONB DEFAULT '[]',
    brief_reason TEXT,
    internal_reasoning TEXT,
    bias_patterns_detected JSONB DEFAULT '[]',
    reflection_triggered BOOLEAN DEFAULT FALSE,
    student_reflection_text TEXT,
    latency_ms INTEGER,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eval.test_results (
    id SERIAL PRIMARY KEY,
    test_run_id VARCHAR(100) NOT NULL,
    test_case_id VARCHAR(100) NOT NULL,
    expected_label VARCHAR(20),
    actual_label VARCHAR(20),
    expected_decision_point BOOLEAN,
    actual_decision_point BOOLEAN,
    confidence FLOAT,
    is_correct BOOLEAN,
    failure_category VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- App schema (future)

CREATE TABLE app.sessions (
    id VARCHAR(100) PRIMARY KEY,
    student_id VARCHAR(100),
    scenario_id VARCHAR(100) NOT NULL,
    is_control BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'active',
    conversation_history JSONB DEFAULT '[]',
    control_profile JSONB,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
