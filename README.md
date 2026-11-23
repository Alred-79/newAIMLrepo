# ClinEQ — AI Simulation & Evaluation Engine

(to bind with common repo )

## Quick Start

```bash
# 1. Clone the repo
git clone git@github.com:Rutgers-ClinEQ/ai-ml.git
cd ai-ml

# 2. Set up environment variables
cp .env.template .env
# Fill in your API keys in .env

# 3. Start PostgreSQL (pgvector) + Redis
docker compose up -d

# 4. Install the AI engine package
pip install -e ai-engine/

# 5. Run tests
pytest ai-engine/tests/
```

## Directory Structure

```
ai-engine/
├── shared/          # Shared Pydantic models and config
├── bias_eval/       # Lightweight + heavyweight bias evaluation
├── patient_sim/     # Patient simulator (scripted state machine)
├── data/            # Scenarios, exemplars, test cases
├── tests/           # Integration tests
├── notebooks/       # Exploration and demo notebooks
└── pyproject.toml

fullstack/
├── backend/         # FastAPI app (reserved)
└── frontend/        # React app (reserved)
```

## Kev Tier-1 bias classifier

The lightweight evaluator supports the self-hosted
[jaredpalmer/kev](https://github.com/jaredpalmer/kev) decision model as an
alternative to Jev. It sends a single `POST /v1/systemone` request with independent
typed questions. No hosted Jev account is required.

Start Kev in its own checkout/environment (Python 3.12 or 3.13 and `uv`):

```bash
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

The initial start downloads the weights; wait for the server to finish loading.
Install Kev from its repository: the PyPI package named `kev` is unrelated.
See the upstream README for hardware requirements and alternate checkpoint sizes.

In this application's `.env`, set:

```dotenv
LIGHTWEIGHT_BACKEND=kev
KEV_BASE_URL=http://127.0.0.1:8009
KEV_API_KEY=
KEV_MODEL=kev-latest
KEV_CHECKPOINT=
KEV_TIMEOUT_SECONDS=1.5
```

`KEV_BASE_URL` is the server root, without `/v1/systemone`. An API key is optional
for a local server; if the server requires one, it is sent as a bearer token.
`kev-latest` is the server's API alias, not a pinned checkpoint. For reproducible
runs, start the server with `--run jaredpalmer/kev-4b@<commit-revision>` and record
that same identifier in `KEV_CHECKPOINT`. The result records this operator-supplied
identifier; it does not verify the deployed weights. An unset identifier is `null`.
The repository's existing Ollama evaluator remains the default when
`LIGHTWEIGHT_BACKEND` is unset; use `ollama` to select it explicitly. The environment
template selects Kev for new setups.

```python
from bias_eval.lightweight import evaluate_tone_sync
from shared.models import EvaluationInput, Message, ScenarioCategory
from shared.tier1_models import ProbeAnchor

result = evaluate_tone_sync(EvaluationInput(
    student_message="Can you tell me more about when the pain started?",
    recent_context=[Message(role="patient", content="I worry you will blame my weight.")],
    scenario_category=ScenarioCategory.WEIGHT,
    chart_context={"symptom_duration": "two weeks"},
    probe=ProbeAnchor(turn_idx=3, text="I worry you will blame my weight."),
))
print(result.model_dump_json(indent=2))
```

`result.tier1.answers` preserves each question's probabilities and provider
confidence separately:

| Question | Output | Meaning |
| --- | --- | --- |
| `tone` | Choice | Surface-language screening, preserving the existing biased/unbiased/uncertain interface |
| `attribution` | Choice | Axis 2: no attribution, evidence-based, stereotype-linked, or uncertain |
| `epistemic_stance` | Choice | Provisional Axis 3 screening |
| `affective_stance` | Choice | Provisional Axis 4 screening |
| `patient_bias_response` | Choice | Separate Axis 5 construct; requested only for patient-to-student or bidirectional scenarios |
| `probe_attention` | Score | Under / appropriate / over attention; requested only with an explicitly supplied, delivered probe |

Clinical action (Axis 1) is not model-scored here. No axis probabilities are
multiplied or combined into a person-level score. For attention, inspect all three
probabilities: a mean of 1 can also mean equal probabilities of under- and
over-attention. Missing probes do not produce omission findings. The current
simulator passes conversation context; callers with chart/probe data must supply
those optional fields through `EvaluationInput`.

Timeouts, HTTP failures, truncated input and malformed/incomplete answers return
`uncertain`, zero tone confidence, `should_escalate=True`, and an unavailable Tier-1
result. There are no retries or automatic provider switches. Any uncertain or
low-confidence question requests escalation even if tone is confident. Potentially
problematic tone, attribution, stance, or anchored attention also requests review;
this routing decision does not combine their probabilities. The timeout
bounds the whole request; it is configurable and is not a latency guarantee.

Results include a question-bank SHA-256 hash, model alias, declared checkpoint,
and elapsed milliseconds. `brief_reason` is a fixed status message, not generated
evidence. Raw Kev signals do not directly trigger the simulator's student-facing
reflection message. The existing heavyweight evaluator and its feedback policy
remain separate. Database/event-store persistence is not added by this change.

These are exploratory screening signals, not validated findings about students.
Agreement, thresholds, and calibration need evaluation against human-coded OSCE
data before research or feedback use. Offline tests verify the HTTP contract and
routing; they do not establish classifier accuracy or the target sub-200 ms latency.

```bash
pip install -e 'ai-engine[dev]'
pytest ai-engine/tests/test_kev.py -q
```

```bash
pytest ai-engine/tests/
```
