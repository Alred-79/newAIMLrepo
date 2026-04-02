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

```bash
pytest ai-engine/tests/
```
