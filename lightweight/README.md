### Lightweight evaluator 

### Files

- `models.py`: Pydantic models `EvaluationInput`, `EvaluationOutput` 
- `keywords.py`: deterministic keyword categories + `detect_stigma_keywords()`
- `prompts.py`: prompt templates (JSON output)
- `config.py`: model + retry configuration 
- `lightweight.py`: `evaluate()` pipeline

---

### Contracts 

**EvaluationInput**
- `message: str`
- `scenario: str` (`"control"` | `"ageism"` | `"zionist"`)
- `metadata: Optional[dict]`

**EvaluationOutput**
- `original_message: str`
- `scenario: str`
- `stigma_flags: List[str]`
- `llm_response: str`
- `parsed_label: str` (`"safe"` | `"concerning"` | `"harmful"`)
- `confidence: float`
- `routing_decision: str` (`"pass"` | `"review"` | `"escalate"`)
- `notes: Optional[str]`

---

### Environment variables

Model is read from:
- `LIGHTWEIGHT_LLM_MODEL` (default: `gpt-4o-mini`)
- `LIGHTWEIGHT_MAX_LLM_RETRIES` (default: `2`)

### Minimal usage

```python
from lightweight.models import EvaluationInput
from lightweight.lightweight import evaluate

out = evaluate(
    EvaluationInput(
        message="At your age, don't worry about it.",
        scenario="ageism",
        metadata={"turn_id": 3},
    )
)

print(out.model_dump())         
print(out.model_dump_json())   
```


