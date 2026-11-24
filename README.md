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
KEV_MAX_RESPONSE_BYTES=1048576
KEV_MAX_INPUT_CHARS=12000
```

Setting rules (checked when the backend is `kev`; an invalid value is logged once at
import by name only, never its value, and every Kev turn then abstains without
sending a request, with an `invalid_config` review trigger):

- `KEV_BASE_URL` must be an absolute `https://` URL, or plain `http://` only to a
  loopback host (`localhost`, `127.0.0.0/8`, `::1`). Empty, relative or malformed
  URLs are invalid, and so is any URL with userinfo (`user:pass@host`), which would
  otherwise be sent as Basic auth in place of the bearer key. Also invalid: any
  query or fragment (`?` or `#`), any non-ASCII character and IDNA (`xn--`) host
  labels, an empty or zero port (`host:` or `host:0`), and `%` in the host (IPv6
  zone ids such as `[fe80::1%25en0]`, percent-encoded hosts). IPv4-mapped IPv6
  hosts such as `[::ffff:127.0.0.1]` are never accepted as loopback for plain
  `http://` (Python 3.13 treats that address as loopback and 3.11/3.12 do not;
  rejecting it keeps one rule on every version). An invalid URL logs one warning
  that names only the setting, never the value. Redirects are never
  followed, even by an injected client, so a redirect response is a
  `transport_error`.
- `KEV_TIMEOUT_SECONDS` bounds the whole request. It must be a plain ASCII decimal
  or scientific number (no underscores, non-ASCII digits, hex, `nan` or `inf`)
  between 0.05 and 60. Empty or whitespace means the default, 1.5.
- `KEV_MAX_RESPONSE_BYTES` (plain integer, 1024 to 67108864, default 1048576) caps
  the response body. The client asks for `Accept-Encoding: identity` and refuses,
  unread, any response with a `Content-Encoding` other than `identity` (for
  example gzip or deflate), so the cap counts the bytes on the wire and a small
  compressed body cannot expand in memory. The body is read in a stream and
  abandoned once over the cap. Both cases are an `invalid_response`.
- `KEV_MAX_INPUT_CHARS` (plain integer, 200 to 30000, default 12000) caps the
  size of the whole request state serialized as compact JSON, exactly as httpx
  sends it (`ensure_ascii=False`, separators `,` and `:`): field names, the student
  message, every recent-context field, the probe and the chart context, with
  quotes, escapes, brackets, commas and colons all counted. The count is taken
  before flattening (which only shortens text) and stops as soon as it passes the
  cap. Longer
  input is not sent; the turn abstains and escalates with the request-level reason
  `input_too_long`. This is a conservative character guard, not a tokenizer-exact
  limit.
- Only values whose type is exactly `str`, `int`, `float` (finite), `bool`,
  `None`, `list`/`tuple` or `dict` with exact-`str` keys, nested at most 8 deep,
  can be sent. Anything else (subclasses of those types, sets, bytes, NaN or
  infinity, arbitrary objects, non-string keys, an integer too large to print,
  deeper nesting, two keys that become equal after flattening, or a chart-context
  key at any depth that, flattened and case-folded, equals a state label:
  `student_message`, `recent_context`, `chart_context` or `probe`) is not sent; the
  turn abstains and escalates with the request-level reason `invalid_input`.
- Before sending, every string in the state, dict keys included, is flattened:
  only the invisible characters used to hide or reorder text are deleted (U+200B,
  U+202A-202E, U+2060-2064, U+2066-2069, U+FEFF, U+FFF9-FFFB, U+E0001 and
  U+E0020-E007F), except that a well-formed subdivision flag (U+1F3F4, 3-7 tag
  letters or digits, then U+E007F) is kept whole; every other format character,
  including ZWNJ/ZWJ, LRM/RLM, Arabic number signs and the soft hyphen, is kept.
  Control characters (including newlines, carriage returns
  and the Unicode line/paragraph separators) become spaces, whitespace runs
  collapse to one space and the ends are trimmed, so text cannot fake another
  labelled field in Kev's prompt. Only the copy sent to Kev changes; the stored
  transcript is not modified.

`KEV_BASE_URL` is the server root, without `/v1/systemone`. An API key is optional
for a local server; if the server requires one, it is sent as a bearer token.
`kev-latest` is the server's API alias, not a pinned checkpoint. For reproducible
runs, start the server with `--run jaredpalmer/kev-4b@<commit-revision>` and record
that same identifier in `KEV_CHECKPOINT`. The result records this operator-supplied
identifier; it does not verify the deployed weights. An unset identifier is `null`.
The repository's existing Ollama evaluator remains the default when
`LIGHTWEIGHT_BACKEND` is unset or empty; use `ollama` to select it explicitly. The
environment template also selects `ollama` and leaves the Kev settings commented
out; Kev is opt-in. Any other value (for example a typo) does not fall back to
Ollama: Tier-1 returns `uncertain` with zero confidence and escalates.

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
| `patient_bias_response` | Choice | Separate Axis 5 construct; asked in every scenario, with `not_applicable` when the patient expressed no prejudice, so the question set does not reveal the scenario type |
| `probe_attention` | Score | Under / appropriate / over attention; requested only with an explicitly supplied, delivered probe (so its presence, and the question-bank hash, show that a probe was delivered) |

Known design gap: `patient_bias_response` never requests review because of the
option Kev chooses (it has no flagged choices, since `not_applicable` is the
expected answer in most scenarios). It requests review only when the answer is
`uncertain` or below the confidence threshold. A confident but inappropriate
response to a prejudiced patient therefore does not request review by itself (the
turn can still escalate through other questions); the answer is still returned in
the Tier-1 result. This is documented, not changed, pending a validated rule.

Clinical action (Axis 1) is not model-scored here. No axis probabilities are
multiplied or combined into a person-level score. For attention, inspect all three
probabilities: a mean of 1 can also mean equal probabilities of under- and
over-attention. Missing probes do not produce omission findings. `probe` and
`chart_context` are dormant in the running app: the simulator passes only the
student message and conversation context, so `probe_attention` is never asked
there. Callers with chart/probe data must supply those optional fields through
`EvaluationInput` themselves.

Timeouts, HTTP failures, input that is too long or cannot be sent, and
malformed/incomplete answers return
`uncertain`, zero tone confidence, `should_escalate=True`, and an unavailable Tier-1
result. There are no retries or automatic provider switches. Any uncertain or
low-confidence question requests escalation even if tone is confident. Potentially
problematic tone, attribution, stance, or anchored attention also requests review;
this routing decision does not combine their probabilities. The timeout
bounds the whole request; it is configurable and is not a latency guarantee.

### Latency and deployment

- Today Tier-1 runs inside the encounter turn: `patient_sim/simulator.py` starts it
  in parallel with response assembly and waits for it before returning the turn.
  One global `KEV_TIMEOUT_SECONDS` (default 1.5) bounds each Kev request. It is a
  ceiling, not an expected latency; no latency has been measured in this
  repository. The project's design budget of about 2 s per turn and sub-200 ms for
  Tier-1 comes from the engineering handoff, not from measurements here.
- A scale-to-zero or otherwise cold Kev deployment can take longer than the timeout
  to answer its first request. That turn then returns `uncertain`, an unavailable
  Tier-1 result and `should_escalate=True` with a `transport_error` review trigger.
  This is by design (abstain, never guess), not a fault to retry around.
- An unsupported `LIGHTWEIGHT_BACKEND` is always logged once as a warning when the
  engine imports its configuration; invalid Kev settings (see the setting rules
  above) are logged only when the backend is `kev`. Only the setting's name is
  logged, never its value. With a bad Kev setting every Kev turn abstains without
  sending a request and records an `invalid_config` review trigger. The other
  request-level reasons are `transport_error` (timeout, connection failure, HTTP
  error status or redirect), `invalid_response` (a malformed, oversized,
  compressed or incomplete Kev answer), `input_too_long` and `invalid_input`
  (see the setting rules above). Answer-level reasons are `low_confidence`,
  `uncertain`, `flagged_choice` and `score_tail`.
- On a Tier-1 timeout or transport error the turn is escalated: when
  `HEAVYWEIGHT_ENABLED` is true (the default), the simulator then awaits the
  heavyweight evaluation before it returns the turn, so that turn waits for the
  Kev timeout and then for the heavyweight call.
- If student reflection is deferred to the end of the encounter, the recommended
  future arrangement is to run Tier-1 in an asynchronous lane with a longer
  timeout, so cold starts do not turn into escalations. This lane does not exist
  in the code today.
- `KEV_CHECKPOINT` is declared by the operator and recorded as-is. The client does
  not verify which weights the server actually loaded.

Results include a question-bank SHA-256 hash, model alias, declared checkpoint,
and elapsed milliseconds. In Kev results, `brief_reason` is one of a few fixed
status messages; it never holds exception text or generated evidence. Raw Kev signals do not directly trigger the simulator's student-facing
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
