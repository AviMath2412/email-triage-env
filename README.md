# Email Triage Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **OpenEnv-compatible** reinforcement learning environment for real-world email triage tasks. Agents classify urgency, rank inbox priority, draft professional replies, and route emails to the correct department — mirroring the work of an executive assistant managing a busy inbox.

---

## Motivation

An executive assistant processes 50–200 emails per day. Getting prioritization wrong means:
- Missed customer crises → churn
- Ignored security alerts → breaches
- Overlooked contract renewals → lost revenue

This environment provides dense, verifiable reward signals unlike open-ended generation tasks.

---

## Quick Start

### Local (fastest for development)

```bash
git clone https://github.com/AviMath2412/email-triage-env
cd email-triage-env
pip install -e .

# Start the server
uvicorn server.app:app --port 7860

# In another terminal
python - <<'EOF'
from client import EmailTriageEnv
from server.models import EmailAction

with EmailTriageEnv("http://localhost:7860") as env:
    result = env.reset(task_id="easy")
    obs = result.observation
    print(f"Email: {obs.single_email.subject}")

    result = env.step(EmailAction(
        action_type="classify",
        email_id=obs.single_email.id,
        urgency="urgent",
        priority=5,
    ))
    print(f"Reward: {result.reward}")
    print(f"Feedback: {result.observation.last_action_feedback}")
EOF
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
# Swagger docs at http://localhost:7860/docs
```

### Hugging Face Spaces

```python
from client import EmailTriageEnv
env = EmailTriageEnv("https://avimath2412-email-triage-env.hf.space")
```

---

## Tasks

| Task | ID | Difficulty | Max Steps | Pass | Excellent |
|------|-----|------------|-----------|------|-----------|
| Single Email Classification | `easy` | Easy | 5 | ≥ 0.70 | ≥ 0.90 |
| Inbox Priority Ranking | `medium` | Medium | 8 | ≥ 0.60 | ≥ 0.85 |
| Full Triage Pipeline | `hard` | Hard | 10 | ≥ 0.55 | ≥ 0.80 |

### Easy — Single Email Classification
Classify one email: `urgency` (urgent/normal/low/spam) + `priority` (1–5).

### Medium — Inbox Priority Ranking
Sort 10 emails by priority. Scored with Kendall-tau. Episode ends when tau ≥ 0.95.

### Hard — Full Triage Pipeline
Classify + draft a professional reply (50+ words) + route to the correct department.

---

## Action Space

| Field | Type | Required For | Description |
|-------|------|-------------|-------------|
| `action_type` | str | All | `classify`, `rank`, `triage`, `done` |
| `email_id` | str | classify, triage | Target email ID |
| `urgency` | enum | classify, triage | `urgent`, `normal`, `low`, `spam` |
| `priority` | int | classify, triage | 1 (lowest) to 5 (highest) |
| `ranked_ids` | list | rank | All IDs in priority order |
| `reply_draft` | str | triage | Professional reply (min 50 words) |
| `route_to` | enum | triage | `support`, `sales`, `engineering`, `hr`, `finance`, `general` |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `single_email` | Email | Email to act on (easy/hard) |
| `inbox` | list[Email] | All emails (medium) |
| `last_action_feedback` | str | Reward explanation |
| `done` | bool | Episode termination flag |
| `cumulative_reward` | float | Running total |
| `step_count` | int | Steps taken so far |
| `correctness_score` | float | Step correctness breakdown |
| `completion_bonus` | float | Efficiency bonus breakdown |

---

## Reward Function

### Easy
| Component | Weight | Notes |
|-----------|--------|-------|
| Urgency correct | 0.50 | Exact match |
| Priority correct | 0.30 | 1.0/0.5/0.25 for off-by-0/1/2 |
| Efficiency bonus | up to 0.20 | Decays with step count |

### Medium
| Component | Weight | Notes |
|-----------|--------|-------|
| Kendall-tau score | 0.85 | Fraction of concordant pairs |
| Improvement bonus | 0.15 | Rewards getting better each step |

### Hard
| Component | Weight | Notes |
|-----------|--------|-------|
| Classification | 0.30 | urgency 50% + priority 50% |
| Reply quality | 0.30 | Length, greeting, acknowledgment, action, sign-off |
| Routing accuracy | 0.40 | Exact department match |
| Efficiency bonus | up to 0.15 | |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| WS | `/ws` | Main WebSocket agent loop |
| POST | `/reset` | HTTP reset |
| POST | `/step` | HTTP step |
| GET | `/state` | Current episode state |
| GET | `/tasks` | List all tasks |
| GET | `/tasks/{id}` | Single task details |
| POST | `/grader` | Grade a completed episode |
| GET | `/baseline` | Run heuristic baseline |
| GET | `/health` | Liveness probe |
| GET | `/docs` | Swagger UI |

---

## Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Keyword heuristic | 0.85 | ~0.74 | 0.92 |
| GPT-4o-mini zero-shot | ~0.94 | ~0.72 | ~0.82 |

Run LLM inference:
```bash
export OPENAI_API_KEY=sk-...
python inference.py --task all --episodes 3
```

---

## Project Structure

```
email-triage-env/
├── server/
│   ├── __init__.py
│   ├── app.py                  # FastAPI server + endpoints
│   ├── environment.py          # Core RL logic
│   ├── models.py               # Pydantic Action/Observation types
│   ├── data.py                 # Email templates and task metadata
│   └── baseline_heuristic.py  # Reference non-LLM agent
├── tests/
│   ├── __init__.py
│   └── test_environment.py    # Deterministic test suite
├── client.py                  # Python SDK
├── inference.py               # LLM evaluation script
├── openenv.yaml               # Environment manifest
├── pyproject.toml             # Package config
├── Dockerfile                 # Container definition
└── README.md
```

---

## Setup & Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
uvicorn server.app:app --reload --port 7860
```

---

## License

MIT
