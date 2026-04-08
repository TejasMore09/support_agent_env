# Customer Support Agent Environment

> An OpenEnv-compliant evaluation environment for LLM agents on real-world customer support tasks.

---

## Overview

This environment simulates a customer support system where an AI agent must handle realistic customer emails through three progressively harder tasks. It mirrors a real-world use case deployed in enterprise support pipelines at companies like Zendesk, Intercom, and Freshdesk.

The agent receives a customer email and must:
1. **Classify** the issue (billing / complaint / query)
2. **Respond** with a professional, empathetic reply
3. **Resolve** the issue end-to-end with concrete action steps

---

## Why This Matters

Customer support automation is one of the highest-ROI applications of LLMs in production today. This environment provides:
- A **reproducible benchmark** for comparing support-agent quality
- **Dense reward signal** across the full trajectory (not just end-of-episode)
- **Deterministic graders** that are exploit-resistant and fair

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `email_id` | `string` | Unique email identifier |
| `email_body` | `string` | Full customer email text |
| `task` | `enum[classify, respond, resolve]` | Current task to complete |
| `step_num` | `int` | Step number within the episode (1-indexed) |

## Action Space

| Field | Type | Description |
|---|---|---|
| `reply` | `string` | Agent's free-text response to the current task |

---

## Tasks

### Task 1 — Email Classification (Easy)
- **Objective**: Identify the email as `billing`, `complaint`, or `query`
- **Grader**: Deterministic keyword + synonym matching
  - Exact match → `1.0`
  - Synonym match → `0.6`
  - No match → `0.0`

### Task 2 — Professional Response Generation (Medium)
- **Objective**: Draft a professional customer support reply
- **Grader**: Multi-criterion scoring
  - Keyword coverage: `40%`
  - Apology present: `20%`
  - Concrete action committed: `20%`
  - Reply length adequacy: `10%`
  - Professionalism (greeting + closing): `10%`

### Task 3 — Full Issue Resolution (Hard)
- **Objective**: Resolve the issue completely — classify, respond, and provide resolution path
- **Grader**: Combined scoring
  - Classification accuracy: `20%`
  - Keyword coverage: `30%`
  - Apology + concrete action: `20%`
  - Expected resolution criteria: `20%`
  - Professionalism: `10%`

---

## Reward Design

- **Dense**: Partial credit at every step — never sparse
- **Deterministic**: Same input always produces same score
- **Range**: `[0.0, 1.0]` for all tasks
- **Reproducible**: Pure rule-based, no ML scoring

---

## Baseline Scores

Scores observed with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Baseline Score |
|---|---|
| classify | 0.60 – 1.00 |
| respond | 0.55 – 0.85 |
| resolve | 0.45 – 0.75 |
| **Average** | **0.55 – 0.87** |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Advance one step |
| `GET` | `/state` | Inspect current state (no side effects) |
| `GET` | `/tasks` | List all tasks with grader descriptions |

---

## Setup & Usage

### Environment Variables (required)

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_huggingface_token
```

### Local Docker Build

```bash
docker build -t support-agent-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=... \
  -e MODEL_NAME=... \
  -e HF_TOKEN=... \
  support-agent-env
```

### Run Inference Baseline

```bash
pip install -r requirements.txt
python inference.py
```

### Test Endpoints Manually

```bash
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"reply": "This is a billing issue."}'
curl http://localhost:7860/state
```

---

## Project Structure

```
support-agent-env/
├── inference.py          # Baseline inference script (required)
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── README.md
├── my_env/
│   ├── __init__.py
│   ├── env.py            # Main environment (reset/step/state/close)
│   ├── models.py         # Pydantic typed models (Observation/Action/Reward)
│   ├── grader.py         # Deterministic graders for all 3 tasks
│   └── data.py           # 10-email dataset (billing/complaint/query)
└── server/
    ├── __init__.py
    └── app.py            # FastAPI HTTP server for HF Space
```

---

## Infra Constraints

- Runtime: Python 3.10
- Memory: ≤ 8 GB
- vCPU: 2
- Inference time: < 20 minutes
- Port: 7860 (HuggingFace Spaces default)