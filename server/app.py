"""
FastAPI server — exposes OpenEnv HTTP endpoints for HuggingFace Space.

Endpoints
---------
GET  /            health check
POST /reset       start a new episode
POST /step        advance by one step
GET  /state       inspect current state (no side effects)
GET  /tasks       list all tasks with descriptions and graders
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from my_env.env import CustomerSupportEnv

app = FastAPI(
    title="Customer Support Agent Environment",
    description="OpenEnv-compliant environment for evaluating LLM agents on customer support tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared env instance (stateful per-session)
env = CustomerSupportEnv()


class ActionRequest(BaseModel):
    reply: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {
        "status": "running",
        "env": "customer-support-agent",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks"],
    }


@app.post("/reset")
async def reset():
    """Start a new episode. Returns initial observation."""
    try:
        result = await env.reset()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: ActionRequest):
    """Advance the environment by one step."""
    try:
        result = await env.step({"reply": action.reply})
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Get current environment state without advancing it."""
    try:
        result = await env.state()
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def tasks():
    """Return task definitions with difficulty and grader descriptions."""
    return {
        "tasks": [
            {
                "id": "classify",
                "name": "Email Classification",
                "difficulty": "easy",
                "description": "Classify the customer email into: billing, complaint, or query.",
                "grader": "Deterministic keyword + synonym matching. Score: 1.0 (exact), 0.6 (synonym), 0.0 (miss).",
                "score_range": [0.0, 1.0],
            },
            {
                "id": "respond",
                "name": "Professional Response Generation",
                "difficulty": "medium",
                "description": "Draft a professional customer support response addressing the issue.",
                "grader": "Multi-criterion: keyword coverage (40%), apology (20%), action (20%), length (10%), professionalism (10%).",
                "score_range": [0.0, 1.0],
            },
            {
                "id": "resolve",
                "name": "Full Issue Resolution",
                "difficulty": "hard",
                "description": "Fully resolve the customer issue end-to-end: classify, respond, and provide a concrete resolution path.",
                "grader": "Combined: classification (20%), keyword coverage (30%), apology+action (20%), resolution criteria (20%), professionalism (10%).",
                "score_range": [0.0, 1.0],
            },
        ]
    }