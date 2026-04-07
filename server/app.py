"""
server/app.py — FastAPI application for the Email Triage Environment.

Exposes standard OpenEnv endpoints (/ws, /reset, /step, /state)
plus additional triage-specific metadata and grading utilities.
"""

from __future__ import annotations

import os
import uvicorn
from fastapi import HTTPException
from openenv.core.env_server import create_fastapi_app

from .models import EmailAction, EmailObservation
from .environment import EmailTriageEnvironment
from .data import TASKS
from .baseline_heuristic import heuristic_agent

SCORE_EPSILON = 1e-3


def _strict_score(value: float) -> float:
    """Clamp score into (0, 1) for external validators."""
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, value))


# ---------------------------------------------------------------------------
# Create the core OpenEnv FastAPI app
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=EmailTriageEnvironment,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
)


# ---------------------------------------------------------------------------
# Metadata & Grading Endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["Metadata"])
async def list_tasks():
    """Returns all available tasks and their success thresholds."""
    return {"tasks": list(TASKS.values())}


@app.get("/tasks/{task_id}", tags=["Metadata"])
async def get_task(task_id: str):
    """Returns details for a specific task including the required schema."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return {
        **TASKS[task_id],
        "action_schema": EmailAction.model_json_schema(),
    }


@app.post("/grader", tags=["Grader"])
async def grade_episode(request: dict):
    """Computes a final grade for a completed episode."""
    task_id = request.get("task_id")
    final_score = _strict_score(float(request.get("final_score", 0.0)))
    actions = request.get("actions", [])

    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    t = TASKS[task_id]
    n_steps = len(actions)
    efficiency = max(0.0, 1.0 - max(0, n_steps - 1) / t["max_steps"])
    grader_score = _strict_score(round(final_score * 0.85 + efficiency * 0.15, 4))

    passed    = final_score >= t["pass_threshold"]
    excellent = final_score >= t["excellent_threshold"]

    return {
        "task_id":      task_id,
        "raw_score":    round(final_score, 4),
        "grader_score": grader_score,
        "passed":       passed,
        "excellent":    excellent,
        "grade":        "excellent" if excellent else ("passing" if passed else "fail"),
        "thresholds": {
            "pass":      t["pass_threshold"],
            "excellent": t["excellent_threshold"],
        },
        "metrics": {
            "action_count": n_steps,
            "efficiency":   round(efficiency, 4),
        },
    }


@app.get("/baseline", tags=["Evaluation"])
async def run_baseline(task_id: str | None = None):
    """Executes the built-in heuristic agent. Used for benchmarking."""
    target_tasks = [task_id] if task_id else list(TASKS.keys())
    if task_id and task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    results = {}
    for tid in target_tasks:
        env = EmailTriageEnvironment()
        obs = env.reset(task_id=tid)

        total_reward, n_steps = 0.0, 0
        while not obs.done and n_steps < TASKS[tid]["max_steps"]:
            action = heuristic_agent(obs, tid)
            obs = env.step(action)
            total_reward += (obs.reward or 0.0)
            n_steps += 1

        results[tid] = {
            "score": round(_strict_score(max(0.0, total_reward)), 4),
            "steps": n_steps,
        }

    return {
        "agent":   "keyword_heuristic_v1",
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """Entry point for `server` CLI script and Docker CMD."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
