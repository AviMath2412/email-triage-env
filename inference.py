"""
inference.py — Standardized inference script for Email Triage Environment.

MANDATORY Requirements (OpenEnv Grading):
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN.
- OpenAI Client for all LLM calls.
- Stdout Format: [START], [STEP], [END] tags.
- Score in [0, 1] range.

Usage:
    export OPENAI_API_KEY=sk-...
    python inference.py --task all --episodes 3
"""

import asyncio
import os
import json
import sys
import argparse
from typing import List, Optional, Dict, Any

from openai import OpenAI
from client import EmailTriageEnv
from server.models import EmailAction, EmailObservation
from server.data import TASKS

# --- Configuration ---
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.getenv("ENV_URL") or os.getenv("PING_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK    = "email-triage-env"
SCORE_EPSILON = 1e-3


# --- Stdout Logging ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def normalize_score(value: float) -> float:
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, value))


# --- LLM System Prompt ---
SYSTEM_PROMPT = """You are an expert executive assistant. You are interacting with
an email triage training environment. Each turn you receive a JSON observation and
must respond with a single JSON action — nothing else.

No markdown. No explanation. No code fences. Raw JSON only.

=== ACTION SCHEMAS ===

EASY TASK (classify):
{"action_type":"classify","email_id":"<id>","urgency":"urgent|normal|low|spam","priority":<1-5>}

MEDIUM TASK (rank):
{"action_type":"rank","ranked_ids":["email_001","email_002",...]}

HARD TASK (triage):
{
  "action_type":"triage",
  "email_id":"<id>",
  "urgency":"urgent|normal|low|spam",
  "priority":<1-5>,
  "reply_draft":"<professional reply, 50+ words>",
  "route_to":"support|sales|engineering|hr|finance|general"
}

=== ROUTING GUIDE ===
engineering — server/API outages, security bugs, technical failures
finance     — invoices, payments, GDPR data requests
support     — customer complaints, refund requests
sales       — pipeline reviews, partnership proposals, contract renewals
hr          — onboarding, benefits, performance reviews, hiring
general     — newsletters, office announcements, low-priority items
"""


def _build_user_prompt(obs: EmailObservation) -> str:
    lines = [
        f"TASK: {obs.task_id.upper()}",
        f"INSTRUCTIONS: {obs.task_description}",
        f"Step: {obs.step_count}/{obs.max_steps}",
    ]
    if obs.last_action_feedback:
        lines += ["", f"LAST FEEDBACK: {obs.last_action_feedback}"]
    if obs.single_email:
        e = obs.single_email
        lines += [
            "", "EMAIL:",
            f"  id: {e.id}",
            f"  from: {e.sender}",
            f"  subject: {e.subject}",
            f"  body: {e.body}",
        ]
    elif obs.inbox:
        lines += ["", f"INBOX ({len(obs.inbox)} emails):"]
        for e in obs.inbox:
            lines.append(f"  [{e.id}] from={e.sender} | subject={e.subject}")
    lines += ["", "Respond with one JSON action only."]
    return "\n".join(lines)


def get_action_from_model(client: OpenAI, obs: EmailObservation) -> EmailAction:
    user_prompt = _build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=600,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        start = raw.find("{")
        if start == -1:
            raise ValueError("No JSON found in model response.")
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return EmailAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model parse error: {exc}", file=sys.stderr)
        return EmailAction(action_type="done")


# --- Episode Runner ---
async def run_episode(client: OpenAI, task_id: str, env_url: str) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = EmailTriageEnv(base_url=env_url)
    try:
        async with env:
            result = await env.reset(task_id=task_id)
            obs = result.observation

            max_steps     = TASKS.get(task_id, {}).get("max_steps", 10)
            pass_threshold = TASKS.get(task_id, {}).get("pass_threshold", 0.70)

            for step in range(1, max_steps + 1):
                if obs.done:
                    break

                action = get_action_from_model(client, obs)
                result = await env.step(action)
                obs    = result.observation

                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action.action_type,
                    reward=reward,
                    done=obs.done or result.done,
                    error=None,
                )

                if obs.done or result.done:
                    break

        total_reward = sum(rewards)
        score = normalize_score(min(max(total_reward, 0.0), 1.0))
        success = total_reward >= pass_threshold

    except Exception as e:
        print(f"[DEBUG] Episode failed: {e}", file=sys.stderr)
        score = normalize_score(0.0)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


async def main() -> None:
    if not HF_TOKEN:
        print("\nERROR: OPENAI_API_KEY / HF_TOKEN environment variable not set.")
        print("Set it with: export HF_TOKEN=sk-...")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run LLM inference against Email Triage Env")
    parser.add_argument("--task",     default="all", choices=["all", "easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    args = parser.parse_args()

    # Pre-flight health check
    import requests
    try:
        requests.get(ENV_URL.rstrip("/") + "/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"\nERROR: Environment not reachable at {ENV_URL}")
        print(f"       Start it first: python -m uvicorn server.app:app --port 7860")
        print(f"       Error: {e}")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks_to_run = [args.task] if args.task != "all" else ["easy", "medium", "hard"]

    for tid in tasks_to_run:
        for ep in range(args.episodes):
            print(f"\n--- Task: {tid} | Episode: {ep + 1}/{args.episodes} ---")
            await run_episode(client, tid, ENV_URL)


if __name__ == "__main__":
    asyncio.run(main())
