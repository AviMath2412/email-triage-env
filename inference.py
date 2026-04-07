"""
inference.py — Standardized inference script for Email Triage Environment.

MANDATORY Requirements (Hugging Face / OpenEnv Grading):
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN.
- OpenAI Client for all LLM calls.
- Stdout Format: [START], [STEP], [END] tags.
- Score in [0, 1] range.
"""

import asyncio
import os
import json
import sys
import argparse
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI
from client import EmailTriageEnv
from server.models import EmailAction, EmailObservation
from server.data import TASKS

# --- Configuration & Environment Setup ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
BENCHMARK = "email-triage-env"
ENV_URL = os.getenv("ENV_URL") or os.getenv("PING_URL") or "http://localhost:7860"

# --- Stdout Logging Helpers ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- LLM Integration ---
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
  "reply_draft":"<professional reply text, 50+ words>",
  "route_to":"support|sales|engineering|hr|finance|general"
}
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
        lines += ["", "EMAIL:", f"  id: {e.id}", f"  from: {e.sender}", f"  subject: {e.subject}", f"  body: {e.body}"]
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
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=600,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        
        # Robust JSON extraction
        start = raw.find("{")
        if start == -1: raise ValueError("No JSON found.")
        
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return EmailAction(**data)
    except Exception as exc:
        # On failure, return 'done' action to gracefully exit the episode
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
            
            # Use max_steps from TASKS metadata
            max_steps = TASKS.get(task_id, {}).get("max_steps", 10)
            pass_threshold = TASKS.get(task_id, {}).get("pass_threshold", 0.70)

            for step in range(1, max_steps + 1):
                if obs.done:
                    break
                
                action = get_action_from_model(client, obs)
                result = await env.step(action)
                obs = result.observation
                
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action.action_type, reward=reward, done=obs.done or result.done, error=None)
                
                if obs.done or result.done:
                    break

        # Calculate score normalized to [0, 1]
        # In this env, rewards are per-step and often binary or close to 1.0 for success.
        # We'll use the cumulative reward / max_steps as a simple proxy, clamping to [0, 1].
        total_reward = sum(rewards)
        score = min(max(total_reward, 0.0), 1.0) # Heuristic for this evaluation spec
        success = total_reward >= pass_threshold
        
    except Exception as e:
        print(f"[DEBUG] Episode failed: {e}", file=sys.stderr)
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    if not API_KEY:
        print("\nERROR: OPENAI_API_KEY / HF_TOKEN environment variable not set.")
        print("Set it with: export HF_TOKEN=sk-...")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["all", "easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    # Pre-flight check
    import requests
    try:
        requests.get(ENV_URL.rstrip("/") + "/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"\nERROR: Environment container not reachable at: {ENV_URL}")
        print(f"       Health check failed: {e}")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    tasks_to_run = [args.task] if args.task != "all" else ["easy", "medium", "hard"]
    
    for tid in tasks_to_run:
        for _ in range(args.episodes):
            await run_episode(client, tid, ENV_URL)

if __name__ == "__main__":
    asyncio.run(main())
