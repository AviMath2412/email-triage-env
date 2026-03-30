"""
baseline.py — Baseline inference script using the OpenAI API.

Runs a GPT model zero-shot against all three tasks and reports graded scores.
Reads OPENAI_API_KEY from environment (never hardcoded).

IMPORTANT: The environment server must be running before you call this.
  Start server:  server   (after `pip install -e .`)
       OR        python server/app.py
       OR        docker run -p 7860:7860 email-triage-env

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py                         # all 3 tasks, 3 episodes each
    python baseline.py --task easy             # single task
    python baseline.py --episodes 5            # more episodes for stable scores
    python baseline.py --env-url http://localhost:7860
    python baseline.py --model gpt-4o          # stronger model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from openai import OpenAI
from client import EmailTriageEnv
from server.models import EmailAction, EmailObservation
from server.data import TASKS

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert executive assistant. You are interacting with
an email triage training environment. Each turn you receive a JSON observation and
must respond with a single JSON action — nothing else.

No markdown. No explanation. No code fences. Raw JSON only.

=== ACTION SCHEMAS ===

EASY TASK — classify one email:
{"action_type":"classify","email_id":"<id>","urgency":"urgent|normal|low|spam","priority":<1-5>}

MEDIUM TASK — rank all emails by priority (most urgent first):
{"action_type":"rank","ranked_ids":["email_001","email_002",...]}

HARD TASK — full triage (classify + write reply + route):
{
  "action_type":"triage",
  "email_id":"<id>",
  "urgency":"urgent|normal|low|spam",
  "priority":<1-5>,
  "reply_draft":"<professional reply text, 50+ words>",
  "route_to":"support|sales|engineering|hr|finance|general"
}

=== URGENCY & PRIORITY GUIDE ===
urgent  (5) — outages, overdue legal/finance, security alerts, customer complaints
normal  (3) — hiring follow-ups, business partnerships, non-urgent internal requests
low     (1-2) — event updates, team building, newsletters
spam    (1) — sales outreach, automated "out of office", shipping trackers

=== ROUTING GUIDE ===
engineering — server outages, API bugs, security vulnerabilities
finance     — invoices, payments, GDPR/Legal requests, audit help
support     — customer tickets, complaints, refund requests
sales       — pipeline reviews, new partnership proposals, demo requests
hr          — benefits enrollment, hiring, onboarding, vacation requests
general     — office news, broad internal updates, newsletter subscriptions
"""


def _build_user_prompt(obs: EmailObservation) -> str:
    """Convert observation to a concise prompt string."""
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
            "",
            "EMAIL:",
            f"  id: {e.id}",
            f"  from: {e.sender}",
            f"  subject: {e.subject}",
            f"  body: {e.body}",
            f"  has_attachment: {e.has_attachment}",
            f"  thread_length: {e.thread_length}",
        ]
    elif obs.inbox:
        lines += ["", f"INBOX ({len(obs.inbox)} emails):"]
        for e in obs.inbox:
            lines.append(f"  [{e.id}] from={e.sender} | subject={e.subject}")

    lines += ["", "Respond with one JSON action only."]
    return "\n".join(lines)


def _call_llm(client: OpenAI, obs: EmailObservation, model: str) -> EmailAction:
    """Call OpenAI and parse the response as an EmailAction."""
    prompt = _build_user_prompt(obs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()

        # Robust JSON extraction: look for the first '{' and extract the object
        start = raw.find("{")
        if start == -1:
            raise ValueError("No JSON block found in LLM response.")
            
        try:
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
            return EmailAction(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback to crude search if raw_decode fails
            end = raw.rfind("}")
            if end != -1:
                return EmailAction(**json.loads(raw[start : end + 1]))
            raise

    except Exception as exc:
        print(f"    [LLM error] {type(exc).__name__}: {exc}")
        if 'raw' in locals():
            print(f"    Raw content: {raw[:200]}...")
        return EmailAction(action_type="done")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    base_url: str,
    openai_client: OpenAI,
    task_id: str,
    episode_num: int,
    model: str,
    verbose: bool = True,
) -> dict:
    """Run one full episode. Returns score dict."""

    async with EmailTriageEnv(base_url=base_url) as env:
        # Pass task_id as a reset kwarg — the server reads it from the payload
        result = await env.reset(task_id=task_id)
        obs = result.observation

        if verbose:
            print(f"\n  Episode {episode_num} | task={task_id}")
            if obs.single_email:
                print(f"  Email: {obs.single_email.subject[:65]}...")
            elif obs.inbox:
                print(f"  Inbox: {len(obs.inbox)} emails")

        total_reward = 0.0
        actions_log  = []
        step = 0

        while not obs.done:
            action = _call_llm(openai_client, obs, model)
            result = await env.step(action)
            obs    = result.observation
            rew    = result.reward or 0.0
            total_reward += rew
            actions_log.append(action.model_dump(exclude_none=True))
            step += 1

            if verbose:
                feedback = (obs.last_action_feedback or "")[:70]
                print(f"    step {step}: {action.action_type} | reward={rew:.3f} | {feedback}...")

            if obs.done or result.done:
                break

    # Hit the grader endpoint directly (REST, no WS needed)
    import requests
    grader_url = base_url.rstrip("/") + "/grader"
    grade = requests.post(grader_url, json={
        "task_id":     task_id,
        "final_score": max(0.0, total_reward),
        "actions":     actions_log,
    }).json()

    if verbose:
        print(f"  → grader_score={grade['grader_score']:.4f}  grade={grade['grade'].upper()}")

    return {
        "episode":      episode_num,
        "task_id":      task_id,
        "total_reward": round(total_reward, 4),
        "grader_score": grade["grader_score"],
        "grade":        grade["grade"],
        "steps":        step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY environment variable not set.\n"
            "Set it with:  export OPENAI_API_KEY=sk-..."
        )

    openai_client = OpenAI(api_key=api_key)
    base_url = args.env_url or os.environ.get("ENV_URL", "http://localhost:7860")
    tasks    = list(TASKS.keys()) if args.task == "all" else [args.task]
    model    = args.model

    print(f"Baseline agent: {model}")
    print(f"Environment:    {base_url}")
    print(f"Tasks:          {tasks}")
    print(f"Episodes/task:  {args.episodes}")

    all_results: dict[str, dict] = {}

    for task_id in tasks:
        print(f"\n{'='*55}")
        print(f"TASK: {task_id.upper()} — {TASKS[task_id]['name']}")
        print(f"{'='*55}")

        episodes = []
        for ep in range(1, args.episodes + 1):
            result = await run_episode(
                base_url, openai_client, task_id, ep, model, verbose=True
            )
            episodes.append(result)

        avg   = sum(r["grader_score"] for r in episodes) / len(episodes)
        passes = sum(1 for r in episodes if r["grade"] != "fail") / len(episodes)
        all_results[task_id] = {
            "avg_grader_score": round(avg, 4),
            "pass_rate":        round(passes, 4),
            "episodes":         episodes,
        }
        print(f"\n  Summary [{task_id}]: avg={avg:.4f}  pass_rate={passes:.0%}")

    # Final table
    print(f"\n{'='*55}")
    print(f"BASELINE SUMMARY  (model: {model})")
    print(f"{'='*55}")
    print(f"{'Task':<10} {'Avg Score':>12} {'Pass Rate':>10}")
    print("-" * 36)
    for tid, res in all_results.items():
        print(f"{tid:<10} {res['avg_grader_score']:>12.4f} {res['pass_rate']:>10.0%}")

    # Persist results
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "evals", "baseline_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"model": model, "env_url": base_url, "results": all_results}, fh, indent=2)
    print(f"\nResults saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenAI baseline agent against the Email Triage Environment"
    )
    parser.add_argument(
        "--task", default="all",
        choices=["all", "easy", "medium", "hard"],
        help="Which task(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Episodes per task for score averaging (default: 3)",
    )
    parser.add_argument(
        "--env-url", default=None,
        help="Environment server URL (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
