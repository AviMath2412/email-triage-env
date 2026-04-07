"""
server/environment.py — Email Triage Environment logic.

Extends openenv.core.env_server.interfaces.Environment, implementing:
  - reset()  → start new episode, return EmailObservation
  - step()   → process action, compute reward, return EmailObservation
  - state    → property returning EmailState

The reward function provides DENSE signal at every step with partial credit.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment

from .models import (
    Email, EmailAction, EmailObservation, EmailState,
    ActionType, UrgencyLabel, Department,
)
from .data import EMAIL_TEMPLATES, TASKS


def _make_email(template_index: int, email_num: int) -> tuple[Email, dict]:
    subj, sender, body, urgency, priority, dept = EMAIL_TEMPLATES[template_index]
    email = Email(
        id=f"email_{email_num:03d}",
        subject=subj,
        sender=sender,
        body=body,
        timestamp=f"2024-03-15T{9 + email_num:02d}:00:00Z",
        has_attachment=random.random() > 0.75,
        thread_length=random.randint(1, 4),
    )
    return email, {"urgency": urgency, "priority": priority, "department": dept}


class EmailTriageEnvironment(Environment):
    """
    Email Triage RL Environment.

    Three tasks of increasing difficulty:
      easy   — classify a single email (urgency + priority)
      medium — rank 10 emails by priority (Kendall-tau scoring)
      hard   — full pipeline: classify + draft reply + route to department
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = EmailState()
        self._emails: list[Email] = []
        self._ground_truths: dict[str, dict] = {}
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._task_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="Email Triage Environment",
            description=(
                "A real-world RL environment where agents triage emails: "
                "classifying urgency, ranking inbox priority, drafting replies, "
                "and routing to the correct department."
            ),
            version="1.0.0",
        )

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> EmailObservation:
        """Start a new episode. Returns initial observation."""
        if seed is not None:
            random.seed(seed)

        task_id = task_id or kwargs.get("task_id") or random.choice(list(TASKS.keys()))
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")

        task = TASKS[task_id]
        self._task_id = task_id
        self._emails = []
        self._ground_truths = {}
        self._cumulative_reward = 0.0
        self._done = False

        self._state = EmailState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=task["max_steps"],
            cumulative_reward=0.0,
            done=False,
        )

        # Generate emails
        if task_id == "easy":
            idx = random.randrange(len(EMAIL_TEMPLATES))
            email, gt = _make_email(idx, 1)
            self._emails = [email]
            self._ground_truths[email.id] = gt

        elif task_id == "medium":
            indices = random.sample(range(len(EMAIL_TEMPLATES)), 10)
            for i, idx in enumerate(indices):
                email, gt = _make_email(idx, i + 1)
                self._emails.append(email)
                self._ground_truths[email.id] = gt

        elif task_id == "hard":
            urgent_indices = [
                i for i, t in enumerate(EMAIL_TEMPLATES)
                if t[3] == UrgencyLabel.URGENT
            ]
            idx = random.choice(urgent_indices)
            email, gt = _make_email(idx, 1)
            self._emails = [email]
            self._ground_truths[email.id] = gt

        self._state.email_count = len(self._emails)
        return self._make_obs()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        action: EmailAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> EmailObservation:
        """Process one action. Returns updated observation with reward signal."""

        if self._task_id is None:
            obs = self._make_obs_error("Call /reset before /step.")
            obs.done = True
            return obs

        if self._done:
            obs = self._make_obs(
                feedback="Episode already complete. No further actions needed.",
                reward=-0.1,
                efficiency_penalty=-0.1,
            )
            obs.done = True
            return obs

        self._state.step_count += 1

        if action.action_type == ActionType.DONE:
            return self._handle_done()

        if self._state.step_count > self._state.max_steps:
            self._done = True
            self._state.done = True
            obs = self._make_obs(
                feedback=f"Max steps ({self._state.max_steps}) exceeded. Episode forced to end.",
                reward=-0.2,
                efficiency_penalty=-0.2,
            )
            obs.done = True
            return obs

        if self._task_id == "easy":
            return self._step_easy(action)
        elif self._task_id == "medium":
            return self._step_medium(action)
        elif self._task_id == "hard":
            return self._step_hard(action)
        else:
            raise RuntimeError(f"Unknown task_id: {self._task_id}")

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> EmailState:
        return self._state

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    def _step_easy(self, action: EmailAction) -> EmailObservation:
        """
        Easy: classify a single email.

        Reward:
          0.50 × urgency correct (exact match)
          0.30 × priority score (1.0 exact / 0.5 off-by-1 / 0.25 off-by-2)
          0.20 × efficiency bonus (decays with step count)
        Penalties:
          -0.10 wrong action type
          -0.05 missing fields
        """
        if action.action_type != ActionType.CLASSIFY:
            return self._make_obs(
                feedback="Wrong action type. Use action_type='classify' for the easy task.",
                reward=-0.10, efficiency_penalty=-0.10,
            )

        if not action.email_id or action.urgency is None or action.priority is None:
            return self._make_obs(
                feedback="Missing fields. classify requires: email_id, urgency, priority.",
                reward=-0.05, efficiency_penalty=-0.05,
            )

        gt = self._ground_truths.get(action.email_id)
        if not gt:
            return self._make_obs(
                feedback=f"Unknown email_id '{action.email_id}'.",
                reward=-0.10, efficiency_penalty=-0.10,
            )

        urgency_score = 1.0 if action.urgency == gt["urgency"] else 0.0
        diff = abs(action.priority - gt["priority"])
        priority_score = 1.0 if diff == 0 else (0.5 if diff == 1 else (0.25 if diff == 2 else 0.0))
        efficiency = max(0.0, 0.20 * (1 - (self._state.step_count - 1) / self._state.max_steps))
        correctness = 0.50 * urgency_score + 0.30 * priority_score
        total = round(correctness + efficiency, 4)

        self._cumulative_reward += total
        self._state.cumulative_reward = round(self._cumulative_reward, 4)
        self._done = True
        self._state.done = True

        feedback = (
            f"Urgency: {'✓' if urgency_score else '✗'} "
            f"(yours={action.urgency.value}, correct={gt['urgency'].value}) | "
            f"Priority: {priority_score:.2f}/1.0 "
            f"(yours={action.priority}, correct={gt['priority']}) | "
            f"Efficiency bonus: +{efficiency:.2f}"
        )
        obs = self._make_obs(
            feedback=feedback,
            reward=total,
            correctness_score=correctness,
            completion_bonus=efficiency,
        )
        obs.done = True
        obs.metadata["ground_truth"] = {k: str(v) for k, v in gt.items()}
        return obs

    def _step_medium(self, action: EmailAction) -> EmailObservation:
        """
        Medium: rank 10 emails by priority using Kendall-tau scoring.

        Reward = tau * 0.85 + improvement_bonus * 0.15
        Episode ends early when tau >= 0.95.
        """
        if action.action_type != ActionType.RANK:
            return self._make_obs(
                feedback="Wrong action type. Use action_type='rank' with ranked_ids.",
                reward=-0.10, efficiency_penalty=-0.10,
            )

        if not action.ranked_ids:
            return self._make_obs(
                feedback="ranked_ids is required — list all email IDs in priority order.",
                reward=-0.05, efficiency_penalty=-0.05,
            )

        all_ids = {e.id for e in self._emails}
        submitted = set(action.ranked_ids)
        missing = all_ids - submitted
        extra   = submitted - all_ids
        if missing or extra:
            return self._make_obs(
                feedback=f"ranked_ids mismatch. Missing: {missing}. Extra: {extra}.",
                reward=-0.10, efficiency_penalty=-0.10,
            )

        correct_order = sorted(
            self._emails,
            key=lambda e: self._ground_truths[e.id]["priority"],
            reverse=True,
        )
        correct_ids = [e.id for e in correct_order]

        n = len(action.ranked_ids)
        pos_sub = {eid: i for i, eid in enumerate(action.ranked_ids)}
        pos_cor = {eid: i for i, eid in enumerate(correct_ids)}
        concordant = sum(
            1
            for i in range(n)
            for j in range(i + 1, n)
            if pos_sub[correct_ids[i]] < pos_sub[correct_ids[j]]
        )
        total_pairs = n * (n - 1) // 2
        tau = concordant / total_pairs if total_pairs else 0.0

        prev_score = self._cumulative_reward
        improvement = tau - prev_score
        total = round(tau * 0.85 + max(0, improvement) * 0.15, 4)

        self._cumulative_reward = tau
        self._state.cumulative_reward = round(tau, 4)
        perfect = tau >= 0.95
        self._done = perfect
        self._state.done = perfect

        feedback = (
            f"Kendall-tau: {tau:.3f} ({concordant}/{total_pairs} pairs correct). "
            f"{'Perfect ranking! Episode complete.' if perfect else 'Keep refining your ranking.'}"
        )
        obs = self._make_obs(
            feedback=feedback,
            reward=total,
            correctness_score=tau,
        )
        obs.done = perfect
        obs.metadata["tau_score"]     = tau
        obs.metadata["correct_order"] = correct_ids
        return obs

    def _step_hard(self, action: EmailAction) -> EmailObservation:
        """
        Hard: full triage pipeline.

        Weighted score:
          30% — classification (urgency + priority)
          30% — reply quality (heuristic)
          40% — routing accuracy
          +0.15 efficiency bonus
        """
        if action.action_type != ActionType.TRIAGE:
            return self._make_obs(
                feedback="Wrong action type. Use action_type='triage' for the hard task.",
                reward=-0.10, efficiency_penalty=-0.10,
            )

        required = [action.email_id, action.urgency, action.priority,
                    action.reply_draft, action.route_to]
        if any(f is None for f in required):
            return self._make_obs(
                feedback="All fields required: email_id, urgency, priority, reply_draft, route_to.",
                reward=-0.05, efficiency_penalty=-0.05,
            )

        gt = self._ground_truths.get(action.email_id)
        if not gt:
            return self._make_obs(
                feedback=f"Unknown email_id '{action.email_id}'.",
                reward=-0.10, efficiency_penalty=-0.10,
            )

        urgency_ok = action.urgency == gt["urgency"]
        diff = abs(action.priority - gt["priority"])
        prio_score = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)
        classify_score = 0.5 * urgency_ok + 0.5 * prio_score

        reply_score = self._score_reply(action.reply_draft, action.email_id)

        route_ok = action.route_to == gt["department"]
        route_score = 1.0 if route_ok else 0.0

        efficiency = max(0.0, 0.15 * (1 - (self._state.step_count - 1) / self._state.max_steps))

        correctness = 0.30 * classify_score + 0.30 * reply_score + 0.40 * route_score
        total = round(correctness + efficiency, 4)

        self._cumulative_reward += total
        self._state.cumulative_reward = round(self._cumulative_reward, 4)
        self._done = True
        self._state.done = True

        feedback = (
            f"Classification: {classify_score:.2f}/1.0 | "
            f"Reply quality: {reply_score:.2f}/1.0 | "
            f"Routing: {'✓' if route_ok else '✗'} "
            f"(yours={action.route_to.value if action.route_to else 'None'}, "
            f"correct={gt['department'].value}) | "
            f"Efficiency: +{efficiency:.2f}"
        )
        obs = self._make_obs(
            feedback=feedback,
            reward=total,
            correctness_score=correctness,
            completion_bonus=efficiency,
        )
        obs.done = True
        obs.metadata["ground_truth"] = {k: str(v) for k, v in gt.items()}
        return obs

    def _handle_done(self) -> EmailObservation:
        self._done = True
        self._state.done = True
        if self._cumulative_reward < 0.05 and self._state.step_count <= 1:
            obs = self._make_obs(
                feedback="Quit immediately without attempting the task.",
                reward=-0.20, efficiency_penalty=-0.20,
            )
        else:
            obs = self._make_obs(feedback="Agent signalled done. Episode complete.")
        obs.done = True
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs_error(self, message: str) -> EmailObservation:
        return EmailObservation(
            task_id="none",
            task_description=message,
            step_count=0,
            max_steps=0,
            last_action_feedback=message,
            cumulative_reward=0.0,
            reward=-0.1,
            done=True,
        )

    def _score_reply(self, reply: str, email_id: str) -> float:
        """Heuristic reply quality scoring (0.0–1.0)."""
        if not reply or len(reply.strip()) < 20:
            return 0.0
        rl = reply.lower()
        score = 0.0

        # Length (25%)
        wc = len(reply.split())
        score += 0.25 if wc >= 50 else (0.10 if wc >= 25 else 0.0)

        # Greeting (20%)
        if any(g in rl for g in ["dear", "hi ", "hello", "good morning", "thank you for"]):
            score += 0.20

        # Acknowledgment of subject matter (20%)
        email = next((e for e in self._emails if e.id == email_id), None)
        if email:
            subject_words = set(email.subject.lower().split()) - {"the", "a", "an", "is", "for", "re:", "—", "-"}
            reply_words   = set(rl.split())
            overlap = len(subject_words & reply_words) / max(1, len(subject_words))
            score += 0.20 * min(1.0, overlap * 2)

        # Committed action (20%)
        if any(p in rl for p in ["will", "we'll", "i'll", "investigating", "refund", "resolved", "fix", "escalated"]):
            score += 0.20

        # Sign-off (15%)
        if any(s in rl for s in ["regards", "sincerely", "best", "thank you", "thanks", "cheers"]):
            score += 0.15

        return round(min(1.0, score), 3)

    def _make_obs(
        self,
        feedback: Optional[str] = None,
        reward: float = 0.0,
        correctness_score: float = 0.0,
        efficiency_penalty: float = 0.0,
        completion_bonus: float = 0.0,
    ) -> EmailObservation:
        task = TASKS[self._task_id]
        single_email = self._emails[0] if len(self._emails) == 1 else None
        inbox        = self._emails    if len(self._emails) > 1  else []

        return EmailObservation(
            task_id=self._task_id,
            task_description=task["description"],
            single_email=single_email,
            inbox=inbox,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            last_action_feedback=feedback,
            cumulative_reward=round(self._cumulative_reward, 4),
            reward=reward,
            done=self._done,
            correctness_score=correctness_score,
            efficiency_penalty=efficiency_penalty,
            completion_bonus=completion_bonus,
        )
