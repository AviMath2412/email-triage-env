"""
client.py — Python SDK for the Email Triage Environment.

Extends openenv-core's EnvClient, implementing the three required abstract methods:
  _step_payload()  — serialize EmailAction → dict for WebSocket
  _parse_result()  — deserialize response dict → StepResult[EmailObservation]
  _parse_state()   — deserialize state dict → EmailState

Usage:
    with EmailTriageEnv(base_url="http://localhost:8000") as env:
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
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from server.models import EmailAction, EmailObservation, EmailState


class EmailTriageEnv(EnvClient[EmailAction, EmailObservation, EmailState]):
    """
    Client for the Email Triage Environment.
    Inherits all connection management and WebSocket protocol from EnvClient.
    """

    def _step_payload(self, action: EmailAction) -> Dict[str, Any]:
        """Serialize EmailAction to dict sent over WebSocket."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[EmailObservation]:
        """Deserialize server response → StepResult[EmailObservation]."""
        obs_data = payload.get("observation", payload)
        obs = EmailObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward") or obs.reward,
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EmailState:
        """Deserialize state response → EmailState."""
        return EmailState(**payload)


__all__ = ["EmailTriageEnv", "EmailAction", "EmailObservation", "EmailState"]
