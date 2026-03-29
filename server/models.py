"""
Standard Pydantic models for Actions, Observations, and Episode State.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UrgencyLabel(str, Enum):
    URGENT = "urgent"   # same-day response required
    NORMAL = "normal"   # 24-48h response window
    LOW    = "low"      # no rush, FYI emails
    SPAM   = "spam"     # no response needed


class Department(str, Enum):
    SUPPORT     = "support"
    SALES       = "sales"
    ENGINEERING = "engineering"
    HR          = "hr"
    FINANCE     = "finance"
    GENERAL     = "general"


class ActionType(str, Enum):
    CLASSIFY = "classify"   # label urgency + priority (easy task)
    RANK     = "rank"       # sort all emails by priority (medium task)
    TRIAGE   = "triage"     # classify + draft reply + route (hard task)
    DONE     = "done"       # signal episode complete


# ---------------------------------------------------------------------------
# Email Data Model
# ---------------------------------------------------------------------------

class Email(Action):
    """
    Represents a single email in the inbox.
    Inherits from Action so it can be embedded cleanly in observations.
    Used as a data container — not an agent action.
    """
    id: str              = Field(..., description="Unique email ID, e.g. 'email_001'")
    subject: str         = Field(..., description="Email subject line")
    sender: str          = Field(..., description="Sender email address")
    body: str            = Field(..., description="Full email body text")
    timestamp: str       = Field(..., description="ISO 8601 timestamp")
    has_attachment: bool = Field(False, description="Whether the email has file attachments")
    thread_length: int   = Field(1, description="Number of messages in thread")


# ---------------------------------------------------------------------------
# Agent Action
# ---------------------------------------------------------------------------

class EmailAction(Action):
    """
    The agent's move. Different fields are populated depending on action_type.

    Easy   → action_type=CLASSIFY, email_id + urgency + priority
    Medium → action_type=RANK,     ranked_ids (all IDs in priority order)
    Hard   → action_type=TRIAGE,   email_id + urgency + priority + reply_draft + route_to
    Done   → action_type=DONE      (signals episode complete)
    """
    action_type: ActionType = Field(..., description="Which action the agent is taking")

    # For CLASSIFY and TRIAGE
    email_id:    Optional[str]          = Field(None, description="ID of email being acted on")
    urgency:     Optional[UrgencyLabel] = Field(None, description="Agent's urgency classification")
    priority:    Optional[int]          = Field(None, ge=1, le=5, description="1=lowest, 5=highest")

    # For RANK only
    ranked_ids:  Optional[list[str]]   = Field(None, description="Email IDs sorted highest→lowest priority")

    # For TRIAGE only
    reply_draft: Optional[str]         = Field(None, description="Drafted reply to the email")
    route_to:    Optional[Department]  = Field(None, description="Department to route this email to")


# ---------------------------------------------------------------------------
# Environment Observation
# ---------------------------------------------------------------------------

class EmailObservation(Observation):
    """
    What the agent sees at each step.

    Inherits from Observation, which already provides:
      .reward (float | None) — the step reward signal
      .done   (bool)         — whether the episode has ended
      .metadata (dict)       — extra debug info

    We add the email-specific fields on top.
    """
    task_id: str                         = Field(...,  description="Active task: easy/medium/hard")
    task_description: str                = Field(...,  description="Plain-English instructions for the agent")
    single_email: Optional[Email]        = Field(None, description="The email to act on (easy/hard tasks)")
    inbox: list[Email]                   = Field(default_factory=list, description="All inbox emails (medium task)")
    step_count: int                      = Field(0,    description="How many steps taken this episode")
    max_steps: int                       = Field(10,   description="Max steps before forced termination")
    last_action_feedback: Optional[str]  = Field(None, description="Explanation of last reward")
    cumulative_reward: float             = Field(0.0,  description="Total reward accumulated this episode")

    # Reward breakdown fields (for transparency and training signal interpretation)
    correctness_score: float             = Field(0.0,  description="How correct was this action")
    efficiency_penalty: float            = Field(0.0,  description="Penalty for wasted/bad steps")
    completion_bonus: float              = Field(0.0,  description="Bonus for clean task completion")


# ---------------------------------------------------------------------------
# Server-side State
# ---------------------------------------------------------------------------

class EmailState(State):
    """
    Server-side episode metadata. Extends the base State which already has:
      .episode_id (str | None)
      .step_count (int)

    We add task-specific tracking fields.
    """
    task_id: Optional[str]   = Field(None, description="Active task ID")
    max_steps: int            = Field(10,   description="Max steps for current task")
    cumulative_reward: float  = Field(0.0,  description="Total reward so far")
    done: bool                = Field(False, description="Whether episode is complete")
    email_count: int          = Field(0,    description="Number of emails in this episode")
