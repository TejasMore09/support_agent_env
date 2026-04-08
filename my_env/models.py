"""
OpenEnv typed models — Observation, Action, Reward
All fields fully typed and documented.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    What the agent sees at each step.
    - email_id:   stable identifier so the agent can reference prior context
    - email_body: the raw customer email text
    - task:       what the agent must do this step
    - step_num:   current step within the episode (1-indexed)
    """
    email_id: str = Field(..., description="Unique identifier for this email")
    email_body: str = Field(..., description="Full text of the customer email")
    task: Literal["classify", "respond", "resolve"] = Field(
        ..., description="Current task the agent must complete"
    )
    step_num: int = Field(..., ge=1, description="Current step number in the episode")


class Action(BaseModel):
    """
    What the agent submits at each step.
    - reply: free-text response / classification / resolution
    """
    reply: str = Field(
        ...,
        min_length=1,
        description="Agent's textual output for the current task"
    )


class Reward(BaseModel):
    """
    Structured reward breakdown returned at each step.
    - total:      final scalar reward [0.0, 1.0]
    - breakdown:  per-criterion scores for interpretability
    - feedback:   human-readable explanation
    """
    total: float = Field(..., ge=0.0, le=1.0, description="Total reward [0, 1]")
    breakdown: dict = Field(default_factory=dict, description="Per-criterion scores")
    feedback: str = Field(default="", description="Grader feedback string")


class StepResult(BaseModel):
    """Full result returned by env.step()"""
    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Full result returned by env.reset()"""
    observation: Observation
    info: dict = Field(default_factory=dict)


class StateResult(BaseModel):
    """Full result returned by env.state()"""
    observation: Observation
    step_num: int
    episode_rewards: list[float]
    done: bool