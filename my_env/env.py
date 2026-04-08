"""
Customer Support Agent Environment
===================================
A real-world OpenEnv environment where an LLM agent handles customer support emails
through 3 progressively harder tasks:

  Task 1 — classify:  Identify the email category (billing/complaint/query)
  Task 2 — respond:   Draft a professional reply addressing the issue
  Task 3 — resolve:   Fully resolve the issue end-to-end

State machine per episode:
  reset() → step(classify) → step(respond) → step(resolve) → done=True

All graders are deterministic and reproducible.
Rewards are dense (partial credit at every step), never sparse.
"""

from __future__ import annotations
import random
from typing import Any

from my_env.data import EMAIL_DATA
from my_env.models import (
    Observation, Action, Reward, StepResult, ResetResult, StateResult
)
from my_env.grader import (
    grade_classification,
    grade_response,
    grade_resolution,
)

TASK_SEQUENCE = ["classify", "respond", "resolve"]


class CustomerSupportEnv:
    """
    OpenEnv-compliant Customer Support environment.

    Methods
    -------
    reset()   → ResetResult
    step()    → StepResult
    state()   → StateResult
    close()   → None
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._current: dict | None = None
        self._step_count: int = 0
        self._episode_rewards: list[float] = []
        self._done: bool = False
        self._last_observation: Observation | None = None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        task = TASK_SEQUENCE[min(self._step_count, len(TASK_SEQUENCE) - 1)]
        return Observation(
            email_id=self._current["id"],
            email_body=self._current["email"],
            task=task,
            step_num=self._step_count + 1,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def reset(self) -> dict:
        """
        Start a new episode. Randomly selects an email from the dataset.
        Returns initial observation with task='classify'.
        """
        self._current = self._rng.choice(EMAIL_DATA)
        self._step_count = 0
        self._episode_rewards = []
        self._done = False
        self._last_observation = self._build_observation()

        result = ResetResult(
            observation=self._last_observation,
            info={
                "email_id": self._current["id"],
                "category": self._current["category"],
                "urgency": self._current["urgency"],
                "total_steps": len(TASK_SEQUENCE),
            }
        )
        return result.model_dump()

    async def step(self, action: dict) -> dict:
        """
        Advance the environment by one step.

        Parameters
        ----------
        action : dict — must contain key "reply" (str)

        Returns
        -------
        StepResult dict with keys: observation, reward, done, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        reply = action.get("reply", "").strip()
        if not reply:
            reply = ""

        current_task = TASK_SEQUENCE[self._step_count]
        email_data = self._current

        # ── Grade current step ────────────────────────────────────────────────
        if current_task == "classify":
            score, feedback = grade_classification(reply, email_data["category"])
        elif current_task == "respond":
            score, feedback = grade_response(reply, email_data)
        else:  # resolve
            score, feedback = grade_resolution(reply, email_data)

        score = round(float(score), 4)
        self._episode_rewards.append(score)
        self._step_count += 1

        self._done = self._step_count >= len(TASK_SEQUENCE)

        # Build next observation (or final)
        if not self._done:
            self._last_observation = self._build_observation()
        # If done, keep last obs but update step_num
        else:
            self._last_observation = Observation(
                email_id=email_data["id"],
                email_body=email_data["email"],
                task="resolve",       # last completed task
                step_num=self._step_count,
            )

        result = StepResult(
            observation=self._last_observation,
            reward=score,
            done=self._done,
            info={
                "task_completed": current_task,
                "feedback": feedback,
                "episode_rewards_so_far": list(self._episode_rewards),
                "cumulative_score": round(
                    sum(self._episode_rewards) / len(self._episode_rewards), 4
                ),
            }
        )
        return result.model_dump()

    async def state(self) -> dict:
        """
        Return the current state of the environment without advancing it.
        Required by OpenEnv spec.
        """
        if self._last_observation is None:
            raise RuntimeError("No active episode. Call reset() first.")

        result = StateResult(
            observation=self._last_observation,
            step_num=self._step_count,
            episode_rewards=list(self._episode_rewards),
            done=self._done,
        )
        return result.model_dump()

    async def close(self) -> None:
        """Clean up resources (no-op for this env)."""
        pass