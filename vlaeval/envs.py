"""Environments.

`ReachEnv` is a deliberately tiny 2-D reaching task with no dependencies. It
exists so the harness can be tested end to end -- protocol, runner, statistics,
failure taxonomy -- without pulling in a simulator. A harness whose own
correctness depends on MuJoCo being installed is a harness nobody verifies.

`LeRobotEnv` is the adapter for real work, and is a stub until a gym environment
is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol as TypingProtocol

import numpy as np


class Env(TypingProtocol):
    """Minimal environment interface the runner depends on."""

    def reset(self, seed: int, options: dict[str, Any]) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]: ...
    @property
    def error(self) -> float: ...


@dataclass
class ReachEnv:
    """Move a point to a target under velocity limits, drift, and an obstacle.

    Each element is there to produce a distinguishable failure mode rather than
    a single undifferentiated "didn't work":

      * a velocity limit, so overshooting is possible;
      * a drift field, so an open-loop policy diverges;
      * an obstacle, so there is a way to fail that is not just being slow.
    """

    action_limit: float = 0.08
    drift: float = 0.004
    obstacle_radius: float = 0.12
    workspace: float = 1.0

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self._rng = rng
        self.position = rng.uniform(-0.8, 0.8, size=2)
        self.target = rng.uniform(-0.8, 0.8, size=2)
        # Keep the task non-trivial: a target already under the tolerance would
        # be a free success and would flatter every policy equally.
        while np.linalg.norm(self.target - self.position) < 0.4:
            self.target = rng.uniform(-0.8, 0.8, size=2)
        self.obstacle = (self.position + self.target) / 2 + rng.normal(0, 0.1, size=2)
        self._drift_direction = rng.normal(size=2)
        self._drift_direction /= np.linalg.norm(self._drift_direction)
        self.collided = False
        self.left_workspace = False
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate([self.position, self.target, self.obstacle])

    @property
    def error(self) -> float:
        return float(np.linalg.norm(self.target - self.position))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(2)
        norm = np.linalg.norm(action)
        if norm > self.action_limit:
            action = action / norm * self.action_limit

        self.position = self.position + action + self._drift_direction * self.drift

        if np.linalg.norm(self.position - self.obstacle) < self.obstacle_radius:
            self.collided = True
        if np.max(np.abs(self.position)) > self.workspace:
            self.left_workspace = True

        terminated = self.collided or self.left_workspace
        return self.observation(), terminated, {
            "collided": self.collided,
            "left_workspace": self.left_workspace,
        }

    def classify_failure(self, timed_out: bool) -> str:
        """Name the failure. An unlabelled failure cannot be prioritised."""
        if self.collided:
            return "collision"
        if self.left_workspace:
            return "left_workspace"
        if timed_out and self.error < 0.15:
            return "timeout_near_target"   # slow, not wrong: a different fix
        return "timeout_far_from_target"


class LeRobotEnv:
    """Adapter for a LeRobot / gymnasium environment.

    Keeping the signature identical to ReachEnv is what lets the harness and its
    statistics be tested against the toy task and then run unchanged on
    gym-pusht or gym-aloha.
    """

    def __init__(self, env_id: str) -> None:
        raise NotImplementedError(
            f"Wire {env_id} in via gymnasium.make: reset(seed=...) -> observation, "
            "step(action) -> (observation, terminated, info). The failure "
            "taxonomy is task-specific and belongs in classify_failure."
        )
