"""Policies.

Three baselines that run anywhere, and an adapter for a real learned policy.

The baselines are not filler. A harness needs a known-bad and a known-good
policy to test that its statistics say the right thing: if the evaluation cannot
separate a random policy from a near-optimal one, it will not separate two real
checkpoints either.
"""

from __future__ import annotations

from typing import Any, Protocol as TypingProtocol

import numpy as np


class Policy(TypingProtocol):
    name: str

    def reset(self, seed: int) -> None: ...
    def act(self, observation: np.ndarray) -> np.ndarray: ...


class RandomPolicy:
    """The floor. Anything that does not beat this has learned nothing."""

    name = "random"

    def __init__(self, scale: float = 0.08) -> None:
        self.scale = scale

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._rng.normal(0, self.scale, size=2)


class GreedyPolicy:
    """Head straight for the target, ignoring the obstacle.

    The interesting baseline: it succeeds whenever nothing is in the way, so its
    failures are almost entirely collisions. A policy that beats it must be
    doing avoidance specifically, and the failure taxonomy shows whether it is.
    """

    name = "greedy"

    def __init__(self, gain: float = 0.5, noise: float = 0.0) -> None:
        self.gain = gain
        self.noise = noise

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        position, target = observation[:2], observation[2:4]
        action = (target - position) * self.gain
        if self.noise:
            action = action + self._rng.normal(0, self.noise, size=2)
        return action


class AvoidantPolicy:
    """Go to the target, but push away from the obstacle when close to it."""

    name = "avoidant"

    def __init__(self, gain: float = 0.5, repulsion: float = 0.05, radius: float = 0.25) -> None:
        self.gain = gain
        self.repulsion = repulsion
        self.radius = radius

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        position, target, obstacle = observation[:2], observation[2:4], observation[4:6]
        action = (target - position) * self.gain

        away = position - obstacle
        distance = float(np.linalg.norm(away))
        if distance < self.radius and distance > 1e-9:
            action = action + (away / distance) * self.repulsion * (self.radius - distance) / self.radius * 10
        return action


class LeRobotPolicy:
    """Adapter for a LeRobot checkpoint (ACT, Diffusion Policy, SmolVLA, pi0).

    Action chunking belongs here rather than in the runner: a chunking policy
    returns several future actions at once and the adapter replays them, so the
    environment loop stays identical for chunking and non-chunking policies and
    the comparison is not contaminated by how they are driven.
    """

    def __init__(self, checkpoint: str, *, chunk_size: int = 1, instruction: str | None = None) -> None:
        self.name = f"lerobot:{checkpoint}"
        raise NotImplementedError(
            "Load with lerobot.policies; call select_action(observation) and, for a "
            "chunking policy, buffer the returned chunk here so the runner's loop "
            "is unchanged. Temporal ensembling also belongs in this adapter."
        )
