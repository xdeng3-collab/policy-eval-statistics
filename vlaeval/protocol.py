"""The evaluation protocol.

Most reported policy comparisons are not comparisons. Two numbers produced under
different initial-state distributions, different step limits, or different
success definitions cannot be subtracted, and nothing in the numbers themselves
says they were produced differently.

So the protocol is an object. It fixes the episode seeds, the step limit, and
the success criterion up front, and every policy is run against the same one. A
result carries the protocol that produced it, and comparing results from
different protocols raises instead of quietly returning a difference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodeSpec:
    """One evaluation episode: a seed and whatever the environment needs."""

    episode_id: int
    seed: int
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episode_id < 0:
            raise ValueError("episode_id must be non-negative")


@dataclass(frozen=True)
class Protocol:
    """A fixed evaluation setup. Two results are comparable iff these match."""

    name: str
    episodes: tuple[EpisodeSpec, ...]
    max_steps: int
    success_threshold: float = 0.05   # task-specific; e.g. distance to target
    instruction: str | None = None    # for language-conditioned policies

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("a protocol needs at least one episode")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        ids = [episode.episode_id for episode in self.episodes]
        if len(set(ids)) != len(ids):
            raise ValueError("episode ids must be unique")

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def fingerprint(self) -> str:
        """Stable hash of everything that makes results comparable.

        Results carry this. Two results with different fingerprints were
        produced under different conditions and must not be compared, however
        similar the numbers look.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "max_steps": self.max_steps,
                "success_threshold": self.success_threshold,
                "instruction": self.instruction,
                "episodes": [
                    {"id": e.episode_id, "seed": e.seed, "options": e.options}
                    for e in self.episodes
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def standard_protocol(
    name: str,
    *,
    n_episodes: int = 50,
    base_seed: int = 20260913,
    max_steps: int = 200,
    success_threshold: float = 0.05,
    instruction: str | None = None,
) -> Protocol:
    """Build a protocol whose seeds derive deterministically from ``base_seed``.

    Derived rather than random: re-running the evaluation months later, on
    another machine, must produce the same initial conditions, or the comparison
    against the earlier number is not one.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    episodes = tuple(
        EpisodeSpec(episode_id=i, seed=base_seed * 1000 + i) for i in range(n_episodes)
    )
    return Protocol(
        name=name, episodes=episodes, max_steps=max_steps,
        success_threshold=success_threshold, instruction=instruction,
    )
