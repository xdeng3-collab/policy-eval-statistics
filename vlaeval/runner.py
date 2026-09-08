"""Running a protocol against a policy."""

from __future__ import annotations

import time

import numpy as np

from .protocol import Protocol
from .results import EpisodeResult, EvalResult


def run(policy, env, protocol: Protocol) -> EvalResult:
    """Run every episode in ``protocol`` and collect the results.

    The policy is reset with the episode's own seed, so a stochastic policy is
    stochastic in the same way every time the protocol is run. Without that, the
    protocol fixes the environment but not the agent, and a rerun is not a
    rerun.
    """
    episodes: list[EpisodeResult] = []
    total_steps = 0
    started = time.perf_counter()

    for spec in protocol.episodes:
        observation = env.reset(spec.seed, spec.options)
        policy.reset(spec.seed)

        success = False
        steps = 0
        for steps in range(1, protocol.max_steps + 1):
            action = policy.act(observation)
            observation, terminated, _ = env.step(action)
            if env.error <= protocol.success_threshold:
                success = True
                break
            if terminated:
                break

        total_steps += steps
        episodes.append(
            EpisodeResult(
                episode_id=spec.episode_id,
                success=success,
                steps=steps,
                failure_mode=None if success else env.classify_failure(steps >= protocol.max_steps),
                final_error=env.error,
            )
        )

    elapsed = time.perf_counter() - started
    return EvalResult(
        policy=getattr(policy, "name", type(policy).__name__),
        protocol_name=protocol.name,
        protocol_fingerprint=protocol.fingerprint(),
        episodes=tuple(episodes),
        seconds_per_step=elapsed / total_steps if total_steps else 0.0,
    )
