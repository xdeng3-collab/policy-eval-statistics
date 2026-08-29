"""Results, intervals, and paired comparison.

A success rate without an interval is not a measurement. At N=50, a policy that
scores 72% has a 95% interval of roughly 58% to 83%; reporting it against
another policy's 68% as a "4 point improvement" is reporting noise.

Comparisons are paired. Both policies ran the same seeded episodes, so the
question is not "are these two rates different" but "on the episodes where they
disagreed, did one win more often" -- which is McNemar's test, and which has far
more power than treating the two samples as independent.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    success: bool
    steps: int
    failure_mode: str | None = None   # None iff success
    final_error: float | None = None

    def __post_init__(self) -> None:
        if self.success and self.failure_mode is not None:
            raise ValueError("a successful episode cannot have a failure mode")
        if not self.success and self.failure_mode is None:
            raise ValueError("a failed episode must say how it failed")


@dataclass(frozen=True)
class EvalResult:
    policy: str
    protocol_name: str
    protocol_fingerprint: str
    episodes: tuple[EpisodeResult, ...]
    seconds_per_step: float = 0.0

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def interval(self, confidence: float = 0.95) -> tuple[float, float]:
        return wilson_interval(self.successes, self.n, confidence)

    def failure_taxonomy(self) -> dict[str, float]:
        """Fraction of *all* episodes ending in each failure mode.

        A success rate says how often a policy worked. This says what to fix,
        and it is the half that usually goes unreported.
        """
        counts = Counter(e.failure_mode for e in self.episodes if not e.success)
        return {mode: count / self.n for mode, count in sorted(counts.items())}

    def mean_steps_on_success(self) -> float:
        steps = [e.steps for e in self.episodes if e.success]
        return sum(steps) / len(steps) if steps else float("nan")

    def summary(self) -> str:
        low, high = self.interval()
        return (
            f"{self.policy}: {self.success_rate:.1%} "
            f"[{low:.1%}, {high:.1%}] over {self.n} episodes"
        )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Not the textbook normal approximation: that one produces intervals extending
    past 0 or 1, and collapses to zero width at 0% and 100% -- exactly the
    regimes a policy evaluation spends its time in.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")

    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(confidence)
    if z is None:
        raise ValueError("confidence must be one of 0.90, 0.95, 0.99")

    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    a_rate: float
    b_rate: float
    both_succeeded: int
    only_a: int
    only_b: int
    neither: int
    p_value: float

    @property
    def difference(self) -> float:
        return self.a_rate - self.b_rate

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    def verdict(self, alpha: float = 0.05) -> str:
        if self.discordant == 0:
            return "identical on every episode"
        if self.p_value > alpha:
            return (
                f"not distinguishable at N={self.both_succeeded + self.discordant + self.neither} "
                f"(p={self.p_value:.3f}); {self.discordant} episodes disagreed"
            )
        winner = self.a if self.only_a > self.only_b else self.b
        return f"{winner} wins (p={self.p_value:.3f}, {self.discordant} discordant episodes)"


def compare(a: EvalResult, b: EvalResult) -> Comparison:
    """Paired comparison of two policies on the same protocol.

    Refuses to compare results from different protocols. That refusal is the
    point: an accidental comparison across setups is the most common way a
    policy improvement gets reported that does not exist.
    """
    if a.protocol_fingerprint != b.protocol_fingerprint:
        raise ValueError(
            f"protocols differ ({a.protocol_name}/{a.protocol_fingerprint} vs "
            f"{b.protocol_name}/{b.protocol_fingerprint}); these results are not comparable"
        )

    a_by_id = {e.episode_id: e.success for e in a.episodes}
    b_by_id = {e.episode_id: e.success for e in b.episodes}
    if set(a_by_id) != set(b_by_id):
        raise ValueError("the two results cover different episodes")

    both = only_a = only_b = neither = 0
    for episode_id, a_ok in a_by_id.items():
        b_ok = b_by_id[episode_id]
        if a_ok and b_ok:
            both += 1
        elif a_ok:
            only_a += 1
        elif b_ok:
            only_b += 1
        else:
            neither += 1

    return Comparison(
        a=a.policy, b=b.policy, a_rate=a.success_rate, b_rate=b.success_rate,
        both_succeeded=both, only_a=only_a, only_b=only_b, neither=neither,
        p_value=exact_mcnemar_p(only_a, only_b),
    )


def exact_mcnemar_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test: a sign test on the discordant pairs.

    Exact rather than the chi-squared approximation, which is unreliable below
    about 25 discordant pairs -- and an evaluation of 50 episodes routinely has
    fewer than that.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0

    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def data_needed_for(difference: float, baseline: float = 0.5, power: float = 0.8) -> int:
    """Rough episode count needed to resolve a difference of ``difference``.

    Useful before running anything: at a 50% baseline, telling 5 points apart
    needs hundreds of episodes. Knowing that in advance stops a 50-episode run
    from being over-read.
    """
    if not 0 < difference < 1:
        raise ValueError("difference must lie in (0, 1)")
    z_alpha, z_beta = 1.96, {0.8: 0.8416, 0.9: 1.2816}.get(power, 0.8416)
    p1, p2 = baseline, min(0.999, baseline + difference)
    pooled = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pooled * (1 - pooled))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (difference ** 2))
