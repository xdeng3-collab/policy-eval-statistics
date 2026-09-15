# vla-eval-harness

An evaluation harness for robot policies, built around the parts that usually go
missing: a protocol that makes two results comparable, intervals on every
success rate, paired significance testing, and a failure taxonomy.

Runs end to end with numpy alone — no simulator required.

```bash
pip install -r requirements.txt
python -m pytest tests -q     # 19 tests
python cli.py evaluate        # three baselines under one protocol
python cli.py power           # how many episodes a difference actually needs
python cli.py mismatch        # what comparing across protocols does
```

## The problem

Policy comparisons in this area are usually two numbers. Two numbers produced
under different initial-state distributions, step limits, or success thresholds
cannot be subtracted — and nothing in the numbers says they were produced
differently.

```
$ python cli.py mismatch
  greedy   under max_steps=400: 42.5%  [c99de503cd097ed8]
  avoidant under max_steps=60:  72.5%  [63fecf5606a5e69e]

  naive reading: greedy is -30.0% better

  compare() refuses: protocols differ (...); these results are not comparable
```

A `Protocol` fixes the episode seeds, the step limit, and the success criterion
up front, and hashes them. Every result carries that fingerprint, so an
accidental cross-protocol comparison is an exception instead of a number.

## Intervals, and how little 50 episodes buys

A success rate without an interval is not a measurement:

| Result | Wilson 95% interval |
|---|---|
| 36 / 50 = 72% | **[58.3%, 82.5%]** |
| 0 / 30 = 0% | [0%, 11.4%] |
| 30 / 30 = 100% | [88.6%, 100%] |

Wilson rather than the textbook normal approximation, which produces intervals
running past 0 and 1 and collapses to zero width at exactly the extremes a
policy evaluation lives at.

Comparisons are **paired**, because both policies ran the same seeded episodes.
The question is not "are these two rates different" but "on the episodes where
they disagreed, did one win more often" — an exact McNemar test, which has far
more power than treating the samples as independent. Exact rather than
chi-squared, which is unreliable below ~25 discordant pairs, and 50 episodes
routinely produce fewer.

It is sobering:

```
$ python cli.py power
  episodes needed to resolve a difference at 80% power, 50% baseline:
     2%    9752 episodes
     5%    1565 episodes
    10%     388 episodes
    20%      93 episodes
```

**Winning 12 episodes and losing 4 — which looks decisive — is p = 0.077.** At
the 50 episodes typically reported, a gap under roughly 20 points is not
resolvable, and reporting it as an improvement is reporting noise. That is a
pytest case, not a remark.

## The taxonomy is the half that says what to fix

```
$ python cli.py evaluate
  random:   0.0% [ 0.0%,  7.1%]   failures: collision 28%, left_workspace 48%, timeout 24%
  greedy:  38.0% [25.9%, 51.8%]   failures: collision 62%
  avoidant: 68.0% [54.2%, 79.2%]  failures: collision 30%, timeout 2%

  greedy vs random:   +38.0% -> greedy wins (p=0.000, 19 discordant)
  avoidant vs greedy: +30.0% -> avoidant wins (p=0.000, 15 discordant)
```

Greedy's failures are *almost entirely collisions*, and the avoidant policy is
better precisely there — 62% down to 30%. A pair of success rates would not have
said that, and "68% vs 38%" gives no hint about what to build next.

Timeouts are split into `timeout_near_target` and `timeout_far_from_target` for
the same reason: one is a policy that is slow, the other is a policy that is
wrong, and they need different fixes.

## Why the toy environment

`ReachEnv` is a 2-D reaching task with a velocity limit, a drift field, and an
obstacle. Each element exists to produce a *distinguishable* failure mode rather
than an undifferentiated "didn't work".

It is there so the harness itself can be tested — protocol, runner, statistics,
taxonomy — without a simulator. **A harness whose own correctness depends on
MuJoCo being installed is a harness nobody verifies.** The suite checks that the
evaluation separates a random policy from a near-optimal one; if it cannot do
that, it will not separate two real checkpoints either.

## Wiring in real policies

`LeRobotEnv` and `LeRobotPolicy` are **seams, not integrations**. Both raise
`NotImplementedError` with the call they expect; neither has been run against a
real checkpoint, and nothing in this repository has ever loaded lerobot. What
they carry is the interface — signatures identical to the toy env and the toy
baselines — so that the statistics can be validated on a task whose ground truth
is known, and then run unchanged on gym-pusht or gym-aloha once someone fills
them in with an ACT, Diffusion Policy, SmolVLA, or pi0 checkpoint.

That split is the point of the repository rather than a gap in it: the part
worth trusting is the part that can be tested without a GPU, and it is tested.
But it would be dishonest to call this "LeRobot support", so it is not called
that.

One design note: **action chunking belongs in the policy adapter, not the
runner.** A chunking policy returns several future actions at once; buffering
them inside the adapter keeps the environment loop identical for chunking and
non-chunking policies, so the comparison is not contaminated by how each one was
driven. Temporal ensembling goes in the same place.

## Layout

```
vlaeval/
  protocol.py   seeded episodes, step limits, fingerprinting
  results.py    Wilson intervals, exact McNemar, failure taxonomy, power
  envs.py       ReachEnv (dependency-free), LeRobot seam (unimplemented)
  policies.py   random / greedy / avoidant baselines, LeRobot seam (unimplemented)
  runner.py     the episode loop
cli.py          evaluate / power / mismatch
tests/          19 tests, weighted toward the statistics
```

## License

MIT
