# power

    command: python cli.py power
    python:  Python 3.9.6
    numpy:   2.0.2
    date:    2026-09-15T03:54:43Z

```
episodes needed to resolve a difference at 80% power, 50% baseline:

     2%     9807 episodes
     5%     1565 episodes
    10%      388 episodes
    15%      170 episodes
    20%       93 episodes
    30%       39 episodes

Run this before the evaluation, not after. At the 50 episodes that get
reported by default, anything under about a 20 point gap is not resolvable,
and reporting it as an improvement is reporting noise.
```

Closed-form, seed-free, machine-independent: these are properties of the
binomial, not of anything that was run. CI asserts the published values rather
than checking that the command exits zero.
