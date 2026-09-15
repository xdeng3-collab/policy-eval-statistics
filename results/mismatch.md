# mismatch

    command: python cli.py mismatch
    python:  Python 3.9.6
    numpy:   2.0.2
    date:    2026-09-15T03:54:44Z

```
  greedy   under max_steps=400: 42.5%  [c99de503cd097ed8]
  avoidant under max_steps=60:  72.5%  [63fecf5606a5e69e]

  naive reading: greedy is -30.0% better

  compare() refuses: protocols differ (reach-v0/c99de503cd097ed8 vs reach-v0/63fecf5606a5e69e); these results are not comparable

The two numbers were produced under different step limits, and nothing in
the numbers says so. Carrying the protocol fingerprint into the result is
what turns a silently wrong comparison into an exception.
```
