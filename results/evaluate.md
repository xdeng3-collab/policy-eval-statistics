# evaluate

    command: python cli.py evaluate
    python:  Python 3.9.6
    numpy:   2.0.2
    date:    2026-09-15T03:54:43Z

```
protocol reach-v0 [bf1aaf7d302f0966] 50 episodes, 120 steps

  random: 0.0% [0.0%, 7.1%] over 50 episodes
     steps when it worked: nan
     failures: collision 28%, left_workspace 48%, timeout_far_from_target 24%

  greedy: 38.0% [25.9%, 51.8%] over 50 episodes
     steps when it worked: 11
     failures: collision 62%

  avoidant: 68.0% [54.2%, 79.2%] over 50 episodes
     steps when it worked: 12
     failures: collision 30%, timeout_far_from_target 2%

paired comparisons (McNemar, exact):
  greedy vs random: +38.0% -> greedy wins (p=0.000, 19 discordant episodes)
  avoidant vs greedy: +30.0% -> avoidant wins (p=0.000, 15 discordant episodes)

The taxonomy is the part that says what to fix: greedy's failures are
almost all collisions, and avoidant is better precisely there. A pair of
success rates alone would not have said that.
```
