#!/usr/bin/env python3
"""Policy evaluation.

    python cli.py evaluate            run the baselines under one protocol
    python cli.py power               how many episodes a difference needs
    python cli.py mismatch            what comparing across protocols does
"""

from __future__ import annotations

import argparse

from vlaeval import (AvoidantPolicy, GreedyPolicy, RandomPolicy, ReachEnv,
                     compare, data_needed_for, run, standard_protocol)


def cmd_evaluate(args: argparse.Namespace) -> int:
    protocol = standard_protocol("reach-v0", n_episodes=args.episodes, max_steps=args.max_steps)
    print(f"protocol {protocol.name} [{protocol.fingerprint()}] "
          f"{protocol.n_episodes} episodes, {protocol.max_steps} steps\n")

    env = ReachEnv()
    results = {}
    for policy in (RandomPolicy(), GreedyPolicy(), AvoidantPolicy()):
        result = run(policy, env, protocol)
        results[policy.name] = result
        taxonomy = ", ".join(f"{k} {v:.0%}" for k, v in result.failure_taxonomy().items())
        print(f"  {result.summary()}")
        print(f"     steps when it worked: {result.mean_steps_on_success():.0f}")
        print(f"     failures: {taxonomy or 'none'}\n")

    print("paired comparisons (McNemar, exact):")
    for a, b in (("greedy", "random"), ("avoidant", "greedy")):
        comparison = compare(results[a], results[b])
        print(f"  {a} vs {b}: {comparison.difference:+.1%} -> {comparison.verdict()}")

    print("\nThe taxonomy is the part that says what to fix: greedy's failures are\n"
          "almost all collisions, and avoidant is better precisely there. A pair of\n"
          "success rates alone would not have said that.")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    print("episodes needed to resolve a difference at 80% power, 50% baseline:\n")
    for difference in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        print(f"  {difference:>5.0%}   {data_needed_for(difference):>6d} episodes")
    print("\nRun this before the evaluation, not after. At the 50 episodes that get\n"
          "reported by default, anything under about a 20 point gap is not resolvable,\n"
          "and reporting it as an improvement is reporting noise.")
    return 0


def cmd_mismatch(args: argparse.Namespace) -> int:
    env = ReachEnv()
    generous = standard_protocol("reach-v0", n_episodes=40, max_steps=400)
    strict = standard_protocol("reach-v0", n_episodes=40, max_steps=60)

    a = run(GreedyPolicy(), env, generous)
    b = run(AvoidantPolicy(), env, strict)
    print(f"  greedy   under max_steps=400: {a.success_rate:.1%}  [{a.protocol_fingerprint}]")
    print(f"  avoidant under max_steps=60:  {b.success_rate:.1%}  [{b.protocol_fingerprint}]")
    print(f"\n  naive reading: greedy is {a.success_rate - b.success_rate:+.1%} better\n")
    try:
        compare(a, b)
    except ValueError as error:
        print(f"  compare() refuses: {error}")
    print("\nThe two numbers were produced under different step limits, and nothing in\n"
          "the numbers says so. Carrying the protocol fingerprint into the result is\n"
          "what turns a silently wrong comparison into an exception.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    sub.add_parser("power").set_defaults(func=cmd_power)
    sub.add_parser("mismatch").set_defaults(func=cmd_mismatch)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
