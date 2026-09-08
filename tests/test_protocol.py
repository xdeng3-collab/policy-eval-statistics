import pytest

from vlaeval import (AvoidantPolicy, GreedyPolicy, RandomPolicy, ReachEnv,
                     compare, run, standard_protocol)
from vlaeval.results import EpisodeResult


def test_the_same_protocol_gives_the_same_fingerprint():
    a = standard_protocol("reach", n_episodes=10, base_seed=1)
    b = standard_protocol("reach", n_episodes=10, base_seed=1)
    assert a.fingerprint() == b.fingerprint()


@pytest.mark.parametrize("change", [
    {"max_steps": 999}, {"n_episodes": 11}, {"base_seed": 2}, {"success_threshold": 0.1},
])
def test_any_change_that_matters_changes_the_fingerprint(change):
    base = dict(n_episodes=10, base_seed=1, max_steps=100, success_threshold=0.05)
    a = standard_protocol("reach", **base)
    b = standard_protocol("reach", **{**base, **change})
    assert a.fingerprint() != b.fingerprint()


def test_a_rerun_reproduces_the_result_exactly():
    protocol = standard_protocol("reach", n_episodes=20, max_steps=100)
    first = run(RandomPolicy(), ReachEnv(), protocol)
    second = run(RandomPolicy(), ReachEnv(), protocol)
    assert first.episodes == second.episodes   # seeded policy and environment


def test_comparing_across_protocols_raises_instead_of_returning_a_number():
    env = ReachEnv()
    generous = run(GreedyPolicy(), env, standard_protocol("reach", n_episodes=20, max_steps=400))
    strict = run(GreedyPolicy(), env, standard_protocol("reach", n_episodes=20, max_steps=40))
    with pytest.raises(ValueError, match="not comparable"):
        compare(generous, strict)


def test_the_harness_separates_a_useless_policy_from_a_working_one():
    # If the evaluation cannot tell random from near-optimal, it will not tell
    # two real checkpoints apart either.
    protocol = standard_protocol("reach", n_episodes=50, max_steps=120)
    env = ReachEnv()
    random_result = run(RandomPolicy(), env, protocol)
    avoidant_result = run(AvoidantPolicy(), env, protocol)
    assert random_result.success_rate < 0.1
    assert avoidant_result.success_rate > 0.5
    assert compare(avoidant_result, random_result).p_value < 0.01


def test_avoidance_shows_up_in_the_taxonomy_not_just_the_rate():
    protocol = standard_protocol("reach", n_episodes=50, max_steps=120)
    env = ReachEnv()
    greedy = run(GreedyPolicy(), env, protocol).failure_taxonomy()
    avoidant = run(AvoidantPolicy(), env, protocol).failure_taxonomy()
    # The whole point of the avoidant policy is fewer collisions specifically.
    assert avoidant.get("collision", 0.0) < greedy.get("collision", 0.0)


def test_a_failed_episode_must_say_how_it_failed():
    with pytest.raises(ValueError):
        EpisodeResult(episode_id=0, success=False, steps=10, failure_mode=None)
    with pytest.raises(ValueError):
        EpisodeResult(episode_id=0, success=True, steps=10, failure_mode="collision")


def test_protocols_reject_malformed_setups():
    with pytest.raises(ValueError):
        standard_protocol("reach", n_episodes=0)
    with pytest.raises(ValueError):
        standard_protocol("reach", n_episodes=5, max_steps=0)
