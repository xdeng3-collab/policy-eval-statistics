"""The statistics decide what gets believed, so they are tested against known
values rather than against themselves."""

import math

import pytest

from vlaeval.results import data_needed_for, exact_mcnemar_p, wilson_interval


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    # The normal approximation fails exactly here, producing intervals that
    # extend past 1 or collapse to zero width. A policy evaluation spends much
    # of its life at 0% and 100%.
    low, high = wilson_interval(0, 30)
    assert low == 0.0 and 0 < high < 0.2
    low, high = wilson_interval(30, 30)
    assert high == 1.0 and 0.8 < low < 1.0


def test_a_typical_result_is_much_less_precise_than_it_looks():
    low, high = wilson_interval(36, 50)   # 72%
    assert low == pytest.approx(0.583, abs=0.01)
    assert high == pytest.approx(0.825, abs=0.01)
    assert high - low > 0.2               # a 24-point-wide interval


def test_interval_narrows_as_episodes_grow():
    widths = [wilson_interval(int(0.7 * n), n) for n in (25, 100, 400, 1600)]
    spans = [high - low for low, high in widths]
    assert all(a > b for a, b in zip(spans, spans[1:]))


def test_mcnemar_matches_the_binomial_by_hand():
    # 10 discordant pairs, all favouring one side: two-sided p = 2 * 2^-10.
    assert exact_mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 5) == pytest.approx(1.0)


def test_a_convincing_looking_win_is_not_significant_at_fifty_episodes():
    # 12 episodes won, 4 lost. Looks decisive; is not.
    assert exact_mcnemar_p(12, 4) > 0.05


def test_small_differences_need_implausible_episode_counts():
    assert data_needed_for(0.05) > 1000
    assert data_needed_for(0.20) < 150
    assert data_needed_for(0.05) > data_needed_for(0.10) > data_needed_for(0.20)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=0.75)
