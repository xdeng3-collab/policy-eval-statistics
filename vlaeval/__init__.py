"""Evaluation harness for robot policies: fixed protocols, intervals, paired tests."""

from .protocol import EpisodeSpec, Protocol, standard_protocol
from .results import (EpisodeResult, EvalResult, Comparison, compare,
                      wilson_interval, exact_mcnemar_p, data_needed_for)
from .envs import Env, ReachEnv, LeRobotEnv
from .policies import Policy, RandomPolicy, GreedyPolicy, AvoidantPolicy, LeRobotPolicy
from .runner import run

__all__ = ["EpisodeSpec", "Protocol", "standard_protocol", "EpisodeResult", "EvalResult",
           "Comparison", "compare", "wilson_interval", "exact_mcnemar_p", "data_needed_for",
           "Env", "ReachEnv", "LeRobotEnv", "Policy", "RandomPolicy", "GreedyPolicy",
           "AvoidantPolicy", "LeRobotPolicy", "run"]
