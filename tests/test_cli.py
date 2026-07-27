"""CLI smoke tests (no GPU, no vllm — the CLI imports vllm lazily).

Building the parser eagerly imports every eval's cli/generate modules, so
this catches dangling imports across the whole package. The parse checks pin
the post-steering-removal flag surface: --steering and friends are gone, and
a representative unsteered invocation parses for each eval.
"""
from __future__ import annotations

import pytest

from honesty_evals.cli import build_parser

SAMPLING = [
    "--max-new-tokens", "256",
    "--temperature", "0.7",
    "--top-p", "0.95",
    "--seed", "42",
]
SERVER = ["--server-url", "http://localhost:8192"]


@pytest.fixture(scope="module")
def parser():
    return build_parser()


def test_build_parser(parser):
    assert parser is not None


@pytest.mark.parametrize(
    "argv",
    [
        ["mask", "generate", "--model", "m", "--run-name", "r",
         "--archetypes", "all", *SERVER, *SAMPLING],
        ["generalized", "generate", "--model", "m", "--run-name", "r",
         "--system-prompt-variant", "dishonest", *SERVER, *SAMPLING],
        ["em", "generate", "--model", "m", "--run-name", "r",
         "--conditions", "base", "--question-set", "first_plot_questions",
         "--n-per-question", "3", *SERVER, *SAMPLING],
        ["among-us", "run", "--model", "m", "--run-name", "r",
         "--num-games", "1", "--game-config", "configs/evals/among_us.yaml",
         *SERVER, *SAMPLING],
        ["auditbench", "run", "--run-name", "r", "--quirk-name", "q",
         *SERVER],
        ["mask", "metrics", "--run-dir", "results/mask/some_run"],
    ],
)
def test_unsteered_invocations_parse(parser, argv):
    args = parser.parse_args(argv)
    assert args.func is not None


@pytest.mark.parametrize(
    "argv",
    [
        ["mask", "generate", "--model", "m", "--run-name", "r",
         "--archetypes", "all", "--steering", "none", *SERVER, *SAMPLING],
        ["serve", "--model", "m", "--tp", "1", "--port", "8192", "--no-steering"],
        ["mask", "metrics", "--run-dir", "d", "--baseline-run-dir", "b"],
        ["auditbench", "run", "--run-name", "r", "--quirk-name", "q",
         "--technique", "sttp", *SERVER],
        ["resolve", "--model", "m", "--steering", "swfc"],
    ],
)
def test_steering_flags_rejected(parser, argv):
    with pytest.raises(SystemExit):
        parser.parse_args(argv)
