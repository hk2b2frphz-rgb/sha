import random
from pathlib import Path

from autoresearch.history import History, Trial
import pytest

from autoresearch.proposer import GemmaProposer, ProposerError, ProposerPipeline
from autoresearch.space import DEFAULT_CONFIG, clamp_config, config_key


class StubGemma(GemmaProposer):
    """GemmaProposer with the subprocess replaced by canned output."""

    def __init__(self, batches):
        super().__init__(script=Path("unused"), out_dir=Path("unused"), model="stub")
        self.batches = list(batches)
        self.calls = 0

    def propose(self, history, avoid=None):  # noqa: D102 - see base class
        self.calls += 1
        batch = self.batches.pop(0) if self.batches else []
        self.last_error = None if batch else "stub returned nothing"
        return [clamp_config(c) for c in batch]


def append(history: History, config: dict, score: float | None = 0.3) -> None:
    history.append(
        Trial(index=history.next_index(), config=clamp_config(config), score=score, seconds=10.0)
    )


def test_first_trial_is_the_baseline_then_random_exploration(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    pipeline = ProposerPipeline(random.Random(1), gemma=None, n_random=3)

    first = pipeline.propose(history)
    assert first.source == "baseline"
    assert config_key(first.config) == config_key(DEFAULT_CONFIG)

    append(history, first.config)
    sources = []
    for _ in range(3):
        proposal = pipeline.propose(history)
        sources.append(proposal.source)
        append(history, proposal.config)
    assert sources == ["random", "random", "random"]


def test_gemma_proposals_are_used_and_queued(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    gemma = StubGemma([[{**DEFAULT_CONFIG, "lora_r": 64}, {**DEFAULT_CONFIG, "lora_r": 128}]])
    pipeline = ProposerPipeline(random.Random(2), gemma=gemma, n_random=0)
    append(history, DEFAULT_CONFIG, 0.4)

    first = pipeline.propose(history)
    assert first.source == "gemma" and first.config["lora_r"] == 64
    append(history, first.config, 0.35)

    # The second proposal comes from the queue, without asking Gemma again.
    second = pipeline.propose(history)
    assert second.source == "gemma" and second.config["lora_r"] == 128
    assert gemma.calls == 1


def test_a_failing_gemma_stops_the_run_instead_of_falling_back(tmp_path: Path):
    """No silent degradation: a broken proposer must be visible, not absorbed."""
    history = History.load_or_create(tmp_path)
    gemma = StubGemma([[], [], []])
    pipeline = ProposerPipeline(random.Random(3), gemma=gemma, n_random=0, gemma_retries=2)
    append(history, DEFAULT_CONFIG, 0.4)

    with pytest.raises(ProposerError) as excinfo:
        pipeline.propose(history)
    assert gemma.calls == 3  # the initial attempt plus two retries
    assert "no fallback" in str(excinfo.value)
    assert "--no-gemma" in str(excinfo.value)


def test_duplicate_only_proposals_are_retried_then_raise(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    duplicate = {**DEFAULT_CONFIG}
    gemma = StubGemma([[duplicate], [duplicate], [duplicate]])
    pipeline = ProposerPipeline(random.Random(4), gemma=gemma, n_random=0, gemma_retries=2)
    append(history, duplicate, 0.4)

    with pytest.raises(ProposerError) as excinfo:
        pipeline.propose(history)
    assert "already evaluated" in str(excinfo.value)


def test_a_fresh_proposal_after_a_duplicate_is_accepted(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    duplicate = {**DEFAULT_CONFIG}
    fresh = {**DEFAULT_CONFIG, "lora_r": 128}
    pipeline = ProposerPipeline(
        random.Random(5), gemma=StubGemma([[duplicate], [fresh]]), n_random=0, gemma_retries=2
    )
    append(history, duplicate, 0.4)

    proposal = pipeline.propose(history)
    assert proposal.source == "gemma"
    assert config_key(proposal.config) not in history.tried_keys()


def test_evolutionary_mode_is_explicit_not_a_fallback(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    pipeline = ProposerPipeline(random.Random(6), gemma=None, n_random=0)
    append(history, DEFAULT_CONFIG, 0.4)

    proposal = pipeline.propose(history)
    assert proposal.source in ("mutation", "crossover")
    assert proposal.parent == 0
