import json
from pathlib import Path

from autoresearch.history import History, Trial
from autoresearch.runner import (
    TrialContext,
    build_eval_command,
    build_train_command,
    prune_adapters,
    run_trial,
)
from autoresearch.space import DEFAULT_CONFIG, clamp_config


def make_ctx(tmp_path: Path, **overrides) -> TrialContext:
    defaults = dict(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        train_manifest=tmp_path / "train.jsonl",
        dev_manifest=tmp_path / "dev.jsonl",
        base_model_file=tmp_path / "manifest.txt",
    )
    defaults.update(overrides)
    return TrialContext(**defaults)


def test_train_command_carries_every_searched_knob(tmp_path: Path):
    config = clamp_config({**DEFAULT_CONFIG, "lora_r": 64, "lora_alpha_mult": 4, "spec_augment": "strong"})
    ctx = make_ctx(tmp_path, num_gpus=4)
    cmd = build_train_command(config, ctx, ctx.trial_dir(1))

    assert cmd[:4] == ["uv", "run", "accelerate", "launch"]
    assert "--num_processes" in cmd and cmd[cmd.index("--num_processes") + 1] == "4"
    assert cmd[cmd.index("--lora-alpha") + 1] == "256"
    assert cmd[cmd.index("--target-modules") + 1] == "q_proj,k_proj,v_proj,out_proj"
    assert float(cmd[cmd.index("--mask-time-prob") + 1]) > 0
    assert cmd[cmd.index("--save-strategy") + 1] == "no"


def test_single_gpu_uses_plain_python(tmp_path: Path):
    cmd = build_train_command(clamp_config(DEFAULT_CONFIG), make_ctx(tmp_path), tmp_path / "t")
    assert cmd[:4] == ["uv", "run", "python", "scripts/train_whisper_lora.py"]


def test_eval_command_adds_the_prompt_only_when_the_config_asks_for_it(tmp_path: Path):
    ctx = make_ctx(tmp_path, prompt_text="用語A、用語B")
    without = build_eval_command(clamp_config({**DEFAULT_CONFIG, "prompt_terms": 0}), ctx, tmp_path)
    with_prompt = build_eval_command(clamp_config({**DEFAULT_CONFIG, "prompt_terms": 1}), ctx, tmp_path)
    assert "--prompt-text" not in without
    assert with_prompt[with_prompt.index("--prompt-text") + 1] == "用語A、用語B"


def test_mock_trial_scores_without_touching_a_gpu(tmp_path: Path):
    ctx = make_ctx(tmp_path, mock=True)
    trial = run_trial(0, clamp_config(DEFAULT_CONFIG), "baseline", None, ctx, budget_sec=60)
    assert trial.ok and 0.0 < trial.score < 1.0
    saved = json.loads((ctx.trial_dir(0) / "config.json").read_text(encoding="utf-8"))
    assert saved["lora_r"] == DEFAULT_CONFIG["lora_r"]


def test_prune_keeps_only_the_best_adapters(tmp_path: Path):
    ctx = make_ctx(tmp_path, keep_adapters=1)
    history = History.load_or_create(tmp_path / "state")
    for index, score in enumerate([0.5, 0.2, 0.4]):
        adapter = ctx.trial_dir(index) / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "weights.bin").write_text("x", encoding="utf-8")
        history.append(
            Trial(index=index, config=clamp_config(DEFAULT_CONFIG), score=score, seconds=1.0)
        )
    prune_adapters(history, ctx)

    assert (ctx.trial_dir(1) / "adapter").exists()
    assert not (ctx.trial_dir(0) / "adapter").exists()
    assert not (ctx.trial_dir(2) / "adapter").exists()
