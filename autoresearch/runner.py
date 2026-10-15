"""Execution of a single trial: LoRA fine-tune -> dev decode -> score.

A trial never merges weights or converts to CTranslate2; the evaluator loads
the adapter on top of the base model directly. That keeps one iteration to a
train + decode, which is what makes tens of trials fit in a 20-hour budget.
The winning config is converted once, at the end of the run.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .history import (
    STAGE_FULL,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TIMEOUT,
    History,
    Trial,
    utcnow,
)
from .space import lora_alpha, spec_augment_probs, target_module_list


class CommandTimeout(RuntimeError):
    pass


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_sec: float | None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess, tee-ing output to log_path, killing the whole group on timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n$ {' '.join(cmd)}\n")
        log_fh.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            **popen_kwargs,
        )
        try:
            returncode = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            _terminate(process)
            raise CommandTimeout(f"timed out after {timeout_sec:.0f}s: {' '.join(cmd[:4])} ...")
    if returncode != 0:
        tail = tail_text(log_path, 40)
        raise RuntimeError(f"command failed (rc={returncode}): {' '.join(cmd[:4])} ...\n{tail}")


def _terminate(process: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=30)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass


def tail_text(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


@dataclass
class TrialContext:
    repo_root: Path
    work_dir: Path
    train_manifest: Path
    dev_manifest: Path
    base_model_file: Path
    base_model: str | None = None
    num_gpus: int = 1
    language: str = "ja"
    objective: str = "cer"
    term_weight: float = 0.5
    terms_file: Path | None = None
    prompt_text: str = ""
    eval_batch_size: int = 8
    seed: int = 42
    keep_adapters: int = 3
    early_stop_margin: float = 0.35
    early_stop_patience: int = 2
    early_stop_dev_limit: int = 24
    mock: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)

    def trial_dir(self, index: int) -> Path:
        return self.work_dir / "trials" / f"trial_{index:04d}"


def early_stop_threshold(history: History, ctx: TrialContext, stage: str | None = None) -> float:
    """dev CER の足切り水準。既存ベストがないうちは足切りしない (0 = 無効)。

    ベストからマージンを取るのは、学習中の評価が dev の一部だけを使う粗い推定
    だから。目的関数が cer_term のときスコアは CER 以上になるので、しきい値は
    自然に甘い側 (打ち切りにくい側) に倒れる。
    """
    if ctx.early_stop_margin <= 0:
        return 0.0
    # Only compare against trials from the same stage: a screening score comes
    # from a few minutes of training and is not comparable with a full run.
    best = history.best(stage)
    if best is None or best.score is None:
        return 0.0
    return float(best.score) * (1.0 + ctx.early_stop_margin)


def build_train_command(
    config: dict[str, Any],
    ctx: TrialContext,
    trial_dir: Path,
    early_stop_cer: float = 0.0,
    max_train_seconds: float = 0.0,
) -> list[str]:
    script = "scripts/train_whisper_lora.py"
    if ctx.num_gpus > 1:
        cmd = [
            "uv", "run", "accelerate", "launch",
            "--num_processes", str(ctx.num_gpus),
            "--mixed_precision", "fp16",
            script,
        ]
    else:
        cmd = ["uv", "run", "python", script]

    mask_time_prob, mask_feature_prob = spec_augment_probs(config)
    cmd += [
        "--manifest", str(ctx.train_manifest),
        "--out-dir", str(trial_dir),
        "--language", ctx.language,
        "--epochs", str(float(config["epochs"])),
        "--lr", str(float(config["lr"])),
        "--batch-size", str(int(config["batch_size"])),
        "--grad-accum", str(int(config["grad_accum"])),
        "--warmup-ratio", str(float(config["warmup_ratio"])),
        "--weight-decay", str(float(config["weight_decay"])),
        "--label-smoothing", str(float(config["label_smoothing"])),
        "--lr-scheduler", str(config["lr_scheduler"]),
        "--lora-r", str(int(config["lora_r"])),
        "--lora-alpha", str(lora_alpha(config)),
        "--lora-dropout", str(float(config["lora_dropout"])),
        "--target-modules", ",".join(target_module_list(config)),
        "--augment-copies", str(int(config["augment_copies"])),
        "--augment-speed", str(float(config["augment_speed"])),
        "--augment-noise-snr-db", str(float(config["augment_noise_snr_db"])),
        "--augment-gain-db", str(float(config["augment_gain_db"])),
        "--mask-time-prob", str(mask_time_prob),
        "--mask-feature-prob", str(mask_feature_prob),
        "--save-strategy", "no",
        "--seed", str(ctx.seed),
    ]
    if max_train_seconds > 0:
        cmd += ["--max-train-seconds", str(int(max_train_seconds))]
    if ctx.early_stop_margin > 0 or ctx.early_stop_patience > 0:
        cmd += [
            "--dev-manifest", str(ctx.dev_manifest),
            "--dev-limit", str(ctx.early_stop_dev_limit),
            "--early-stop-cer", str(early_stop_cer),
            "--early-stop-patience", str(ctx.early_stop_patience),
        ]
    if ctx.base_model:
        cmd += ["--base-model", ctx.base_model]
    else:
        cmd += ["--base-model-file", str(ctx.base_model_file)]
    return cmd


def build_eval_command(config: dict[str, Any], ctx: TrialContext, trial_dir: Path) -> list[str]:
    cmd = [
        "uv", "run", "python", "scripts/eval_whisper_hf.py",
        "--manifest", str(ctx.dev_manifest),
        "--adapter", str(trial_dir / "adapter"),
        "--out-dir", str(trial_dir / "eval"),
        "--language", ctx.language,
        "--batch-size", str(ctx.eval_batch_size),
        "--beam-size", str(int(config["beam_size"])),
        "--no-repeat-ngram-size", str(int(config["no_repeat_ngram_size"])),
        "--objective", ctx.objective,
        "--term-weight", str(ctx.term_weight),
    ]
    if ctx.base_model:
        cmd += ["--base-model", ctx.base_model]
    else:
        cmd += ["--base-model-file", str(ctx.base_model_file)]
    if ctx.terms_file is not None:
        cmd += ["--terms", str(ctx.terms_file)]
    if int(config["prompt_terms"]) == 1 and ctx.prompt_text:
        cmd += ["--prompt-text", ctx.prompt_text]
    return cmd


def read_train_progress(trial_dir: Path) -> dict[str, Any]:
    """train_progress.json (学習中の dev CER 推移) を試行の記録に取り込む。"""
    path = trial_dir / "train_progress.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        "early_stopped": bool(data.get("early_stopped")),
        "early_stop_reason": data.get("early_stop_reason") or None,
        "dev_cer_curve": data.get("dev_cer_curve") or [],
        "epochs_completed": data.get("epochs_completed"),
        "steps_completed": data.get("steps_completed"),
        "time_budget_hit": bool(data.get("time_budget_hit")),
    }


def mock_score(config: dict[str, Any]) -> dict[str, float]:
    """Deterministic surrogate objective used by --mock for wiring tests.

    Shaped so the optimum sits away from the baseline: the loop is expected to
    move lr toward ~2e-4, add a little augmentation, and prefer rank 32-64.
    """
    import math

    lr_penalty = (math.log10(float(config["lr"])) + 3.7) ** 2
    rank_penalty = abs(math.log2(float(config["lora_r"])) - 5.5) * 0.05
    aug_bonus = 0.04 * min(int(config["augment_copies"]), 1) + 0.02 * (float(config["augment_speed"]) > 0)
    epoch_penalty = abs(float(config["epochs"]) - 8.0) * 0.004
    beam_bonus = 0.01 * (int(config["beam_size"]) > 1)
    cer = 0.32 + 0.05 * lr_penalty + rank_penalty + epoch_penalty - aug_bonus - beam_bonus
    cer = max(0.02, min(1.0, cer))
    return {"cer": cer, "wer": cer * 1.4, "term_recall": max(0.0, 1.0 - cer * 1.2), "score": cer}


def run_trial(
    index: int,
    config: dict[str, Any],
    source: str,
    parent: int | None,
    ctx: TrialContext,
    *,
    budget_sec: float,
    early_stop_cer: float = 0.0,
    stage: str = STAGE_FULL,
    max_train_seconds: float = 0.0,
) -> Trial:
    trial_dir = ctx.trial_dir(index)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trial = Trial(
        index=index, config=config, source=source, parent=parent, stage=stage, started_at=utcnow()
    )
    started = time.monotonic()

    if ctx.mock:
        metrics = mock_score(config)
        trial.metrics = metrics
        trial.score = metrics["score"]
        trial.status = STATUS_COMPLETED
        trial.seconds = time.monotonic() - started
        trial.finished_at = utcnow()
        return trial

    log_path = trial_dir / "trial.log"
    try:
        run_command(
            build_train_command(config, ctx, trial_dir, early_stop_cer, max_train_seconds),
            cwd=ctx.repo_root,
            log_path=log_path,
            timeout_sec=budget_sec,
            env=ctx.extra_env,
        )
        remaining = budget_sec - (time.monotonic() - started)
        if remaining <= 0:
            raise CommandTimeout("no time left for evaluation")
        run_command(
            build_eval_command(config, ctx, trial_dir),
            cwd=ctx.repo_root,
            log_path=log_path,
            timeout_sec=remaining,
            env=ctx.extra_env,
        )
        summary = json.loads((trial_dir / "eval" / "summary.json").read_text(encoding="utf-8"))
        trial.metrics = {
            "cer": summary.get("cer"),
            "wer": summary.get("wer"),
            "term_recall": summary.get("term_recall"),
            "rtf": summary.get("rtf"),
            "n": summary.get("n"),
        }
        trial.score = float(summary["score"])
        trial.status = STATUS_COMPLETED
        trial.metrics.update(read_train_progress(trial_dir))
    except CommandTimeout as exc:
        trial.status = STATUS_TIMEOUT
        trial.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a bad config must not kill the run
        trial.status = STATUS_FAILED
        trial.error = f"{type(exc).__name__}: {exc}"[:2000]

    trial.seconds = time.monotonic() - started
    trial.finished_at = utcnow()
    return trial


def prune_adapters(history: History, ctx: TrialContext) -> None:
    """Keep only the best few adapters on disk; failed trials keep their logs."""
    keep = {t.index for t in history.elites(ctx.keep_adapters)}
    # The adapter that will be exported must survive even if a screening trial
    # happens to have scored lower than the full-length winner.
    final_best = history.final_best()
    if final_best is not None:
        keep.add(final_best.index)
    for trial in history.trials:
        if trial.index in keep:
            continue
        adapter_dir = ctx.trial_dir(trial.index) / "adapter"
        trainer_dir = ctx.trial_dir(trial.index) / "trainer"
        for path in (adapter_dir, trainer_dir):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
