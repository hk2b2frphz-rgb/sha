"""The auto-research loop itself: budgeted, resumable, self-improving."""
from __future__ import annotations

import json
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analyst import GemmaAnalyst
from .data import collect_terms, split_manifest
from .history import STAGE_FULL, STAGE_SCREEN, History, utcnow
from .proposer import GemmaProposer, ProposerError, ProposerPipeline
from .report import write_report
from .space import config_key
from .runner import (
    CommandTimeout,
    TrialContext,
    early_stop_threshold,
    prune_adapters,
    run_command,
    run_trial,
)

# Used to decide whether another trial fits before the first one has finished.
DEFAULT_TRIAL_ESTIMATE_SEC = 45 * 60.0


@dataclass
class LoopSettings:
    repo_root: Path
    work_dir: Path
    train_manifest: Path
    eval_manifest: Path | None = None
    dev_manifest: Path | None = None
    holdout_manifest: Path | None = None
    base_model: str | None = None
    base_model_file: Path = Path("manifest.txt")
    terms_file: Path | None = None
    guidance_file: Path | None = None
    num_gpus: int = 1
    eval_batch_size: int = 8
    language: str = "ja"
    objective: str = "cer"
    term_weight: float = 0.5
    budget_hours: float = 20.0
    reserve_minutes: float = 40.0
    max_trial_hours: float = 6.0
    max_trials: int = 0
    max_consecutive_failures: int = 5
    # Two-stage search: screen many configs under a fixed short training budget,
    # then re-run the best few at full length. screen_minutes=0 disables it.
    screen_minutes: float = 25.0
    screen_budget_fraction: float = 0.6
    finalists: int = 3
    dev_ratio: float = 0.5
    seed: int = 42
    keep_adapters: int = 3
    early_stop_margin: float = 0.35
    early_stop_patience: int = 2
    early_stop_dev_limit: int = 24
    n_random: int = 4
    elite_k: int = 4
    use_gemma: bool = True
    gemma_model: str = "google/gemma-4-E4B-it"
    gemma_timeout_sec: int = 900
    gemma_proposals: int = 3
    write_analysis: bool = True
    prompt_term_count: int = 30
    final_holdout_eval: bool = True
    final_export: bool = False
    ct2_quantization: str = "float16"
    mock: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


class StopFlag:
    """Set by SIGTERM/SIGINT/SIGUSR1 so a killed job still checkpoints cleanly."""

    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def install(self) -> None:
        names = ["SIGTERM", "SIGINT", "SIGUSR1"]
        for name in names:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass

    def _handle(self, signum, _frame) -> None:  # noqa: ANN001
        self.requested = True
        self.reason = f"signal {signum}"
        print(f"[loop] stop requested ({self.reason}); finishing after the current trial", flush=True)


def prepare_data(settings: LoopSettings) -> tuple[Path, Path | None, Path | None]:
    """Return (dev_manifest, holdout_manifest, terms_file), splitting if needed."""
    data_dir = settings.work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dev = settings.dev_manifest
    holdout = settings.holdout_manifest
    if dev is None:
        if settings.eval_manifest is None:
            raise SystemExit("either --dev-manifest or --eval-manifest is required")
        dev = data_dir / "dev.jsonl"
        holdout = holdout or data_dir / "holdout.jsonl"
        if dev.exists() and holdout.exists():
            print(f"[loop] reusing existing split: {dev} / {holdout}")
        else:
            n_dev, n_hold = split_manifest(
                settings.eval_manifest,
                dev,
                holdout,
                dev_ratio=settings.dev_ratio,
                seed=settings.seed,
            )
            print(f"[loop] split {settings.eval_manifest}: dev={n_dev} holdout={n_hold}")
    if holdout is not None and not holdout.exists():
        holdout = None

    terms_file = settings.terms_file
    if terms_file is None:
        # build_whisper_manifest.py drops the term column, so also look at the
        # TTS input JSONL that sits next to the training data.
        terms = collect_terms(
            settings.train_manifest,
            settings.train_manifest.parent / "tts_input.jsonl",
            dev,
            limit=200,
        )
        if terms:
            terms_file = data_dir / "terms.txt"
            terms_file.write_text("\n".join(terms) + "\n", encoding="utf-8")
            print(f"[loop] {len(terms)} domain terms -> {terms_file}")
        else:
            print(
                "[loop] WARNING: no domain terms found; term recall and the "
                "prompt_terms knob are inactive. Pass --terms to enable them."
            )
    return dev, holdout, terms_file


def load_guidance(settings: LoopSettings) -> str:
    """Human-written notes handed to the proposer (karpathy/autoresearch's program.md).

    A place to put domain knowledge the search cannot discover on its own, e.g.
    "this domain's terms lose their geminate consonants" or "keep beam <= 3".
    """
    path = settings.guidance_file
    if path is None or not Path(path).exists():
        return ""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text:
        print(f"[loop] proposer guidance: {path} ({len(text)} chars)")
    return text


def build_context(settings: LoopSettings, dev: Path, terms_file: Path | None) -> TrialContext:
    prompt_text = ""
    if terms_file is not None and terms_file.exists():
        terms = [t.strip() for t in terms_file.read_text(encoding="utf-8").splitlines() if t.strip()]
        prompt_text = "、".join(terms[: settings.prompt_term_count])
    return TrialContext(
        repo_root=settings.repo_root,
        work_dir=settings.work_dir,
        train_manifest=settings.train_manifest,
        dev_manifest=dev,
        base_model_file=settings.base_model_file,
        base_model=settings.base_model,
        num_gpus=settings.num_gpus,
        language=settings.language,
        objective=settings.objective,
        term_weight=settings.term_weight,
        terms_file=terms_file,
        prompt_text=prompt_text,
        eval_batch_size=settings.eval_batch_size,
        seed=settings.seed,
        keep_adapters=settings.keep_adapters,
        early_stop_margin=settings.early_stop_margin,
        early_stop_patience=settings.early_stop_patience,
        early_stop_dev_limit=settings.early_stop_dev_limit,
        mock=settings.mock,
        extra_env=settings.extra_env,
    )


def run_final_holdout(
    settings: LoopSettings, ctx: TrialContext, history: History, holdout: Path, budget_sec: float
) -> dict[str, Any] | None:
    best = history.final_best()
    if best is None:
        return None
    adapter = ctx.trial_dir(best.index) / "adapter"
    if settings.mock:
        return {"note": "mock run; holdout evaluation skipped", "score": best.score}
    if not adapter.exists():
        return {"error": f"best adapter was pruned: {adapter}"}
    out_dir = settings.work_dir / "final" / "holdout_eval"
    cmd = [
        "uv", "run", "python", "scripts/eval_whisper_hf.py",
        "--manifest", str(holdout),
        "--adapter", str(adapter),
        "--out-dir", str(out_dir),
        "--language", settings.language,
        "--batch-size", str(settings.eval_batch_size),
        "--beam-size", str(int(best.config["beam_size"])),
        "--no-repeat-ngram-size", str(int(best.config["no_repeat_ngram_size"])),
        "--objective", settings.objective,
        "--term-weight", str(settings.term_weight),
    ]
    if settings.base_model:
        cmd += ["--base-model", settings.base_model]
    else:
        cmd += ["--base-model-file", str(settings.base_model_file)]
    if ctx.terms_file is not None:
        cmd += ["--terms", str(ctx.terms_file)]
    if int(best.config["prompt_terms"]) == 1 and ctx.prompt_text:
        cmd += ["--prompt-text", ctx.prompt_text]
    try:
        run_command(
            cmd,
            cwd=settings.repo_root,
            log_path=settings.work_dir / "final" / "final.log",
            timeout_sec=budget_sec,
            env=settings.extra_env,
        )
        return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    except (CommandTimeout, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:500]}


def run_final_export(
    settings: LoopSettings, ctx: TrialContext, history: History, budget_sec: float
) -> dict[str, Any] | None:
    best = history.final_best()
    if best is None or settings.mock:
        return None
    adapter = ctx.trial_dir(best.index) / "adapter"
    if not adapter.exists():
        return {"error": f"best adapter was pruned: {adapter}"}
    merged_dir = settings.work_dir / "final" / "merged_hf"
    ct2_dir = settings.work_dir / "final" / "ct2"
    cmd = [
        "uv", "run", "python", "scripts/export_whisper_model.py",
        "--adapter", str(adapter),
        "--merge-dir", str(merged_dir),
        "--ct2-dir", str(ct2_dir),
        "--quantization", settings.ct2_quantization,
        "--language", settings.language,
    ]
    if settings.base_model:
        cmd += ["--base-model", settings.base_model]
    else:
        cmd += ["--base-model-file", str(settings.base_model_file)]
    try:
        run_command(
            cmd,
            cwd=settings.repo_root,
            log_path=settings.work_dir / "final" / "final.log",
            timeout_sec=budget_sec,
            env=settings.extra_env,
        )
        return {"merged_hf": str(merged_dir), "ct2": str(ct2_dir)}
    except (CommandTimeout, RuntimeError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:500]}


def run_loop(settings: LoopSettings) -> int:
    started = time.monotonic()
    deadline = started + settings.budget_hours * 3600.0
    reserve_sec = settings.reserve_minutes * 60.0

    settings.work_dir.mkdir(parents=True, exist_ok=True)
    dev, holdout, terms_file = prepare_data(settings)
    ctx = build_context(settings, dev, terms_file)

    history = History.load_or_create(
        settings.work_dir / "state",
        run_meta={
            "train_manifest": str(settings.train_manifest),
            "dev_manifest": str(dev),
            "holdout_manifest": str(holdout) if holdout else None,
            "objective": settings.objective,
            "base_model": settings.base_model or f"file:{settings.base_model_file}",
        },
    )
    history.start_session(
        {
            "budget_hours": settings.budget_hours,
            "num_gpus": settings.num_gpus,
            "gemma": settings.gemma_model if settings.use_gemma else None,
            "mock": settings.mock,
        }
    )

    gemma = None
    if settings.use_gemma and not settings.mock:
        gemma = GemmaProposer(
            script=settings.repo_root / "scripts" / "propose_with_gemma.py",
            out_dir=settings.work_dir / "proposals",
            model=settings.gemma_model,
            timeout_sec=settings.gemma_timeout_sec,
            n_proposals=settings.gemma_proposals,
            objective_note=f"minimize dev {settings.objective}",
            guidance=load_guidance(settings),
        )
    rng = random.Random(settings.seed + len(history.trials) * 7919)
    pipeline = ProposerPipeline(
        rng, gemma, n_random=settings.n_random, elite_k=settings.elite_k
    )

    stop = StopFlag()
    stop.install()
    failures: list[str] = []
    report_path = settings.work_dir / "report.md"
    consecutive_failures = 0

    print(
        f"[loop] budget={settings.budget_hours}h reserve={settings.reserve_minutes}min "
        f"gpus={settings.num_gpus} resumed_trials={len(history.trials)}",
        flush=True,
    )

    def run_one(index, config, source, parent, stage, trial_budget, max_train_seconds):  # noqa: ANN001
        nonlocal consecutive_failures
        threshold = early_stop_threshold(history, ctx, stage)
        print(
            f"[loop] trial {index} stage={stage} source={source} "
            f"budget={trial_budget / 60:.0f}min early_stop_cer={threshold:.4f} "
            f"config={json.dumps(config, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )
        trial = run_trial(
            index,
            config,
            source,
            parent,
            ctx,
            budget_sec=trial_budget,
            early_stop_cer=threshold,
            stage=stage,
            max_train_seconds=max_train_seconds,
        )
        history.append(trial)
        prune_adapters(history, ctx)
        best = history.best(stage)
        print(
            f"[loop] trial {index} -> status={trial.status} score={trial.score} "
            f"({trial.seconds / 60:.1f} min); best[{stage}]={best.score if best else None} "
            f"(trial {best.index if best else '-'})",
            flush=True,
        )
        if trial.error:
            print(f"[loop] trial {index} error: {trial.error}", flush=True)
        write_report(history, report_path, {"status": "running", "updated_at": utcnow()})
        consecutive_failures = 0 if trial.ok else consecutive_failures + 1
        return trial

    def time_left(until: float) -> float:
        return until - time.monotonic() - reserve_sec

    def out_of_time(until: float, trial_cap_sec: float = 0.0) -> str:
        """Reason to stop, or "" when another trial still fits."""
        remaining = time_left(until)
        estimate = 0.0 if settings.mock else (history.mean_trial_seconds() or DEFAULT_TRIAL_ESTIMATE_SEC)
        # A screening run has a fixed training cap.  In particular, do not use
        # the 45-minute cold-start estimate for a 25-minute screening trial:
        # doing so can reject every trial near a short stage deadline even
        # though the runner will enforce the smaller wall-clock budget.
        if trial_cap_sec > 0:
            estimate = min(estimate, trial_cap_sec)
        if remaining <= 0 or remaining < estimate * 1.05:
            return (
                f"not enough time left ({remaining / 60:.1f} min < estimated trial "
                f"{estimate / 60:.1f} min)"
            )
        return ""

    def failure_limit_hit(index: int) -> str:
        if (
            settings.max_consecutive_failures
            and consecutive_failures >= settings.max_consecutive_failures
        ):
            return (
                f"{consecutive_failures} trials failed in a row; stopping so the "
                f"cause can be fixed (see {ctx.trial_dir(index) / 'trial.log'})"
            )
        return ""

    # ---- stage 1: screening -------------------------------------------------
    # Every screening trial trains for the same short wall clock, so many
    # configurations can be compared for the price of one full run. This is the
    # single change that moves the trial count from ~15 to ~50 in a night.
    screening = settings.screen_minutes > 0
    screen_stage = STAGE_SCREEN if screening else STAGE_FULL
    screen_deadline = deadline
    if screening:
        screen_deadline = started + settings.budget_hours * 3600.0 * settings.screen_budget_fraction
        print(
            f"[loop] stage 1 (screening): {settings.screen_minutes} min of training per trial, "
            f"until {settings.screen_budget_fraction:.0%} of the budget is used",
            flush=True,
        )

    stop_reason = "budget exhausted"
    while True:
        if stop.requested:
            stop_reason = f"stop requested ({stop.reason})"
            break
        if settings.max_trials and len(history.trials) >= settings.max_trials:
            stop_reason = f"max_trials={settings.max_trials} reached"
            break
        reason = out_of_time(
            screen_deadline,
            settings.screen_minutes * 60.0 if screening else 0.0,
        )
        if reason:
            stop_reason = f"screening done: {reason}" if screening else reason
            break

        index = history.next_index()
        try:
            proposal = pipeline.propose(history, stage=screen_stage)
        except ProposerError as exc:
            stop_reason = f"proposer stopped the run: {exc}"
            print(f"[loop] ERROR: {exc}", flush=True)
            break
        trial = run_one(
            index,
            proposal.config,
            proposal.source,
            proposal.parent,
            screen_stage,
            min(time_left(screen_deadline), settings.max_trial_hours * 3600.0),
            settings.screen_minutes * 60.0 if screening else 0.0,
        )
        reason = failure_limit_hit(index)
        if reason:
            stop_reason = reason
            break

    print(f"[loop] stage 1 stopping: {stop_reason}", flush=True)

    # ---- stage 2: full-length runs for the finalists -------------------------
    # Screening ranks configs under a short budget; that ranking does not
    # automatically hold for full training, so the top few are re-run properly
    # and only those results are used to pick the model that ships.
    finals_reason = ""
    if screening and not stop.requested and history.completed(STAGE_SCREEN):
        finalists = history.elites(settings.finalists, stage=STAGE_SCREEN)
        print(
            f"[loop] stage 2 (finals): re-running the top {len(finalists)} configs at full length",
            flush=True,
        )
        tried_full = {config_key(t.config) for t in history.trials if t.stage == STAGE_FULL}
        for finalist in finalists:
            if stop.requested:
                finals_reason = f"stop requested ({stop.reason})"
                break
            reason = out_of_time(deadline)
            if reason:
                finals_reason = reason
                break
            if config_key(finalist.config) in tried_full:
                continue
            index = history.next_index()
            trial = run_one(
                index,
                finalist.config,
                f"finalist(trial {finalist.index})",
                finalist.index,
                STAGE_FULL,
                min(time_left(deadline), settings.max_trial_hours * 3600.0),
                0.0,
            )
            tried_full.add(config_key(finalist.config))
            reason = failure_limit_hit(index)
            if reason:
                finals_reason = reason
                break
        print(f"[loop] stage 2 finished{': ' + finals_reason if finals_reason else ''}", flush=True)

    extra: dict[str, Any] = {"stop_reason": stop_reason}
    if screening:
        extra["screening"] = (
            f"{len(history.completed(STAGE_SCREEN))} screened at {settings.screen_minutes} min, "
            f"{len(history.completed(STAGE_FULL))} re-run at full length"
            + (f" ({finals_reason})" if finals_reason else "")
        )
    remaining = deadline - time.monotonic()
    if settings.final_holdout_eval and holdout is not None and remaining > 0:
        summary = run_final_holdout(settings, ctx, history, holdout, remaining)
        if summary:
            extra["holdout"] = json.dumps(
                {k: v for k, v in summary.items() if k in ("n", "cer", "wer", "term_recall", "score", "error")},
                ensure_ascii=False,
            )
            final_dir = settings.work_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "holdout_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    remaining = deadline - time.monotonic()
    if settings.final_export and remaining > 0:
        exported = run_final_export(settings, ctx, history, remaining)
        if exported:
            extra["export"] = json.dumps(exported, ensure_ascii=False)

    # The written discussion comes last: the search results are already safe on
    # disk, so a slow or broken analyst only costs the prose.
    analysis = None
    if settings.write_analysis and settings.use_gemma and not settings.mock:
        analyst = GemmaAnalyst(
            script=settings.repo_root / "scripts" / "write_report_with_gemma.py",
            out_dir=settings.work_dir / "final",
            model=settings.gemma_model,
            timeout_sec=settings.gemma_timeout_sec,
        )
        print("[loop] asking Gemma to write the report discussion ...", flush=True)
        analysis = analyst.analyze(history, extra)
        if analysis is None:
            # Not silently skipped: the report records it and the job exits
            # non-zero, so "the LLM part never ran" cannot look like success.
            failures.append(f"report discussion: {analyst.last_error}")
            extra["analysis_error"] = analyst.last_error or "unknown"
            print(f"[loop] ERROR: report discussion failed: {analyst.last_error}", flush=True)

    history.save()
    write_report(history, report_path, extra, analysis, settings.gemma_model if analysis else None)
    best = history.final_best()
    if best is not None:
        (settings.work_dir / "best_config.json").write_text(
            json.dumps(best.config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[loop] best trial {best.index} (stage={best.stage}) score={best.score:.4f}", flush=True
        )
        print(f"[loop] best adapter: {ctx.trial_dir(best.index) / 'adapter'}", flush=True)
    print(f"[loop] report: {report_path}", flush=True)

    if failures:
        print("[loop] FAILED steps:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    return 0
