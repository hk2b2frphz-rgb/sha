#!/usr/bin/env python3
"""Entry point for the Whisper auto-research loop (see README_autoresearch.md).

  uv run python scripts/run_autoresearch.py \
      --train-manifest out/whisper_turbo/train_manifest.jsonl \
      --eval-manifest  out/whisper_turbo/test_manifest.jsonl \
      --work-dir experiments/autoresearch --budget-hours 20 --num-gpus 4

Re-running with the same --work-dir resumes: completed trials are kept and the
proposer continues from the existing history.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.loop import LoopSettings, run_loop  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="self-improving Whisper fine-tuning loop")
    data = parser.add_argument_group("data")
    data.add_argument("--train-manifest", type=Path, required=True, help='{"id","audio","text"} JSONL')
    data.add_argument("--eval-manifest", type=Path, default=None, help="test manifest; split into dev/holdout")
    data.add_argument("--dev-manifest", type=Path, default=None, help="explicit dev set (skips splitting)")
    data.add_argument("--holdout-manifest", type=Path, default=None, help="explicit holdout set")
    data.add_argument("--dev-ratio", type=float, default=0.5, help="fraction of --eval-manifest used as dev")
    data.add_argument("--terms", type=Path, default=None, help="domain terms, one per line")
    data.add_argument(
        "--guidance",
        type=Path,
        default=None,
        help="human notes handed to the proposer (see PROPOSER_GUIDANCE.md)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--base-model", default=None, help="base model path / HF id")
    model.add_argument("--base-model-file", type=Path, default=Path("manifest.txt"))
    model.add_argument("--language", default="ja")
    model.add_argument("--num-gpus", type=int, default=1, help="DDP processes for training")
    model.add_argument("--eval-batch-size", type=int, default=8)

    search = parser.add_argument_group("search")
    search.add_argument("--work-dir", type=Path, default=Path("experiments/autoresearch"))
    search.add_argument("--budget-hours", type=float, default=20.0, help="wall-clock budget for the loop")
    search.add_argument("--reserve-minutes", type=float, default=40.0, help="time kept for the final steps")
    search.add_argument("--max-trial-hours", type=float, default=6.0, help="hard cap per trial")
    search.add_argument("--max-trials", type=int, default=0, help="0 = limited only by the time budget")
    search.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="stop after this many failures in a row (0 = never stop)",
    )
    search.add_argument("--objective", default="cer", choices=["cer", "wer", "cer_term"])
    search.add_argument("--term-weight", type=float, default=0.5)
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--n-random", type=int, default=4, help="random configs before the LLM takes over")
    search.add_argument("--elite-k", type=int, default=4)
    search.add_argument("--keep-adapters", type=int, default=3, help="adapters kept on disk")
    search.add_argument(
        "--screen-minutes",
        type=float,
        default=25.0,
        help="fixed training minutes per screening trial (0 = no screening stage)",
    )
    search.add_argument(
        "--screen-budget-fraction",
        type=float,
        default=0.6,
        help="share of the budget spent on screening before the finalists are re-run",
    )
    search.add_argument(
        "--finalists", type=int, default=3, help="top screened configs re-run at full length"
    )
    search.add_argument(
        "--early-stop-margin",
        type=float,
        default=0.35,
        help="abort a trial whose mid-training dev CER exceeds best*(1+margin) (0 = off)",
    )
    search.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="abort when mid-training dev CER stops improving for N epochs (0 = off)",
    )
    search.add_argument(
        "--early-stop-dev-limit", type=int, default=24, help="utterances decoded per mid-training eval"
    )

    proposer = parser.add_argument_group("proposer")
    proposer.add_argument("--gemma-model", default="google/gemma-4-E4B-it")
    proposer.add_argument("--no-gemma", action="store_true", help="evolutionary search only")
    proposer.add_argument("--gemma-timeout-sec", type=int, default=900)
    proposer.add_argument("--gemma-proposals", type=int, default=3)
    proposer.add_argument(
        "--no-analysis", action="store_true", help="skip the Gemma-written discussion in report.md"
    )

    final = parser.add_argument_group("final")
    final.add_argument("--no-holdout-eval", action="store_true")
    final.add_argument("--final-export", action="store_true", help="merge best LoRA and convert to CT2")
    final.add_argument("--ct2-quantization", default="float16")
    final.add_argument("--mock", action="store_true", help="wiring test: synthetic scores, no GPU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = LoopSettings(
        repo_root=Path(__file__).resolve().parents[1],
        work_dir=args.work_dir,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        dev_manifest=args.dev_manifest,
        holdout_manifest=args.holdout_manifest,
        base_model=args.base_model,
        base_model_file=args.base_model_file,
        terms_file=args.terms,
        num_gpus=max(1, args.num_gpus),
        eval_batch_size=args.eval_batch_size,
        language=args.language,
        objective=args.objective,
        term_weight=args.term_weight,
        budget_hours=args.budget_hours,
        reserve_minutes=args.reserve_minutes,
        max_trial_hours=args.max_trial_hours,
        max_trials=args.max_trials,
        max_consecutive_failures=args.max_consecutive_failures,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        keep_adapters=args.keep_adapters,
        screen_minutes=args.screen_minutes,
        screen_budget_fraction=args.screen_budget_fraction,
        finalists=args.finalists,
        guidance_file=args.guidance,
        early_stop_margin=args.early_stop_margin,
        early_stop_patience=args.early_stop_patience,
        early_stop_dev_limit=args.early_stop_dev_limit,
        n_random=args.n_random,
        elite_k=args.elite_k,
        use_gemma=not args.no_gemma,
        gemma_model=args.gemma_model,
        gemma_timeout_sec=args.gemma_timeout_sec,
        gemma_proposals=args.gemma_proposals,
        write_analysis=not args.no_analysis,
        final_holdout_eval=not args.no_holdout_eval,
        final_export=args.final_export,
        ct2_quantization=args.ct2_quantization,
        mock=args.mock,
    )
    return run_loop(settings)


if __name__ == "__main__":
    raise SystemExit(main())
