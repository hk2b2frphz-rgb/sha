#!/usr/bin/env python3
"""Fine-tune Whisper large-v3 on a synthesize_speech.py manifest (LoRA).

Tuned for renting a GPU by the hour: the defaults minimise GPU-hours rather
than chasing the last decimal of accuracy.

  * encoder frozen  -- terminology adaptation lives in the decoder, and
    skipping the encoder backward pass cuts ~35% of step time
  * LoRA on the decoder attention projections only (~1% of parameters)
  * bf16 weights, no quantisation -- large-v3 is ~3 GB, so on a 24 GB card
    8-bit would only add dequantisation overhead and slow training down
  * log-mel computed on the fly in dataloader workers -- overlaps with GPU
    compute, and avoids paying for tens of GB of feature cache on disk
  * --max-hours puts a hard ceiling on the bill

Usage:
    python scripts/finetune_whisper.py \
        --manifest out/audio/manifest.jsonl \
        --out-dir out/whisper-ft \
        --max-hours 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------
# metrics (local implementation: avoids installing jiwer/evaluate on the
# rented box, which is dead time you are paying for)
# --------------------------------------------------------------------------
def _levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_rate(refs: list[str], hyps: list[str], *, char: bool) -> float:
    num = den = 0
    for ref, hyp in zip(refs, hyps):
        r = list(ref.replace(" ", "")) if char else ref.split()
        h = list(hyp.replace(" ", "")) if char else hyp.split()
        num += _levenshtein(r, h)
        den += len(r)
    return num / den if den else 0.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class ManifestDataset(torch.utils.data.Dataset):
    """Reads manifest.jsonl as emitted by scripts/synthesize_speech.py."""

    def __init__(self, rows: list[dict], root: Path, processor, sample_rate: int = 16000):
        self.rows = rows
        self.root = root
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # soundfile rather than torchaudio.load: torchaudio >=2.11 routes loading
        # through TorchCodec, which drags in an extra package and ffmpeg. Resampling
        # still goes through torchaudio.functional, which is pure tensor ops.
        import soundfile as sf
        import torchaudio

        row = self.rows[idx]
        wav_path = self.root / row["wav"]
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio).mean(1)  # to mono
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        features = self.processor.feature_extractor(
            audio.numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(row["sentence"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {"input_features": torch.stack([f["input_features"] for f in features])}

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # the model prepends decoder_start_token itself
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# --------------------------------------------------------------------------
# cost ceiling
# --------------------------------------------------------------------------
def make_time_budget_callback(max_hours: float):
    from transformers import TrainerCallback

    class TimeBudget(TrainerCallback):
        def __init__(self) -> None:
            self.deadline = time.time() + max_hours * 3600

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() > self.deadline:
                print(
                    f"\n[budget] {max_hours}h wall-clock limit reached at step "
                    f"{state.global_step} -- stopping so the instance can be destroyed.",
                    flush=True,
                )
                control.should_training_stop = True
                control.should_save = True
            return control

    return TimeBudget()


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest.jsonl from synthesize_speech.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3, help="LoRA tolerates a high LR")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-ratio", type=float, default=0.05, help="held-out fraction")
    ap.add_argument("--eval-max", type=int, default=200, help="cap eval set (eval costs GPU-hours)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=200, help="checkpoint often on spot instances")
    ap.add_argument("--max-hours", type=float, default=0, help="hard wall-clock cap (0 = none)")
    ap.add_argument("--train-encoder", action="store_true", help="also adapt the encoder (slower)")
    ap.add_argument("--resume", action="store_true", help="resume from last checkpoint if present")
    ap.add_argument("--merge", action="store_true", help="merge LoRA into the base model at the end")
    ap.add_argument("--ct2", action="store_true", help="also convert to CTranslate2 (faster-whisper)")
    ap.add_argument("--ct2-quant", default="float16", help="CTranslate2 quantisation")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = Path(args.manifest).resolve()
    root = manifest.parent
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {manifest}")

    missing = [r["wav"] for r in rows if not (root / r["wav"]).exists()][:5]
    if missing:
        raise SystemExit(f"missing audio referenced by manifest, e.g. {missing}")

    random.shuffle(rows)
    n_eval = min(args.eval_max, max(1, int(len(rows) * args.eval_ratio))) if len(rows) > 20 else 0
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    print(f"[data] {len(train_rows)} train / {len(eval_rows)} eval utterances", flush=True)

    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task=args.task
    )

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device found. Whisper large-v3 fine-tuning is not practical on CPU -- "
            "rent a bf16-capable GPU (RTX 3090/4090) with scripts/vast_train.sh."
        )

    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    if args.train_encoder:
        target = r".*\.(q_proj|v_proj)"
    else:
        # decoder-only adaptation: no encoder backward pass
        target = r".*decoder.*\.(q_proj|v_proj)"
        model.model.encoder.requires_grad_(False)

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    train_ds = ManifestDataset(train_rows, root, processor)
    eval_ds = ManifestDataset(eval_rows, root, processor) if eval_rows else None
    collator = Collator(processor, model.config.decoder_start_token_id)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "cer": error_rate(refs, hyps, char=True),
            "wer": error_rate(refs, hyps, char=False),
        }

    out_dir = Path(args.out_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        bf16=True,
        optim="adamw_bnb_8bit",
        dataloader_num_workers=args.num_workers,
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=1,
        eval_strategy="no" if eval_ds is None else "epoch",
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=args.seed,
    )

    callbacks = [make_time_budget_callback(args.max_hours)] if args.max_hours > 0 else []

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_ds else None,
        processing_class=processor,
        callbacks=callbacks,
    )

    resume = args.resume and any(out_dir.glob("checkpoint-*"))
    if resume:
        print("[train] resuming from last checkpoint", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    hours = (time.time() - t0) / 3600
    print(f"[train] finished in {hours:.2f} h", flush=True)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"[out] LoRA adapter -> {adapter_dir}", flush=True)

    if eval_ds is not None:
        print(f"[eval] {trainer.evaluate()}", flush=True)

    if args.merge or args.ct2:
        merged_dir = out_dir / "merged"
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        print(f"[out] merged model -> {merged_dir}", flush=True)

        if args.ct2:
            import subprocess

            ct2_dir = out_dir / "ct2"
            subprocess.run(
                [
                    "ct2-transformers-converter",
                    "--model", str(merged_dir),
                    "--output_dir", str(ct2_dir),
                    "--copy_files", "tokenizer.json", "preprocessor_config.json",
                    "--quantization", args.ct2_quant,
                    "--force",
                ],
                check=True,
            )
            print(f"[out] faster-whisper model -> {ct2_dir}", flush=True)

    (out_dir / "DONE").write_text(f"{hours:.3f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
