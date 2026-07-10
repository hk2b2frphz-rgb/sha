#!/usr/bin/env python3
"""whisper-large-v3-turbo を LoRA で fine-tune する (HF transformers + peft)。

入力は build_whisper_manifest.py が出す学習 manifest JSONL:
  {"id", "audio", "text", "duration_sec"}

ベースモデルのパスは manifest.txt (1行1パス、先頭の非コメント行) から読む。
HPC 上ではこの manifest.txt を編集してローカルの重みを指す。

出力:
  <out-dir>/adapter/            LoRA アダプタ + processor
  <merge-dir> (任意)            base に adapter をマージした HF形式モデル
                                -> PBS 側で ct2-transformers-converter に渡す

複数GPU (V100×4) では accelerate launch / torchrun 経由で DDP 実行する。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="whisper-large-v3-turbo LoRA fine-tune")
    parser.add_argument("--manifest", type=Path, required=True, help="学習 manifest JSONL")
    parser.add_argument("--base-model", type=str, default=None, help="ベースモデルのパス/HF id (直接指定)")
    parser.add_argument(
        "--base-model-file",
        type=Path,
        default=Path("manifest.txt"),
        help="ベースモデルのパスを記した txt (--base-model 未指定時に使用)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="アダプタ等の出力先")
    parser.add_argument("--merge-dir", type=Path, default=None, help="マージ済みHFモデルの保存先 (任意)")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8, help="per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-label-len", type=int, default=225, help="ラベルtoken上限 (whisperは448)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_base_model(args: argparse.Namespace) -> str:
    if args.base_model:
        return args.base_model
    if not args.base_model_file.exists():
        raise SystemExit(
            f"ベースモデル未指定で {args.base_model_file} も無い。"
            " --base-model を渡すか manifest.txt を用意してください。"
        )
    for raw in args.base_model_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            return line
    raise SystemExit(f"{args.base_model_file} に有効なパスがありません")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # 先頭の BOS はモデルが強制付与するため落とす。
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Audio, Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    base_model = resolve_base_model(args)
    print(f"[train] base model: {base_model}")

    processor = WhisperProcessor.from_pretrained(base_model, language=args.language, task=args.task)

    rows = []
    for raw in args.manifest.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rec = __import__("json").loads(raw)
            rows.append({"audio": rec["audio"], "text": rec["text"]})
    if not rows:
        raise SystemExit(f"学習manifestが空です: {args.manifest}")
    print(f"[train] {len(rows)} training examples")

    dataset = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    max_label_len = args.max_label_len

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["text"]).input_ids[:max_label_len]
        return batch

    dataset = dataset.map(prepare, remove_columns=dataset.column_names, num_proc=1)

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    # 日本語 transcribe を強制し、言語自動判定によるブレを防ぐ。
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    # gradient checkpointing + PEFT の勾配伝播のため入力に requires_grad を立てる。
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    adapter_dir = args.out_dir / "adapter"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    trainer.train()

    if trainer.is_world_process_zero():
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        print(f"[train] saved adapter -> {adapter_dir}")

        if args.merge_dir is not None:
            from peft import PeftModel

            print("[train] merging LoRA adapter into base ...")
            base = WhisperForConditionalGeneration.from_pretrained(base_model)
            base.generation_config.language = args.language
            base.generation_config.task = args.task
            base.generation_config.forced_decoder_ids = None
            base.config.forced_decoder_ids = None
            base.config.suppress_tokens = []
            merged = PeftModel.from_pretrained(base, str(adapter_dir))
            merged = merged.merge_and_unload()
            args.merge_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(args.merge_dir)
            processor.save_pretrained(args.merge_dir)
            print(f"[train] saved merged model -> {args.merge_dir}")


if __name__ == "__main__":
    main()
