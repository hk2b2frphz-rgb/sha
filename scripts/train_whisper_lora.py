#!/usr/bin/env python3
"""whisper-large-v3-turbo を fine-tune する (HF transformers + peft)。

--finetune-mode で 2 方式を選ぶ:
  lora  LoRA/PEFT アダプタだけを学習 (既定)
  full  重みを直接学習。既定で **encoder は凍結**し decoder だけ動かす

encoder 凍結の full-FT を用意しているのは、この学習データが TTS 合成音声だから。
encoder まで動かすと合成音の音響特性に過適合して実音声で崩れる一方、decoder だけ
なら「音は同じ・表記だけ専門用語に寄せる」という本来の狙いに合う。

入力は build_whisper_manifest.py が出す学習 manifest JSONL:
  {"id", "audio", "text", "duration_sec"}

ベースモデルのパスは manifest.txt (1行1パス、先頭の非コメント行) から読む。
HPC 上ではこの manifest.txt を編集してローカルの重みを指す。

出力 (lora):
  <out-dir>/adapter/            LoRA アダプタ + processor
  <merge-dir> (任意)            base に adapter をマージした HF形式モデル
出力 (full):
  <merge-dir> か <out-dir>/model/   学習済み HF形式モデル + processor
いずれも -> PBS 側で ct2-transformers-converter に渡す。

複数GPU (V100×4) では accelerate launch / torchrun 経由で DDP 実行する。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.asr_text import error_rate, normalize_text  # noqa: E402  (sys.path bootstrap above)


# 学習方式ごとの既定学習率。full-FT は重みを直接動かすので LoRA より 1 桁小さくする
# (Whisper large 系の目安は事前学習の 1/40 = 5e-6〜1e-5)。
FINETUNE_MODES = ("lora", "full")
DEFAULT_LR = {"lora": 1e-4, "full": 1e-5}
MIXED_PRECISION_MODES = ("auto", "fp16", "bf16")


def resolve_learning_rate(mode: str, lr: float | None) -> float:
    """--lr 明示があればそれ、無ければ方式ごとの既定値。"""
    return float(lr) if lr is not None else DEFAULT_LR[mode]


def resolve_freeze_encoder(mode: str, flag: bool | None) -> bool:
    """encoder を凍結するか。full は既定で凍結、lora は常に対象外。"""
    if mode == "lora":
        if flag:
            raise SystemExit(
                "--freeze-encoder は --finetune-mode full 用です "
                "(lora ではベース重みが元から全て凍結されています)"
            )
        return False
    return True if flag is None else bool(flag)


def count_parameters(model: Any) -> tuple[int, int]:
    """(学習対象パラメータ数, 全パラメータ数)。"""
    trainable = 0
    total = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


def freeze_encoder(model: Any) -> None:
    """encoder の全パラメータを requires_grad=False にする。

    encoder は train モードのままにしておく。SpecAugment と dropout は
    self.training を見て掛かるので、凍結後も decoder 側から見た入力の
    揺らぎとして効き、正則化になる。
    """
    for param in model.get_encoder().parameters():
        param.requires_grad = False


def verify_encoder_frozen(model: Any) -> int:
    """encoder が完全に凍結されていることを検証し、パラメータ数を返す。"""
    params = list(model.get_encoder().parameters())
    if not params:
        raise RuntimeError("Whisper encoder has no parameters; refusing decoder-only fine-tuning")
    trainable = [param for param in params if param.requires_grad]
    if trainable:
        trainable_count = sum(int(param.numel()) for param in trainable)
        raise RuntimeError(
            "decoder-only fine-tuning requires a fully frozen encoder, but "
            f"{len(trainable)} tensors ({trainable_count:,} parameters) are still trainable"
        )
    return sum(int(param.numel()) for param in params)


def truncate_label_ids(input_ids: list[int], max_length: int, eos_token_id: int | None) -> list[int]:
    """ラベルを切り詰める。切詰め時も末尾の EOT/EOS は必ず残す。"""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    ids = list(input_ids)
    if len(ids) <= max_length:
        return ids
    truncated = ids[:max_length]
    if eos_token_id is None:
        raise ValueError("eos_token_id is required when labels are truncated")
    truncated[-1] = int(eos_token_id)
    return truncated


def resolve_mixed_precision(
    mode: str,
    *,
    cuda_available: bool,
    bf16_supported: bool,
) -> tuple[bool, bool]:
    """``(fp16, bf16)`` を返す。auto は対応 GPU なら bf16 を優先する。"""
    if mode not in MIXED_PRECISION_MODES:
        raise ValueError(f"unsupported mixed precision mode: {mode}")
    if mode == "auto":
        if not cuda_available:
            return False, False
        return (False, True) if bf16_supported else (True, False)
    if not cuda_available:
        raise SystemExit(f"--mixed-precision {mode} requires CUDA")
    if mode == "bf16" and not bf16_supported:
        raise SystemExit("--mixed-precision bf16 is not supported by this CUDA device/runtime")
    return mode == "fp16", mode == "bf16"


def resolve_checkpoint_policy(has_dev: bool, requested_save_strategy: str) -> tuple[str, str, bool]:
    """``(eval_strategy, save_strategy, load_best_model_at_end)`` を返す。"""
    if has_dev:
        # transformers は best model 復元時に eval/save strategy の一致を要求する。
        return "epoch", "epoch", True
    return "no", requested_save_strategy, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="whisper-large-v3-turbo fine-tune (LoRA / full)")
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
    parser.add_argument(
        "--finetune-mode",
        default="lora",
        choices=list(FINETUNE_MODES),
        help="lora=アダプタのみ / full=重みを直接学習 (既定で encoder 凍結)",
    )
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="full-FT で encoder を凍結する (既定: 凍結)。--no-freeze-encoder で全層学習",
    )
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"学習率 (既定は方式依存: lora={DEFAULT_LR['lora']:g}, full={DEFAULT_LR['full']:g})",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--mixed-precision",
        default="auto",
        choices=list(MIXED_PRECISION_MODES),
        help="auto=GPUに応じてbf16/fp16を選択、または fp16 / bf16 を明示",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler",
        default="linear",
        choices=["linear", "cosine", "constant_with_warmup"],
        help="学習率スケジュール",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,out_proj",
        help="LoRA を挿す線形層 (カンマ区切り)",
    )
    # 音声側のデータ拡張。--augment-copies>0 のとき、拡張済みコピーを学習集合に足す。
    parser.add_argument("--augment-copies", type=int, default=0, help="拡張コピーの本数 (0で無効)")
    parser.add_argument("--augment-speed", type=float, default=0.0, help="話速の揺らぎ幅 (±割合)")
    parser.add_argument("--augment-noise-snr-db", type=float, default=0.0, help="加算ノイズのSNR (0で無効)")
    parser.add_argument("--augment-gain-db", type=float, default=0.0, help="音量の揺らぎ幅 (±dB)")
    # SpecAugment (メル特徴のマスク)。0 のとき無効。
    parser.add_argument("--mask-time-prob", type=float, default=0.0)
    parser.add_argument("--mask-feature-prob", type=float, default=0.0)
    parser.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["epoch", "no"],
        help="中間チェックポイント (探索ループでは no にしてディスクを節約する)",
    )
    # 学習中の足切り。--dev-manifest を渡したときだけ有効になる。
    parser.add_argument("--dev-manifest", type=Path, default=None, help="学習中に測る dev manifest")
    parser.add_argument("--dev-limit", type=int, default=24, help="学習中の評価に使う発話数")
    parser.add_argument(
        "--early-stop-cer",
        type=float,
        default=0.0,
        help="評価時点の dev CER がこれを超えたら学習を打ち切る (0で無効)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="dev CER がこの回数だけ改善しなければ打ち切る (0で無効)",
    )
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=0.0,
        help="学習の実時間上限。全設定を同じ計算量で比べる代理試行で使う (0で無効)",
    )
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
    lines = args.base_model_file.read_text(encoding="utf-8").splitlines()
    # manifest.txt may also hold batch-evaluation settings. Prefer its explicit
    # BASE_MODEL setting, while retaining the original one-model-per-file form.
    for raw in lines:
        line = raw.strip()
        if line.startswith("BASE_MODEL=") and line.removeprefix("BASE_MODEL=").strip():
            return line.removeprefix("BASE_MODEL=").strip()
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#"):
            if "=" not in line:
                return line
    raise SystemExit(f"{args.base_model_file} に有効なパスがありません")


def load_manifest_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """{"audio","text"} だけを取り出す。limit>0 で先頭 N 件に絞る。"""
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        audio = rec.get("audio") or rec.get("wav")
        text = rec.get("text") or rec.get("sentence") or ""
        if not audio:
            continue
        rows.append({"audio": str(audio), "text": str(text)})
        if limit and len(rows) >= limit:
            break
    return rows


class EarlyStopPolicy:
    """学習中の dev CER を見て打ち切りを判断する (transformers 非依存)。

    threshold: 既存ベストから見て見込みがない水準。1回でも超えたら止める。
    patience : 改善が止まった回数。エポックを使い切る前に切り上げる。
    """

    def __init__(self, threshold: float = 0.0, patience: int = 0, min_delta: float = 1e-4) -> None:
        self.threshold = float(threshold or 0.0)
        self.patience = int(patience or 0)
        self.min_delta = min_delta
        self.history: list[float] = []
        self.best: float | None = None
        self.since_improvement = 0

    def update(self, cer: float) -> tuple[bool, str]:
        cer = float(cer)
        self.history.append(cer)
        if self.best is None or cer < self.best - self.min_delta:
            self.best = cer
            self.since_improvement = 0
        else:
            self.since_improvement += 1

        if self.threshold > 0 and cer > self.threshold:
            return True, f"dev CER {cer:.4f} exceeds the early-stop threshold {self.threshold:.4f}"
        if self.patience > 0 and self.since_improvement >= self.patience:
            return True, f"dev CER has not improved for {self.since_improvement} evaluations"
        return False, ""


def augment_waveform(
    wav: Any,
    sample_rate: int,
    seed: int,
    *,
    speed: float = 0.0,
    noise_snr_db: float = 0.0,
    gain_db: float = 0.0,
) -> Any:
    """話速・音量・雑音を乱数で揺らした波形を返す (16kHz mono float32 前提)。

    seed を例文ごとに固定するので、同じ設定なら毎回同じ拡張データになり、
    試行間のスコア差がデータの偶然で揺れない。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    audio = np.asarray(wav, dtype="float32")

    if speed > 0:
        import librosa

        factor = float(rng.uniform(1.0 - speed, 1.0 + speed))
        if abs(factor - 1.0) > 1e-3:
            # 話速変換 (ピッチも一緒に動く Kaldi 流の speed perturbation)。
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=int(sample_rate / factor)
            ).astype("float32")

    if gain_db > 0:
        audio = audio * float(10.0 ** (rng.uniform(-gain_db, gain_db) / 20.0))

    if noise_snr_db > 0:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        if rms > 0:
            noise_rms = rms / (10.0 ** (noise_snr_db / 20.0))
            audio = audio + rng.normal(0.0, noise_rms, size=audio.shape).astype("float32")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.ascontiguousarray(audio, dtype="float32")


def build_training_rows(rows: list[dict[str, Any]], copies: int) -> list[dict[str, Any]]:
    """元データ + 拡張コピー。aug_seed<0 の行は拡張しない。"""
    expanded = [{**row, "aug_seed": -1} for row in rows]
    for copy_index in range(max(0, copies)):
        for row_index, row in enumerate(rows):
            expanded.append({**row, "aug_seed": copy_index * 100003 + row_index + 1})
    return expanded


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

    mode = str(args.finetune_mode)
    freeze_enc = resolve_freeze_encoder(mode, args.freeze_encoder)
    learning_rate = resolve_learning_rate(mode, args.lr)

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    base_model = resolve_base_model(args)
    print(f"[train] base model: {base_model}")
    print(f"[train] finetune mode: {mode} (lr={learning_rate:g})")

    processor = WhisperProcessor.from_pretrained(base_model, language=args.language, task=args.task)

    rows = load_manifest_rows(args.manifest)
    if not rows:
        raise SystemExit(f"学習manifestが空です: {args.manifest}")
    training_rows = build_training_rows(rows, args.augment_copies)
    print(
        f"[train] {len(rows)} training examples "
        f"(+{len(training_rows) - len(rows)} augmented copies)"
    )

    dataset = Dataset.from_list(training_rows)

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    max_label_len = args.max_label_len

    # datasets の Audio 特徴量デコードは新しめの版で torchcodec/ffmpeg を要求し
    # map() が落ちることがある。soundfile + librosa で自前デコードして回避する
    # (16kHzモノラルへ変換)。
    import librosa
    import soundfile as sf

    def load_16k_mono(path: str) -> "np.ndarray":
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return np.ascontiguousarray(wav, dtype="float32")

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        wav = load_16k_mono(batch["audio"])
        if int(batch.get("aug_seed", -1)) >= 0:
            wav = augment_waveform(
                wav,
                16000,
                int(batch["aug_seed"]) + args.seed,
                speed=args.augment_speed,
                noise_snr_db=args.augment_noise_snr_db,
                gain_db=args.augment_gain_db,
            )
        batch["input_features"] = feature_extractor(
            wav, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = truncate_label_ids(
            tokenizer(batch["text"]).input_ids,
            max_label_len,
            tokenizer.eos_token_id,
        )
        return batch

    dataset = dataset.map(prepare, remove_columns=dataset.column_names, num_proc=1)

    # 学習中の足切り用 dev セット (少数の発話を毎エポック復号して CER を見る)。
    eval_dataset = None
    if args.dev_manifest is not None and args.dev_manifest.exists():
        dev_rows = load_manifest_rows(args.dev_manifest, limit=args.dev_limit)
        dev_rows = [{**row, "aug_seed": -1} for row in dev_rows]
        if dev_rows:
            eval_dataset = Dataset.from_list(dev_rows).map(
                prepare, remove_columns=["audio", "text", "aug_seed"], num_proc=1
            )
            print(f"[train] in-training dev set: {len(dev_rows)} utterances")

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    # 日本語 transcribe を強制し、言語自動判定によるブレを防ぐ。
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # 学習中は gradient checkpointing と両立しないので KV キャッシュを切るが、
    # 学習中 dev 評価の generate ではキャッシュを使わないと極端に遅くなる。
    model.config.use_cache = False
    model.generation_config.use_cache = True
    # SpecAugment: エンコーダのメル特徴に時間/周波数マスクを掛ける (学習時のみ有効)。
    if args.mask_time_prob > 0 or args.mask_feature_prob > 0:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
        model.config.mask_feature_prob = args.mask_feature_prob
        print(
            f"[train] SpecAugment on (time={args.mask_time_prob}, feature={args.mask_feature_prob})"
        )
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        # 凍結ベースへ gradient checkpointing 越しに勾配を渡すため、LoRA のみ必要。
        model.enable_input_require_grads()
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        print(f"[train] LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if freeze_enc:
            freeze_encoder(model)
            frozen_encoder_params = verify_encoder_frozen(model)
            print(f"[train] verified frozen encoder: {frozen_encoder_params:,} parameters")
        trainable_params, total_params = count_parameters(model)
        print(
            f"[train] full fine-tune (encoder {'frozen' if freeze_enc else 'trainable'}): "
            f"trainable params: {trainable_params:,} || all params: {total_params:,} || "
            f"trainable%: {100.0 * trainable_params / max(1, total_params):.4f}"
        )

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    progress_path = args.out_dir / "train_progress.json"
    policy = EarlyStopPolicy(args.early_stop_cer, args.early_stop_patience)
    stop_reason = ""

    def compute_metrics(eval_preds: Any) -> dict[str, float]:
        predictions, labels = eval_preds.predictions, eval_preds.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        hyps = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rates = [
            error_rate(list(normalize_text(ref)), list(normalize_text(hyp)))
            for ref, hyp in zip(refs, hyps)
        ]
        return {"cer": sum(rates) / len(rates) if rates else 1.0}

    class EarlyStopCallback(TrainerCallback):
        """毎エポックの dev CER を見て、見込みがなければ学習を打ち切る。"""

        def on_evaluate(self, args_, state, control, metrics=None, **kwargs):  # noqa: ANN001
            nonlocal stop_reason
            cer = (metrics or {}).get("eval_cer")
            if cer is None:
                return control
            should_stop, reason = policy.update(cer)
            print(
                f"[train] epoch {state.epoch:.2f}: dev CER={cer:.4f} "
                f"(best={policy.best:.4f}, threshold={policy.threshold or 0:.4f})",
                flush=True,
            )
            if should_stop:
                stop_reason = reason
                print(f"[train] early stop: {reason}", flush=True)
                control.should_training_stop = True
            return control

    class TimeBudgetCallback(TrainerCallback):
        """実時間で学習を打ち切る。設定ごとの 1 ステップの重さの違いを吸収して、
        代理試行の「同じ計算量での比較」を成立させる。"""

        def __init__(self, limit_sec: float) -> None:
            self.limit_sec = limit_sec
            self.started = time.monotonic()
            self.hit = False

        def on_step_end(self, args_, state, control, **kwargs):  # noqa: ANN001
            if not self.hit and time.monotonic() - self.started >= self.limit_sec:
                self.hit = True
                elapsed = time.monotonic() - self.started
                print(
                    f"[train] time budget reached ({elapsed:.0f}s >= {self.limit_sec:.0f}s) "
                    f"at step {state.global_step}; stopping",
                    flush=True,
                )
                control.should_training_stop = True
            return control

    adapter_dir = args.out_dir / "adapter"
    cuda_available = bool(torch.cuda.is_available())
    bf16_supported = bool(
        cuda_available
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16, use_bf16 = resolve_mixed_precision(
        args.mixed_precision,
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
    )
    print(
        "[train] mixed precision: "
        + ("bf16" if use_bf16 else "fp16" if use_fp16 else "disabled")
    )
    has_dev = eval_dataset is not None
    eval_strategy, effective_save_strategy, load_best_model = resolve_checkpoint_policy(
        has_dev, args.save_strategy
    )
    if has_dev and args.save_strategy != "epoch":
        print("[train] dev evaluation requires epoch checkpoints; overriding --save-strategy to epoch")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        lr_scheduler_type=args.lr_scheduler,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy=effective_save_strategy,
        save_total_limit=1,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="cer" if has_dev else None,
        greater_is_better=False if has_dev else None,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        seed=args.seed,
        eval_strategy=eval_strategy,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        predict_with_generate=eval_dataset is not None,
        generation_max_length=args.max_label_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )
    if eval_dataset is not None:
        trainer.add_callback(EarlyStopCallback())
    time_budget = None
    if args.max_train_seconds > 0:
        time_budget = TimeBudgetCallback(args.max_train_seconds)
        trainer.add_callback(time_budget)
        print(f"[train] wall-clock training budget: {args.max_train_seconds:.0f}s")
    trainer.train()
    if has_dev and trainer.state.best_model_checkpoint:
        print(
            "[train] restored best dev-CER checkpoint: "
            f"{trainer.state.best_model_checkpoint} (CER={trainer.state.best_metric})"
        )

    if trainer.is_world_process_zero():
        # 探索ループはこのファイルを読んで、試行が早期打ち切りされたかを記録する。
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "finetune_mode": mode,
                    "freeze_encoder": freeze_enc,
                    "learning_rate": learning_rate,
                    "mixed_precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "disabled",
                    "dev_cer_curve": policy.history,
                    "best_dev_cer": policy.best,
                    "best_model_checkpoint": trainer.state.best_model_checkpoint,
                    "best_model_metric": trainer.state.best_metric,
                    "early_stopped": bool(stop_reason),
                    "early_stop_reason": stop_reason,
                    "epochs_requested": args.epochs,
                    "epochs_completed": trainer.state.epoch,
                    "steps_completed": trainer.state.global_step,
                    "time_budget_sec": args.max_train_seconds or None,
                    "time_budget_hit": bool(time_budget and time_budget.hit),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if mode == "lora":
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
                merged = PeftModel.from_pretrained(base, str(adapter_dir))
                merged = merged.merge_and_unload()
                args.merge_dir.mkdir(parents=True, exist_ok=True)
                merged.save_pretrained(args.merge_dir)
                processor.save_pretrained(args.merge_dir)
                print(f"[train] saved merged model -> {args.merge_dir}")
        else:
            # full-FT の出力はそれ自体がマージ済みモデル。3GB級なので二重に置かず、
            # --merge-dir があればそこへ、無ければ <out-dir>/model へ 1 部だけ保存する。
            model.config.use_cache = True
            model_dir = args.merge_dir if args.merge_dir is not None else args.out_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
            print(f"[train] saved fine-tuned model -> {model_dir}")


if __name__ == "__main__":
    main()
