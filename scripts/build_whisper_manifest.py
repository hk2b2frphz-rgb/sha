#!/usr/bin/env python3
"""TTS合成 manifest + TTS入力JSONL -> Whisper学習 manifest。

synthesize_speech.py は manifest.jsonl の "sentence" にひらがな版(tts_text)を
書き込むため、そのままでは学習ターゲット(元の漢字文)が失われる。ここで
prepare_train_from_csv.py が出した tts_input.jsonl を id で join し直し、
音声パスと「元の漢字文」を対応付けた学習用 manifest を作る。

出力JSONL 1行:
  {"id", "audio", "text", "duration_sec"}
    audio : wav の絶対パス
    text  : 元の Sentance (漢字) = ASR学習の正解テキスト
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS合成結果を Whisper学習 manifest へ変換")
    parser.add_argument("--tts-input", type=Path, required=True, help="prepare_train_from_csv.py の出力JSONL")
    parser.add_argument(
        "--synth-dir",
        type=Path,
        action="append",
        required=True,
        help="synthesize_speech.py の --out-dir。シャード分だけ複数回指定可",
    )
    parser.add_argument("--out", type=Path, required=True, help="学習 manifest JSONL")
    return parser.parse_args()


def load_targets(path: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        targets[str(rec["id"])] = str(rec["sentence"])
    if not targets:
        raise SystemExit(f"tts_input が空です: {path}")
    return targets


def main() -> None:
    args = parse_args()
    targets = load_targets(args.tts_input)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_target = 0
    missing_wav = 0
    seen: set[str] = set()
    with args.out.open("w", encoding="utf-8") as out_fh:
        for synth_dir in args.synth_dir:
            manifest = synth_dir / "manifest.jsonl"
            if not manifest.exists():
                raise SystemExit(f"合成manifestが見つかりません: {manifest}")
            for raw in manifest.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec_id = str(rec["id"])
                if rec_id in seen:
                    continue
                text = targets.get(rec_id)
                if text is None:
                    missing_target += 1
                    continue
                wav = Path(str(rec["wav"]))
                wav_path = wav if wav.is_absolute() else synth_dir / wav
                if not wav_path.exists():
                    missing_wav += 1
                    continue
                out_fh.write(
                    json.dumps(
                        {
                            "id": rec_id,
                            "audio": str(wav_path.resolve()),
                            "text": text,
                            "duration_sec": rec.get("duration_sec"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                seen.add(rec_id)
                written += 1

    if not written:
        raise SystemExit("学習manifestが空になりました。id/wavの対応を確認してください。")
    print(
        f"wrote {written} records -> {args.out} "
        f"(missing_target={missing_target}, missing_wav={missing_wav})"
    )


if __name__ == "__main__":
    main()
