#!/usr/bin/env python3
"""共有された test wav 群 -> whisper-streaming 評価 manifest。

evaluate_whisper_streaming.py は 1行 {"id","wav","sentence"} を読み、sentence を
参照(正解)として CER/WER を出す。ここでは wav ディレクトリと、任意の参照TSVから
その manifest を組む。

参照TSV (--references) はタブ区切り 2列:  <key>\t<reference_text>
  key は wav のファイル名(拡張子有無どちらも可) か id。ヘッダ行があってもよい。
参照が無い wav は sentence="" で出力する (推論のみ、指標は無意味になる)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="test wav群を評価manifestへ変換")
    parser.add_argument("--wav-dir", type=Path, required=True, help="wavディレクトリ (再帰探索)")
    parser.add_argument("--references", type=Path, default=None, help="key<TAB>text の参照TSV (任意)")
    parser.add_argument("--out", type=Path, required=True, help="出力 manifest.jsonl")
    return parser.parse_args()


def load_references(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    refs: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        key = cols[0].strip()
        text = cols[1].strip()
        if key.lower() in ("id", "key", "filename", "wav"):  # ヘッダ行
            continue
        # ファイル名・stem・そのままの3通りで引けるように登録
        refs[key] = text
        refs[Path(key).name] = text
        refs[Path(key).stem] = text
    return refs


def main() -> None:
    args = parse_args()
    refs = load_references(args.references)
    wavs = sorted(p for p in args.wav_dir.rglob("*.wav") if p.is_file())
    if not wavs:
        raise SystemExit(f"wav が見つかりません: {args.wav_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for wav in wavs:
            ref = refs.get(wav.name) or refs.get(wav.stem) or ""
            if ref:
                matched += 1
            fh.write(
                json.dumps(
                    {"id": wav.stem, "wav": str(wav.resolve()), "sentence": ref},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(wavs)} records -> {args.out} (with_reference={matched})")
    if refs and matched == 0:
        print("[WARN] 参照TSVを渡したが1件も一致しませんでした。key(ファイル名/stem)を確認してください。")


if __name__ == "__main__":
    main()
