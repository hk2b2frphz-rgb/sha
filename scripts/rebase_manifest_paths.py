#!/usr/bin/env python3
"""manifest の audio パスを別マシンのディレクトリに貼り替える。

build_whisper_manifest.py は audio を絶対パスで書く。同じマシンで学習するなら
それでよいが、CPU コンテナで音声を作って GPU を借りて学習する構成では、借りた
マシン上に同じ絶対パスは存在しない。この差だけを埋める。

WAV は basename で突き合わせる (build_whisper_manifest.py が id 由来の一意な
ファイル名を付けるため)。見つからない行があれば既定で失敗する
--allow-missing を付けた場合のみ、その行を落として続行する。

Usage:
    python scripts/rebase_manifest_paths.py \
        --manifest train_manifest.jsonl --root /workspace/data --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def index_wavs(root: Path) -> dict[str, Path]:
    """root 以下の *.wav を basename -> path で引けるようにする。"""
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in root.rglob("*.wav"):
        if path.name in index:
            duplicates.add(path.name)
        index[path.name] = path
    if duplicates:
        raise SystemExit(
            f"{root} に同名の WAV が複数あります: {sorted(duplicates)[:5]}"
            " -- basename での突き合わせができません"
        )
    return index


def rebase_rows(
    rows: list[dict], index: dict[str, Path], *, allow_missing: bool
) -> tuple[list[dict], list[str]]:
    out: list[dict] = []
    missing: list[str] = []
    for row in rows:
        name = Path(str(row.get("audio", ""))).name
        found = index.get(name)
        if found is None:
            missing.append(name or row.get("id", "<no id>"))
            continue
        out.append({**row, "audio": str(found.resolve())})
    if missing and not allow_missing:
        raise SystemExit(
            f"{len(missing)} 件の WAV が見つかりません (例: {missing[:5]})。"
            " --allow-missing で除外して続行できます。"
        )
    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="WAV を探すディレクトリ")
    ap.add_argument("--out", type=Path, default=None, help="出力先 (既定は標準出力)")
    ap.add_argument("--in-place", action="store_true", help="--manifest を直接書き換える")
    ap.add_argument("--allow-missing", action="store_true", help="欠損行を落として続行")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"--root がディレクトリではありません: {args.root}")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"manifest が空です: {args.manifest}")

    rebased, missing = rebase_rows(rows, index_wavs(args.root), allow_missing=args.allow_missing)
    if not rebased:
        raise SystemExit("貼り替えられた行がありません")

    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rebased)
    target = args.manifest if args.in_place else args.out
    if target is None:
        sys.stdout.write(text)
    else:
        target.write_text(text, encoding="utf-8")

    note = f" (dropped {len(missing)})" if missing else ""
    print(f"rebased {len(rebased)}/{len(rows)} rows -> {target or 'stdout'}{note}", file=sys.stderr)


if __name__ == "__main__":
    main()
