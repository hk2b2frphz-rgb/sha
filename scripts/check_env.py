#!/usr/bin/env python3
"""
環境診断スクリプト。GPU マシンで実行するだけで、CUDA / torch / qwen-tts /
gemma_runtime をステップごとに確認し、最後に診断結果をまとめて表示する。

依存パッケージ不要 (標準ライブラリのみ)。環境が壊れていても動くように、
各チェックはサブプロセスで実行する。

使い方 (リポジトリ直下で):
  python3 scripts/check_env.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

TIMEOUT_SEC = 300  # uv sync 直後の初回 import はモデル DL 等で遅いことがある


@dataclass
class StepResult:
    name: str
    ok: bool
    output: str


RESULTS: list[StepResult] = []


def run(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC
        )
    except FileNotFoundError:
        return False, f"コマンドが見つかりません: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, f"タイムアウト ({TIMEOUT_SEC}s)"
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


def step(name: str, cmd: list[str], note: str = "") -> StepResult:
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    if note:
        print(f"  ({note})")
    print(f"  $ {' '.join(cmd)}")
    print("-" * 60)
    ok, out = run(cmd)
    print(out if out else "(出力なし)")
    print(f"--> {'OK' if ok else 'NG'}")
    result = StepResult(name, ok, out)
    RESULTS.append(result)
    return result


def main() -> None:
    print("term2speech 環境診断を開始します")

    # ---- 1. GPU とドライバ --------------------------------------------
    step(
        "1. GPU / ドライバ確認 (nvidia-smi)",
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
    )

    if shutil.which("uv") is None:
        print("\nuv が見つかりません。先に uv をインストールしてください。")
        sys.exit(1)

    # ---- 2. メイン環境 (Qwen3-TTS 用) ---------------------------------
    torch_check = (
        "import torch;"
        "print('torch', torch.__version__);"
        "print('built for CUDA', torch.version.cuda);"
        "print('cuda available:', torch.cuda.is_available());"
        "print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-');"
        "cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None;"
        "print('compute capability:', cap);"
        "print('bf16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else '-')"
    )
    step("2. メイン環境: torch", ["uv", "run", "python", "-c", torch_check])

    step(
        "3. メイン環境: qwen_tts の import",
        ["uv", "run", "python", "-c",
         "from qwen_tts import Qwen3TTSModel; print('qwen_tts import OK')"],
        note="libcudart エラーはここで出ることが多い",
    )

    step(
        "4. メイン環境: nvidia/torch 系パッケージ一覧",
        ["uv", "pip", "list"],
    )

    # ---- 3. gemma_runtime 環境 ----------------------------------------
    step(
        "5. gemma_runtime: torch",
        ["uv", "run", "--project", "gemma_runtime", "python", "-c", torch_check],
    )

    step(
        "6. gemma_runtime: transformers の import",
        ["uv", "run", "--project", "gemma_runtime", "python", "-c",
         "import transformers; print('transformers', transformers.__version__)"],
    )

    # ---- 4. 診断まとめ -------------------------------------------------
    print(f"\n{'=' * 60}")
    print("診断まとめ")
    print("=" * 60)
    for r in RESULTS:
        print(f"  [{'OK' if r.ok else 'NG'}] {r.name}")

    all_out = "\n".join(r.output for r in RESULTS)
    findings: list[str] = []

    if "libcudart.so.13" in all_out:
        findings.append(
            "libcudart.so.13 (CUDA 13) を要求するパッケージが混入しています。\n"
            "    CUDA 13 は V100 非対応のため、cu12 系へ揃える必要があります。\n"
            "    STEP 4 の一覧に 'cu13' を含むパッケージがないか確認してください。"
        )
    elif "libcudart" in all_out and any(not r.ok for r in RESULTS):
        findings.append(
            "libcudart 関連のエラーが出ています。NG になった STEP の出力を確認してください。"
        )

    if "V100" in all_out and "bf16 supported: False" in all_out:
        findings.append(
            "V100 は bfloat16 非対応です。synthesize_speech.py は --dtype float16 を付けて実行してください。"
        )

    if "cuda available: False" in all_out:
        findings.append(
            "torch から GPU が見えていません。ドライバと torch の CUDA バージョンの組み合わせを確認してください。"
        )

    if not findings and all(r.ok for r in RESULTS):
        findings.append("問題は見つかりませんでした。パイプラインを実行できます。")

    print()
    for f in findings:
        print(f"  * {f}")

    print(
        "\nNG がある場合は、このスクリプトの全出力をコピーして共有してください。"
    )
    sys.exit(0 if all(r.ok for r in RESULTS) else 1)


if __name__ == "__main__":
    main()
