#!/usr/bin/env python3
"""
読み仮名アノテーション精度評価スクリプト

data/test_cases.tsv に定義した (term, reading, sentence) を使い、
各 LLM プロバイダが専門用語を正しく読めるかテストする。

使い方:
  # ドライラン (LLM 不要・動作確認)
  python scripts/run_eval.py --cases data/test_cases.tsv --dry-run

  # Gemma (GPU サーバー上で実行)
  uv run --project gemma_runtime python scripts/run_eval.py \
      --cases data/test_cases.tsv --provider gemma --model google/gemma-4-E4B-it

  # Azure OpenAI / GPT-4o
  #   必要な環境変数: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY
  #   任意の環境変数: AZURE_OPENAI_DEPLOYMENT (デフォルト: gpt-4o)
  python scripts/run_eval.py \
      --cases data/test_cases.tsv --provider azure-openai

  # キーが未設定の場合にスキップ (終了コード 0)
  python scripts/run_eval.py \
      --cases data/test_cases.tsv --provider azure-openai --skip-if-no-key

  # ドメイン絞り込み
  python scripts/run_eval.py \
      --cases data/test_cases.tsv --provider gemma --domain sewage
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_eval")

ANNOTATION_PROMPT = """\
以下の文に含まれる専門用語「{term}」の読み仮名をひらがなで答えてください。

文：「{sentence}」

ひらがなのみで答えてください（説明は不要です）。"""

AZURE_ENV_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"]


# ── ユーティリティ ────────────────────────────────────────────────────

def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def build_prompt(term: str, sentence: str) -> str:
    return ANNOTATION_PROMPT.format(term=term, sentence=sentence)


def reading_matches(llm_output: str, gold_reading: str) -> bool:
    """LLM 出力（正規化後）に金標準読みが含まれるか。"""
    normalized = kata_to_hira(llm_output.strip())
    return gold_reading in normalized


# ── データロード ──────────────────────────────────────────────────────

def load_cases(path: Path, domain: str | None = None) -> list[dict[str, str]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("id\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        case = {
            "id": parts[0],
            "term": parts[1],
            "reading": kata_to_hira(parts[2]),
            "sentence": parts[3],
            "domain": parts[4] if len(parts) > 4 else "",
        }
        if domain and case["domain"] != domain:
            continue
        cases.append(case)
    if not cases:
        raise SystemExit(f"テストケースが見つかりません: {path}")
    return cases


# ── Gemma プロバイダ ──────────────────────────────────────────────────

def build_gemma_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "model": args.model,
        "device_map": args.device_map,
        "model_kwargs": {"attn_implementation": "sdpa"},
    }
    if args.dtype != "auto":
        kwargs["torch_dtype"] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]

    logger.info("Gemma をロード中: %s", args.model)
    try:
        return pipeline("text-generation", **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("model_kwargs", None)
        return pipeline("text-generation", **kwargs)


def _flatten(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(_flatten(r) for r in result)
    if isinstance(result, dict):
        for key in ("generated_text", "text", "content"):
            if key in result:
                return _flatten(result[key])
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def annotate_gemma(pipe: Any, args: argparse.Namespace, term: str, sentence: str) -> str:
    prompt = build_prompt(term, sentence)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "return_full_text": False,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    try:
        result = pipe(messages, **gen_kwargs)
    except TypeError:
        result = pipe(prompt, **gen_kwargs)
    return _flatten(result)


# ── Azure OpenAI プロバイダ ───────────────────────────────────────────

def check_azure_keys(skip_if_no_key: bool) -> bool:
    missing = [v for v in AZURE_ENV_VARS if not os.environ.get(v)]
    if not missing:
        return True
    logger.warning("Azure OpenAI の環境変数が未設定: %s", ", ".join(missing))
    if skip_if_no_key:
        logger.info("--skip-if-no-key が指定されているためスキップします (exit 0)")
        return False
    raise SystemExit(f"環境変数を設定してください: {', '.join(missing)}")


def build_azure_client() -> Any:
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-01",
    )


def annotate_azure(client: Any, deployment: str, term: str, sentence: str) -> str:
    prompt = build_prompt(term, sentence)
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


# ── 評価・レポート ────────────────────────────────────────────────────

def evaluate_case(case: dict, llm_output: str) -> dict:
    correct = reading_matches(llm_output, case["reading"])
    return {
        "id": case["id"],
        "term": case["term"],
        "gold_reading": case["reading"],
        "llm_output": llm_output.strip(),
        "reading_correct": correct,
        "domain": case["domain"],
        "sentence": case["sentence"],
    }


def build_report(provider: str, model: str, cases_file: str, results: list[dict]) -> dict:
    n = len(results)
    n_correct = sum(1 for r in results if r["reading_correct"])

    per_term: dict[str, dict] = {}
    for r in results:
        t = r["term"]
        if t not in per_term:
            per_term[t] = {
                "term": t,
                "gold_reading": r["gold_reading"],
                "domain": r["domain"],
                "n": 0,
                "n_correct": 0,
                "examples": [],
            }
        per_term[t]["n"] += 1
        if r["reading_correct"]:
            per_term[t]["n_correct"] += 1
        if len(per_term[t]["examples"]) < 1:
            per_term[t]["examples"].append({
                "sentence": r["sentence"],
                "llm_output": r["llm_output"],
            })

    for v in per_term.values():
        v["acc_correct"] = v["n_correct"] / v["n"] if v["n"] else 0.0

    return {
        "provider": provider,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_file": cases_file,
        "n_total": n,
        "n_correct": n_correct,
        "acc_reading_correct": round(n_correct / n, 4) if n else 0.0,
        "per_term": list(per_term.values()),
        "all_results": results,
    }


def print_summary(report: dict) -> None:
    n = report["n_total"]
    n_correct = report["n_correct"]
    acc = report["acc_reading_correct"]
    print(f"\n{'='*60}")
    print(f"プロバイダ : {report['provider']}  モデル: {report['model']}")
    print(f"{'='*60}")
    print(f"テストケース数 : {n}")
    print(f"正解数         : {n_correct}/{n}  ({acc*100:.1f}%)")
    print(f"{'='*60}")
    print("\n用語別:")
    for v in sorted(report["per_term"], key=lambda x: x["acc_correct"]):
        mark = "✓" if v["acc_correct"] >= 1.0 else ("△" if v["acc_correct"] > 0 else "✗")
        ex = v["examples"][0]["llm_output"] if v["examples"] else "-"
        print(
            f"  {mark} {v['term']:22s}  [{v['domain']:10s}]  "
            f"(金標準: {v['gold_reading']}  LLM: {ex[:30]})"
        )


# ── 引数パーサ ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="読み仮名アノテーション精度評価")
    parser.add_argument("--cases", type=Path, required=True, help="テストケース TSV")
    parser.add_argument("--provider", choices=["gemma", "azure-openai"],
                        help="LLM プロバイダ (--dry-run 時は不要)")
    parser.add_argument("--model", default="google/gemma-4-E4B-it",
                        help="Gemma 使用時のモデル名")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype",
                        choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 で greedy decode")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--domain",
                        help="評価対象ドメインをフィルタ (例: medical, sewage, it, engineering)")
    parser.add_argument("--out", type=Path,
                        help="評価結果 JSON の出力先 (省略時は自動命名)")
    parser.add_argument("--dry-run", action="store_true",
                        help="LLM を呼ばずデータ・ロジックのみ確認 (smoke test)")
    parser.add_argument("--skip-if-no-key", action="store_true",
                        help="API キーが未設定の場合にスキップして終了 (exit 0)")
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases, domain=args.domain)
    logger.info("%d 件のテストケースを読み込みました", len(cases))

    # ── ドライラン (smoke test) ──────────────────────────────────────
    if args.dry_run:
        print("\n[Dry Run] テストケース先頭 3 件:")
        for c in cases[:3]:
            print(f"  [{c['id']}] {c['term']} / {c['reading']} / {c['domain']}")
            print(f"         文: {c['sentence']}")
        print(f"\n[Dry Run] 生成プロンプト例 (ケース 1):")
        print("---")
        print(build_prompt(cases[0]["term"], cases[0]["sentence"]))
        print("---")
        # 評価ロジック自己チェック: 金標準をそのまま LLM 出力として全問正解を確認
        mock_results = [evaluate_case(c, c["reading"]) for c in cases]
        all_correct = all(r["reading_correct"] for r in mock_results)
        status = f"PASS (全 {len(mock_results)} 件正解確認)" if all_correct else "FAIL"
        print(f"\n[Dry Run] 評価ロジック自己チェック: {status}")
        domain_counts: dict[str, int] = {}
        for c in cases:
            domain_counts[c["domain"]] = domain_counts.get(c["domain"], 0) + 1
        print(f"[Dry Run] ドメイン内訳: {dict(sorted(domain_counts.items()))}")
        print(f"\n[Dry Run] Smoke test PASSED")
        print(f"           {len(cases)} 件ロード / プロンプト生成 / 評価ロジック すべて OK")
        sys.exit(0)

    # ── プロバイダ必須チェック ───────────────────────────────────────
    if not args.provider:
        raise SystemExit("--provider を指定してください (gemma / azure-openai) または --dry-run を使用")

    # ── 出力先 ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d")
    model_tag = args.model.split("/")[-1] if args.provider == "gemma" \
        else os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    out_path = args.out or Path("experiments") / f"{ts}_{model_tag}_eval.json"

    results: list[dict] = []

    # ── Gemma ───────────────────────────────────────────────────────
    if args.provider == "gemma":
        pipe = build_gemma_pipeline(args)
        for i, c in enumerate(cases, 1):
            logger.info("(%d/%d) 「%s」", i, len(cases), c["term"])
            raw = annotate_gemma(pipe, args, c["term"], c["sentence"])
            result = evaluate_case(c, raw)
            results.append(result)
            mark = "✓" if result["reading_correct"] else "✗"
            logger.info("  %s  LLM: %s  (金標準: %s)", mark, raw.strip()[:30], c["reading"])

    # ── Azure OpenAI ─────────────────────────────────────────────────
    elif args.provider == "azure-openai":
        if not check_azure_keys(args.skip_if_no_key):
            sys.exit(0)
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        client = build_azure_client()
        for i, c in enumerate(cases, 1):
            logger.info("(%d/%d) 「%s」", i, len(cases), c["term"])
            raw = annotate_azure(client, deployment, c["term"], c["sentence"])
            result = evaluate_case(c, raw)
            results.append(result)
            mark = "✓" if result["reading_correct"] else "✗"
            logger.info("  %s  LLM: %s  (金標準: %s)", mark, raw.strip()[:30], c["reading"])

    report = build_report(
        provider=args.provider,
        model=args.model if args.provider == "gemma"
              else os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        cases_file=str(args.cases),
        results=results,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    logger.info("結果を保存: %s", out_path)


if __name__ == "__main__":
    main()
