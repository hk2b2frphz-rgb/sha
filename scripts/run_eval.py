#!/usr/bin/env python3
"""Evaluate Japanese reading annotation accuracy for LLM providers."""
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

文: {sentence}

ひらがなのみで答えてください。説明は不要です。
"""

AZURE_ENV_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"]


def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in text)


def build_prompt(term: str, sentence: str) -> str:
    return ANNOTATION_PROMPT.format(term=term, sentence=sentence)


def build_batch_prompt(cases: list[dict[str, str]]) -> str:
    payload = [
        {"id": case["id"], "term": case["term"], "sentence": case["sentence"]}
        for case in cases
    ]
    return (
        "For each item, estimate the Japanese reading of term in hiragana only.\n"
        "Return JSON only, with this exact shape:\n"
        '{"results":[{"id":"1","reading":"ひらがな"}]}\n'
        "Do not include explanations, markdown, or extra keys.\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def reading_matches(llm_output: str, gold_reading: str) -> bool:
    normalized = kata_to_hira(llm_output.strip())
    return gold_reading in normalized


def chunked(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [items[index : index + size] for index in range(0, len(items), size)]


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    object_start = stripped.find("{")
    array_start = stripped.find("[")
    starts = [index for index in (object_start, array_start) if index >= 0]
    if not starts:
        raise ValueError("no JSON object or array found in model output")
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end < start:
        raise ValueError("unterminated JSON in model output")
    return stripped[start : end + 1]


def parse_batch_readings(raw: str) -> dict[str, str]:
    parsed = json.loads(_extract_json_text(raw))
    rows = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        raise ValueError("batch output must be a list or an object with a results list")

    readings: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id", "")).strip()
        reading = str(row.get("reading", "")).strip()
        if case_id and reading:
            readings[case_id] = kata_to_hira(reading)
    return readings


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
        return "\n".join(_flatten(row) for row in result)
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


def check_azure_keys(skip_if_no_key: bool) -> bool:
    missing = [name for name in AZURE_ENV_VARS if not os.environ.get(name)]
    if not missing:
        return True
    logger.warning("Azure OpenAI environment variables are missing: %s", ", ".join(missing))
    if skip_if_no_key:
        logger.info("--skip-if-no-key was set; skipping Azure OpenAI evaluation")
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


def annotate_azure_batch(client: Any, deployment: str, args: argparse.Namespace, cases: list[dict[str, str]]) -> dict[str, str]:
    prompt = build_batch_prompt(cases)
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max(args.max_new_tokens, len(cases) * 32),
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    return parse_batch_readings(raw)


def evaluate_case(case: dict[str, str], llm_output: str) -> dict[str, Any]:
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


def build_report(provider: str, model: str, cases_file: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    n_correct = sum(1 for row in results if row["reading_correct"])

    per_term: dict[str, dict[str, Any]] = {}
    for row in results:
        term = row["term"]
        if term not in per_term:
            per_term[term] = {
                "term": term,
                "gold_reading": row["gold_reading"],
                "domain": row["domain"],
                "n": 0,
                "n_correct": 0,
                "examples": [],
            }
        per_term[term]["n"] += 1
        if row["reading_correct"]:
            per_term[term]["n_correct"] += 1
        if len(per_term[term]["examples"]) < 1:
            per_term[term]["examples"].append({
                "sentence": row["sentence"],
                "llm_output": row["llm_output"],
            })

    for row in per_term.values():
        row["acc_correct"] = row["n_correct"] / row["n"] if row["n"] else 0.0

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


def print_summary(report: dict[str, Any]) -> None:
    n = report["n_total"]
    n_correct = report["n_correct"]
    acc = report["acc_reading_correct"]
    print(f"\n{'=' * 60}")
    print(f"プロバイダ: {report['provider']}  モデル: {report['model']}")
    print(f"{'=' * 60}")
    print(f"テストケース数: {n}")
    print(f"正解数: {n_correct}/{n}  ({acc * 100:.1f}%)")
    print(f"{'=' * 60}")
    print("\n用語別:")
    for row in sorted(report["per_term"], key=lambda item: item["acc_correct"]):
        mark = "OK" if row["acc_correct"] >= 1.0 else ("PART" if row["acc_correct"] > 0 else "NG")
        example = row["examples"][0]["llm_output"] if row["examples"] else "-"
        print(
            f"  {mark:4s} {row['term']:22s}  [{row['domain']:10s}]  "
            f"(gold: {row['gold_reading']}  LLM: {example[:30]})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="読み仮名アノテーション精度評価")
    parser.add_argument("--cases", type=Path, required=True, help="テストケース TSV")
    parser.add_argument("--provider", choices=["gemma", "azure-openai"], help="LLM provider")
    parser.add_argument("--model", default="google/gemma-4-E4B-it", help="Gemma model name")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 uses greedy decode")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=10, help="Azure OpenAI/GPT cases per request")
    parser.add_argument("--domain", help="Filter evaluation domain, e.g. medical, sewage, it, engineering")
    parser.add_argument("--out", type=Path, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Check data and evaluation logic without LLM calls")
    parser.add_argument("--skip-if-no-key", action="store_true", help="Exit 0 when API keys are missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases, domain=args.domain)
    logger.info("%d test cases loaded", len(cases))

    if args.dry_run:
        print("\n[Dry Run] First 3 cases:")
        for case in cases[:3]:
            print(f"  [{case['id']}] {case['term']} / {case['reading']} / {case['domain']}")
            print(f"         sentence: {case['sentence']}")
        print("\n[Dry Run] Prompt example:")
        print("---")
        print(build_prompt(cases[0]["term"], cases[0]["sentence"]))
        print("---")
        mock_results = [evaluate_case(case, case["reading"]) for case in cases]
        all_correct = all(row["reading_correct"] for row in mock_results)
        status = f"PASS ({len(mock_results)} correct)" if all_correct else "FAIL"
        domain_counts: dict[str, int] = {}
        for case in cases:
            domain_counts[case["domain"]] = domain_counts.get(case["domain"], 0) + 1
        print(f"\n[Dry Run] Evaluation logic: {status}")
        print(f"[Dry Run] Domain counts: {dict(sorted(domain_counts.items()))}")
        sys.exit(0)

    if not args.provider:
        raise SystemExit("--provider を指定してください (gemma / azure-openai) または --dry-run を使用")

    timestamp = datetime.now().strftime("%Y-%m-%d")
    model_tag = args.model.split("/")[-1] if args.provider == "gemma" else os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    out_path = args.out or Path("experiments") / f"{timestamp}_{model_tag}_eval.json"

    results: list[dict[str, Any]] = []
    if args.provider == "gemma":
        pipe = build_gemma_pipeline(args)
        for index, case in enumerate(cases, 1):
            logger.info("(%d/%d) %s", index, len(cases), case["term"])
            raw = annotate_gemma(pipe, args, case["term"], case["sentence"])
            result = evaluate_case(case, raw)
            results.append(result)
            mark = "OK" if result["reading_correct"] else "NG"
            logger.info("  %s  LLM: %s  (gold: %s)", mark, raw.strip()[:30], case["reading"])

    elif args.provider == "azure-openai":
        if not check_azure_keys(args.skip_if_no_key):
            sys.exit(0)
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        client = build_azure_client()
        batches = chunked(cases, args.batch_size)
        for batch_index, batch in enumerate(batches, 1):
            logger.info("batch %d/%d (%d cases)", batch_index, len(batches), len(batch))
            batch_readings = annotate_azure_batch(client, deployment, args, batch)
            missing_ids = [case["id"] for case in batch if case["id"] not in batch_readings]
            if missing_ids:
                raise RuntimeError(f"Azure OpenAI batch response missing ids: {', '.join(missing_ids)}")
            for case in batch:
                raw = batch_readings[case["id"]]
                result = evaluate_case(case, raw)
                results.append(result)
                mark = "OK" if result["reading_correct"] else "NG"
                logger.info("  %s  %s  LLM: %s  (gold: %s)", mark, case["term"], raw.strip()[:30], case["reading"])

    report = build_report(
        provider=args.provider,
        model=args.model if args.provider == "gemma" else os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        cases_file=str(args.cases),
        results=results,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    logger.info("結果を保存: %s", out_path)


if __name__ == "__main__":
    main()
