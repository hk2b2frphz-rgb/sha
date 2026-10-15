"""Research-grade reading evaluation harness."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .error_classifier import classify_error
from .gold_loader import load_gold_files
from .gold_schema import GoldEntry
from .metrics import best_reference_score
from .normalize import normalize_reading


def _prediction_text(record: dict[str, Any]) -> str:
    for key in ("prediction", "reading", "llm_output", "tts_text", "output"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def load_prediction_records(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("all_results"), list):
            return data["all_results"]
        if isinstance(data.get("results"), list):
            return data["results"]
    raise ValueError(f"unsupported prediction JSON shape: {path}")


def evaluate_records(records: list[dict[str, Any]], gold: dict[str, GoldEntry]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        term = str(record.get("term", ""))
        entry = gold.get(term)
        prediction = _prediction_text(record)
        if not entry:
            results.append({"id": record.get("id", index), "term": term, "status": "missing_gold"})
            continue

        references = list(entry.reading_values)
        normalized_prediction = normalize_reading(prediction)
        contained_reference = next((ref for ref in references if normalize_reading(ref) in normalized_prediction), None)
        # Legacy experiment JSON stores full LLM/TTS text, not only the reading.
        # Treat a contained gold reading as exact, then use distance metrics only
        # when no acceptable reading appears in the output.
        score = (
            best_reference_score(contained_reference, references)
            if contained_reference
            else best_reference_score(prediction, references)
        )
        row = {
            "id": record.get("id", index),
            "term": term,
            "prediction": prediction,
            "prediction_normalized": normalized_prediction,
            "gold_readings": references,
            "best_reference": score.reference,
            "exact_match": score.exact_match,
            "character_distance": score.character_distance,
            "mora_distance": score.mora_distance,
            "mora_error_rate": round(score.mora_error_rate, 6),
            "error_type": classify_error(prediction, references, score),
            "domain": entry.domain or record.get("domain", ""),
        }
        if "resolver_metadata" in record:
            row["resolver_metadata"] = record["resolver_metadata"]
        results.append(row)

    evaluated = [r for r in results if r.get("status") != "missing_gold"]
    n = len(evaluated)
    n_exact = sum(1 for r in evaluated if r["exact_match"])
    avg_mer = sum(float(r["mora_error_rate"]) for r in evaluated) / n if n else 0.0
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_total": len(results),
        "n_evaluated": n,
        "n_missing_gold": len(results) - n,
        "n_exact_match": n_exact,
        "exact_match_accuracy": round(n_exact / n, 6) if n else 0.0,
        "average_mora_error_rate": round(avg_mer, 6),
        "results": results,
    }


def evaluate_prediction_file(predictions: Path | str, gold_paths: list[Path | str]) -> dict[str, Any]:
    return evaluate_records(load_prediction_records(predictions), load_gold_files(gold_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reading predictions against multi-reference gold data.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_prediction_file(args.predictions, args.gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_evaluated", "exact_match_accuracy", "average_mora_error_rate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
