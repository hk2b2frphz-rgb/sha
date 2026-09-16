"""Run the first-round method comparison experiment.

Gold readings are loaded only from human-curated TSV files. Dictionary tools
are optional prediction methods and are never used to generate references.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.gold_loader import load_gold_tsv
from eval.harness import evaluate_records
from resolvers import HybridResolver, Resolver, SmartHybridResolver, build_available_dict_resolvers, build_llm_resolvers
from resolvers.dict_resolvers import BACKENDS

LOGGER = logging.getLogger("method_comparison")

BENCHMARKS = {
    "medical": Path("data/benchmark_terms.tsv"),
    "sewage": Path("data/benchmark_terms_sewage_60.tsv"),
    "general": Path("data/test_cases.tsv"),
}


def _package_available(package: str) -> bool:
    module_name = package.replace("-", "_")
    if package == "SudachiDict-core":
        module_name = "sudachidict_core"
    return importlib.util.find_spec(module_name) is not None


def attempt_optional_installs(timeout_sec: int = 300) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for backend, (_factory, install, packages) in BACKENDS.items():
        missing = [p for p in packages if not _package_available(p)]
        if not missing:
            statuses.append({"backend": backend, "attempted": False, "success": True, "message": "already installed"})
            continue
        command = [sys.executable, "-m", "pip", "install", *missing]
        try:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            success = proc.returncode == 0
            message = (proc.stdout + "\n" + proc.stderr).strip()[-2000:]
        except Exception as exc:  # pragma: no cover - environment dependent
            success = False
            message = f"{exc.__class__.__name__}: {exc}"
        statuses.append(
            {
                "backend": backend,
                "attempted": True,
                "success": success,
                "missing_before": missing,
                "command": " ".join(command),
                "message": message,
                "install_instruction": install,
            }
        )
    return statuses


def load_terms(path: Path) -> list[str]:
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        rows = (line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t")
        return [row["term"] for row in reader if row.get("term")]


def records_for_resolver(resolver: Resolver, terms: list[str]) -> list[dict[str, Any]]:
    records = []
    for index, term in enumerate(terms, 1):
        detail = resolver.resolve_details(term)
        records.append(
            {
                "id": index,
                "term": term,
                "prediction": detail.normalized_reading,
                "resolver_metadata": {
                    "confidence": detail.confidence,
                    "source": detail.source,
                    **detail.metadata,
                },
            }
        )
    return records


def aggregate_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    evaluated = [r for r in evaluation["results"] if r.get("status") != "missing_gold"]
    n = len(evaluated)
    return {
        "n": n,
        "accuracy": evaluation["exact_match_accuracy"],
        "average_mer": evaluation["average_mora_error_rate"],
        "average_character_distance": round(sum(r["character_distance"] for r in evaluated) / n, 6) if n else 0.0,
        "n_exact": evaluation["n_exact_match"],
    }


def bootstrap_accuracy_ci(outcomes: list[bool], n_resamples: int = 10000, seed: int = 13) -> dict[str, float]:
    if not outcomes:
        return {"lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    n = len(outcomes)
    values = []
    ints = [1 if x else 0 for x in outcomes]
    for _ in range(n_resamples):
        values.append(sum(ints[rng.randrange(n)] for _ in range(n)) / n)
    values.sort()
    return {"lower": round(values[int(0.025 * n_resamples)], 6), "upper": round(values[int(0.975 * n_resamples) - 1], 6)}


def mcnemar_test(a: list[bool], b: list[bool]) -> dict[str, float | int]:
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    total = b01 + b10
    if total == 0:
        return {"b01": b01, "b10": b10, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / total
    # Survival function for chi-square df=1: erfc(sqrt(x/2)).
    import math

    p_value = math.erfc(math.sqrt(chi2 / 2))
    return {"b01": b01, "b10": b10, "chi2": round(chi2, 6), "p_value": round(p_value, 6)}


def mcnemar_exact_test(a: list[bool], b: list[bool]) -> dict[str, float | int | str]:
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    total = b01 + b10
    if total == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0, "method": "no_discordance"}
    try:
        from scipy.stats import binomtest

        p_value = binomtest(min(b01, b10), total, 0.5, alternative="two-sided").pvalue
        method = "scipy.stats.binomtest"
    except Exception:
        # Exact two-sided binomial test for p=0.5.
        import math

        tail = sum(math.comb(total, k) for k in range(0, min(b01, b10) + 1)) / (2**total)
        p_value = min(1.0, 2 * tail)
        method = "fallback_exact_binomial"
    return {"b01": b01, "b10": b10, "p_value": round(p_value, 6), "method": method}


def wilcoxon_distance_test(a_distances: list[float], b_distances: list[float]) -> dict[str, float | str | int]:
    if not a_distances or len(a_distances) != len(b_distances):
        return {"n": 0, "statistic": 0.0, "p_value": 1.0, "method": "unavailable"}
    diffs = [x - y for x, y in zip(a_distances, b_distances)]
    nonzero = sum(1 for value in diffs if value)
    if nonzero == 0:
        return {"n": len(diffs), "nonzero": 0, "statistic": 0.0, "p_value": 1.0, "method": "no_difference"}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(a_distances, b_distances, alternative="two-sided", zero_method="wilcox")
        return {
            "n": len(diffs),
            "nonzero": nonzero,
            "statistic": round(float(result.statistic), 6),
            "p_value": round(float(result.pvalue), 6),
            "method": "scipy.stats.wilcoxon",
        }
    except Exception as exc:
        return {"n": len(diffs), "nonzero": nonzero, "statistic": 0.0, "p_value": 1.0, "method": f"unavailable: {exc}"}


def complementarity(a: list[bool], b: list[bool]) -> dict[str, int | float]:
    a_fail_b_success = sum((not x) and y for x, y in zip(a, b))
    a_success_b_fail = sum(x and (not y) for x, y in zip(a, b))
    either_success = sum(x or y for x, y in zip(a, b))
    both_fail = sum((not x) and (not y) for x, y in zip(a, b))
    n = len(a)
    return {
        "a_fail_b_success": a_fail_b_success,
        "a_success_b_fail": a_success_b_fail,
        "either_success": either_success,
        "both_fail": both_fail,
        "complementarity_rate": round((a_fail_b_success + a_success_b_fail) / n, 6) if n else 0.0,
    }


def oracle_upper(a: list[bool], b: list[bool]) -> dict[str, int | float]:
    either_success = sum(x or y for x, y in zip(a, b))
    n = len(a)
    return {
        "n": n,
        "n_exact": either_success,
        "accuracy": round(either_success / n, 6) if n else 0.0,
    }


def make_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Japanese Reading Estimation Method Comparison",
        "",
        "## Design Notes",
        "",
        "- Gold readings are loaded only from the human-curated TSV files in `data/`; dictionary analyzers are prediction-side methods only.",
        "- LLM methods use saved `experiments/*_neweval.json` predictions and do not run live inference.",
        "- First-round hybrid methods use a dictionary prediction only when it is single-token/high-confidence or when dictionary and LLM readings agree; otherwise they fall back to the saved LLM output.",
        "- Smart hybrid methods score prediction-side signals only: OOV/unknown flags, kana consistency, tokenization confidence, multiple dictionary agreement, saved-LLM agreement, and surface lexical markers. They do not read gold labels.",
        "- Optional dictionary dependencies are skipped when unavailable, with install commands recorded below.",
        "",
        "## Optional Dictionary Status",
        "",
        "| Resolver | Available | Message | Install |",
        "|---|---:|---|---|",
    ]
    for status in results["dictionary_status"]:
        lines.append(f"| {status['name']} | {status['available']} | {status['message']} | `{status['install']}` |")

    lines += ["", "## Optional Install Attempts", "", "| Backend | Attempted | Success | Message |", "|---|---:|---:|---|"]
    for attempt in results.get("install_attempts", []):
        message = str(attempt.get("message", "")).replace("\n", " ")
        if len(message) > 180:
            message = message[:177] + "..."
        lines.append(
            f"| {attempt.get('backend', '')} | {attempt.get('attempted', False)} | {attempt.get('success', False)} | {message} |"
        )

    lines += ["", "## Main Results", "", "| Method | Domain | n | Accuracy | 95% CI | MER | Char Edit |", "|---|---|---:|---:|---|---:|---:|"]
    for method, domains in results["summary"].items():
        for domain, row in domains.items():
            ci = row["accuracy_ci"]
            lines.append(
                f"| {method} | {domain} | {row['n']} | {row['accuracy']:.3f} | [{ci['lower']:.3f}, {ci['upper']:.3f}] | {row['average_mer']:.3f} | {row['average_character_distance']:.3f} |"
            )

    lines += ["", "## McNemar Exact-Match Tests", ""]
    for domain, matrix in results["mcnemar"].items():
        lines += [f"### {domain}", "", "| Method A | Method B | A fail/B success | A success/B fail | chi2 | p |", "|---|---|---:|---:|---:|---:|"]
        for pair, row in matrix.items():
            a, b = pair.split(" vs ", 1)
            lines.append(f"| {a} | {b} | {row['b01']} | {row['b10']} | {row['chi2']:.3f} | {row['p_value']:.3f} |")
        lines.append("")

    lines += ["## Exact McNemar Binomial Tests", ""]
    for domain, matrix in results["mcnemar_exact"].items():
        lines += [f"### {domain}", "", "| Method A | Method B | A fail/B success | A success/B fail | p | Method |", "|---|---|---:|---:|---:|---|"]
        for pair, row in matrix.items():
            a, b = pair.split(" vs ", 1)
            lines.append(f"| {a} | {b} | {row['b01']} | {row['b10']} | {row['p_value']:.3f} | {row['method']} |")
        lines.append("")

    lines += ["## Wilcoxon Character-Distance Tests", ""]
    for domain, matrix in results["wilcoxon_character_distance"].items():
        lines += [f"### {domain}", "", "| Method A | Method B | n | Nonzero | statistic | p | Method |", "|---|---|---:|---:|---:|---:|---|"]
        for pair, row in matrix.items():
            a, b = pair.split(" vs ", 1)
            lines.append(
                f"| {a} | {b} | {row['n']} | {row.get('nonzero', 0)} | {row['statistic']:.3f} | {row['p_value']:.3f} | {row['method']} |"
            )
        lines.append("")

    lines += ["## Error Analysis", "", "| Method | Domain | Error Type | Count |", "|---|---|---|---:|"]
    for method, domains in results["error_analysis"].items():
        for domain, counts in domains.items():
            for label, count in counts.items():
                lines.append(f"| {method} | {domain} | {label} | {count} |")

    lines += ["", "## Complementarity", ""]
    for domain, matrix in results["complementarity"].items():
        lines += [f"### {domain}", "", "| Method A | Method B | A fail/B success | A success/B fail | Either succeeds | Both fail | Rate |", "|---|---|---:|---:|---:|---:|---:|"]
        for pair, row in matrix.items():
            a, b = pair.split(" vs ", 1)
            lines.append(
                f"| {a} | {b} | {row['a_fail_b_success']} | {row['a_success_b_fail']} | {row['either_success']} | {row['both_fail']} | {row['complementarity_rate']:.3f} |"
            )
        lines.append("")

    lines += ["## Oracle Upper Bounds", ""]
    for domain, matrix in results["oracle_upper"].items():
        lines += [f"### {domain}", "", "| Method A | Method B | n | Oracle exact | Oracle accuracy |", "|---|---|---:|---:|---:|"]
        for pair, row in matrix.items():
            a, b = pair.split(" vs ", 1)
            lines.append(f"| {a} | {b} | {row['n']} | {row['n_exact']} | {row['accuracy']:.3f} |")
        lines.append("")

    lines += ["## Key Findings", ""]
    best_by_domain = results["best_by_domain"]
    for domain, best in best_by_domain.items():
        lines.append(f"- {domain}: best observed accuracy is {best['accuracy']:.3f} from {best['method']}.")
    lines.append("- Dictionary rows appear only for dependencies that installed successfully in this environment; skipped rows are not imputed.")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    install_attempts = [] if args.no_install else attempt_optional_installs(timeout_sec=args.install_timeout)
    dict_resolvers, dict_status = build_available_dict_resolvers()
    llm_resolvers = build_llm_resolvers(ROOT / "experiments")
    if not llm_resolvers:
        LOGGER.warning("No *_neweval.json LLM prediction files found; running dictionary resolvers only.")

    resolvers: list[Resolver] = [*dict_resolvers, *llm_resolvers]
    if not resolvers:
        raise RuntimeError("No resolvers available. Install at least one dictionary backend or add *_neweval.json predictions.")
    if dict_resolvers:
        primary_dict = dict_resolvers[0]
        for llm in llm_resolvers:
            resolvers.append(HybridResolver(primary_dict, llm))
        e4b_resolvers = [r for r in llm_resolvers if "e4b" in r.name]
        if e4b_resolvers:
            primary_llm = e4b_resolvers[0]
            peer_llms = [r for r in llm_resolvers if r is not primary_llm]
            resolvers.append(SmartHybridResolver(dict_resolvers, primary_llm, peer_llms))

    all_results: dict[str, Any] = {
        "install_attempts": install_attempts,
        "dictionary_status": [asdict(s) for s in dict_status],
        "methods": [r.name for r in resolvers],
        "benchmarks": {k: str(v) for k, v in BENCHMARKS.items()},
        "summary": {},
        "per_term": {},
        "mcnemar": {},
        "mcnemar_exact": {},
        "wilcoxon_character_distance": {},
        "error_analysis": {},
        "complementarity": {},
        "oracle_upper": {},
        "best_by_domain": {},
    }

    outcomes: dict[str, dict[str, list[bool]]] = {domain: {} for domain in BENCHMARKS}
    character_distances: dict[str, dict[str, list[float]]] = {domain: {} for domain in BENCHMARKS}
    for resolver in resolvers:
        all_results["summary"][resolver.name] = {}
        all_results["per_term"][resolver.name] = {}
        all_results["error_analysis"][resolver.name] = {}
        for domain, path in BENCHMARKS.items():
            terms = load_terms(path)
            gold = load_gold_tsv(ROOT / path)
            records = records_for_resolver(resolver, terms)
            evaluation = evaluate_records(records, gold)
            summary = aggregate_evaluation(evaluation)
            exacts = [bool(r["exact_match"]) for r in evaluation["results"] if r.get("status") != "missing_gold"]
            distances = [float(r["character_distance"]) for r in evaluation["results"] if r.get("status") != "missing_gold"]
            summary["accuracy_ci"] = bootstrap_accuracy_ci(exacts, args.bootstrap_samples)
            all_results["summary"][resolver.name][domain] = summary
            all_results["per_term"][resolver.name][domain] = evaluation["results"]
            outcomes[domain][resolver.name] = exacts
            character_distances[domain][resolver.name] = distances
            all_results["error_analysis"][resolver.name][domain] = dict(Counter(r["error_type"] for r in evaluation["results"] if not r.get("exact_match")))

    method_names = [r.name for r in resolvers]
    for domain in BENCHMARKS:
        all_results["mcnemar"][domain] = {}
        all_results["mcnemar_exact"][domain] = {}
        all_results["wilcoxon_character_distance"][domain] = {}
        all_results["complementarity"][domain] = {}
        all_results["oracle_upper"][domain] = {}
        for i, a in enumerate(method_names):
            for b in method_names[i + 1 :]:
                pair = f"{a} vs {b}"
                all_results["mcnemar"][domain][pair] = mcnemar_test(outcomes[domain][a], outcomes[domain][b])
                all_results["mcnemar_exact"][domain][pair] = mcnemar_exact_test(outcomes[domain][a], outcomes[domain][b])
                all_results["wilcoxon_character_distance"][domain][pair] = wilcoxon_distance_test(
                    character_distances[domain][a], character_distances[domain][b]
                )
                all_results["complementarity"][domain][pair] = complementarity(outcomes[domain][a], outcomes[domain][b])
                all_results["oracle_upper"][domain][pair] = oracle_upper(outcomes[domain][a], outcomes[domain][b])

        best_method = max(method_names, key=lambda m: all_results["summary"][m][domain]["accuracy"])
        all_results["best_by_domain"][domain] = {
            "method": best_method,
            "accuracy": all_results["summary"][best_method][domain]["accuracy"],
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(make_markdown_report(all_results), encoding="utf-8")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reading-estimation methods across benchmarks.")
    parser.add_argument("--out-json", type=Path, default=ROOT / "experiments/comparison_results.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "experiments/comparison_report.md")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--install-timeout", type=int, default=300)
    parser.add_argument("--no-install", action="store_true", help="Skip optional dictionary pip install attempts.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
