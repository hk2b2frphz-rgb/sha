import json
from pathlib import Path

base = Path("experiments")
files = [
    "2026-06-13_e2b_eval.json",
    "2026-06-13_e4b_eval.json",
    "2026-06-13_sewage_e2b_eval.json",
    "2026-06-13_sewage_e4b_eval.json",
]

for fname in files:
    data = json.loads((base / fname).read_text(encoding="utf-8"))
    print(f"\n=== {data['model']} / {fname} ===")
    seen = {}
    for r in data["all_results"]:
        t = r["term"]
        if t not in seen:
            seen[t] = r["tts_text"]
            print(f"{t}\t{r['tts_text']}")
