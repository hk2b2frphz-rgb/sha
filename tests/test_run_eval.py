import json

from scripts.run_eval import build_batch_prompt, chunked, parse_batch_readings


def test_chunked_splits_cases_by_batch_size():
    cases = [{"id": str(index)} for index in range(25)]
    batches = chunked(cases, 10)
    assert [len(batch) for batch in batches] == [10, 10, 5]


def test_parse_batch_readings_accepts_results_object_and_katakana():
    raw = json.dumps(
        {
            "results": [
                {"id": "1", "reading": "\u30ab\u30c3\u30bb\u30a4\u30aa\u30c7\u30a4"},
                {"id": "2", "reading": "\u3060\u3063\u3061\u3063\u305d"},
            ]
        },
        ensure_ascii=False,
    )
    assert parse_batch_readings(raw) == {
        "1": "\u304b\u3063\u305b\u3044\u304a\u3067\u3044",
        "2": "\u3060\u3063\u3061\u3063\u305d",
    }


def test_parse_batch_readings_strips_markdown_fence():
    raw = '```json\n{"results":[{"id":"1","reading":"\u3042\u3044"}]}\n```'
    assert parse_batch_readings(raw) == {"1": "\u3042\u3044"}


def test_build_batch_prompt_requests_json_only():
    prompt = build_batch_prompt([{"id": "1", "term": "\u6c5a\u6ce5", "sentence": "\u6c5a\u6ce5\u3092\u51e6\u7406\u3059\u308b"}])
    assert "Return JSON only" in prompt
    assert '"id": "1"' in prompt
    assert "\u6c5a\u6ce5" in prompt
