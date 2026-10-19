from eval.metrics import best_reference_score, character_distance, exact_match, mora_error_rate, morae


def test_exact_match_normalizes_script_and_width():
    assert exact_match("ｺｳｼｮｸﾊﾞｲ", "こうしょくばい")


def test_mora_split_contracts_small_kana():
    assert morae("きゃりー") == ["きゃ", "り", "ー"]


def test_character_distance():
    assert character_distance("こうしょくばい", "ひかりしょくばい") > 0


def test_mora_error_rate():
    assert mora_error_rate("きゃり", "きゃりー") == 1 / 3


def test_multi_reference_uses_best_reference():
    score = best_reference_score("ヒカリショクバイ", ["こうしょくばい", "ひかりしょくばい"])
    assert score.exact_match
    assert score.reference == "ひかりしょくばい"

