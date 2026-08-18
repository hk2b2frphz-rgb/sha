import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.biased_wer import aggregate, biased_wer, segment_with_bias_terms  # noqa: E402

# 文字単位に割る素朴なトークナイザ。形態素解析器がバイアス語を割ってしまう
# 状況を再現するので、用語が1トークンに保たれるかを厳しく試せる。
CHARS = lambda s: list(s)  # noqa: E731
TERMS = ["活性汚泥法", "汚泥", "配水池"]


def test_bias_term_survives_a_tokenizer_that_would_split_it():
    tokens, is_bias = segment_with_bias_terms("活性汚泥法で処理する", TERMS, CHARS)

    assert tokens[0] == "活性汚泥法", "用語が分割されると用語単位で数えられない"
    assert is_bias[0] is True
    assert all(b is False for b in is_bias[1:])


def test_longest_term_wins_over_a_contained_one():
    tokens, is_bias = segment_with_bias_terms("活性汚泥法", TERMS, CHARS)

    # 「汚泥」も語彙にあるが、より長い「活性汚泥法」を優先する
    assert tokens == ["活性汚泥法"]
    assert is_bias == [True]


def test_perfect_transcript_scores_zero_on_both():
    r = biased_wer("配水池を点検する", "配水池を点検する", TERMS, CHARS)

    assert r.b_wer == 0.0
    assert r.u_wer == 0.0
    assert r.b_total == 1


def test_a_wrong_term_is_charged_to_b():
    # 「配水池」を「排水池」と誤った。一般語(を点検する)は正しい。
    r = biased_wer("配水池を点検する", "排水池を点検する", TERMS, CHARS)

    assert r.b_wer == 1.0, "用語をすべて外したのだから B-WER は 1.0"
    assert r.b_total == 1


def test_a_wrong_ordinary_word_moves_u_wer_but_not_b_wer():
    r = biased_wer("配水池を点検する", "配水池を天検する", TERMS, CHARS)

    assert r.b_wer == 0.0
    assert r.u_wer and r.u_wer > 0


def test_b_wer_is_far_more_sensitive_than_overall_wer():
    ref = "最終沈殿池を出た下水は消毒設備へ送られる"
    terms = ["最終沈殿池"]
    wrong = ref.replace("最終沈殿池", "最終賃殿池")

    ok = biased_wer(ref, ref, terms, CHARS)
    bad = biased_wer(ref, wrong, terms, CHARS)

    assert ok.b_wer == 0.0
    assert bad.b_wer == 1.0
    # 用語を外しても全体WERは 0.31 止まり。B-WER なら 1.0 と振り切れる。
    # 全体WERだけ見ていると用語の取りこぼしが埋もれる、というのがこの指標の要点。
    assert bad.wer is not None and bad.wer < bad.b_wer / 2


def test_a_mangled_term_also_charges_some_insertions_to_u():
    """用語を外すと U-WER も少し動く。指標の性質なので明示しておく。

    参照では「配水池」が1トークン、仮説では認識を外して複数トークンに割れる。
    その差は挿入として扱われ、挿入は仮説側の語のクラス(=非バイアス)に付く。
    形態素解析器を使う本番では割れ方が粗い分この影響は小さいが、ゼロではない。
    B-WER と U-WER を並べて読むときはこの分を織り込むこと。
    """
    r = biased_wer("配水池を点検する", "排水池を点検する", TERMS, CHARS)

    assert r.b_wer == 1.0, "用語は完全に外しているので B-WER は 1.0"
    assert r.u_errors > 0, "割れた分が挿入として U 側に付く"
    assert r.u_wer is not None and r.u_wer < r.b_wer


def test_insertion_is_charged_to_the_class_of_the_inserted_word():
    r = biased_wer("点検する", "配水池点検する", TERMS, CHARS)

    assert r.b_errors == 1, "挿入された語がバイアス語なら B 側に付ける"
    assert r.u_errors == 0


def test_deletion_of_a_term_is_charged_to_b():
    r = biased_wer("配水池を点検", "を点検", TERMS, CHARS)

    assert r.b_errors == 1
    assert r.b_total == 1


def test_aggregate_pools_counts_rather_than_averaging_rates():
    a = biased_wer("配水池", "排水池", TERMS, CHARS)          # 1/1 誤り
    b = biased_wer("配水池を点検する", "配水池を点検する", TERMS, CHARS)  # 0/1 誤り

    pooled = aggregate([a, b])

    assert pooled.b_total == 2
    assert pooled.b_errors == 1
    assert pooled.b_wer == 0.5


def test_no_bias_terms_present_leaves_b_wer_undefined():
    r = biased_wer("点検を実施した", "点検を実施した", TERMS, CHARS)

    assert r.b_total == 0
    assert r.b_wer is None, "分母がないのに 0.0 と報告すると完璧に見えてしまう"
