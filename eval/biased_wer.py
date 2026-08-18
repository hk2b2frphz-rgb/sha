"""B-WER / U-WER — ドメイン用語に絞った誤り率と、それ以外の誤り率。

文脈バイアス付きASRの評価で使われる分け方をそのまま採る:

    U-WER : バイアス語彙に「入っていない」語の誤り率 (一般語がどれだけ崩れたか)
    B-WER : バイアス語彙に「入っている」語の誤り率   (狙った専門用語が取れたか)

全体WERだけを見ると、専門用語は文中の1語でしかないため、用語を全部間違えても
WERはわずかしか動かない。B-WERはその1語だけを分母にするので、ドメイン適応の
効き方が直接見える。fine-tuningでB-WERが下がりU-WERが上がっていたら、
汎用性能を犠牲にして用語を覚えたということで、これも見なければ判断できない。

日本語特有の注意:
形態素解析器は「活性汚泥法」を「活性/汚泥/法」に割ってしまう。割られると
用語単位の勘定ができないので、バイアス語は解析前に取り出して1トークンとして
扱う。参照・仮説の両方で同じ処理をする。

誤りの帰属は次の通り (この分野の標準的な扱い):
  置換・脱落 -> 参照側の語が属するクラス
  挿入       -> 挿入された仮説側の語が属するクラス
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class BiasedWer:
    b_wer: float | None
    u_wer: float | None
    wer: float | None
    b_errors: int
    b_total: int
    u_errors: int
    u_total: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "b_wer": self.b_wer,
            "u_wer": self.u_wer,
            "wer_from_alignment": self.wer,
            "b_errors": self.b_errors,
            "b_total": self.b_total,
            "u_errors": self.u_errors,
            "u_total": self.u_total,
        }


def segment_with_bias_terms(
    text: str,
    bias_terms: Sequence[str],
    tokenize: Callable[[str], list[str]],
) -> tuple[list[str], list[bool]]:
    """``text`` をトークン列にする。バイアス語は分割せず1トークンとして残す。

    戻り値は (トークン列, 各トークンがバイアス語か) の対。
    長い用語を先に当てるので、「汚泥」と「汚泥濃縮槽」が両方あっても
    後者が優先される。
    """
    terms = sorted({t for t in bias_terms if t}, key=len, reverse=True)
    tokens: list[str] = []
    is_bias: list[bool] = []
    cursor = 0
    while cursor < len(text):
        hit = next((t for t in terms if text.startswith(t, cursor)), None)
        if hit:
            tokens.append(hit)
            is_bias.append(True)
            cursor += len(hit)
            continue
        # 次のバイアス語の開始位置まで一気に進める
        nxt = len(text)
        for t in terms:
            i = text.find(t, cursor + 1)
            if i != -1:
                nxt = min(nxt, i)
        chunk = text[cursor:nxt]
        for tok in tokenize(chunk):
            if tok:
                tokens.append(tok)
                is_bias.append(False)
        cursor = nxt
    return tokens, is_bias


def _align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int, int]]:
    """Levenshtein の編集操作列を返す。要素は (op, ref_idx, hyp_idx)。

    op は "eq" | "sub" | "del" | "ins"。idx は該当しない側が -1。
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("eq" if cost == 0 else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
            continue
        ops.append(("ins", -1, j - 1))
        j -= 1
    ops.reverse()
    return ops


def biased_wer(
    reference: str,
    hypothesis: str,
    bias_terms: Sequence[str],
    tokenize: Callable[[str], list[str]],
) -> BiasedWer:
    ref_tokens, ref_bias = segment_with_bias_terms(reference, bias_terms, tokenize)
    hyp_tokens, hyp_bias = segment_with_bias_terms(hypothesis, bias_terms, tokenize)

    b_total = sum(ref_bias)
    u_total = len(ref_tokens) - b_total
    b_err = u_err = 0

    for op, ri, hi in _align(ref_tokens, hyp_tokens):
        if op == "eq":
            continue
        if op == "ins":
            # 挿入は仮説側の語が属するクラスへ
            if hyp_bias[hi]:
                b_err += 1
            else:
                u_err += 1
            continue
        # 置換・脱落は参照側の語が属するクラスへ
        if ref_bias[ri]:
            b_err += 1
        else:
            u_err += 1

    total = b_total + u_total
    return BiasedWer(
        b_wer=b_err / b_total if b_total else None,
        u_wer=u_err / u_total if u_total else None,
        wer=(b_err + u_err) / total if total else None,
        b_errors=b_err,
        b_total=b_total,
        u_errors=u_err,
        u_total=u_total,
    )


def aggregate(results: Iterable[BiasedWer]) -> BiasedWer:
    """発話ごとの結果をコーパス全体の率にまとめる。

    発話ごとの率を平均するのではなく、誤り数と語数をそれぞれ合計してから
    割る。短い発話が1語間違えただけで率が跳ねるのを平均に混ぜないため。
    """
    b_err = b_tot = u_err = u_tot = 0
    for r in results:
        b_err += r.b_errors
        b_tot += r.b_total
        u_err += r.u_errors
        u_tot += r.u_total
    total = b_tot + u_tot
    return BiasedWer(
        b_wer=b_err / b_tot if b_tot else None,
        u_wer=u_err / u_tot if u_tot else None,
        wer=(b_err + u_err) / total if total else None,
        b_errors=b_err,
        b_total=b_tot,
        u_errors=u_err,
        u_total=u_tot,
    )
