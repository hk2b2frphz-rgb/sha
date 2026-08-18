#!/usr/bin/env python3
"""用語ごとに構造の異なる例文を機械的に割り当てる。

前回の失敗: 「文字列として全部違う」を条件にしたところ、"次回確認日を定めた○○と
前回の結果も参照し、..." のような同一の書き出し骨格を穴埋めしただけの文が
量産された (72%が同一骨格)。文字列としては非重複でも、音響的な文脈の多様性が
実質無い。

この対策として、文の「骨格」を (shape, context, subject, tail) の4軸の組で
定義し、各用語には全軸が重複しない組を1つ割り当てる。軸のサイズを十分大きく
取ることで、用語数(1265)よりずっと大きい組み合わせ空間から非復元抽出でき、
骨格の完全一致がそもそも起こらない。

さらに shape 自体に用語の出現位置 (文頭/文中/文末) と文体 (敬体/常体/体言止め)
を混ぜているため、同じ context/subject/tail を再利用しても書き出しの文字面は
shape で変わる。

生成後、以下を機械検証する:
  - 用語が文中に一字一句含まれる
  - 全文がユニーク
  - 先頭12文字の重複が閾値を超えない (前回問題の再発防止の直接チェック)

Usage:
    python scripts/generate_diverse_sentences.py \
        --in data/generated_sentences.csv --out data/generated_sentences.csv --seed 42
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path

# ---- 軸1: 文の骨格 (shape) -------------------------------------------------
# {ctx}=状況節, {subj}=主体, {term}=用語, {tail}=述部。
# 用語の出現位置・文体・語順をshapeごとに変えることで、同じctx/subj/tailを
# 再利用しても書き出しの文字面が変わるようにする。
#
# 設計ルール(厳守): {ctx} は既にで/に/際/受け/先立ち/ところ/ため等の接続で終わる
# 完成した副詞節なので、直後には必ず「、」だけを置く。{ctx}のすぐ後ろに
# 「際」「にあたり」「前に」「にもかかわらず」「資料を」等の語や助詞を直接
# くっつけない (「〜作業中に中に」「〜受けにもかかわらず」のような二重接続や
# 助詞の衝突が発生するため、実際にこれで壊れた文が223件出た教訓)。
# {tail} は「確認した」のような完了形の述語なので、文末 / 「そう{tail}と」/
# 「{tail}のは」の位置でのみ使う。名詞スロット (「〜を{tail}を行った」等) には
# 使わない。
SHAPES = [
    "{ctx}、{subj}は{term}について{tail}。",
    "{subj}は{ctx}、{term}の状況を{tail}。",
    "{term}に関して、{subj}が{tail}。",
    "{ctx}、{term}の扱いが議題に上がった。",
    "{subj}によると、{term}に問題は無かったという。",
    "{term}の件は、{subj}が{tail}。",
    "{ctx}、{subj}が指摘したのは{term}についてだった。",
    "{subj}が{tail}のは、{term}を確認したためだ。",
    "{ctx}、{term}を巡って{subj}と協議した。",
    "{term}――{ctx}、{subj}はそう{tail}と振り返る。",
    "{ctx}、{subj}は{term}を重点項目とした。",
    "{subj}: 「{term}の扱いはどうなっていますか」",
    "{ctx}、{subj}は{term}を{tail}。",
    "{ctx}、{term}を含む資料を{subj}が{tail}。",
    "{ctx}、{subj}から{term}についての報告があった。",
    "{ctx}、{term}が話題になったと{subj}は話す。",
    "{subj}の説明では、{term}は想定内の範囲だった。",
    "{ctx}、{term}の確認を{subj}へ依頼した。",
    "{term}について、{subj}は{tail}。",
    "{ctx}、{term}の対応が後回しになっていないか{subj}が確認した。",
]

# ---- 軸2: 状況節 (ctx) ------------------------------------------------------
CTXS = [
    "月例点検の結果を受け", "夜間工事の安全会議で", "施工計画の協議に先立ち",
    "議会答弁を準備する過程で", "委託業者との現場打合せで", "長寿命化計画の見直し時に",
    "運転日報を確認したところ", "事故原因の検証会議で", "工事完成図との照合時に",
    "次年度の実施計画を作る際", "施設台帳を更新するため", "関係機関との合同協議で",
    "監査資料を取りまとめる際", "交代勤務者への申し送りで", "試運転の評価会で",
    "住民説明会の準備中に", "台風接近時の緊急点検で", "老朽化更新の優先順位付けで",
    "広域化に向けた協議で", "研修資料を作成する際", "苦情対応の記録を整理中に",
    "予算査定のヒアリングで", "水質事故の初動対応で", "設計変更の審査時に",
    "災害訓練の振り返りで", "新任職員への引継ぎで", "定期点検の立会中に",
    "更新工事の入札説明会で", "他事業体との視察対応で", "維持管理計画の改定作業で",
    "夜間巡視の申し送りで", "断水復旧作業の途中で", "耐震診断の結果報告で",
    "漏水調査の現地立会で", "環境モニタリング会議で", "第三者委員会の審議で",
    "現場代理人との打合せで", "水質検査の年次報告で", "設備更新の稟議作成中に",
    "点検業者からの報告受領時に", "業務継続計画の見直しで", "監視制御システムの更新協議で",
    "施設見学会の案内準備で", "料金審議会の資料作成中に", "工事安全パトロールで",
    "処理能力の再評価作業で", "系統切替えの事前打合せで", "汚泥処分委託の契約更新時に",
    "新設管路の竣工検査で", "維持管理業務の年度総括で",
]

# ---- 軸3: 主体 (subj) -------------------------------------------------------
SUBJS = [
    "現場代理人", "運転員", "維持管理担当者", "設計班", "工事監督員",
    "水質担当者", "施設管理者", "委託業者", "調査チーム", "事業担当課",
    "保全担当", "点検業者", "運転管理会社", "施工業者", "監理技術者",
    "住民代表", "議員", "本庁担当者", "支所職員", "危機管理担当",
    "電気設備担当", "機械設備担当", "経理担当", "計画担当", "広報担当",
]

# ---- 軸4: 述部 (tail) --------------------------------------------------------
TAILS = [
    "確認した", "報告した", "説明した", "指摘した", "記録に残した",
    "検討した", "見直した", "共有した", "整理した", "取りまとめた",
    "指示した", "把握した", "点検した", "評価した", "協議した",
    "示した", "提案した", "更新した", "照会した", "調整した",
    "判断した", "承認を得た", "資料化した", "引き継いだ", "問い合わせた",
    "改めた", "洗い出した", "手配した", "処理した", "監視した",
    "測定した", "算定した", "見送った", "反映させた", "総括した",
    "説明を求めた", "現地確認した", "回覧した", "議事録に残した", "決裁を仰いだ",
]


import re

_FIELD_RE = re.compile(r"\{(\w+)\}")


def leading_axis(shape: str) -> str:
    """shape内で最初に出てくるプレースホルダ名。文頭の文字面がどの軸に
    依存するかを表す。ctx/subj はプールが小さい (50/25) ため、これらが
    文頭に来る行は先頭12文字が衝突しやすい。term は1265件すべて一意なので
    文頭にあれば衝突しない。"""
    m = _FIELD_RE.findall(shape)
    return m[0] if m else "term"


SHAPE_LEAD = [leading_axis(s) for s in SHAPES]


def build_sentence(term: str, combo: tuple[int, int, int, int]) -> str:
    si, ci, ui, ti = combo
    return SHAPES[si].format(ctx=CTXS[ci], subj=SUBJS[ui], term=term, tail=TAILS[ti])


def assign_unique_combos(
    rng: random.Random, n: int, *, ctx_lead_cap: int = 2, subj_lead_cap: int = 2
) -> list[tuple[int, int, int, int]]:
    """(shape, ctx, subj, tail) の組を n 個、重複なく割り当てる。

    ctx/subj が文頭に来る行は、そのctx/subj値ごとに ctx_lead_cap/subj_lead_cap
    回までしか使わない (プールが小さく、文頭で使うと先頭文字が衝突するため)。
    term が文頭に来る行には制限を掛けない (termは全件一意)。
    """
    space = len(SHAPES) * len(CTXS) * len(SUBJS) * len(TAILS)
    assert space >= n, f"組み合わせ空間 {space} が用語数 {n} より小さい"

    ctx_lead_used: dict[int, int] = collections.Counter()
    subj_lead_used: dict[int, int] = collections.Counter()
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []

    while len(out) < n:
        si = rng.randrange(len(SHAPES))
        ci = rng.randrange(len(CTXS))
        ui = rng.randrange(len(SUBJS))
        ti = rng.randrange(len(TAILS))
        combo = (si, ci, ui, ti)
        if combo in seen:
            continue
        lead = SHAPE_LEAD[si]
        if lead == "ctx" and ctx_lead_used[ci] >= ctx_lead_cap:
            continue
        if lead == "subj" and subj_lead_used[ui] >= subj_lead_cap:
            continue
        seen.add(combo)
        out.append(combo)
        if lead == "ctx":
            ctx_lead_used[ci] += 1
        elif lead == "subj":
            subj_lead_used[ui] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-prefix-reuse", type=int, default=2, help="先頭12文字の重複許容数")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.inp.open(encoding="utf-8")))
    rng = random.Random(args.seed)
    combos = assign_unique_combos(rng, len(rows))
    rng.shuffle(combos)  # 用語の並び順とshape/ctx/subj/tailの相関を消す

    out_rows = []
    for row, combo in zip(rows, combos):
        term = row["Word"]
        sentence = build_sentence(term, combo)
        assert term in sentence, f"用語が文に含まれない: {term} / {sentence}"
        out_rows.append({**row, "Sentance": sentence})

    # ---- 検証: 前回問題になった「先頭12文字の使い回し」を直接チェック ----
    prefixes = collections.Counter(r["Sentance"][:12] for r in out_rows)
    over = {p: n for p, n in prefixes.items() if n > args.max_prefix_reuse}
    sentences = [r["Sentance"] for r in out_rows]
    dup_sentences = len(sentences) - len(set(sentences))

    print(f"件数: {len(out_rows)}")
    print(f"先頭12文字ユニーク数: {len(prefixes)} / {len(out_rows)}")
    print(f"先頭12文字が{args.max_prefix_reuse}件を超えて重複: {len(over)} グループ")
    print(f"完全一致の重複文: {dup_sentences}")
    if over:
        for p, n in sorted(over.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {n:3d}件  {p}...")
        raise SystemExit("先頭12文字の重複が閾値を超えています。軸の組み合わせ数が不足しています。")
    if dup_sentences:
        raise SystemExit(f"完全一致の重複文が {dup_sentences} 件あります。")

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
