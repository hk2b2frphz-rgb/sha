# 学習データの作り方 (上下水道ドメイン)

TTS で「専門用語だけをひらがな読みにした音声」を作り、正解テキストは漢字のまま与える。
これで whisper に「この音 → この漢字表記」を覚えさせる。1265 件・1.69 時間の音声を
CPU だけで約 35 分で作れる。

    音声:  「ひらいせつびについて調べてみたら、思ったより奥が深かった。」
    正解:  「避雷設備について調べてみたら、思ったより奥が深かった。」

ベースモデルはこの音声を「飛来設備」と誤変換する。読みは取れているので問題は表記側にあり、
だから encoder を凍結して decoder だけ学習させる (`CLAUDE.md` 参照)。

---

## 全体の流れ

```
用語収集 (公開資料)            → data/terms_*.tsv        1346語
  ↓ scripts/build_term_csv.py     重複除去・機械検証      1265語
data/generated_sentences.csv
  ↓ scripts/generate_diverse_sentences.py   例文を再生成
data/generated_sentences.csv (Sentance列が入れ替わる)
  ↓ scripts/run_cpu_dataprep.sh   TTS合成〜manifest作成
out/whisper_turbo/{train,dev}.jsonl + tts_data/shard_00/wav/*.wav
```

---

## 1. 用語の収集 (`data/terms_*.tsv`)

タブ区切り、列は `term / reading / sentence / category / source`。
`＃` または `#` で始まる行はコメント。

| ファイル | 語数 | 出典 |
|---|---|---|
| `terms_sewage_repo60.tsv` | 60 | リポジトリ既存のベンチマーク60語 (S001-S015は人手、以降は公開資料で確認済み) |
| `terms_water_supply.tsv` | 68 | 明石市水道局 水道用語集、前澤化成工業 排水設備用語集 |
| `terms_codex.tsv` | 669 | 日本水道協会「水道用語辞典」第2版、日本下水道協会「下水道用語集」、e-Gov法令検索ほか |
| `terms_codex2.tsv` | 549 | 国土交通省の各種ガイドライン (管きょ更生工法、ストックマネジメント、点検・調査マニュアル)、水質基準項目 |

`terms_codex*.tsv` は codex に収集させた。フラグシップモデルを明示すること:

```bash
codex exec --skip-git-repo-check -s workspace-write -m gpt-5.6-sol "<収集指示>"
```

**読みの正しさが最重要。** 読みが間違っていると「誤った音 → 正しい漢字」を教えることになり、
学習データとして有害になる。収集時は次を厳守させた:

- 用語集が読みを明示しているものだけを採用し、推測した読みは出さない
- 誤読の実例を指示に含める (「伏越し」は「ふせこし」であり「ふくえつし」は誤り)
- 出典を `source` 列に必ず書かせる

## 2. 用語の統合と検証 (`scripts/build_term_csv.py`)

```bash
source .venv-cpu/bin/activate
python scripts/build_term_csv.py data/terms_*.tsv \
    --out data/generated_sentences.csv \
    --report /tmp/rejected.txt
```

1346語 → **1265語**。以下に該当する行を機械的に落とす:

| 検査 | 落とす理由 |
|---|---|
| 漢字を含まない | カタカナ語・英字略語は表記ゆれ学習の対象にならない |
| reading がひらがな以外 | 読みとして使えない |
| 用語が sentence に一字一句含まれない | 後段の読み置換ができない |
| reading == term | 置換しても音が変わらない |
| 用語の重複 | 同一用語は1件だけ残す |
| **同じ用語に別の読み** | どちらかが誤り。両方落として人が確認する |

## 3. 例文の生成 (`scripts/generate_diverse_sentences.py`)

用語と読みは保持したまま、例文だけを作り直す。

```bash
python scripts/generate_diverse_sentences.py \
    --in data/generated_sentences.csv \
    --out data/generated_sentences.csv --seed 42
```

### なぜ専用スクリプトが要るか

codex に「全部違う文を書け」と指示すると、文字列としては非重複でも
**同じ骨格を穴埋めしただけ**の文を量産する。実際に 2 回失敗した:

1. 1回目: 同じ書き出しが最大 75 回そのまま重複
2. 2回目 (修正依頼後): 書き出しは変えず内側の節だけ差し替え。用語を除いて正規化すると
   **72% が同一骨格**。「全文ユニーク」のチェックはすり抜けていた
3. さらに骨格を 1264 種類に分けても、全文が「三人称過去形の役所報告調」という
   **一つの話法しか無い**状態だった

つまり検証基準を「文字列の非重複」に置いている限り、何度頼み直しても同じ結果になる。
そこで**衝突しないことを構造的に保証する生成器**を書いた。

### 仕組み

文を 4 軸の組で定義し、各用語に重複しない組を割り当てる:

| 軸 | 数 | 例 |
|---|---|---|
| shape (骨格) | 52 | `{term}について、{subj}は{tail}。` |
| ctx (状況節) | 50 | 月例点検の結果を受け / 台風接近時の緊急点検で |
| subj (主体) | 25 | 現場代理人 / 水質担当者 / 議員 |
| tail (述部) | 40 | 確認した / 議事録に残した / 決裁を仰いだ |

52 shape のうち **35 は用語が文頭に来る** (term-leading)。用語は1265件すべて一意なので
文頭が衝突しない。ctx/subj が文頭に来る shape はプールが小さい (50/25) ため、
値ごとに 2 回までに制限している。

shape は 7 つの文体に分かれており、役所報告調は 21% まで下げてある:

| 文体 | 割合 | 例 |
|---|---|---|
| 報告 | 21% | 月例点検の結果を受け、運転員は◯について確認した。 |
| 雑感 (一人称) | 15% | 先輩から◯の話を聞いて、少し驚いた。 |
| 指示 | 15% | ◯の取り扱いには、十分注意すること。 |
| 辞書引用 | 15% | 用語集には、◯の項目が記載されている。 |
| 質問 | 12% | ◯に関して、何か質問はありますか。 |
| 記述 | 11% | ◯という課題は、以前から指摘されてきた。 |
| 業務連絡 | 10% | ◯の件で、折り返しご連絡いたします。 |

### 生成時の自動検証

閾値を超えたら**生成自体が失敗する**ようにしてある:

- 先頭12文字が 2 件を超えて重複 → `SystemExit`
- 完全一致の重複文 → `SystemExit`
- 用語が文に含まれない → `assert`

この検証を厳しくしたことで、書いている最中に 2 種類の文法破綻が見つかった:

| 破綻 | 件数 | 原因 |
|---|---|---|
| `〜に対する改めたを…行った` | 223 | tail (完了形の述語) を名詞スロットで使っていた |
| `〜作業中に中に` / `〜受けにもかかわらず` | 13 | 既に助詞で終わる ctx の後ろに、さらに助詞を足していた |

そのため shape には設計ルールがある (スクリプト冒頭のコメント参照):

- `{ctx}` の直後は必ず読点。語や助詞を継ぎ足さない
- `{tail}` は文末 / `そう{tail}と` / `{tail}のは` の位置でのみ使う

## 4. TTS 合成と manifest 作成 (`scripts/run_cpu_dataprep.sh`)

```bash
DATA_REPO=Tsuka25/term2speech-data HF_TOKEN=hf_xxx \
    bash scripts/run_cpu_dataprep.sh data/generated_sentences.csv
```

内部で 4 段階:

1. `prepare_train_from_csv.py` — Sentance 内の Word を Reading に置換して `tts_text` を作る。
   正解テキスト (`sentence`) は漢字のまま残す
2. `synthesize_speech_kokoro.py --device cpu` — Kokoro-82M で `tts_text` を読み上げ。
   声は `jf_alpha` (女性)。**評価用は `jm_kumo` (男性) にして話者を分ける**
3. `build_whisper_manifest.py` — 音声と正解を突き合わせて manifest 化
4. `split_whisper_manifest.py` — train 1155 / dev 110 に分割

`DATA_REPO` を渡すと HF へ自動退避する。**渡さないとコンテナ終了で消える。**

所要時間の実測: 1265 文で約 32 分 (4 vCPU、約 1.5 秒/文)。GPU は不要。

## 5. 出来上がったデータの検証

作った後に必ず確認する。データが壊れていても学習は動いてしまう。

```bash
source .venv-cpu/bin/activate
python - <<'EOF'
import json, os
rows=[json.loads(l) for l in open("out/whisper_turbo/train_manifest.jsonl",encoding="utf-8")]
print("件数:", len(rows))
print("用語が正解文に無い  :", sum(1 for r in rows if r["term"] not in r["text"]))
print("読みがtts_textに無い:", sum(1 for r in rows if r["reading"] not in r["tts_text"]))
print("tts_textに漢字が残存:", sum(1 for r in rows if r["term"] in r["tts_text"]))  # 0 でないと学習が成立しない
print("音声ファイル欠損    :", sum(1 for r in rows if not os.path.exists(r["audio"])))
d=[r["duration_sec"] for r in rows]
print(f"音声長: 計{sum(d)/3600:.2f}h 平均{sum(d)/len(d):.1f}s 最短{min(d):.1f}s 最長{max(d):.1f}s")
EOF
```

### 音声を実際に聞かせる

数値が正常でも、無音や合成失敗は検出できない。ベースモデルに聞かせるのが確実:

```bash
python - <<'EOF'
import json, random
from faster_whisper import WhisperModel
rows=[json.loads(l) for l in open("out/whisper_turbo/train_manifest.jsonl",encoding="utf-8")]
m=WhisperModel("out/ct2_base_turbo", device="cpu", compute_type="int8")
random.seed(11)
for r in random.sample(rows,8):
    segs,_=m.transcribe(r["audio"], language="ja", beam_size=1)
    print("発話:", r["tts_text"])
    print("認識:", "".join(s.text for s in segs).strip())
    print("正解:", r["text"], "\n")
EOF
```

期待する結果は「**読みは合っているが漢字が違う**」。実測例:

| 発話 | ベースモデルの認識 | 正解 |
|---|---|---|
| しょうさんたいちっそ | 小三大地 | 硝酸態窒素 |
| ろっかくろむかごうぶつ | 六角炉無化合物 | 六価クロム化合物 |
| ひらいせつび | 飛来設備 | 避雷設備 |
| しんしゅくかとうかん | 新宿化等官 | 伸縮可とう管 |

音声が明瞭なのに誤変換されるなら、狙い通りのデータになっている。
逆に認識が全く読みと合っていなければ、TTS 側を疑う。

---

## 評価データ (学習データとは別に作る)

`data/test_sentences_100.tsv` (100文)。学習データと分ける点:

- **例文が別**。学習で使った文を読ませても評価にならない (重複ゼロを機械確認済み)
- **話者が別**。学習は `jf_alpha`、評価は `jm_kumo`
- 用語は**読みが検証済みのものだけ**を使う

最終的な評価は**実録音**で行うこと。合成音声だけで学習した場合、
合成音の音響特性に適応しただけで実音声では伸びない可能性がある。

---

## 落とし穴

- **コンテナ再起動でファイルが巻き戻ることがある。** git 管理外の
  `data/generated_sentences.csv` が古い版に戻り、それに気づかず TTS を回して
  HF に間違ったデータを上げた。再起動後は中身を検証する
  (例: 疑問文の件数が 210 前後なら新版、0 なら旧版)
- **HF へ上げ直すときは古いフォルダを消してから。** 混在が一番厄介
- 生成物 (`out/`) と用語CSV は `.gitignore` 済み。用語が公開リポジトリに出ないようにしている
