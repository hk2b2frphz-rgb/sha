# 第26回：登録語の優遇幅を訂正音声で校正する仮説

ユーザーは第25回を「弱い」、過去音声の遡及回収を「いらん機能」と評価。求めるものは「構造的にそれはうまくいくよね」と理解できる技術。どちらも推奨案から外す。誤置換防止という課題へのユーザー合意はまだない。

## 新しい仮説（特許性・効果未確認）

登録語への優遇量を、実音声では訂正語が競合語に勝ち、訂正区間を隠した対照入力では勝たない範囲として登録時に求める。範囲がなければその登録の自動優遇を有効化しない。ユーザーの一回の訂正を語別校正に使う。

候補固定・加算スコアの簡略モデル：訂正語 y と競合 c の無優遇スコア差を、実音声で d_real、対象区間を隠した音声で d_mask とする。語別加算量 b について d_real+b>0 かつ d_mask+b<=0、すなわち -d_real<b<=-d_mask を要求。b>=0、複数対照・余裕幅も考慮する。非線形のプロンプト長やLLM指示をこの加算量と同一視しない。

構造的に言えるのは、登録に使った固定候補・入力に対して二条件が成立すること。未来の発話の正確性、マスク入力の自然さ、競合候補網羅、同音語区別は保証しない。音声が影響したことは正解の証明ではない。候補語の読み等を採点できる基盤が前提。

## 近接文献

- [US12387718B2公報転載](https://patents.justia.com/patent/12387718)：検索出力で請求項1全文を確認。元音声と複数マスク音声の予測を比較し、内部言語モデルの偏りを除く技術。Google Patents本文取得失敗。原公報照合・日本対応未確認。音声を隠した比較それ自体は新規点としない。
- [Internal Language Model Estimation for Domain-Adaptive End-to-End Speech Recognition](https://arxiv.org/abs/2011.01991)：著者公開概要確認。内部言語モデル推定による分野適応。方式詳細未比較。
- US11423883B2、WO2020226789A1、US12051407B2、US11664021B2：文脈バイアスの検索候補、請求項精査前。
- [2023音響情報を使う後段訂正論文](https://arxiv.org/abs/2302.11192)：著者概要確認。音響参照による文脈訂正は既存。

特許として検討する差は「訂正登録時に上限と下限を算出し、可行範囲の有無で語別優遇を登録する」こと。既存の閾値校正・キーワード登録拒否・内部LM補正との組合せで容易とされ得る。差の先行技術調査は未了。実験未実施。

検索：patent speech recognition contextual biasing acoustic evidence subtract language prior counterfactual / speech recognition contextual biasing hallucination masked audio likelihood ratio correction / ASR counterfactual biasing / speech recognition audio internal language model subtraction patent。
