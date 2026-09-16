# 近接先行技術索引

## 第48回追加（2026-09-16）

詳細：[出願候補の外枠・合理性・特許性](2026-09-16_48_invention-framework-and-patentability.md)。以下は補強案の境界確認。登録時の競合検出・対話を単独の固有点にしない。

| 文献 | 論点 | 確認範囲 |
|---|---|---|
| [US5754977A](https://patents.justia.com/patent/5754977) | 登録時に新旧音声等の混同を検出、通知・保留・変更 | 登録1～3・7～10・13、FIG.4等の本文。日本未調査 |
| [US20220392432A1](https://patents.justia.com/patent/20220392432) | 訂正履歴＋個人クラス＋アプリ文脈で候補追加・再順位付け | 公開1・4・7、FIG.4～5。全項・日本未精査 |
| [US20150106100A1](https://patents.google.com/patent/US20150106100A1/en) | 経験的混同度、新語追加の既存語への悪影響 | 公開1～4、Maximum Utility Grammar Augmentation。全項・日本未精査 |

## 第47回の更新（2026-09-16、過去の評価より優先）

詳細と差分・検索ログは[第47回](2026-09-16_47_collision-mapping-prior-art.md)。衝突は置換表の同一キーに複数正解語が対応すること。JP6718182B1の1対1限定という旧説明は撤回。

| 文献 | 主な論点 | 今回の確認レベル |
|---|---|---|
| [JP7414078B2](https://patents.google.com/patent/JP7414078B2/ja) | 同一置換元の衝突を生成時に一候補へ解消 | 登録全8項、図8～9・【0044】等 |
| [JP2022526467A](https://patents.google.com/patent/JP2022526467A/ja)／US20220270595A1 | 実発話→誤認識対応表、頻度・品詞条件 | 日本公開1・5・7～9を追加確認、米対応項も照合 |
| [US20090254819A1](https://patents.google.com/patent/US20090254819A1/en) | 1対多訂正辞書と候補別文脈スコア | 公開1・4・6～8・15・17・19～21、本文【0097】【0108】等。B2未精査 |
| [JP2025514770A](https://patents.google.com/patent/JP2025514770A/en) | 誤認識ペアと文脈による後修正 | 公開1～16・18～22関連記載、本文【0037】【0039】。全24項対比未了 |
| [US20250335933A1](https://patents.google.com/patent/US20250335933A1/en) | 定義質問・保存、定義競合時の文脈区別 | 本文FIG.3・【0145】を追加確認。全項未精査 |
| [JP7322782B2](https://patents.google.com/patent/JP7322782B2/ja) | 音声住所の同音誤変換を地域情報で置換 | 登録1～2、瓦町等の本文例。全6項未精査 |
| [US12633290B2](https://patents.google.com/patent/US12633290B2/en)／US20230252994A1 | 音素候補と分野・利用者文脈による後修正 | 登録1～10、FIG.7等。全18項・日本対応未精査 |
| [JP6481643B2](https://patents.google.com/patent/JP6481643B2/ja) | 置換語ペア＋文脈保存による要求変更の先読み | 登録1、図2等。ASR誤訂正と同一視しない |
| [JP2018045001A](https://patents.google.com/patent/JP2018045001A/en) | 話題別置換表 | 実施例3・表4・図8を再確認 |
| [JP6718182B1](https://patents.google.com/patent/JP6718182B1/ja) | 誤変換辞書の対応付けと候補取得 | 登録1・3再読。1対1／解析1回限定は撤回 |


更新：2026-09-15。ここに載せるのは、この調査で**本文又は請求項を確認し、設計仮説との関係を個別メモに記録した文献**である。検索結果に一度現れただけで本文未確認の文献は載せない。権利の現況は、特記のない限りこの索引では判断しない。

## 第45回の精査更新（古い確認レベルより優先）

第46回追加（詳細は[第46回](2026-09-15_46_homonym-context-two-pass.md)）：[US20210142789A1／US11961511B2](https://patents.google.com/patent/US20210142789A1/en)は低信頼ASR領域を検出し、文脈手掛かり・ベクトル類似度・音韻距離・n-gram・埋込みで置換する二段／複数モデルの誤り解消。 [US9558740B1](https://patents.google.com/patent/US9558740B1/en)は曖昧性解消が必要かを第1段階で判定し、必要時だけ第2段階で候補を選ぶ。 [CN115019787B](https://patents.google.com/patent/CN115019787B/en)は信頼度が拮抗する同音候補を検出し、説明語彙・言語モデルを検索して候補を提示。今回案の「同じ音響キーに複数の正解専門語と用例・意味を紐付け、衝突時だけ第2パスで候補固有情報を使う」はこの差分として整理。ただし、2パス・文脈補正・同音語判定自体は既知。

第45回追加（詳細は[第45回](2026-09-15_45_consensus-enrollment-prior-art.md)）：[JP6718182B1／JP2020184024A](https://patents.google.com/patent/JP6718182B1/ja)は正しい入力用語を音声データ化→音声解析し、異なる解析語を誤変換語として辞書登録、将来の正しい候補抽出までを請求項1～3で直接開示。[US20220270595A1／JP2022526467A](https://patents.google.com/patent/US20220270595A1/en)は専門語を話者が発音し、ASRの誤認識語を対応表に登録、複数話者の複数誤認識語、頻度・品詞条件による置換まで開示。今回案の広い「実発話→誤認識表記→正解表記の置換登録」は既知性が高い。

同回の構成要素確認：[JP4852448B2](https://patents.google.com/patent/JP4852448B2/ja)はn-best候補＋人手訂正から誤り修正モデルを更新（請求項1～2、4）。[JP5148671B2](https://patents.google.com/patent/JP5148671B2/ja)は複数方式又は複数回の認識結果の差分から誤認識部分を抽出（請求項5）。[JP4270770B2](https://patents.google.com/patent/JP4270770B2/ja)は再発話と音響的対応付けによる候補置換（請求項1～5）。[US7529668B2](https://patents.google.com/patent/US7529668B2/en)は複数発音の音素アラインメント・多数派コンセンサスで精製辞書を作成（請求項1、14～18、20、33～37）。[US6122613A](https://patents.google.com/patent/US6122613A/en)は異なる複数認識器の結果統合（請求項1～4）。[US20170004831A1](https://patents.google.com/patent/US20170004831A1/en)はn-best、再発話、手修正、元音声との対、将来モデル更新を開示。[US8280733B2](https://patents.google.com/patent/US8280733B2/en)は音響距離・信頼度・同一発音の一定回数による選択的追加。[US8977547B2](https://patents.google.com/patent/US8977547B2/en)は複数発話の音響安定性閾値、再発話要求、失敗分岐を登録処理として直接開示。複数ASR・n-best・反復・安定性閾値の単純加算は固有点としない。

第45回の暫定差分：登録時に誤認識候補の再現性を定量化し、安定時だけ置換規則へ昇格、不安定時は追加発音又は保留へ分岐する制御。ただし、未開示と確定しておらず、特許性も未判定。

第44回追加候補：[US10388272B2](https://patents.justia.com/patent/10388272)は複数ASRへの仮説系列共有とn-best再順位付けの検索参照、[US6212498B1](https://patents.google.com/patent/US6212498B1/en)は発音による音響モデル登録、詳細は[第44回](2026-09-15_44_consensus-enrollment-brainstorm.md)。複数ASR・n-best・反復発音を単独の新規点と扱わない。

第43回追加（詳細は[第43回](2026-09-15_43_probe-recognition-replacement-enrollment.md)）：[JP2018045001A](https://patents.google.com/patent/JP2018045001A/en)はASR出力の置換表・信頼度・話題条件、公開全11項と関連本文確認。[JP7414078B2](https://patents.google.com/patent/JP7414078B2/ja)は類似語と母音／子音の優先度による変換表生成、登録1～6と本文冒頭確認、全8項未精査。[JP7268389B2](https://patents.google.com/patent/JP7268389B2/en)は本文のOCR誤認識辞書・登録箇所のみ確認、音声の直接一致例から除外。いずれも今回公式現況未確認。

第41回追加：[US20250335933A1](https://patents.google.com/patent/US20250335933A1/en)は未知語判定・定義質問・回答の辞書保存・将来利用。FIG.3の302～316と対話例、公開請求項1・対応表を確認。公開全20項・公式現況未精査。日本はGoogle対応表で未確認。詳細は[第41回](2026-09-15_41_voice-enrollment-meaning-clarification.md)。補助例US20090281786A1は検索出力の本文・公開1・6等を確認、質問は意味取得でなく登録可否の確認。全体精査・日本対応未確認。

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [JP5819924B2](https://patents.google.com/patent/JP5819924B2/ja)／JP2014067062A | 単語内文字方法、文字構成、訂正時語彙構築、日本語 | 登録全16項、親出願対応表。現況・包袋未確認 | [第40回](2026-09-15_40_kanji-repair-patentability-review.md) |
| [JP5622566B2](https://patents.google.com/patent/JP5622566B2/ja)／JP2010525415A | 単語を発声して文字指定、訂正・レキシコン更新 | 登録全6項、図4等の本文。候補一覧だけという旧説明を訂正 | [第40回](2026-09-15_40_kanji-repair-patentability-review.md) |
| [US20260065907A1](https://patents.google.com/patent/US20260065907A1/en) | LLM自然言語訂正、構造／文脈／意味と発音、元音声・ASR複数候補 | 公開全20項とFIG.1～4周辺等。対応表はUS・WO2026051679A1、日本は未確認 | [第40回](2026-09-15_40_kanji-repair-patentability-review.md) |
| [JP2002287787A](https://patents.google.com/patent/JP2002287787A/ja) | 説明語・文字・文脈キュー、日本語「の」、文字検証と音響類似性 | 公開請求項1・14・18・25～26・31～40、本文【0027】～【0028】【0042】等を比較 | [第40回](2026-09-15_40_kanji-repair-patentability-review.md) |
| [JP4537755B2](https://patents.google.com/patent/JP4537755B2/ja) | 履歴＋テンプレートから訂正文法を作り認識・訂正 | 登録請求項1を再確認。全項の権利対比は未了 | [第40回](2026-09-15_40_kanji-repair-patentability-review.md) |
| [JP2021144271A](https://patents.google.com/patent/JP2021144271A/en) | グループ名・属性記号による文字入力候補提示 | 要約・本文冒頭。音声漢字訂正の中心比較としては優先度低、請求項未精査 | [第40回](2026-09-15_40_kanji-repair-patentability-review.md) |

## 既存索引

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [JP7771015B2](https://patents.google.com/patent/JP7771015B2/ja)／JP2024015818A | 修正履歴、修正前後の差分文字列、話題・時刻の重み、後段訂正モデル更新 | 登録請求項1〜13と実施形態を確認。訂正は請求項3以下 | [第19回](2026-09-15_19_retrieval-grounded-post-correction.md) |
| [US20260011325A1](https://patents.google.com/patent/US20260011325A1/en) | ASR仮説＋文脈リストによるモデル外の後段スペル訂正 | 公開本文・例示を確認 | [第19回](2026-09-15_19_retrieval-grounded-post-correction.md) |
| [JP2025135075A](https://patents.google.com/patent/JP2025135075A/en) | 利用者固有語彙＋LLMによる後修正 | 公開請求項を確認 | [第19回](2026-09-15_19_retrieval-grounded-post-correction.md) |
| [WO2006113350A2](https://patents.google.com/patent/WO2006113350A2/en) | ASR出力と最終編集文書の対による後段訂正モデル適応 | 公開請求項・本文を確認 | [第19回](2026-09-15_19_retrieval-grounded-post-correction.md) |
| [US9858038B2](https://patents.google.com/patent/US9858038B2/en) | 集約利用者置換データ・個人置換データによる訂正候補 | 登録請求項・本文を確認 | [第18回](2026-09-15_18_correction-transfer-rag.md)、[第19回](2026-09-15_19_retrieval-grounded-post-correction.md) |
| [JP2008243080A](https://patents.google.com/patent/JP2008243080A/ja) | 類似用例検索、音韻・意味属性による音声翻訳の認識訂正 | 公開本文を確認 | [第19回](2026-09-15_19_retrieval-grounded-post-correction.md) |
| [US10522133B2](https://patents.google.com/patent/US10522133B2/en) | 過去の誤認識・訂正履歴を照合し、自動訂正又は警告 | 登録請求項1ほかを確認 | [第20回](2026-09-15_20_explicit-correction-loop.md) |
| [US9514743B2](https://patents.google.com/patent/US9514743B2/en) | 訂正要求発話の検出、訂正発話の複数仮説から候補生成 | 登録本文を確認 | [第20回](2026-09-15_20_explicit-correction-loop.md) |
| [JP3718088B2](https://patents.google.com/patent/JP3718088B2/ja) | 訂正用音声から候補生成・選択、辞書更新 | 登録請求項1〜3を確認 | [第20回](2026-09-15_20_explicit-correction-loop.md) |
| [JP4537755B2](https://patents.google.com/patent/JP4537755B2/ja) | 対話履歴を用いた訂正発話検出と直前発話の修正 | 登録請求項・本文を確認 | [第20回](2026-09-15_20_explicit-correction-loop.md) |
| [JP2001306091A](https://patents.google.com/patent/JP2001306091A/ja) | 訂正用の再発話と元発話区間の対応付け | 公開本文を確認 | [第21回](2026-09-15_21_target_then_phonetic_retrieval.md) |
| [WO2003025904A1](https://patents.google.com/patent/WO2003025904A1/en) | 訂正語を音素列化し、認識結果中の類似音素列で訂正位置を特定 | 公開請求項・本文を確認 | [第21回](2026-09-15_21_target_then_phonetic_retrieval.md) |
| [US20210043196A1](https://patents.google.com/patent/US20210043196A1/en) | 音素記号列を辞書の対応語で置換 | 公開請求項を確認 | [第21回](2026-09-15_21_target_then_phonetic_retrieval.md) |
| [US20050203751A1](https://patents.google.com/patent/US20050203751A1) | OOV訂正、訂正語と誤認識区間の音素アラインメント、語彙追加 | 公開本文を確認 | [第21回](2026-09-15_21_target_then_phonetic_retrieval.md) |
| [US20210005204A1](https://patents.google.com/patent/US20210005204A1) | 誤認識／正しい音素列の履歴、音素編集距離等による候補 | 公開本文を確認 | [第21回](2026-09-15_21_target_then_phonetic_retrieval.md) |
| [US8204739B2](https://patents.google.com/patent/US8204739B2/en) | 利用者コミュニティの訂正・新語共有と確率調整 | 登録本文を確認 | [第18回](2026-09-15_18_correction-transfer-rag.md) |
| [US8532994B2](https://patents.google.com/patent/US8532994B2/en) | 本人・ソーシャルグラフの他者の語彙による個人化 | 登録本文を確認 | [第18回](2026-09-15_18_correction-transfer-rag.md) |
| [JP6488588B2](https://patents.google.com/patent/JP6488588B2/ja) | ソーシャルグラフのデータを連絡先語彙に利用 | 本文確認済み。請求項比較は未了 | [第18回](2026-09-15_18_correction-transfer-rag.md) |
| [US20190035385A1](https://patents.google.com/patent/US20190035385A1) | 正誤入力、訂正、n-best候補、訂正による学習 | 公開請求項を確認 | [第18回](2026-09-15_18_correction-transfer-rag.md) |
| [US20200105247A1](https://patents.google.com/patent/US20200105247A1/en) | クエリ人気度と音韻類似度による訂正候補の再順位付け | 公開本文を確認 | [第18回](2026-09-15_18_correction-transfer-rag.md) |

## 使い方

### 第36回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US9953644B2](https://patents.google.com/patent/US9953644B2/en)／US20160155445A1 | 概念の存在／正しさを別々に判定し、怪しい区間への対象指示型質問を生成 | 登録請求項1・15、本文の入力特徴・例、Google対応表を確認。日本同族は同表で未確認 | [第36回](2026-09-15_36_decide-then-clarify-prior-art.md) |
| [US11823659B2](https://patents.google.com/patent/US11823659B2/en)／US20210183366A1 | 低信頼度又は曖昧さ検出、追加情報要求、個人用認識キー更新、全体ASR学習 | 登録請求項1・9〜15、Google対応表を確認。日本同族は同表で未確認 | [第36回](2026-09-15_36_decide-then-clarify-prior-art.md) |
| [JPWO2014041607A1](https://patents.google.com/patent/JPWO2014041607A1/en) | 修正対象語と同音・異表記の候補群を表示し、利用者選択で置換 | 本文の候補生成・選択・置換処理を確認。請求項比較未了 | [第37回](2026-09-15_37_japanese-semantic-component-repair.md) |
| [JP2002287787A](https://patents.google.com/patent/JP2002287787A/ja) | 同音異義文字、文字列・語句・文脈キューを用いる言語モデル | 公開本文の同音異義語・文脈キュー・文字識別処理を確認。請求項比較未了 | [第37回](2026-09-15_37_japanese-semantic-component-repair.md) |
| [JP2013250931A](https://patents.google.com/patent/JP2013250931A/ja) | 音声認識語の意味を検索し、意味不明なら他の利用者へ質問し回答を取得・表示 | 公開本文の意味検索、質問、回答取得処理を確認。請求項比較未了 | [第38回](2026-09-15_38_definition-after-semantic-correction.md) |
| [JP3710157B2](https://patents.google.com/patent/JP3710157B2/ja) | 語の読みと漢字構成要素の符号を併用する漢字語句入力 | 登録本文の読み・構成要素併用、候補絞込み・選択処理を確認。自然言語の字義説明との比較は未了 | [第39回](2026-09-15_39_semantic-kanji-repair-dialogue.md) |

### 第32回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US12444412B2](https://patents.google.com/patent/US12444412B2/en)／US20250095641A1 | LLMがツール・検索・ユーザー入力を使いASRエンティティ訂正。メモリはタスク応答キャッシュ | 登録全20項と本文関連箇所確認。Google対応表で日本同族未確認 | [第33回](2026-09-15_33_samsung-amazon-close-review.md) |

### 第30回精査追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US10621285B2](https://patents.google.com/patent/US10621285B2/en)／US20200034423A1 | GSP意味表現、対話依存構造、質問回答による知識更新 | 登録全21項・公開本文の関連実施形態確認。公式現況・包袋未確認 | [第30回](2026-09-15_30_elemental-cognition-review.md) |
| [JP7178995B2](https://patents.google.com/patent/JP7178995B2/ja)／JP2019526139A | 同族の日本請求項。質問・GSP更新・後続利用、信頼度で終了 | 登録全17項・対応関係確認。原簿・包袋未確認 | [第30回](2026-09-15_30_elemental-cognition-review.md) |

### 第28回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US20200034423A1](https://patents.google.com/patent/US20200034423A1/en) | 対話エンジンが文中の語義をユーザーに質問し知識を獲得 | 本文FIG.16関連確認。ASR後修正全工程・日本対応未確認 | [第28回](2026-09-15_28_system-initiated-meaning-question.md) |
| [JPH075891A](https://patents.google.com/patent/JPH075891A/ja)／JP3397372B2 | 質問回答から認識語彙を変更し元音声の未知語を再評価 | 公開要約・本文関連箇所確認。登録版未精査 | [第28回](2026-09-15_28_system-initiated-meaning-question.md) |

### 第27回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US20260065907A1](https://patents.google.com/patent/US20260065907A1/en) | 自然言語の文脈・意味説明と発音を用いるLLM訂正 | 公開概要・本文発明概要を確認。全請求項・日本対応未確認 | [第27回](2026-09-15_27_user-usage-description.md) |

### 第26回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US12387718B2](https://patents.justia.com/patent/12387718) | 音声マスク前後の予測比較による内部言語モデル偏りの除去 | 公報転載の請求項1を検索出力で確認。原公報・日本対応未確認 | [第26回](2026-09-15_26_registration-bias-window.md) |

### 第25回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US20090106027A1](https://patents.google.com/patent/US20090106027A1/en)／JP5094120B2 | 音声部分置換・結合による認識用標準パターン生成 | 米公開本文関連箇所と対応表確認。日本本文・登録請求項未精査 | [第25回](2026-09-15_25_context-invariant-enrollment.md) |
| Nuisance Attribute Projection（2007、非特許文献） | 不要変動方向を抑える比較 | MIT書誌・関連ISCA論文概要確認。原論文数式比較未了 | [第25回](2026-09-15_25_context-invariant-enrollment.md) |

### 第24回追記

| 文献 | 主な論点 | 確認レベル | 詳細メモ |
|---|---|---|---|
| [US12444402B2](https://patents.google.com/patent/US12444402B2/en)／US20230115538A1 | 音響埋込みで固有名詞を検索、候補選択時に埋込みと選択名称をDB登録 | 登録全14項と本文関連箇所確認。Google対応表で日本同族未確認 | [第33回](2026-09-15_33_samsung-amazon-close-review.md) |
| [US20230402034A1](https://patents.google.com/patent/US20230402034A1/en) | 過去の利用者訂正を代替仮説へ反映 | 公開本文確認。登録版US12322388B2は未精査 | [第24回](2026-09-15_24_audio-memory-prior-art.md) |

以下は索引の更新方針。

- 新しい近接文献は、本文又は請求項を確認した後に必ずこの表へ追加する。
- 「確認レベル」は、文献全体の権利解釈ではない。どこまで読んだかの監査記録である。
- 日本の権利現況が重要な文献は、別途J-PlatPatの公式画面で確認し、確認日と結果を個別メモへ追記する。
