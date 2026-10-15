# 第69回：TTS候補の読み選別と正誤例を使う損失

確認日：2026-10-13。問い：日本語等で複数TTSから合成音声を生成し、再認識して読みの正しい候補を採用する研究、又は正読／誤読を損失に反映する研究があるか。TTSは音声生成、書き起こしはASRと解釈。今回は非特許文献の確認で、特許性判断ではない。

## 結論と区別

直接近い論文がある。ただし、複数の独立TTSモデルを使う方式と、同一TTSの複数話者・複数サンプリングを区別する。また、TTSを改善する学習と、合成音声でASRを改善する学習を区別する。以下の文献を「日本語で複数の独立TTSから候補を採用する構成」の完全一致と断定しない。

## 本文で確認した一次資料

### TKTO：正読／誤読を使う日本語TTS学習

[Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](https://aclanthology.org/2026.acl-short.59/)（Kotoge・Sasaki、ACL 2026 short、初稿2025-10-07）。[本文](https://aclanthology.org/2026.acl-short.59.pdf) §3、§4.1、Table 1。

「辛い」のkarai／tsuraiを含む5,000文について、男女の話者それぞれ5音声を生成。正読かつ低CERを望ましい例、誤読かつ高CERを望ましくない例にする。CERにはWhisperを利用。正誤ペアがない例も利用できるKTOをトークン単位へ拡張し、重要な音声トークンを重くする。学習対象はCosyVoice2系TTS。Table 1の男性条件で読み正解率0.668→0.958だが、主に「辛い」の評価で一般専門語の保証ではない。複数独立TTSではない。正読／誤読ラベルを具体的にどう付与したかの詳細は追加確認事項。

### Sarashina2.2-TTS：かなASRで合成例を選別

[Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis](https://arxiv.org/html/2606.25369v1)（2026-06-24、技術報告）。§3.2.1–3.2.2、§4.1。

特定の漢字の読みを必要とする例文と全文かなをLLMで生成。読み・アクセントを制御する内部TTSで、多様な話者プロンプトから合成する。かなASRで対象漢字区間と全文を検証し、誤り閾値を超える音声を除外してTTSの追加学習に利用。複数独立TTSではない。読みを制御するPronSteeringは公開モデルに含まれないと本文が明記。

### Kana-domain ASR rewards：発音誤りを報酬にする

[Pronunciation-Oriented Reinforcement Learning for Japanese Text-to-Speech with Kana-Domain ASR Rewards](https://arxiv.org/html/2610.07575v1)（2026-10-06、プレプリント）。§2–4、Table 1。

同一TTSから入力ごとに16候補を生成し、かなWhisperと正解かなのCERから報酬を作りGRPOでTTSを更新。通常表記CERを報酬にする条件に対し、対象漢字かな誤り率9.65%→7.17%（相対25.7%減）。0.5Bの一モデルで、正解読みが必要。KL正則化なしでは音声長が過度に伸びる問題も報告。ASRのFTではない。

### Self-Refining / Twister：ASRを改善するための音素比較

[A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data](https://arxiv.org/html/2506.11130v1)（2025-06、プレプリント）。§III-B–C。

台湾華語のTTS合成音声を補助ASRで再認識し、入力文と認識文を音素化してPERで選別する。本文は同音語等の表記差を合成失敗と誤判定しないため、と説明する。選別・MFA整列・拡張後にWhisperを学習。日本語や複数独立TTSの実証ではないが、今回のASR改善目的に近い。

### CONTRAPROST：複数話者候補からの採用

[Speech Is More than Words: Do Speech-to-Text Translation Systems Leverage Prosody?](https://aclanthology.org/2024.wmt-1.119/)（WMT 2024）。[本文](https://www2.statmt.org/wmt24/pdf/2024.wmt-1.119.pdf) §2.4、Appendix C。

OpenAI TTSの6話者で音声候補を生成し、WhisperのWERが0でない候補を除外、残りを韻律の評価で選ぶ。候補生成→ASR検証→採用の近接例だが、単一サービスの複数話者。合成言語は英語で、日本語は翻訳の出力先。日本語合成音声の実験と混同しない。

### W3AR：ASR内部情報を語単位のTTS報酬に利用

[Speech Recognition Model Improves Text-to-Speech Synthesis using Fine-Grained Reward](https://arxiv.org/html/2511.17555v1)（初稿2025-11、[AAAI掲載本文](https://ojs.aaai.org/index.php/AAAI/article/download/40631/44592)）。Method / ASR-driven Reward Modeling / Policy Optimization。

Whisper等のクロスアテンションの集中度・単調性から語単位の報酬を作り、複数サンプルの相対報酬でTTSを最適化。文全体だけでなく局所的な合成不良を損失に反映する例。日本語専門語への直接実証ではない。

## 本案の設計で重要な点（以下は技術的整理）

同音異義語は「終沈／集塵」のように読みが同じで表記・意味が異なる問題。異読語は「辛い」のように同じ表記で読みが異なる問題。TKTOとSarashinaの中心は後者。

| 合成音声の実際の発音 | ASRの出力 | ASR学習での意味 |
|---|---|---|
| 正しい | 正しい表記 | 基本的な教師対として利用可能 |
| 正しい | 誤った表記 | 改善したいASR難例。TTS失敗として一律除外しない |
| 間違っている | 正しい表記 | ASRが文脈で補った可能性。文字一致は発音品質を保証しない |
| 間違っている | 間違った表記 | 教師文との不整合を確認し、除外・再合成等を検討 |

発音の真値は自動的には得られない。確認済みの読み、かなASR等の検証器、必要時の聴取を使う。かなASR自体にも誤りはある。完全に同じ読みの語は、かな比較だけでは表記を選べない。

ASRを学習する場合、同じ音声に対する正解／誤認識テキストを比較する損失と、正読／誤読音声を比較するTTSの損失を区別する。誤読音声に元の表記を教師として付けると音と表記の不整合を学習させうる。学習対象ASRが正解した例だけ採用すると、修正したい難例を落とす選別バイアスが生じうる。いずれも設計論点であり、この調査で新規性を認定したものではない。

複数独立TTSによる日本語候補生成と、かなでの発音品質／表記でのASR難易度を分離した採用・重み付けを一体化した研究の完全一致は今回未特定。未発見を新規性と解釈しない。ASR側の選好損失まで同一構成で検証した文献、関連特許は追加調査事項。
