# 第53回：Speech LLM・duplex・Qwen-ASR・Seed-ASR

2026-10-09。ユーザーはこれらを研究の起点として技術的特徴を確認したい。今回は基盤技術の整理であり、新規発明・特許性の判定ではない。

## 分類

Speech LLMは音声表現をLLMで扱う技術の総称として解釈。duplexは話す／聞くを同時に扱う対話方式。Qwen-ASRとSeed-ASRは具体的な認識モデル。ストリーミング文字起こしと双方向の音声対話を混同しない。

## 一次資料で確認した特徴

- [Qwen3-ASR論文](https://arxiv.org/html/2601.21337v1)、[公式実装](https://github.com/QwenLM/Qwen3-ASR)：AuT音声エンコーダ→projector→Qwen3 LLM。0.6B/1.7B版、日本語を含む30言語、同一モデルでoffline/streaming。重み・推論・FT手順公開。背景文脈を与えられるが、ASR用SFTは自由な指示追従を意図的に抑制。汎用対話モデルと同一視しない。別モデルのForcedAlignerは正解文字列の保証ではない。§2と公式README確認。
- [Seed-ASR論文](https://arxiv.org/html/2407.04675v2)、[公式ページ](https://bytedancespeech.github.io/seedasr_tech_report/)：LUISE音声エンコーダ→converter→LLM。連続音声表現と自然言語文脈から文字起こし。Context SFTは〈文脈・音声・正解〉で訓練。§3.4のjoint beam searchは文脈あり／なしのスコアを併用し、文脈なしスコアによる候補枝刈りで幻覚を抑制。多言語版は日本語の学習・評価を含む。Qwenのような公開重み・FT一式は今回未確認。§3.2–3.5、4.2確認。
- [Moshi論文](https://arxiv.org/html/2410.00037v2)、[公式実装](https://github.com/kyutai-labs/moshi)：full-duplexの代表。Mimi音声codecで音声をトークン化し、利用者・システムの音声を別ストリームとして同時に扱う。Inner Monologueは音声に先行する整列済みテキストを使う。発話重なり・割込み等を扱う構造であり、専門語認識精度の保証ではない。§3、要旨確認。
- [Qwen-Audio-3.0-ASR論文](https://arxiv.org/html/2609.07549v2)：2026年9月の別レポート。Qwen3-ASR公開モデルと仕様を混同しない。MoE LLM、指示条件付き認識、重要語／広い候補群の二段階hotword、履歴文脈、同一デコード内の整文、専用Streaming版を記載。§3.4は正しい／競合する／ノイズを含む／空の文脈を混ぜた学習も説明。公開重み・再現性は未評価。

## 今回の判断

実験基盤は公開重みと日本語対応があるQwen3-ASRが扱いやすい、文脈利用の設計はSeed-ASRが参考になる、対話タイミングはMoshi等のduplexを別軸にする、という研究上の推奨。性能順位の断定ではない。音声と文脈を認識中に統合する処理は既存。いずれも利用するだけで自動的に学習し続けるわけではなく、文脈入力・重み更新・対話制御を分けて検討する。

実装・日本語専門語の評価・新規特許検索は未実施。直前のリスペリング論文はユーザー申告で自社論文と判明（第52回に追記）。
