# ローカル日本語ASRと専門語指定（第56回）

確認日：2026-10-09。ユーザーはB200を利用可能。大型モデルも許容。API比較に続く実装候補調査であり、特許性の判断ではない。

## 結論と限界

最初の比較候補はQwen3-ASR-1.7B、VibeVoice-ASR、Fun-ASR-Nano-2512。大型の研究比較枠としてQwen3-Omni-30B-A3B-Instruct、日本語の対照としてWhisper large-v3／Kotoba-Whisper v2.0を置く。これは機能・用途からの評価優先順位で、日本語精度の実測順位ではない。全候補を同一条件で比較した日本語専門語ベンチマークは今回確認できていない。モデルのダウンロード・実行・GPU計測なし。

## 公式仕様の確認

| モデル | 規模 | 専門語指定と位置付け | 出典 |
|---|---|---|---|
| Qwen3-ASR-1.7B | 名称1.7B、HF全体表示約2B | 日本語、オフライン／ストリーミング。公式サンプルのcontextに語句を指定可能。読み付き辞書として強制対応させる機能とは区別。Apache-2.0。 | [モデル](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)、[context使用例](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_transformers.py) |
| VibeVoice-ASR | 公式呼称7B、HF全体表示9B | Customized Hotwordsに専門語・背景情報。日本語のMLC評価を公開。長時間音声・話者分離・時刻を一体処理。MIT。 | [モデル](https://huggingface.co/microsoft/VibeVoice-ASR)、[公式説明・日本語評価](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md) |
| Fun-ASR-Nano-2512 | 0.8B（公式表） | 日本語対応。hotwordsの語句リスト引数あり。日本語専門語への効果は要実測。7.7BのFun-ASRは別モデルで、公式表は非公開扱い。Apache-2.0。 | [モデル・使用例](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| Qwen3-Omni-30B-A3B-Instruct | 30B級MoE | 日本語音声入力、音声認識と指示・文脈を扱える汎用モデル。専門語の説明を添える実験候補。専用辞書APIやASR専用モデルへの精度優位を確認したわけではない。 | [モデル](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Kotoba-Whisper v2.0 | 756M | 日本語向け学習。公式評価でもlarge-v3との優劣はデータセットで変わる。読み付き辞書を主要機能とするモデルではない。 | [モデル・比較表](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) |

## 評価案

### 追補：Qwen3-ASRのcontext長

ストリーミングでもinit_streaming_state(context=..., language=..., chunk_size_sec=2.0)で文脈を渡せる。公式ラッパーはvLLMのみ対応し、標準チャンク2秒。これは処理待ちを含む最終遅延の保証値ではない。単語数上限は未規定。例えば区切り込み平均10トークン/項目と仮定すると100語=1,000、1,000語=10,000トークン。これは換算仮定で、実際の専門語リストをトークナイズした実測値でも、精度維持を保証した語数でもない。評価は100→300→1,000語等で増やして実施する案。

公式[config.json](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/config.json)のtext_config.max_position_embeddingsは65,536。これはcontext文字列専用枠ではなくモデル全体の系列長設定。音声トークン・特殊トークン・生成出力の枠も必要。[公式推論ラッパー](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)ではcontextをsystemメッセージへ渡しており、確認した処理にはcontextだけを固定長で切り詰める上限は見当たらない。実際の許容長はサーバー設定等でも制限される。max_new_tokensの512等は生成出力の上限。長大な専門語リストでの認識品質を保証する数値ではなく、推奨context長は今回確認できていない。

- 同じ実音声に、専門語の指定なし／表記だけ／読み・意味も記した文脈（受け付けるモデルのみ）を与える。
- 全文CERに加え、専門語の正答率、専門語を含まない発話への誤挿入、発話終了後の確定遅延を測る。
- 「終沈（しゅうちん）」等を文脈に書くのは実験条件であり、公式の読み登録形式ではない。入力できることと読みを正しく利用することを分ける。
- B200があっても大規模化を精度向上と同一視しない。まずBF16で品質比較。実メモリ・遅延は音声長、キャッシュ、同時処理数、推論実装を含めて計測する。

参考確認：Voxtral-Mini-4B-Realtime-2602とNemotron-3.5-ASR-streaming-0.6bは日本語ストリーミング候補。ただし各公開チェックポイントのカスタム語彙機能は今回未確認。商用APIの機能を公開重みに転用して記載しない。[Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)、[Nemotron](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)。
