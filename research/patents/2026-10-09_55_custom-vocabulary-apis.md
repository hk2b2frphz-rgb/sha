# 第55回：日本語ASR APIの辞書機能と精度比較

確認日：2026-10-09。ユーザーはカスタム辞書が使えるAPIを日本語精度順で知りたい。実APIの実行・課金・音声送信はしていない。

## 精度順位の根拠と範囲

[koedeskの日本語比較表](https://koedesk.app/benchmark/)、[測定方法と利益相反](https://koedesk.app/blog/stt-benchmark/)を確認。2026年7月、FLEURS日本語200発話。単語指定機能を公式確認できた掲載モデルをCER順に抜き出すとMAI-Transcribe-1.5、Scribe v2、Chirp 3、Nova-3、AmiVoice -a-general。最初の二つは同一の統計的上位群。辞書追加後の専門語精度ランキングではない。主催者はScribe利用アプリ事業者。全サービスの一般順位・会話精度順位に拡張しない。

MAI-Transcribe-2は現在公式資料にあるが、1.5の測定値を2へ転用しない。標準Azure Speechにも転用しない。AmiVoiceの値も-a-generalの測定であり、全エンジンを代表しない。

## 辞書仕様の一次資料

| API | 確認した機能・制約 | 出典 |
|---|---|---|
| Azure MAI-Transcribe | phraseListによる語句ヒント。1.5の公式サンプル確認、現行2にも機能あり。今回確認したのはファイル送信のREST方式。public preview | [公式サンプル](https://microsoft-foundry.github.io/forgebook/notebook/mai-transcribe-1-5/)、[Learn](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe) |
| ElevenLabs Scribe v2 | keyterms。公式資料でbatch最大1000語、realtime最大50語。日本語対応。v2とv2 Realtimeの精度を混同しない | [公式](https://elevenlabs.io/docs/overview/capabilities/speech-to-text) |
| Google Chirp 3 | PhraseSet/model adaptation、最大1000句。日本語とstreaming対応 | [公式](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3) |
| Deepgram Nova-3 | keyterm prompting。日本語対応。読みの専用フィールドではない | [keyterm](https://developers.deepgram.com/docs/keyterm)、[言語](https://developers.deepgram.com/docs/models-languages-overview) |
| AmiVoice | ハイブリッドはwritten/spokenで表記と読みを登録。End to Endは単語強調。方式を分ける | [公式](https://docs.amivoice.com/amivoice-api/manual/user-dictionary) |
| Speechmatics | additional_vocabで表記、sounds_likeで読み。日本語は全角ひらがな／カタカナ。batch/realtime対応。同一比較表に不在のため順位未付与 | [公式](https://beta-docs.speechmatics.com/speech-to-text/features/custom-dictionary) |
| Amazon Transcribe | 日本語を含む対応言語でCustom Vocabulary。Phrase/DisplayAs。SoundsLikeとIPAは現在無視される。同一比較表に不在のため順位未付与 | [概要](https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary.html)、[形式](https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary-create-table.html) |

Qwenの[認識精度向上API資料](https://www.alibabacloud.com/help/ja/model-studio/improve-asr-accuracy)でもホットワードと文脈機能を確認。同一日本語比較結果、各モデル・リージョンの日本語ホットワード適用範囲は今回未確認のため順位に入れない。

## 推奨の意味

一般日本語精度の比較候補はScribe/MAI/Chirp。登録時の読み推定を検証する用途には、読みを明示的に渡せるAmiVoiceハイブリッドとSpeechmaticsが適するという機能上の判断。採用前には辞書有無で専門語正解率・誤挿入・確定遅延を測る。辞書、語句ヒント、学習済みカスタムモデルは別物。

## 直前のブレストの保存（中断）

ユーザーはリスペリングのように、小さな変更と改善理由が直結する案を希望。仮説として「専門語部分の音声を隠しても当てられる合成例かを調べ、文脈依存を考慮して学習例を選別」を探索したが、ユーザーがAPI調査へ切り替えたため未完了・未採用。音声マスク比較は既調査US12387718B2と近接（今回Google本文取得失敗）。[Google 2020の合成対象テキスト選択](https://research.google/pubs/improving-speech-recognition-using-gan-based-speech-synthesis-and-contrastive-unspoken-text-selection/)、[Sentence-Select 2022](https://research.google/pubs/sentence-select-large-scale-language-model-data-selection-for-rare-word-speech-recognition/)は機関要旨確認。特許性・効果の主張はしない。
