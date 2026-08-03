# 専門用語向け Whisper decoder FT（Miltoka・3ジョブ構成）

`data/annotations.tsv` の `term`（表記）と `reading`（読み）から学習文と音声を作り、
`whisper-large-v3-turbo` の **encoderを凍結したままdecoderだけ** fine-tuneし、
CTranslate2形式まで変換する手順です。評価ジョブは既存のものをそのまま使います。

```text
annotations.tsv
  └─ 1. xan_s / small: Qwen3.6-27B + vLLM → corpus.jsonl
       └─ 2. xvn_s / middle2: Qwen3-TTS × 4 GPU → train/dev manifest
            └─ 3. xan_s / middle: Whisper decoder FT → HF → CTranslate2
```

各段は同じ `RUN_ROOT` を使います。前段の成果物が完全でなければ次段は失敗し、
生成済みの文・WAVは再開時に再利用されます。

## 入力

UTF-8のTSVを用意します。読みはLLMに推測させず、この値だけを正として使います。

```tsv
term	reading
活性汚泥法	かっせいおでいほう
最終沈殿池	さいしゅうちんでんち
```

空欄、同一用語に対する複数の読み、重複行は文章生成前にエラーになります。

## 実行環境

利用者が指定する環境変数は次の3つだけです。いずれも投入shellで設定します。

```bash
export REPO=/absolute/path/to/term2speech
export MIL=/absolute/path/to/miltoka
export PROXY_URL='http://user:password%40example@proxy.example.com:8080'
```

PBSと投入helperが残りのパスを次の規則で埋めます。

| 用途 | 自動設定されるパス |
|---|---|
| Qwen3.6文章生成 | `$MIL/.venv_vllm_qwen_10000/bin/vllm` |
| Qwen3-TTS高速合成 | `$MIL/.venv-vllm-omni/bin/vllm` |
| vLLM-Omni Python | `$MIL/.venv-vllm-omni/bin/python` |
| client・公式TTS fallback・Whisper学習 | `$REPO/.venv/bin/python` |

文章生成用の`.venv_vllm_qwen_10000`とTTS用の`.venv-vllm-omni`は別環境です。
投入helperは両方の実行ファイルを検査し、不足していればジョブ投入前に対象パスを表示して停止します。

Qwen3.6-27BのBF16重みだけで約56 GBを使います。stage 1はtensor-parallel対象GPUの
合計メモリが65 GB未満なら公式の `Qwen/Qwen3.6-27B-FP8` を自動選択します。ローカル配置済みの
重みを使う場合は `LLM_MODEL=/shared/models/...` を明示してください。
`OFFLINE=1` では `prestage_offline.sh` と揃えてFP8を既定にします。BF16等へ変える場合は、
prestage時とPBS投入時の両方へ同じ `LLM_MODEL` を渡してください。

Qwen3-TTSの既定は `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`、日本語ネイティブの
`Ono_Anna` です。V100はBF16とFlashAttention 2を使えないため、float16＋SDPAを使います。
公式実装はリストbatchに対応しています。MiltokaにCUDA 12系のvLLM-Omni環境がある場合は
`TTS_ENGINE=vllm_omni` で `/v1/audio/speech/batch` を使えます。CUDA 13系wheelはV100に
適合しないため、利用可否をジョブ冒頭で検査します。

公式資料:

- [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api/)

## 3ジョブの投入（推奨）

3本とも `scripts/setup_proxy.sh` をsourceします。proxyはリポジトリへ保存せず、
`qsub -v` の `PROXY_URL` で渡してください。ユーザー名・パスワードに `@`、`,`、`:`、`/`、`#` が
含まれる場合は、それぞれ `%40`、`%2C`、`%3A`、`%2F`、`%23` のようにURL encodeします。
特に`,`はPBSの変数区切りなので、必ず `%2C` にしてください。設定確認だけを先に行う場合:

```bash
PROXY_URL='http://user:password%40example@proxy.example.com:8080'
qsub -v "PROXY_URL=$PROXY_URL,PROXY_DEBUG=1" scripts/run_net_check.pbs
```

ジョブログには資格情報をマスクしたproxy URLと、Hugging Face、PyPI、CUDA 12.1 wheelへの
疎通結果が出ます。`NO_PROXY`が未設定なら、既存helperの既定値
`localhost,127.0.0.1,::1` が使われます。

`annotations.tsv` は通常 `$REPO/data/annotations.tsv` に置きます。そこになければ
`$REPO/annotations.tsv` も自動検索します。投入helperが時刻付き`RUN_ROOT`を作成し、
3本のPBSを`afterok`で直列につなぎます。

```bash
export REPO=/absolute/path/to/term2speech
export MIL=/absolute/path/to/miltoka
PROXY_URL='http://user:password%40example@proxy.example.com:8080'
export PROXY_URL

bash "$REPO/scripts/submit_whisper_domain_ft.sh"
```

成功すると、共通出力先と3つのジョブIDが表示されます。

```text
run_root=/absolute/path/to/term2speech/out/whisper_domain_ft/run_YYYYmmdd_HHMMSS
text=123.server
tts=124.server
train=125.server
```

proxy内の`,`はPBSの変数区切りと衝突するため、必ず`%2C`へencodeしてください。
入力や出力先を変更する高度な実行では、`ANNOTATIONS`や`RUN_ROOT`を追加でexportしてから
同じhelperを実行できます。通常の実行では不要です。

### stage 1: 学習文生成

既定は用語ごとに正例12文と、同じ分野だが辞書中の用語を一切含まない対照文12文を
生成します。vLLMへ並列リクエストを送り、Qwen3.6はnon-thinkingのJSON出力に固定します。

```bash
# 通常はsubmit_whisper_domain_ft.shがこのstageを投入します。
# 生成数などを変える場合だけ、投入前に追加設定します。
export SENTENCES_PER_TERM=16
export REPLAY_RATIO=1.0
bash "$REPO/scripts/submit_whisper_domain_ft.sh"
```

正例は用語をexactly once含むこと、文長、改行・markup・数字、異常反復、重複を検査します。
`tts_text` は合格後にPythonが `term → reading` と置換するため、LLMが読みに干渉しません。
これは常用するTTS入力ではなく、漢字TTSが読めなかった場合のfallbackです。TTS直前にも
権威TSVから再計算するため、`tts_text` が欠けていても停止せず補完し、不一致でも権威readingを使います。
分野を固定したい場合は、例えば `DOMAIN_HINT=上下水道施設の運転保守` をPBSへ渡せます。
未指定時は用語ごとに分野を推定する汎用promptです。大量の用語でもcontextを圧迫しないよう、
prompt内の禁止語一覧は代表集合に制限しつつ、Python側ではTSV中の全用語を検査します。
既存serverを `VLLM_BASE_URL` で指定した場合、`LLM_MODEL` 未指定なら `/v1/models` の
先頭IDを取得してリクエストに使います。複数modelをserveするserverでは `LLM_MODEL` も明示してください。

### stage 2: Qwen3-TTS

`xvn_s / middle2` のV100 4枚に決定的にシャードし、各GPU内でもbatch合成します。
各専門語文はまず漢字を残した `sentence` で合成し、自然な韻律を優先します。そのWAVを
promptなしのfaster-whisperで逆認識し、結果をかな・モーラへ変換して正解readingと比較します。
音素文字列の完全一致ではなく、`セイ↔セー`、`ホウ↔ホー`など日本語の長音表記差を同一視した
重み付きモーラ距離を使います。一方、通常モーラ・促音・撥音の置換/欠落は内容誤りとして扱います。
不一致、境界スコア、空認識、ASR例外のいずれでもジョブは停止せず、その1件だけ `tts_text` で再合成します。

```bash
# 通常はautoです。MiltokaのvLLM-Omniを優先し、不適合なら公式qwen-ttsへfallbackします。
export TTS_ENGINE=auto
bash "$REPO/scripts/submit_whisper_domain_ft.sh"
```

発音確認の既定は `large-v3-turbo`、V100では `float16` です。判定器をロードできない場合も
停止せず、そのshardは全件reading fallbackで完成させます。その状態はmanifestに記録され、
次回 `--resume` では漢字TTS＋確認を再試行します。
採用上限は `PRONUNCIATION_PASS_THRESHOLD=0.15`、境界記録上限は
`PRONUNCIATION_UNCERTAIN_THRESHOLD=0.35` で変更できます。境界域も採用せずreading fallbackです。

manifestでは次を分離して保持します。

- `sentence`: Whisperの正解（漢字表記）
- `tts_text`: 読みへ置換したfallback文
- `synthesis_text`: 実際にTTSへ渡した文
- `pronunciation_check`: `passed` / `fallback_reading` / `checker_unavailable`
- `pronunciation_asr_text`: 漢字候補WAVのpromptなしASR結果
- `pronunciation_observed_reading`: ASR結果を形態素読みへ変換した値
- `pronunciation_mora_distance`: 許容長音差を含む正規化距離
- `pronunciation_content_edits`: 実質的なモーラ置換・挿入・欠落数
- `reading_fallback_used`: 読み仮名で再合成したか

WAVはfinite値、長さ、RMS、clippingを検査します。一般文に加え、少量の無音・低レベル雑音と
空transcriptのcontrolも作り、無音区間で専門語が湧き出す学習偏りを抑えます。最後に用語・kindを
層化して `train_manifest.jsonl` と `dev_manifest.jsonl` に分割します。
ASRによる確認は、明らかな読み違いをreading fallbackへ回しつつ漢字文の韻律を最大限残すための
実用的なgateです。音素単位の完全な証明ではないため、本学習前の用語別標本試聴も併用してください。

### stage 3: decoder FTとCTranslate2変換

このPBSは常にfull FT＋encoder凍結です。TTS生成を再実行しません。

```bash
# 通常はsubmit_whisper_domain_ft.shが前段成功後に自動実行します。
export EPOCHS=3
export LR=5e-6
bash "$REPO/scripts/submit_whisper_domain_ft.sh"
```

既定はA100 2枚（対応時BF16、その他はFP16）、3 epoch、学習率 `5e-6`、label smoothing、weight decay、
速度・gain・雑音・SpecAugmentです。dev CERが最良だったcheckpointへ戻してから保存します。
変換は一時ディレクトリで行い、`model.bin` 等を検証してから最終ディレクトリへ移します。

完成したモデル:

```text
<RUN_ROOT>/model/ct2/
```

既存評価へ渡す例:

```bash
qsub -v "MODEL_DIR=<RUN_ROOT>/model/ct2,WAV_DIR=data/test_wav,REFS=data/test_refs.txt" \
  scripts/run_whisper_eval.pbs
```

## 湧き出し対策

今回の経路では次を既定にしています。

- 専門語正例と、専門語を含まない同分野のreplay文を一対一で混ぜる
- 無音・低レベル雑音の空transcript controlを少量混ぜる
- encoderを完全凍結し、低い学習率でdecoderだけを更新する
- label末尾のEOTを切り落とさない
- dev最良checkpointへ復元する
- Whisper既定の`suppress_tokens`を解除しない
- 復号はVAD有効、直前文字列の内部履歴・明示prompt再注入を無効、temperature 0
- no-speech/log-prob/compression-ratioと反復制約を設定可能にする
- 評価に空音声false-positive文字数、挿入数、反復率を出す

専門用語だけのCERが下がっても、通常文・無音で挿入が増えたモデルは採用しないでください。
既存評価集合に無音が無い場合は、空参照の無音・環境雑音も追加して比較してください。

## 主な成果物

```text
<RUN_ROOT>/
  corpus.jsonl
  corpus.summary.json
  tts/shard_00..03/wav/*.wav
  combined_manifest.jsonl
  train_manifest.jsonl
  dev_manifest.jsonl
  model/hf/
  model/ct2/
```

`corpus.jsonl` と各manifestはIDで対応します。入力や設定を変える場合は同じ出力へ上書きせず、
新しい `RUN_ROOT` を使ってください。
