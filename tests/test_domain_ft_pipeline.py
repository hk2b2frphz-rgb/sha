from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_pipeline_is_split_into_three_resource_specific_pbs_jobs():
    text = read("run_generate_training_text.pbs")
    tts = read("run_synthesize_training_audio.pbs")
    train = read("run_whisper_decoder_train.pbs")

    assert "#PBS -q xan_s" in text and "select=1:res=small" in text
    assert "Qwen/Qwen3.6-27B" in text and '"$VLLM_CMD" serve' in text

    assert "#PBS -q xvn_s" in tts and "select=1:res=middle2" in tts
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in tts
    assert "--num-shards" in tts and "--api-base" in tts
    assert "--pronunciation-check-model" in tts
    assert "large-v3-turbo" in tts
    assert "VLLM_OMNI_CMD" in tts
    assert "vLLM-Omni synthesis failed" in tts

    assert "#PBS -q xan_s" in train and "select=1:res=middle" in train
    assert "--finetune-mode full" in train
    assert "--freeze-encoder" in train
    assert "ct2-transformers-converter" in train
    assert "synthesize_speech.py" not in train


def test_all_pipeline_jobs_share_the_same_run_root_default():
    expected = 'RUN_ROOT="${RUN_ROOT:-out/whisper_domain_ft/default}"'
    for name in (
        "run_generate_training_text.pbs",
        "run_synthesize_training_audio.pbs",
        "run_whisper_decoder_train.pbs",
    ):
        assert expected in read(name)


def test_pbs_jobs_derive_runtime_paths_from_repo_and_miltoka_roots():
    text = read("run_generate_training_text.pbs")
    tts = read("run_synthesize_training_audio.pbs")
    train = read("run_whisper_decoder_train.pbs")

    assert 'REPO="${REPO:-${PBS_O_WORKDIR:-$(pwd)}}"' in text
    assert 'VLLM_CMD="$MIL/.venv_vllm_qwen_10000/bin/vllm"' in text
    assert 'CLIENT_PYTHON="$REPO/.venv/bin/python"' in text
    assert 'export PATH="$VLLM_BIN_DIR:$PATH"' in text
    assert "command -v ninja" in text
    assert 'VLLM_OMNI_CMD="$MIL/.venv-vllm-omni/bin/vllm"' in tts
    assert 'VLLM_OMNI_PYTHON="$MIL/.venv-vllm-omni/bin/python"' in tts
    assert 'TTS_PYTHON="$REPO/.venv/bin/python"' in tts
    assert 'export PATH="$VLLM_OMNI_BIN_DIR:$PATH"' in tts
    assert "command -v ninja" in tts
    assert 'TRAIN_PYTHON="$REPO/.venv/bin/python"' in train


def test_submit_helper_only_requires_three_user_environment_variables():
    submit = read("submit_whisper_domain_ft.sh")

    assert ': "${REPO:?' in submit
    assert ': "${MIL:?' in submit
    assert ': "${PROXY_URL:?' in submit
    assert '.venv_vllm_qwen_10000/bin/ninja' in submit
    assert '.venv-vllm-omni/bin/ninja' in submit
    assert "run_$(date +%Y%m%d_%H%M%S)" in submit
    assert 'depend=afterok:$text_job' in submit
    assert 'depend=afterok:$tts_job' in submit


def test_job_shell_files_remain_ascii_for_strict_qsub_parsers():
    for name in (
        "run_generate_training_text.pbs",
        "run_synthesize_training_audio.pbs",
        "run_whisper_decoder_train.pbs",
        "submit_whisper_domain_ft.sh",
    ):
        read(name).encode("ascii")


def test_whisper_export_paths_do_not_clear_pretrained_suppress_tokens():
    for name in (
        "train_whisper_lora.py",
        "export_whisper_model.py",
    ):
        source = read(name)
        assert "suppress_tokens = []" not in source
