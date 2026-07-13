from pathlib import Path

from scripts.train_whisper_lora import resolve_base_model


class Args:
    base_model = None

    def __init__(self, base_model_file: Path):
        self.base_model_file = base_model_file


def test_resolve_base_model_reads_named_value_from_shared_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "WAV_DIR=data/test_wav\nBASE_MODEL=/models/whisper\nMODEL=x\t/models/x\n",
        encoding="utf-8",
    )
    assert resolve_base_model(Args(manifest)) == "/models/whisper"
