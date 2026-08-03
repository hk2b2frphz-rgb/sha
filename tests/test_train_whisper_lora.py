from pathlib import Path

import pytest

from scripts.train_whisper_lora import (
    DEFAULT_LR,
    count_parameters,
    freeze_encoder,
    resolve_base_model,
    resolve_checkpoint_policy,
    resolve_freeze_encoder,
    resolve_learning_rate,
    resolve_mixed_precision,
    truncate_label_ids,
    verify_encoder_frozen,
)


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


def test_learning_rate_default_is_lower_for_full_finetune():
    # full-FT は重みを直接動かすので、LoRA と同じ 1e-4 では壊れる。
    assert resolve_learning_rate("full", None) == DEFAULT_LR["full"]
    assert resolve_learning_rate("lora", None) == DEFAULT_LR["lora"]
    assert DEFAULT_LR["full"] < DEFAULT_LR["lora"]


def test_explicit_learning_rate_wins_over_mode_default():
    assert resolve_learning_rate("full", 3e-5) == 3e-5


def test_full_finetune_freezes_encoder_by_default():
    assert resolve_freeze_encoder("full", None) is True
    assert resolve_freeze_encoder("full", False) is False


def test_lora_mode_rejects_freeze_encoder():
    assert resolve_freeze_encoder("lora", None) is False
    with pytest.raises(SystemExit):
        resolve_freeze_encoder("lora", True)


class FakeParam:
    def __init__(self, numel: int):
        self._numel = numel
        self.requires_grad = True

    def numel(self) -> int:
        return self._numel


class FakeModule:
    def __init__(self, params: list[FakeParam]):
        self._params = params

    def parameters(self):
        return iter(self._params)


class FakeWhisper:
    def __init__(self, encoder_params: list[FakeParam], decoder_params: list[FakeParam]):
        self.encoder = FakeModule(encoder_params)
        self._all = encoder_params + decoder_params

    def get_encoder(self):
        return self.encoder

    def parameters(self):
        return iter(self._all)


def test_freeze_encoder_leaves_only_the_decoder_trainable():
    model = FakeWhisper([FakeParam(600), FakeParam(400)], [FakeParam(250)])
    assert count_parameters(model) == (1250, 1250)

    freeze_encoder(model)

    assert count_parameters(model) == (250, 1250)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert verify_encoder_frozen(model) == 1000


def test_verify_encoder_frozen_rejects_even_one_trainable_tensor():
    model = FakeWhisper([FakeParam(600), FakeParam(400)], [FakeParam(250)])
    model.encoder._params[0].requires_grad = False

    with pytest.raises(RuntimeError, match="still trainable"):
        verify_encoder_frozen(model)


def test_truncate_labels_keeps_eot_at_the_new_end():
    assert truncate_label_ids([1, 2, 3, 4, 99], 4, eos_token_id=99) == [1, 2, 3, 99]


def test_labels_that_fit_are_not_rewritten():
    assert truncate_label_ids([1, 2, 3], 4, eos_token_id=99) == [1, 2, 3]


def test_truncated_labels_require_an_eot_token():
    with pytest.raises(ValueError, match="eos_token_id"):
        truncate_label_ids([1, 2], 1, eos_token_id=None)


@pytest.mark.parametrize(
    ("cuda_available", "bf16_supported", "expected"),
    [
        (False, False, (False, False)),
        (True, False, (True, False)),
        (True, True, (False, True)),
    ],
)
def test_auto_mixed_precision_selects_the_safest_supported_mode(
    cuda_available: bool,
    bf16_supported: bool,
    expected: tuple[bool, bool],
):
    assert resolve_mixed_precision(
        "auto",
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
    ) == expected


def test_explicit_bf16_rejects_an_unsupported_gpu():
    with pytest.raises(SystemExit, match="not supported"):
        resolve_mixed_precision("bf16", cuda_available=True, bf16_supported=False)


def test_dev_checkpoint_policy_keeps_eval_and_save_aligned_for_best_model_restore():
    assert resolve_checkpoint_policy(True, "no") == ("epoch", "epoch", True)
    assert resolve_checkpoint_policy(False, "no") == ("no", "no", False)
