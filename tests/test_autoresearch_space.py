import random

from autoresearch.space import (
    DEFAULT_CONFIG,
    SPACE,
    clamp_config,
    config_key,
    crossover,
    describe_space,
    lora_alpha,
    mutate_config,
    sample_config,
    spec_augment_probs,
    target_module_list,
)


def is_legal(config: dict) -> bool:
    for name, param in SPACE.items():
        value = config[name]
        if param.kind == "choice":
            if value not in param.choices:
                return False
        elif not (param.low <= float(value) <= param.high):
            return False
    return True


def test_sampled_and_mutated_configs_stay_inside_the_space():
    rng = random.Random(0)
    for _ in range(50):
        config = sample_config(rng)
        assert is_legal(config)
        assert is_legal(mutate_config(config, rng, n_changes=3))
        assert is_legal(crossover(config, sample_config(rng), rng))


def test_default_config_is_legal_and_covers_every_parameter():
    assert set(DEFAULT_CONFIG) == set(SPACE)
    assert is_legal(clamp_config(DEFAULT_CONFIG))


def test_clamp_repairs_out_of_range_missing_and_unknown_keys():
    repaired = clamp_config({"lr": 10.0, "epochs": -5, "nonsense": 1, "lora_r": 33})
    assert repaired["lr"] == SPACE["lr"].high
    assert repaired["epochs"] == SPACE["epochs"].low
    assert repaired["lora_r"] == 32  # snapped to the nearest legal choice
    assert "nonsense" not in repaired
    assert repaired["beam_size"] == DEFAULT_CONFIG["beam_size"]  # filled from the default
    assert is_legal(repaired)


def test_clamp_falls_back_for_unusable_values():
    repaired = clamp_config({"lr": "not-a-number", "target_modules": "does_not_exist"})
    assert isinstance(repaired["lr"], float)
    assert repaired["target_modules"] in SPACE["target_modules"].choices


def test_config_key_identifies_equivalent_configs():
    a = clamp_config(DEFAULT_CONFIG)
    b = clamp_config(dict(DEFAULT_CONFIG))
    assert config_key(a) == config_key(b)
    b["lora_r"] = 64
    assert config_key(a) != config_key(b)


def test_derived_training_values():
    config = clamp_config({**DEFAULT_CONFIG, "lora_r": 64, "lora_alpha_mult": 4})
    assert lora_alpha(config) == 256
    assert target_module_list(config) == ["q_proj", "k_proj", "v_proj", "out_proj"]
    assert spec_augment_probs({**config, "spec_augment": "off"}) == (0.0, 0.0)
    assert spec_augment_probs({**config, "spec_augment": "strong"})[0] > 0


def test_space_description_lists_every_knob():
    text = describe_space()
    for name in SPACE:
        assert name in text
