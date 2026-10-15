"""Search space for the Whisper auto-research loop.

Every knob here is something a trial can actually change end to end: the
training script and the evaluator both accept the corresponding flag. Keeping
the space declarative lets the same definition drive random sampling, mutation,
validation of LLM-proposed configs, and the prompt text shown to the proposer.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["float", "log_float", "int", "choice"]

# LoRA target-module presets. The loop searches over preset names rather than
# raw module lists so a proposer cannot invent modules Whisper does not have.
TARGET_MODULE_PRESETS: dict[str, list[str]] = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "qkvo_fc": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
}

# SpecAugment presets -> (mask_time_prob, mask_feature_prob).
SPEC_AUGMENT_PRESETS: dict[str, tuple[float, float]] = {
    "off": (0.0, 0.0),
    "light": (0.05, 0.0),
    "strong": (0.10, 0.05),
}


@dataclass(frozen=True)
class Param:
    name: str
    kind: Kind
    description: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = field(default_factory=tuple)
    round_to: int | None = None

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            return rng.choice(list(self.choices))
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "log_float":
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
            return self._round(value)
        return self._round(rng.uniform(self.low, self.high))

    def mutate(self, value: Any, rng: random.Random) -> Any:
        if self.kind == "choice":
            options = [c for c in self.choices if c != value]
            return rng.choice(options) if options else value
        if self.kind == "int":
            span = max(1, int(round((self.high - self.low) * 0.25)))
            return self.clamp(int(value) + rng.randint(-span, span))
        if self.kind == "log_float":
            return self.clamp(float(value) * math.exp(rng.gauss(0.0, 0.5)))
        span = (self.high - self.low) * 0.25
        return self.clamp(float(value) + rng.gauss(0.0, span))

    def clamp(self, value: Any) -> Any:
        if self.kind == "choice":
            if value in self.choices:
                return value
            # Numeric choices: snap to the nearest legal option instead of
            # discarding an otherwise reasonable proposal.
            if all(isinstance(c, (int, float)) for c in self.choices):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return self.choices[0]
                return min(self.choices, key=lambda c: abs(float(c) - numeric))
            return self.choices[0]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return self.sample(random.Random(0))
        if not math.isfinite(numeric):
            return self.low
        numeric = min(max(numeric, float(self.low)), float(self.high))
        return int(round(numeric)) if self.kind == "int" else self._round(numeric)

    def _round(self, value: float) -> float:
        return round(value, self.round_to) if self.round_to is not None else value


SPACE: dict[str, Param] = {
    param.name: param
    for param in [
        # ---- optimization ----
        Param("epochs", "float", "training epochs", low=2.0, high=14.0, round_to=1),
        Param("lr", "log_float", "learning rate", low=1e-5, high=6e-4, round_to=7),
        Param("batch_size", "choice", "per-device batch size", choices=(4, 8, 16)),
        Param("grad_accum", "choice", "gradient accumulation steps", choices=(1, 2, 4)),
        Param("warmup_ratio", "float", "LR warmup fraction", low=0.0, high=0.3, round_to=3),
        Param("weight_decay", "float", "AdamW weight decay", low=0.0, high=0.1, round_to=4),
        Param("label_smoothing", "float", "label smoothing factor", low=0.0, high=0.2, round_to=3),
        Param(
            "lr_scheduler",
            "choice",
            "LR schedule",
            choices=("linear", "cosine", "constant_with_warmup"),
        ),
        # ---- LoRA capacity ----
        Param("lora_r", "choice", "LoRA rank", choices=(8, 16, 32, 64, 128)),
        Param("lora_alpha_mult", "choice", "lora_alpha = mult * lora_r", choices=(1, 2, 4)),
        Param("lora_dropout", "float", "LoRA dropout", low=0.0, high=0.2, round_to=3),
        Param(
            "target_modules",
            "choice",
            "which projections get LoRA adapters",
            choices=tuple(TARGET_MODULE_PRESETS),
        ),
        # ---- data augmentation (waveform level, applied when building features) ----
        Param("augment_copies", "choice", "extra augmented copies of the train set", choices=(0, 1, 2)),
        Param("augment_speed", "choice", "speed perturbation range (+/-)", choices=(0.0, 0.05, 0.1)),
        Param("augment_noise_snr_db", "choice", "gaussian noise SNR (0=off)", choices=(0, 20, 15, 10)),
        Param("augment_gain_db", "choice", "random gain jitter (+/- dB)", choices=(0.0, 3.0, 6.0)),
        Param("spec_augment", "choice", "SpecAugment strength", choices=tuple(SPEC_AUGMENT_PRESETS)),
        # ---- decoding ----
        Param("beam_size", "choice", "beam search width at eval time", choices=(1, 3, 5)),
        Param("no_repeat_ngram_size", "choice", "block repeated n-grams (0=off)", choices=(0, 3)),
        Param("prompt_terms", "choice", "feed domain terms as a Whisper prompt", choices=(0, 1)),
    ]
}

# Baseline = the hyper-parameters the existing run_whisper_train.pbs ships with.
# Trial 0 always evaluates this so every later result has a reference point.
DEFAULT_CONFIG: dict[str, Any] = {
    "epochs": 5.0,
    "lr": 1e-4,
    "batch_size": 8,
    "grad_accum": 1,
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "label_smoothing": 0.0,
    "lr_scheduler": "linear",
    "lora_r": 32,
    "lora_alpha_mult": 2,
    "lora_dropout": 0.05,
    "target_modules": "qkvo",
    "augment_copies": 0,
    "augment_speed": 0.0,
    "augment_noise_snr_db": 0,
    "augment_gain_db": 0.0,
    "spec_augment": "off",
    "beam_size": 1,
    "no_repeat_ngram_size": 0,
    "prompt_terms": 0,
}


def lora_alpha(config: dict[str, Any]) -> int:
    return int(config["lora_r"]) * int(config["lora_alpha_mult"])


def target_module_list(config: dict[str, Any]) -> list[str]:
    return TARGET_MODULE_PRESETS[str(config["target_modules"])]


def spec_augment_probs(config: dict[str, Any]) -> tuple[float, float]:
    return SPEC_AUGMENT_PRESETS[str(config["spec_augment"])]


def sample_config(rng: random.Random) -> dict[str, Any]:
    return {name: param.sample(rng) for name, param in SPACE.items()}


def clamp_config_with_changes(raw: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Coerce a mapping into a legal config and report every adjustment made.

    The caller is expected to print the changes: a proposer that keeps needing
    corrections is a proposer that is not really working, and that has to be
    visible rather than silently absorbed.
    """
    raw = raw or {}
    config: dict[str, Any] = {}
    changes: list[str] = []
    for name, param in SPACE.items():
        if name in raw and raw[name] is not None:
            value = param.clamp(raw[name])
            if value != raw[name]:
                changes.append(f"{name}: {raw[name]!r} -> {value!r}")
            config[name] = value
        else:
            config[name] = DEFAULT_CONFIG[name]
            changes.append(f"{name}: missing -> default {DEFAULT_CONFIG[name]!r}")
    for name in raw:
        if name not in SPACE:
            changes.append(f"{name}: unknown key, dropped")
    return config, changes


def clamp_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    return clamp_config_with_changes(raw)[0]


def mutate_config(base: dict[str, Any], rng: random.Random, n_changes: int = 2) -> dict[str, Any]:
    config = dict(clamp_config(base))
    names = rng.sample(list(SPACE), k=min(max(1, n_changes), len(SPACE)))
    for name in names:
        config[name] = SPACE[name].mutate(config[name], rng)
    return config


def crossover(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    a, b = clamp_config(a), clamp_config(b)
    return {name: (a[name] if rng.random() < 0.5 else b[name]) for name in SPACE}


def config_key(config: dict[str, Any]) -> str:
    """Stable identity used to avoid re-running an already evaluated config."""
    canonical = {name: clamp_config(config)[name] for name in sorted(SPACE)}
    for name, value in canonical.items():
        if SPACE[name].kind in ("float", "log_float"):
            canonical[name] = float(f"{float(value):.6g}")
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False)


def describe_space() -> str:
    """Human/LLM readable space description used in the proposer prompt."""
    lines = []
    for name, param in SPACE.items():
        if param.kind == "choice":
            domain = "one of " + json.dumps(list(param.choices), ensure_ascii=False)
        elif param.kind == "int":
            domain = f"integer in [{int(param.low)}, {int(param.high)}]"
        elif param.kind == "log_float":
            domain = f"float in [{param.low:g}, {param.high:g}] (log scale)"
        else:
            domain = f"float in [{param.low:g}, {param.high:g}]"
        lines.append(f"- {name}: {domain}  # {param.description}")
    return "\n".join(lines)
