"""Experiment configuration: YAML file + `key.sub=value` command-line overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    root: str = "data/leandojo_benchmark_4_v7"
    split: str = "random"                 # random | novel_premises
    max_train_examples: Optional[int] = None
    val_limit: Optional[int] = 2000       # val steps used for checkpoint selection


@dataclass
class ModelConfig:
    name: str = "google/byt5-small"
    max_len: int = 1024


@dataclass
class NegativesConfig:
    strategy: str = "in_file"             # in_file | random  (the ablation)
    num_negatives: int = 3
    num_in_file: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 2000
    max_steps: int = 40000
    loss: str = "mse"                     # mse (ReProver) | infonce
    temperature: float = 0.05
    grad_clip: float = 1.0
    bf16: bool = True
    num_workers: int = 4
    log_every: int = 50
    eval_every: int = 5000
    ckpt_every: int = 5000


@dataclass
class EvalConfig:
    batch_size: int = 64
    premise_batch_size: int = 128
    k_save: int = 100
    n_boot: int = 1000


@dataclass
class Config:
    seed: int = 0
    output_dir: str = "runs"
    run_name: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def resolved_run_name(self) -> str:
        return self.run_name or f"{self.data.split}__{self.negatives.strategy}__seed{self.seed}"

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.resolved_run_name()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _apply(obj, key_path: List[str], value, full_key: str):
    name = key_path[0]
    if not dataclasses.is_dataclass(obj) or name not in {f.name for f in dataclasses.fields(obj)}:
        raise KeyError(f"unknown config key {full_key!r}")
    if len(key_path) == 1:
        setattr(obj, name, value)
    else:
        _apply(getattr(obj, name), key_path[1:], value, full_key)


def _merge(obj, d: dict, prefix: str = ""):
    for k, v in d.items():
        full = f"{prefix}{k}"
        if isinstance(v, dict):
            sub = getattr(obj, k, None)
            if not dataclasses.is_dataclass(sub):
                raise KeyError(f"unknown config section {full!r}")
            _merge(sub, v, full + ".")
        else:
            _apply(obj, [k], v, full)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    cfg = Config()
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _apply(cfg, k.split("."), yaml.safe_load(v), k)
    return cfg
