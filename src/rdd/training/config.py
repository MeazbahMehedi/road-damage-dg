"""Resolve and validate training settings."""

import re
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    model: str
    train_domain: str
    data_root: str
    project_dir: str
    imgsz: int
    epochs: int
    batch: int
    seed: int
    patience: int = 30
    workers: int = 2
    device: str = "auto"
    fold: int | str | None = 0
    n_folds: int = 5
    wandb: bool = False

    def __post_init__(self) -> None:
        for name in ("model", "train_domain", "data_root", "project_dir", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for value in (self.train_domain, Path(self.model).stem):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(f"invalid domain or model identifier: {value!r}")
        for name in ("imgsz", "epochs", "batch", "seed", "patience", "workers", "n_folds"):
            value = getattr(self, name)
            minimum = 3 if name == "n_folds" else 1 if name in {"imgsz", "epochs", "batch"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.imgsz % 32 or self.seed >= 2**32:
            raise ValueError("imgsz must be divisible by 32 and seed must fit in 32 bits")
        if self.fold is not None and self.fold != "final":
            if type(self.fold) is not int or not 0 <= self.fold < self.n_folds:
                raise ValueError("fold must be in [0, n_folds), final, or null")
        if type(self.wandb) is not bool:
            raise ValueError("wandb must be true or false")


def read_settings(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    unknown = set(data) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(map(str, unknown))}")
    return data


def load_config(path: Path, *, smoke: bool = False, **overrides) -> TrainConfig:
    data = read_settings(path)
    required = {"model", "train_domain", "data_root", "project_dir", "imgsz", "epochs", "batch", "seed"}
    if required - data.keys():
        raise ValueError(f"{path}: missing required settings {sorted(required - data.keys())}")
    if smoke:
        data.update(read_settings(path.with_name("smoke.yaml")))
        if overrides.get("fold") is not None:
            raise ValueError("the six-image smoke dataset has a single fixed split")
    data.update({name: value for name, value in overrides.items() if value is not None})
    return TrainConfig(**data)


def parse_fold(value: str) -> int | str:
    return value if value == "final" else int(value)


def make_exp_id(config: TrainConfig) -> str:
    fold = f"f{config.fold}" if type(config.fold) is int else config.fold or "smoke"
    return f"{Path(config.model).stem}_{config.train_domain}_{fold}_s{config.seed}"
