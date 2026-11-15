"""Train a reviewed Bangladesh fold, or run a separate six-image CPU smoke test."""

import argparse
import json
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml

from rdd.data.get_data_yaml import get_data_yaml, label_path, split_images
from rdd.data.make_folds import check_readiness, read_plan
from rdd.data.make_splits import SPLITS
from rdd.models.build_model import build_model
from rdd.training.config import TrainConfig, load_config, make_exp_id, parse_fold
from rdd.utils.logging import sha256_file, write_json


def verify_fold(root: Path, dataset: Path, fold: int | str, plan: dict,
                splits: tuple[str, ...]) -> list[Path]:
    """Compare runtime lists and selected files to the frozen fold plan."""
    name = f"fold{fold}" if type(fold) is int else fold
    spec = plan["folds"][name]
    owners = {}
    for split in SPLITS:
        for route in spec[split]["routes"]:
            if owners.setdefault(route, split) != split:
                raise ValueError(f"{name}: route {route} crosses splits")
    if name != "final" and set(plan["holdout"]) & owners.keys():
        raise ValueError(f"{name}: final hold-out route appears in cross-validation")
    images = []
    for split in splits:
        resolved = split_images(dataset, split)
        expected = [(root / p).resolve() for p in spec[split]["images"]]
        if resolved != expected:
            raise ValueError(f"{dataset}: {split} differs from the frozen list")
        for image in resolved:
            rel = image.relative_to(root).as_posix()
            digests = plan["files"][rel]
            if (sha256_file(image) != digests["image"]
                    or sha256_file(label_path(image)) != digests["label"]):
                raise ValueError(f"image or label changed since splitting: {rel}")
        images.extend(resolved)
    if len(set(images)) != len(images):
        raise ValueError(f"{dataset}: an image occurs in multiple splits")
    return images


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def configure_tracking(enabled: bool) -> None:
    from ultralytics import settings

    if enabled:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("wandb=true requires W&B to be installed and configured") from exc
    settings.update({"wandb": enabled})


def git_state() -> dict:
    repo = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def train(cfg: TrainConfig, *, smoke: bool = False, resume: bool = False) -> Path:
    exp_id = make_exp_id(cfg)
    root = Path(cfg.data_root).resolve()
    run = Path(cfg.project_dir).resolve() / exp_id
    if smoke and (cfg.train_domain != "China" or cfg.fold is not None or cfg.device != "cpu"):
        raise ValueError("smoke must use the separate China sample on CPU")
    if not smoke and (cfg.train_domain != "Bangladesh" or cfg.fold is None):
        raise ValueError("the C2 baseline requires a Bangladesh fold; use --smoke for sample data")
    plan = None if smoke else check_readiness(root, cfg.n_folds)
    dataset = get_data_yaml(cfg.train_domain, root, cfg.fold)
    if plan is not None:
        verify_fold(root, dataset, cfg.fold, plan, ("train", "val"))
    else:
        counts = [len(split_images(dataset, split)) for split in SPLITS]
        if counts != [4, 1, 1]:
            raise ValueError(f"expected a six-image 4/1/1 smoke dataset, found {counts}")

    config_path = run / "config_used.yaml"
    checkpoint = run / "weights/last.pt"
    if resume:
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"resuming requires {checkpoint} and {config_path}")
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != asdict(cfg):
            raise ValueError("resume configuration differs from the original run")
        metadata = json.loads((run / "env.json").read_text(encoding="utf-8"))
        if metadata["dataset_yaml_sha256"] != sha256_file(dataset):
            raise ValueError("dataset YAML changed since the original run")
        if plan is not None and metadata["plan_sha256"] != sha256_file(dataset.parent / "plan.json"):
            raise ValueError("fold plan changed since the original run")
    elif run.exists():
        raise FileExistsError(f"run already exists: {run}. Use --resume for an interrupted run.")

    import torch
    import ultralytics

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    configure_tracking(cfg.wandb)
    model = build_model(checkpoint if resume else cfg.model)
    if resume:
        trainer_args = model.ckpt.get("train_args", {})
        if (Path(trainer_args.get("data", "")).resolve() != dataset
                or Path(trainer_args.get("project", "")).resolve() != run.parent
                or trainer_args.get("name") != exp_id):
            raise ValueError("resume requires the original dataset and run paths")
        if model.ckpt.get("optimizer") is None:
            raise ValueError("last.pt has no optimizer state. It may be from a completed run.")
        model.train(resume=True, device=device)
    else:
        run.mkdir(parents=True, exist_ok=False)
        config_path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
        write_json(run / "env.json", {
            **git_state(), "python": platform.python_version(), "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__, "device": device, "smoke": smoke,
            "dataset_yaml_sha256": sha256_file(dataset),
            "n_labeled": len(split_images(dataset, "train")),
            "plan_sha256": sha256_file(dataset.parent / "plan.json") if plan else None,
        })
        model.train(data=str(dataset), epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch,
                    patience=cfg.patience, deterministic=True, seed=cfg.seed, workers=cfg.workers,
                    device=device, project=str(run.parent), name=exp_id, exist_ok=True, amp=not smoke)
    best = run / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training returned without producing {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/baseline_bd.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold", type=parse_fold)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config, smoke=args.smoke, fold=args.fold, seed=args.seed,
                          data_root=args.data_root)
        print(train(cfg, smoke=args.smoke, resume=args.resume))
    except (OSError, KeyError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        parser.exit(2, f"rdd train: {exc}\n")


if __name__ == "__main__":
    main()
