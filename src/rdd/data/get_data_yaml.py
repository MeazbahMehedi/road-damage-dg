"""Read C1 country datasets and C2 fold datasets."""

from math import isfinite
from pathlib import Path
import re

import yaml

from rdd.data.convert_voc_to_yolo import CLASSES
from rdd.data.make_splits import SPLITS


def get_data_yaml(domain: str, data_root: Path, fold: int | str | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", domain):
        raise ValueError(f"invalid dataset domain {domain!r}")
    if fold is None:
        path = data_root / domain / "dataset.yaml"
    elif fold == "final" or (type(fold) is int and fold >= 0):
        name = f"fold{fold}" if type(fold) is int else "final"
        path = data_root / domain / "folds" / f"{name}.yaml"
    else:
        raise ValueError(f"invalid fold {fold!r}")
    if not path.is_file():
        raise FileNotFoundError(f"dataset YAML missing: {path}")
    return path.resolve()


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" not in parts[:-1]:
        raise ValueError(f"image path has no images directory: {image}")
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_label_counts(path: Path) -> tuple[int, ...]:
    counts = [0] * len(CLASSES)
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            category, x, y, width, height = line.split()
            category = int(category)
            x, y, width, height = map(float, (x, y, width, height))
            if (category not in range(len(CLASSES))
                    or not all(isfinite(v) for v in (x, y, width, height))
                    or not 0 < width <= 1 or not 0 < height <= 1
                    or not 0 <= x <= 1 or not 0 <= y <= 1
                    or min(x-width/2, y-height/2, 1-x-width/2, 1-y-height/2) < -1e-6):
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid YOLO box") from exc
        box = (category, x, y, width, height)
        if box in seen:
            raise ValueError(f"{path}:{number}: duplicate YOLO box")
        seen.add(box)
        counts[category] += 1
    return tuple(counts)


def split_images(dataset: Path, split: str) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"invalid split {split!r}")
    spec = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    if spec.get("names") not in (dict(enumerate(CLASSES)), list(CLASSES)):
        raise ValueError(f"{dataset}: unexpected class mapping")
    base = (dataset.parent / spec["path"]).resolve()
    target = base / spec[split]
    if target.is_dir():
        images = sorted(p for p in target.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    elif target.is_file() and target.suffix == ".txt":
        images = [(target.parent / line).resolve()
                  for line in target.read_text(encoding="utf-8").splitlines() if line]
    else:
        raise FileNotFoundError(f"{dataset}: missing {split} path: {target}")
    if not images or len(images) != len(set(images)):
        raise ValueError(f"{dataset}: {split} must contain distinct images")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"image missing: {image}")
    return images
