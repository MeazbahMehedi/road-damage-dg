"""Freeze route-disjoint Bangladesh folds from a reviewed image manifest."""

import argparse
import csv
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from PIL import Image

from rdd.data.convert_voc_to_yolo import CLASSES
from rdd.data.get_data_yaml import label_path, read_label_counts
from rdd.data.make_splits import SPLITS
from rdd.utils.logging import sha256_file
from rdd.utils.seed import stable_rank


@dataclass(frozen=True)
class Frame:
    image: str
    route: str
    counts: tuple[int, ...]


def make_folds(frames: list[Frame], holdout: set[str], n_folds: int, seed: int) -> dict:
    """Balance routes on class counts and frame counts, using the seed for ties."""
    routes = defaultdict(list)
    for frame in frames:
        routes[frame.route].append(frame)
    if not holdout or holdout - routes.keys():
        raise ValueError("hold-out must name at least one route in the manifest")
    dev = routes.keys() - holdout
    if n_folds < 3 or n_folds > len(dev):
        raise ValueError("need between 3 and the number of development routes as folds")

    features = {route: [len(routes[route]), *(
        sum(f.counts[c] for f in routes[route]) for c in range(len(CLASSES))
    )] for route in dev}
    totals = [sum(values) for values in zip(*features.values())]
    order = sorted(dev, key=lambda r: (
        -sum(value / max(total, 1) for value, total in zip(features[r], totals)),
        stable_rank(seed, r),
    ))
    buckets = [set() for _ in range(n_folds)]
    loads = [[0] * len(totals) for _ in buckets]
    for index, route in enumerate(order):
        def cost(bucket: int) -> float:
            return sum(((load + value) ** 2 - load ** 2) / max(total, 1) ** 2
                       for load, value, total in zip(loads[bucket], features[route], totals))

        choice = index if index < n_folds else min(range(n_folds), key=cost)
        buckets[choice].add(route)
        loads[choice] = [a + b for a, b in zip(loads[choice], features[route])]

    plan = {f"fold{i}": {"train": dev - test - buckets[(i + 1) % n_folds],
                        "val": buckets[(i + 1) % n_folds], "test": test}
            for i, test in enumerate(buckets)}
    plan["final"] = {"train": dev - buckets[0], "val": buckets[0], "test": holdout}
    return {key: {split: sorted(groups) for split, groups in spec.items()}
            for key, spec in plan.items()}


def load_frames(root: Path, manifest: Path) -> tuple[list[Frame], dict]:
    """Validate manifest paths, YOLO labels and exact duplicate route ownership."""
    seen_stems = set()
    hash_routes = {}
    hashes = {}
    frames = []
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "route_id"} <= set(reader.fieldnames or []):
            raise ValueError("manifest must include image and route_id columns")
        for row in reader:
            relative = Path(row["image"])
            route = row["route_id"].strip()
            image = (root / relative).resolve()
            stem = image.stem.casefold()
            if (relative.is_absolute() or ".." in relative.parts or not route
                    or relative.as_posix() != row["image"] or root not in image.parents
                    or relative.parts[:2] != ("Bangladesh", "images")
                    or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}
                    or stem in seen_stems):
                raise ValueError(f"invalid or duplicate frame or route: {row}")
            seen_stems.add(stem)
            with Image.open(image) as decoded:
                if min(decoded.size) < 10:
                    raise ValueError(f"image smaller than 10 pixels: {relative}")
                decoded.load()
            if image.suffix.lower() in {".jpg", ".jpeg"}:
                with image.open("rb") as stream:
                    stream.seek(-2, 2)
                    if stream.read() != b"\xff\xd9":
                        raise ValueError(f"JPEG missing end marker: {relative}")
            label = label_path(image)
            if root not in label.resolve().parents:
                raise ValueError(f"label escapes data root: {label}")
            digest = sha256_file(image)
            if hash_routes.setdefault(digest, route) != route:
                raise ValueError(f"exact duplicate spans routes: {relative}")
            counts = read_label_counts(label)
            hashes[relative.as_posix()] = {"image": digest, "label": sha256_file(label)}
            frames.append(Frame(relative.as_posix(), route, counts))
    return sorted(frames, key=lambda frame: frame.image), hashes


def write_folds(root: Path, manifest: Path, holdout: set[str], n_folds: int,
                seed: int, *, leave_one_route_out: bool = False) -> Path:
    """Publish frozen stem lists, runtime lists, dataset YAMLs, and an audit plan."""
    root = root.resolve()
    manifest = manifest.resolve()
    if root not in manifest.parents:
        raise ValueError("manifest must live inside data_root")
    frames, hashes = load_frames(root, manifest)
    destination = root / "Bangladesh" / "folds"
    if destination.exists():
        raise FileExistsError(f"folds already exist: {destination}")
    route_count = len({frame.route for frame in frames})
    if leave_one_route_out or route_count < 10:
        n_folds = route_count - len(holdout)
    route_plan = make_folds(frames, holdout, n_folds, seed)
    entries = {}
    with TemporaryDirectory(dir=destination.parent, prefix=".folds-") as work:
        stage = Path(work) / "folds"
        stage.mkdir()
        (stage / "stems").mkdir()
        for name, spec in route_plan.items():
            entries[name] = {}
            for split, routes in spec.items():
                selected = [frame for frame in frames if frame.route in routes]
                counts = {code: sum(frame.counts[i] for frame in selected)
                          for i, code in enumerate(CLASSES)}
                if absent := [code for code, count in counts.items() if not count]:
                    warnings.warn(f"{name} {split}: no instances of {', '.join(absent)}")
                entries[name][split] = {"images": [f.image for f in selected],
                                        "routes": routes, "counts": counts}
                (stage / f"{name}_{split}.txt").write_text(
                    "".join(f"{(root / f.image).as_posix()}\n" for f in selected), encoding="utf-8")
                (stage / "stems" / f"{name}_{split}.txt").write_text(
                    "".join(f"{Path(f.image).stem}\n" for f in selected), encoding="utf-8")
            (stage / f"{name}.yaml").write_text(yaml.safe_dump({
                "path": destination.as_posix(),
                **{split: f"{name}_{split}.txt" for split in SPLITS},
                "names": dict(enumerate(CLASSES)),
            }, sort_keys=False), encoding="utf-8")
        plan = {
            "manifest": manifest.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(manifest),
            "n_folds": n_folds, "seed": seed, "n_routes": route_count,
            "n_images": len(frames), "holdout": sorted(holdout), "folds": entries,
            "files": hashes,
        }
        (stage / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8", newline="\n")
        stage.rename(destination)
    return destination


def read_plan(root: Path) -> dict:
    path = root / "Bangladesh/folds/plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Bangladesh fold plan missing: {path}. Run rdd.data.make_folds first.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(root / plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Bangladesh manifest changed after splitting. Review and regenerate folds.")
    return plan


def check_readiness(root: Path, n_folds: int) -> dict:
    """Require a reviewed manifest before launching a real baseline run."""
    plan = read_plan(root)
    path = root / "Bangladesh/readiness.json"
    if not path.is_file():
        raise FileNotFoundError(f"Bangladesh readiness review missing: {path}")
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("manifest_sha256") != plan["manifest_sha256"]:
        raise ValueError("readiness review belongs to a different manifest")
    if review.get("plan_sha256") != sha256_file(root / "Bangladesh/folds/plan.json"):
        raise ValueError("readiness review belongs to a different fold plan")
    if plan["n_folds"] != n_folds:
        raise ValueError(f"configured n_folds must match generated plan ({plan['n_folds']})")
    if plan["n_images"] < 300:
        raise ValueError("Bangladesh gate needs at least 300 labeled frames")
    # Support fewer routes when the review explicitly records the exception.
    exception = review.get("route_exception")
    if plan["n_routes"] < 10 and not (isinstance(exception, str) and exception.strip()):
        raise ValueError("fewer than 10 routes requires a documented route_exception and LORO")
    if plan["n_routes"] < 10 and plan["n_folds"] != plan["n_routes"] - len(plan["holdout"]):
        raise ValueError("use leave-one-route-out for fewer than 10 routes")
    reviewed = review.get("second_annotator_images")
    if (review.get("labeling_guide_reviewed") is not True or type(reviewed) is not int
            or not (plan["n_images"] + 9) // 10 <= reviewed <= plan["n_images"]):
        raise ValueError("review the labeling guide and independently annotate about 10% of frames")
    classes = review.get("reported_classes")
    counts = plan["folds"]["final"]["test"]["counts"]
    if (not isinstance(classes, list) or not classes
            or any(code not in CLASSES or counts[code] < 30 for code in classes)):
        raise ValueError("declare the classes to report with at least 30 final test instances each")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout-routes", type=Path, required=True, help="One route ID per line")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leave-one-route-out", action="store_true")
    args = parser.parse_args()
    try:
        holdout = set(args.holdout_routes.read_text(encoding="utf-8").split())
        print(write_folds(args.data_root, args.manifest, holdout, args.n_folds, args.seed,
                          leave_one_route_out=args.leave_one_route_out))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"make-folds: {exc}\n")


if __name__ == "__main__":
    main()
