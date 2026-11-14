"""Build an audited, country-aware YOLO dataset from extracted RDD2022 data."""

import argparse
import csv
import json
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from rdd.data.convert_voc_to_yolo import CLASSES, read_annotation
from rdd.data.dataset_stats import write_stats
from rdd.data.make_splits import SPLITS, assign_splits
from rdd.data.plot_stats import write_figures
from rdd.utils.logging import sha256_file, write_json

COUNTRIES = {"Japan": "Japan", "India": "India", "Czech": "Czech",
             "Norway": "Norway", "United_States": "United_States",
             "China_Drone": "China", "China_MotorBike": "China"}
FIELDS = ("source_image", "source_xml", "country", "domain", "split", "group", "filename_block",
          "image_sha256", "xml_sha256", "image", "label", "width", "height",
          "boxes", *CLASSES, "clipped", "skipped", "excluded_classes",
          "area_lt_1pct", "area_1_to_10pct", "area_ge_10pct")


def read_groups(path: Path) -> dict[str, str]:
    groups = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["source_image", "group"]:
            raise ValueError("groups CSV header must be source_image,group")
        for row in reader:
            key, group = row["source_image"], row["group"]
            if not key or not group or not group.strip() or key in groups:
                raise ValueError(f"missing or duplicate groups CSV entry: {key!r}")
            groups[key] = group.strip()
    return groups


def by_stem(folder: Path, suffixes: set[str]) -> dict[str, Path]:
    found = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            key = path.stem.casefold()
            if key in found:
                raise ValueError(f"ambiguous filename stems: {found[key]} and {path}")
            found[key] = path
    return found


def discover(raw: Path, domain: str) -> list[tuple[Path, Path]]:
    candidates = [path for path in raw.rglob("train")
                  if path.is_dir() and path.parent.name == domain
                  and (path / "images").is_dir()
                  and (path / "annotations/xmls").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"{domain}: expected one labeled train tree under {raw}, "
                         f"found {len(candidates)}")
    train = candidates[0]
    images = by_stem(train / "images", {".jpg", ".jpeg", ".png"})
    xmls = by_stem(train / "annotations/xmls", {".xml"})
    if not images or images.keys() != xmls.keys():
        missing = sorted(images.keys() - xmls.keys())
        orphan = sorted(xmls.keys() - images.keys())
        raise ValueError(f"{domain}: empty or unpaired inputs; missing XML ({len(missing)}): "
                         f"{missing[:5]}, orphan XML ({len(orphan)}): {orphan[:5]}")
    return [(images[key], xmls[key]) for key in sorted(images)]


def prepare(config_path: Path, root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw, output, logs = ((root / config[key]).resolve() for key in ("raw", "output", "logs"))
    domains = config["domains"]
    if not isinstance(domains, list) or not domains or len(set(domains)) != len(domains):
        raise ValueError("domains must be a nonempty list without repeats")
    if any(domain not in COUNTRIES for domain in domains):
        raise ValueError(f"unsupported domain: {domains}")
    seed, unknown = config["seed"], config["unknown"]
    if type(seed) is not int or unknown not in {"skip", "error"}:
        raise ValueError("seed must be an integer and unknown must be skip or error")
    if not raw.is_dir():
        raise ValueError(f"raw dataset directory does not exist: {raw}")
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("raw and output must be separate directory trees")
    if output.exists():
        raise ValueError(f"output already exists: {output}; choose a fresh output")
    groups_path = (root / config["groups"]).resolve() if config.get("groups") else None
    groups = read_groups(groups_path) if groups_path else {}
    block_size = config.get("block_size", 1)
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    rows = []
    owners: dict[tuple[str, str], str] = {}
    for domain in sorted(domains):
        for index, (image, xml) in enumerate(discover(raw, domain)):
            key = image.relative_to(raw).as_posix()
            if groups_path and key not in groups:
                raise ValueError(f"groups CSV missing {key}")
            digest = sha256_file(image)
            group = groups.get(key, "")
            country = COUNTRIES[domain]
            block = f"{domain}:{index // block_size}" if block_size > 1 else ""
            for kind, value in (("image hash", digest), ("route group", group)):
                if value and owners.setdefault((kind, value), country) != country:
                    raise ValueError(f"{key}: {kind} shared across countries: {value}")
            rows.append({"source_image": key, "source_xml": xml.relative_to(raw).as_posix(),
                         "country": country, "domain": domain, "group": group, "filename_block": block,
                         "image_sha256": digest, "xml_sha256": sha256_file(xml)})
    if groups.keys() - {row["source_image"] for row in rows}:
        raise ValueError("groups CSV contains paths outside the selected input inventory")

    output.parent.mkdir(parents=True, exist_ok=True)
    split_maps = {}
    for country in sorted({row["country"] for row in rows}):
        split_maps.update(assign_splits(
            {r["source_image"]: (r["image_sha256"], r["group"], r["filename_block"])
             for r in rows if r["country"] == country}, config["ratios"], seed))
    excluded = Counter()
    with TemporaryDirectory(prefix=".rdd-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        subsets = sorted({r["country"] for r in rows})
        for country in subsets:
            for split in SPLITS:
                for kind in ("images", "labels"):
                    (stage / country / kind / split).mkdir(parents=True)
            (stage / country / "dataset.yaml").write_text(yaml.safe_dump({
                "path": (output / country).as_posix(),
                **{split: f"images/{split}" for split in SPLITS},
                "names": dict(enumerate(CLASSES)),
            }, sort_keys=False), encoding="utf-8")

        for row in rows:
            image = raw / row["source_image"]
            annotation = read_annotation(image, raw / row["source_xml"], unknown)
            split = split_maps[row["source_image"]]
            country = row["country"]
            name = f'{row["domain"]}__{image.name}'
            img_rel = Path(country, "images", split, name)
            label_rel = Path(country, "labels", split, name).with_suffix(".txt")
            shutil.copyfile(image, stage / img_rel)
            labels = [box.yolo(annotation.width, annotation.height) for box in annotation.boxes]
            (stage / label_rel).write_text("".join(f"{line}\n" for line in labels),
                                           encoding="utf-8")
            counts = Counter(box.class_id for box in annotation.boxes)
            areas = [(box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1])
                     / (annotation.width * annotation.height) for box in annotation.boxes]
            excluded.update(annotation.skipped)
            row.update(split=split, image=img_rel.as_posix(), label=label_rel.as_posix(),
                       width=annotation.width, height=annotation.height, boxes=len(labels),
                       clipped=annotation.clipped, skipped=sum(annotation.skipped.values()),
                       excluded_classes=json.dumps(annotation.skipped, sort_keys=True),
                       area_lt_1pct=sum(a < .01 for a in areas),
                       area_1_to_10pct=sum(.01 <= a < .10 for a in areas),
                       area_ge_10pct=sum(a >= .10 for a in areas),
                       **{name: counts[i] for i, name in enumerate(CLASSES)})

        manifest = stage / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        write_stats(stage / "stats.csv", rows)
        shutil.copyfile(stage / "stats.csv", stage / "dataset_stats.csv")
        write_figures(stage, rows)
        for country in subsets:
            (stage / "splits" / country).mkdir(parents=True)
            for split in SPLITS:
                rel_paths = [Path(r["image"]).relative_to(country).as_posix()
                             for r in rows if r["country"] == country and r["split"] == split]
                (stage / country / f"{split}.txt").write_text(
                    "".join(f"{p}\n" for p in rel_paths), encoding="utf-8")
                (stage / "splits" / country / f"{split}.txt").write_text(
                    "".join(f"{country}/{p}\n" for p in rel_paths), encoding="utf-8")

        timestamp = datetime.now(timezone.utc)
        run_id = f"prepare-{timestamp:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
        record = dict(run_id=run_id, created_utc=timestamp.isoformat(),
                      config_path=str(config_path), config=config,
                      config_sha256=sha256_file(config_path), output=str(output), raw=str(raw),
                      groups_sha256=sha256_file(groups_path) if groups_path else None,
                      manifest_sha256=sha256_file(manifest), stats_sha256=sha256_file(stage / "stats.csv"),
                      python=platform.python_version(), pipeline=version("road-damage-dg"),
                      dependencies={name: version(name) for name in ("Pillow", "PyYAML")},
                      code_sha256={p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): sha256_file(p)
                                   for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))},
                      images=len(rows), boxes=sum(r["boxes"] for r in rows),
                      clipped=sum(r["clipped"] for r in rows), excluded_classes=dict(excluded))
        logs.mkdir(parents=True, exist_ok=True)
        write_json(stage / "run.json", record)
        stage.rename(output)
        write_json(logs / f"{run_id}.json", record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="base for relative paths")
    parser.add_argument("--config", type=Path, default=Path("configs/data/rdd2022.yaml"))
    args = parser.parse_args()
    try:
        output = prepare((args.root / args.config).resolve(), args.root.resolve())
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"rdd-prepare: {exc}\n")
    print(f"Prepared {output}; see manifest.csv, stats.csv and run.json")


if __name__ == "__main__":
    main()
