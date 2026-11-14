import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from rdd.data.prepare import prepare
from rdd.utils.logging import sha256_file
from test_convert import annotation


def build_release(root: Path):
    raw = root / "raw" / "RDD2022"
    for domain, offset in (("Japan", 0), ("China_Drone", 30), ("China_MotorBike", 60)):
        train = raw / domain / "train"
        (train / "images").mkdir(parents=True)
        (train / "annotations/xmls").mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (100, 50), (i + offset, 0, 0)).save(train / f"images/{i}.png")
            contents = annotation(("D10", (0, 10, 25, 40))) if i else annotation()
            (train / f"annotations/xmls/{i}.xml").write_text(contents)
    test = raw / "Japan" / "test" / "images"
    test.mkdir(parents=True)
    (test / "unlabeled.jpg").write_bytes(b"intentionally not read")
    config = dict(raw="raw", output="prepared", logs="logs", seed=42, unknown="skip",
                  domains=["Japan", "China_Drone", "China_MotorBike"], block_size=2,
                  ratios=dict(train=.7, val=.15, test=.15))
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, raw, config


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_end_to_end(tmp_path):
    config, _, settings = build_release(tmp_path)
    output = prepare(config, tmp_path)
    manifest = rows(output / "manifest.csv")
    assert len(manifest) == 30
    assert {r["country"] for r in manifest} == {"China", "Japan"}
    assert all("/train/" in r["source_image"] for r in manifest)
    block_splits = {}
    for row in manifest:
        block = row["filename_block"]
        assert block
        assert block_splits.setdefault(block, row["split"]) == row["split"]
        assert sha256_file(output / row["image"]) == row["image_sha256"]
        lines = (output / row["label"]).read_text().splitlines()
        assert len(lines) == int(row["boxes"])
        assert all(line.startswith("1 ") for line in lines)
    for country in ("China", "Japan"):
        dataset = yaml.safe_load((output / country / "dataset.yaml").read_text())
        assert Path(dataset["path"]) == output / country
        for split in ("train", "val", "test"):
            images = {r["image"].split("/", 1)[1] for r in manifest
                      if r["country"] == country and r["split"] == split}
            assert images
            assert images == set((output / country / f"{split}.txt").read_text().splitlines())
            assert {f"{country}/{p}" for p in images} == set(
                (output / "splits" / country / f"{split}.txt").read_text().splitlines())
    assert (output / "stats.csv").read_bytes() == (output / "dataset_stats.csv").read_bytes()
    for name in ("images_by_country.png", "boxes_by_class.png"):
        with Image.open(output / name) as figure:
            figure.verify()
    stats = rows(output / "stats.csv")
    assert sum(int(r["images"]) for r in stats) == 30
    assert sum(int(r["boxes"]) for r in stats) == 27
    assert sum(int(r["D10"]) for r in stats) == 27
    assert sum(int(r["area_ge_10pct"]) for r in stats) == 27
    assert sum(int(r["negative_images"]) for r in stats) == 3
    log = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert log["manifest_sha256"] == sha256_file(output / "manifest.csv")
    assert log["stats_sha256"] == sha256_file(output / "stats.csv")
    with pytest.raises(ValueError, match="already exists"):
        prepare(config, tmp_path)
    settings["output"] = "repeat"
    config.write_text(yaml.safe_dump(settings))
    repeat = prepare(config, tmp_path)
    assert (output / "manifest.csv").read_bytes() == (repeat / "manifest.csv").read_bytes()
    assert (output / "stats.csv").read_bytes() == (repeat / "stats.csv").read_bytes()


@pytest.mark.parametrize("failure", ["missing", "invalid", "cross_country_duplicate"])
def test_invalid_input_does_not_publish(tmp_path, failure):
    config, raw, _ = build_release(tmp_path)
    xml = raw / "Japan/train/annotations/xmls/0.xml"
    if failure == "missing":
        xml.unlink()
        match = "missing XML"
    elif failure == "invalid":
        xml.write_text("<annotation>")
        match = "no element found"
    else:
        (raw / "China_Drone/train/images/0.png").write_bytes(
            (raw / "Japan/train/images/0.png").read_bytes())
        match = "shared across countries"
    with pytest.raises(ValueError, match=match):
        prepare(config, tmp_path)
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "logs").exists()
