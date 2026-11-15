from pathlib import Path

import pytest
from PIL import Image

from rdd.data.get_data_yaml import get_data_yaml, split_images
from rdd.data.make_folds import Frame, check_readiness, make_folds, read_plan, write_folds
from rdd.training.train import verify_fold


def test_route_isolation_coverage_and_determinism():
    frames = [Frame(f"{i}-{j}.jpg", f"R{i}", (1, 2, i % 3, 4))
              for i in range(12) for j in range(2)]
    hold = {"R10", "R11"}
    plan = make_folds(frames, hold, 5, 42)
    assert plan == make_folds(list(reversed(frames)), hold, 5, 42)
    seen = []
    for i in range(5):
        fold = plan[f"fold{i}"]
        train, val, test = (set(fold[split]) for split in ("train", "val", "test"))
        assert not (train & val or train & test or val & test)
        assert (train | val | test) == {f"R{i}" for i in range(10)}
        seen.extend(test)
    assert sorted(seen) == sorted(f"R{i}" for i in range(10))
    assert set(plan["final"]["test"]) == hold
    assert not hold & set(plan["final"]["val"] + plan["final"]["train"])


def test_generated_files_and_manifest_guard(tmp_path):
    data = tmp_path / "Bangladesh"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    lines = ["image,route_id"]
    for i in range(5):
        rel = Path("Bangladesh") / "images" / f"BD_{i}.png"
        Image.new("RGB", (20, 20), (i, 0, 0)).save(tmp_path / rel)
        (data / "labels" / f"BD_{i}.txt").write_text(
            "".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in range(4)))
        lines.append(f"{rel.as_posix()},R{i}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n")
    write_folds(tmp_path, manifest, {"R4"}, 5, 42)
    plan = read_plan(tmp_path)
    assert plan["n_folds"] == 4
    yaml_path = get_data_yaml("Bangladesh", tmp_path, "final")
    assert split_images(yaml_path, "test") == [data / "images/BD_4.png"]
    with pytest.raises(FileNotFoundError, match=r"Japan.*dataset.yaml"):
        get_data_yaml("Japan", tmp_path)
    with pytest.raises(FileNotFoundError, match="readiness review"):
        check_readiness(tmp_path, 4)
    assert verify_fold(tmp_path, yaml_path, "final", plan, ("test",))
    (data / "labels/BD_4.txt").write_text("")
    with pytest.raises(ValueError, match="image or label changed"):
        verify_fold(tmp_path, yaml_path, "final", plan, ("test",))
