from pathlib import Path

import pytest
import yaml

from rdd.training.config import load_config, make_exp_id

BASELINE = Path(__file__).resolve().parents[1] / "configs/train/baseline_bd.yaml"


def test_smoke_and_id():
    assert make_exp_id(load_config(BASELINE)) == "yolov8s_Bangladesh_f0_s1"
    smoke = load_config(BASELINE, smoke=True)
    assert (smoke.model, smoke.imgsz, smoke.epochs, smoke.workers, smoke.device) == (
        "yolov8n.pt", 320, 1, 0, "cpu",
    )
    assert make_exp_id(smoke) == "yolov8n_China_smoke_s1"


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_unknown_config(tmp_path, missing):
    data = yaml.safe_load(BASELINE.read_text())
    if missing:
        del data["imgsz"]
    else:
        data["epoch"] = 1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="missing.*imgsz" if missing else "unknown.*epoch"):
        load_config(config)
