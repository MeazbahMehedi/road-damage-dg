from math import isnan, sqrt

from rdd.data.convert_voc_to_yolo import CLASSES
from rdd.evaluation.aggregate import METRICS, aggregate


def test_aggregate_separates_final_and_flags_sparse_folds():
    base = dict(phase="C2", model="yolov8s.pt", split="test", condition="clean", method="baseline",
                train_domain="Bangladesh", test_domain="Bangladesh", imgsz="640", seed="1")
    base.update({f"n_inst_{code}": "32" for code in CLASSES})

    def row(fold, value):
        return dict(base, fold=fold, **dict.fromkeys(METRICS, value))

    rows = [row("0", .25), row("1", .75), row("final", .9)]
    rows[0].update(ap_D40="nan", n_inst_D40="0")
    cv, final = aggregate(rows)
    assert cv["map50_mean"] == .5 and cv["map50_std"] == sqrt(.125)
    assert cv["unreliable_D40"] and cv["ap_D40_n"] == 1
    assert cv["ap_D40_mean"] == .75 and isnan(cv["ap_D40_std"])
    assert final["protocol"] == "final" and final["map50_mean"] == .9
