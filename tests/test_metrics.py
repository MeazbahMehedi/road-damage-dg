from math import isnan

import pytest

from rdd.evaluation.metrics import f1, per_class_columns


def test_f1():
    assert f1(0.0, 0.0) == 0.0
    assert f1(0.5, 1.0) == pytest.approx(2 / 3)


def test_ap_uses_class_index():
    columns = per_class_columns([0.8, 0.4], [3, 0])
    assert columns["ap_D40"] == 0.8
    assert columns["ap_D00"] == 0.4
    assert isnan(columns["ap_D10"]) and isnan(columns["ap_D20"])
    with pytest.raises(ValueError):
        per_class_columns([0.5], [])
