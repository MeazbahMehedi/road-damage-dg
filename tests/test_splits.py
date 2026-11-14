from collections import Counter

import pytest

from rdd.data.make_splits import assign_splits

RATIOS = {"train": .7, "val": .15, "test": .15}


def test_reproducible_and_balanced():
    items = {str(i): (str(i), "") for i in range(20)}
    first = assign_splits(items, RATIOS, 42)
    assert first == assign_splits(dict(reversed(list(items.items()))), RATIOS, 42)
    assert first != assign_splits(items, RATIOS, 43)
    assert Counter(first.values()) == {"train": 14, "val": 3, "test": 3}


def test_duplicate_connects_routes():
    items = {"a": ("h1", "r1"), "b": ("h2", "r1"), "c": ("h2", "r2"),
             "d": ("h3", "r2"), "e": ("h4", ""), "f": ("h5", "")}
    splits = assign_splits(items, RATIOS, 42)
    assert splits["a"] == splits["b"] == splits["c"] == splits["d"]
    assert set(splits.values()) == set(RATIOS)


def test_inadequate_groups():
    with pytest.raises(ValueError, match="three independent"):
        assign_splits({"a": ("a", "same"), "b": ("b", "same")}, RATIOS, 42)
