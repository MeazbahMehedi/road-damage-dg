"""Assign indivisible image groups to approximate per-domain split targets."""

from collections import defaultdict
from math import isclose
from typing import Mapping

from rdd.utils.seed import stable_rank

SPLITS = ("train", "val", "test")


def assign_splits(items: Mapping[str, tuple[str, ...]], ratios: Mapping[str, float],
                  seed: int) -> dict[str, str]:
    """Map source image to split. Keys in each tuple represent identity, route and optional filename block."""
    if (set(ratios) != set(SPLITS) or any(not 0 < ratios[s] < 1 for s in SPLITS)
            or not isclose(sum(ratios.values()), 1)):
        raise ValueError("train/val/test ratios must be positive and sum to 1")

    # Merge all connected images: a duplicate can link two different route groups.
    parents = {key: key for key in items}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen: dict[tuple[int, str], str] = {}
    for key in sorted(items):
        for kind, value in enumerate(items[key]):
            if value:
                previous = seen.setdefault((kind, value), key)
                parents[find(key)] = find(previous)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(items):
        groups[find(key)].append(key)
    if len(groups) < len(SPLITS):
        raise ValueError("need at least three independent groups for train/val/test")

    counts = dict.fromkeys(SPLITS, 0)
    assignments = {}
    ordered = sorted(groups.values(), key=lambda keys: (-len(keys), stable_rank(seed, keys[0])))
    for index, keys in enumerate(ordered):
        empty = [s for s in SPLITS if not counts[s]]
        allowed = empty if len(ordered) - index == len(empty) else SPLITS
        split = max(allowed, key=lambda s: len(items) * ratios[s] - counts[s])
        counts[split] += len(keys)
        assignments.update(dict.fromkeys(keys, split))
    return assignments
