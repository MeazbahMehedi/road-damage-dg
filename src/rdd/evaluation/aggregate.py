"""Summarize fold and seed scores while keeping the final hold-out separate."""

import argparse
import csv
from collections import defaultdict
from math import isfinite, nan
from pathlib import Path
from statistics import mean, stdev

from rdd.data.convert_voc_to_yolo import CLASSES
from rdd.evaluation.results import read_results

GROUP = ("model", "split", "condition", "method", "train_domain", "test_domain", "imgsz")
METRICS = ("map50", "map50_95", "precision", "recall", "f1", *(f"ap_{c}" for c in CLASSES))


def aggregate(rows: list[dict], minimum_instances: int = 30) -> list[dict]:
    if minimum_instances < 1:
        raise ValueError("minimum_instances must be positive")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row["phase"] != "C2":
            continue
        protocol = "final" if row["fold"] == "final" else "cv" if str(row["fold"]).isdigit() else "fixed"
        key = (*(row[column] for column in GROUP), protocol)
        unit = (*key, row["fold"], row["seed"])
        if unit in seen:
            raise ValueError(f"ambiguous repeated fold and seed: {unit}")
        seen.add(unit)
        groups[key].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = dict(zip((*GROUP, "protocol"), key))
        summary.update(n_runs=len(group), n_folds=len({row["fold"] for row in group}),
                       n_seeds=len({row["seed"] for row in group}))
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            values = [value for value in values if isfinite(value)]
            summary[f"{metric}_mean"] = mean(values) if values else nan
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else nan
            summary[f"{metric}_n"] = len(values)
        for code in CLASSES:
            support = min(int(row[f"n_inst_{code}"]) for row in group)
            summary[f"min_inst_{code}"] = support
            summary[f"unreliable_{code}"] = support < minimum_instances
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/results/baseline.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/baseline_bd_summary.csv"))
    parser.add_argument("--minimum-instances", type=int, default=30)
    args = parser.parse_args()
    try:
        summaries = aggregate(read_results(args.input), args.minimum_instances)
        if not summaries:
            raise ValueError("no real C2 rows to summarize")
        if args.input.resolve() == args.output.resolve():
            raise ValueError("summary output must be separate from source results")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, tuple(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"wrote {len(summaries)} groups to {args.output}")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"rdd aggregate: {exc}\n")


if __name__ == "__main__":
    main()
