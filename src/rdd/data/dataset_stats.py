"""Summarize converted annotations by domain and project split."""

import csv
from collections import Counter
from pathlib import Path

from rdd.data.convert_voc_to_yolo import CLASSES

STAT_FIELDS = ("country", "domain", "split", "images", "negative_images", "boxes",
               *CLASSES, "clipped", "skipped", "area_lt_1pct", "area_1_to_10pct",
               "area_ge_10pct", "image_sizes")


def write_stats(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, STAT_FIELDS)
        writer.writeheader()
        for domain, split in sorted({(r["domain"], r["split"]) for r in rows}):
            subset = [r for r in rows if (r["domain"], r["split"]) == (domain, split)]
            sums = Counter()
            sizes = Counter()
            for row in subset:
                sums.update({key: row[key] for key in STAT_FIELDS[5:-1]})
                sizes[f'{row["width"]}x{row["height"]}'] += 1
            writer.writerow(dict(country=subset[0]["country"], domain=domain, split=split, images=len(subset),
                                 negative_images=sum(not r["boxes"] for r in subset),
                                 image_sizes=";".join(f"{k}:{v}" for k, v in sorted(sizes.items())),
                                 **sums))
