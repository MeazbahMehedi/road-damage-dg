"""Draw standalone dataset summary figures from preparation statistics."""

from collections import Counter
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw, ImageFont

from rdd.data.convert_voc_to_yolo import CLASSES


def draw_counts(path: Path, title: str, counts: Mapping[str, int]) -> None:
    """Write a labeled horizontal bar chart of integer counts."""
    image = Image.new("RGB", (860, 96 + 54 * len(counts)), "#f8fafc")
    draw = ImageDraw.Draw(image)
    heading = ImageFont.load_default(size=24)
    font = ImageFont.load_default(size=16)
    draw.text((24, 22), title, fill="#0f172a", font=heading)
    maximum = max(max(counts.values(), default=0), 1)
    for index, (label, value) in enumerate(counts.items()):
        y = 76 + 54 * index
        draw.text((24, y), label, fill="#334155", font=font)
        if value:
            draw.rectangle((215, y, 215 + round(515 * value / maximum), y + 22), fill="#2563a6")
        draw.text((748, y), f"{value:,}", fill="#0f172a", font=font)
    image.save(path)


def write_figures(directory: Path, rows: list[dict]) -> None:
    """Plot image counts by country and retained box counts by class."""
    country_counts = Counter(row["country"] for row in rows)
    class_counts = {code: sum(row[code] for row in rows) for code in CLASSES}
    draw_counts(directory / "images_by_country.png", "Images by country", dict(sorted(country_counts.items())))
    draw_counts(directory / "boxes_by_class.png", "Retained boxes by class", class_counts)
