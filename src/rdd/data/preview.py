"""Draw converted YOLO annotations for an image listed in the manifest."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from rdd.data.convert_voc_to_yolo import CLASSES


def draw_preview(dataset: Path, source_image: str, output: Path) -> None:
    """Save a preview without changing the prepared image or label."""
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        row = next((r for r in csv.DictReader(stream) if r["source_image"] == source_image), None)
    if row is None:
        raise ValueError(f"image not found in manifest: {source_image}")
    with Image.open(dataset / row["image"]) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, canvas.width // 60))
    for line in (dataset / row["label"]).read_text(encoding="utf-8").splitlines():
        category, x, y, width, height = line.split()
        x, y, width, height = map(float, (x, y, width, height))
        left, right = (x - width / 2) * canvas.width, (x + width / 2) * canvas.width
        top, bottom = (y - height / 2) * canvas.height, (y + height / 2) * canvas.height
        color = ("#fde047", "#38bdf8", "#f472b6", "#86efac")[int(category)]
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, canvas.width // 260))
        draw.text((left + 2, top + 2), CLASSES[int(category)], font=font,
                  fill=color, stroke_width=1, stroke_fill="black")
    # Exclusive creation protects source images and existing previews.
    with output.open("xb") as stream:
        canvas.save(stream, format="PNG")


def main() -> None:
    """Parse the dataset path, exact manifest key and output PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source_image")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        draw_preview(args.dataset, args.source_image, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"rdd preview: {exc}\n")


if __name__ == "__main__":
    main()
