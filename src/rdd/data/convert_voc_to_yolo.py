"""Read RDD2022 VOC XML and convert pixel-edge boxes to normalized YOLO boxes."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

CLASSES = ("D00", "D10", "D20", "D40")


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]

    def yolo(self, width: int, height: int) -> str:
        x1, y1, x2, y2 = self.xyxy
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        return f"{self.class_id} " + " ".join(f"{v:.10f}" for v in values)


@dataclass(frozen=True)
class Annotation:
    width: int
    height: int
    boxes: tuple[Box, ...]
    skipped: Counter[str]
    clipped: int


def read_annotation(image: Path, xml: Path, unknown: str) -> Annotation:
    try:
        with Image.open(image) as opened:
            width, height = opened.size
            opened.load()  # Detect truncated/corrupt images before reporting success.
        root = ET.parse(xml).getroot()
        if root.tag != "annotation":
            raise ValueError("expected an annotation root")
        declared = (int(root.findtext("size/width", "")), int(root.findtext("size/height", "")))
        if declared != (width, height):
            raise ValueError(f"XML size {declared} differs from image size {(width, height)}")
        boxes = []
        skipped: Counter[str] = Counter()
        clipped = 0
        for index, obj in enumerate(root.findall("object"), 1):
            name = obj.findtext("name", "").strip()
            if not name:
                raise ValueError(f"object {index}: missing class name")
            if name not in CLASSES:
                if unknown != "skip":
                    raise ValueError(f"object {index}: unsupported class {name!r}")
                skipped[name] += 1
                continue
            original = tuple(int(obj.findtext(f"bndbox/{key}", "")) for key in
                             ("xmin", "ymin", "xmax", "ymax"))
            edges = tuple(min(max(v, 0), limit) for v, limit in
                          zip(original, (width, height, width, height)))
            if edges[0] >= edges[2] or edges[1] >= edges[3]:
                raise ValueError(f"object {index}: degenerate box {original}")
            clipped += edges != original
            boxes.append(Box(CLASSES.index(name), edges))
        return Annotation(width, height, tuple(boxes), skipped, clipped)
    except (ET.ParseError, OSError, ValueError) as exc:
        raise ValueError(f"{image} / {xml}: {exc}") from exc
