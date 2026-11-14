from pathlib import Path

import pytest
from PIL import Image

from rdd.data.convert_voc_to_yolo import read_annotation


def annotation(*objects: tuple[str, tuple[int, int, int, int]]) -> str:
    body = ""
    for name, edges in objects:
        coords = "".join(f"<{k}>{v}</{k}>" for k, v in
                         zip(("xmin", "ymin", "xmax", "ymax"), edges))
        body += f"<object><name>{name}</name><bndbox>{coords}</bndbox></object>"
    return f"<annotation><size><width>100</width><height>50</height></size>{body}</annotation>"


def test_count_and_geometry(tmp_path: Path):
    image, xml = tmp_path / "image.png", tmp_path / "image.xml"
    Image.new("RGB", (100, 50)).save(image)
    xml.write_text(annotation(("D00", (10, 10, 30, 30)), ("D40", (-2, 0, 100, 50)),
                              ("Repair", (0, 0, 10, 10))))
    result = read_annotation(image, xml, "skip")
    assert len(result.boxes) + sum(result.skipped.values()) == 3
    assert result.skipped == {"Repair": 1}
    assert result.clipped == 1
    assert [box.yolo(100, 50) for box in result.boxes] == [
        "0 0.2000000000 0.4000000000 0.2000000000 0.4000000000",
        "3 0.5000000000 0.5000000000 1.0000000000 1.0000000000",
    ]
    with pytest.raises(ValueError, match="unsupported class"):
        read_annotation(image, xml, "error")


@pytest.mark.parametrize("contents, message", [
    ("<annotation>", "no element found"),
    (annotation().replace("100", "99"), "differs from image size"),
    (annotation(("D20", (-5, 10, -1, 30))), "degenerate box"),
])
def test_bad_annotations_fail(tmp_path, contents, message):
    image, xml = tmp_path / "image.png", tmp_path / "image.xml"
    Image.new("RGB", (100, 50)).save(image)
    xml.write_text(contents)
    with pytest.raises(ValueError, match=message):
        read_annotation(image, xml, "skip")
