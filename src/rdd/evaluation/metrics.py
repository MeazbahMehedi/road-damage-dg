"""Detection metric conventions shared by baseline and evaluation commands."""

from math import nan

from rdd.data.convert_voc_to_yolo import CLASSES


def f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def per_class_columns(values, class_index) -> dict[str, float]:
    output = {f"ap_{code}": nan for code in CLASSES}
    seen = set()
    for value, index in zip(values, class_index, strict=True):
        if int(index) != index or index not in range(len(CLASSES)) or index in seen:
            raise ValueError(f"invalid or repeated AP class index: {index}")
        seen.add(index)
        output[f"ap_{CLASSES[int(index)]}"] = float(value)
    return output
