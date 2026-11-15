"""Load a detection model or a locally saved checkpoint."""

from pathlib import Path


def build_model(name: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the C2 dependencies from requirements-c2.txt") from exc
    try:
        return YOLO(str(name), task="detect")
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Could not load {name}. Check the weights path or download access: {exc}") from exc
