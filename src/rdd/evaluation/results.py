"""Write evaluation rows with a fixed schema and repeat-run protection."""

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

COLUMNS = (
    "exp_id", "phase", "model", "imgsz", "train_domain", "test_domain", "split", "fold",
    "condition", "method", "n_labeled", "seed", "map50", "map50_95", "precision", "recall", "f1",
    "ap_D00", "ap_D10", "ap_D20", "ap_D40", "n_test_images",
    "n_inst_D00", "n_inst_D10", "n_inst_D20", "n_inst_D40", "latency_ms", "notes",
)
KEY = ("exp_id", "test_domain", "split", "fold")


def read_results(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: incompatible results CSV header")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: incomplete results CSV row")
    return rows


def append_result(row: dict, path: Path, *, overwrite: bool = False) -> None:
    missing = set(COLUMNS) - row.keys()
    unknown = row.keys() - set(COLUMNS)
    if missing or unknown:
        raise ValueError(f"invalid result columns: missing={sorted(missing)}, unknown={sorted(unknown)}")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("x", encoding="utf-8") as lease:
        lease.write("results update in progress\n")
    temporary = None
    try:
        rows = read_results(path) if path.exists() else []
        key = tuple(str(row[field]) for field in KEY)
        matches = [i for i, old in enumerate(rows) if tuple(old[field] for field in KEY) == key]
        if len(matches) > 1 or (matches and not overwrite):
            raise ValueError(f"duplicate result: {key}")
        if matches:
            rows[matches[0]] = row
        else:
            rows.append(row)
        with NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()
