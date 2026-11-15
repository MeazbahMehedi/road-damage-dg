"""Score a trained checkpoint on a frozen evaluation split."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from rdd.data.convert_voc_to_yolo import CLASSES
from rdd.data.get_data_yaml import get_data_yaml, label_path, read_label_counts, split_images
from rdd.data.make_folds import check_readiness
from rdd.evaluation.metrics import f1, per_class_columns
from rdd.evaluation.results import append_result
from rdd.models.build_model import build_model
from rdd.training.config import TrainConfig, make_exp_id, parse_fold
from rdd.training.train import git_state, resolve_device, verify_fold
from rdd.utils.logging import sha256_file, write_json


def evaluate(weights: Path, *, domain: str, split: str, fold: int | str | None = None,
             exp_id: str | None = None, data_root: Path | None = None, imgsz: int | None = None,
             batch: int | None = None, output: Path | None = None, overwrite: bool = False) -> dict:
    """Evaluate a checkpoint and record both metrics and ground truth support."""
    weights = weights.resolve(strict=True)
    run = weights.parents[1]
    cfg = TrainConfig(**yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8")))
    metadata = json.loads((run / "env.json").read_text(encoding="utf-8"))
    smoke = metadata["smoke"]
    trained_id = make_exp_id(cfg)
    if exp_id is not None and exp_id != trained_id:
        raise ValueError(f"--exp-id must match the training run ({trained_id})")
    if fold is None and domain == cfg.train_domain:
        fold = cfg.fold
    if split not in {"val", "test"}:
        raise ValueError("evaluate on val or test")
    if smoke and (domain != "China" or fold is not None or split != "val"):
        raise ValueError("smoke evaluation is limited to the sample validation split")
    if domain == cfg.train_domain and not smoke and fold != cfg.fold:
        raise ValueError("evaluate the Bangladesh baseline on its own fold to avoid overlap")
    root = Path(data_root or cfg.data_root).resolve()
    dataset = get_data_yaml(domain, root, fold)
    if domain == "Bangladesh":
        if fold is None:
            raise ValueError("Bangladesh evaluation requires a frozen fold")
        plan = check_readiness(root, cfg.n_folds)
        if metadata["plan_sha256"] and sha256_file(dataset.parent / "plan.json") != metadata["plan_sha256"]:
            raise ValueError("evaluation plan differs from the plan recorded at training")
        images = verify_fold(root, dataset, fold, plan, (split,))
    else:
        images = split_images(dataset, split)

    counts = Counter()
    for image in images:
        counts.update(dict(zip(CLASSES, read_label_counts(label_path(image)))))
    if not sum(counts.values()):
        raise ValueError("the evaluation split has no benchmark ground truth boxes")
    size = cfg.imgsz if imgsz is None else imgsz
    batch_size = cfg.batch if batch is None else batch
    if size <= 0 or size % 32 or batch_size < 1:
        raise ValueError("imgsz must be a positive multiple of 32 and batch must be positive")
    fold_key = "smoke" if smoke else (fold if fold is not None else "")
    default_output = "smoke.csv" if smoke else "baseline.csv"
    output = output or Path("experiments/results") / default_output
    # A run-local report also prevents accidental repeated hold-out evaluation.
    report = run / "evaluation" / f"{domain}_{fold_key}_{split}"
    result_path = report / "result.json"
    if result_path.is_file() and (not overwrite or (fold == "final" and split == "test")):
        raise FileExistsError(f"evaluation already recorded: {result_path}")

    device = resolve_device(cfg.device)
    metrics = build_model(weights).val(
        data=str(dataset), split=split, imgsz=size, batch=batch_size,
        device=device, workers=cfg.workers, conf=0.001, plots=True,
        project=str(report.parent), name=report.name, exist_ok=True,
    )
    if tuple(metrics.nt_per_class) != tuple(counts[code] for code in CLASSES):
        raise ValueError("Ultralytics target counts differ from the evaluated labels")
    box = metrics.box
    ap = per_class_columns(box.ap50, box.ap_class_index)
    # The validator may return no AP entries when there are no correct detections.
    present = {int(i) for i in box.ap_class_index}
    for index, code in enumerate(CLASSES):
        if not counts[code]:
            ap[f"ap_{code}"] = float("nan")
        elif index not in present:
            if present:
                raise ValueError(f"validator did not return AP for present class {code}")
            ap[f"ap_{code}"] = 0.0

    row = dict(exp_id=trained_id, phase="C2-smoke" if smoke else "C2", model=cfg.model,
               imgsz=size, train_domain=cfg.train_domain, test_domain=domain, split=split,
               fold=fold_key, condition="clean", method="baseline",
               n_labeled=metadata["n_labeled"],
               seed=cfg.seed, map50=float(box.map50), map50_95=float(box.map),
               precision=float(box.mp), recall=float(box.mr),
               f1=f1(float(box.mp), float(box.mr)), **ap, n_test_images=len(images),
               **{f"n_inst_{code}": counts[code] for code in CLASSES}, latency_ms="",
               notes="pipeline smoke only" if smoke else "")
    report.mkdir(parents=True, exist_ok=True)
    json_row = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
    write_json(result_path, {
        "row": json_row, "weights_sha256": sha256_file(weights),
        "dataset_yaml_sha256": sha256_file(dataset), "device": device,
        "train_environment": metadata, "evaluation_code": git_state(),
    })
    append_result(row, output, overwrite=overwrite)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--fold", type=parse_fold)
    parser.add_argument("--exp-id")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        row = evaluate(args.weights, domain=args.domain, split=args.split, fold=args.fold,
                       exp_id=args.exp_id, data_root=args.data_root, imgsz=args.imgsz,
                       batch=args.batch, output=args.output, overwrite=args.overwrite)
        print(f"{row['exp_id']}: {row['test_domain']} {row['split']} mAP50-95={row['map50_95']:.6f}")
    except (OSError, KeyError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        parser.exit(2, f"rdd evaluate: {exc}\n")


if __name__ == "__main__":
    main()
