"""Train and evaluate CV folds for each selected seed. Final hold-out is separate."""

import argparse
from pathlib import Path

from rdd.evaluation.evaluate import evaluate
from rdd.training.config import load_config
from rdd.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/baseline_bd.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    try:
        base = load_config(args.config, data_root=args.data_root)
        folds = list(range(base.n_folds)) if args.folds is None else args.folds
        if len(set(folds)) != len(folds) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("folds and seeds must be distinct")
        configs = [load_config(args.config, fold=fold, seed=seed, data_root=args.data_root)
                   for fold in folds for seed in args.seeds]
        for cfg in configs:
            best = train(cfg)
            evaluate(best, domain=cfg.train_domain, split="test", fold=cfg.fold)
    except (OSError, KeyError, RuntimeError, ValueError, ImportError) as exc:
        parser.exit(2, f"run-cv: {exc}\n")


if __name__ == "__main__":
    main()
