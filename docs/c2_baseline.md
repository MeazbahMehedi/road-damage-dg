# C2 Bangladesh baseline

The primary baseline trains and evaluates on Bangladesh. The RDD2022 pipeline remains available for later domain shift comparisons, including Japan. Until Bangladesh data is ready, use the six-image China Drone sample for the CPU smoke run.

## Install and smoke test

Use Python 3.12. For a Windows CPU environment, install PyTorch first, then install the pinned C2 requirements from the repository root.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-c2.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Put the six official China Drone image/XML pairs under `data/raw/rdd2022_smoke/China_Drone/train/`. If the 4/1/1 prepared dataset is absent, run `python -m rdd.data.prepare --config configs/data/c2_smoke.yaml`. The pairs are `China_Drone_000000` through `China_Drone_000005`. See the [C1 record](phase1_validation.md) for provenance.

```powershell
python -m rdd.training.train --config configs/train/baseline_bd.yaml --smoke
python -m rdd.evaluation.evaluate --weights runs/yolov8n_China_smoke_s1/weights/best.pt --domain China --split val
```

The smoke run uses YOLOv8n, 320 px, one epoch, and zero workers. It writes `experiments/results/smoke.csv` with phase `C2-smoke`. Use a new seed for a new smoke run.

## Prepare the Bangladesh folds

Export verified YOLO annotations and unique frame names from the collection and labeling workflow. Put images under `data/processed/local_yolo/Bangladesh/images/` and labels under the matching `Bangladesh/labels/`. Keep one `manifest.csv` at the data root with `image,route_id` columns. Paths are relative to the data root, for example `Bangladesh/images/BD_R007_V003_000240.jpg`. Keep an empty label for a reviewed image with no target boxes. List the reserved routes, one per line, in `holdout_routes.txt`. Aim for about 20% of routes and choose them before model development.

```powershell
python -m rdd.data.make_folds --data-root data/processed/local_yolo --manifest data/processed/local_yolo/manifest.csv --holdout-routes data/processed/local_yolo/holdout_routes.txt
```

The command checks labels, rejects exact duplicates assigned to different routes, and freezes the remaining routes in five groups. Fold k uses group k for test and the next group for validation. The remaining groups are for training. With fewer than ten total routes it uses one development route per group. Set `n_folds` to the count in the generated plan if it differs from five. `final.yaml` uses group 0 for validation, the other development groups for training, and the reserved routes for test. The builder writes the YAML files, machine-specific image lists, portable stem lists, and `plan.json` under `Bangladesh/folds/`. Copy the reviewed stem lists to `splits/bangladesh/` for version control once the splits are frozen.

After reviewing the collection and labeling, write `Bangladesh/readiness.json` next to `folds/`:

```json
{
  "manifest_sha256": "<copy from folds/plan.json>",
  "plan_sha256": "<Get-FileHash of folds/plan.json, lowercase>",
  "labeling_guide_reviewed": true,
  "second_annotator_images": 30,
  "reported_classes": ["D00", "D10", "D20", "D40"],
  "privacy_reviewed": true
}
```

The numeric example assumes 300 frames. Record the actual count reviewed independently. The gate requires at least 300 labeled frames and about 10% independently annotated. For fewer than 10 routes, add a documented `route_exception` and use leave-one-route-out. Report class-level conclusions only for declared classes with at least 30 final test instances. Record `privacy_reviewed` after checking faces, plates, and redistribution permissions before external use. The JSON captures the human review. The code checks that it refers to the frozen manifest and fold plan.

## Train and evaluate

```powershell
python -m rdd.training.train --config configs/train/baseline_bd.yaml --fold 0 --seed 1
python -m rdd.evaluation.evaluate --weights runs/yolov8s_Bangladesh_f0_s1/weights/best.pt --domain Bangladesh --split val --fold 0
```

Once settings are frozen, run the matching fold on `--split test`. The [Kaggle launcher](../notebooks/kaggle_train.ipynb) stages a writable copy and checks the regenerated plan against the frozen one. Keep the GitHub read token in a Kaggle Secret named `RDD_GITHUB_TOKEN` for private repositories. Set the input directory and exact commit in the first notebook cell. The notebook defaults to one validation run. Change its evaluation split to test only after settings are frozen.

Each run writes `config_used.yaml`, `env.json`, and `weights/{best,last}.pt` under `runs/<exp_id>/`. For interruptions, use the same config, paths, fold and seed with `--resume`. Restore the run directory and data at the same paths on Kaggle. An optimizer-bearing `last.pt` is required. Completed runs have stripped checkpoints. Keep model checkpoints and raw imagery in ignored storage.

```powershell
python scripts/run_cv.py --folds 0 1 2 3 4 --seeds 1 2 3
python -m rdd.evaluation.aggregate
```

The CV runner evaluates fold test routes. Run it after fixing settings and pass only folds and seeds that still need results. It leaves final hold-out evaluation to a separate `--fold final` training and test command, using a prespecified seed. Normal evaluation writes `experiments/results/baseline.csv` and run-local plots and `result.json`. Copy actual results back from Kaggle before tracking the CSV.

## How to read the CSV

| Fields | Meaning |
| --- | --- |
| `exp_id, phase, model, imgsz` | Run identifier, coding phase or smoke marker, initialization, and evaluation input size. |
| `train_domain, test_domain, split, fold` | Training and evaluation domains, evaluation split, and its fold or final hold-out. |
| `condition, method, n_labeled, seed` | Evaluation condition, training strategy, labeled training image count, and model seed. |
| `map50, map50_95` | Mean AP at IoU 0.50 and averaged over IoU 0.50–0.95. |
| `precision, recall, f1` | Macro precision and recall at the operating point selected by Ultralytics and their harmonic mean. |
| `ap_D00, ap_D10, ap_D20, ap_D40` | AP at IoU 0.50 by class ID. An absent class is `nan`. |
| `n_test_images, n_inst_D*` | Number of images and labeled instances in the evaluated split, including when the split is validation. |
| `latency_ms, notes` | Reserved benchmark latency and explanatory notes. |

Ultralytics 8.3.160 selects the confidence point by maximizing the smoothed mean per-class F1 curve. The CSV uses the harmonic mean of its macro precision and recall, which can differ from the average of class-level F1 scores. The reported mAP uses the precision–recall curve with a validator confidence floor of 0.001. [Pinned metric implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.160/ultralytics/utils/metrics.py)

The summary reports unweighted means and sample standard deviations over available fold–seed rows, together with run and metric counts. Standard deviation is undefined for a single observation. The summary omits absent-class APs from their means and flags a class if its weakest evaluated fold has fewer than 30 instances. Final hold-out rows and CV rows remain separate. These standard deviations describe observed variation across runs.


## Local verification

The six-image smoke run completed on Windows with Python 3.12.10, PyTorch 2.7.1 CPU and Ultralytics 8.3.160. Training used four images and produced both checkpoints and the run metadata. Evaluation used one validation image and wrote a smoke CSV row with the two present classes and missing-class AP marked as `nan`. A separate two-epoch probe resumed from `last.pt` after an injected interruption. The pinned Ultralytics loader also opened generated fold lists on a temporary fixture. The Kaggle notebook code cells compiled locally. A real Bangladesh run and a Kaggle session are pending.
