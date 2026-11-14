# Robust and Efficient Road Damage Detection Under Domain Shift

This master's project will measure how road damage detectors transfer across regions and conditions, how cheaply they can adapt, and what accuracy they retain on edge hardware. This repository currently covers C1, foundation and RDD2022 preparation. C1 maps to research Phase 0 and the data portion of research Phase 2. Training, adaptation and deployment are reserved for later phases.

## Install and test

Use Python 3.12. From this repository root, on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS, use `python3.12 -m venv .venv` and replace the interpreter above with `.venv/bin/python`. Alternatively, use `conda env create -f environment.yml` from this directory. C1 runs on CPU and has no PyTorch, YOLO or tracking account requirement.

## Prepare RDD2022

1. Obtain the [official Figshare v1 archive](https://doi.org/10.6084/m9.figshare.21431547.v1) (about 13.3 GB; MD5 `b62bd51d2ffcfaa76c60f234f0cc2bb3`). The outer ZIP contains country ZIPs; extract the relevant inner archives as well. [Publisher download links and label definitions](https://github.com/sekilab/RoadDamageDetector#dataset) are also available. Allow disk space for both the extracted images and a processed copy. The dataset is [CC BY 4.0](https://api.figshare.com/v2/articles/21431547); cite Arya et al., [RDD2022](https://doi.org/10.1002/gdj3.260). The repository's code license does not cover the dataset.
2. Place the extracted directories under `data/raw/rdd2022`, keeping the original names and the paths `<domain>/train/images/` and `<domain>/train/annotations/xmls/`. An extra enclosing archive directory is fine. Official challenge test images are unlabeled and ignored.
3. Run preparation from the repository root:

```powershell
.\.venv\Scripts\python.exe -m rdd.data.prepare --config configs/data/rdd2022.yaml
```

For a Japan-only run, use `configs/data/rdd_japan.yaml`. Each run needs a fresh output directory. The CLI also exposes this command as `rdd-prepare` in an activated environment.

The output has one directory per country with `images/{train,val,test}`, matching `labels/{train,val,test}`, a YOLO `dataset.yaml`, and split text files relative to that country's directory. Rebuild or update the generated YAML's absolute `path` if you move the dataset. China pools Drone and MotorBike for the country split while retaining capture domains in the manifest and statistics.

- `manifest.csv`: source and output paths, SHA-256 hashes, split membership, class counts, skipped classes and clipped box counts.
- `stats.csv` and `dataset_stats.csv`: images, negatives, instances per class, image-size counts and relative box area buckets by capture domain and split. Add domain rows to get country counts for China.
- `run.json` and `experiments/logs/prepare-*.json`: effective config, seed, environment versions, counts, and hashes of the manifest and stats.

See the [data protocol and validation gate](docs/experiment_plan.md) for the conversion convention, leakage limits, and checks to perform before baseline training.

The [C1 verification record](docs/phase1_validation.md) documents the official sample used for the smoke test.

On Bash, run `bash scripts/run_data_pipeline.sh --config configs/data/rdd2022.yaml` from the repository root with the environment active. The dataset also includes image lists under `splits/<Country>/` with paths relative to the prepared dataset root and two PNG figures, `images_by_country.png` and `boxes_by_class.png`.

To check a converted annotation, copy a `source_image` value from the manifest and render it as a PNG. The following command uses the six-image China Drone sample and assumes its prepared directory exists.

```powershell
.\.venv\Scripts\python.exe -m rdd.data.preview data/processed/rdd2022_block_smoke China_Drone/train/images/China_Drone_000000.jpg data/block-preview.png
```
