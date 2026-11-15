# C1 verification record

Verified with Python 3.12.10 on Windows: 11 tests passed and pip check found no broken requirements.

The CLI was exercised on six image/XML pairs (China_Drone_000000 through China_Drone_000005) from the [official Figshare v1 release](https://doi.org/10.6084/m9.figshare.21431547.v1). The subset was obtained from the nested China_Drone ZIP using byte ranges and the extracted members were checked against the ZIP entry CRCs. The raw subset is kept under ignored local data, separate from the full release directory.

To repeat using those six pairs under `data/raw/rdd2022_smoke/China_Drone/train/`:

```powershell
.\.venv\Scripts\python.exe -m rdd.data.prepare --config configs/data/rdd2022_smoke.yaml
```

The original image-level smoke run produced the following results with seed 42:

- 6 images split 4 train / 1 validation / 1 test.
- 12 retained boxes: 5 D00, 6 D10 and 1 D20. All 12 YOLO rows matched the source class and reconstructed XML edges to within 0.000001 pixel. There were no clipped or excluded boxes.
- Two runs to separate output directories produced identical SHA-256 hashes for the manifest and statistics CSVs. The manifest hash was `4489c656dd4ee5b06d595b2c5d7a6547922a8c521dc2136a0e7751e8157a0e37`.

These checks establish sample compatibility and repeatability. They do not establish the class balance, annotation quality or split independence of the full release. Follow the [validation gate](experiment_plan.md#validation-gate) before freezing splits or training a baseline. No model was trained or evaluated.

The current smoke config uses two-image filename blocks. It kept all three blocks intact and yielded 2 train, 2 validation and 2 test images. The seed and fixed groups determine the split, so the small sample cannot match the target ratios. The 12 retained boxes had the same class counts as before. The PowerShell and Git Bash runs produced identical manifests, statistics and figures. The preview CLI drew and labeled a D10 box on the first image, and the rendered PNG was opened for inspection.

Validation on the five selected RDD2022 countries is pending. Review all four class counts, per-country box counts and example overlays once the selected official archives are available.

## Verification of the supplied C1 ZIP

The workspace also contains a separate source archive, `road-damage-dg-step1-2.zip` (SHA-256 `b3fc82fc14c8fdce54f82a72e62534dd64cf937d742c95421512d663f24fc0e0`). It was extracted to a temporary directory for testing. The active repository and the source archive were kept separate.

- All 13 tests passed in the ZIP implementation. The active repository's own 11 tests also passed.
- The ZIP converter ran on the same six official China Drone pairs. It retained all 12 boxes. Reconstructing pixel edges from the six-decimal YOLO coordinates matched the original XML within 0.001 pixel. The six rendered overlays are in the ignored local file `data/c1-archive-verification/zip-sample-overlays.png`.
- A CSV calculated directly from the ZIP converter's YOLO labels is available locally at `data/c1-archive-verification/sample_stats.csv`. All six images were 512 × 512 and had boxes. The sample median box area was 2.9118% of image area.
- With seed 42 and group size 2, the ZIP splitter assigned 4 train, 2 val and 0 test images. A repeat produced identical split lists. The active repository's splitter assigns a group to each split when there are exactly three independent groups.
- The full-data command in the active checkout failed during discovery with `China_Drone: expected one labeled train tree ... found 0`. The raw directory was empty. The official S3 archive endpoints returned HTTP 403 from this machine. The Figshare range probe received 393,216 bytes in 18 seconds before timing out. Full-release counts and splits remain unavailable.

## Japan public comparison audit

The Japan preview used 12 official image/XML pairs from the Figshare v1 Japan archive. The subset was chosen from the smallest compressed images to make the transfer practical. It is a compatibility sample. The archive directory contained 10,506 paired training entries. The installed pipeline prepared all 12 downloaded pairs with seed 42 and two-image blocks for the preview. The real Japan config still uses 100-image blocks.

| Split | Images | D00 | D10 | D20 | D40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 8 | 10 | 1 | 2 | 0 |
| Validation | 2 | 2 | 1 | 0 | 0 |
| Test | 2 | 1 | 2 | 0 | 0 |

All 19 retained boxes matched the source XML classes and reconstructed pixel edges. Eight D44 boxes were excluded under the four-class protocol. There were zero dimension mismatches, clipped boxes, or exact duplicate images in this sample. All 12 overlays were opened for review. Two runs matched byte for byte on the split lists, manifest and statistics.

Full Japan C1 validation remains pending. The publisher's direct endpoint returned HTTP 403 during prior checks. A fresh Figshare byte-range probe transferred about 23 KB/s before timing out. The default Japan run found no extracted Japan tree. Complete the Japan audit and audit the other selected public domains before C4. Real C2 training is governed by the separate Bangladesh readiness gate.
