# Bangladesh road image collection

Bangladesh supplies the primary baseline and can also serve as the target for a later Japan-to-Bangladesh study. This protocol defines what to record so frames, labels, and held-out routes can be reconstructed later. Collection and labeling are pending.

## Before recording

- Assign a stable route ID to each geographic route or corridor, such as `R007`. Reuse it on every visit, in both directions and across days. Give each video its own ID, such as `V003`. Merge materially overlapping routes into one group before splitting and record the mapping.
- Use a forward-facing smartphone on a steady dashboard or windshield mount as the default. Keep the angle and height consistent when practical. This reduces camera setup differences, but camera, weather, road surface and collection differences can still contribute to the Japan-to-Bangladesh gap. Record any alternate setup as a separate capture condition.
- Record at least 1080p, keep the originals, and apply resizing in the pipeline. Review a short pilot for focus, glare, and motion blur. Aim for several recording days and a mix of highway, urban, and rural roads and conditions where feasible.
- Mount the phone before moving and leave it alone while driving. A passenger can operate it if needed. Check applicable local recording and sharing rules before collecting. Review release copies and blur faces and plates before sharing. Keep raw video and any precise route or GPS information private.

## Files and metadata

```text
data/raw/bangladesh/
├── videos/<route_id>/<video_id>.mp4
├── metadata/routes.csv
├── frames/
└── labels/
```

Use the [metadata header](templates/bangladesh_routes.csv). Each row describes one video. Use ISO dates, resolution such as `1920x1080`, numeric fps and camera height in cm when known, and `true` or `false` for `has_gps`. Leave unknown measurements blank. Store approximate speed in km/h and use `highway`, `urban`, or `rural` for `road_type`. Keep exact locations and coordinates out of the shareable metadata. Use IDs with letters, digits, and hyphens so underscores remain separators. Frames inherit their video's route ID and use `BD_<route_id>_<video_id>_<frame_no>.jpg`, for example `BD_R007_V003_000240.jpg`. Keep the original video frame number.

## Splits and label budgets

Aim for at least 10 distinct route groups as a starting point. Prioritize independent routes over many similar frames. Reserve final test routes before assigning the other routes to development folds. Keep test routes out of training, augmentation, pseudo-labeling, and model selection. Test annotations are needed for scoring and stay reserved for final evaluation. Freeze the route assignments, record overlap checks, and use the [labeling guide](labeling_guide.md) before annotating the main collection.

Draw nested, seeded sets of 25, 50, 100, and 300 labeled images from adaptation routes and reuse each set across methods. Use the same held-out test routes for the Japan zero-shot model, fine-tuned variants, and a Bangladesh-only reference trained from the same COCO initialization. Report the Bangladesh-only reference's label budget and report per-class test instance counts. Treat AP below roughly 30–50 test instances for a class as uncertain. Repeat model runs with several seeds and keep the data splits fixed. When estimating test uncertainty, resample at the route level if enough test routes are available.
