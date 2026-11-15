# Bangladesh labeling guide

Pilot this guide on a small set of frames before labeling the main collection. Keep the same class IDs as the Japan baseline.

| ID | Class | How to identify it |
| --- | --- | --- |
| 0 · D00 | Longitudinal crack | A visible crack that runs mainly along the road's direction. |
| 1 · D10 | Transverse crack | A visible crack that runs mainly across the road's direction. |
| 2 · D20 | Alligator crack | A connected network of cracks forming irregular cells. Box the affected patch. |
| 3 · D40 | Pothole | A visible depression with pavement material missing. Check that shadows, puddles, and patches are distinguishable before assigning the class. |

Draw a tight axis-aligned box around the visible damage and keep it inside the image. Label distinct damages separately. Use one box for a connected alligator patch instead of counting its constituent cracks again. For partially visible damage, box the visible extent when class evidence is sufficient. Flag ambiguous cases for review. A reviewed image with no visible target damage gets an empty label. An unreviewed image remains unlabeled.

Use the pilot to agree on minimum visibility, diverging cracks, repairs, and borderline cases. Keep adjudicated examples and record rule changes before applying them to the main labels. Have a second annotator independently label about 10% of the images across routes and conditions, then review class and box disagreements. Preserve the route and video IDs through export.
