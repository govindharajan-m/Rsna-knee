# MRI Preprocessing

I reconstruct each valid series by reading DICOM metadata, ordering slices from spatial geometry when possible, and decoding one series at a time into a raw NumPy volume. I retain original geometry, ordering strategy, spacing checks, source paths, and quality flags separately from processed arrays.

I treat percentile clipping, min-max, z-score, and robust normalization as configurable experimental choices. I do not claim that the baseline percentile-min-max setting is final. I also leave target spacing and image size unset by default, so I can compare later experiments such as 336 × 336 and 384 × 384 from explicit configuration.

I use an image-derived foreground candidate as initial ROI groundwork. It records its bounding box, margin, confidence, and fallback status. I do not treat this candidate ROI as verified anatomy; a failed or unreliable estimate returns the full volume.

I build cache entries under `preprocessing version / configuration fingerprint`. I invalidate an entry when the transform configuration or ordered source-file signature changes. I save the active configuration alongside cache artifacts and keep cache files out of Git.

I use `python scripts/prepare_dataset.py --data-root <PATH> --qc-samples 8` to build the cache, write a Parquet study manifest, calculate intensity summaries, run basic leakage checks, and save montage images for manual review.

## MRI-only training baseline

`scripts/train_baseline.py` uses the strict `reconstruct_ordered_volume` path
from `data/volume.py` to decode every selected pixel slice and fail on an
unreadable or dimension-inconsistent series. Each study tensor has fixed
`[series channel, depth, row, column]` layout with channels `[Axial, Coronal,
Sagittal]`. Per plane, the candidate rule is the lexicographically first
`SeriesInstanceUID`; it does not rank by image quality or SeriesDescription.
The initial `[24, 96, 96]` target size is a computational setting, not a
physical-spacing claim. Training uses finite-value percentile clipping and
per-volume scaling to `[0, 1]`.

The final Kaggle notebook checks for an official `test_series.csv`. If it is not
provided, the notebook uses the gold-training manifest to measure a one-to-one
mapping from manifest plane names to dominant axes of the DICOM slice normal.
Test series are assigned using that mapping, without inventing patient-side
direction labels. If the measured mapping is absent, ambiguous, or overlapping,
inference fails rather than guessing.
