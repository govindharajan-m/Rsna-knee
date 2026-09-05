# MRI Preprocessing

I reconstruct each valid series by reading DICOM metadata, ordering slices from spatial geometry when possible, and decoding one series at a time into a raw NumPy volume. I retain original geometry, ordering strategy, spacing checks, source paths, and quality flags separately from processed arrays.

I treat percentile clipping, min-max, z-score, and robust normalization as configurable experimental choices. I do not claim that the baseline percentile-min-max setting is final. I also leave target spacing and image size unset by default, so I can compare later experiments such as 336 × 336 and 384 × 384 from explicit configuration.

I use an image-derived foreground candidate as initial ROI groundwork. It records its bounding box, margin, confidence, and fallback status. I do not treat this candidate ROI as verified anatomy; a failed or unreliable estimate returns the full volume.

I build cache entries under `preprocessing version / configuration fingerprint`. I invalidate an entry when the transform configuration or ordered source-file signature changes. I save the active configuration alongside cache artifacts and keep cache files out of Git.

I use `python scripts/prepare_dataset.py --data-root <PATH> --qc-samples 8` to build the cache, write a Parquet study manifest, calculate intensity summaries, run basic leakage checks, and save montage images for manual review.
