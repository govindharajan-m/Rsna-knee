# Dataset Audit

I use `python scripts/audit_dataset.py --data-root <PATH>` to create a metadata-only inventory of a supplied DICOM dataset. The command writes a Markdown report, JSON summary, series inventory, study inventory, and diagnostic log under the configured audit output directory.

I do not infer a preferred MRI sequence from filenames or from an external solution. I review measured series metadata, modality distribution, dimensions, spacing, orientation, laterality, missing fields, and quality flags before making a later sequence-selection decision.

## Competition Dataset: Initial Measured Audit

I verified the RSNA Knee Abnormality Detection competition dataset in the Kaggle execution environment.

The competition data is available under:

`/kaggle/input/competitions/rsna-knee-abnormality-detection`

The supplied training metadata contains:

- 4,407 training studies.
- 4,407 unique training studies represented in `train_series.csv`.
- 3–14 series per study in the observed training metadata.
- Mean of approximately 5.53 series per study.
- `train.csv` contains 4,407 rows and 14 columns.
- `train.csv` contains `StudyInstanceUID`, `Report`, and 12 finding columns.
- The 12 finding columns contain binary values (`0.0` and `1.0`) for the explicitly labeled subset.
- 58 studies have explicit binary labels across all 12 findings.
- The remaining 4,349 studies have reports but do not have explicit binary labels.

The 12 findings are:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

## Series Metadata

`train_series.csv` provides:

- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Fluid_Sensitive`
- `Fat_Suppression`
- `Anatomical_Plane`

The observed training studies contain multiple series and multiple anatomical planes. I therefore do not treat a study as a single image or assume that one fixed series is sufficient for every study.

## DICOM Geometry Audit

I inspected DICOM metadata from a sample of 100 training series, covering 3,429 DICOM files.

The observed native image dimensions varied substantially, including dimensions from `256 × 256` through `1280 × 1280`, with both square and non-square dimensions present.

Observed `SliceThickness` values ranged from approximately 0.6 mm to 5.0 mm, with a median of approximately 3.0 mm.

`SpacingBetweenSlices` was present in 2,974 of the 3,429 inspected DICOM files. It was absent in 455 files.

The following fields were present in all 3,429 inspected DICOM files:

- `ImagePositionPatient`
- `ImageOrientationPatient`
- `InstanceNumber`

A sampled series demonstrated that DICOM filenames are not reliable anatomical slice ordering keys. The first three files returned by filename sorting had `InstanceNumber` values of 5, 10, and 3.

For volume reconstruction, I therefore use physical image geometry as the primary ordering signal. `ImagePositionPatient` and `ImageOrientationPatient` provide the information required to derive physical slice coordinates and establish anatomical ordering. `InstanceNumber`, `SliceThickness`, and `SpacingBetweenSlices` are treated as supporting metadata and quality-control signals rather than blindly trusted ordering assumptions.

Across the sampled 100 series:

- Mean number of slices per series: approximately 34.3.
- Range of slices per series: 16–176.
- Mean median physical slice spacing: approximately 3.69 mm.
- Median median physical slice spacing: approximately 3.30 mm.
- Range of median physical slice spacing: approximately 0.60–6.78 mm.

Within the sampled series, physical slice spacing was generally internally consistent, based on the close agreement between the minimum, median, and maximum adjacent physical-position spacing measurements.

## Current Preprocessing Implications

The initial audit supports the following implementation requirements:

1. I must not rely on DICOM filenames for anatomical slice ordering.
2. I should derive slice ordering from physical DICOM geometry.
3. I should not assume a fixed native image dimension.
4. I should not assume a single native voxel spacing across series.
5. I should use available physical-position metadata when reconstructing volumes.
6. I should treat missing `SpacingBetweenSlices` as recoverable when sufficient physical-position information is available.
7. I should retain series-level metadata such as anatomical plane, fluid sensitivity, and fat suppression for later sequence-selection and quality-control decisions.
8. I should keep report-derived labels separate from explicit labels during later weak-supervision development.
9. I should not treat missing explicit labels as negative findings.

This document records measured observations from the initial Kaggle audit. It does not define the final model architecture or claim competition performance.

## Training Resources

I use `python scripts/audit_training_resources.py --data-root <PATH> --labels-path <PATH>` when an explicit label table is available. The resource audit records label encoding and report availability without placing report text in logs or manifests.