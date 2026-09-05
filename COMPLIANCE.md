# Competition Compliance Working Record

I use this document as a working record for the RSNA Knee MRI Abnormality Detection competition. It is not a substitute for the current official rules. I verify requirements incrementally before relying on them.

## Verification convention

- `[ ] not yet verified` means I need an authoritative current source before I rely on the item.
- `[x] verified/documented` records an item with an authoritative source noted here.

## Competition Identity
[x] verified — **Competition**: RSNA Knee MRI Abnormality Detection (Kaggle)
- Source: sample_submission.csv in competition data
- 12 findings per study (probabilities 0-1)

## Target Labels (12 Findings)
[x] verified — The competition requires predictions for exactly these 12 findings:
1. ACL (Anterior Cruciate Ligament)
2. MCL (Medial Collateral Ligament)
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA (Osteoarthritis)
6. Lateral OA
7. PF OA (Patellofemoral OA)
8. Effusion
9. Synovitis
10. Baker's (Cyst)
11. Contusion
12. Fracture

Source: sample_submission.csv column headers

## Evaluation Metric
[ ] not yet verified — Macro-AUC mentioned in public solutions but not confirmed as official metric.
Source: knee_mri_training_the_twelve_finding_model.py comments
Will verify against official Kaggle competition rules page.

## Submission Format
[x] verified — CSV with:
- Column 1: StudyInstanceUID (DICOM study identifier)
- Columns 2-13: Probability predictions for each finding (0.0 to 1.0)
- Separator: comma
- Header row required

Source: sample_submission.csv

## Dataset Structure
[x] verified from context — Training dataset contains:
- 4,407 studies total
- 58 studies with structured labels (known findings)
- 4,349 studies with free-text radiology reports only
- Input: DICOM MRI files or reconstructed volumes
- Labels must be extracted from reports (report NLP problem)

Source: knee_mri_training_the_twelve_finding_model.py

## Input Data Format
[x] verified — DICOM MRI format
- Multiple slices per study (variable count)
- Variable in-plane dimensions and spacing
- Requires DICOM parsing and slice ordering
- Frame of Reference: 3D Cartesian space

Source: Existing kneescope12 infrastructure and DICOM audit capabilities

## Output Probability Requirements
[x] verified — Predictions must be continuous probabilities (0.0-1.0 range)
- Not binary {0, 1} 
- Soft labels recommended (hedged language → intermediate probabilities)
- Example: report saying "suspected tear" → ~0.8, not hard 0 or 1

Source: knee_mri_training_the_twelve_finding_model.py methodology section

## Account requirements
[ ] not yet verified — To be verified against the current official competition rules.

## Team requirements
[ ] not yet verified — To be verified against the current official competition rules.

## External data

[ ] not yet verified — To be verified against the current official competition rules.

I will not use any external dataset unless I have recorded its source, license, access conditions, and competition status in the External Resource Registry.

## Pretrained models

[ ] not yet verified — To be verified against the current official competition rules.

I will verify whether pretrained weights are allowed, any training-data restrictions, model-license terms, and disclosure requirements before introducing a pretrained model.

## External code

[ ] not yet verified — To be verified against the current official competition rules.

I will maintain attribution and license records for external code. I will independently implement project components and will not copy a public solution's implementation into this project.

## Licensing

[ ] not yet verified — To be verified against the current official competition rules.

I will confirm that project and dependency licenses are compatible with the competition and final-submission requirements.

## Private data

[ ] not yet verified — To be verified against the current official competition rules.

I will verify whether private or institutional data is permitted before accessing or combining it with competition material.

## Test-set restrictions

[ ] not yet verified — To be verified against the current official competition rules.

I will verify restrictions on accessing, inspecting, labeling, or otherwise using test-set information. My intended final inference path uses permitted MRI/test inputs only and does not depend on unavailable radiology reports.

## Human annotation and prediction restrictions

[ ] not yet verified — To be verified against the current official competition rules.

I will verify all restrictions on manual labeling, human prediction, and human-in-the-loop use of test examples.

## Submission limits

[ ] not yet verified — To be verified against the current official competition rules.

I will record daily, total, team, and deadline-related limits before submitting.

## Final-submission requirements

[ ] not yet verified — To be verified against the current official competition rules.

I will verify requirements for selecting, documenting, and reproducing a final submission.

## Reproducibility requirements

[ ] not yet verified — To be verified against the current official competition rules.

I intend to retain configurations, source revision, environment details, seeds, and dataset versions. I will verify any official reproducibility requirements separately.

## Winner requirements

[ ] not yet verified — To be verified against the current official competition rules.

I will verify any code-sharing, documentation, interview, audit, or publication obligations that apply to prize recipients.

## Computational environment

[ ] not yet verified — To be verified against the current official competition rules.

I will verify restrictions on hardware, hosted services, internet use, external APIs, and execution environments.

## Documentation requirements

[ ] not yet verified — To be verified against the current official competition rules.

I will verify disclosure, attribution, model-card, and methodology documentation requirements.


## External Resource Registry

I will populate this registry as I introduce resources. A blank registry is intentional: I have not verified any external resource for competition use yet.

| Resource name | Source | Version | Purpose | License | Accessibility | Competition compliance status | Notes |
| ------------- | ------ | ------- | ------- | ------- | ------------- | ----------------------------- | ----- |
| _No resources registered yet._ | — | — | — | — | — | Not yet verified | I will add a source and verification notes before use. |

## Verification log

| Date | Item | Official source | Finding | Recorded by |
| ---- | ---- | --------------- | ------- | ----------- |
| _No entries yet._ | — | — | — | — |
