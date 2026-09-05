# Dataset Audit

I use `python scripts/audit_dataset.py --data-root <PATH>` to create a metadata-only inventory of a supplied DICOM dataset. The command writes a Markdown report, JSON summary, series inventory, study inventory, and diagnostic log under the configured audit output directory.

I do not infer a preferred MRI sequence from filenames or from an external solution. I review the measured series-description prevalence, modality distribution, dimensions, spacing, orientation, laterality, missing fields, and quality flags before making a later sequence-selection decision.

I have not run this audit against the competition dataset in this workspace because no competition data path is available here. I will add measured findings to generated audit outputs only after running the command on an explicitly supplied data root.

I use `python scripts/audit_training_resources.py --data-root <PATH> --labels-path <PATH>` when an explicit label table is available. The resource audit records label encoding and report availability without placing report text in logs or manifests.
