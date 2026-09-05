# Data Leakage Audit

I keep cross-validation at the study or appropriately verified patient grouping level; I do not split individual MRI slices independently.

I record studies with multiple series, exact duplicate cached volumes, metadata-similar series, and optional patient-group overlap when a permitted identifier mapping is explicitly available. I treat geometry similarity as a review candidate, not proof of duplicate image content.

I have not made a leakage finding for the competition data because no competition dataset is available in this workspace. I will record measured findings in `data_leakage.json` after cache preparation runs on a supplied data root.
