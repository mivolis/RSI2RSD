# C4 parent-15 proxy CE: lambda=0.7, beta=0.0015

Purpose: collaborator report for the one-cycle parent and comprehensive proxy-CE observables. Export target: landscape-A4 PDF.

Source: `Projects/RSD_C4_CTTA/Experiments/Yingchuan_Sweep_20260923/Parent15_O_20260924/Evidence/l07_b00015/source_manifest.json` and the Sapelo parent paths it lists. All five seeds have 15 verified parent blocks.

Hard all-image proxy CE is computed on every adaptation-batch image against the fixed saved pseudo-label, before and after the same update. The historical selected-only CE is reported separately and verified against source logs. Soft-target and offline true-label CE are additional diagnostics, not substitutes for the training objective.

Step, block and cell records: `O_steps.csv`, `O_blocks.csv`, `O_cells.csv`. No extension, new fork or J measurement is included. This finite exploratory parent cannot establish sustained deterioration or a phase boundary.
