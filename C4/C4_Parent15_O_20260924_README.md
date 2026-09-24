# C4 lambda=0.7 parent-15 and full proxy CE supplement (2026-09-24)

This is a new one-cycle, 15-block supplement after the earlier three-cycle sweep was stopped. No extension, fork or Jacobian job is part of this run. The first two beta groups reuse five previously completed parents each; later beta groups will use one zhai_p L40S GPU sequentially. Seeds are 202, 303, 404, 505 and 606. The confirmation audit remains closed.

"O" is reported without collapsing distinct definitions. The original selected-only hard pseudo-label CE is the training objective; the all-image and unselected hard CE are additional measurements. Soft mixed-target CE and offline true-label CE are separate diagnostics. Each CSV has before/after values at every minibatch and image-weighted block and seed summaries; empty masks are NA.

| beta | status | report | step data | block data | seed data | source paths |
|---|---|---|---|---|---|---|
| 0.0005 | Five parent-15 trajectories and O analysis verified | [PDF](C4_Parent15_O_l07_b00005_EN.pdf) | [CSV](C4_Parent15_O_l07_b00005_O_steps.csv) | [CSV](C4_Parent15_O_l07_b00005_O_blocks.csv) | [CSV](C4_Parent15_O_l07_b00005_O_cells.csv) | [JSON](C4_Parent15_O_l07_b00005_source_manifest.json) |
| 0.001 | Five parent-15 trajectories and O analysis verified | [PDF](C4_Parent15_O_l07_b0001_EN.pdf) | [CSV](C4_Parent15_O_l07_b0001_O_steps.csv) | [CSV](C4_Parent15_O_l07_b0001_O_blocks.csv) | [CSV](C4_Parent15_O_l07_b0001_O_cells.csv) | [JSON](C4_Parent15_O_l07_b0001_source_manifest.json) |

Beta 0.0015 and 0.002 are not yet complete. [Analysis and run code](C4_Parent15_O_Code_20260924.tar.gz). Bulk batch tapes and checkpoints remain on Sapelo at `/scratch/yz54720/RSI2RSD/C4/`. The analysis checks hashes of each consumed step archive, update log and block summary, then reconciles selected CE with original logs. This is narrower than full checkpoint-manifest verification. The reports do not claim sustained deterioration or an identified Jacobian.
