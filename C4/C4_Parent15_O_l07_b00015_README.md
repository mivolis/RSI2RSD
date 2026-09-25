# Parent-15 O supplement: lambda 0.7, beta 0.0015

Five new one-cycle parent trajectories (seeds 202/303/404/505/606) came from Sapelo array 48515078. All five tasks ended `COMPLETED 0:0`; each parent has complete status, 15 block markers and 1,410 update records. CPU analysis job 48515945 ended `COMPLETED 0:0`, checked SHA256 of every consumed step archive, update log and block summary, independently reproduced the selected hard CE, and emitted 7,050 step rows, 75 block rows and five cell rows. This is a source-data audit, not a full checkpoint-manifest audit.

- [English PDF](C4_Parent15_O_l07_b00015_EN.pdf) and [source Markdown](C4_Parent15_O_l07_b00015_RESULTS.md); four A4 landscape pages rendered and visually inspected.
- [Per-step CSV](C4_Parent15_O_l07_b00015_O_steps.csv): unsmoothed before/after CE for all, selected and unselected images.
- [Per-block CSV](C4_Parent15_O_l07_b00015_O_blocks.csv): 75 image-weighted block summaries and audit CE.
- [Per-cell CSV](C4_Parent15_O_l07_b00015_O_cells.csv): five parent summaries.
- [Source manifest](C4_Parent15_O_l07_b00015_source_manifest.json): exact parent paths. Bulk raw files remain on Sapelo.

Selected-only hard pseudo-label CE is the historical training objective. All-image and unselected hard CE, soft mixed-target CE, and offline true-label CE are labelled separately. No new extension, fork, J or confirmation audit is involved.
