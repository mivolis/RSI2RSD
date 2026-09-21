# C4: self-referential continual test-time adaptation

Exploratory implementation of the integrated C4 protocol (September 21, 2026). Work remains on `experiment/C4`; stable versions can be merged by the repository maintainers later.

## Current experiment

- CIFAR-100-C, severity 5, all 15 corruptions in a fixed repeated sequence.
- Frozen source anchor, trainable student, EMA teacher. Feedback vector: teacher-target weight lambda and teacher refresh beta.
- Meeting pilot: lambda 0 and 1, beta 0.001, seed 101, two complete cycles.
- Fixed image-identity splits: adaptation / reserved gate / exploratory audit / untouched confirmation.
- Save per-update predictions, masks, pseudo-labels, IDs, paired randomness, local probes, full boundary checkpoints and fixed-reference-panel measurements.

## Evidence status

Preparation and software checks are complete for the natural learner and fixed-reference panel. Scientific GPU results are not yet available. Synthetic tests are software checks, never scientific evidence. Replay/fork analyses will be reported separately when implemented and validated.

## Reproduction

Install the pinned requirements, obtain the pinned public CoTTA transforms and RobustBench checkpoint, and run `prepare_data.py`. The setup and pilot scripts document the exact commands. The code implements the C4 hard-label update, not unmodified CoTTA.

`report.py --verify` recomputes measurements from retained raw records and validates committed block hashes. Only complete blocks enter summaries; interrupted attempts are retained.

## Outputs

Curated reports and artifact manifests will be added under this directory. Large raw archives and checkpoints will be preserved with checksums and a retrieval index. Credentials and private source documents are excluded. One seed does not establish a phase transition or a causal feedback mechanism.
