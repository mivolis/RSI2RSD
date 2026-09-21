# C4 artifact availability — 2026-09-21

Repository branch: `experiment/C4`; all project files under `C4/`. Main is unchanged.

## Compact evidence

- `compact_results.tar.gz`: natural pilot and original H20/H94 forks on Vast.
- `sapelo_compact_results.tar.gz`: six Sapelo forks, H20/H94/H1410 after blocks 5/15;
  config, schedules, metrics, contrasts, figures, completion manifests, runtime and
  verification records. SHA256 `61604acd38cce60ef872f5b2a9044c5dc4b6a4de077dd28f6699bc11c982956e`; 1,491,405 bytes.
- `RESULTS.md`: complete scientific interpretation, including negative results.
- `C4-5_after5_H1410.png`, `C4-5_after15_H1410.png`: full-cycle fork curves.

## Vast full records

All 7,250 required source files were recovered and SHA256-verified locally before
instance 51934516 was destroyed. A round-trip-verified archive includes these
plus provenance (7,276 files, 10,873,969,648 bytes). Its SHA256 is
`130a4663aad72593adb6dba086d32edd567f6c9333f9e6707d7a1e69725f763e`.

The full archive is **published and available** in the
[Vast raw evidence pre-release](https://github.com/mivolis/RSI2RSD/releases/tag/c4-pilot-20260921),
tag `c4-pilot-20260921`, created from `experiment/C4` commit
`9be0f1aa8ef17d3a127f2f46e8a681cfce596e8f`.
All 12 parts and three manifests are present; all 15 GitHub-displayed SHA256
digests match the local verified evidence. Record: `VAST_RELEASE_VERIFICATION.json`.
This compares GitHub's displayed digests; no second full download was performed.

Download `.part00` through `.part11` plus `RAW_ARCHIVE_INDEX.json`,
`RAW_FILES_SHA256.json` and `SHA256SUMS`. Verify the parts, concatenate them in
numeric order to recreate `c4_vast_raw_20260921.tar.gz`, then extract in an empty
directory. The release includes exact reconstruction commands.

Local retained copy: experiment `remote/`; verified archive: `Delivery/raw/`.
No private manuscript or credential is included. Public CIFAR input images are
excluded from the archive; their exact source/hash provenance is retained.

## Sapelo full records

Canonical host: Sapelo; job `48416583`, node `rb7-1`, partition `zhai_p`.
Canonical directory: `/scratch/yz54720/RSI2RSD/C4/migration_results/job48416583/`.
Code: `/scratch/yz54720/RSI2RSD/C4/`; Slurm script `slurm/sapelo_forks.sbatch`.
Logs: `logs/slurm-48416583.out`, `.err`; scheduler record `logs/sacct-48416583.txt`.

Job completed, exit `0:0`. All six fork verifications passed. Each H1410 parent
has 4,364 hashed files and 4,230 raw future records. Full states, logits, target
probabilities, masks, sample IDs and paired randomness are retained remotely.
The final local bulk mirror is **complete and hash-verified** as of September 21,
2026, 14:27 EDT. All six completion manifests passed (9,588 referenced files),
including all 9,144 future raw batch records. The verification record is
`SAPELO_MIRROR_VERIFICATION.json` in C4 (canonical local record:
`Sapelo/mirror_verification.json`). Full local mirror: `Sapelo/job48416583/`.

A separate full Sapelo archive is **complete and round-trip verified locally**:
9,632 files including provenance, 14,678,916,359 bytes. Its SHA256 is
`0147e6a1a1c55799074e076ba6a3fa80691f86af65b7e18f080a1d60c95f1ad1`.
All archive members were read back and rehashed. `SAPELO_RAW_ARCHIVE_INDEX.json`,
`SAPELO_RAW_FILES_SHA256.json` and `SAPELO_SHA256SUMS` describe its contents.

The 16 parts and three manifests are **uploading to a separate draft release**,
tag `c4-sapelo-48416583-20260921`, targeting `experiment/C4`; they are not yet a
completed collaborator download. Verified local archive: `Delivery/sapelo_raw/`.
Compact evidence also matches host manifests.
Cross-host repeats are not independent seeds.
