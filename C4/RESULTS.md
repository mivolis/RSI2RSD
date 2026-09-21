# C4 exploratory results — 2026-09-21

**Completed:** paired natural pilot (30 blocks and 2,820 updates per arm), plus
the two prespecified forks at horizons 20 and 94. One seed (101), severity 5,
two corruption cycles. Raw-record checks passed on the GPU host. Full local
recovery and the checksummed raw release are being finalized.

## Natural trajectories

Lower values are better. Lambda is the source/EMA mixing weight (`--gamma`).
The fixed reference Q always contains the same 1,500 image-corruption pairs.

| Measure | Frozen source | Lambda 0 | Lambda 1 |
|---|---:|---:|---:|
| Initial fixed-Q CE | 2.45255 | 2.45255 | 2.45255 |
| Fixed-Q CE after cycle 1 | 2.45255 | 2.58033 | 2.83018 |
| Fixed-Q CE after cycle 2 | 2.45255 | 2.43198 | 2.89772 |
| Fixed-Q error after cycle 2 | 45.47% | 49.93% | 52.93% |
| Mean same-domain error change, visit 2 minus visit 1 | — | -2.12 pp | -4.97 pp |
| Selected adaptation samples | — | 43.09% | 43.31% |

Both arms exhibit large transient changes and recovery. These are not monotone
deterioration trajectories. Same-domain return error improves in both arms,
while the lambda=1 fixed-Q CE endpoint is worse than the source and increases
slightly from cycle 1 to cycle 2. CE and error answer different questions and
must both be retained. Changing domains must not be confused with learner drift.

Proxy CE falls in 2,815/2,820 lambda=0 updates and 2,817/2,820 lambda=1 updates.
Among the 60 scheduled local probes per arm, proxy CE falls while current-domain
held-out probe CE rises in 22 (lambda 0) and 17 (lambda 1). These finite-sample
measurements demonstrate proxy/audit disagreement, not population-valid updates.

## Prespecified update forks

All values below are fixed-Q CE differences. `A_closed` and `A_open` compare
accepting with rejecting the candidate under endogenous and replayed targets;
`I_feedback = A_closed - A_open`. Positive means a worse effect of acceptance.

| Parent checkpoint | Immediate effect | A_closed at H94 | A_open at H94 | I_feedback at H94 |
|---|---:|---:|---:|---:|
| After block 5 | -0.04593 | -0.11964 | -0.11654 | -0.00310 |
| After block 15 | +0.04656 | +0.13892 | +0.44335 | -0.30443 |

The block-5 candidate improves Q immediately but is worse than rejection at
H5 and H20. Closed and replay effects are identical at these short endpoints,
so that reversal does **not** identify feedback amplification. At H94 the
candidate is beneficial again. The block-15 candidate is immediately harmful;
at H94 endogenous feedback reduces its relative harm compared with this replay
intervention. Neither fork supports a uniform harmful-feedback explanation.

Both nodes were specified before outcomes. H20 and H94 repeat prefixes of the
same forks and are not independent replications. Reject/replay identity is
explicitly verified over the first 20 steps; longer equality is an implication
of the deterministic recurrence, not an independently executed fourth branch.

## Integrity and scope

- All 60 pilot blocks have checksummed completion markers and raw batch records.
- Each fork retains complete accept/reject states, future schedules, target
  probabilities, labels, masks, predictions and full checkpoints at audit nodes.
- Independent verification reconstructs losses from logits, checks the replay
  tape and sample keys, and recomputes reported contrasts.
- Official dataset MD5 and lossless extraction hashes passed. A clean source
  sanity check used 1,000 adaptation-pool IDs only (20.8% error); confirmation
  outcomes remain untouched.
- Initial preflight hit a uint8-label indexing bug before adaptation. The repair
  and regression tests are on the experiment branch; failed artifacts remain.
- Pilot compute took 7.09/7.25 minutes per arm. Peak VRAM was not instrumented;
  observed running allocations were about 2.8 GB per GPU, not a peak estimate.

No hypercompression, inheritance-loss, Jacobian, phase-transition, independent
replication, benchmark-superiority or general RSI claim is established here.
The natural algorithm is a simplified, explicitly specified CTTA learner, not
an implementation of full CoTTA.

## Next decision

Preserve these mixed and negative findings. The user requested moving the
planned full-cycle fork extension to two Sapelo zhai_p L40S GPUs; transfer and
environment preparation are underway. Verify replay identity and keep host
provenance separate. Do not tune the present runs to manufacture deterioration.

Figures and machine-readable tables are in the accompanying compact results
archive. Large raw artifacts will have an explicit checksummed release index;
until that verification finishes, the remote-only remainder is not claimed to
be safely delivered.
