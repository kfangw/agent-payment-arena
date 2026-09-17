# Replication branch

This branch adds, on top of `main`, the runners, analysis scripts and result
files of the settlement-decision study that `main` does not carry. Everything
here is byte-identical to the compact reproducibility archive that accompanies
the write-up; the archive's `MANIFEST.sha256` verifies each file.

## What is added

```
src/arena/experiments/settlement/security_replay.py
                  main-run replay with delay and expiry counters (the only
                  file of the package that differs from main; the result
                  files record its SHA-256 as module_sha256)
scripts/replication/transfer_cell.py     cross-rail transfer runner
scripts/replication/b6_cell.py           regime-conditional two-threshold rule
                                         on the halting rail (B6)
scripts/replication/adversarial_struct.py
                  random-constant check of the augmented-state interval structure
scripts/replication/analysis/            commitment premium (stage chain and
                  flow integration), arrival-shape check, waiting-set sweep,
                  and the exact model they share (model.py)
scripts/replication/tables/              regenerate the six generated row
                  files and the refined-grid A - B4 rows from results
results_replication/fits/                fitted parameters, seeds, sample
                  sizes and archived mean payoffs of the main run, per cell
results_replication/security/            nine security, delay and expiry files
results_replication/transfer/            six transfer result files
results_replication/b6/                  three B6 result files
results_replication/diagnostics/         e-slow-deep, experiment2, halt-F2,
                  misspec-halt, security-h, wait-sweep
```

Not on this branch: `results/blocks/` of the archive (23 MB of per-cell
block sums behind the bootstrap intervals). The archive carries them; the
runners here rebuild every policy from `results_replication/fits/` and check
the replayed mean payoffs against the archived ones before reporting.

## Running

From the repository root with the package importable (`uv sync`, or
`PYTHONPATH=src`):

```
python3 scripts/replication/transfer_cell.py --help
python3 scripts/replication/b6_cell.py --help
python3 scripts/replication/adversarial_struct.py <draws> <seed>
python3 scripts/replication/analysis/wait_interval_sweep.py [n] [seed]
python3 scripts/replication/analysis/commitment_premium.py
python3 scripts/replication/tables/table_rows.py results_replication <out_dir>
python3 scripts/replication/tables/table_b4_rows.py <blocks_dir>
```

`table_b4_rows.py` and the interval columns of `table_rows.py` need the
block sums, so they run against the archive's `results/` tree, not this
branch alone. The flow integration in `analysis/` needs scipy.
