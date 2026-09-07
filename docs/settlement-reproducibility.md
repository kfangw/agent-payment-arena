# Settlement experiment reproducibility guide

This repository contains the code, locked Python environment, declared seeds,
and artifact checks needed to reproduce the settlement experiment results. Python 3.12 and
`uv` are required. The simulations do not call external services or require
API keys.

## Environment

```bash
git clone https://github.com/kfangw/agent-payment-arena.git
cd agent-payment-arena
git checkout feat/settlement-checks
UV_CACHE_DIR=/tmp/agent-payment-arena-uv uv sync --frozen --dev
UV_CACHE_DIR=/tmp/agent-payment-arena-uv uv run pytest
```

`uv.lock` fixes every transitive dependency. Result envelopes record the Git
revision, resolved-parameter hash, seed, tuning size, evaluation size, and UTC
creation time. Output files are write-once to prevent accidental replacement.

## Fast integrity run

The smoke profile follows the complete core path at a reduced sample size. It
runs all nine base cells and the three B5 outage cells.

```bash
UV_CACHE_DIR=/tmp/agent-payment-arena-uv \
  uv run python scripts/run_settlement_cells.py --profile smoke --workers 3
```

## Confirmatory run

The full profile uses 200,000 tuning draws and 5,663,400 evaluation draws per
cell. Seeds 1 through 9 follow `CONFIRMATORY_RUNS` in
`src/arena/experiments/settlement/design.py`. B5 uses seeds 7, 8, and 9 and a
161-point suspicion grid.

```bash
UV_CACHE_DIR=/tmp/agent-payment-arena-uv \
  uv run python scripts/run_settlement_cells.py --profile full --workers 3
```

The launcher is resumable. It skips an output only when the expected file
already exists. Delete or move an output deliberately before repeating that
cell because individual runners refuse to overwrite results.

The supplementary analyses retain separate launchers:

| Analysis | Command | Output |
| --- | --- | --- |
| B3 grid and refined B4 | `uv run python scripts/run_s1.py` | `results/` |
| Oracle B4 bound | `uv run python scripts/run_s2.py` | `results/` |
| Injection axes | `uv run python scripts/run_s3.py` | `results_inject/` |
| Deeper settlement | `uv run python scripts/run_s4.py` | `results/` |
| Residual diagnosis | `uv run python scripts/run_r4.py` | `results_r4/` |
| Reduced S2 to S4 set | `uv run python scripts/run_rest.py` | several result directories |
| Parameter and shape checks | `uv run python scripts/run_t567.py` | `results_t5/` to `results_t7/` |

## Preserving result data

Generated data are excluded from Git because the full JSON set is large. The
archive command stores any number of result directories under explicit labels,
writes a SHA-256 digest for every file into `MANIFEST.json`, normalizes archive
metadata, and verifies the archive immediately.

```bash
uv run python scripts/package_artifacts.py pack \
  --input base=results \
  --input injections=results_inject \
  --input residual=results_r4 \
  --input b5=results_settlement/b5 \
  --output artifacts/settlement-results.tar.gz

uv run python scripts/package_artifacts.py verify \
  artifacts/settlement-results.tar.gz
```

Publish the archive as a GitHub Release asset or in a DOI-bearing repository.
Do not commit the large archive to the ordinary Git history. Preserve the
archive's companion `.metadata.json` file with the release; it records the
archive digest, input labels, file count, and uncompressed byte count. It does
not expose source-machine paths.

## B5 definition

B5 observes the public settlement regime at arrival. It rejects in the halted
regime and otherwise applies B3 thresholds tuned on the independent tuning
split. The B5 runner regenerates A, refined B3, and B5 from the declared seed.
It therefore has no absolute paths or dependency on previously generated JSON.
The output stores episode block sums, shared block counts, confidence intervals,
the sign-flip test, and the equivalence margin needed to audit the reported value.
