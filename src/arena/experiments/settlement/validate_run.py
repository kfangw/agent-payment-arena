"""Acceptance checks for the run refactor: block sums reconstruct the
mean, and the output is a write-once envelope.

Run:  python -m arena.experiments.settlement.validate_run
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .run import CHAIN_BLOCK, block_stats, main


def block_sums_reconstruct_mean() -> None:
    """sum(block_sums) / sum(block_counts) equals the flat mean."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(9_500)
    episodes = np.arange(len(arr)) // CHAIN_BLOCK
    sums, counts = block_stats(arr, episodes)
    assert counts.sum() == len(arr)
    assert abs(sums.sum() / counts.sum() - arr.mean()) < 1e-9
    # last block is a short remainder, not 1000
    assert counts[-1] == len(arr) % CHAIN_BLOCK
    print(f"block sums ok: {len(counts)} episodes reconstruct the mean")


def run_writes_envelope() -> None:
    """A small chain cell writes one envelope with block sums, and a
    second write to the same path is refused."""
    with tempfile.TemporaryDirectory() as d:
        argv = [
            "--env",
            "E-fast",
            "--flow",
            "F1",
            "--cw",
            "mid",
            "--n-eval",
            "4000",
            "--n-tune",
            "2000",
            "--seed",
            "1",
            "--out",
            d,
        ]
        main(argv)
        path = next(Path(d).glob("settlement_*.json"))
        env = json.loads(path.read_text())
        for key in ("kind", "cell", "params_hash", "code", "created_utc", "payload"):
            assert key in env, f"missing {key}"
        assert env["kind"] == "settlement"
        pol = env["payload"]["policies"]
        assert "A" in pol and "B1" in pol
        counts = env["payload"]["block_counts"]
        sums = pol["A"]["block_sums"]
        assert len(counts) == len(sums)
        recon = sum(sums) / sum(counts)
        assert abs(recon - env["payload"]["means"]["A"]) < 1e-9
        try:
            main(argv)
        except FileExistsError:
            print("run ok: envelope written, overwrite refused, sums reconstruct")
            return
        raise AssertionError("second run should have refused to overwrite")


def main_checks() -> None:
    """Run every check; raise on the first failure."""
    block_sums_reconstruct_mean()
    run_writes_envelope()
    print("\nrun refactor: all checks passed")


if __name__ == "__main__":
    main_checks()
