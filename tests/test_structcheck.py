"""Detection regression test for arena.experiments.settlement.structcheck.

The chain contrast only shows the checks do not false-positive on a table
that already holds the property.  These tests inject each of the five fault
kinds into a valid suspicion row and confirm the corresponding check turns
False, so a real violation cannot pass silently.  Labels are built directly;
no dynamic program is solved.
"""

from __future__ import annotations

import numpy as np

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY
from arena.experiments.settlement.structcheck import check_pi

NPI = 501
PI = np.linspace(0.0, 1.0, NPI)
M, H = 0.35, 1.0
SIGMA = 1.0  # pi_hat = m/(m+h) = 0.2593, index ~130


def _valid_verify_row():
    """grant[0..99] | verify[100..129] | reject[130..500] — no direct
    grant-reject adjacency, so c5 does not fire."""
    lab = np.full(NPI, REJECT, dtype=np.int8)
    lab[:100] = GRANT
    lab[100:130] = VERIFY
    return lab


def _valid_adjacent_row():
    """grant[0..129] | reject[130..500] — boundary at pi_hat, c5 fires."""
    lab = np.full(NPI, REJECT, dtype=np.int8)
    lab[:130] = GRANT
    return lab


def _check(lab):
    return check_pi(lab, PI, SIGMA, M, H)


def test_valid_rows_pass_all():
    for lab in (_valid_verify_row(), _valid_adjacent_row()):
        r = _check(lab)
        assert all(r[c] for c in ("c1", "c2", "c3", "c4", "c5")), r


def test_floating_grant_flags_c1_or_c4():
    lab = _valid_verify_row()
    lab[400] = GRANT  # a grant stranded inside reject
    r = _check(lab)
    assert not (r["c1"] and r["c4"])


def test_endpoint_verify_flags_c3():
    lab = _valid_verify_row()
    lab[NPI - 1] = VERIFY  # verify touches pi = 1
    assert not _check(lab)["c3"]


def test_split_grant_flags_c4():
    lab = _valid_verify_row()
    lab[50] = REJECT  # split the grant interval in two
    assert not _check(lab)["c4"]


def test_empty_reject_flags_c2():
    lab = _valid_verify_row()
    lab[lab == REJECT] = GRANT  # no reject region left
    assert not _check(lab)["c2"]


def test_shifted_boundary_flags_c5():
    lab = np.full(NPI, REJECT, dtype=np.int8)
    lab[:150] = GRANT  # boundary ~0.30, pi_hat ~0.26: 20 steps off
    assert not _check(lab)["c5"]
