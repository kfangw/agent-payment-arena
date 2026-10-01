"""Unit tests for experiment G (spec 9.4 information contract, spec 5 estimator edges).

    export PYTHONPATH=$PWD/src && python3 scripts/tests_G.py   (writes results/calibration_pilot/tests_G.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
import run_G as G  # noqa: E402

from arena.experiments.settlement.flows import make_flows  # noqa: E402
from arena.experiments.settlement.outage import draw_outage_batch  # noqa: E402


def _draws():
    w = G.world_env()
    d = draw_outage_batch(w, make_flows()[G.FLOW], 5000, np.random.default_rng([G.LEARN_ROOT, 950]),
                          payments_per_episode=G.PPE)
    return w, G.distort(d, 2.0)


def _expect_fail(fn):
    try:
        fn()
    except AssertionError as e:
        return str(e)
    return None


def test_contract_passes():
    w, d = _draws()
    g = G.CalGuard(d, w.tau)
    c = G.cal_learn(g, 2000, w.tau)
    assert g.read == G.FIELDS_CAL, g.read
    assert g._theta.n_read == c["n_ans"]
    return dict(fields=sorted(g.read), theta_reads=g._theta.n_read, n_ans=c["n_ans"])


def test_unanswered_intent_trips():
    """Deliberately read the intent of an unanswered payment: must raise."""
    w, d = _draws()
    g = G.CalGuard(d, w.tau)
    unans = np.nonzero(np.asarray(d.t_ans) > w.tau)[0]
    assert len(unans) > 0
    msgs = dict(
        one_unanswered=_expect_fail(lambda: g.theta[unans[:1]]),
        scalar_unanswered=_expect_fail(lambda: g.theta[int(unans[0])]),
        all_by_slice=_expect_fail(lambda: g.theta[:100]),
        whole_array=_expect_fail(lambda: np.asarray(g.theta)),
        iterate=_expect_fail(lambda: list(g.theta)),
    )

    def leaky_cal(gg, n, tau):          # a Comp_cal that also uses unanswered labels
        y = np.asarray(gg.theta[np.arange(n)]) == 1
        return y.mean()
    msgs["leaky_estimator"] = _expect_fail(lambda: leaky_cal(g, 2000, w.tau))
    assert all(m is not None for m in msgs.values()), msgs
    return msgs


def test_extra_field_trips():
    """Reading v or payoff inputs changes the field set, so the plan-phase assert fails."""
    w, d = _draws()
    g = G.CalGuard(d, w.tau)
    G.cal_learn(g, 500, w.tau)
    _ = g.v
    msg = _expect_fail(lambda: (_ for _ in ()).throw(AssertionError(f"Comp_cal read {g.read}"))
                       if g.read != G.FIELDS_CAL else None)
    assert msg is not None and "v" in g.read
    return dict(fields=sorted(g.read), tripped=True)


def test_mle_edges():
    s = np.linspace(0.01, 0.5, 50)
    k0, f0 = G.zeta_mle(s, np.zeros(50, bool))
    k1, f1 = G.zeta_mle(s, np.ones(50, bool))
    assert (k0, f0) == (200, "no_misuse") and (k1, f1) == (-200, "all_misuse")
    rng = np.random.default_rng(1)
    pi = rng.beta(1.6, 6.4, 200_000)
    y = rng.random(200_000) < pi
    k, f = G.zeta_mle(np.clip(1.4 * pi, 0, 1), y)
    return dict(no_misuse=[k0, f0], all_misuse=[k1, f1], beta_zeta1_4=[k, f, float(np.exp(0.01 * k))])


if __name__ == "__main__":
    out = {}
    for name in ("test_contract_passes", "test_unanswered_intent_trips", "test_extra_field_trips", "test_mle_edges"):
        try:
            out[name] = dict(result=globals()[name](), pass_=True)
        except Exception as e:  # noqa: BLE001
            out[name] = dict(error=repr(e), pass_=False)
        print(name, "PASS" if out[name]["pass_"] else "FAIL", json.dumps(out[name].get("result", out[name].get("error")))[:300])
    paths.CALIBRATION_PILOT.mkdir(parents=True, exist_ok=True)
    (paths.CALIBRATION_PILOT / "tests_G.json").write_text(json.dumps(out, indent=1, default=str))
    sys.exit(0 if all(v["pass_"] for v in out.values()) else 1)
