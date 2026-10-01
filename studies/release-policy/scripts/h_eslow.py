"""Experiment H: hazard-input sensitivity on the slow rail E-slow x F2 (c_w mid, seed 5).

    PYTHONPATH=src python3 scripts/h_eslow.py repro   # step 1 (from the package root)
    PYTHONPATH=src python3 scripts/h_eslow.py spec    # step 2 (DP tables only, no payoff)
    PYTHONPATH=src python3 scripts/h_eslow.py run     # step 3 (multipliers from MULTS)
    PYTHONPATH=src python3 scripts/h_eslow.py analyze # step 4 (summary.json)

World, draws and calibration follow run.run_chain exactly: rng = default_rng(5);
tuning batch (200,000) then evaluation batch (5,663,400) by simulate.draw_batch;
q_hat = share of tuning answers within tau; rho_hat = rho_hat_from_q(q_hat, tau);
policy "A" = policies.compile_A(ch, "A", rho=rho_hat).  Episode blocks are
run.CHAIN_BLOCK = 1000 payments (5,664 blocks; the last holds 400).

The perturbed compiler input is f_hat = clip(k * f, 0, 1) for all stages at once;
q_hat, m, h, C, c_w stay nominal.  Payoff always comes from the TRUE channel, and
the window exercise rule ex_unit is the true-world one (same convention as the
E-fitted injection queue, which passes the world's ex to replay_outage).

The replay here is a vectorised re-implementation of simulate.replay with the same
float operation order; step 1 checks it bit-for-bit against the archived block
sums and against simulate.replay on a prefix.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY, WAIT, rho_hat_from_q, sigma_list
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.policies import PI_GRID, CompiledPolicy, compile_A
from arena.experiments.settlement.simulate import draw_batch, replay
from paths import ARCHIVED_BLOCKS, H_ESLOW

OUT = H_ESLOW
NPZ = OUT / "npz"
ARCH = ARCHIVED_BLOCKS / "blocks_E-slow_F2_mid_s5.npz"
SEED, N_TUNE, N_EVAL = 5, 200_000, 5_663_400
CHAIN_BLOCK = 1000             # = run.CHAIN_BLOCK (run.py cannot be imported under Python 3.11)
DEN = 49.5                      # bp denominator, same convention as analyze.py
NB, BOOT_SEED = 2_000, 20260929
# Fixed before any perturbed payoff was computed.
PLAN_MULTS = [0.8, 0.9, 1.1, 1.2]
EXTRA_MULTS = [0.5, 0.3, 0.1, 2.0]
MULTS = PLAN_MULTS + EXTRA_MULTS
ACT = ["grant", "reject", "verify", "wait"]


# ------------------------------------------------------------------ world
def world():
    kind, ch, rho = envs_for("mid")["E-slow"]
    assert kind == "chain"
    return ch, rho


def load():
    t0 = time.time()
    ch, _ = world()
    fl = make_flows()["F2"]
    rng = np.random.default_rng(SEED)
    tune_d = draw_batch(ch, fl, N_TUNE, rng)
    q_hat = float((tune_d.t_ans <= ch.tau).mean())
    del tune_d
    d = draw_batch(ch, fl, N_EVAL, rng)
    d.p_true = None
    rho_hat = rho_hat_from_q(q_hat, ch.tau)
    sig = sigma_list(ch.f)
    ex_unit = np.maximum(sig * (1 + ch.m) - 1.0, 0.0)
    return dict(ch=ch, d=d, q_hat=q_hat, rho_hat=rho_hat, ex=ex_unit, load_s=time.time() - t0)


def compile_k(ch, rho_hat, k):
    f_hat = np.clip(k * ch.f, 0.0, 1.0)
    return compile_A(replace(ch, f=f_hat), f"A_k{k:g}", rho=rho_hat), f_hat


# ------------------------------------------------------------------ lookup + replay
def lookup_idx(pol: CompiledPolicy, d, chunk=500_000):
    """(iv, ip) per payment, same rule as CompiledPolicy.__call__."""
    lv = np.log(pol.v_grid)
    iv = np.empty(len(d), dtype=np.int64)
    for s in range(0, len(d), chunk):
        sl = slice(s, s + chunk)
        iv[sl] = np.argmin(np.abs(lv[None, :] - np.log(np.maximum(d.v[sl], 1e-12))[:, None]), axis=1)
    ip = np.round(d.pi0 * (len(PI_GRID) - 1)).astype(np.int64)
    return iv, ip


def vreplay(ch, d, rows, ex):
    """Vectorised simulate.replay. rows[k, stage] = action of payment k at stage."""
    n = len(d)
    N, FIN = ch.N, ch.N + 1
    v, th, fa, ta = d.v, d.theta, d.fail_at, d.t_ans
    pay = np.zeros(n)

    def leg(ids, stage):
        good = np.where(th[ids] == 0, v[ids] * ch.m, v[ids] * (-ch.h))
        return np.where((fa[ids] >= stage) & (fa[ids] <= N), -v[ids], good)

    alive = np.arange(n)
    for s in range(FIN + 1):
        if len(alive) == 0:
            break
        a = rows[alive, s].copy()
        if s == FIN:
            a[a == WAIT] = REJECT
        g = alive[a == GRANT]
        pay[g] += leg(g, s)
        vv = alive[a == VERIFY]
        if len(vv):
            pay[vv] -= ch.C
            ids, j = vv, s
            for t in range(1, ch.tau + 1):
                if len(ids) == 0:
                    break
                pay[ids] -= ch.cw * v[ids]
                ans = ta[ids] == t
                hit = ids[ans & (th[ids] == 0)]
                if ex[min(j, FIN)] > 0.0 and len(hit):
                    pay[hit] += leg(hit, j)
                ids = ids[~ans]
                if j <= N:
                    ids = ids[fa[ids] != j]
                j = min(j + 1, FIN)
        w = alive[a == WAIT]
        pay[w] -= ch.cw * v[w]
        alive = w[fa[w] != s]
    return pay


def block_sums(pay):
    n = len(pay)
    starts = range(0, n, CHAIN_BLOCK)
    sums = np.array([float(pay[i:i + CHAIN_BLOCK].sum()) for i in starts])
    cnts = np.array([min(CHAIN_BLOCK, n - i) for i in starts], dtype=float)
    return sums, cnts


def eval_policy(c, pol, iv, ip):
    rows = pol.tables[iv, :, ip]                      # (n, N+2) int8
    t0 = time.time()
    pay = vreplay(c["ch"], c["d"], rows, c["ex"])
    return pay, rows[:, 0].copy(), time.time() - t0


# ------------------------------------------------------------------ steps
def step_repro():
    c = load()
    ch, d = c["ch"], c["d"]
    arch = np.load(ARCH, allow_pickle=True)
    pol = compile_A(ch, "A", rho=c["rho_hat"])
    iv, ip = lookup_idx(pol, d)
    pay, first, rs = eval_policy(c, pol, iv, ip)
    sums, cnts = block_sums(pay)
    a2 = arch["A2"]
    # cross-check the vectorised replay against simulate.replay on a prefix
    npre = 20_000
    sub = replace(d, v=d.v[:npre], theta=d.theta[:npre], pi0=d.pi0[:npre],
                  fail_at=d.fail_at[:npre], t_ans=d.t_ans[:npre])
    ref = replay(ch, sub, pol, c["ex"])
    res = dict(
        q_hat=c["q_hat"], rho_hat=c["rho_hat"], load_s=c["load_s"], replay_s=rs,
        n_blocks=len(sums), n_blocks_arch=len(a2),
        counts_equal=bool(np.array_equal(cnts, arch["block_counts"])),
        blocks_bit_equal=bool(np.array_equal(sums, a2)),
        max_abs_block_diff=float(np.max(np.abs(sums - a2))) if len(sums) == len(a2) else None,
        mean_payoff=float(pay.mean()), mean_payoff_arch=float(a2.sum() / arch["block_counts"].sum()),
        mean_exposure=float(d.v.mean()),
        prefix_loop_vs_vector_equal=bool(np.array_equal(ref, pay[:npre])),
        prefix_n=npre,
    )
    print(json.dumps(res, indent=1))
    NPZ.mkdir(parents=True, exist_ok=True)
    (OUT / "repro.json").write_text(json.dumps(res, indent=1))
    # Per-payment payoffs equal simulate.replay; archived block sums may differ from
    # ours by float summation order only (checked: <= 1 ulp per block).
    res["blocks_equal_within_1e-14_rel"] = bool(np.allclose(sums, a2, rtol=1e-14, atol=0))
    res["n_blocks_not_bit_equal"] = int((sums != a2).sum())
    res["max_ulp_diff"] = int(np.max(np.abs(sums.view(np.int64) - a2.view(np.int64))))
    print(json.dumps(res, indent=1))
    (OUT / "repro.json").write_text(json.dumps(res, indent=1))
    if res["blocks_equal_within_1e-14_rel"] and res["counts_equal"]:
        np.savez(NPZ / "k1.npz", block_sums=sums, block_counts=cnts, first=first,
                 meta=json.dumps(dict(k=1.0, replay_s=rs, q_hat=c["q_hat"], rho_hat=c["rho_hat"])))


def step_spec():
    """DP tables only: f/c_w, and table changes per multiplier. No payoff."""
    ch, _ = world()
    # rho_hat from the same tuning draw (payoff-free)
    fl = make_flows()["F2"]
    rng = np.random.default_rng(SEED)
    tune_d = draw_batch(ch, fl, N_TUNE, rng)
    q_hat = float((tune_d.t_ans <= ch.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, ch.tau)
    base = compile_A(ch, "A", rho=rho_hat).tables
    # arrival weights for the table cells: F2 tuning draw (not the evaluation set)
    pol0 = compile_A(ch, "A", rho=rho_hat)
    iv, ip = lookup_idx(pol0, tune_d)
    out = dict(f=ch.f.tolist(), cw=ch.cw, f_over_cw=(ch.f / ch.cw).tolist(),
               k_cross=(ch.cw / ch.f).tolist(), sigma=sigma_list(ch.f).tolist(),
               m=ch.m, h=ch.h, C=ch.C, tau=ch.tau, q_hat=q_hat, rho_hat=rho_hat,
               nominal_action_share_by_stage={}, mults={})
    for s in range(base.shape[1]):
        out["nominal_action_share_by_stage"][s] = [float((base[:, s, :] == a).mean()) for a in range(4)]
    cand = sorted(set(MULTS + [0.45, 0.4, 0.2, 0.05, 0.02, 1.5, 3.0, 5.0]))
    for k in cand:
        pol, f_hat = compile_k(ch, rho_hat, k)
        T = pol.tables
        ch_mask = T != base
        trans = {}
        for s in range(T.shape[1]):
            for a in range(4):
                for b in range(4):
                    nab = int(((base[:, s, :] == a) & (T[:, s, :] == b)).sum())
                    if a != b and nab:
                        trans[f"s{s}:{ACT[a]}->{ACT[b]}"] = nab
        out["mults"][f"{k:g}"] = dict(
            f_hat=f_hat.tolist(), stages_below_cw=[int(i) for i in np.where(f_hat < ch.cw)[0]],
            table_changed_share=float(ch_mask.mean()),
            table_changed_share_by_stage=[float(ch_mask[:, s, :].mean()) for s in range(T.shape[1])],
            first_decision_changed_share_tune=float((T[iv, 0, ip] != base[iv, 0, ip]).mean()),
            transitions=trans)
    (OUT / "spec_tables.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "transitions"}
                      for k, v in out["mults"].items()}, indent=1))
    print("f/cw", [round(x, 3) for x in out["f_over_cw"]])
    for k, v in out["mults"].items():
        top = sorted(v["transitions"].items(), key=lambda kv: -kv[1])[:8]
        print(k, top)


def step_run():
    c = load()
    ch, d = c["ch"], c["d"]
    base = np.load(NPZ / "k1.npz")
    pol0 = compile_A(ch, "A", rho=c["rho_hat"])
    iv, ip = lookup_idx(pol0, d)
    first0 = base["first"]
    pay0 = None
    for k in MULTS:
        path = NPZ / f"k{k:g}.npz"
        if path.exists():
            print("skip", k)
            continue
        t0 = time.time()
        pol, f_hat = compile_k(ch, c["rho_hat"], k)
        tc = time.time() - t0
        pay, first, rs = eval_policy(c, pol, iv, ip)
        if pay0 is None:
            pay0, _, _ = eval_policy(c, pol0, iv, ip)
        sums, cnts = block_sums(pay)
        meta = dict(k=k, f_hat=f_hat.tolist(), compile_s=tc, replay_s=rs,
                    table_changed_share=float((pol.tables != pol0.tables).mean()),
                    first_changed_share=float((first != first0).mean()),
                    first_changed_count=int((first != first0).sum()),
                    payoff_changed_share=float((pay != pay0).mean()),
                    first_action_counts=np.bincount(first, minlength=4).tolist(),
                    first_action_counts_nominal=np.bincount(first0, minlength=4).tolist())
        np.savez(path, block_sums=sums, block_counts=cnts, first=first, meta=json.dumps(meta))
        print(k, json.dumps({kk: vv for kk, vv in meta.items() if kk != "f_hat"}), flush=True)


def step_analyze():
    arch = np.load(ARCH, allow_pickle=True)
    c = arch["block_counts"].astype(float)
    csum = c.sum()
    nom = np.load(NPZ / "k1.npz")["block_sums"]
    assert np.allclose(nom, arch["A2"], rtol=1e-14, atol=0)
    ks = [k for k in MULTS if (NPZ / f"k{k:g}.npz").exists()]
    runs = {k: np.load(NPZ / f"k{k:g}.npz") for k in ks}
    series, labels = [], []
    for k in ks:
        bs = runs[k]["block_sums"]
        series += [nom - bs, bs - arch["B4_refined"], bs - arch["B4_declared"]]
        labels += [(k, "L"), (k, "D_ref"), (k, "D_dec")]
    series += [nom - arch["B4_refined"], nom - arch["B4_declared"]]
    labels += [(1.0, "D_ref"), (1.0, "D_dec")]
    S = np.vstack(series)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(c)
    M = np.column_stack([S.T, c])
    stats = np.empty((NB, S.shape[0]))
    for j0 in range(0, NB, 20):
        m = min(20, NB - j0)
        idx = rng.integers(0, n, (m, n))
        off = (np.arange(m) * n)[:, None]
        W = np.bincount((idx + off).ravel(), minlength=m * n).reshape(m, n).astype(float)
        P = W @ M
        stats[j0:j0 + m] = P[:, :-1] / P[:, -1:]
    lo = np.percentile(stats, 2.5, axis=0) / DEN * 1e4
    hi = np.percentile(stats, 97.5, axis=0) / DEN * 1e4
    pt = S.sum(axis=1) / csum / DEN * 1e4
    res = {}
    for (k, kind), p, l, h in zip(labels, pt, lo, hi):
        res.setdefault(f"{k:g}", {})[kind] = dict(bp=float(p), lo=float(l), hi=float(h))
    for k in ks:
        meta = json.loads(str(runs[k]["meta"]))
        res[f"{k:g}"].update({kk: meta[kk] for kk in ("table_changed_share", "first_changed_share",
                              "first_changed_count", "payoff_changed_share", "first_action_counts",
                              "first_action_counts_nominal", "f_hat", "replay_s", "compile_s")})
    summary = dict(cell="E-slow x F2", cw="mid", seed=SEED, n_eval=N_EVAL, n_tune=N_TUNE,
                   block=CHAIN_BLOCK, n_blocks=int(n), bp_denominator=DEN,
                   mean_exposure_cell=float(json.loads(str(arch["meta"]))["mean_exposure"]),
                   bootstrap=dict(resamples=NB, seed=BOOT_SEED, kind="paired block, percentile, ratio of sums"),
                   repro=json.loads((OUT / "repro.json").read_text()),
                   multipliers_plan=PLAN_MULTS, multipliers_extra=EXTRA_MULTS, results=res,
                   note="L = J(nominal) - J(k); D = J(k) - J(B4) with archived B4 block sums "
                        "(B4_refined, B4_declared) from blocks_E-slow_F2_mid_s5.npz; k=1 D is nominal A2 - B4.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'k':>5} {'L bp':>8} {'L 95%':>20} {'D_ref':>8} {'D_ref 95%':>20} {'D_dec':>8} "
          f"{'tab%':>7} {'first%':>7} {'pay%':>7}")
    for k in ["1"] + [f"{k:g}" for k in ks]:
        r = res[k]
        L = r.get("L", dict(bp=0, lo=0, hi=0))
        print(f"{k:>5} {L['bp']:8.3f} [{L['lo']:8.3f},{L['hi']:8.3f}] {r['D_ref']['bp']:8.3f} "
              f"[{r['D_ref']['lo']:8.3f},{r['D_ref']['hi']:8.3f}] {r['D_dec']['bp']:8.3f} "
              f"{100*r.get('table_changed_share', 0):7.3f} {100*r.get('first_changed_share', 0):7.3f} "
              f"{100*r.get('payoff_changed_share', 0):7.3f}")


if __name__ == "__main__":
    t = time.time()
    {"repro": step_repro, "spec": step_spec, "run": step_run, "analyze": step_analyze}[sys.argv[1]]()
    print(f"wall {time.time() - t:.1f}s")
