"""Vectorized replay for experiment G, written to reproduce arena's replay_outage
(outage.py, 705574b) bit for bit.

replay_outage walks one payment at a time in Python.  This module runs the same state
machine on all payments at once with numpy: every payment keeps its own (stage i, tick t,
payoff), and each payoff receives the same float64 operations in the same order as in
replay_outage (payoff -= C, payoff -= cw*v per waited tick, payoff += v*mult at a
grant), so the per-payment results are identical, not just close.  run_G.py checks this
against replay_outage (max |diff| must be exactly 0) before it trusts the fast path.

Policies:
  table policy : a CompiledOutagePolicy (label tables per v grid point) read at the
                 score the policy sees (pi_seen).  Same lookup as CompiledOutagePolicy.__call__
                 (nearest log v grid point, first on ties; pi bin = round(pi * 500), half to even).
  forced policy: FixedActionOutageWatchPolicy(k, action): WAIT while elapsed < k, the
                 stage is not final and ticks remain, then the fixed action.
"""
from __future__ import annotations

import numpy as np

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY
from arena.experiments.settlement.outage import PI_GRID_OUTAGE, WAIT


def v_index(v_grid, v, chunk=500_000):
    """Nearest v grid point in log space, as CompiledOutagePolicy.__call__ (argmin, first on ties)."""
    lg = np.log(np.asarray(v_grid, dtype=float))
    v = np.asarray(v, dtype=float)
    out = np.empty(len(v), dtype=np.int64)
    for s in range(0, len(v), chunk):
        lv = np.log(np.maximum(v[s:s + chunk], 1e-12))
        out[s:s + chunk] = np.argmin(np.abs(lg[None, :] - lv[:, None]), axis=1)
    return out


def v_index_scalar(v_grid, v):
    """The scalar formula of CompiledOutagePolicy.__call__, for spot checks."""
    return np.array([int(np.argmin(np.abs(np.log(v_grid) - np.log(max(float(x), 1e-12))))) for x in v])


def pi_bin(pi):
    """round(pi * 500): numpy rounds half to even, as Python's round does."""
    return np.round(np.asarray(pi, dtype=float) * (len(PI_GRID_OUTAGE) - 1)).astype(np.int64)


def stack_tables(pol):
    return np.stack(pol.tables).astype(np.int8)          # (n_v, N+2, H+1, 2, 501)


def _settle(env, d, idx, i, t):
    """_settle_from vectorized: 1 settled, 0 failed, from stage i at tick t."""
    N, H = env.N, env.H
    f = np.asarray(env.f, dtype=float)
    i = i.copy()
    t = t.copy()
    res = np.full(len(idx), -1, dtype=np.int64)
    live = np.ones(len(idx), dtype=bool)
    while live.any():
        a = np.nonzero(live)[0]
        ia, ta = i[a], t[a]
        done1 = ia > N
        res[a[done1]] = 1
        live[a[done1]] = False
        rest = ~done1
        a, ia, ta = a[rest], ia[rest], ta[rest]
        done0 = (H - ta) <= 0
        res[a[done0]] = 0
        live[a[done0]] = False
        rest = ~done0
        a, ia, ta = a[rest], ia[rest], ta[rest]
        k = idx[a]
        up = d.paths[k, ta] == 0
        fail = np.zeros(len(a), dtype=bool)
        fail[up] = d.u_stage[k[up], ia[up]] < f[ia[up]]
        res[a[fail]] = 0
        live[a[fail]] = False
        adv = up & ~fail
        i[a[adv]] += 1
        t[a[~fail]] += 1
    return res


def replay_vec(env, d, action_fn):
    """Replay every payment of d.  action_fn(idx, i, l, r) -> action array for payments idx
    at stage i (FIN = N+1), remaining ticks l >= 0, regime r.  Returns per-payment payoffs."""
    n = len(d)
    N, H, FIN = env.N, env.H, env.N + 1
    f = np.asarray(env.f, dtype=float)
    m, h, C, cw, tau = float(env.m), float(env.h), float(env.C), float(env.cw), int(env.tau)
    v_all = np.asarray(d.v, dtype=float)
    theta = np.asarray(d.theta)
    out = np.zeros(n)
    i = np.zeros(n, dtype=np.int64)
    t = np.zeros(n, dtype=np.int64)
    live = np.ones(n, dtype=bool)
    verify_idx, verify_i, verify_t = [], [], []
    ex_pos = _EX["ex"]
    while live.any():
        k = np.nonzero(live)[0]
        ik = i[k]
        ik[ik > N] = FIN
        i[k] = ik
        tk = t[k]
        if tk.max() > H:
            raise IndexError("tick beyond the regime path (replay_outage would raise too)")
        l = H - tk
        r = d.paths[k, tk].astype(np.int64)
        expire = (l <= 0) & (ik < FIN)
        live[k[expire]] = False
        keep = ~expire
        k, ik, tk, l, r = k[keep], ik[keep], tk[keep], l[keep], r[keep]
        a = np.asarray(action_fn(k, ik, np.maximum(l, 0), r))
        # GRANT
        g = a == GRANT
        if g.any():
            kg = k[g]
            won = np.where(ik[g] == FIN, 1, _settle(env, d, kg, ik[g], tk[g]))
            mult = np.where(won == 1, np.where(theta[kg] == 0, m, -h), -1.0)
            out[kg] += v_all[kg] * mult
            live[kg] = False
        # REJECT and unknown actions end the payment
        other = ~np.isin(a, (GRANT, VERIFY, WAIT))
        live[k[other]] = False
        # VERIFY: handled after the main loop (the payment ends inside the window)
        vv = a == VERIFY
        if vv.any():
            kv = k[vv]
            out[kv] -= C
            verify_idx.append(kv)
            verify_i.append(ik[vv])
            verify_t.append(tk[vv])
            live[kv] = False
        # WAIT
        w = a == WAIT
        if w.any():
            kw, iw, tw, rw = k[w], ik[w], tk[w], r[w]
            out[kw] -= cw * v_all[kw]
            up = (rw == 0) & (iw <= N)
            fail = np.zeros(len(kw), dtype=bool)
            fail[up] = d.u_stage[kw[up], iw[up]] < f[iw[up]]
            live[kw[fail]] = False
            i[kw[up & ~fail]] += 1
            t[kw[~fail]] += 1
    if verify_idx:
        _verify(env, d, out, np.concatenate(verify_idx), np.concatenate(verify_i),
                np.concatenate(verify_t), ex_pos, v_all, theta, f, m, cw, tau)
    return out


def _verify(env, d, out, kv, iv, tv, ex_pos, v_all, theta, f, m, cw, tau):
    """The verify window of replay_outage, vectorized (payoff -= C already applied)."""
    N, H, FIN = env.N, env.H, env.N + 1
    i, t = iv.copy(), tv.copy()
    w = np.minimum(tau, np.maximum(H - t, 0))
    s = np.zeros(len(kv), dtype=np.int64)
    live = s < w
    while live.any():
        a = np.nonzero(live)[0]
        k = kv[a]
        out[k] -= cw * v_all[k]
        s[a] += 1
        ans = d.t_ans[k] == s[a]
        if ans.any():
            aa, ka = a[ans], k[ans]
            li = H - t[aa]
            lr = d.paths[ka, t[aa]].astype(np.int64)
            ok = (theta[ka] == 0) & (ex_pos[i[aa], li, lr] > 0)
            if ok.any():
                ko, ao = ka[ok], aa[ok]
                won = np.where(i[ao] == FIN, 1, _settle(env, d, ko, i[ao], t[ao]))
                out[ko] += v_all[ko] * np.where(won == 1, m, -1.0)
            live[aa] = False
        na = ~ans
        a, k = a[na], k[na]
        up = (d.paths[k, t[a]] == 0) & (i[a] <= N)
        fail = np.zeros(len(a), dtype=bool)
        fail[up] = d.u_stage[k[up], i[a][up]] < f[i[a][up]]
        live[a[fail]] = False
        adv = a[up & ~fail]
        i[adv] += 1
        i[adv[i[adv] > N]] = FIN
        a2 = a[~fail]
        t[a2] += 1
        exp = (H - t[a2]) <= 0
        live[a2[exp]] = False
        a3 = a2[~exp]
        live[a3[s[a3] >= w[a3]]] = False


_EX = {}


def set_exercise(ex_pos):
    _EX["ex"] = ex_pos


def table_action_fn(T, iv, ip):
    """Action function of a compiled policy: T stacked tables, iv v index per payment,
    ip pi bin per payment (from the score the policy sees)."""
    def fn(k, i, l, r):
        return T[iv[k], i, l, r, ip[k]]
    return fn


def forced_action_fn(env, horizon, action):
    H, FIN = env.H, env.N + 1

    def fn(k, i, l, r):
        elapsed = H - l
        wait = (elapsed < horizon) & (i < FIN) & (l > 0)
        return np.where(wait, WAIT, action)
    return fn


def replay_table(env, d, T, iv, pi_seen, ex_pos):
    set_exercise(ex_pos)
    return replay_vec(env, d, table_action_fn(T, iv, pi_bin(pi_seen)))


def replay_forced(env, d, horizon, action, ex_pos):
    set_exercise(ex_pos)
    return replay_vec(env, d, forced_action_fn(env, horizon, action))
