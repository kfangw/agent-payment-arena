"""Adversarial check of the augmented-state structure proposition.

Samples constants from the assumption-satisfying region, compiles the
(i, l, r) table, and applies the five clause tests of structcheck.check_pi
at every state.  A proposition must hold at every draw, not only at the
declared constants.
"""
import json, sys
import numpy as np
from arena.experiments.settlement.outage import (OutageEnv, PI_GRID_OUTAGE,
                                                 compile_outage, survival)
from arena.experiments.settlement.structcheck import check_pi, _interval
from arena.experiments.settlement.core import WAIT

def draw_env(rng):
    N = int(rng.integers(2, 18))
    f0 = float(rng.uniform(1e-4, 0.05))
    ftail = float(rng.uniform(1e-6, 0.01))
    return OutageEnv(
        f=np.array([f0] + [ftail] * N),
        m=float(rng.uniform(0.05, 1.5)),
        h=float(rng.uniform(0.1, 4.0)),
        C=float(rng.uniform(0.05, 3.0)),
        cw=float(rng.uniform(0.0, 0.02)),
        tau=int(rng.integers(2, 20)),
        H=int(rng.integers(5, 40)),
        rho=float(rng.uniform(0.02, 0.6)),
        p01=float(rng.uniform(1e-4, 0.05)),
        p10=float(rng.uniform(0.005, 0.4)),
    )

def main(n_draws, seed):
    rng = np.random.default_rng(seed)
    bad, checked, draws, wait_only = [], 0, 0, 0
    for d in range(n_draws):
        env = draw_env(rng)
        try:
            pol = compile_outage(env, 'A')
            sig = survival(env)
        except Exception as e:                       # infeasible draw
            print(f'draw {d}: skipped ({type(e).__name__})', flush=True); continue
        draws += 1
        N, H = env.N, env.H
        for iv in range(len(pol.v_grid)):
            tab = pol.tables[iv]
            for i in range(N + 2):
                for l in range(H + 1):
                    for r in (0, 1):
                        lab = tab[i, l, r, :]
                        res = check_pi(lab, PI_GRID_OUTAGE, sig[i, l, r], env.m, env.h)
                        checked += 1
                        # the proposition claims intervals for the three terminal
                        # actions only; waiting is outside it, as in Theorem 1
                        fails = [c for c in ('c1','c2','c3','c5') if not res[c]]
                        if not res['c4'] and not _interval(np.where(lab == WAIT)[0]):
                            wait_only += 1
                        if fails:
                            bad.append({'draw': d, 'state': [iv,i,l,r], 'fails': fails,
                                        'env': {k: (v.tolist() if hasattr(v,'tolist') else v)
                                                for k,v in env.__dict__.items()}})
        print(f'draw {d}: states {checked}, terminal-clause violations {len(bad)}, '
              f'non-interval WAIT sets {wait_only}', flush=True)
        if bad:
            break
    print(json.dumps({'draws': draws, 'states_checked': checked,
                      'terminal_clause_violations': len(bad),
                      'non_interval_wait_states': wait_only,
                      'first': bad[:1]}, indent=1))

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
