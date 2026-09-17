#!/usr/bin/env python3
"""Regenerate the six generated row files of the manuscript from the archive:
table_decomp_rows.tex (Table S5) from the archived means of the main run,
table_exp2_rows.tex (Table S8) from results/diagnostics/experiment2/,
table_b5_rows.tex (Table S12, upper block) from the B5 diagnostic files,
table_b6_rows.tex (Table S12, lower block) from results/b6/,
table_delay_rows.tex (Table S16) from results/security/,
table_transfer_rows.tex (Table S15) from results/transfer/.
Basis points use the declared mean exposure of $49.50 where the table does.

Run from code/:  python3 table_rows.py ../results <out_dir>
"""
import json, sys, os
from arena.experiments.settlement.naming import canon_keys

CELLS = [('E-fast','F1',1),('E-fast','F2',2),('E-fast','F3',3),('E-slow','F1',4),('E-slow','F2',5),
         ('E-slow','F3',6),('E-outage','F1',7),('E-outage','F2',8),('E-outage','F3',9)]
D = 49.50

def margin_place(lo, hi, eps=0.0495):
    """Where an interval sits relative to the equivalence margin: inside it,
    above it, below its negative, or straddling one of the two edges.  The
    diagnostic tables print this in place of a verdict (supplement, the
    pre-registered design section)."""
    if -eps <= lo and hi <= eps:
        return 'inside'
    if lo > eps:
        return 'above'
    if hi < -eps:
        return 'below'
    return 'straddles'

def decomp(res):
    rows = []
    for env, fl, s in CELLS:
        m = canon_keys(json.load(open(f'{res}/fits/results/duel_{env}_{fl}_mid_s{s}.json'))['payload']['means'])
        ab1 = m['A'] - m['B1']; ask = m['A'] - m['A_noV']; wait = m['A'] - m['A_noW']
        rows.append(f"{env}$\\times${fl} & {ab1:.3f} & {ask:.3f} & {wait:.3f} & ${ab1-ask-wait:+.3f}$ & ${m['B3']-m['A_noW']:+.3f}$ \\\\")
    return '\n'.join(rows) + '\n\\bottomrule\n'

def b5(res):
    rows = []
    for fl, s in (('F1', 7), ('F2', 8), ('F3', 9)):
        d = json.load(open(f'{res}/fits/b5/b5_E-outage_{fl}_mid_s{s}.json'))['a_minus_b5']
        lo, hi = d['simultaneous_ci95_three_cells']
        v = margin_place(lo, hi)
        rows.append(f"E-outage$\\times${fl} & ${d['mean']:+.4f}$ & $[{lo:.4f},\\,{hi:.4f}]$ & ${d['mean']/D*1e4:.1f}$ & {v} \\\\")
    return '\n'.join(rows) + '\n\\bottomrule\n'

def delay(res):
    out = []
    for env, fl, _ in CELLS:
        d = json.load(open(f'{res}/security/security_{env}_{fl}.json'))['outcomes']
        names = [('A', '$\\Amod$')]
        # on the halting rail B4's tuned watch length is zero, so B3 and B4 are one rule and one row
        if env == 'E-outage':
            names += [('B3', 'B3, B4'), ('B5', 'B5')]
        else:
            names += [('B3', 'B3'), ('B4', 'B4')]
        for j, (n, nm) in enumerate(names):
            o = d[n]
            lab = f" {env}$\\times${fl}" if j == 0 else " "
            out.append(f"{lab} & {nm} & {o['query_frequency']*100:.2f} & {o['mean_delay_seconds']:.1f} & {o['expired_answer_rate']*100:.2f} \\\\")
        out.append('\\addlinespace[2pt]')
    out.pop()
    return '\n'.join(out) + '\n\\bottomrule\n'

def transfer(res):
    rows = []
    for tgt in ('E-slow', 'E-outage'):
        for fl in ('F1', 'F2', 'F3'):
            t = json.load(open(f'{res}/transfer/transfer_E-fast_to_{tgt}_{fl}.json'))
            f = t['fits']; ts, tt = t['tick_seconds_source'], t['tick_seconds']
            secs = (f['source_B4'][0] * ts, f['source_B4'][0] * tt, f['target_B4'][0] * tt)
            def cell(k):
                p = t['paired'][k]
                return f"${p['bp']:+.1f}$ $[{p['bp_ci95'][0]:+.1f},\\,{p['bp_ci95'][1]:+.1f}]$"
            rows.append(f" {tgt}$\\times${fl} & {secs[0]:g} & {secs[1]:g} & {secs[2]:g} & {cell('A_minus_B4xfer')} & {cell('B4tuned_minus_B4xfer')} & {cell('B3tuned_minus_B3xfer')} \\\\")
        if tgt == 'E-slow':
            rows.append('\\addlinespace[2pt]')
    return '\n'.join(rows) + '\n\\bottomrule\n'

def b6(res):
    rows = []
    for fl in ('F1', 'F2', 'F3'):
        p = json.load(open(f'{res}/b6/b6_E-outage_{fl}.json'))['paired']
        a, b = p['A_minus_B6_refined'], p['B5_minus_B6_refined']
        def cell(q):
            lo, hi = q['block_ci95']
            return f"${q['mean']:+.4f}$ & $[{lo:.4f},\\,{hi:.4f}]$ & ${q['bp']:.1f}$"
        rows.append(f"E-outage$\\times${fl} & {cell(a)} & {cell(b)} \\\\")
    return '\n'.join(rows) + '\n\\bottomrule\n'

EXP2_AXES = (('kappa', 'Prior miscalibration $\\zeta$'), ('noise', 'Measurement error $\\eta$'),
             ('delta', 'Regime shift $\\delta$'), ('lambda', 'Response bias $\\lambda$'))
EXP2_FILES = {('E-slow', 'kappa'): 'inject_E-slow_F2_kappa_s2003', ('E-slow', 'noise'): 'inject_E-slow_F2_noise_s2002',
              ('E-slow', 'delta'): 'inject_E-slow_F2_delta_s2004', ('E-slow', 'lambda'): 'inject_E-slow_F2_lambda_s2005',
              ('E-outage', 'kappa'): 'inject_E-outage_F2_kappa_s2006', ('E-outage', 'noise'): 'inject_E-outage_F2_noise_s2001',
              ('E-outage', 'delta'): 'inject_E-outage_F2_delta_s2006', ('E-outage', 'lambda'): 'inject_E-outage_F2_lambda_s2006',
              ('E-slow', 'b4'): 'b4_E-slow_F2_{ax}_s5', ('E-outage', 'b4'): 'b4_E-outage_F2_{ax}_s8'}

def exp2(res):
    d = f'{res}/diagnostics/experiment2'
    out = []
    for ax, name in EXP2_AXES:
        cells = {}
        for env in ('E-slow', 'E-outage'):
            p1 = json.load(open(f'{d}/{EXP2_FILES[(env, ax)]}.json'))['payload']
            b4 = json.load(open(f'{d}/{EXP2_FILES[(env, "b4")].format(ax=ax)}.json'))['payload']['advantage']
            base = p1['no_injection']['mean']
            cells[env] = ({round(r['level'], 4): dict(r, retention=r['mean'] / base) for r in p1['advantage']},
                          {round(r['level'], 4): r for r in b4})
        levels = [round(r['level'], 4) for r in json.load(open(f'{d}/{EXP2_FILES[("E-slow", ax)]}.json'))['payload']['advantage']]
        for j, lv in enumerate(levels):
            lab = name if j == 0 else ''
            lvs = {0.25: '$1/4$', 0.5: '$1/2$', round(2/3, 4): '$2/3$'}.get(lv, f'{lv:g}') if ax == 'lambda' else f'${lv:g}$'
            parts = [lab, lvs]
            for env in ('E-slow', 'E-outage'):
                r1, r4 = cells[env][0][lv], cells[env][1][lv]
                parts.append(f"{r1['mean']:.3f} & {100*r1['retention']:.1f}")
                sg = lambda x: f"{x:+.3f}" if abs(x) >= 5e-4 else "0.000"
                parts.append(f"${sg(r4['mean'])}$ $[{sg(r4['ci95'][0])},\\,{sg(r4['ci95'][1])}]$")
            out.append(' & '.join(parts) + ' \\\\')
        out.append('\\addlinespace[2pt]')
    out.pop()
    return '\n'.join(out) + '\n\\bottomrule\n'

if __name__ == '__main__':
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for name, fn in (('table_decomp_rows.tex', decomp), ('table_exp2_rows.tex', exp2), ('table_b5_rows.tex', b5),
                     ('table_b6_rows.tex', b6), ('table_delay_rows.tex', delay), ('table_transfer_rows.tex', transfer)):
        open(os.path.join(out, name), 'w').write(fn(res))
        print('wrote', name)
