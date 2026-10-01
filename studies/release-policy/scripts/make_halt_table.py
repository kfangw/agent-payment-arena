"""Generate the table rows for the measured halt law (5-minute threshold) and the declared law.

Definitions (same as in data/halt/fitted_rail.py):
    - halt: rows of gaps.csv (all gaps above 60 s) with gap_minutes > 5.
    - mean halt (min) = mean gap_minutes of those halts.
    - p10 = 1 / mean halt (min).  (fitted_rail.py FIT_P10 = 1 / 22.8)
    - normal_minutes = span_days*24*60 - total halt minutes.
    - p01 = number of halts / normal_minutes.  (fitted_rail.py FIT_P01 = 18 / 517975.2)
    - stationary halt share = p01 / (p01 + p10).

Expected values (5-minute row of the threshold sweep):
    18 halts, 22.8 min, p01 3.48e-5, p10 0.0439, 0.0792%
The declared row is not computed; its constants are copied as given:
    24 per year, 60.0 min, p01 4.63e-5, p10 0.0167, 0.28%

Run:
    python3 scripts/make_halt_table.py [--gaps PATH]
If a computed value differs from the expected one (at the printed precision) the
script prints the difference and exits with status 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, ROOT  # noqa: E402

DEFAULT_GAPS = ROOT / 'data' / 'halt' / 'gaps.csv'
DEFAULT_META = ROOT / 'data' / 'halt' / 'collect-meta.json'
OUT_PATH = RESULTS / 'generated' / 'table_halt_rows.tex'

THRESHOLD_MIN = 5.0

# Declared row: constants copied from the threshold-sweep table, not computed here.
# The stationary share is printed as 0.28, the precision of the rails table.
DECLARED = {
    'label': 'Declared (Table~\\ref{tab:envs})',
    'count_label': '24 per year',
    'mean_min': 60.0,
    'p01': 4.63e-5,
    'p10': 0.0167,
    'share_pct': 0.28,
}

# Expected values: the 5-minute row of the threshold sweep.
EXPECTED_FIT = {
    'count': 18,
    'mean_min': 22.8,
    'p01': 3.48e-5,
    'p10': 0.0439,
    'share_pct': 0.0792,
}


def load_gaps(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f)
        return [float(row['gap_minutes']) for row in reader]


def compute_fit(gaps_path: Path, meta_path: Path):
    gaps_min = load_gaps(gaps_path)
    halts = [g for g in gaps_min if g > THRESHOLD_MIN]
    count = len(halts)
    mean_min = sum(halts) / count

    meta = json.loads(meta_path.read_text())
    span_days = meta['span_days']
    total_span_min = span_days * 24 * 60
    total_halt_min = sum(halts)
    normal_min = total_span_min - total_halt_min

    p01 = count / normal_min
    p10 = 1 / mean_min
    share = p01 / (p01 + p10)

    return {
        'count': count,
        'mean_min': mean_min,
        'p01': p01,
        'p10': p10,
        'share_pct': share * 100,
    }


def round_sig(x: float, sig: int = 3) -> float:
    if x == 0:
        return 0.0
    from math import floor, log10
    d = sig - int(floor(log10(abs(x)))) - 1
    return round(x, d)


def fmt_p(x: float) -> str:
    # Scientific notation, three significant digits.
    mant, exp = f'{x:.2e}'.split('e')
    exp_i = int(exp)
    return f'{mant}e{exp_i:+d}'.replace('e+0', 'e').replace('e-0', 'e-')


def check(computed: dict, expected: dict) -> list[str]:
    diffs = []
    if computed['count'] != expected['count']:
        diffs.append(f"count: computed={computed['count']} expected={expected['count']}")
    if round(computed['mean_min'], 1) != expected['mean_min']:
        diffs.append(
            f"mean_min: computed={computed['mean_min']:.4f} "
            f"(rounded {round(computed['mean_min'], 1)}) expected={expected['mean_min']}")
    if round_sig(computed['p01'], 3) != expected['p01']:
        diffs.append(
            f"p01: computed={computed['p01']:.6e} "
            f"(rounded {round_sig(computed['p01'], 3):.2e}) expected={expected['p01']:.2e}")
    if round(computed['p10'], 4) != expected['p10']:
        diffs.append(
            f"p10: computed={computed['p10']:.6f} "
            f"(rounded {round(computed['p10'], 4)}) expected={expected['p10']}")
    if round(computed['share_pct'], 4) != expected['share_pct']:
        diffs.append(
            f"share_pct: computed={computed['share_pct']:.6f} "
            f"(rounded {round(computed['share_pct'], 4)}) expected={expected['share_pct']}")
    return diffs


def render_tex(fit: dict) -> str:
    # p01 in scientific notation, p10 with four decimals.
    fit_row = (
        f"Fitted, 5-min threshold & {fit['count']} in 360 days & "
        f"{fit['mean_min']:.1f} & {fmt_p(fit['p01'])} & {fit['p10']:.4f} & "
        f"{fit['share_pct']:.4f}\\% \\\\"
    )
    declared_row = (
        f"{DECLARED['label']} & {DECLARED['count_label']} & "
        f"{DECLARED['mean_min']:.1f} & {fmt_p(DECLARED['p01'])} & "
        f"{DECLARED['p10']:.4f} & {DECLARED['share_pct']:.2f}\\% \\\\"
    )
    return fit_row + '\n' + declared_row + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gaps', type=Path, default=DEFAULT_GAPS)
    ap.add_argument('--meta', type=Path, default=DEFAULT_META)
    ap.add_argument('--out', type=Path, default=OUT_PATH)
    args = ap.parse_args()

    fit = compute_fit(args.gaps, args.meta)
    diffs = check(fit, EXPECTED_FIT)

    print('computed:', json.dumps(fit, ensure_ascii=False, indent=1))
    print('expected:', json.dumps(EXPECTED_FIT, ensure_ascii=False, indent=1))

    if diffs:
        print('MISMATCH:')
        for d in diffs:
            print(' -', d)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_tex(fit))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
