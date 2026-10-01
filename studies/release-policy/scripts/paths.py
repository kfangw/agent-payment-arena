"""Locations used by the experiment scripts, all relative to the package root.

Set the environment variable ARENA_RESULTS to move the whole results tree.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("ARENA_RESULTS", ROOT / "results"))

WRONG_CONSTANTS = RESULTS / "wrong_constants"
R_CROSS = RESULTS / "r_cross"
DRAW_INJECTION = RESULTS / "draw_injection"
H_ESLOW = RESULTS / "h_eslow"
D_ANSWERS = RESULTS / "D_answers"
COMPUTE_COST = RESULTS / "compute_cost"
REFERENCE_B4 = RESULTS / "reference_b4"          # b4ref_E-fitted_F{1,2,3}_mid_s{7,8,9}.json
ARCHIVED_BLOCKS = RESULTS / "archived_blocks"    # per-policy block sums, regenerated (not shipped)
EXPOSURE_BAND = RESULTS / "exposure_band"        # B4 tuned per exposure band (xband.py); runs/ holds per-flow output
REGULARIZED_B4 = RESULTS / "regularized_b4"      # regularized small-sample B4 (run_E1.py)
D_PLAN = D_ANSWERS / "plan"                      # plan files of run_D.py (if regenerated)
D_PLAN_SHIPPED = RESULTS / "response_records" / "plan"   # the shipped copy
CACHE = RESULTS / "cache"                        # compiled-table caches (regenerated, not shipped)
CALIBRATION_PILOT = RESULTS / "calibration_pilot"
EXPOSURE_LAW = RESULTS / "exposure_law"
TUNING_VARIATION = RESULTS / "tuning_variation"  # run_C.py
NO_CHANNEL = RESULTS / "no_channel"              # run_D2.py
MEASURED_STREAMS = RESULTS / "measured_streams"  # run_E.py, analyze_E.py
X402_AMOUNTS = ROOT / "data" / "x402_amounts"    # measured payment-size histogram (hist.json)
HARM_FRONTIER = RESULTS / "harm_frontier"        # run_HF.py
SUBMISSION_HAZARD = RESULTS / "submission_hazard"  # run_F0.py
MISUSE_ANSWER = RESULTS / "misuse_answer"        # run_L.py
REAL_LABEL_FLOW = RESULTS / "real_label_flow"    # run_F4.py, analyze_F4.py
CARD_FRAUD = ROOT / "data" / "card_fraud"        # prep_F4.py: creditcard.csv (not shipped), F4_scores.npz, F4_calibration.json
