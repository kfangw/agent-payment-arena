# Experiment F4 spec: a flow built from public labelled fraud data

Fixed before any result was seen. Scripts: `scripts/prep_F4.py`, `scripts/run_F4.py`, `scripts/analyze_F4.py`; data: `data/card_fraud/`; results: `results/real_label_flow/` (`summary.json`, `summary.md`, `plan/`, `verify/`).

## 0. Purpose

The flows F1 to F3 are synthetic: the suspicion score and the intent are drawn jointly from a declared model, so the model tests itself. F4 takes one flow from real labelled data. Every payment gets a calibrated score together with the true label of a real transaction. What is borrowed is the joint distribution of score and label only. A card fraud is a different kind of misuse from the misuse of a delegated agent, and the data contain no exposure. The experiment shows what the compiled policy does on the score distribution and the calibration error of a real scorer, and it claims nothing beyond that.

## 1. Questions

Two questions, in this order.

1. Where does the score distribution of a real scorer sit relative to the places the model assumes? Report the share of payments whose score falls in the verify band [a, b) = [0.1625, 0.63125), the calibration inside that band, and the distortion zeta-hat obtained by fitting x = min(zeta pi, 1) to the labels.
2. On that distribution, is the policy compiled from constants alone (Comp) equal to B4 tuned on labels within the 10 bp margin? How large are the value of the question channel (Comp minus Comp without the verify action) and the share of misuse released?

The answer to the second question is likely to be trivial (section 6). The first question is the point of the experiment.

## 2. Data

ULB "Credit Card Fraud Detection" (Dal Pozzolo et al., 2015, IEEE SSCI, doi 10.1109/SSCI.2015.33): 284,807 card transactions, 492 frauds (0.172%), features V1 to V28 (PCA components), Time, Amount, label Class. Two days of European card transactions in September 2013. Source: OpenML data set 1597 or Kaggle `mlg-ulb/creditcardfraud`. The raw file is not shipped; see `data/card_fraud/README.md`.

A second data set (IEEE-CIS) is not used.

## 3. Scores (prep_F4.py)

The target is a calibrated suspicion score pi0 = P(misuse | x) together with its label.

1. Split by time into three parts, ordered by the Time column: train, calibration, evaluation. This is not a random split: a deployed scorer is trained on the past and used on the future. Frauds cluster within the two days, so the number of frauds per part is not known in advance. Rule fixed in advance: start with 60/20/20; if the evaluation part has fewer than 60 frauds or the calibration part has fewer than 100, use 50/25/25; if that also falls short, report the numbers and continue. The split is decided once, from label counts alone, and never changed after seeing scores or results. Applied before any score was computed: 60/20/20 gives 360, 57 and 75 frauds in train, calibration and evaluation, so the calibration part falls short; 50/25/25 gives 269, 129 and 94. The split used is 50/25/25.
2. One classifier on the train part: `HistGradientBoostingClassifier` with default settings and `random_state=0`. Features: V1 to V28 and log(1 + Amount). Time is not used.
3. Calibrate on the calibration part with `IsotonicRegression(out_of_bounds="clip")` (`CalibratedClassifierCV(cv="prefit")` is deprecated in scikit-learn 1.6 and later and is not used). Apply it to the evaluation part to get pi0, clipped to [1e-9, 1 - 1e-9].
4. The pairs (pi0, Class) of the evaluation part are the raw data of flow F4.
5. Calibration checks. Quantile-bin ECE is not used: at a base rate of 0.17% nine of ten quantile bins have pi0 close to 0 and the statistic stays below 0.001 whatever the calibration. Instead:
   * A table over the fixed bins [0, 0.05), [0.05, a), [a, 0.3), [0.3, b), [b, 0.9), [0.9, 1] with the count, mean pi0 and observed fraud rate in each; a = 0.1625 and b = 0.63125 are the verify band of the main run (`band_fn` of `scripts/run_H.py`).
   * The share of rows in the band [a, b) and, inside it, the observed fraud rate against the mean pi0. This is the first reported quantity.
   * zeta-hat: the maximum-likelihood fit of the distortion on the whole evaluation part, with the function `zeta_mle(s, y)` of `scripts/run_G.py` unchanged (a different function would not be comparable with the distortion-sensitivity experiment). `zeta_mle` fits P(y = 1 | s) = min(s / zeta, 1) with s = pi0; its output is exp(0.01 k) for the grid index k. Report zeta-hat with a bootstrap interval from 200 row resamples.
   * For reference: overall ECE over the fixed bins, AUC and average precision.
6. Outputs: `data/card_fraud/F4_scores.npz` (pi0, label, amount, split bounds) and `data/card_fraud/F4_calibration.json` (bin table, band share, zeta-hat and interval, ECE, AUC and AP, rows and frauds per part).

## 4. Flow

`EmpiricalFlow` is defined in `scripts/run_F4.py`; `flows.py` is not touched. The interface is that of the other flows, `Flow.sample(n, rng) -> (v, p_true, theta, pi0)`. `sample` draws rows of the evaluation part with replacement (row indices from `rng.integers`) and uses the row's pi0 both as p_true and as pi0. The intent theta is the row's actual label; it is not redrawn as Bernoulli(p). This keeps "a real label for every score": if the calibration were perfect the two choices would agree in expectation, and the difference is the calibration error. p_true is set to pi0 only because some harness checks read p_true as an oracle; the experiment does not use it.

Exposure v is the declared lognormal of the main run (median 30, sigma 1, truncated to [0.5, 2000]). The Amount column is a card transaction amount, not an x402 exposure, and is not used. The random-number order is: first the exposure draw, then the row indices.

The base rate is set by the data (about 0.13% in the evaluation part). No reweighting of the base rate is done: that would make the flow synthetic again.

## 5. Design

* World: E-fitted, main-run constants (m 0.35, h 1, C 0.5, c_w 0.006, tau 10, q 0.75, rho 0.1294).
* Four policies. Comp: the main-run table, not recompiled (the table does not depend on the flow). B4_F4: tuned on the learning draw with the existing tuner on its 161-point grid. Cutoff: grant iff pi0 < pihat(0) = 0.2552. Comp_nv: the compiled policy without the verify action (the table of experiment D2).
* Draw sizes: 200,000 learning payments (`default_rng([111, 700, 40])`) and 500,000 evaluation payments (seed 111, as in `scripts/run_H.py`). The evaluation part has about 71,000 rows, so 500,000 payments resample each row about seven times.
* Uncertainty at two levels. The result is limited by the roughly 100 frauds of the evaluation part, not by the length of the draw; a draw-level interval alone would be falsely tight.
  * Row level (main): resample the rows of the evaluation part with replacement, build a new `EmpiricalFlow`, draw an evaluation sample from it and replay the four policies; 200 repetitions. The 95% interval of Comp minus B4_F4 is the percentile interval of the 200 differences. B4_F4 is not retuned in a repetition. The replay time per repetition is measured in the smoke run first; if it exceeds 30 s the number of repetitions is cut to 100 and this is reported.
  * Draw level (auxiliary): paired block bootstrap with the function used by the other experiments (`analyze_H.py`), so that F4 is reported in the same form as the other flows. Both intervals are shown side by side.
* Metrics: J(Comp) - J(B4_F4) in bp with both intervals; J(Comp) - J(Cutoff); J(Comp) - J(Comp_nv) (value of the channel); the share and the dollar exposure of misuse granted (Comp, B4_F4, Cutoff); the verify share of Comp at arrival; the share of payments in the band. bp = (J_comp - J_other) / (realized mean exposure of the evaluation draw) x 10^4. All metrics come from one evaluation draw, as in experiments D2 and L.
* Smoke run: 100,000 evaluation payments first, to check ranges and replay time, and the vectorized replay against `replay_outage` of the repository on a prefix.

## 6. Reading rules (fixed in advance; not changed after seeing results)

Expected: after isotonic calibration pi0 has a single peak near 0 and a tail of a few hundred rows. Payments in the band number a few dozen. All four policies converge on "grant almost everything, reject a few high scores". Then Comp minus B4 is about 0, the channel is worth about 0, and the share of misuse released is the share of frauds the classifier scores below 0.2552, the same for the four policies. If so, the answer to question 2 is trivial and the headline is question 1 (band share, calibration in the band, zeta-hat).

* Band share and zeta-hat: report "on the real scorer the band holds x% of payments, zeta-hat = y [interval]". If the interval contains 1, report that the calibration error lies inside the first grid point of the distortion-sensitivity curve (rule R-zeta-1); otherwise (R-zeta-2) read that curve at zeta-hat.
* Comp minus B4_F4: if the row-level 95% interval lies inside [-10, +10] bp, report "equal within the margin on the real-label flow" (R-b4-1). If it falls below -10, report it as it is, "behind B4 by x bp" (R-b4-2). If it exceeds +10, do not write "ahead"; write "equal or above" (R-b4-3). If it exceeds 10 on both sides, report that this data cannot support a margin verdict and give the width (R-b4-4).
* Value of the channel, Comp minus Comp_nv: if the point value is within 2 bp (the reading of "about 1 bp"), report that the channel is hardly used in F4 (R-ch-1); this agrees with the statement that below the floor the policy is a single cutoff. If larger, report it (R-ch-2) and check the fixed-bin table for a thick band.
* If Comp trails B4_F4, compare the shortfall with the value of the distortion-sensitivity curve at zeta-hat and split it into the part explained by calibration error and the remainder.
* Whatever the outcome: all numbers are reported.

## 7. Outputs

Quantities reported, in this order: share of payments in the band; fraud rate in the band against mean pi0; zeta-hat with interval; the fixed-bin table; Comp - B4_F4 in bp with the row and the draw interval; Comp - Cutoff in bp; Comp - Comp_nv in bp; the three misuse pass rates; the three misuse exposures in dollars; the verify share; ECE; AUC and AP; rows and frauds of each part; the base rate. They are recomputed from the stored block sums by `scripts/analyze_F4.py`, and `results/real_label_flow/verify/recheck.json` holds an independent recomputation from the raw files.

## 8. As run

The shipped results follow this design with these departures: (1) the block size of the draw-level bootstrap is 50 payments (one episode), the function of `analyze_H.py`; (2) the "about 1 bp" threshold of the channel rule was read as 2 bp; (3) the auxiliary variant of B4_F4 tuned on the calibration part and the two measured payment-size laws were not run; (4) the 200 row-level repetitions all use the same evaluation seed (111) and resample the rows only, so the row-level interval of Comp - B4_F4 is much narrower than the draw-level interval; the draw-level interval is the honest width for that comparison; (5) the split followed the rule of section 3.1 (50/25/25).
