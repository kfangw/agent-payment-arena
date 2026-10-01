# Experiment C spec: B4 tuning variation in the main-result interval

Fixed before any result was seen. Script: `scripts/run_C.py`; results: `results/tuning_variation/` (`summary.json`, `plan/`).

Motivation: a comment on an earlier version noted that the same cell gives a different difference on another seed and learning sample (5.8 [5.1, 6.5] instead of 4.3 ± 1).

## 1. Question

The E-fitted row of the main-result table (Comp - B4: F1 +3.8, F2 +4.3, F3 +1.4 bp) has an interval from a paired block bootstrap with B4 fixed to one learning sample. If the tuning sample of B4 changes, how much does the difference move?

## 2. Design

* World: E-fitted, flows F1, F2, F3, main-run constants, and the main run's compiled table (true constants). The evaluation set is the main run's (per flow 5,663,400 payments, see the change log for the seeds). Comp is replayed once per flow.
* B4: the existing tuner (161-point grid, k in {0, 1, 2, 4, 8, 16, 30}). Five independent learning samples of 200,000 payments, `default_rng([112, 700, s])`, s = 1..5, are drawn; B4 is tuned five times and each B4_s is replayed on the evaluation set.
* Metric: per flow D_s = J(Comp) - J(B4_s) in bp (of the mean exposure 49.50). The mean, minimum, maximum and standard deviation of the five values. And an interval that includes tuning variation: the 2.5% and 97.5% percentiles of a two-way resampling, a block bootstrap (1,000 resamples) times a resampling of the tuning samples (5 drawn with replacement from 5).
* Check: the (k, a, b) of the main run's B4 is not required to be reproduced at s, since the learning samples differ. Instead, J of Comp must agree with the main run's value (the source data of the main-result table).

## 3. Reading rules (fixed in advance)

* If, for every flow, the lower end of the two-way interval is at least -10 bp and the upper end at most +10 bp, keep "equal within the margin" and report "half-width of the interval with tuning variation <= x bp".
* If the interval exceeds the margin in any flow, the text changes to "on average within the margin; the interval with tuning variation reaches ±y", and "matches in payoff" in the abstract gets "on average".
* If the signs of the five values differ (B4 is ahead at some s), state that.

## 4. Run time

Per flow: draw 1 min + Comp replay 10 s + 5 B4 tunings x 1.6 min + 5 replays x 10 s, about 10 min. Three flows, 30 min.

## Change log

* Evaluation set. The first version of this spec said "the main run's seed 8". The main run uses seeds 7, 8, 9 for F1, F2, F3 and draws with `default_rng(seed)` first the 200,000 tuning payments and then the 5,663,400 evaluation payments (`scripts/run_queue.py`, the repro step of `scripts/run_D.py`). This evaluation set was used (`run_D2.draw_main`). It is the same draw as experiment D2; the block sums of Comp are checked against those of D2.
* Agreement with the main run's J(Comp). The original block-sum files of the main run are not part of this package. Instead, in experiment D2, B4 was retuned on the main run's tuning split and replayed on the same evaluation set; Comp - B4 equals the main-run values (`results/generated/lead_cells.json`) in all three flows to the first decimal, and the intervals too. The D of this B4_main is also reported for reference.
* The five B4 tunings use the same tuner as `run_H.py` (vectorized forced replays + `tune_watch_policy`, 161-point grid, `horizon_grid(H)`), run in two parallel processes. The learning-sample seed `default_rng([112, 700, s])` is the same for all flows (as specified).
* The generator for resampling the tuning samples in the two-way interval is `default_rng([20260930, 5])`, and for the block resampling `default_rng(20260930)`, both restarted per flow. In each resample five of the five D_s are drawn with replacement and averaged.
