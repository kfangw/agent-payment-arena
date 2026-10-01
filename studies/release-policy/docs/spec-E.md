# Experiment E spec: measured x402 payment sizes

Fixed before any result was seen. Scripts: `scripts/run_E.py`, `scripts/analyze_E.py`; data: `data/x402_amounts/`; results: `results/measured_streams/` (`summary.json`, `summary.md`, `manifest.json`, `plan/`).

Motivation: a sigma = 1 log-normal exposure law does not agree with the cited statement that payments of $1 or more make up 95% of the value, and no test distribution is the actual distribution.

## 1. Question

Can the amount distribution of actual x402 payments be obtained from on-chain data? If yes, what does the compiled policy do on that distribution?

## 2. Data (exploration, 2-hour limit)

* Candidate source: Dune (public x402 dashboards and queries; the USDC transfers on Base that an x402 facilitator submitted). Search for the Coinbase facilitator addresses and the x402 dashboards.
* What is downloaded: a histogram or quantiles of the payment amounts (USDC) of the last 30 to 90 days. No individual address or identifying information is used. The text gives only the source (dashboard name and query date).
* Judgment: if the sample is smaller than 10,000 payments or the facilitator identification is uncertain, drop it and keep only a record of the attempt.

## 3. Design (if data exist)

* Experiment H's code and world unchanged. Only the exposure law changes: an empirical law added as L6, by inverse-transform sampling from the measured quantiles, clipped to [0.01, 2000].
* Four policies (Comp, Cutoff, B4_L, B4_L0); metrics as in experiment H, section 5. Evaluation seed 111.

## 4. Reading rules (fixed in advance)

* If Comp - Cutoff at L6 is inside the margin of 10 bp (relative to L6's mean), write "on measured sizes the policy equals a single cutoff"; if outside, write its size and the share of payments above v*. Either way, rewrite the conditional sentence of "Today's payment sizes" on the measured basis.
* Report the count and value shares of payments above v* as they are.
* State the limits of the data (one facilitator, the period, one chain) in the same paragraph.

## 5. Time limit

Exploration 2 hours. After the data are in hand, the run takes 20 minutes and writing the results into the text 1 hour.

## Change log

(Written after the data were obtained and before the run. The reading rules were not changed.)

* Source. The address list of facilitators used by the public Dune dashboard "x402 Facilitators" (published by the Dune team) was used as listed. From the public table `tokens.transfers` (blockchain base, USDC, tx_from in the list), the amounts were downloaded binned to 3 significant figures (query date 2026-09-30). No individual address or recipient was downloaded. A description of the query is in `data/x402_amounts/README.md`.
* The Coinbase facilitator addresses have no transfers after 2026-01-09 (the addresses appear to have been replaced), so two windows were downloaded. A: all listed Base facilitators, the last 90 days (from 2026-07-02), 144,494 payments, $66,116. B: the Coinbase addresses, their last 90 active days (2025-10-12 to 2026-01-09), 43,734,389 payments, $17,431,903. Both exceed the 10,000-payment criterion. Window A fits "the last 30 to 90 days" of section 2; window B fits the Coinbase facilitator that the text cites. A is L6 and B is L7; both are run and the rule of section 4 is applied to each. The text reports both in one sentence.
* The empirical distribution is used as point masses at the 3-figure bins (inverse-transform sampling). The clip at 0.01 and 2000 is as specified (A has many payments below $0.01, so the clip raises the mean; the realized mean is recorded).
* Summary computed before clipping: A mean 0.4576, median 0.01, count at or above v* (3.13) 1.08%, value 84.3% (a few large individual transfers such as $7,730 dominate). B mean 0.3986, median 0.02, $1.00 is 26.2% of the count, count at or above v* 0.26%, value 27.2%.
