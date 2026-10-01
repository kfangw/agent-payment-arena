# Experiment H smoke checks (spec sec 9)

Records: `smoke/progress.log`, `smoke/checks.json`, `smoke/timing.json`, `smoke/summary.md`. Generated 2026-09-30 21:16.

| Check | Content | Numbers | Verdict |
|---|---|---|---|
| 9.1 | Reproduction of the seed-108 reference value of experiment D (F2 rebuilt with the L0 law, true-q table, vectorized replay) | J = 8.938881202421832, D = 8.938881202421832, difference 0.0e+00. L0 law equals the vdist of make_flows F2: True. Full-size draw 59.8 s, replay Comp 11.8 s, Cutoff 7.9 s, B4 (k = 1) 12.3 s, B4 (k = 30) 21.3 s | PASS |
| extra | Vectorized replay = replay_outage of the repository | first 20,000 payments, L0 (seed 108), L1 and L3 (seed 111), the four policies and 9 forced policies (k = 0, 4, 30), including the first action of Comp (vectorized = repository policy call): 42 cases, per-payment maximum difference 0 in all, grant counts equal | PASS |
| 9.2 | Identical decisions below the table's lowest v, pi grid of 501 points, running and halted arrivals | mismatches (running / halted): v = 0.01: arrivals 0 / 0, all states 0, v = 0.05: arrivals 0 / 0, all states 0, v = 0.2: arrivals 0 / 0, all states 0, v = 0.5: arrivals 0 / 0, all states 0. Arrival actions in the $0.5 row (running and halted together, 1,002 points): grant 128, reject 797, verify 0, wait 77 | PASS |
| 9.3 | Recomputation of pihat(0) | sigma(0) = (1 - 0.005)(1 - 3e-05)^16 = 0.9945225074449573, pihat(0) = 0.2551795105735829, third decimal 0.255 | PASS |
| 9.4 | sigma_3 of L3 | analytic sigma_3 = 4.078867, mu_3 = -9.9280, median 4.879e-05. Untruncated: mean 0.2000, value share at or above $1 0.9500. Sample of 1 million (before clip): mean 0.1594, share 0.9367; after clip to [0.01, 2000]: mean 0.1502, share 0.8764, which leave +-2%. Clipped at the lower bound 90.4%, at the upper bound 0.001%, value share above $2000 (analytic) 0.413. As instructed, the analytic sigma_3 is kept and the realized values are recorded | PASS (analytic); sample condition not met, recorded |
| 9.5 | Smoke run (100,000 evaluation payments, L0, L1, four policies, 100 bootstrap resamples) | 0 errors, `smoke/summary.md` written | PASS |

## Run-time estimate (spec sec 7)

One law = full-size draw 59.8 s + 4 replays 53.2 s + B4 learning (200,000-payment draw, 21 forced replays, 161-point grid tuning) 96.7 s + other 15 s = 225 s. Six laws 0.37 h, limit 2.5 h. L4 dropped: no.

Smoke numbers are not results.
