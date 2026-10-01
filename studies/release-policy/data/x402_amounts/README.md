# Measured x402 payment sizes (`hist.json`)

`hist.json` is the histogram of on-chain payment amounts used by `scripts/run_E.py` (laws L6 and L7, design notes in `docs/spec-E.md`). It is the only external data of experiment E and is read by `scripts/run_E.py` and `scripts/analyze_E.py`.

## Source

* Platform: Dune, public table `tokens.transfers`.
* Filter: blockchain `base`, token symbol `USDC`, and the sender of the transfer (`tx_from`) in the set of x402 facilitator addresses. The address list is the one used by the public Dune dashboard "x402 Facilitators", published by the Dune team; it was used as listed and was not edited.
* Query date: 2026-09-30.
* No individual address, transaction hash or recipient was downloaded. The query returns only, per amount, the number of transfers and their summed dollar value.

## What the query does

In outline (the SQL text itself is not part of this package):

1. Select the USDC transfers on Base whose sender is in the facilitator address list, inside a date window.
2. Convert the raw token amount to dollars (USDC has 6 decimals) and round it to 3 significant figures.
3. Group by the rounded amount and return `n` (count of transfers) and `usd` (sum of the exact dollar amounts in the group).

Two windows were queried:

| Window key | Content | Payments | Sum in dollars | Distinct amounts |
|---|---|---:|---:|---:|
| `A_all90d` | all listed Base facilitators, transfers from 2026-07-02 on (the last 90 days at the query date) | 144,494 | 66,116 | 1,195 |
| `B_cb90d` | the Coinbase facilitator addresses of the list, 2025-10-12 to 2026-01-09 (their last 90 active days; these addresses have no transfers after 2026-01-09) | 43,734,389 | 17,431,903 | 4,038 |

## File format

`hist.json` has three keys: `source` (text), `windows` (the two window descriptions above) and `rows`. Each row is `{"win": <window key>, "amt3": <amount rounded to 3 significant figures, dollars>, "n": <number of transfers>, "usd": <summed dollar value>}`. `scripts/run_E.py` uses `amt3` as the support and `n / sum(n)` as the weights of an empirical exposure law, clipped to [0.01, 2000]; window `A_all90d` is law L6 and `B_cb90d` is law L7.

## Limits

One facilitator address list, one chain (Base), one token (USDC), a 90-day window each; window B ends on 2026-01-09 and is not recent data. Amounts are binned to 3 significant figures. Transfers that a facilitator sends for other purposes than settling a payment, if any, are not separated.


# Revert share of facilitator transactions (`revert_share.json`)

`revert_share.json` is the only external data of experiment F0 (`docs/spec-F0.md`, `scripts/run_F0.py`). It is read by `scripts/run_F0.py`.

## Source

* Platform: Dune, public table `base.transactions`.
* Filter: blockchain `base`, and `from` in the same set of x402 facilitator addresses as above (the list of the public Dune dashboard "x402 Facilitators", used as listed). The `success` flag of each transaction decides whether it reverted.
* Query date: 2026-10-01.
* No individual address, transaction hash or recipient was downloaded. The query returns only two counts per window: all transactions, and reverted transactions.

## Windows

The same two windows as above (same address sets and date ranges).

| Window key | Transactions | Reverted | Revert share |
|---|---:|---:|---:|
| `A_all90d` | 147,059 | 596 | 0.405% |
| `B_cb90d` | 44,276,970 | 538,473 | 1.216% |

## File format

`revert_share.json` has the keys `source` (text), `windows` (for each window key: `desc`, `n_tx`, `n_revert`, `revert_share`), `declared_f0` (0.005, the declared stage-0 failure probability of E-fitted) and `c_w` (0.006, the cost constant it is compared with).

## Limits

The data do not show whether a reverted payment is submitted again; if the resubmission succeeds, the failure rate of a payment is lower than the revert share. A revert can also have causes that are not a failure of the payment. The two windows differ by a factor of 3 in the revert share.

