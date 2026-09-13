# exp1_simulation_control — evidence package

> **SIMULATED DATA.** This run was produced from the synthetic corpus in
> `ziggy/simulate.py`. It validates the pipeline and the evaluation harness
> end to end and acts as a positive control. **It is not evidence about real
> markets.** See `REPORT.md` for the real-data run.

_Generated 2026-09-13T05:10:08.492248+00:00 · config `configs/simulation.yaml` · runtime 791.6s_

## The question

> Given everything publicly knowable at the snapshot, can we automatically
> produce a short ranked list of situations that contains substantially more
> consequential future market activity than a naive baseline?

## Headline result

On the **holdout** period (2025-07-15 → 2026-08-31, 285 sessions), ranking with **gbm_wf** and taking the top **20** names per snapshot:

- **precision@20 = 0.257** [0.243, 0.271] against a base rate of 0.100

- **lift@20 = 2.56×** [2.42, 2.70]

- **5.1** of the 20 shortlisted names were consequential on an average day

- the shortlist captured **2.4%** of the day's total absolute excess movement while being 1.5% of the universe (**1.60×** its share)

- selected names moved **1.60×** as far as the average name


![lift bars](fig_lift_bars.png)

![lift vs k](fig_lift_vs_k.png)

## Protocol

- Snapshot convention: **20:00 America/New_York**; the earliest action is the next session's open.

- Period 2021-01-01 → 2026-08-31; 1421 snapshots, 1,934,936 candidate rows, 1362 names ranked per snapshot.

- Chronological splits with a 21-session embargo: train 2021-01-04→2024-04-23 (831), validation 2024-05-23→2025-06-11 (263), holdout 2025-07-15→2026-08-31 (285).

- Label: `label_cs_q90` — |excess move| over 5 sessions in the top decile of that snapshot's own cross-section.

- Model chosen on **validation lift@20**; the holdout was scored 1 time with the choice already fixed.

- 146 features available to the ranker.


## All rankers on the holdout

| model | precision | precision_ci | lift | capture | mag_ratio | ndcg | hit_any |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gbm_wf | 0.257 | [0.243, 0.271] | 2.559 | 0.024 | 1.601 | 0.283 | 0.996 |
| logistic | 0.256 | [0.239, 0.273] | 2.555 | 0.024 | 1.582 | 0.275 | 1.000 |
| logistic_wf | 0.256 | [0.238, 0.273] | 2.550 | 0.024 | 1.591 | 0.276 | 1.000 |
| gbm | 0.255 | [0.241, 0.270] | 2.541 | 0.024 | 1.591 | 0.282 | 0.996 |
| deterministic | 0.195 | [0.185, 0.205] | 1.942 | 0.020 | 1.352 | 0.237 | 0.989 |
| baseline_trailing_vol | 0.192 | [0.178, 0.205] | 1.910 | 0.020 | 1.338 | 0.226 | 0.975 |
| baseline_prior_abs_move | 0.169 | [0.159, 0.179] | 1.680 | 0.019 | 1.254 | 0.213 | 0.989 |
| baseline_inverse_liquidity | 0.129 | [0.117, 0.142] | 1.285 | 0.017 | 1.135 | 0.190 | 0.932 |
| baseline_earnings_only | 0.124 | [0.116, 0.131] | 1.235 | 0.017 | 1.101 | 0.197 | 0.950 |
| baseline_n_filings | 0.117 | [0.108, 0.125] | 1.164 | 0.016 | 1.063 | 0.179 | 0.936 |
| baseline_liquidity | 0.116 | [0.105, 0.128] | 1.158 | 0.016 | 1.065 | 0.178 | 0.939 |
| baseline_volume_spike | 0.116 | [0.108, 0.125] | 1.155 | 0.016 | 1.055 | 0.179 | 0.918 |
| baseline_any_disclosure | 0.115 | [0.107, 0.123] | 1.150 | 0.016 | 1.067 | 0.180 | 0.943 |
| baseline_random | 0.099 | [0.091, 0.106] | 0.984 | 0.015 | 1.008 | 0.169 | 0.879 |


### Paired comparisons at k=20 (selected model minus baseline, same sessions)

| reference | diff | lo | hi | p_value | n |
| --- | --- | --- | --- | --- | --- |
| baseline_random | 1.575 | 1.416 | 1.744 | 0.001 | 280 |
| baseline_any_disclosure | 1.409 | 1.269 | 1.545 | 0.001 | 280 |
| baseline_volume_spike | 1.404 | 1.231 | 1.583 | 0.001 | 280 |
| baseline_liquidity | 1.401 | 1.212 | 1.596 | 0.001 | 280 |
| baseline_n_filings | 1.395 | 1.247 | 1.537 | 0.001 | 280 |
| baseline_earnings_only | 1.324 | 1.192 | 1.464 | 0.001 | 280 |
| baseline_inverse_liquidity | 1.274 | 1.087 | 1.455 | 0.001 | 280 |
| baseline_prior_abs_move | 0.879 | 0.740 | 1.023 | 0.001 | 280 |
| baseline_trailing_vol | 0.649 | 0.505 | 0.796 | 0.001 | 280 |


## Stability through time

| year | n_sessions | base_rate | precision | lift | capture | mag_ratio | ndcg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | 153 | 0.100 | 0.275 | 2.736 | 0.025 | 1.692 | 0.286 |
| 2025 | 229 | 0.100 | 0.260 | 2.597 | 0.024 | 1.627 | 0.283 |
| 2026 | 161 | 0.100 | 0.250 | 2.494 | 0.024 | 1.561 | 0.277 |


![stability](fig_stability.png)


## By volatility regime (holdout)

| regime | n_sessions | base_rate | precision | lift | capture | mag_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| calm | 94 | 0.100 | 0.240 | 2.390 | 0.024 | 1.559 |
| normal | 93 | 0.100 | 0.258 | 2.567 | 0.024 | 1.605 |
| stressed | 93 | 0.100 | 0.273 | 2.721 | 0.024 | 1.639 |


## Robustness to the definition of 'consequential'

| label | k | precision | lift | base_rate |
| --- | --- | --- | --- | --- |
| label_abs | 20 | 0.146 | 3.306 | 0.048 |
| label_vol_expansion | 20 | 0.057 | 1.235 | 0.044 |


## Point-in-time and leakage audits

**acceptance_timezone** — `consistent_with_eastern`

```json
{
  "assumed_timezone": "America/New_York",
  "share_within_edgar_hours_06_22": 1.0,
  "8k_modal_acceptance_hour": 16,
  "verdict": "consistent_with_eastern",
  "note": "a UTC misreading would push the modal 8-K hour into the small hours and drop the in-hours share far below 97%"
}
```

**feature_availability** — `clean`

```json
{
  "events_checked": 137500,
  "violations_disclosure_after_snapshot": 0,
  "min_gap_hours": 0.0,
  "median_gap_hours": 7.516666666666667,
  "status": "clean"
}
```

**forward_columns** — `clean`

```json
{
  "n_features": 146,
  "forbidden_columns_present": [],
  "status": "clean"
}
```

**label_coverage** — ``

```json
{
  "rows": 1934936,
  "rows_with_label": 1928205,
  "label_coverage": 0.9965213319717035,
  "rows_truncated_by_delisting": 954,
  "rows_without_entry_price": 6731
}
```

**shuffle_control** — `clean`

```json
{
  "k": 20,
  "real_lift": 2.5589802265167236,
  "shuffled_score_lift": 1.0215835571289062,
  "status": "clean"
}
```

**leakage_canary** — `detector_works`

```json
{
  "k": 20,
  "cheating_lift": 9.940949440002441,
  "status": "detector_works",
  "note": "this ranker is allowed to see the label; it is never used in the experiment"
}
```

**permutation_test** — ``

```json
{
  "observed_lift": 2.5589802265167236,
  "null_mean": 1.0050071636835733,
  "null_std": 0.035771119165991704,
  "null_p95": 1.0574447333812713,
  "p_value": 0.01639344262295082,
  "n_permutations": 60
}
```


## Survivorship

```json
{
  "tickers_ever_in_universe": 1724,
  "tickers_disappeared_before_end": 259,
  "annual_disappearance_rate": 0.02657251563175901,
  "expected_annual_delisting_rate_us_equities": 0.04,
  "tickers_first_seen_after_start": 205,
  "names_per_session_median": 1365.0,
  "names_per_session_min": 1292,
  "names_per_session_max": 1417,
  "mean_daily_membership_churn": 4.584507042253521
}
```


## Reproduce

```bash
pip install -r requirements.txt
python -m ziggy.cli all --config configs/simulation.yaml
```
