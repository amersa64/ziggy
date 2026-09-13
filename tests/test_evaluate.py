from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ziggy.evaluate import (
    block_bootstrap_ci,
    paired_block_bootstrap,
    per_session_metrics,
    permutation_null,
    summarise,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    rows = []
    for s in pd.bdate_range("2021-01-04", "2021-09-30"):
        n = 300
        sig = rng.normal(size=n)
        mag = np.abs(rng.normal(size=n) * 0.03 + 0.02 * sig)
        rows.append(pd.DataFrame({"session": s, "informative": sig, "noise": rng.normal(size=n), "mag": mag}))
    d = pd.concat(rows, ignore_index=True)
    d["label"] = (d.groupby("session")["mag"].rank(pct=True) > 0.9).astype(float)
    return d


def test_uninformative_score_has_lift_one(scored):
    ps = per_session_metrics(scored, "noise", "label", "mag", [20])
    assert abs(float(ps["lift"].mean()) - 1.0) < 0.1


def test_informative_score_beats_random(scored):
    a = per_session_metrics(scored, "informative", "label", "mag", [20])["lift"].mean()
    b = per_session_metrics(scored, "noise", "label", "mag", [20])["lift"].mean()
    assert a > b + 0.5


def test_perfect_oracle_achieves_the_maximum_lift(scored):
    d = scored.assign(oracle=scored["mag"])
    ps = per_session_metrics(d, "oracle", "label", "mag", [20])
    # 20 picks from a 300-name universe with a 10% base rate: every pick is a hit.
    assert float(ps["precision"].mean()) > 0.99
    # 6.7% of the universe holding a far larger share of the total movement.
    assert float(ps["capture"].mean()) > 2.5 * (20 / 300)


def test_capture_is_bounded_and_exceeds_the_naive_share(scored):
    ps = per_session_metrics(scored, "informative", "label", "mag", [20])
    assert (ps["capture"].between(0, 1)).all()
    assert float(ps["capture"].mean()) > 20 / 300


def test_lift_decreases_as_the_shortlist_grows(scored):
    s = summarise(per_session_metrics(scored, "informative", "label", "mag", [10, 50, 100]), n_boot=200)
    lifts = s.sort_values("k")["lift"].to_numpy()
    assert lifts[0] > lifts[1] > lifts[2]


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(2.0, 0.5, 400)
    mean, lo, hi = block_bootstrap_ci(v, n_boot=500, block=10, seed=1)
    assert lo < mean < hi
    assert hi - lo < 0.5


def test_paired_bootstrap_finds_no_difference_between_identical_series():
    rng = np.random.default_rng(2)
    v = rng.normal(1.0, 0.3, 300)
    res = paired_block_bootstrap(v, v.copy(), n_boot=300, seed=2)
    assert abs(res["diff"]) < 1e-9
    assert res["p_value"] > 0.5


def test_permutation_null_collapses_to_one(scored):
    res = permutation_null(scored, "informative", "label", "mag", 20, n_perm=15, seed=3)
    assert abs(res["null_mean"] - 1.0) < 0.15
    assert res["observed_lift"] > res["null_p95"]


def test_ties_are_broken_randomly_not_alphabetically():
    d = pd.DataFrame({
        "session": ["s"] * 100, "score": [1.0] * 100,
        "ticker": [f"T{i:03d}" for i in range(100)],
        "mag": np.linspace(0, 1, 100),
    })
    d["label"] = (d["mag"] > 0.9).astype(float)
    a = per_session_metrics(d, "score", "label", "mag", [10], seed=1)["precision"].iloc[0]
    b = per_session_metrics(d, "score", "label", "mag", [10], seed=2)["precision"].iloc[0]
    c = per_session_metrics(d, "score", "label", "mag", [10], seed=1)["precision"].iloc[0]
    assert a == c            # same seed reproduces
    assert (a, b) != (1.0, 1.0)  # and does not systematically pick the winners


def test_univariate_lift_matches_the_reference_metric(scored):
    """The fast path must agree exactly with per_session_metrics."""
    from ziggy.evaluate import univariate_lift

    u = univariate_lift(scored, ["informative", "noise"], "label", "mag", 20, seed=7)
    ref = float(per_session_metrics(scored.assign(_f=scored["informative"]),
                                    "_f", "label", "mag", [20], seed=7)["lift"].mean())
    assert u.set_index("feature").loc["informative", "lift_high"] == pytest.approx(ref, rel=1e-9)


def test_univariate_lift_ranks_an_informative_feature_above_noise(scored):
    from ziggy.evaluate import univariate_lift

    u = univariate_lift(scored, ["informative", "noise"], "label", "mag", 20, seed=7)
    assert u["feature"].iloc[0] == "informative"
    assert u.set_index("feature").loc["noise", "best_lift"] < 1.3


def test_univariate_lift_finds_two_sided_signals_via_extremeness():
    rng = np.random.default_rng(11)
    rows = []
    for s in pd.bdate_range("2021-01-04", "2021-12-31"):
        n = 250
        two_sided = rng.normal(size=n)
        rows.append(pd.DataFrame({"session": s, "two_sided": two_sided}))
    d = pd.concat(rows, ignore_index=True)
    # magnitude depends on |two_sided|, so neither direction alone captures it
    d["mag"] = np.abs(rng.normal(0, 0.02, len(d))) + 0.03 * d["two_sided"].abs()
    d["label"] = (d.groupby("session")["mag"].rank(pct=True) > 0.9).astype(float)

    from ziggy.evaluate import univariate_lift

    r = univariate_lift(d, ["two_sided"], "label", "mag", 20, seed=3).iloc[0]
    assert r["direction"] == "extreme"
    assert r["lift_extreme"] > max(r["lift_high"], r["lift_low"]) + 0.5


def test_univariate_lift_skips_features_with_no_coverage(scored):
    from ziggy.evaluate import univariate_lift

    d = scored.assign(constant=1.0, mostly_missing=np.nan)
    u = univariate_lift(d, ["informative", "constant", "mostly_missing"], "label", "mag", 20)
    assert set(u["feature"]) == {"informative"}
