from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ziggy.features.assemble import cross_sectional_rank, feature_columns
from ziggy.rank.baselines import BASELINES, score_baseline
from ziggy.rank.models import DeterministicScorer, build_ranker, walk_forward_scores
from ziggy.splits import make_splits


@pytest.fixture
def sessions_idx():
    return pd.DatetimeIndex(pd.bdate_range("2021-01-04", "2025-12-31"))


def test_splits_are_chronological_and_disjoint(sessions_idx, cfg):
    s = make_splits(sessions_idx, cfg)
    assert s.train.max() < s.validation.min() < s.validation.max() < s.holdout.min()
    for a, b in ((s.train, s.validation), (s.validation, s.holdout), (s.train, s.holdout)):
        assert len(set(a) & set(b)) == 0


def test_embargo_gap_is_at_least_the_longest_label_horizon(sessions_idx, cfg):
    s = make_splits(sessions_idx, cfg)
    emb = int(cfg.splits["embargo_sessions"])
    pos = {d: i for i, d in enumerate(sessions_idx)}
    assert pos[s.validation.min()] - pos[s.train.max()] > emb - 1
    assert emb >= max(cfg.labels["horizons"])


def test_split_fractions_are_respected(sessions_idx, cfg):
    s = make_splits(sessions_idx, cfg)
    n = len(sessions_idx)
    assert abs(len(s.holdout) / n - cfg.splits["holdout_frac"]) < 0.03


@pytest.fixture
def toy_matrix():
    rng = np.random.default_rng(0)
    rows = []
    for s in pd.bdate_range("2021-01-04", "2023-12-29"):
        n = 120
        rows.append(pd.DataFrame({
            "session": s, "ticker": [f"T{i:03d}" for i in range(n)],
            "sec_n_high_salience": rng.poisson(0.2, n).astype(float),
            "sec_n_filings": rng.poisson(0.5, n).astype(float),
            "sec_item_results": (rng.random(n) < 0.05).astype(float),
            "volume_z_20d": rng.normal(size=n), "abs_ret_1d": np.abs(rng.normal(0, 0.02, n)),
            "vol_21d": np.abs(rng.normal(0.4, 0.1, n)), "log_adv20_universe": rng.normal(16, 1.5, n),
            "has_disclosure": (rng.random(n) < 0.3).astype(float),
        }))
    d = pd.concat(rows, ignore_index=True)
    latent = 0.9 * d["sec_n_high_salience"] + 0.5 * d["volume_z_20d"] - 0.3 * d["log_adv20_universe"] / 16
    # Signal-to-noise chosen so a linear model can find it: the point of the
    # test is that fitting works, not how faint a signal sklearn can detect.
    d["consequence_magnitude"] = np.abs(rng.normal(0, 0.03, len(d)) + 0.04 * latent)
    d["label_cs_q90"] = (d.groupby("session")["consequence_magnitude"].rank(pct=True) > 0.9).astype(float)
    return d


def test_feature_columns_never_include_labels(toy_matrix):
    cols = feature_columns(toy_matrix)
    assert "consequence_magnitude" not in cols and "label_cs_q90" not in cols
    assert "sec_n_filings" in cols


def test_cross_sectional_rank_is_within_day_only():
    d = pd.DataFrame({"session": ["a"] * 3 + ["b"] * 3, "x": [1.0, 2, 3, 100, 200, 300]})
    r = cross_sectional_rank(d, ["x"])["x"].to_numpy()
    assert np.allclose(r[:3], r[3:])       # levels differ, ranks do not


def test_deterministic_scorer_needs_no_fitting_and_is_reproducible(toy_matrix):
    s = DeterministicScorer()
    cols = feature_columns(toy_matrix)
    a = s.score(toy_matrix, cols)
    b = DeterministicScorer().score(toy_matrix, cols)
    assert np.allclose(a, b)
    assert not s.needs_fit


def test_fitted_rankers_learn_the_planted_signal(toy_matrix):
    from ziggy.evaluate import per_session_metrics

    cols = feature_columns(toy_matrix)
    train = toy_matrix[toy_matrix["session"] < "2023-01-01"]
    test = toy_matrix[toy_matrix["session"] >= "2023-01-01"].copy()
    for name in ("logistic", "gbm"):
        r = build_ranker(name, seed=0).fit(train, cols, "label_cs_q90")
        test["score"] = r.score(test, cols).values
        lift = per_session_metrics(test, "score", "label_cs_q90", "consequence_magnitude", [20])["lift"].mean()
        assert lift > 1.2, f"{name} failed to learn a planted signal (lift {lift:.2f})"


def test_walk_forward_never_trains_on_the_year_it_scores(toy_matrix):
    cols = feature_columns(toy_matrix)
    ev = pd.DatetimeIndex(toy_matrix[toy_matrix["session"] >= "2022-01-01"]["session"].unique())
    _, rows = walk_forward_scores("logistic", toy_matrix, cols, "label_cs_q90", ev,
                                  min_train_sessions=100, seed=0)
    for r in rows:
        if r["status"] == "fitted":
            assert pd.Timestamp(r["train_end"]) < pd.Timestamp(year=r["year"], month=1, day=1)


def test_every_baseline_produces_a_usable_score(toy_matrix):
    for name in BASELINES:
        s = score_baseline(name, toy_matrix, seed=0)
        assert len(s) == len(toy_matrix)
        assert np.isfinite(s.to_numpy()).all()
