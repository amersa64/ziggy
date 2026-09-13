"""The load-bearing test: features must not change when the future is removed.

Every feature is recomputed on a panel truncated at session D and compared to
the same feature computed on the full panel. If any feature reads forward in
time -- a centred window, a global normalisation, a backward-filled series that
propagates from the future -- the two disagree and this test fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ziggy.features.price import compute_breadth, compute_market_context, compute_price_features
from tests.conftest import make_panel

CUTOFFS = [-1, -20, -60]


@pytest.mark.parametrize("offset", CUTOFFS)
def test_price_features_are_identical_when_the_future_is_truncated(panel, bench, sessions, offset):
    cutoff = sessions[offset - 1]
    full = compute_price_features(panel, bench)
    trunc = compute_price_features(
        panel[panel["date"] <= cutoff], bench[bench["date"] <= cutoff]
    )
    a = full[full["session"] == cutoff].set_index("ticker").sort_index()
    b = trunc[trunc["session"] == cutoff].set_index("ticker").sort_index()
    assert not a.empty and len(a) == len(b)
    cols = [c for c in a.columns if c != "session"]
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        assert np.allclose(x[~both_nan], y[~both_nan], rtol=1e-5, atol=1e-7, equal_nan=True), (
            f"feature {c!r} changed when the future was removed -- it is reading forward"
        )


def test_market_context_is_causal(bench, sessions):
    cutoff = sessions[-30]
    full = compute_market_context(bench)
    trunc = compute_market_context(bench[bench["date"] <= cutoff])
    a = full[full["session"] == cutoff].drop(columns=["session"]).iloc[0]
    b = trunc[trunc["session"] == cutoff].drop(columns=["session"]).iloc[0]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6)


def test_breadth_uses_only_the_same_day_cross_section(panel, bench, sessions):
    cutoff = sessions[-40]
    pf = compute_price_features(panel, bench)
    full = compute_breadth(pf)
    trunc = compute_breadth(pf[pf["session"] <= cutoff])
    a = full[full["session"] == cutoff].drop(columns=["session"]).iloc[0]
    b = trunc[trunc["session"] == cutoff].drop(columns=["session"]).iloc[0]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6)


def test_universe_membership_is_unchanged_by_future_data(panel, sessions, cfg):
    from ziggy.universe import build_price_panel, build_universe

    cutoff = sessions[-25]
    full = build_universe(build_price_panel(panel, sessions), cfg, sessions)
    sub = pd.DatetimeIndex(sessions[sessions <= cutoff])
    trunc = build_universe(build_price_panel(panel[panel["date"] <= cutoff], sub), cfg, sub)
    a = set(full[full["session"] == cutoff]["ticker"])
    b = set(trunc[trunc["session"] == cutoff]["ticker"])
    assert a == b


def test_a_deliberately_leaky_feature_is_caught(panel, bench, sessions):
    """Control: the test above must be able to fail."""
    cutoff = sessions[-20]

    def leaky(p):
        px = p.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
        # A centred window silently uses tomorrow.
        return px.rolling(5, center=True, min_periods=1).mean().stack(future_stack=True)

    full = leaky(panel)
    trunc = leaky(panel[panel["date"] <= cutoff])
    a = full.loc[cutoff].sort_index()
    b = trunc.loc[cutoff].sort_index()
    assert not np.allclose(a.to_numpy(), b.to_numpy()), (
        "the truncation harness failed to notice an obviously leaky feature"
    )
