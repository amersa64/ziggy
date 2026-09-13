from __future__ import annotations

import numpy as np
import pandas as pd

from ziggy.labels import (add_consequence_labels, calibrate_abs_threshold,
                          compute_forward_returns)
from tests.conftest import make_panel


def _fixed_panel(sessions, tickers, closes):
    rows = []
    for t in tickers:
        rows.append(pd.DataFrame({
            "date": sessions, "ticker": t, "open": closes, "high": closes,
            "low": closes, "close": closes, "adj_close": closes, "volume": 1e6,
        }))
    return pd.concat(rows, ignore_index=True)


def test_forward_return_is_next_open_to_close_of_t_plus_h():
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=10))
    prices = np.array([100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    panel = _fixed_panel(sessions, ["A"], prices)
    bench = _fixed_panel(sessions, ["SPY"], np.full(10, 50.0))
    lab = compute_forward_returns(panel, bench, sessions, [3], "SPY")
    row = lab[(lab.ticker == "A") & (lab.session == sessions[0])].iloc[0]
    # entry = open of session 1 (=101), exit = close of session 3 (=103)
    assert row["fwd_ret_3d"] == np.float32(103 / 101 - 1)
    # a flat benchmark leaves the excess equal to the raw return
    assert abs(row["fwd_exret_3d"] - row["fwd_ret_3d"]) < 1e-6


def test_snapshot_close_is_never_the_entry_price():
    """Entering at the snapshot's own close would be trading on the snapshot."""
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=8))
    prices = np.array([100.0, 200, 201, 202, 203, 204, 205, 206])
    panel = _fixed_panel(sessions, ["A"], prices)
    bench = _fixed_panel(sessions, ["SPY"], np.full(8, 50.0))
    lab = compute_forward_returns(panel, bench, sessions, [1], "SPY")
    row = lab[(lab.ticker == "A") & (lab.session == sessions[0])].iloc[0]
    # The 100 -> 200 jump happens between the snapshot and the entry, so it must
    # NOT appear in the forward return.
    assert abs(float(row["fwd_ret_1d"])) < 1e-6


def test_delisting_truncates_rather_than_dropping():
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=20))
    panel = make_panel(["A", "DEAD"], sessions, seed=5, stop={"DEAD": 12})
    bench = make_panel(["SPY"], sessions, seed=6)
    lab = compute_forward_returns(panel, bench, sessions, [5], "SPY")
    dead = lab[lab.ticker == "DEAD"]
    trunc = dead[dead["fwd_truncated_5d"] == 1.0]
    assert len(trunc) > 0, "a name that stops trading must still produce labels"
    assert dead["fwd_ret_5d"].notna().sum() >= len(trunc)


def test_cross_sectional_label_has_the_designed_base_rate(cfg):
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=60))
    panel = make_panel([f"T{i:03d}" for i in range(200)], sessions, seed=3)
    bench = make_panel(["SPY"], sessions, seed=4)
    lab = compute_forward_returns(panel, bench, sessions, [5], "SPY")
    out = add_consequence_labels(lab, cfg)
    rate = float(out["label_cs_q90"].mean())
    assert 0.08 < rate < 0.12, rate


def test_absolute_threshold_uses_only_the_training_rows():
    df = pd.DataFrame({"consequence_magnitude": np.concatenate([np.full(100, 0.01), np.full(100, 10.0)])})
    train_only = df.iloc[:100]
    assert calibrate_abs_threshold(train_only, 0.9) < 0.02
    assert calibrate_abs_threshold(df, 0.9) > 1.0


def test_magnitude_is_absolute_excess_not_signed(cfg):
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=40))
    panel = make_panel([f"T{i}" for i in range(30)], sessions, seed=8)
    bench = make_panel(["SPY"], sessions, seed=9)
    out = add_consequence_labels(compute_forward_returns(panel, bench, sessions, [5], "SPY"), cfg)
    assert (out["consequence_magnitude"].dropna() >= 0).all()


def test_volnorm_label_is_not_satisfiable_by_ranking_on_volatility(cfg):
    """The normalised label must break the 'volatile names move more' shortcut.

    The panel is built with deliberately heterogeneous per-name volatility, which
    is precisely the free lunch an absolute-magnitude label hands to a ranker
    that simply sorts on trailing sigma.
    """
    from ziggy.evaluate import per_session_metrics
    from ziggy.features.price import compute_price_features

    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=200))
    rng = np.random.default_rng(21)
    vols = np.linspace(0.008, 0.055, 150)          # 10x spread in daily sigma
    rows = []
    for i, v in enumerate(vols):
        p = 100 * np.exp(np.cumsum(rng.normal(0, v, len(sessions))))
        rows.append(pd.DataFrame({
            "date": sessions, "ticker": f"T{i:03d}", "open": p, "high": p * 1.01,
            "low": p * 0.99, "close": p, "adj_close": p, "volume": 3e6,
        }))
    panel = pd.concat(rows, ignore_index=True)
    bench = make_panel(["SPY"], sessions, seed=22)

    pf = compute_price_features(panel, bench)
    lab = compute_forward_returns(panel, bench, sessions, [5], "SPY")
    out = add_consequence_labels(lab, cfg, trailing_vol=pf[["session", "ticker", "vol_21d"]])
    out = out.merge(pf[["session", "ticker", "vol_21d"]], on=["session", "ticker"], how="left")
    out = out.dropna(subset=["vol_21d", "consequence_magnitude", "consequence_magnitude_volnorm"])

    raw = float(per_session_metrics(out, "vol_21d", "label_cs_q90",
                                    "consequence_magnitude", [20])["lift"].mean())
    norm = float(per_session_metrics(out, "vol_21d", "label_cs_q90_volnorm",
                                     "consequence_magnitude_volnorm", [20])["lift"].mean())
    # Sorting on sigma buys a large edge on the absolute label and *less than
    # nothing* once the label is expressed in units of that same sigma: trailing
    # volatility mean-reverts, so the highest-sigma names tend to move less than
    # their own recent volatility implied.
    assert raw > 1.8, raw
    assert norm < 0.8, norm


def test_volnorm_label_can_be_derived_from_a_built_matrix(cfg):
    from ziggy.labels import add_volnorm_label

    df = pd.DataFrame({
        "session": ["a"] * 20, "consequence_magnitude": np.linspace(0.01, 0.2, 20),
        "vol_21d": np.full(20, 0.4),
    })
    out = add_volnorm_label(df, cfg)
    assert out["label_cs_q90_volnorm"].sum() == 2
    assert out["consequence_magnitude_volnorm"].iloc[-1] > out["consequence_magnitude_volnorm"].iloc[0]
