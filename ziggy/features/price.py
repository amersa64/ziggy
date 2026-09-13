"""Price, volume and flow features.

Everything here is computed from bars up to *and including* session D, which is
legitimate: the 20:00 ET snapshot is after the 16:00 close, so session D's own
bar is public information. Nothing reads D+1.

Wide (date x ticker) matrices are used instead of groupby-apply because the
panel is ~2-3M rows and the rolling statistics are identical per column.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


def _pivot(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").sort_index()


def _roll_mean(x: pd.DataFrame, w: int, minp: int | None = None) -> pd.DataFrame:
    return x.rolling(w, min_periods=minp or max(2, w // 2)).mean()


def _roll_std(x: pd.DataFrame, w: int, minp: int | None = None) -> pd.DataFrame:
    return x.rolling(w, min_periods=minp or max(3, w // 2)).std()


def compute_price_features(
    panel: pd.DataFrame, bench: pd.DataFrame, benchmark: str = "SPY"
) -> pd.DataFrame:
    """Long frame of per-(session, ticker) price features.

    ``panel`` is the tidy OHLCV panel; ``bench`` the same for benchmark symbols.
    """
    px = _pivot(panel, "adj_close")
    close = _pivot(panel, "close")
    openp = _pivot(panel, "open")
    high = _pivot(panel, "high")
    low = _pivot(panel, "low")
    vol = _pivot(panel, "volume")
    dv = close * vol

    ret = px.pct_change(fill_method=None)
    logret = np.log(px).diff()

    bpx = bench.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    bpx = bpx.reindex(px.index).ffill()
    bret = bpx.pct_change(fill_method=None)
    mkt = bret[benchmark] if benchmark in bret.columns else pd.Series(0.0, index=px.index)

    feats: dict[str, pd.DataFrame] = {}

    # -- returns and relative strength -----------------------------------------
    for h in (1, 5, 21, 63, 252):
        r = px.pct_change(h, fill_method=None)
        feats[f"ret_{h}d"] = r
        br = (1.0 + mkt).rolling(h, min_periods=h).apply(np.prod, raw=True) - 1.0
        feats[f"exret_{h}d"] = r.sub(br, axis=0)
    feats["abs_ret_1d"] = ret.abs()
    feats["abs_exret_1d"] = feats["exret_1d"].abs()
    feats["mom_12_1"] = feats["ret_252d"] - feats["ret_21d"]
    feats["accel_5_21"] = feats["ret_5d"] - feats["ret_21d"] / 4.2

    # -- volatility -------------------------------------------------------------
    vol21 = _roll_std(logret, 21) * np.sqrt(TRADING_DAYS)
    vol63 = _roll_std(logret, 63) * np.sqrt(TRADING_DAYS)
    vol5 = _roll_std(logret, 5, minp=4) * np.sqrt(TRADING_DAYS)
    feats["vol_21d"] = vol21
    feats["vol_63d"] = vol63
    feats["vol_ratio_5_21"] = vol5 / vol21.replace(0, np.nan)
    feats["vol_ratio_21_63"] = vol21 / vol63.replace(0, np.nan)
    feats["vol_of_vol"] = _roll_std(vol21, 63)
    # Parkinson range volatility: uses the day's high/low, robust to gaps.
    hl = np.log(high / low.replace(0, np.nan))
    feats["parkinson_21d"] = np.sqrt(_roll_mean(hl**2, 21) / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)
    feats["range_pct_1d"] = ((high - low) / close.replace(0, np.nan))
    feats["gap_pct"] = openp / close.shift(1).replace(0, np.nan) - 1.0
    feats["intraday_ret"] = close / openp.replace(0, np.nan) - 1.0

    # -- volume -----------------------------------------------------------------
    logvol = np.log1p(vol)
    v_mean, v_std = _roll_mean(logvol.shift(1), 20), _roll_std(logvol.shift(1), 20)
    feats["volume_z_20d"] = (logvol - v_mean) / v_std.replace(0, np.nan)
    adv20 = dv.rolling(20, min_periods=10).median()
    adv60 = dv.rolling(60, min_periods=30).median()
    feats["log_adv20"] = np.log1p(adv20)
    feats["adv_trend_20_60"] = adv20 / adv60.replace(0, np.nan)
    feats["dollar_volume_ratio"] = dv / adv20.replace(0, np.nan)
    feats["turnover_accel"] = dv.rolling(5, min_periods=3).mean() / adv20.replace(0, np.nan)
    # Amihud: price impact per dollar traded. Higher = thinner, more jumpy.
    feats["amihud_21d"] = _roll_mean(ret.abs() / dv.replace(0, np.nan), 21) * 1e9

    # -- position in range ------------------------------------------------------
    hi252 = px.rolling(252, min_periods=120).max()
    lo252 = px.rolling(252, min_periods=120).min()
    feats["dist_52w_high"] = px / hi252.replace(0, np.nan) - 1.0
    feats["dist_52w_low"] = px / lo252.replace(0, np.nan) - 1.0
    feats["drawdown_63d"] = px / px.rolling(63, min_periods=30).max().replace(0, np.nan) - 1.0
    feats["pct_up_days_21d"] = (ret > 0).rolling(21, min_periods=10).mean()

    # -- market sensitivity -----------------------------------------------------
    m = mkt.reindex(px.index)
    cov = logret.rolling(126, min_periods=60).cov(m)
    var = m.rolling(126, min_periods=60).var()
    beta = cov.div(var.replace(0, np.nan), axis=0)
    feats["beta_126d"] = beta
    idio = logret.sub(beta.mul(m, axis=0))
    feats["idio_vol_63d"] = _roll_std(idio, 63) * np.sqrt(TRADING_DAYS)
    feats["idio_ret_5d"] = idio.rolling(5, min_periods=4).sum()
    feats["abs_idio_ret_5d"] = feats["idio_ret_5d"].abs()

    # -- assemble long ----------------------------------------------------------
    frames = []
    for name, mat in feats.items():
        s = mat.stack(future_stack=True)
        s.name = name
        frames.append(s)
    out = pd.concat(frames, axis=1).reset_index()
    out = out.rename(columns={"date": "session", "level_1": "ticker"})
    if "ticker" not in out.columns:
        out = out.rename(columns={out.columns[1]: "ticker"})
    num = [c for c in out.columns if c not in ("session", "ticker")]
    out[num] = out[num].astype("float32")
    log.info("price features: %d rows x %d features", len(out), len(num))
    return out


def compute_market_context(bench: pd.DataFrame, vix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Whole-market regime context, one row per session."""
    bpx = bench.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    out = pd.DataFrame(index=bpx.index)
    for sym in bpx.columns:
        s = bpx[sym]
        r = s.pct_change(fill_method=None)
        key = sym.lower().lstrip("^")
        out[f"mkt_{key}_ret_1d"] = r
        out[f"mkt_{key}_ret_5d"] = s.pct_change(5, fill_method=None)
        out[f"mkt_{key}_ret_21d"] = s.pct_change(21, fill_method=None)
        out[f"mkt_{key}_vol_21d"] = r.rolling(21, min_periods=10).std() * np.sqrt(TRADING_DAYS)
        out[f"mkt_{key}_drawdown"] = s / s.rolling(252, min_periods=60).max() - 1.0
    if vix is not None and not vix.empty:
        v = vix.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
        col = v.columns[0]
        s = v[col].reindex(out.index).ffill()
        out["vix_level"] = s
        out["vix_chg_5d"] = s.pct_change(5, fill_method=None)
        out["vix_z_63d"] = (s - s.rolling(63, min_periods=30).mean()) / s.rolling(63, min_periods=30).std()
    return out.reset_index().rename(columns={"date": "session"})


def compute_breadth(panel_features: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional dispersion and breadth, computed within each session."""
    g = panel_features.groupby("session", observed=True)
    out = pd.DataFrame(
        {
            "breadth_pct_above_0_21d": g["ret_21d"].apply(lambda s: (s > 0).mean()),
            "xs_dispersion_ret_5d": g["ret_5d"].std(),
            "xs_median_vol_21d": g["vol_21d"].median(),
            "xs_mean_abs_ret_1d": g["abs_ret_1d"].mean(),
        }
    ).reset_index()
    return out
