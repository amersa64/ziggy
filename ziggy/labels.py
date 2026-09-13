"""Forward "consequential activity" labels.

This is the only module allowed to look forward in time, and it is never used to
build a feature. Definitions:

* **Entry** is the open of the session *after* the snapshot. The 20:00 ET
  snapshot cannot be traded; the earliest action is the next open.
* **Exit** is the close of session ``t+h``.
* **Excess** subtracts the benchmark over the identical window, and a
  beta-adjusted variant uses the beta that was estimable at the snapshot.

Consequential-activity definitions (all reported; ``cs_q90`` is primary):

``cs_q90``
    ``|excess move|`` in the top decile of that snapshot's own cross-section.
    Base rate is exactly 10% by construction, which makes "lift" mean precisely
    "how many times better than a coin flip over the same candidate set", with
    no regime drift in the denominator.
``abs``
    ``|excess move|`` above a fixed threshold calibrated on the *training* split
    only. Regime-sensitive on purpose: it answers "did we find real moves",
    not "did we find relatively large ones".
``vol_expansion`` / ``volume_shock``
    Realized volatility or volume over the forward window relative to the name's
    own trailing behaviour. Consequence without requiring a directional move.

Delisting is handled rather than dropped: if a name stops trading inside the
window, the exit uses its last observed close and ``label_truncated`` is set.
Silently dropping those rows would discard exactly the most consequential
outcomes (bankruptcies, cash acquisitions).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _pivot(panel: pd.DataFrame, field: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    m = panel.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    return m.reindex(sessions).sort_index()


def compute_forward_returns(
    panel: pd.DataFrame,
    bench: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    horizons: list[int],
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Forward returns from next-session open to close of ``t+h``."""
    close = _pivot(panel, "close", sessions)
    adj = _pivot(panel, "adj_close", sessions)
    openp = _pivot(panel, "open", sessions)
    vol = _pivot(panel, "volume", sessions)

    # Put the open on the same adjustment basis as adj_close.
    factor = (adj / close.replace(0, np.nan))
    adj_open = openp * factor

    bpx = bench.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    bpx = bpx.reindex(sessions).ffill()
    bopen = bench.pivot_table(index="date", columns="ticker", values="open", aggfunc="last")
    bclose = bench.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    bfac = (bpx / bclose.reindex(sessions).replace(0, np.nan))
    badj_open = bopen.reindex(sessions) * bfac
    bmk_close = bpx[benchmark] if benchmark in bpx.columns else pd.Series(np.nan, index=sessions)
    bmk_open = badj_open[benchmark] if benchmark in badj_open.columns else pd.Series(np.nan, index=sessions)

    entry = adj_open.shift(-1)
    bentry = bmk_open.shift(-1)
    logret = np.log(adj).diff()
    # Carrying the last observed close forward gives the exit price for a name
    # that stops trading inside the window. Rows whose *entry* never happens
    # (the name's last bar is the snapshot itself) stay NaN and are excluded.
    adj_held = adj.ffill()

    out = {}
    for h in horizons:
        clean_exit = adj.shift(-h)
        held_exit = adj_held.shift(-h)
        exitp = clean_exit.where(clean_exit.notna(), held_exit)
        truncated = clean_exit.isna() & held_exit.notna() & entry.notna()

        r = exitp / entry.replace(0, np.nan) - 1.0
        br = bmk_close.shift(-h) / bentry.replace(0, np.nan) - 1.0
        out[f"fwd_ret_{h}d"] = r
        out[f"fwd_exret_{h}d"] = r.sub(br, axis=0)
        out[f"fwd_truncated_{h}d"] = truncated.astype(float)
        out[f"fwd_no_exit_{h}d"] = (exitp.isna() & entry.notna()).astype(float)

        # Realized behaviour over the forward window (direction-free consequence).
        if h == 1:
            fwd_vol = logret.shift(-1).abs() * np.sqrt(252)
        else:
            fwd_vol = logret.shift(-h).rolling(h, min_periods=max(2, h // 2)).std() * np.sqrt(252)
        out[f"fwd_vol_{h}d"] = fwd_vol
        out[f"fwd_volume_{h}d"] = vol.shift(-h).rolling(h, min_periods=1).mean()

    frames = []
    for name, mat in out.items():
        s = mat.stack(future_stack=True)
        s.name = name
        frames.append(s)
    df = pd.concat(frames, axis=1).reset_index()
    df = df.rename(columns={df.columns[0]: "session", df.columns[1]: "ticker"})
    num = [c for c in df.columns if c not in ("session", "ticker")]
    df[num] = df[num].astype("float32")
    return df


def attach_beta_adjusted(labels: pd.DataFrame, features: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Beta-adjusted excess, using the beta estimable at the snapshot."""
    if "beta_126d" not in features.columns:
        return labels
    b = features[["session", "ticker", "beta_126d"]]
    out = labels.merge(b, on=["session", "ticker"], how="left")
    for h in horizons:
        mkt = out[f"fwd_ret_{h}d"] - out[f"fwd_exret_{h}d"]  # recover benchmark return
        beta = out["beta_126d"].clip(-3, 3).fillna(1.0)
        out[f"fwd_abret_{h}d"] = (out[f"fwd_ret_{h}d"] - beta * mkt).astype("float32")
    return out.drop(columns=["beta_126d"])


def add_consequence_labels(
    df: pd.DataFrame,
    cfg,
    trailing_vol: pd.DataFrame | None = None,
    abs_threshold: float | None = None,
) -> pd.DataFrame:
    """Attach binary consequence labels and the continuous magnitude."""
    lab = cfg.labels
    h = int(lab["primary_horizon"])
    q = float(lab["consequential"]["cross_sectional_quantile"])
    out = df.copy()

    col = f"fwd_exret_{h}d"
    out["consequence_magnitude"] = out[col].abs().astype("float32")

    # Cross-sectional top decile within each snapshot.
    g = out.groupby("session", observed=True)["consequence_magnitude"]
    out["label_cs_q90"] = (g.rank(pct=True, na_option="keep") > q).astype("float32")
    out.loc[out["consequence_magnitude"].isna(), "label_cs_q90"] = np.nan

    # Absolute threshold (calibrated on train elsewhere and passed in).
    if abs_threshold is not None:
        out["label_abs"] = (out["consequence_magnitude"] >= abs_threshold).astype("float32")
        out.loc[out["consequence_magnitude"].isna(), "label_abs"] = np.nan

    # Volatility expansion relative to the name's own trailing volatility.
    if trailing_vol is not None and "vol_21d" in trailing_vol.columns:
        # The caller may already carry vol_21d as a feature; join under a private
        # name so the merge cannot produce vol_21d_x / vol_21d_y.
        tv = trailing_vol[["session", "ticker", "vol_21d"]].rename(columns={"vol_21d": "_trailing_vol_21d"})
        out = out.merge(tv, on=["session", "ticker"], how="left")
        ratio = out[f"fwd_vol_{h}d"] / out["_trailing_vol_21d"].replace(0, np.nan)
        out["fwd_vol_ratio"] = ratio.astype("float32")
        out["label_vol_expansion"] = (
            ratio >= float(lab["secondary"]["vol_expansion_ratio"])
        ).astype("float32")
        out.loc[ratio.isna(), "label_vol_expansion"] = np.nan
        out = out.drop(columns=["_trailing_vol_21d"])

    return out


def calibrate_abs_threshold(train_df: pd.DataFrame, quantile: float = 0.90) -> float:
    """Absolute |excess| threshold, estimated on the training split ONLY."""
    v = train_df["consequence_magnitude"].dropna()
    if v.empty:
        return float("nan")
    return float(np.quantile(v, quantile))
