"""Assemble the candidate matrix.

One row per (snapshot, ticker) in the point-in-time universe, carrying every
feature the ranker is allowed to see. The ``available_at`` bookkeeping travels
with the matrix so :func:`ziggy.store.assert_pit` can verify the whole thing
rather than trusting the construction code.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Features whose absence genuinely means "zero", versus those where absence
# means "not applicable" and must stay missing.
ZERO_FILL_PREFIXES = ("sec_n_", "sec_item_", "sec_filings_", "sec_8k_", "sec_form4_",
                      "sec_salient_", "insider_")
SENTINEL_FILL = {"sec_sessions_since_filings": 999.0, "sec_sessions_since_8k": 999.0,
                 "sec_days_since_same_form": 999.0, "sec_days_since_any": 999.0}

ID_COLS = ["session", "ticker", "snapshot_ts"]


def build_candidate_matrix(
    universe: pd.DataFrame,
    snapshots: pd.DataFrame,
    price_features: pd.DataFrame,
    grid_sec: pd.DataFrame,
    event_features: pd.DataFrame,
    insider_features: pd.DataFrame,
    text_features: pd.DataFrame | None,
    market_context: pd.DataFrame,
    breadth: pd.DataFrame,
    macro_features: pd.DataFrame,
) -> pd.DataFrame:
    df = universe[["session", "ticker", "adv20", "liq_rank", "close"]].copy()
    df = df.merge(snapshots[["session", "snapshot_ts"]], on="session", how="left")

    df = df.merge(price_features, on=["session", "ticker"], how="left")
    if grid_sec is not None and len(grid_sec):
        df = df.merge(grid_sec, on=["session", "ticker"], how="left")
    if insider_features is not None and len(insider_features):
        df = df.merge(insider_features, on=["session", "ticker"], how="left")

    if event_features is not None and len(event_features):
        ev = event_features.rename(columns={"snapshot_session": "session"})
        ev = ev.drop(columns=[c for c in ("available_at",) if c in ev.columns])
        df = df.merge(ev, on=["session", "ticker"], how="left")
    if text_features is not None and len(text_features):
        tf = text_features.rename(columns={"snapshot_session": "session"})
        df = df.merge(tf, on=["session", "ticker"], how="left")

    for extra in (market_context, breadth, macro_features):
        if extra is not None and len(extra):
            df = df.merge(extra, on="session", how="left")

    # Missing-value policy, made explicit, and applied column-block at a time:
    # assigning ~170 columns one by one fragments the frame badly at this size.
    sentinel_cols = [c for c in df.columns if c in SENTINEL_FILL]
    zero_cols = [c for c in df.columns
                 if c not in SENTINEL_FILL and any(c.startswith(p) for p in ZERO_FILL_PREFIXES)]
    if sentinel_cols:
        df[sentinel_cols] = df[sentinel_cols].fillna(
            {c: SENTINEL_FILL[c] for c in sentinel_cols}
        )
    if zero_cols:
        df[zero_cols] = df[zero_cols].fillna(0.0)

    df = df.assign(
        has_disclosure=(df.get("sec_n_filings", pd.Series(0.0, index=df.index)) > 0).astype("float32"),
        log_adv20_universe=np.log1p(df["adv20"]).astype("float32"),
    )

    num = [c for c in df.columns if c not in ("session", "ticker", "snapshot_ts")]
    df = df.astype({c: "float32" for c in num})
    df = df.sort_values(["session", "ticker"]).reset_index(drop=True)
    log.info("candidate matrix: %d rows x %d columns", len(df), df.shape[1])
    return df


def feature_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Columns the ranker may consume. Label and identity columns are excluded."""
    exclude = (exclude or set()) | {
        "session", "ticker", "snapshot_ts", "adv20", "close", "available_at",
    }
    # Kept deliberately in lockstep with ziggy.audit.forward_column_audit, which
    # re-checks the result and aborts the run if anything label-shaped slips in.
    banned_prefix = ("fwd_", "label_", "y_", "target_")
    banned_substr = ("consequence", "future")
    cols = []
    for c in df.columns:
        if c in exclude or any(c.startswith(p) for p in banned_prefix):
            continue
        if any(b in c for b in banned_substr):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def cross_sectional_rank(df: pd.DataFrame, cols: list[str], group: str = "session") -> pd.DataFrame:
    """Within-session percentile ranks.

    Using the same-day cross-section is point-in-time safe: every value in the
    comparison set was knowable at that same snapshot. It also removes market-wide
    level shifts, which is what makes a single model usable across regimes.
    """
    g = df.groupby(group, observed=True)
    out = {}
    for c in cols:
        out[c] = g[c].rank(pct=True, na_option="keep").astype("float32")
    ranked = pd.DataFrame(out, index=df.index)
    return ranked.fillna(0.5)
