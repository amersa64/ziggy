"""Macro context joined as of each snapshot.

The join is ``merge_asof`` on ``available_at <= snapshot_ts``. For revisable
series this picks the newest *vintage* the snapshot could have seen, so a 2021
snapshot gets the 2021 print of payrolls, not the number as revised in 2024.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _series_asof(panel: pd.DataFrame, sid: str, snaps: pd.DataFrame) -> pd.DataFrame:
    sub = panel[(panel["series_id"] == sid)].dropna(subset=["value", "available_at"]).copy()
    if sub.empty:
        return pd.DataFrame({"session": snaps["session"], f"macro_{sid}": np.nan})
    sub["available_at"] = pd.to_datetime(sub["available_at"], utc=True)
    # Within one availability instant keep the most recent observation.
    sub = sub.sort_values(["available_at", "date"]).groupby("available_at", as_index=False).tail(1)
    sub = sub.sort_values("available_at")[["available_at", "value"]]
    merged = pd.merge_asof(
        snaps.sort_values("snapshot_ts"),
        sub,
        left_on="snapshot_ts",
        right_on="available_at",
        direction="backward",
    )
    return pd.DataFrame({"session": merged["session"], f"macro_{sid}": merged["value"].values})


def compute_macro_features(macro_panel: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """One row per session with levels, 21-session changes and 252-session z-scores."""
    snaps = snapshots[["session", "snapshot_ts"]].copy()
    snaps["snapshot_ts"] = pd.to_datetime(snaps["snapshot_ts"], utc=True)
    out = snaps[["session"]].copy()
    if macro_panel is None or macro_panel.empty:
        log.warning("no macro panel: macro features will be absent")
        return out
    for sid in sorted(macro_panel["series_id"].unique()):
        s = _series_asof(macro_panel, sid, snaps)
        out = out.merge(s, on="session", how="left")
        col = f"macro_{sid}"
        v = out[col]
        out[f"{col}_chg21"] = v.diff(21)
        mu, sd = v.rolling(252, min_periods=60).mean(), v.rolling(252, min_periods=60).std()
        out[f"{col}_z252"] = (v - mu) / sd.replace(0, np.nan)
    num = [c for c in out.columns if c != "session"]
    out[num] = out[num].astype("float32")
    log.info("macro features: %d sessions x %d columns", len(out), len(num))
    return out
