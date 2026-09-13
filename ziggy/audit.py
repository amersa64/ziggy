"""Leakage, timestamp and survivorship audits.

A point-in-time claim that is not tested is a point-in-time hope. These checks
run as part of the experiment and their output goes into the evidence package
whether it is flattering or not.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def acceptance_timezone_audit(filings: pd.DataFrame, assumed_tz: str = "America/New_York") -> dict:
    """Check the assumption that EDGAR acceptance stamps are Eastern wall-clock.

    EDGAR is open 06:00-22:00 ET. Under the correct interpretation essentially
    all filings fall inside that window and 8-K acceptances pile up just after
    the 16:00 close. Under a wrong (UTC) reading the same mass would appear at
    02:00-03:00 local, i.e. outside EDGAR's operating hours -- which is the
    signature this audit looks for.
    """
    acc = pd.to_datetime(filings["accepted_at"], utc=True).dt.tz_convert(assumed_tz)
    hour = acc.dt.hour
    inside = ((hour >= 6) & (hour < 22)).mean()
    hist = hour.value_counts().sort_index()
    eightk = filings[filings["form_base"] == "8-K"]
    mode_hour = None
    if len(eightk):
        h8 = pd.to_datetime(eightk["accepted_at"], utc=True).dt.tz_convert(assumed_tz).dt.hour
        mode_hour = int(h8.value_counts().idxmax())
    verdict = "consistent_with_eastern" if inside > 0.97 else "SUSPECT"
    return {
        "assumed_timezone": assumed_tz,
        "share_within_edgar_hours_06_22": float(inside),
        "8k_modal_acceptance_hour": mode_hour,
        "hour_histogram": {int(k): int(v) for k, v in hist.items()},
        "verdict": verdict,
        "note": "a UTC misreading would push the modal 8-K hour into the small "
                "hours and drop the in-hours share far below 97%",
    }


def feature_availability_audit(matrix: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Every disclosure feeding a snapshot must have been accepted before it."""
    if events.empty:
        return {"status": "no_events"}
    ev = events.rename(columns={"snapshot_session": "session"})
    snaps = matrix[["session", "snapshot_ts"]].drop_duplicates()
    m = ev.merge(snaps, on="session", how="inner")
    acc = pd.to_datetime(m["last_accepted_at"], utc=True)
    snap = pd.to_datetime(m["snapshot_ts"], utc=True)
    violations = int((acc > snap).sum())
    gap_hours = ((snap - acc).dt.total_seconds() / 3600.0)
    return {
        "events_checked": int(len(m)),
        "violations_disclosure_after_snapshot": violations,
        "min_gap_hours": float(gap_hours.min()),
        "median_gap_hours": float(gap_hours.median()),
        "status": "clean" if violations == 0 else "LEAK",
    }


def forward_column_audit(feature_cols: list[str]) -> dict:
    """No feature name may look like a label. Cheap, and it has caught real bugs."""
    banned = [c for c in feature_cols
              if c.startswith(("fwd_", "label_", "target_", "y_"))
              or "consequence" in c or "future" in c]
    return {"n_features": len(feature_cols), "forbidden_columns_present": banned,
            "status": "clean" if not banned else "LEAK"}


def shuffle_control(
    df: pd.DataFrame, score_col: str, label_col: str, magnitude_col: str, k: int, seed: int = 0
) -> dict:
    """Destroy the score, keep everything else. Lift must fall to ~1.0."""
    from ziggy.evaluate import per_session_metrics

    rng = np.random.default_rng(seed)
    d = df[["session", score_col, label_col, magnitude_col]].dropna().copy()
    real = per_session_metrics(d, score_col, label_col, magnitude_col, [k], seed=seed)
    d["_shuffled"] = d.groupby("session", observed=True)[score_col].transform(
        lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy()
    )
    sh = per_session_metrics(d, "_shuffled", label_col, magnitude_col, [k], seed=seed)
    return {
        "k": k,
        "real_lift": float(real["lift"].mean()),
        "shuffled_score_lift": float(sh["lift"].mean()),
        "status": "clean" if abs(float(sh["lift"].mean()) - 1.0) < 0.12 else "SUSPECT",
    }


def leakage_canary(
    df: pd.DataFrame, feature_cols: list[str], label_col: str, magnitude_col: str,
    k: int, seed: int = 0,
) -> dict:
    """Positive control: deliberately inject a future-derived feature.

    A leak detector that has never fired proves nothing. Here we build a ranker
    that *does* see the future and confirm the evaluation lights up, so that the
    clean result above is a measurement rather than an absence of measurement.
    """
    from ziggy.evaluate import per_session_metrics

    d = df.dropna(subset=[label_col, magnitude_col]).copy()
    rng = np.random.default_rng(seed)
    # A noisy view of the outcome: obviously illegal, used only as a control.
    d["_cheating_score"] = d[magnitude_col] + rng.normal(0, d[magnitude_col].std() * 0.5, len(d))
    m = per_session_metrics(d, "_cheating_score", label_col, magnitude_col, [k], seed=seed)
    lift = float(m["lift"].mean())
    return {
        "k": k, "cheating_lift": lift,
        "status": "detector_works" if lift > 3.0 else "DETECTOR_BLIND",
        "note": "this ranker is allowed to see the label; it is never used in the experiment",
    }


def label_coverage_audit(labelled: pd.DataFrame, horizon: int) -> dict:
    tot = len(labelled)
    mag = labelled["consequence_magnitude"]
    return {
        "rows": int(tot),
        "rows_with_label": int(mag.notna().sum()),
        "label_coverage": float(mag.notna().mean()) if tot else np.nan,
        "rows_truncated_by_delisting": int(labelled.get(f"fwd_truncated_{horizon}d", pd.Series(dtype=float)).fillna(0).sum()),
        "rows_without_entry_price": int(tot - mag.notna().sum()),
    }


def run_all(
    filings: pd.DataFrame,
    events: pd.DataFrame,
    matrix: pd.DataFrame,
    feature_cols: list[str],
    scored: pd.DataFrame | None,
    score_col: str,
    label_col: str,
    magnitude_col: str,
    k: int,
    horizon: int,
    seed: int = 0,
) -> dict:
    out = {
        "acceptance_timezone": acceptance_timezone_audit(filings),
        "feature_availability": feature_availability_audit(matrix, events),
        "forward_columns": forward_column_audit(feature_cols),
        "label_coverage": label_coverage_audit(matrix, horizon) if "consequence_magnitude" in matrix else {},
    }
    if scored is not None and score_col in scored.columns:
        out["shuffle_control"] = shuffle_control(scored, score_col, label_col, magnitude_col, k, seed)
        out["leakage_canary"] = leakage_canary(scored, feature_cols, label_col, magnitude_col, k, seed)
    return out
