"""Disclosure features.

Three layers, in increasing order of how much work they cost:

1. **What was disclosed tonight** -- form mix, 8-K item families, submission
   size, how far after the close it landed. A filing accepted at 21:40 ET is a
   different animal from one accepted at 08:05 ET.
2. **How unusual this disclosure is for this issuer** -- size relative to the
   same issuer's previous filing of the same form, gap since the last one,
   filing intensity over the trailing quarter. "AAPL filed an 8-K" is not
   information; "AAPL filed its first non-scheduled 8-K in seven months and it
   is three times the length of the last one" might be.
3. **Insider behaviour** -- open-market purchases and sales from the quarterly
   Form 345 structured datasets, scaled by the name's own dollar volume.

Every trailing quantity is computed on the snapshot grid with strictly past
windows, so a value at session D never contains D+1.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ziggy.providers.sec_edgar import HIGH_SALIENCE_FAMILIES, item_family

log = logging.getLogger(__name__)

# Item families kept as explicit indicator features. Chosen for economic
# salience and non-trivial frequency, not by looking at outcomes.
TRACKED_FAMILIES = [
    "results", "ma_assets", "executive_change", "material_agreement",
    "impairment", "restructuring_costs", "non_reliance", "auditor_change",
    "control_change", "listing_compliance", "debt_obligation", "debt_acceleration",
    "unregistered_sale", "bankruptcy", "cybersecurity", "reg_fd", "other_events",
    "shareholder_vote", "security_holder_rights",
]


def _wide(df: pd.DataFrame, sessions: pd.DatetimeIndex, tickers: pd.Index,
          value: str, fill: float = 0.0) -> pd.DataFrame:
    m = df.pivot_table(index="snapshot_session", columns="ticker", values=value, aggfunc="sum")
    return m.reindex(index=sessions, columns=tickers).fillna(fill)


def _stack(mat: pd.DataFrame, name: str) -> pd.Series:
    s = mat.stack(future_stack=True)
    s.name = name
    return s


def compute_filing_history_features(filings: pd.DataFrame) -> pd.DataFrame:
    """Per-filing novelty proxies that need the issuer's own filing history."""
    df = filings.sort_values(["cik", "form_base", "accepted_at"]).copy()
    g = df.groupby(["cik", "form_base"], observed=True)
    prior_size = g["size"].shift(1)
    prior_time = g["accepted_at"].shift(1)
    df["size_log_ratio_same_form"] = np.log1p(df["size"]) - np.log1p(prior_size)
    df["days_since_same_form"] = (df["accepted_at"] - prior_time).dt.total_seconds() / 86400.0
    df["has_prior_same_form"] = prior_size.notna()

    # 8-K item-set novelty: has this issuer filed this item family lately?
    df["fam_key"] = df["item_list"].apply(
        lambda xs: "|".join(sorted({item_family(x) for x in xs})) if xs else ""
    )
    g2 = df.sort_values(["cik", "accepted_at"]).groupby("cik", observed=True)
    df = df.sort_values(["cik", "accepted_at"])
    df["days_since_any_filing"] = (
        g2["accepted_at"].diff().dt.total_seconds() / 86400.0
    )
    return df


def compute_event_features(events: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Same-evening disclosure features, one row per (snapshot_session, ticker)."""
    if events.empty:
        return pd.DataFrame()
    ev = events.copy()

    ev["sec_n_filings"] = ev["n_filings"].astype("float32")
    ev["sec_n_8k"] = ev["n_8k"].astype("float32")
    ev["sec_n_periodic"] = ev["n_periodic"].astype("float32")
    ev["sec_n_form4"] = ev["n_form4"].astype("float32")
    ev["sec_n_13d"] = ev["n_13d"].astype("float32")
    ev["sec_n_amendments"] = ev["n_amendments"].astype("float32")
    ev["sec_log_total_size"] = np.log1p(ev["total_size"]).astype("float32")
    ev["sec_log_max_size"] = np.log1p(ev["max_size"]).astype("float32")
    ev["sec_n_high_salience"] = ev["n_high_salience"].astype("float32")
    ev["sec_n_item_codes"] = ev["item_codes"].fillna("").apply(
        lambda s: len([x for x in s.split("|") if x])
    ).astype("float32")

    fams = ev["item_families"].fillna("")
    for fam in TRACKED_FAMILIES:
        ev[f"sec_item_{fam}"] = fams.str.contains(rf"(?:^|\|){fam}(?:\||$)", regex=True).astype("float32")

    # Timing within the snapshot window: minutes past the 16:00 ET close.
    acc_et = pd.to_datetime(ev["last_accepted_at"], utc=True).dt.tz_convert("America/New_York")
    minutes = acc_et.dt.hour * 60 + acc_et.dt.minute
    ev["sec_minutes_past_close"] = (minutes - 16 * 60).astype("float32")
    ev["sec_after_hours"] = (minutes >= 16 * 60).astype("float32")
    ev["sec_before_open"] = (minutes < 9 * 60 + 30).astype("float32")

    # Per-issuer novelty, aggregated from the filing level to the event level.
    fh = compute_filing_history_features(filings)
    agg = (
        fh.groupby(["snapshot_session", "ticker"], observed=True)
        .agg(
            sec_size_log_ratio=("size_log_ratio_same_form", "max"),
            sec_days_since_same_form=("days_since_same_form", "min"),
            sec_days_since_any=("days_since_any_filing", "min"),
        )
        .reset_index()
    )
    ev = ev.merge(agg, on=["snapshot_session", "ticker"], how="left")

    keep = ["snapshot_session", "ticker", "available_at"] + [
        c for c in ev.columns if c.startswith("sec_")
    ]
    return ev[keep]


def compute_grid_sec_features(
    events: pd.DataFrame, sessions: pd.DatetimeIndex, tickers: pd.Index
) -> pd.DataFrame:
    """Trailing disclosure-intensity features on the dense (session, ticker) grid."""
    if events.empty:
        return pd.DataFrame({"session": [], "ticker": []})
    counts = {
        "filings": _wide(events, sessions, tickers, "n_filings"),
        "8k": _wide(events, sessions, tickers, "n_8k"),
        "form4": _wide(events, sessions, tickers, "n_form4"),
        "salient": _wide(events, sessions, tickers, "n_high_salience"),
    }
    out = []
    for key, mat in counts.items():
        # Strictly past windows: shift(1) excludes tonight's own disclosure.
        past = mat.shift(1)
        out.append(_stack(past.rolling(21, min_periods=1).sum(), f"sec_{key}_21d_prior"))
        out.append(_stack(past.rolling(63, min_periods=1).sum(), f"sec_{key}_63d_prior"))

    # Sessions since the previous filing of any kind / previous 8-K.
    for key in ("filings", "8k"):
        mat = counts[key].shift(1) > 0
        idx = pd.Series(np.arange(len(mat.index)), index=mat.index)
        last_true = mat.mul(idx, axis=0).where(mat).ffill()
        gap = (-last_true).add(idx, axis=0)
        out.append(_stack(gap.fillna(999.0).clip(upper=999.0), f"sec_sessions_since_{key}"))

    df = pd.concat(out, axis=1).reset_index()
    df = df.rename(columns={df.columns[0]: "session", df.columns[1]: "ticker"})
    num = [c for c in df.columns if c not in ("session", "ticker")]
    df[num] = df[num].astype("float32")
    return df


def compute_insider_features(
    insider: pd.DataFrame, cal, sessions: pd.DatetimeIndex, tickers: pd.Index
) -> pd.DataFrame:
    """Trailing open-market insider buying/selling on the snapshot grid.

    Form 4 is due two business days after the transaction, so the *filing*
    timestamp (not the transaction date) is when the market could know. We map
    filings to snapshots exactly as with any other document.
    """
    if insider is None or insider.empty:
        return pd.DataFrame({"session": [], "ticker": []})
    df = insider.dropna(subset=["ticker", "filing_date"]).copy()
    # The Form345 datasets carry a filing date, not an acceptance instant; assume
    # the conservative 17:30 ET filing time, matching the EDGAR fallback.
    accepted = (
        pd.to_datetime(df["filing_date"]).dt.normalize()
        + pd.Timedelta(hours=17, minutes=30)
    ).dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    df["snapshot_session"] = cal.snapshot_session(accepted).values
    df = df.dropna(subset=["snapshot_session"])

    om = df[df["is_open_market"]]
    daily = (
        om.groupby(["snapshot_session", "ticker"], observed=True)
        .agg(
            buy_value=("signed_value", lambda s: float(s[s > 0].sum())),
            sell_value=("signed_value", lambda s: float(-s[s < 0].sum())),
            n_trans=("signed_value", "size"),
        )
        .reset_index()
    )
    if daily.empty:
        return pd.DataFrame({"session": [], "ticker": []})
    daily["net_value"] = daily["buy_value"] - daily["sell_value"]

    out = []
    for col in ("buy_value", "sell_value", "net_value", "n_trans"):
        mat = _wide(daily, sessions, tickers, col)
        for w in (5, 21, 63):
            out.append(_stack(mat.rolling(w, min_periods=1).sum(), f"insider_{col}_{w}d"))
    df2 = pd.concat(out, axis=1).reset_index()
    df2 = df2.rename(columns={df2.columns[0]: "session", df2.columns[1]: "ticker"})
    eps = 1.0
    for w in (5, 21, 63):
        b, s = df2[f"insider_buy_value_{w}d"], df2[f"insider_sell_value_{w}d"]
        df2[f"insider_buy_ratio_{w}d"] = b / (b + s + eps)
        df2[f"insider_log_net_{w}d"] = np.sign(df2[f"insider_net_value_{w}d"]) * np.log1p(
            df2[f"insider_net_value_{w}d"].abs()
        )
    num = [c for c in df2.columns if c not in ("session", "ticker")]
    df2[num] = df2[num].astype("float32")
    return df2
