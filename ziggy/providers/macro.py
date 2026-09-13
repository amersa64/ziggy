"""Macro series with an explicit availability model.

Two distinct hazards are handled separately:

1. **Publication lag.** An observation dated 2021-03-31 was not knowable on
   2021-03-31. Every series carries ``lag_days``; the value becomes available at
   18:00 ET on ``observation_date + lag_days``, comfortably before the 20:00
   snapshot of that session.
2. **Revision.** Unemployment, payrolls, CPI and industrial production are
   revised for years afterwards. Using today's value as though it were known in
   2021 is a silent leak. For those series we pull dated ALFRED *vintages* and
   the feature layer as-of joins the latest vintage available at the snapshot.
"""
from __future__ import annotations

import io
import logging

import pandas as pd

from ziggy.net import Fetcher

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
ALFRED_CSV = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}&vintage_date={vd}"
AVAILABILITY_HOUR_ET = 18


def _parse_csv(content: bytes, series_id: str) -> pd.DataFrame:
    if not content:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(io.BytesIO(content))
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(columns=["date", "value"])
    date_col = df.columns[0]
    val_col = next((c for c in df.columns[1:] if series_id.upper() in c.upper()), df.columns[1])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[val_col].replace(".", pd.NA), errors="coerce"),
        }
    )
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_series(fetcher: Fetcher, series_id: str) -> pd.DataFrame:
    """Latest vintage of a series (appropriate only for non-revisable data)."""
    res = fetcher.get(FRED_CSV.format(sid=series_id))
    if res.status != 200:
        log.warning("FRED %s unavailable (%s)", series_id, res.status)
        return pd.DataFrame(columns=["date", "value"])
    return _parse_csv(res.content, series_id)


def fetch_vintage(fetcher: Fetcher, series_id: str, vintage_date: str) -> pd.DataFrame:
    """The series exactly as it stood on ``vintage_date`` (ALFRED)."""
    res = fetcher.get(ALFRED_CSV.format(sid=series_id, vd=vintage_date))
    if res.status != 200:
        return pd.DataFrame(columns=["date", "value"])
    return _parse_csv(res.content, series_id)


def _available_at(dates: pd.Series, lag_days: int) -> pd.Series:
    stamped = pd.to_datetime(dates) + pd.Timedelta(days=int(lag_days)) + pd.Timedelta(hours=AVAILABILITY_HOUR_ET)
    return (
        stamped.dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.as_unit("ns")
    )


def build_macro_panel(fetcher: Fetcher, cfg) -> pd.DataFrame:
    """Long point-in-time macro table.

    Columns: ``series_id, date, value, vintage_date, available_at, revisable``.
    One row per (series, observation, vintage) that a snapshot could have seen.
    """
    spec = cfg.macro["series"]
    cadence = int(cfg.macro.get("vintage_cadence_days", 30))
    start = pd.Timestamp(cfg.experiment["start_date"])
    end = pd.Timestamp(cfg.experiment["end_date"])
    frames = []

    for s in spec:
        sid, lag, revisable = s["id"], int(s["lag_days"]), bool(s.get("revisable", False))
        if not revisable:
            df = fetch_series(fetcher, sid)
            if df.empty:
                continue
            df = df[(df["date"] >= start - pd.Timedelta(days=400)) & (df["date"] <= end)]
            df["series_id"] = sid
            df["vintage_date"] = pd.NaT
            df["available_at"] = _available_at(df["date"], lag)
            df["revisable"] = False
            frames.append(df)
            continue

        # Revisable: walk vintages across the study window.
        vintages = pd.date_range(start, end, freq=f"{cadence}D")
        got = 0
        for vd in vintages:
            v = fetch_vintage(fetcher, sid, vd.strftime("%Y-%m-%d"))
            if v.empty:
                continue
            v = v[v["date"] <= vd]
            if v.empty:
                continue
            v = v.tail(24).copy()  # two years of history per vintage is ample
            v["series_id"] = sid
            v["vintage_date"] = vd
            # The vintage itself is what became knowable, at the vintage date.
            v["available_at"] = _available_at(pd.Series([vd] * len(v)), 0).values
            v["revisable"] = True
            frames.append(v)
            got += 1
        log.info("%s: %d vintages", sid, got)

    if not frames:
        return pd.DataFrame(
            columns=["series_id", "date", "value", "vintage_date", "available_at", "revisable"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[["series_id", "date", "value", "vintage_date", "available_at", "revisable"]]
