"""Trading calendar and the snapshot convention.

The single most important invariant in this repository:

    A snapshot for session D is taken at 20:00 America/New_York on D and may use
    information publicly released at or before that instant, and nothing else.

Anything released after 20:00 ET on D belongs to the *next* session's snapshot.
A hypothetical downstream agent acts no earlier than the open of session D+1.
"""
from __future__ import annotations

from datetime import time
from functools import lru_cache

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised implicitly
    import exchange_calendars as xcals

    _HAVE_XCALS = True
except Exception:  # pragma: no cover
    _HAVE_XCALS = False

NY = "America/New_York"
UTC = "UTC"


@lru_cache(maxsize=8)
def _sessions(start: str, end: str, calendar: str = "XNYS") -> pd.DatetimeIndex:
    """US equity trading sessions between ``start`` and ``end`` inclusive."""
    if _HAVE_XCALS:
        # Pad the construction bounds: get_calendar() refuses a range whose
        # endpoints are not themselves sessions.
        pad_start = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        pad_end = (pd.Timestamp(end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        cal = xcals.get_calendar(calendar, start=pad_start, end=pad_end)
        idx = pd.DatetimeIndex(pd.to_datetime(cal.sessions))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
        idx = idx.as_unit("ns")
        return idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    # Fallback: weekdays only. Loudly worse; used only if the calendar package
    # is unavailable. Holidays would leak in as phantom snapshots.
    return pd.DatetimeIndex(pd.bdate_range(start, end))


class TradingCalendar:
    def __init__(self, start: str, end: str, snapshot_time_local: str = "20:00", tz: str = NY):
        self.start, self.end, self.tz = start, end, tz
        hh, mm = snapshot_time_local.split(":")
        self.snapshot_time = time(int(hh), int(mm))
        self.sessions = _sessions(start, end)

    # -- sessions --------------------------------------------------------------
    def session_index(self) -> pd.DatetimeIndex:
        return self.sessions

    def is_session(self, d) -> bool:
        return pd.Timestamp(d).normalize() in set(self.sessions)

    def next_session(self, d, offset: int = 1) -> pd.Timestamp | None:
        d = pd.Timestamp(d).normalize()
        pos = self.sessions.searchsorted(d, side="right") - 1 + offset
        if pos < 0 or pos >= len(self.sessions):
            return None
        return self.sessions[pos]

    def shift_sessions(self, d, n: int) -> pd.Timestamp | None:
        """Session ``n`` places after ``d`` (``d`` itself must be a session)."""
        d = pd.Timestamp(d).normalize()
        pos = self.sessions.searchsorted(d, side="left")
        if pos >= len(self.sessions) or self.sessions[pos] != d:
            return None
        tgt = pos + n
        if tgt < 0 or tgt >= len(self.sessions):
            return None
        return self.sessions[tgt]

    # -- snapshots -------------------------------------------------------------
    def snapshot_timestamps(self) -> pd.DatetimeIndex:
        """UTC instants of every snapshot in the configured range."""
        local = pd.DatetimeIndex(
            [pd.Timestamp.combine(d.date(), self.snapshot_time) for d in self.sessions]
        )
        return (
            local.tz_localize(self.tz, nonexistent="shift_forward", ambiguous=True)
            .tz_convert(UTC)
            .as_unit("ns")
        )

    def snapshot_table(self) -> pd.DataFrame:
        """One row per snapshot: session date, snapshot instant, tradable session."""
        ts = self.snapshot_timestamps()
        nxt = [self.shift_sessions(d, 1) for d in self.sessions]
        return pd.DataFrame(
            {
                "session": self.sessions,
                "snapshot_ts": ts,
                "next_session": pd.DatetimeIndex(pd.Series(nxt, dtype="datetime64[ns]")),
            }
        )

    def assign_snapshot(self, published_ts) -> pd.Series | pd.Timestamp | None:
        """Map publication instants to the snapshot that may first see them.

        A document published at 21:00 ET on Tuesday is assigned to Wednesday's
        snapshot, because Tuesday's snapshot closed at 20:00 ET.
        """
        snaps = self.snapshot_timestamps()
        scalar = not isinstance(published_ts, (pd.Series, pd.DatetimeIndex, np.ndarray, list))
        vals = pd.DatetimeIndex(
            pd.to_datetime([published_ts] if scalar else published_ts, utc=True)
        ).as_unit("ns")
        index = published_ts.index if isinstance(published_ts, pd.Series) else pd.RangeIndex(len(vals))
        # Compare in integer nanoseconds to sidestep tz-awareness coercion.
        pos = np.searchsorted(snaps.asi8, vals.asi8, side="left")
        taken = np.where(pos < len(snaps), np.clip(pos, 0, len(snaps) - 1), -1)
        out_int = np.full(len(vals), np.iinfo(np.int64).min, dtype="int64")  # NaT sentinel
        ok = (taken >= 0) & ~pd.isna(vals)
        out_int[ok] = snaps.asi8[taken[ok]]
        out_idx = pd.DatetimeIndex(out_int.view("datetime64[ns]")).tz_localize(UTC)
        out = pd.Series(out_idx, index=index)
        if scalar:
            v = out.iloc[0]
            return v if pd.notna(v) else None
        return out

    def snapshot_session(self, published_ts) -> pd.Series:
        """Session date of the snapshot that first sees ``published_ts``."""
        snap = self.assign_snapshot(published_ts)
        if not isinstance(snap, pd.Series):
            snap = pd.Series(pd.DatetimeIndex([snap]).tz_convert(UTC) if snap is not None
                             else pd.DatetimeIndex([pd.NaT]).tz_localize(UTC))
        return snap.dt.tz_convert(self.tz).dt.normalize().dt.tz_localize(None)


def build_calendar_with_warmup(cfg, warmup_days: int) -> TradingCalendar:
    """Calendar extended backwards so trailing windows (252-session returns, the
    universe history requirement) are warm on the first study session."""
    e = cfg.experiment
    start = (pd.Timestamp(e["start_date"]) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    return TradingCalendar(
        start=start,
        end=str(e["end_date"]),
        snapshot_time_local=str(e["snapshot_time_local"]),
        tz=str(e["timezone"]),
    )


def build_calendar(cfg) -> TradingCalendar:
    e = cfg.experiment
    return TradingCalendar(
        start=str(e["start_date"]),
        end=str(e["end_date"]),
        snapshot_time_local=str(e["snapshot_time_local"]),
        tz=str(e["timezone"]),
    )
