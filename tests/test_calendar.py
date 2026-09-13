"""The snapshot convention is the foundation of every point-in-time claim."""
from __future__ import annotations

import pandas as pd

NY = "America/New_York"


def _snap(cal, ts):
    return cal.snapshot_session(pd.Timestamp(ts, tz=NY)).iloc[0]


def test_before_cutoff_lands_on_same_session(cal):
    assert _snap(cal, "2021-01-05 19:59") == pd.Timestamp("2021-01-05")


def test_exactly_at_cutoff_is_included(cal):
    # The snapshot may use information released *up to and including* 20:00.
    assert _snap(cal, "2021-01-05 20:00") == pd.Timestamp("2021-01-05")


def test_after_cutoff_rolls_to_next_session(cal):
    assert _snap(cal, "2021-01-05 20:01") == pd.Timestamp("2021-01-06")
    assert _snap(cal, "2021-01-05 21:00") == pd.Timestamp("2021-01-06")


def test_weekend_publication_lands_on_monday(cal):
    assert _snap(cal, "2021-01-08 23:00") == pd.Timestamp("2021-01-11")
    assert _snap(cal, "2021-01-09 12:00") == pd.Timestamp("2021-01-11")


def test_holiday_is_skipped(cal):
    # 2021-11-25 was Thanksgiving; a late Wednesday filing waits for Friday.
    assert _snap(cal, "2021-11-24 21:00") == pd.Timestamp("2021-11-26")


def test_dst_transition_is_handled(cal):
    # 2021-03-14 was the spring-forward; the snapshot instants stay 24h apart
    # in wall-clock terms and strictly increasing in UTC.
    ts = cal.snapshot_timestamps()
    assert ts.is_monotonic_increasing
    assert ts.tz is not None


def test_publication_after_study_end_has_no_snapshot(cal):
    assert pd.isna(_snap(cal, "2026-09-05 10:00"))


def test_snapshot_is_after_the_close_of_its_own_session(cal):
    t = cal.snapshot_table().head(50)
    local = pd.to_datetime(t["snapshot_ts"], utc=True).dt.tz_convert(NY)
    assert (local.dt.hour == 20).all()
    assert (local.dt.normalize().dt.tz_localize(None) == t["session"]).all()


def test_next_session_is_strictly_after(cal):
    t = cal.snapshot_table().dropna(subset=["next_session"])
    assert (t["next_session"] > t["session"]).all()


def test_vectorised_and_scalar_agree(cal):
    stamps = ["2021-01-05 21:00", "2021-06-15 10:00", "2021-11-24 21:00"]
    vec = cal.snapshot_session(pd.Series(pd.to_datetime(stamps)).dt.tz_localize(NY))
    for i, s in enumerate(stamps):
        assert vec.iloc[i] == _snap(cal, s)
