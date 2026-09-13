"""Point-in-time bookkeeping: events, availability and the store guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ziggy.audit import (
    acceptance_timezone_audit,
    feature_availability_audit,
    forward_column_audit,
    leakage_canary,
    shuffle_control,
)
from ziggy.events import attach_snapshots, build_events, map_filings_to_tickers
from ziggy.providers.sec_edgar import normalise_filings
from ziggy.store import assert_pit


def _filings(cal, n=300, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.to_datetime(rng.choice(pd.bdate_range("2021-02-01", "2021-11-30"), n))
    hours = rng.integers(6, 22, n)
    acc = days + pd.to_timedelta(hours, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m")
    raw = pd.DataFrame({
        "cik": rng.choice([111, 222, 333], n),
        "accession": [f"{i:018d}" for i in range(n)],
        "form": rng.choice(["8-K", "4", "10-Q"], n, p=[0.4, 0.4, 0.2]),
        "filing_date": days.strftime("%Y-%m-%d"), "report_date": None,
        "acceptance_raw": acc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "primary_document": "d.htm", "primary_doc_description": "",
        "items": rng.choice(["2.02,9.01", "5.02", "", "1.01"], n),
        "size": rng.integers(10_000, 2_000_000, n),
        "is_xbrl": 1, "is_inline_xbrl": 1, "act": "34", "file_number": "001",
    })
    nf = normalise_filings(raw, "America/New_York")
    tm = pd.DataFrame({"cik": [111, 222, 333], "ticker": ["AAA", "BBB", "CCC"]})
    return map_filings_to_tickers(attach_snapshots(nf, cal), tm)


def test_no_filing_is_ever_visible_before_it_was_accepted(cal):
    f = _filings(cal)
    events = build_events(f)
    snaps = cal.snapshot_table().rename(columns={"session": "snapshot_session"})
    m = events.merge(snaps, on="snapshot_session", how="left")
    assert (pd.to_datetime(m["last_accepted_at"], utc=True)
            <= pd.to_datetime(m["snapshot_ts"], utc=True)).all()


def test_availability_audit_reports_clean_for_a_correct_build(cal):
    f = _filings(cal)
    events = build_events(f)
    matrix = cal.snapshot_table()[["session", "snapshot_ts"]]
    res = feature_availability_audit(matrix, events)
    assert res["status"] == "clean"
    assert res["violations_disclosure_after_snapshot"] == 0
    assert res["min_gap_hours"] >= 0


def test_availability_audit_detects_an_injected_violation(cal):
    f = _filings(cal)
    events = build_events(f)
    events.loc[events.index[:5], "last_accepted_at"] = pd.Timestamp("2030-01-01", tz="UTC")
    res = feature_availability_audit(cal.snapshot_table()[["session", "snapshot_ts"]], events)
    assert res["status"] == "LEAK"
    assert res["violations_disclosure_after_snapshot"] == 5


def test_assert_pit_accepts_past_and_rejects_future():
    ok = pd.DataFrame({"snapshot_ts": pd.to_datetime(["2021-01-05"], utc=True),
                       "available_at": pd.to_datetime(["2021-01-04"], utc=True)})
    assert_pit(ok)
    bad = ok.assign(available_at=pd.to_datetime(["2021-01-06"], utc=True))
    with pytest.raises(AssertionError):
        assert_pit(bad)


def test_forward_column_audit_blocks_label_shaped_features():
    assert forward_column_audit(["vol_21d", "sec_n_8k"])["status"] == "clean"
    assert forward_column_audit(["vol_21d", "fwd_ret_5d"])["status"] == "LEAK"
    assert forward_column_audit(["consequence_magnitude"])["status"] == "LEAK"


def test_acceptance_timezone_audit_flags_a_utc_misreading():
    # Eastern wall-clock filings, correctly localised: all inside EDGAR hours.
    et = pd.to_datetime(["2021-01-05 16:30", "2021-01-05 08:15", "2021-01-06 18:45"])
    good = pd.DataFrame({
        "accepted_at": et.tz_localize("America/New_York").tz_convert("UTC"),
        "form_base": ["8-K"] * 3,
    })
    assert acceptance_timezone_audit(good)["verdict"] == "consistent_with_eastern"
    # The same stamps wrongly treated as UTC land at 03:30/11:15/13:45 ET.
    bad = pd.DataFrame({"accepted_at": et.tz_localize("UTC"), "form_base": ["8-K"] * 3})
    assert acceptance_timezone_audit(bad)["share_within_edgar_hours_06_22"] < 1.0


@pytest.fixture
def toy_scored():
    rng = np.random.default_rng(4)
    rows = []
    for s in pd.bdate_range("2021-01-04", "2021-06-30"):
        n = 200
        sig = rng.normal(size=n)
        mag = np.abs(rng.normal(size=n) * 0.03 + 0.025 * sig)
        rows.append(pd.DataFrame({"session": s, "score": sig, "mag": mag}))
    d = pd.concat(rows, ignore_index=True)
    d["label"] = (d.groupby("session")["mag"].rank(pct=True) > 0.9).astype(float)
    return d


def test_shuffling_the_score_destroys_the_edge(toy_scored):
    res = shuffle_control(toy_scored, "score", "label", "mag", 20, seed=1)
    assert res["status"] == "clean"
    assert res["real_lift"] > res["shuffled_score_lift"] + 0.5


def test_the_leak_detector_fires_on_a_cheating_ranker(toy_scored):
    res = leakage_canary(toy_scored, [], "label", "mag", 20, seed=1)
    assert res["status"] == "detector_works"
    assert res["cheating_lift"] > 3.0
