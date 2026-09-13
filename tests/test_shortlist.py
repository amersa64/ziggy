"""The shortlist is the product; it has to be correct and legible."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from ziggy.shortlist import HUMAN_LABELS, _reasons, build_shortlist, format_shortlist
from ziggy.rank.models import DETERMINISTIC_WEIGHTS
from ziggy.store import Store


@pytest.fixture
def wired(cfg, tmp_path):
    """A miniature but structurally complete set of pipeline outputs."""
    c = copy.deepcopy(cfg)
    c.raw["paths"] = {k: str(tmp_path / k) for k in ("raw", "interim", "processed", "artifacts")}
    store = Store(c)
    rng = np.random.default_rng(0)
    sessions = pd.to_datetime(["2021-03-01", "2021-03-02"])
    tickers = [f"T{i:02d}" for i in range(40)]

    rows = []
    for s in sessions:
        for t in tickers:
            rows.append({"session": s, "ticker": t})
    m = pd.DataFrame(rows)
    n = len(m)
    for col in DETERMINISTIC_WEIGHTS:
        m[col] = rng.random(n).astype("float32")
    for col in ("ret_5d", "exret_5d", "dist_52w_high", "sec_log_total_size",
                "sec_n_filings", "sec_n_8k", "sec_n_periodic", "sec_n_form4",
                "sec_minutes_past_close", "insider_log_net_21d"):
        m[col] = rng.normal(size=n).astype("float32")
    store.write(m, "processed", "candidates")

    scored = m[["session", "ticker"]].copy()
    scored["score_deterministic"] = rng.random(n)
    scored["score_gbm"] = rng.random(n)
    store.write(scored, "processed", "scored")

    ev = pd.DataFrame({
        "snapshot_session": [sessions[1]] * 3,
        "ticker": tickers[:3],
        "forms": ["8-K", "8-K|4", "10-Q"],
        "item_codes": ["2.02,9.01", "5.02", ""],
        "item_families": ["results|exhibits", "executive_change", ""],
        "last_accepted_at": pd.to_datetime(["2021-03-02 21:05"] * 3, utc=True),
        "primary_url": [f"https://www.sec.gov/Archives/{t}.htm" for t in tickers[:3]],
    })
    store.write(ev, "interim", "events")
    store.write_json({"selected_model": "gbm"}, "artifacts", "manifest")
    return c


def test_shortlist_is_the_requested_length_and_ordered_by_score(wired):
    sl = build_shortlist(wired, session="2021-03-02", k=12)
    assert len(sl) == 12
    assert sl["rank"].tolist() == list(range(1, 13))
    assert sl["score"].is_monotonic_decreasing


def test_shortlist_defaults_to_the_selected_model_and_latest_session(wired):
    sl = build_shortlist(wired)
    assert set(sl["model"]) == {"gbm"}
    assert set(sl["session"]) == {"2021-03-02"}


def test_disclosure_evidence_is_attached_when_present(wired):
    sl = build_shortlist(wired, session="2021-03-02", k=40).set_index("ticker")
    assert sl.loc["T00", "item_families"] == "results|exhibits"
    assert sl.loc["T00", "source_url"].startswith("https://www.sec.gov/")
    # A name that filed nothing must not borrow another name's evidence.
    assert pd.isna(sl.loc["T39", "item_families"])
    assert pd.isna(sl.loc["T39", "source_url"])


def test_every_row_carries_a_human_readable_reason(wired):
    sl = build_shortlist(wired, session="2021-03-02", k=15)
    assert (sl["why"].str.len() > 0).all()
    assert not sl["why"].str.contains("_").any(), "raw feature names leaked into the explanation"


def test_reasons_name_the_dominant_signal():
    ranks = pd.Series({f: 0.5 for f in DETERMINISTIC_WEIGHTS})
    ranks["sec_item_ma_assets"] = 1.0
    ranks["volume_z_20d"] = 1.0
    why = _reasons(ranks)
    assert HUMAN_LABELS["sec_item_ma_assets"] in why
    assert HUMAN_LABELS["volume_z_20d"] in why


def test_reasons_are_honest_when_nothing_stands_out():
    ranks = pd.Series({f: 0.5 for f in DETERMINISTIC_WEIGHTS})
    assert "no single standout signal" in _reasons(ranks)


def test_negatively_weighted_features_are_reported_in_their_helpful_direction():
    # log_adv20_universe carries a negative weight, so a LOW rank (illiquid) is
    # what should be surfaced, never a high one.
    ranks = pd.Series({f: 0.5 for f in DETERMINISTIC_WEIGHTS})
    ranks["log_adv20_universe"] = 0.01
    assert HUMAN_LABELS["log_adv20_universe"] in _reasons(ranks)
    ranks["log_adv20_universe"] = 0.99
    assert HUMAN_LABELS["log_adv20_universe"] not in _reasons(ranks)


def test_unknown_model_fails_loudly(wired):
    with pytest.raises(KeyError):
        build_shortlist(wired, session="2021-03-02", model="does_not_exist")


def test_session_without_candidates_fails_loudly(wired):
    with pytest.raises(ValueError):
        build_shortlist(wired, session="2021-03-09")


def test_terminal_rendering_includes_tickers_and_sources(wired):
    # k spans the whole universe so the three names that filed are certain to
    # appear; at small k the shortlist may legitimately contain no filers.
    text = format_shortlist(build_shortlist(wired, session="2021-03-02", k=40))
    assert "Shortlist for 2021-03-02" in text
    assert "why:" in text
    assert "T00" in text and "results|exhibits" in text
    assert any(line.strip().startswith("https://www.sec.gov/") for line in text.splitlines())
