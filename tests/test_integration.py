"""End-to-end run against a stubbed network.

The real ingestion takes hours. An integration bug that only shows up at the
seam between ingest, build and experiment would therefore be discovered after
those hours, not before. This exercises the whole chain --
``ingest_all`` -> ``build_dataset`` -> ``run_experiment`` -> ``build_report``
-> ``build_shortlist`` -- on a corpus small enough to run in seconds, with every
network call replaced.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import ziggy.pipeline as pipe
import ziggy.providers.macro as macro_provider
import ziggy.providers.sec_edgar as sec
from ziggy.config import load_config
from ziggy.providers.market import MarketDataProvider, _finalise

N_TICKERS = 14
TICKERS = [f"ZZ{i:02d}" for i in range(N_TICKERS)]
SESSIONS = pd.DatetimeIndex(pd.bdate_range("2020-06-01", "2021-12-31"))


class FakeMarketProvider(MarketDataProvider):
    name = "fake"
    covers_delisted = True
    adjustment = "test"

    def get_daily(self, tickers, start, end):
        rng = np.random.default_rng(5)
        frames = []
        for t in list(tickers):
            if t not in TICKERS and t not in ("SPY", "QQQ", "IWM", "^VIX"):
                continue
            n = len(SESSIONS)
            p = 40 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
            if t == "^VIX":
                p = 14 + np.abs(rng.normal(0, 4, n))
            frames.append(pd.DataFrame({
                "date": SESSIONS, "ticker": t, "open": p * 0.998, "high": p * 1.01,
                "low": p * 0.99, "close": p, "adj_close": p,
                "volume": rng.integers(3_000_000, 9_000_000, n).astype(float),
            }))
        return _finalise(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def _fake_filings(ciks):
    rng = np.random.default_rng(6)
    rows = []
    for i, cik in enumerate(ciks):
        days = rng.choice(SESSIONS, size=40, replace=False)
        for j, d in enumerate(pd.to_datetime(sorted(days))):
            form = str(rng.choice(["8-K", "4", "10-Q", "10-K"], p=[0.35, 0.4, 0.2, 0.05]))
            hour = int(rng.integers(7, 22))
            rows.append({
                "cik": cik, "accession": f"{cik:010d}-21-{i:03d}{j:03d}", "form": form,
                "filing_date": d.strftime("%Y-%m-%d"), "report_date": None,
                "acceptance_raw": (d + pd.Timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "primary_document": "doc.htm", "primary_doc_description": "",
                "items": str(rng.choice(["2.02,9.01", "5.02", "1.01", "", "8.01"])) if form == "8-K" else "",
                "size": int(rng.integers(20_000, 2_000_000)), "is_xbrl": 1,
                "is_inline_xbrl": 1, "act": "34", "file_number": "001-00000",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_cfg(tmp_path):
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg.raw["paths"] = {k: str(tmp_path / k) for k in ("raw", "interim", "processed", "artifacts")}
    cfg.raw["experiment"].update({"start_date": "2021-01-04", "end_date": "2021-12-31"})
    cfg.raw["universe"].update({"min_history_sessions": 60, "min_median_dollar_volume_20d": 1.0,
                                "min_price": 1.0, "max_names": N_TICKERS})
    cfg.raw["labels"].update({"horizons": [1, 5], "primary_horizon": 5})
    cfg.raw["splits"].update({"embargo_sessions": 5})
    cfg.raw["ranking"].update({"top_k": [3, 5], "primary_k": 5, "models": ["deterministic", "logistic"]})
    cfg.raw["ranking"]["walk_forward"]["enabled"] = False
    cfg.raw["evaluation"].update({"bootstrap_samples": 50, "permutation_samples": 5})
    cfg.raw["sec"]["document_budget"] = 40
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def stubbed_network(monkeypatch):
    ciks = [900000 + i for i in range(N_TICKERS)]
    tmap = pd.DataFrame({"cik": ciks, "ticker": TICKERS,
                         "company": [f"Co {i}" for i in range(N_TICKERS)], "exchange": "XTST"})
    monkeypatch.setattr(sec, "fetch_ticker_map", lambda fetcher: tmap)
    monkeypatch.setattr(
        sec, "fetch_filings_bulk",
        lambda fetcher, cs, **kw: (_fake_filings(cs), pd.DataFrame({"cik": cs, "sic": "7372"})),
    )
    monkeypatch.setattr(sec, "fetch_form345_quarter", lambda f, y, q: {})
    monkeypatch.setattr(sec, "fetch_filing_text",
                        lambda f, cik, acc, doc: "the company reported results and an impairment charge " * 40)
    monkeypatch.setattr(pipe, "build_market_provider", lambda cfg, fetcher, which=None: FakeMarketProvider())

    def fake_macro(fetcher, cfg):
        avail = (pd.DatetimeIndex(SESSIONS) + pd.Timedelta(days=1, hours=18)).tz_localize(
            "America/New_York", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        return pd.DataFrame({"series_id": "DGS10", "date": SESSIONS,
                             "value": np.linspace(1.0, 2.0, len(SESSIONS)),
                             "vintage_date": pd.NaT, "available_at": avail, "revisable": False})

    monkeypatch.setattr(macro_provider, "build_macro_panel", fake_macro)
    return tmap


def test_full_chain_runs_and_stays_point_in_time(tiny_cfg, stubbed_network):
    from ziggy.experiment.run import run_experiment
    from ziggy.report.build_report import build_report
    from ziggy.shortlist import build_shortlist
    from ziggy.store import Store

    stats = pipe.ingest_all(tiny_cfg)
    assert stats["prices"] > 0 and stats["filings"] > 0 and stats["universe"] > 0
    assert stats["documents"] > 0, "documents were never fetched"

    store = Store(tiny_cfg)
    assert store.exists("interim", "filing_text")

    matrix = pipe.build_dataset(tiny_cfg)
    assert len(matrix) > 0
    assert matrix["session"].min() >= pd.Timestamp("2021-01-04")
    assert {"label_cs_q90", "consequence_magnitude"}.issubset(matrix.columns)
    # Derived labels must reach the matrix, not only the experiment.
    assert "label_cs_q90_volnorm" in matrix.columns

    manifest = run_experiment(tiny_cfg, matrix)
    assert manifest["holdout_evaluations"] == 1
    assert manifest["audits"]["feature_availability"]["status"] == "clean"
    assert manifest["audits"]["forward_columns"]["status"] == "clean"
    assert manifest["audits"]["acceptance_timezone"]["verdict"] == "consistent_with_eastern"
    assert manifest["provenance"]["git_commit"]
    assert manifest["selected_model"] in ("deterministic", "logistic")

    report = build_report(tiny_cfg)
    text = report.read_text()
    assert "## Verdict" in text and "## Headline result" in text

    sl = build_shortlist(tiny_cfg, k=3)
    assert len(sl) == 3
    assert (sl["why"].str.len() > 0).all()


def test_ingestion_is_idempotent_and_cached(tiny_cfg, stubbed_network, monkeypatch):
    """A second ingest must reuse what is on disk rather than re-fetch."""
    pipe.ingest_all(tiny_cfg)
    calls = {"n": 0}

    def exploding(*a, **k):
        calls["n"] += 1
        raise AssertionError("re-fetched an already-cached table")

    monkeypatch.setattr(sec, "fetch_ticker_map", exploding)
    monkeypatch.setattr(sec, "fetch_filings_bulk", exploding)
    pipe.ingest_all(tiny_cfg)
    assert calls["n"] == 0
