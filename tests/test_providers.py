"""Provider parsing, against fixtures shaped like the real payloads."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ziggy.features.macro import compute_macro_features
from ziggy.net import Fetcher, RateLimiter
from ziggy.providers.market import OHLCV_COLUMNS, CsvDirectoryProvider, StooqProvider, _finalise
from ziggy.providers.news import NullNewsProvider, build_news_provider
from ziggy.providers.sec_edgar import _rows_from_block, fetch_company_filings, parse_insider_transactions

SUBMISSIONS = {
    "cik": "320193", "name": "Apple Inc.", "sic": "3571", "sicDescription": "Electronic Computers",
    "exchanges": ["Nasdaq"],
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-21-000010", "0000320193-21-000011"],
            "filingDate": ["2021-01-27", "2021-01-28"],
            "reportDate": ["2020-12-26", ""],
            "acceptanceDateTime": ["2021-01-27T16:31:22.000Z", "2021-01-28T06:03:00.000Z"],
            "act": ["34", "34"], "form": ["8-K", "10-Q"], "fileNumber": ["001-36743", "001-36743"],
            "items": ["2.02,9.01", ""], "size": [120000, 900000],
            "isXBRL": [1, 1], "isInlineXBRL": [1, 1],
            "primaryDocument": ["a8k.htm", "aapl-10q.htm"],
            "primaryDocDescription": ["8-K", "10-Q"],
        },
        "files": [{"name": "CIK0000320193-submissions-001.json"}],
    },
}
SHARD = {
    "accessionNumber": ["0000320193-20-000001"], "filingDate": ["2020-05-01"],
    "reportDate": [""], "acceptanceDateTime": ["2020-05-01T18:00:00.000Z"], "act": ["34"],
    "form": ["8-K"], "fileNumber": ["001-36743"], "items": ["5.02"], "size": [40000],
    "isXBRL": [0], "isInlineXBRL": [0], "primaryDocument": ["x.htm"], "primaryDocDescription": [""],
}


class FakeFetcher:
    """Serves the fixtures above without touching the network."""

    def __init__(self):
        self.calls = []

    def get_json(self, url, **kw):
        self.calls.append(url)
        if "submissions-001" in url:
            return SHARD
        if "CIK0000320193" in url:
            return SUBMISSIONS
        return None


def test_submissions_block_is_parsed_into_rows():
    df = _rows_from_block(SUBMISSIONS["filings"]["recent"], 320193)
    assert len(df) == 2
    assert set(df["form"]) == {"8-K", "10-Q"}
    assert df["items"].iloc[0] == "2.02,9.01"


def test_older_shards_are_followed():
    f = FakeFetcher()
    df = fetch_company_filings(f, 320193)
    assert len(df) == 3, "the paginated shard was not fetched"
    assert "0000320193-20-000001" in set(df["accession"])
    assert df.attrs["sic"] == "3571"


def test_block_with_ragged_columns_does_not_crash():
    bad = dict(SUBMISSIONS["filings"]["recent"])
    bad["items"] = ["2.02"]          # shorter than accessionNumber
    df = _rows_from_block(bad, 1)
    # A ragged column falls back to the declared default rather than misaligning
    # item codes onto the wrong filings, which would be far worse than losing them.
    assert len(df) == 2
    assert (df["items"] == "").all()


def test_empty_block_returns_empty_frame():
    assert _rows_from_block({}, 1).empty


def test_insider_transactions_are_signed_by_acquire_dispose():
    parts = {
        "submission": pd.DataFrame({
            "ACCESSION_NUMBER": ["a", "b"], "FILING_DATE": ["01-MAR-2021", "02-MAR-2021"],
            "PERIOD_OF_REPORT": ["01-MAR-2021", "02-MAR-2021"], "ISSUERCIK": ["1", "1"],
            "ISSUERTRADINGSYMBOL": ["aapl", "aapl"], "DOCUMENT_TYPE": ["4", "4"],
        }),
        "nonderiv_trans": pd.DataFrame({
            "ACCESSION_NUMBER": ["a", "b"], "TRANS_DATE": ["01-MAR-2021", "02-MAR-2021"],
            "TRANS_CODE": ["P", "S"], "TRANS_ACQUIRED_DISP_CD": ["A", "D"],
            "TRANS_SHARES": ["100", "50"], "TRANS_PRICEPERSHARE": ["10", "20"],
            "SHRS_OWND_FOLWNG_TRANS": ["100", "50"], "DIRECT_INDIRECT_OWNERSHIP": ["D", "D"],
        }),
    }
    out = parse_insider_transactions(parts)
    assert out["ticker"].tolist() == ["AAPL", "AAPL"]
    assert out["signed_value"].tolist() == [1000.0, -1000.0]
    assert out["is_open_market"].all()


def test_insider_grants_are_not_treated_as_open_market():
    parts = {
        "submission": pd.DataFrame({"ACCESSION_NUMBER": ["a"], "FILING_DATE": ["01-MAR-2021"],
                                    "ISSUERCIK": ["1"], "ISSUERTRADINGSYMBOL": ["x"],
                                    "DOCUMENT_TYPE": ["4"]}),
        "nonderiv_trans": pd.DataFrame({"ACCESSION_NUMBER": ["a"], "TRANS_DATE": ["01-MAR-2021"],
                                        "TRANS_CODE": ["A"], "TRANS_ACQUIRED_DISP_CD": ["A"],
                                        "TRANS_SHARES": ["100"], "TRANS_PRICEPERSHARE": ["0"],
                                        "SHRS_OWND_FOLWNG_TRANS": ["100"],
                                        "DIRECT_INDIRECT_OWNERSHIP": ["D"]}),
    }
    assert not parse_insider_transactions(parts)["is_open_market"].any()


def test_stooq_symbol_mapping():
    assert StooqProvider.to_symbol("AAPL") == "aapl.us"
    assert StooqProvider.to_symbol("BRK.B") == "brk-b.us"
    assert StooqProvider.to_symbol("^VIX") == "^vix"


def test_market_finalise_drops_bad_bars_and_dedupes():
    raw = pd.DataFrame({
        "date": ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-06"],
        "ticker": ["a", "a", "a", "a"],
        "open": [1, 1, 2, 3], "high": [1, 1, 2, 3], "low": [1, 1, 2, 3],
        "close": [1.0, 1.0, 0.0, 3.0], "adj_close": [1, 1, 0, 3], "volume": [10, 10, 10, 10],
    })
    out = _finalise(raw)
    assert list(out.columns) == OHLCV_COLUMNS
    assert len(out) == 2                       # dedupe + drop the zero close
    assert out["ticker"].tolist() == ["A", "A"]


def test_csv_provider_round_trip(tmp_path):
    d = tmp_path / "csv_prices"
    d.mkdir()
    pd.DataFrame({"date": ["2021-01-04", "2021-01-05"], "open": [1, 2], "high": [1, 2],
                  "low": [1, 2], "close": [1, 2], "volume": [5, 6]}).to_csv(d / "ABC.csv", index=False)
    out = CsvDirectoryProvider(d).get_daily(["ABC"], "2021-01-01", "2021-12-31")
    assert len(out) == 2 and out["adj_close"].tolist() == [1.0, 2.0]


def test_news_provider_is_deliberately_absent(cfg):
    p = build_news_provider(cfg)
    assert isinstance(p, NullNewsProvider)
    assert p.get_articles(["AAPL"], "2021-01-01", "2021-12-31").empty
    assert len(p.integrity_requirements()) >= 4


def test_unimplemented_news_provider_refuses_loudly(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c.raw["news"]["provider"] = "some_scraped_feed"
    with pytest.raises(NotImplementedError):
        build_news_provider(c)


def test_macro_asof_never_uses_a_future_vintage():
    snaps = pd.DataFrame({
        "session": pd.to_datetime(["2021-06-01", "2021-07-01", "2021-08-02"]),
        "snapshot_ts": pd.to_datetime(["2021-06-02 00:00", "2021-07-02 00:00", "2021-08-03 00:00"], utc=True),
    })
    panel = pd.DataFrame({
        "series_id": ["UNRATE"] * 3,
        "date": pd.to_datetime(["2021-04-01", "2021-04-01", "2021-04-01"]),
        "value": [6.1, 6.0, 5.8],                       # the same month, revised twice
        "vintage_date": pd.to_datetime(["2021-05-15", "2021-07-15", "2021-09-15"]),
        "available_at": pd.to_datetime(["2021-05-15", "2021-07-15", "2021-09-15"], utc=True),
        "revisable": [True] * 3,
    })
    out = compute_macro_features(panel, snaps).set_index("session")
    assert out.loc["2021-06-01", "macro_UNRATE"] == pytest.approx(6.1)   # first print
    assert out.loc["2021-08-02", "macro_UNRATE"] == pytest.approx(6.0)   # first revision
    # The September revision must never reach an August snapshot.
    assert out["macro_UNRATE"].max() == pytest.approx(6.1)


def test_rate_limiter_enforces_a_minimum_gap():
    import time

    rl = RateLimiter(20.0)
    t0 = time.monotonic()
    for _ in range(5):
        rl.acquire()
    assert time.monotonic() - t0 >= 0.15


def test_fetcher_cache_round_trip(tmp_path):
    f = Fetcher(tmp_path, "ziggy-test", max_per_second=0, cache_enabled=True, namespace="t")
    f._store("http://example.com/x", b"payload")
    assert f.cached("http://example.com/x") == b"payload"
    assert f.cached("http://example.com/missing") is None
