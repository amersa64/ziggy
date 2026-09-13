"""Pipeline wiring, exercised without a network."""
from __future__ import annotations

import copy

import pandas as pd
import pytest

import ziggy.pipeline as pipe
import ziggy.providers.sec_edgar as sec
from ziggy.store import Store


@pytest.fixture
def isolated_cfg(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c.raw["paths"] = {k: str(tmp_path / k) for k in ("raw", "interim", "processed", "artifacts")}
    return c


@pytest.fixture
def normalised_filings():
    raw = pd.DataFrame({
        "cik": [1, 1, 2], "accession": ["a-1", "a-2", "b-1"],
        "form": ["10-Q", "10-Q", "8-K"],
        "filing_date": ["2021-01-05", "2021-04-05", "2021-02-01"],
        "report_date": [None] * 3,
        "acceptance_raw": ["2021-01-05T16:00:00.000Z", "2021-04-05T16:00:00.000Z",
                           "2021-02-01T16:00:00.000Z"],
        "primary_document": ["d.htm"] * 3, "primary_doc_description": [""] * 3,
        "items": ["", "", "2.02"], "size": [900000, 950000, 50000],
        "is_xbrl": 1, "is_inline_xbrl": 1, "act": "34", "file_number": "001",
    })
    return sec.normalise_filings(raw, "America/New_York")


def test_document_ingestion_produces_novelty_features(isolated_cfg, normalised_filings, monkeypatch):
    texts = {
        "a-1": "alpha beta gamma " * 200,
        "a-2": "alpha beta gamma " * 200 + "impairment restatement litigation " * 80,
        "b-1": "short filing text here",
    }
    monkeypatch.setattr(
        sec, "fetch_filing_text",
        lambda f, cik, acc, doc: next(
            (v for k, v in texts.items() if k.replace("-", "") == acc), ""
        ),
    )
    store = Store(isolated_cfg)
    out = pipe.ingest_documents(isolated_cfg, store, normalised_filings, force=True)

    assert len(out) == 3
    row = out.set_index("accession").loc["a-2"]
    assert row["text_cosine_prior"] < 1.0
    assert row["text_new_token_mass"] > 0.3
    assert row["text_neg_delta"] > 0
    # The issuer's first filing of that form has nothing to compare against.
    assert pd.isna(out.set_index("accession").loc["a-1", "text_cosine_prior"])
    assert store.exists("interim", "text_features")


def test_document_budget_is_respected_and_spent_on_the_largest(isolated_cfg, normalised_filings, monkeypatch):
    isolated_cfg.raw["sec"]["document_budget"] = 1
    seen = []

    def fake(f, cik, acc, doc):
        seen.append(acc)
        return "some text here for the filing"

    monkeypatch.setattr(sec, "fetch_filing_text", fake)
    pipe.ingest_documents(isolated_cfg, Store(isolated_cfg), normalised_filings, force=True)
    assert len(seen) == 1
    # a-2 is the largest submission; a 50KB 8-K has little text to be novel about.
    assert seen[0] == "a2"


def test_documents_can_be_switched_off(isolated_cfg, normalised_filings, monkeypatch):
    isolated_cfg.raw["sec"]["fetch_documents"] = False
    monkeypatch.setattr(sec, "fetch_filing_text",
                        lambda *a: pytest.fail("documents fetched despite being disabled"))
    assert pipe.ingest_documents(isolated_cfg, Store(isolated_cfg), normalised_filings, force=True).empty


def test_ingest_all_refuses_to_build_a_universe_without_prices(isolated_cfg, monkeypatch):
    monkeypatch.setattr(pipe, "ingest_ticker_map",
                        lambda cfg, store, force=False: pd.DataFrame({"cik": [1], "ticker": ["AAA"]}))
    monkeypatch.setattr(pipe, "ingest_prices", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipe, "ingest_benchmarks", lambda *a, **k: (pd.DataFrame(), pd.DataFrame()))
    with pytest.raises(RuntimeError, match="no price data"):
        pipe.ingest_all(isolated_cfg)


def test_extracted_text_is_persisted_for_downstream_use(isolated_cfg, normalised_filings, monkeypatch):
    """Experiment 2 reads the documents; it must not have to re-fetch them."""
    monkeypatch.setattr(sec, "fetch_filing_text", lambda f, cik, acc, doc: "material impairment charge")
    store = Store(isolated_cfg)
    pipe.ingest_documents(isolated_cfg, store, normalised_filings, force=True)

    assert store.exists("interim", "filing_text")
    txt = store.read("interim", "filing_text")
    assert len(txt) == 3
    assert {"accession", "cik", "form_base", "accepted_at", "source_url", "text"}.issubset(txt.columns)
    assert txt["text"].str.contains("impairment").all()
    assert txt["source_url"].str.startswith("https://www.sec.gov/").all()


def test_text_persistence_can_be_switched_off(isolated_cfg, normalised_filings, monkeypatch):
    isolated_cfg.raw["sec"]["persist_text"] = False
    monkeypatch.setattr(sec, "fetch_filing_text", lambda *a: "some text")
    store = Store(isolated_cfg)
    pipe.ingest_documents(isolated_cfg, store, normalised_filings, force=True)
    assert not store.exists("interim", "filing_text")
