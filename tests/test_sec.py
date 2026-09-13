from __future__ import annotations

import pandas as pd

from ziggy.providers.sec_edgar import HIGH_SALIENCE_FAMILIES, extract_text, item_family, normalise_filings


def _raw(**over):
    base = {
        "cik": [320193], "accession": ["0000320193-21-000001"], "form": ["8-K"],
        "filing_date": ["2021-01-27"], "report_date": [None],
        "acceptance_raw": ["2021-01-27T16:31:22.000Z"], "primary_document": ["a8k.htm"],
        "primary_doc_description": [""], "items": ["2.02,9.01"], "size": [120000],
        "is_xbrl": [1], "is_inline_xbrl": [1], "act": ["34"], "file_number": ["001"],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_acceptance_is_read_as_eastern_wall_clock():
    df = normalise_filings(_raw(), "America/New_York")
    # 16:31 ET in January is 21:31 UTC. A naive UTC reading would leave 16:31Z.
    assert df["accepted_at"].iloc[0] == pd.Timestamp("2021-01-27 21:31:22", tz="UTC")


def test_missing_acceptance_falls_back_conservatively():
    df = normalise_filings(_raw(acceptance_raw=[None]), "America/New_York")
    assert bool(df["accepted_at_imputed"].iloc[0])
    assert df["accepted_at"].iloc[0] == pd.Timestamp("2021-01-27 22:30", tz="UTC")


def test_items_are_split_and_families_mapped():
    df = normalise_filings(_raw(), "America/New_York")
    assert df["item_list"].iloc[0] == ["2.02", "9.01"]
    assert df["n_items"].iloc[0] == 2
    assert item_family("2.02") == "results"
    assert item_family("2.06") == "impairment"
    assert "results" in HIGH_SALIENCE_FAMILIES


def test_unknown_item_code_does_not_raise():
    assert item_family("99.99") == "other_events"


def test_amendments_are_flagged_and_base_form_extracted():
    df = normalise_filings(_raw(form=["8-K/A"]), "America/New_York")
    assert bool(df["is_amendment"].iloc[0])
    assert df["form_base"].iloc[0] == "8-K"


def test_source_url_points_at_the_document():
    df = normalise_filings(_raw(), "America/New_York")
    url = df["source_url"].iloc[0]
    assert url.endswith("/a8k.htm") and "000032019321000001" in url


def test_extract_text_strips_markup_and_scripts():
    out = extract_text(b"<html><style>x{}</style><p>Net&nbsp;income rose</p><script>y</script></html>")
    assert out == "Net income rose"
