"""Textual novelty: deterministic, causal, and actually sensitive to change."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from ziggy.features.text import (aggregate_text_to_events, compute_text_novelty,
                                 filing_text_stats, hashed_vector, tokenize)

BOILERPLATE = ("the company reported revenue growth in its core segment and "
               "continues to expect stable demand across its markets ") * 50
BAD_NEWS = ("we recorded a material impairment charge and disclosed a restatement "
            "following litigation and a covenant default ") * 30


def _docs(texts, form="10-Q", cik=1):
    return pd.DataFrame({
        "cik": cik, "form_base": form,
        "accepted_at": pd.date_range("2021-01-01", periods=len(texts), freq="90D"),
        "accession": [f"a{i}" for i in range(len(texts))],
        "text": texts,
    })


def test_an_identical_refiling_is_maximally_similar():
    out = compute_text_novelty(_docs([BOILERPLATE, BOILERPLATE]))
    assert out["text_cosine_prior"].iloc[1] == 1.0
    assert out["text_new_token_mass"].iloc[1] == 0.0
    assert out["text_len_log_ratio"].iloc[1] == 0.0


def test_the_first_filing_has_no_comparison():
    out = compute_text_novelty(_docs([BOILERPLATE]))
    assert pd.isna(out["text_cosine_prior"].iloc[0])


def test_new_language_lowers_similarity_and_raises_novelty():
    out = compute_text_novelty(_docs([BOILERPLATE, BOILERPLATE + BAD_NEWS]))
    r = out.iloc[1]
    assert r["text_cosine_prior"] < 0.95
    assert r["text_new_token_mass"] > 0.2
    assert r["text_len_log_ratio"] > 0
    assert r["text_neg_delta"] > 0


def test_comparison_is_per_issuer_and_per_form():
    docs = pd.DataFrame({
        "cik": [1, 2, 1], "form_base": ["10-Q", "10-Q", "10-K"],
        "accepted_at": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        "accession": ["a", "b", "c"], "text": [BOILERPLATE] * 3,
    })
    out = compute_text_novelty(docs).set_index("accession")
    # A different issuer, and a different form from the same issuer, are not
    # valid comparisons -- each is the first of its own series.
    assert pd.isna(out.loc["b", "text_cosine_prior"])
    assert pd.isna(out.loc["c", "text_cosine_prior"])


def test_comparison_is_always_backwards_in_time():
    """Shuffling input order must not change any filing's comparison."""
    docs = _docs([BOILERPLATE, BOILERPLATE + BAD_NEWS, BOILERPLATE])
    a = compute_text_novelty(docs).set_index("accession")["text_cosine_prior"]
    b = compute_text_novelty(docs.iloc[::-1].reset_index(drop=True)).set_index("accession")["text_cosine_prior"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_hashing_is_stable_across_processes():
    """Python's builtin hash() is salted per process; ours must not be."""
    code = (
        "from ziggy.features.text import hashed_vector, tokenize;"
        "v = hashed_vector(tokenize('alpha beta gamma delta'));"
        "print(round(float(v.sum()), 10))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
                       cwd=".").stdout.strip()
        for seed in (0, 1, 12345)
    }
    assert len(outs) == 1 and outs != {""}, outs


def test_empty_text_does_not_crash():
    stats = filing_text_stats("")
    assert stats["n_tokens"] == 0
    out = compute_text_novelty(_docs(["", BOILERPLATE]))
    assert len(out) == 2


def test_hashed_vector_is_unit_norm():
    v = hashed_vector(tokenize(BOILERPLATE))
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_aggregation_takes_the_most_novel_filing_of_the_evening():
    text_feats = pd.DataFrame({
        "accession": ["a", "b"], "text_cosine_prior": [0.99, 0.40],
        "text_new_token_mass": [0.01, 0.33], "text_dropped_mass": [0.0, 0.1],
        "text_len_log_ratio": [0.0, 1.2], "text_neg_density": [0.0, 0.05],
        "text_unc_density": [0.01, 0.02], "text_neg_delta": [0.0, 0.05],
        "text_unc_delta": [0.0, 0.01], "text_n_tokens": [100, 900],
    })
    filings = pd.DataFrame({
        "accession": ["a", "b"], "snapshot_session": pd.to_datetime(["2021-05-04"] * 2),
        "ticker": ["AAA", "AAA"],
    })
    out = aggregate_text_to_events(text_feats, filings)
    assert len(out) == 1
    assert out["text_cosine_prior"].iloc[0] == 0.40      # least similar wins
    assert out["text_new_token_mass"].iloc[0] == 0.33
