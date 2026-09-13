"""Textual novelty of filings.

The intuition: a routine 10-Q is ~95% identical to the last one. The interesting
filing is the one that says something it did not say before. We measure that
without any language model, using hashed token counts:

* ``text_cosine_prior``    -- cosine similarity to the issuer's previous filing
                              of the same form. Low = unusual.
* ``text_new_token_mass``  -- share of this filing's tokens absent from the prior.
* ``text_dropped_mass``    -- share of the prior's tokens missing here.
* ``text_len_log_ratio``   -- length change, the crudest and most robust signal.
* ``text_risk_delta``      -- change in density of hedging/negative language.

This is deterministic, cheap and auditable, which is the point: the expensive
reasoning happens downstream in Experiment 2, not here.
"""
from __future__ import annotations

import logging
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z][a-z'\-]{2,}")
HASH_DIM = 4096

# Small, fixed lexicons. Not tuned on outcomes; drawn from the standard
# Loughran-McDonald sentiment word families.
NEGATIVE_TERMS = {
    "adverse", "adversely", "decline", "declined", "deficiency", "delay", "delays",
    "deteriorate", "discontinued", "impairment", "impaired", "litigation", "loss",
    "losses", "restructuring", "shortfall", "terminate", "terminated", "termination",
    "unable", "weakness", "writedown", "write-down", "default", "breach", "investigation",
    "subpoena", "restatement", "going-concern", "covenant", "downgrade", "layoff", "layoffs",
}
UNCERTAIN_TERMS = {
    "may", "might", "could", "uncertain", "uncertainty", "risk", "risks", "believe",
    "approximately", "possible", "potential", "anticipate", "assume", "contingent",
    "depend", "fluctuate", "indefinite", "likelihood", "pending", "preliminary", "unclear",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    """CRC32, not ``hash()``: the builtin is salted per process, which would
    make every run of this pipeline produce different novelty numbers."""
    return zlib.crc32(token.encode("utf-8"))


def hashed_vector(tokens: list[str], dim: int = HASH_DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for t, c in Counter(tokens).items():
        v[_stable_hash(t) % dim] += c
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(np.dot(a, b))


def filing_text_stats(text: str) -> dict:
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return {"n_tokens": 0, "neg_density": np.nan, "unc_density": np.nan, "vec": np.zeros(HASH_DIM, np.float32), "vocab": set()}
    c = Counter(toks)
    return {
        "n_tokens": n,
        "neg_density": sum(c[w] for w in NEGATIVE_TERMS if w in c) / n,
        "unc_density": sum(c[w] for w in UNCERTAIN_TERMS if w in c) / n,
        "vec": hashed_vector(toks),
        "vocab": set(c),
    }


def compute_text_novelty(docs: pd.DataFrame) -> pd.DataFrame:
    """Novelty features per filing.

    ``docs`` needs ``cik, form_base, accepted_at, accession, text``. Comparison
    is always against the issuer's own chronologically previous filing of the
    same form -- never a later one.
    """
    if docs.empty:
        return pd.DataFrame(
            columns=["accession", "text_cosine_prior", "text_new_token_mass",
                     "text_dropped_mass", "text_len_log_ratio", "text_neg_density",
                     "text_unc_density", "text_neg_delta", "text_n_tokens"]
        )
    df = docs.sort_values(["cik", "form_base", "accepted_at"]).reset_index(drop=True)
    rows = []
    prev_key, prev_stats = None, None
    for r in df.itertuples(index=False):
        st = filing_text_stats(getattr(r, "text") or "")
        key = (r.cik, r.form_base)
        if prev_key == key and prev_stats is not None and st["n_tokens"] and prev_stats["n_tokens"]:
            cos = _cosine(st["vec"], prev_stats["vec"])
            new_mass = len(st["vocab"] - prev_stats["vocab"]) / max(len(st["vocab"]), 1)
            drop_mass = len(prev_stats["vocab"] - st["vocab"]) / max(len(prev_stats["vocab"]), 1)
            len_ratio = float(np.log1p(st["n_tokens"]) - np.log1p(prev_stats["n_tokens"]))
            neg_delta = float(st["neg_density"] - prev_stats["neg_density"])
            unc_delta = float(st["unc_density"] - prev_stats["unc_density"])
        else:
            cos = new_mass = drop_mass = len_ratio = neg_delta = unc_delta = np.nan
        rows.append(
            {
                "accession": r.accession,
                "text_cosine_prior": cos,
                "text_new_token_mass": new_mass,
                "text_dropped_mass": drop_mass,
                "text_len_log_ratio": len_ratio,
                "text_neg_density": st["neg_density"],
                "text_unc_density": st["unc_density"],
                "text_neg_delta": neg_delta,
                "text_unc_delta": unc_delta,
                "text_n_tokens": st["n_tokens"],
            }
        )
        prev_key, prev_stats = key, st
    return pd.DataFrame(rows)


def aggregate_text_to_events(
    text_feats: pd.DataFrame, filings: pd.DataFrame
) -> pd.DataFrame:
    """Roll per-filing novelty up to the (snapshot, ticker) event."""
    if text_feats.empty or filings.empty:
        return pd.DataFrame(columns=["snapshot_session", "ticker"])
    m = filings[["accession", "snapshot_session", "ticker"]].merge(
        text_feats, on="accession", how="inner"
    )
    if m.empty:
        return pd.DataFrame(columns=["snapshot_session", "ticker"])
    out = (
        m.groupby(["snapshot_session", "ticker"], observed=True)
        .agg(
            text_cosine_prior=("text_cosine_prior", "min"),
            text_new_token_mass=("text_new_token_mass", "max"),
            text_dropped_mass=("text_dropped_mass", "max"),
            text_len_log_ratio=("text_len_log_ratio", "max"),
            text_neg_density=("text_neg_density", "max"),
            text_unc_density=("text_unc_density", "max"),
            text_neg_delta=("text_neg_delta", "max"),
            text_unc_delta=("text_unc_delta", "max"),
            text_n_tokens=("text_n_tokens", "max"),
        )
        .reset_index()
    )
    return out
