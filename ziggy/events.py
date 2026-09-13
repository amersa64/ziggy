"""Candidate and event construction.

A *candidate* is a (snapshot, ticker) pair: "should the expensive reasoner look
at this name tonight?". Every tradable name in the point-in-time universe is a
candidate every session -- that is what makes this an honest retrieval problem
rather than a pre-filtered one. Names with no disclosure simply carry zeroed
event features and have to earn their rank from price/flow behaviour.

An *event* is the cluster of everything a single issuer disclosed inside one
snapshot window ``(previous snapshot, this snapshot]``. Clustering matters: a
company that files an 8-K, a 10-Q and six Form 4s on the same evening is one
situation, not eight.
"""
from __future__ import annotations

import logging

import pandas as pd

from ziggy.providers.sec_edgar import HIGH_SALIENCE_FAMILIES, item_family

log = logging.getLogger(__name__)


def attach_snapshots(filings: pd.DataFrame, cal) -> pd.DataFrame:
    """Map each filing to the first snapshot that could legally have seen it."""
    df = filings.copy()
    df["snapshot_session"] = cal.snapshot_session(df["accepted_at"]).values
    df["available_at"] = df["accepted_at"]
    before = len(df)
    df = df.dropna(subset=["snapshot_session"])
    if len(df) < before:
        log.info("dropped %d filings outside the study calendar", before - len(df))
    return df


def map_filings_to_tickers(filings: pd.DataFrame, ticker_map: pd.DataFrame) -> pd.DataFrame:
    """Attach tickers to filings via CIK.

    A CIK with several share classes produces one row per class; each class is
    ranked independently because they trade independently.
    """
    tm = ticker_map[["cik", "ticker"]].drop_duplicates()
    out = filings.merge(tm, on="cik", how="inner")
    log.info("filings with a ticker mapping: %d of %d", len(out), len(filings))
    return out


def build_events(filings: pd.DataFrame) -> pd.DataFrame:
    """Collapse filings into one event row per (snapshot_session, ticker).

    Vectorised named aggregations rather than a per-group ``apply``: at real
    EDGAR volume this is roughly a million groups and the apply-based version
    costs minutes rather than seconds.
    """
    if filings.empty:
        return pd.DataFrame()
    df = filings.copy()
    df["fam_list"] = df["item_list"].apply(lambda xs: [item_family(x) for x in xs])

    fb = df["form_base"]
    df["_is_8k"] = (fb == "8-K").astype("int32")
    df["_is_periodic"] = fb.isin(["10-Q", "10-K", "20-F", "40-F"]).astype("int32")
    df["_is_form4"] = (fb == "4").astype("int32")
    df["_is_13d"] = fb.isin(["SC 13D", "SC 13G"]).astype("int32")
    df["_is_amend"] = df["is_amendment"].astype("int32")
    df["_n_salient"] = df["fam_list"].apply(
        lambda fams: sum(1 for f in fams if f in HIGH_SALIENCE_FAMILIES)
    ).astype("int32")

    keys = ["snapshot_session", "ticker"]
    g = df.groupby(keys, observed=True, sort=False)
    ev = g.agg(
        n_filings=("accession", "size"),
        n_8k=("_is_8k", "sum"),
        n_periodic=("_is_periodic", "sum"),
        n_form4=("_is_form4", "sum"),
        n_13d=("_is_13d", "sum"),
        n_amendments=("_is_amend", "sum"),
        n_high_salience=("_n_salient", "sum"),
        total_size=("size", "sum"),
        max_size=("size", "max"),
        first_accepted_at=("accepted_at", "min"),
        last_accepted_at=("accepted_at", "max"),
    ).reset_index()

    def _join_unique(frame: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
        sub = frame[keys + [col]].dropna(subset=[col])
        sub = sub[sub[col].astype(str).str.len() > 0].drop_duplicates()
        if sub.empty:
            return pd.DataFrame(columns=keys + [name])
        out = (
            sub.sort_values(keys + [col])
            .groupby(keys, observed=True, sort=False)[col]
            .agg("|".join)
            .reset_index()
            .rename(columns={col: name})
        )
        return out

    ev = ev.merge(_join_unique(df, "form_base", "forms"), on=keys, how="left")
    items = df[keys + ["item_list"]].explode("item_list")
    ev = ev.merge(_join_unique(items, "item_list", "item_codes"), on=keys, how="left")
    fams = df[keys + ["fam_list"]].explode("fam_list")
    ev = ev.merge(_join_unique(fams, "fam_list", "item_families"), on=keys, how="left")

    # Keep a handle back to the underlying documents for downstream inspection.
    biggest = df.sort_values("size", ascending=False).drop_duplicates(keys)
    ev = ev.merge(biggest[keys + ["source_url"]].rename(columns={"source_url": "primary_url"}),
                  on=keys, how="left")
    accs = (
        df.sort_values(keys + ["size"], ascending=[True, True, False])
        .groupby(keys, observed=True, sort=False)["accession"]
        .agg(lambda s: "|".join(s.astype(str).head(12)))
        .reset_index()
        .rename(columns={"accession": "accessions"})
    )
    ev = ev.merge(accs, on=keys, how="left")

    for c in ("forms", "item_codes", "item_families"):
        ev[c] = ev[c].fillna("")
    ev["available_at"] = ev["last_accepted_at"]
    log.info("events: %d (snapshot,ticker) clusters", len(ev))
    return ev
