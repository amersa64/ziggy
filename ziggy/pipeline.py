"""End-to-end ingestion and dataset construction.

Ordering matters and is deliberate:

1. Prices first, for every ticker SEC knows about. Prices define the
   point-in-time tradability universe.
2. The universe then tells us which CIKs are worth a submissions request, which
   turns "every filer on EDGAR" into a few thousand companies.
3. Filings, insider datasets and macro follow.
4. Only then are features built, and labels last -- in a separate module that
   nothing upstream imports.
"""
from __future__ import annotations

import logging

import pandas as pd

from ziggy.calendar import build_calendar, build_calendar_with_warmup
from ziggy.events import attach_snapshots, build_events, map_filings_to_tickers
from ziggy.features.assemble import build_candidate_matrix
from ziggy.features.macro import compute_macro_features
from ziggy.features.price import compute_breadth, compute_market_context, compute_price_features
from ziggy.features.sec import compute_event_features, compute_grid_sec_features, compute_insider_features
from ziggy.features.text import aggregate_text_to_events, compute_text_novelty
from ziggy.labels import add_consequence_labels, attach_beta_adjusted, compute_forward_returns
from ziggy.net import generic_fetcher, sec_fetcher
from ziggy.providers import macro as macro_provider
from ziggy.providers import sec_edgar as sec
from ziggy.providers.market import build_market_provider
from ziggy.store import Store
from ziggy.universe import build_price_panel, build_universe, survivorship_report

log = logging.getLogger(__name__)

PRICE_WARMUP_DAYS = 520  # enough to satisfy the 252-session feature windows


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest_ticker_map(cfg, store: Store, force: bool = False) -> pd.DataFrame:
    if store.exists("raw", "ticker_map") and not force:
        return store.read("raw", "ticker_map")
    tm = sec.fetch_ticker_map(sec_fetcher(cfg))
    store.write(tm, "raw", "ticker_map")
    return tm


def ingest_prices(cfg, store: Store, tickers: list[str], force: bool = False) -> pd.DataFrame:
    if store.exists("raw", "prices") and not force:
        return store.read("raw", "prices")
    start = (pd.Timestamp(cfg.experiment["start_date"]) - pd.Timedelta(days=PRICE_WARMUP_DAYS)).strftime("%Y-%m-%d")
    end = str(cfg.experiment["end_date"])
    fetcher = generic_fetcher(cfg, namespace=cfg.market["provider"])
    provider = build_market_provider(cfg, fetcher)
    px = provider.get_daily(tickers, start, end)
    if px.empty:
        for alt in cfg.market.get("fallback_providers", []):
            log.warning("primary market provider returned nothing; trying %s", alt)
            px = build_market_provider(cfg, generic_fetcher(cfg, namespace=alt), alt).get_daily(
                tickers, start, end
            )
            if not px.empty:
                break
    store.write(px, "raw", "prices")
    store.write_json({"provider": provider.name, "covers_delisted": provider.covers_delisted,
                      "adjustment": provider.adjustment, "start": start, "end": end,
                      "tickers_requested": len(tickers),
                      "tickers_returned": int(px["ticker"].nunique()) if len(px) else 0},
                     "raw", "prices_provenance")
    return px


def ingest_benchmarks(cfg, store: Store, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if store.exists("raw", "benchmarks") and not force:
        b = store.read("raw", "benchmarks")
        v = store.read("raw", "vix") if store.exists("raw", "vix") else pd.DataFrame()
        return b, v
    start = (pd.Timestamp(cfg.experiment["start_date"]) - pd.Timedelta(days=PRICE_WARMUP_DAYS)).strftime("%Y-%m-%d")
    end = str(cfg.experiment["end_date"])
    fetcher = generic_fetcher(cfg, namespace=cfg.market["provider"])
    provider = build_market_provider(cfg, fetcher)
    bench = provider.get_daily(list(cfg.market["benchmarks"]), start, end)
    vix = provider.get_daily([cfg.market["vix_symbol"]], start, end)
    store.write(bench, "raw", "benchmarks")
    store.write(vix, "raw", "vix")
    return bench, vix


def ingest_filings(cfg, store: Store, ciks: list[int], force: bool = False) -> pd.DataFrame:
    if store.exists("raw", "filings") and not force:
        return store.read("raw", "filings")
    forms = set(cfg.sec["forms"]["core"]) | set(cfg.sec["forms"]["extended"])
    start = (pd.Timestamp(cfg.experiment["start_date"]) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    filings, profiles = sec.fetch_filings_bulk(
        sec_fetcher(cfg), ciks, start_date=start, forms=forms
    )
    store.write(filings, "raw", "filings")
    store.write(profiles, "raw", "company_profiles")
    return filings


def ingest_insider(cfg, store: Store, force: bool = False) -> pd.DataFrame:
    if store.exists("raw", "insider") and not force:
        return store.read("raw", "insider")
    if not cfg.sec.get("use_form345_datasets", True):
        return pd.DataFrame()
    f = sec_fetcher(cfg, namespace="sec_bulk")
    start = pd.Timestamp(cfg.experiment["start_date"])
    end = pd.Timestamp(cfg.experiment["end_date"])
    frames = []
    for ts in pd.period_range(start, end, freq="Q"):
        parts = sec.fetch_form345_quarter(f, ts.year, ts.quarter)
        tx = sec.parse_insider_transactions(parts)
        if not tx.empty:
            frames.append(tx)
            log.info("form345 %dQ%d: %d transactions", ts.year, ts.quarter, len(tx))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    store.write(out, "raw", "insider")
    return out


def ingest_macro(cfg, store: Store, force: bool = False) -> pd.DataFrame:
    if store.exists("raw", "macro") and not force:
        return store.read("raw", "macro")
    panel = macro_provider.build_macro_panel(generic_fetcher(cfg, namespace="fred"), cfg)
    store.write(panel, "raw", "macro")
    return panel


def ingest_documents(cfg, store: Store, filings: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Primary documents for the filings most likely to carry novel text."""
    if store.exists("interim", "text_features") and not force:
        return store.read("interim", "text_features")
    if not cfg.sec.get("fetch_documents", True):
        return pd.DataFrame()
    budget = int(cfg.sec.get("document_budget", 0))
    cand = filings[filings["form_base"].isin(["8-K", "10-Q", "10-K"])].copy()
    cand = cand[cand["primary_document"].astype(str).str.len() > 0]
    # Spend the budget on the largest filings of each form: a 400-byte 8-K has
    # no text to be novel about.
    cand = cand.sort_values("size", ascending=False).head(budget)
    log.info("fetching %d filing documents (budget %d)", len(cand), budget)
    f = sec_fetcher(cfg, namespace="sec_docs")
    texts = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    def one(row):
        return row.accession, sec.fetch_filing_text(f, row.cik, row.accession_nodash, row.primary_document)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(one, r) for r in cand.itertuples(index=False)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="documents", unit="doc"):
            try:
                acc, txt = fut.result()
            except Exception:
                continue
            if txt:
                texts.append({"accession": acc, "text": txt})
    if not texts:
        return pd.DataFrame()
    # `ticker` is attached later, at the event stage, so it may or may not be
    # present here depending on the caller.
    meta = ["accession", "cik", "ticker", "form_base", "accepted_at", "items",
            "source_url", "size"]
    td = pd.DataFrame(texts).merge(
        cand[[c for c in meta if c in cand.columns]], on="accession", how="left"
    )
    feats = compute_text_novelty(td)
    store.write(feats, "interim", "text_features")

    # Persist the extracted text itself, not only the features derived from it.
    # Experiment 2's whole job is to read these documents; handing it a cosine
    # distance and a URL would make it re-fetch everything this run already has.
    if cfg.sec.get("persist_text", True):
        keep = ["accession", "cik", "ticker", "form_base", "accepted_at", "items",
                "source_url", "size", "text"]
        store.write(td[[c for c in keep if c in td.columns]], "interim", "filing_text")
    return feats


def ingest_all(cfg, force: bool = False) -> dict:
    store = Store(cfg)
    cal = build_calendar(cfg)

    tm = ingest_ticker_map(cfg, store, force)
    tickers = sorted(tm["ticker"].dropna().unique().tolist())
    log.info("candidate tickers from SEC mapping: %d", len(tickers))

    prices = ingest_prices(cfg, store, tickers, force)
    bench, vix = ingest_benchmarks(cfg, store, force)
    if prices.empty:
        raise RuntimeError("no price data: cannot build a universe")

    study = cal.session_index()
    full = build_calendar_with_warmup(cfg, PRICE_WARMUP_DAYS).session_index()
    panel = build_price_panel(prices, full)
    universe = build_universe(panel, cfg, full, study_sessions=study)
    store.write(universe, "interim", "universe")
    store.write_json(survivorship_report(panel, universe, study), "artifacts", "survivorship_report")

    uni_tickers = set(universe["ticker"].unique())
    ciks = sorted(tm[tm["ticker"].isin(uni_tickers)]["cik"].unique().tolist())
    log.info("CIKs to query on EDGAR: %d", len(ciks))

    filings_raw = ingest_filings(cfg, store, ciks, force)
    insider = ingest_insider(cfg, store, force)
    macro = ingest_macro(cfg, store, force)

    # Documents last, and deliberately so: it is by far the longest-running step
    # and everything else is already usable without it. A run interrupted here
    # still produces a complete experiment, minus the text-novelty features.
    text_feats = pd.DataFrame()
    if cfg.sec.get("fetch_documents", True) and not filings_raw.empty:
        normalised = sec.normalise_filings(filings_raw, cfg.sec["acceptance_timezone"])
        text_feats = ingest_documents(cfg, store, normalised, force)

    return {
        "ticker_map": len(tm), "prices": len(prices), "universe": len(universe),
        "filings": len(filings_raw), "insider": len(insider), "macro": len(macro),
        "benchmarks": len(bench), "vix": len(vix), "documents": len(text_feats),
    }


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_dataset(cfg, force: bool = False) -> pd.DataFrame:
    store = Store(cfg)
    cal = build_calendar(cfg)
    full_cal = build_calendar_with_warmup(cfg, PRICE_WARMUP_DAYS)
    sessions = full_cal.session_index()      # includes warmup: feature windows
    study = cal.session_index()              # the period actually evaluated
    snaps = cal.snapshot_table()

    def _in_study(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["session"].isin(study)]

    prices = store.read("raw", "prices")
    bench = store.read("raw", "benchmarks")
    vix = store.read("raw", "vix") if store.exists("raw", "vix") else pd.DataFrame()
    tm = store.read("raw", "ticker_map")
    universe = store.read("interim", "universe")

    # -- filings -> events ----------------------------------------------------
    filings_raw = store.read("raw", "filings") if store.exists("raw", "filings") else pd.DataFrame()
    if not filings_raw.empty:
        filings = sec.normalise_filings(filings_raw, cfg.sec["acceptance_timezone"])
        # Snapshots are attached on the *warmup* calendar so that disclosure from
        # before the study window still feeds the trailing filing-intensity
        # windows. Without this the first quarter of the study would believe
        # every issuer had been silent for a year.
        filings = map_filings_to_tickers(attach_snapshots(filings, full_cal), tm)
        filings = filings[filings["ticker"].isin(set(universe["ticker"]))]
        events = build_events(filings)
        store.write(filings.drop(columns=["item_list", "fam_list"], errors="ignore"), "interim", "filings_normalised")
        store.write(events, "interim", "events")
    else:
        filings, events = pd.DataFrame(), pd.DataFrame()

    # -- features -------------------------------------------------------------
    panel = build_price_panel(prices, sessions)
    price_feats = compute_price_features(panel, bench, cfg.labels["benchmark"])
    tickers = pd.Index(sorted(universe["ticker"].unique()))

    grid_sec = compute_grid_sec_features(events, sessions, tickers) if len(events) else pd.DataFrame()
    ev_feats = compute_event_features(events, filings) if len(events) else pd.DataFrame()

    insider = store.read("raw", "insider") if store.exists("raw", "insider") else pd.DataFrame()
    ins_feats = (
        compute_insider_features(insider, full_cal, sessions, tickers) if len(insider) else pd.DataFrame()
    )

    text_feats = pd.DataFrame()
    if store.exists("interim", "text_features") and len(filings):
        tf = store.read("interim", "text_features")
        text_feats = aggregate_text_to_events(tf, filings)

    mkt_ctx = compute_market_context(bench, vix)
    breadth = compute_breadth(price_feats)
    macro_panel = store.read("raw", "macro") if store.exists("raw", "macro") else pd.DataFrame()
    macro_feats = compute_macro_features(macro_panel, snaps)

    matrix = build_candidate_matrix(
        universe, snaps, price_feats, grid_sec, ev_feats, ins_feats, text_feats,
        mkt_ctx, breadth, macro_feats,
    )
    # The universe is already restricted to the study window; this makes the
    # warmup/study boundary explicit rather than implicit in the upstream join.
    matrix = _in_study(matrix)

    # -- labels (the only forward-looking step) --------------------------------
    horizons = list(cfg.labels["horizons"])
    fwd = compute_forward_returns(panel, bench, sessions, horizons, cfg.labels["benchmark"])
    fwd = attach_beta_adjusted(fwd, price_feats[["session", "ticker", "beta_126d"]], horizons)
    matrix = matrix.merge(fwd, on=["session", "ticker"], how="left")
    matrix = add_consequence_labels(matrix, cfg, trailing_vol=price_feats[["session", "ticker", "vol_21d"]])

    store.write(matrix, "processed", "candidates")
    log.info("dataset: %d rows, %d columns", len(matrix), matrix.shape[1])
    return matrix
