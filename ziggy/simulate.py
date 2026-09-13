"""Synthetic point-in-time corpus.

This exists for one reason: to exercise and validate the pipeline and the
evaluation harness end-to-end without a network, and to act as a *positive
control* -- a world where a known amount of signal is present by construction,
so that a pipeline reporting "no signal" can be distinguished from a pipeline
that is simply broken.

It is NOT evidence about real markets, and every artifact generated from it is
stamped ``simulated``.

The generating process:

* a market factor with volatility regimes, plus sector factors;
* per-name idiosyncratic returns with fat tails and persistent volatility;
* IPOs mid-sample and delistings (price to zero, or an acquisition pop);
* a disclosure process -- periodic filings on a quarterly cadence, Form 4s at a
  per-name rate, and 8-Ks whose item mix mirrors real EDGAR frequencies;
* **the planted relationship**: an 8-K carrying a high-salience item family
  raises the *scale* of the next few days' idiosyncratic move, more so for
  smaller and more volatile names, and insider open-market buying raises it
  slightly. The effect is on volatility, not direction, which is exactly the
  claim Experiment 1 tests and deliberately weaker than a directional edge.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ziggy.calendar import build_calendar
from ziggy.store import Store
from ziggy.universe import build_price_panel, build_universe, survivorship_report

log = logging.getLogger(__name__)

N_NAMES = 1800
SECTORS = 11

ITEM_POOL = [
    ("2.02,9.01", 0.24), ("8.01", 0.16), ("5.02", 0.12), ("7.01,9.01", 0.09),
    ("1.01,9.01", 0.07), ("5.07", 0.07), ("2.01,9.01", 0.04), ("3.02", 0.04),
    ("2.03", 0.03), ("5.03", 0.03), ("2.05", 0.02), ("2.06", 0.02),
    ("4.02", 0.01), ("4.01", 0.01), ("3.01", 0.01), ("1.03", 0.005),
    ("5.01", 0.005), ("1.05", 0.005), ("2.04", 0.005),
]
SALIENT_ITEMS = {"2.02", "2.01", "2.05", "2.06", "4.01", "4.02", "3.01", "1.03", "5.01", "2.04", "1.05"}


def _sessions(cfg) -> pd.DatetimeIndex:
    cal = build_calendar(cfg)
    start = pd.Timestamp(cfg.experiment["start_date"]) - pd.Timedelta(days=540)
    full = build_calendar_range(cfg, start)
    return full


def build_calendar_range(cfg, start: pd.Timestamp) -> pd.DatetimeIndex:
    from ziggy.calendar import TradingCalendar

    cal = TradingCalendar(start.strftime("%Y-%m-%d"), str(cfg.experiment["end_date"]),
                          cfg.experiment["snapshot_time_local"], cfg.experiment["timezone"])
    return cal.session_index()


def generate(cfg, n_names: int = N_NAMES) -> dict:
    rng = np.random.default_rng(cfg.seed)
    store = Store(cfg)
    sessions = _sessions(cfg)
    T = len(sessions)
    log.info("simulating %d names over %d sessions (%s..%s)", n_names, T,
             sessions[0].date(), sessions[-1].date())

    tickers = np.array([f"SIM{i:04d}" for i in range(n_names)])
    sector = rng.integers(0, SECTORS, n_names)
    # Size: log-uniform market caps, driving liquidity and volatility.
    log_cap = rng.uniform(np.log(3e8), np.log(2e12), n_names)
    size_z = (log_cap - log_cap.mean()) / log_cap.std()
    base_vol = np.clip(0.42 - 0.10 * size_z + rng.normal(0, 0.08, n_names), 0.12, 1.20)
    beta = np.clip(1.0 + 0.25 * rng.normal(0, 1, n_names) - 0.12 * size_z, 0.2, 2.8)

    # -- listing lifecycle ------------------------------------------------------
    first_idx = np.zeros(n_names, dtype=int)
    ipo = rng.random(n_names) < 0.18
    first_idx[ipo] = rng.integers(0, int(T * 0.75), ipo.sum())
    last_idx = np.full(n_names, T - 1, dtype=int)
    delist = rng.random(n_names) < 0.16          # ~2.3%/yr over a 7-year window
    dl_idx = rng.integers(int(T * 0.15), T - 1, n_names)
    last_idx[delist] = np.maximum(dl_idx[delist], first_idx[delist] + 300)
    last_idx = np.minimum(last_idx, T - 1)
    acquired = delist & (rng.random(n_names) < 0.55)   # rest are failures

    # -- market and sector factors ---------------------------------------------
    regime = np.zeros(T)
    r, vol_state = 0, np.array([0.55, 0.95, 1.9])
    for t in range(T):
        if rng.random() < 0.012:
            r = rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15])
        regime[t] = vol_state[r]
    mkt = rng.normal(0.0004, 0.009, T) * regime
    sec_f = rng.normal(0, 0.006, (T, SECTORS)) * regime[:, None]
    vix = np.clip(11 + 13 * regime + rng.normal(0, 1.6, T), 9, 85)

    # -- disclosure process -----------------------------------------------------
    item_codes = [c for c, _ in ITEM_POOL]
    item_p = np.array([p for _, p in ITEM_POOL], dtype=float)
    item_p /= item_p.sum()

    eightk_rate = np.clip(0.010 + 0.006 * rng.random(n_names), 0.004, 0.03)
    form4_rate = np.clip(0.020 + 0.030 * rng.random(n_names), 0.004, 0.09)

    alive = (np.arange(T)[None, :] >= first_idx[:, None]) & (np.arange(T)[None, :] <= last_idx[:, None])
    eightk = (rng.random((n_names, T)) < eightk_rate[:, None]) & alive
    # Periodic reports: roughly every 63 sessions, offset per name.
    offs = rng.integers(0, 63, n_names)
    periodic = ((np.arange(T)[None, :] - offs[:, None]) % 63 == 0) & alive
    form4 = (rng.random((n_names, T)) < form4_rate[:, None]) & alive

    # Assign item codes and salience to each 8-K.
    n8 = int(eightk.sum())
    chosen = rng.choice(len(item_codes), size=n8, p=item_p)
    salient_flag = np.array([
        any(c in SALIENT_ITEMS for c in item_codes[i].split(",")) for i in chosen
    ])
    salience_grid = np.zeros((n_names, T))
    rows8, cols8 = np.where(eightk)
    salience_grid[rows8, cols8] = salient_flag.astype(float)

    # -- the planted relationship -----------------------------------------------
    # A salient 8-K accepted in snapshot window t raises the scale of the
    # idiosyncratic move over sessions t+1..t+5. Small, volatile names react more.
    react = np.clip(1.0 - 0.22 * size_z, 0.5, 2.2)
    shock = np.zeros((n_names, T))
    for lag in range(1, 6):
        shock[:, lag:] += salience_grid[:, :-lag] * react[:, None] * (1.0 if lag <= 2 else 0.55)
    shock += 0.35 * np.roll(eightk.astype(float), 1, axis=1) * react[:, None]
    shock += 0.30 * np.roll(periodic.astype(float), 1, axis=1) * react[:, None]

    idio_scale = base_vol[:, None] / np.sqrt(252) * regime[None, :] * (1.0 + 0.85 * shock)
    idio = rng.standard_t(df=4, size=(n_names, T)) / np.sqrt(2.0) * idio_scale
    ret = beta[:, None] * mkt[None, :] + sec_f[:, sector].T + idio
    ret = np.clip(ret, -0.55, 0.80)

    # Delisting outcomes land on the final bar.
    for i in np.where(delist)[0]:
        j = last_idx[i]
        ret[i, j] = rng.uniform(0.15, 0.45) if acquired[i] else -rng.uniform(0.25, 0.70)

    price = np.zeros((n_names, T))
    p0 = np.exp(rng.uniform(np.log(4), np.log(320), n_names))
    for i in range(n_names):
        a, b = first_idx[i], last_idx[i]
        price[i, a:b + 1] = p0[i] * np.exp(np.cumsum(ret[i, a:b + 1]))
    price = np.where(alive, np.maximum(price, 0.2), np.nan)

    # Volume: baseline from size, with event and volatility driven spikes.
    base_shares = np.exp(rng.normal(13.0, 1.1, n_names) + 0.55 * size_z)
    vmult = 1.0 + 2.2 * (eightk | periodic) + 1.1 * np.abs(ret) / (base_vol[:, None] / np.sqrt(252))
    volume = base_shares[:, None] * vmult * np.exp(rng.normal(0, 0.35, (n_names, T)))
    volume = np.where(alive, volume, np.nan)

    openp = price * (1 + rng.normal(0, 0.004, (n_names, T)))
    high = np.fmax(price, openp) * (1 + np.abs(rng.normal(0, 0.006, (n_names, T))))
    low = np.fmin(price, openp) * (1 - np.abs(rng.normal(0, 0.006, (n_names, T))))

    def tidy(mat_map: dict[str, np.ndarray], names: np.ndarray) -> pd.DataFrame:
        frames = {}
        for key, m in mat_map.items():
            frames[key] = pd.DataFrame(m.T, index=sessions, columns=names).stack(future_stack=True)
        df = pd.DataFrame(frames).reset_index()
        df.columns = ["date", "ticker"] + list(mat_map)
        return df.dropna(subset=["close"])

    prices = tidy({"open": openp, "high": high, "low": low, "close": price,
                   "adj_close": price, "volume": volume}, tickers)
    log.info("simulated price rows: %d", len(prices))

    # -- benchmarks and VIX -----------------------------------------------------
    bench_rows = []
    for sym, b, extra in (("SPY", 1.0, 0.0), ("QQQ", 1.15, 0.002), ("IWM", 1.1, -0.001)):
        p = 300 * np.exp(np.cumsum(b * mkt + rng.normal(extra / 252, 0.002, T)))
        bench_rows.append(pd.DataFrame({"date": sessions, "ticker": sym, "open": p * 0.999,
                                        "high": p * 1.004, "low": p * 0.996, "close": p,
                                        "adj_close": p, "volume": 8e7}))
    bench = pd.concat(bench_rows, ignore_index=True)
    vix_df = pd.DataFrame({"date": sessions, "ticker": "^VIX", "open": vix, "high": vix * 1.05,
                           "low": vix * 0.95, "close": vix, "adj_close": vix, "volume": 0.0})

    # -- filings ----------------------------------------------------------------
    ciks = 1000000 + np.arange(n_names)
    ticker_map = pd.DataFrame({"cik": ciks, "ticker": tickers,
                               "company": [f"Simulated Co {i}" for i in range(n_names)],
                               "exchange": "SIM"})

    def filing_rows(mask: np.ndarray, form_fn, size_lo, size_hi, items=None) -> pd.DataFrame:
        ri, ci = np.where(mask)
        if len(ri) == 0:
            return pd.DataFrame()
        d = sessions[ci]
        # Acceptance times: a realistic bimodal mix of pre-open and post-close.
        u = rng.random(len(ri))
        hour = np.where(u < 0.30, rng.integers(6, 9, len(ri)),
                        np.where(u < 0.80, rng.integers(16, 21, len(ri)), rng.integers(9, 16, len(ri))))
        minute = rng.integers(0, 60, len(ri))
        acc = pd.to_datetime(d) + pd.to_timedelta(hour, "h") + pd.to_timedelta(minute, "m")
        return pd.DataFrame({
            "cik": ciks[ri],
            "accession": [f"{ciks[a]:010d}-{b % 100:02d}-{i:06d}" for i, (a, b) in enumerate(zip(ri, ci))],
            "form": form_fn(len(ri)),
            "filing_date": pd.to_datetime(d).strftime("%Y-%m-%d"),
            "report_date": None,
            "acceptance_raw": acc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "primary_document": "doc.htm",
            "primary_doc_description": "",
            "items": items if items is not None else "",
            "size": rng.integers(size_lo, size_hi, len(ri)),
            "is_xbrl": 1, "is_inline_xbrl": 1, "act": "34", "file_number": "001-00000",
        })

    items_per_8k = np.array(item_codes, dtype=object)[chosen]
    f8 = filing_rows(eightk, lambda n: "8-K", 20_000, 900_000, items=items_per_8k)
    fp = filing_rows(periodic, lambda n: rng.choice(["10-Q", "10-K"], n, p=[0.75, 0.25]), 400_000, 6_000_000)
    f4 = filing_rows(form4, lambda n: "4", 3_000, 30_000)
    filings = pd.concat([x for x in (f8, fp, f4) if len(x)], ignore_index=True)
    start_keep = (pd.Timestamp(cfg.experiment["start_date"]) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    filings = filings[filings["filing_date"] >= start_keep].reset_index(drop=True)
    log.info("simulated filings: %d", len(filings))

    # -- insider transactions ---------------------------------------------------
    ri, ci = np.where(form4)
    keep = rng.random(len(ri)) < 0.45
    ri, ci = ri[keep], ci[keep]
    is_buy = rng.random(len(ri)) < 0.35
    val = np.exp(rng.normal(11.5, 1.4, len(ri)))
    insider = pd.DataFrame({
        "accession": [f"ins-{i}" for i in range(len(ri))],
        "issuer_cik": ciks[ri], "ticker": tickers[ri],
        "filing_date": sessions[ci],
        "transaction_date": sessions[np.maximum(ci - 2, 0)],
        "transaction_code": np.where(is_buy, "P", "S"),
        "acquired_disposed": np.where(is_buy, "A", "D"),
        "shares": val / 50.0, "price": 50.0, "shares_after": 0.0,
        "direct_indirect": "D", "document_type": "4",
        "value": val, "signed_value": np.where(is_buy, val, -val), "is_open_market": True,
    })

    # -- macro ------------------------------------------------------------------
    macro_rows = []
    for sid, lag, revisable, base, scale in [
        ("DGS10", 1, False, 2.5, 0.6), ("DGS2", 1, False, 2.2, 0.8),
        ("T10Y2Y", 1, False, 0.3, 0.5), ("BAMLH0A0HYM2", 1, False, 4.2, 1.1),
        ("VIXCLS", 1, False, 0.0, 0.0), ("UNRATE", 21, True, 4.2, 0.5),
        ("CPIAUCSL", 21, True, 290.0, 12.0),
    ]:
        if sid == "VIXCLS":
            vals = vix
        else:
            vals = base + scale * np.cumsum(rng.normal(0, 0.02, T)) / np.sqrt(T) * 8
        avail = (pd.to_datetime(sessions) + pd.Timedelta(days=lag) + pd.Timedelta(hours=18)) \
            .tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        macro_rows.append(pd.DataFrame({"series_id": sid, "date": sessions, "value": vals,
                                        "vintage_date": pd.NaT, "available_at": avail,
                                        "revisable": revisable}))
    macro = pd.concat(macro_rows, ignore_index=True)

    # -- persist exactly where ingestion would have put it ----------------------
    store.write(ticker_map, "raw", "ticker_map")
    store.write(prices, "raw", "prices")
    store.write(bench, "raw", "benchmarks")
    store.write(vix_df, "raw", "vix")
    store.write(filings, "raw", "filings")
    store.write(insider, "raw", "insider")
    store.write(macro, "raw", "macro")
    store.write_json({"provider": "simulated", "covers_delisted": True,
                      "adjustment": "none_needed", "SIMULATED": True,
                      "warning": "synthetic data; not evidence about real markets"},
                     "raw", "prices_provenance")

    study = build_calendar(cfg).session_index()
    panel = build_price_panel(prices, sessions)
    universe = build_universe(panel, cfg, sessions, study_sessions=study)
    store.write(universe, "interim", "universe")
    store.write_json(survivorship_report(panel, universe, study), "artifacts", "survivorship_report")

    return {"simulated": True, "names": n_names, "sessions": T, "price_rows": len(prices),
            "filings": len(filings), "insider": len(insider),
            "universe_rows": len(universe), "delisted": int(delist.sum()), "ipos": int(ipo.sum())}
