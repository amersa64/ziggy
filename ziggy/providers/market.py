"""Historical daily market data behind a swappable provider interface.

The experiment only ever asks for ``get_daily``. Which vendor answers is a
configuration choice, because vendor coverage -- especially of *delisted*
securities -- is the single biggest threat to the honesty of a historical
study like this one. Each provider declares ``covers_delisted`` so the report
can state the survivorship position rather than guess at it.
"""
from __future__ import annotations

import io
import logging
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ziggy.net import Fetcher

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


class MarketDataProvider(ABC):
    name: str = "abstract"
    covers_delisted: bool = False
    adjustment: str = "unknown"

    @abstractmethod
    def get_daily(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Tidy daily bars. Columns: :data:`OHLCV_COLUMNS`."""

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS})


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return MarketDataProvider._empty()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    df["ticker"] = df["ticker"].astype("string").str.upper()
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[OHLCV_COLUMNS]
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[(df["close"] > 0) & (df["volume"].fillna(0) >= 0)]
    return df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
class StooqProvider(MarketDataProvider):
    """stooq.com free daily history.

    Prices are split-adjusted; dividends are *not* reinvested, so ``adj_close``
    equals ``close``. For a 1-to-21 session study the dividend drift is small
    relative to the effects measured, but it is a known bias and is reported.
    """

    name = "stooq"
    covers_delisted = True  # stooq retains history for many delisted US symbols
    adjustment = "split_adjusted"
    URL = "https://stooq.com/q/d/l/?s={sym}&i=d&d1={d1}&d2={d2}"

    def __init__(self, fetcher: Fetcher, workers: int = 4):
        self.fetcher, self.workers = fetcher, workers

    @staticmethod
    def to_symbol(ticker: str) -> str:
        t = ticker.strip().upper()
        if t.startswith("^"):
            return t.lower()
        return f"{t.replace('.', '-').lower()}.us"

    def _one(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        url = self.URL.format(
            sym=self.to_symbol(ticker),
            d1=pd.Timestamp(start).strftime("%Y%m%d"),
            d2=pd.Timestamp(end).strftime("%Y%m%d"),
        )
        res = self.fetcher.get(url)
        if res.status != 200 or len(res.content) < 40:
            return self._empty()
        head = res.content[:60].decode("utf-8", "replace")
        if not head.startswith("Date"):
            if "limit" in head.lower():
                log.error("stooq rate limit hit: %s", head.strip()[:80])
            return self._empty()
        df = pd.read_csv(io.BytesIO(res.content))
        df.columns = [c.strip().lower() for c in df.columns]
        if "close" not in df.columns:
            return self._empty()
        df["ticker"] = ticker.upper()
        df["adj_close"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = float("nan")
        return df

    def get_daily(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._one, t, start, end): t for t in tickers}
            for f in tqdm(as_completed(futs), total=len(futs), desc="stooq", unit="tkr"):
                try:
                    d = f.result()
                except Exception as exc:
                    log.warning("stooq %s failed: %s", futs[f], exc)
                    continue
                if not d.empty:
                    frames.append(d)
        return _finalise(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


class StooqBulkProvider(MarketDataProvider):
    """The stooq bulk daily archive: one zip for the whole US market.

    Vastly kinder to the vendor than tens of thousands of per-symbol requests and
    it includes symbols that have since been delisted.
    """

    name = "stooq_bulk"
    covers_delisted = True
    adjustment = "split_adjusted"
    URLS = [
        "https://static.stooq.com/db/h/d_us_txt.zip",
        "https://stooq.com/db/d/?b=d_us_txt",
    ]

    def __init__(self, fetcher: Fetcher, cache_dir: Path):
        self.fetcher, self.cache_dir = fetcher, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path | None:
        target = self.cache_dir / "d_us_txt.zip"
        if target.exists() and target.stat().st_size > 1_000_000:
            return target
        for url in self.URLS:
            res = self.fetcher.get(url)
            if res.status == 200 and len(res.content) > 1_000_000:
                target.write_bytes(res.content)
                return target
        return None

    def get_daily(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        path = self.download()
        if path is None:
            log.warning("stooq bulk archive unavailable")
            return self._empty()
        want = {t.upper() for t in tickers} if tickers else None
        frames = []
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".txt")]
            for m in tqdm(members, desc="stooq_bulk", unit="file"):
                sym = Path(m).stem.upper()
                tkr = sym[:-3].replace("-", ".") if sym.endswith(".US") else sym
                if want is not None and tkr not in want:
                    continue
                with zf.open(m) as fh:
                    try:
                        d = pd.read_csv(fh)
                    except Exception:
                        continue
                d.columns = [c.strip().strip("<>").lower() for c in d.columns]
                if "close" not in d.columns or "date" not in d.columns:
                    continue
                d["ticker"] = tkr
                d["adj_close"] = d["close"]
                if "volume" not in d.columns:
                    d["volume"] = float("nan")
                frames.append(d[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])
        out = _finalise(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        return out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]


class YahooProvider(MarketDataProvider):
    """Yahoo Finance chart endpoint. Carries dividend-adjusted closes.

    Delisted tickers are generally *not* retrievable, which is exactly the
    survivorship hole this study must not hide.
    """

    name = "yahoo"
    covers_delisted = False
    adjustment = "total_return"
    URL = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
    )

    def __init__(self, fetcher: Fetcher, workers: int = 4):
        self.fetcher, self.workers = fetcher, workers

    def _one(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        p2 = int(pd.Timestamp(end, tz="UTC").timestamp()) + 86400
        sym = ticker.replace(".", "-")
        js = self.fetcher.get_json(self.URL.format(sym=sym, p1=p1, p2=p2))
        try:
            res = js["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            ts = res["timestamp"]
        except (TypeError, KeyError, IndexError):
            return self._empty()
        adj = None
        try:
            adj = res["indicators"]["adjclose"][0]["adjclose"]
        except (KeyError, IndexError, TypeError):
            pass
        d = pd.DataFrame(
            {
                "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None),
                "ticker": ticker.upper(),
                "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                "close": q.get("close"),
                "adj_close": adj if adj is not None else q.get("close"),
                "volume": q.get("volume"),
            }
        )
        return d

    def get_daily(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._one, t, start, end): t for t in tickers}
            for f in tqdm(as_completed(futs), total=len(futs), desc="yahoo", unit="tkr"):
                try:
                    d = f.result()
                except Exception as exc:
                    log.warning("yahoo %s failed: %s", futs[f], exc)
                    continue
                if not d.empty:
                    frames.append(d)
        return _finalise(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


class CsvDirectoryProvider(MarketDataProvider):
    """Offline provider: one CSV per ticker in a directory. Used for fixtures."""

    name = "csv"
    covers_delisted = True
    adjustment = "as_supplied"

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def get_daily(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        want = {t.upper() for t in tickers} if tickers else None
        frames = []
        for p in sorted(self.directory.glob("*.csv")):
            tkr = p.stem.upper()
            if want is not None and tkr not in want:
                continue
            d = pd.read_csv(p)
            d.columns = [c.lower() for c in d.columns]
            d["ticker"] = tkr
            if "adj_close" not in d.columns:
                d["adj_close"] = d["close"]
            frames.append(d)
        out = _finalise(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        if out.empty:
            return out
        return out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]


def build_market_provider(cfg, fetcher: Fetcher, which: str | None = None) -> MarketDataProvider:
    which = which or cfg.market["provider"]
    if which == "stooq":
        return StooqProvider(fetcher)
    if which == "stooq_bulk":
        return StooqBulkProvider(fetcher, cfg.raw_dir / "stooq")
    if which == "yahoo":
        return YahooProvider(fetcher)
    if which == "csv":
        return CsvDirectoryProvider(cfg.raw_dir / "csv_prices")
    raise ValueError(f"unknown market provider {which!r}")
