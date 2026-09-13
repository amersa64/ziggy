"""Reachability probe for the external data sources.

Run before a long ingestion. In restricted environments (corporate proxies,
sandboxes) the failure mode is a silent 403 at CONNECT, which otherwise only
shows up two hours into a download.
"""
from __future__ import annotations

import requests

SOURCES = [
    ("SEC submissions API", "https://data.sec.gov/submissions/CIK0000320193.json"),
    ("SEC archives", "https://www.sec.gov/files/company_tickers.json"),
    ("SEC structured data", "https://www.sec.gov/files/structureddata/data/"
                            "insider-transactions-data-sets/2021q1_form345.zip"),
    ("stooq daily", "https://stooq.com/q/d/l/?s=aapl.us&i=d"),
    ("stooq bulk", "https://static.stooq.com/db/h/d_us_txt.zip"),
    ("yahoo chart", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=5d&interval=1d"),
    ("FRED csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"),
    ("ALFRED vintages", "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=UNRATE&vintage_date=2021-06-01"),
]


def check_sources(cfg, timeout: int = 20) -> list[dict]:
    out = []
    headers = {"User-Agent": cfg.http["user_agent"]}
    for name, url in SOURCES:
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            status = "OK" if r.status_code == 200 else f"HTTP{r.status_code}"
            r.close()
        except requests.RequestException as exc:
            status = type(exc).__name__
        out.append({"name": name, "url": url, "status": status})
    return out
