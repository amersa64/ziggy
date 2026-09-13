"""SEC EDGAR ingestion.

Design notes
------------
* The *submissions* API (``data.sec.gov/submissions/CIK##########.json``) is the
  backbone: one request per company returns that company's entire filing history
  with ``acceptanceDateTime`` (the instant EDGAR accepted the document), form
  type, 8-K item codes, submission size and the primary document name. That is
  the whole event skeleton without downloading a single filing document.
* ``acceptanceDateTime`` is serialised with a trailing ``Z`` but the wall clock
  is US Eastern. We localise using ``sec.acceptance_timezone`` and verify the
  choice empirically in :mod:`ziggy.audit` (an ET reading puts the 8-K mode just
  after the 16:00 close; a UTC reading would put it near midnight).
* Insider transactions come from the quarterly *Form 345 structured data sets*
  rather than from parsing 1.5M individual Form 4 XML documents.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ziggy.net import Fetcher

log = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_SHARD_URL = "https://data.sec.gov/submissions/{name}"
FORM345_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{yr}q{q}_form345.zip"
ARCHIVE_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
ARCHIVE_DIR_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"


# --------------------------------------------------------------------------- #
# Ticker <-> CIK mapping
# --------------------------------------------------------------------------- #
def fetch_ticker_map(fetcher: Fetcher) -> pd.DataFrame:
    """CIK / ticker / company-name / exchange mapping as published by SEC.

    Caveat recorded in the limitations section of the report: this file is the
    *current* mapping. It is not point-in-time, so a company that changed ticker
    is mapped under its latest symbol. It is used only to attach filings to price
    series, never as a universe filter.
    """
    rows: list[dict] = []
    ex = fetcher.get_json(COMPANY_TICKERS_EXCHANGE_URL)
    if ex and "data" in ex:
        fields = [f.lower() for f in ex["fields"]]
        for rec in ex["data"]:
            d = dict(zip(fields, rec))
            rows.append(
                {
                    "cik": int(d["cik"]),
                    "ticker": str(d["ticker"]).upper(),
                    "company": d.get("name"),
                    "exchange": d.get("exchange"),
                }
            )
    base = fetcher.get_json(COMPANY_TICKERS_URL)
    if base:
        known = {(r["cik"], r["ticker"]) for r in rows}
        for rec in base.values():
            key = (int(rec["cik_str"]), str(rec["ticker"]).upper())
            if key not in known:
                rows.append(
                    {"cik": key[0], "ticker": key[1], "company": rec.get("title"), "exchange": None}
                )
    df = pd.DataFrame(rows).drop_duplicates(subset=["cik", "ticker"])
    log.info("ticker map: %d cik/ticker pairs, %d distinct CIKs", len(df), df.cik.nunique())
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Filing metadata
# --------------------------------------------------------------------------- #
_FILING_COLS = [
    "cik", "accession", "form", "filing_date", "report_date", "acceptance_raw",
    "primary_document", "primary_doc_description", "items", "size", "is_xbrl",
    "is_inline_xbrl", "act", "file_number",
]


def _rows_from_block(block: dict, cik: int) -> pd.DataFrame:
    if not block or not block.get("accessionNumber"):
        return pd.DataFrame(columns=_FILING_COLS)
    n = len(block["accessionNumber"])

    def col(key, default=None):
        v = block.get(key)
        return v if isinstance(v, list) and len(v) == n else [default] * n

    return pd.DataFrame(
        {
            "cik": cik,
            "accession": col("accessionNumber"),
            "form": col("form"),
            "filing_date": col("filingDate"),
            "report_date": col("reportDate"),
            "acceptance_raw": col("acceptanceDateTime"),
            "primary_document": col("primaryDocument"),
            "primary_doc_description": col("primaryDocDescription"),
            "items": col("items", ""),
            "size": col("size", 0),
            "is_xbrl": col("isXBRL", 0),
            "is_inline_xbrl": col("isInlineXBRL", 0),
            "act": col("act", ""),
            "file_number": col("fileNumber", ""),
        }
    )


def fetch_company_filings(fetcher: Fetcher, cik: int) -> pd.DataFrame:
    """Full filing history for one CIK, including the older paginated shards."""
    payload = fetcher.get_json(SUBMISSIONS_URL.format(cik=cik))
    if not payload:
        return pd.DataFrame(columns=_FILING_COLS)
    filings = payload.get("filings", {}) or {}
    frames = [_rows_from_block(filings.get("recent", {}), cik)]
    for shard in filings.get("files", []) or []:
        name = shard.get("name")
        if not name:
            continue
        sub = fetcher.get_json(SUBMISSIONS_SHARD_URL.format(name=name))
        if sub:
            frames.append(_rows_from_block(sub, cik))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FILING_COLS)
    out.attrs["sic"] = payload.get("sic")
    out.attrs["sic_description"] = payload.get("sicDescription")
    out.attrs["name"] = payload.get("name")
    out.attrs["exchanges"] = payload.get("exchanges")
    return out


def fetch_filings_bulk(
    fetcher: Fetcher,
    ciks: list[int],
    *,
    workers: int = 6,
    start_date: str | None = None,
    forms: set[str] | None = None,
    desc: str = "submissions",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filing metadata for many companies. Returns (filings, company_profile)."""
    frames, profiles = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_company_filings, fetcher, c): c for c in ciks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc, unit="co"):
            cik = futs[fut]
            try:
                df = fut.result()
            except Exception as exc:  # keep going; a single dead CIK is not fatal
                log.warning("CIK %s failed: %s", cik, exc)
                continue
            if df.empty:
                continue
            profiles.append(
                {
                    "cik": cik,
                    "sic": df.attrs.get("sic"),
                    "sic_description": df.attrs.get("sic_description"),
                    "company": df.attrs.get("name"),
                    "exchanges": ",".join(df.attrs.get("exchanges") or []),
                }
            )
            if forms is not None:
                df = df[df["form"].isin(forms)]
            if start_date is not None:
                df = df[df["filing_date"] >= start_date]
            if not df.empty:
                frames.append(df)
    filings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FILING_COLS)
    return filings, pd.DataFrame(profiles)


def normalise_filings(filings: pd.DataFrame, acceptance_tz: str) -> pd.DataFrame:
    """Type-cast, localise acceptance timestamps, and split 8-K item codes."""
    df = filings.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    # EDGAR writes Eastern wall-clock with a spurious "Z"; strip it and localise.
    raw = df["acceptance_raw"].astype("string").str.replace("Z", "", regex=False)
    naive = pd.to_datetime(raw, errors="coerce")
    localised = naive.dt.tz_localize(acceptance_tz, ambiguous=True, nonexistent="shift_forward")
    df["accepted_at"] = localised.dt.tz_convert("UTC").dt.as_unit("ns")
    # Filings before 2002 have no acceptance stamp; fall back to 17:30 ET on the
    # filing date, which is the conservative (latest plausible) assumption.
    fallback = (
        df["filing_date"]
        .add(pd.Timedelta(hours=17, minutes=30))
        .dt.tz_localize(acceptance_tz, ambiguous=True, nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.as_unit("ns")
    )
    df["accepted_at_imputed"] = df["accepted_at"].isna()
    df["accepted_at"] = df["accepted_at"].fillna(fallback)

    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype("int64")
    df["items"] = df["items"].fillna("").astype("string")
    df["item_list"] = df["items"].apply(
        lambda s: [i.strip() for i in str(s).split(",") if i.strip()] if s else []
    )
    df["n_items"] = df["item_list"].str.len()
    df["form"] = df["form"].astype("string")
    df["form_base"] = df["form"].str.replace(r"/A$", "", regex=True)
    df["is_amendment"] = df["form"].str.endswith("/A").fillna(False)
    df["accession_nodash"] = df["accession"].astype("string").str.replace("-", "", regex=False)
    df["source_url"] = [
        ARCHIVE_DOC_URL.format(cik=c, acc_nodash=a, doc=d) if isinstance(d, str) and d else
        ARCHIVE_DIR_URL.format(cik=c, acc_nodash=a)
        for c, a, d in zip(df["cik"], df["accession_nodash"], df["primary_document"])
    ]
    return df


# --------------------------------------------------------------------------- #
# 8-K item taxonomy
# --------------------------------------------------------------------------- #
# Current-report item codes grouped into economically meaningful families. The
# grouping is deliberate: individual codes are sparse, families are not.
ITEM_FAMILIES: dict[str, str] = {
    "1.01": "material_agreement", "1.02": "material_agreement", "1.03": "bankruptcy",
    "1.04": "mine_safety", "1.05": "cybersecurity",
    "2.01": "ma_assets", "2.02": "results", "2.03": "debt_obligation",
    "2.04": "debt_acceleration", "2.05": "restructuring_costs", "2.06": "impairment",
    "3.01": "listing_compliance", "3.02": "unregistered_sale", "3.03": "security_holder_rights",
    "4.01": "auditor_change", "4.02": "non_reliance",
    "5.01": "control_change", "5.02": "executive_change", "5.03": "bylaw_change",
    "5.04": "plan_blackout", "5.05": "ethics_waiver", "5.06": "shell_status_change",
    "5.07": "shareholder_vote", "5.08": "director_nominations",
    "6.01": "abs_informational", "6.02": "abs_servicer_change", "6.03": "abs_credit_enhancement",
    "6.04": "abs_failure", "6.05": "abs_securities_act",
    "7.01": "reg_fd", "8.01": "other_events", "9.01": "exhibits",
}

# Families that historically move prices when they are not already expected.
HIGH_SALIENCE_FAMILIES = {
    "results", "ma_assets", "bankruptcy", "impairment", "restructuring_costs",
    "non_reliance", "auditor_change", "control_change", "executive_change",
    "listing_compliance", "debt_acceleration", "cybersecurity", "material_agreement",
    "unregistered_sale", "debt_obligation",
}


def item_family(code: str) -> str:
    return ITEM_FAMILIES.get(code.strip(), "other_events")


# --------------------------------------------------------------------------- #
# Insider transactions (Form 3/4/5 structured data sets)
# --------------------------------------------------------------------------- #
def fetch_form345_quarter(fetcher: Fetcher, year: int, quarter: int) -> dict[str, pd.DataFrame]:
    """Download and parse one quarterly Form 345 dataset zip."""
    url = FORM345_URL.format(yr=year, q=quarter)
    res = fetcher.get(url)
    if res.status != 200 or not res.content:
        log.warning("no Form345 dataset for %dQ%d (%s)", year, quarter, res.status)
        return {}
    out: dict[str, pd.DataFrame] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
            for member in ("SUBMISSION.tsv", "NONDERIV_TRANS.tsv", "REPORTINGOWNER.tsv"):
                if member not in zf.namelist():
                    continue
                with zf.open(member) as fh:
                    out[member.replace(".tsv", "").lower()] = pd.read_csv(
                        fh, sep="\t", dtype=str, on_bad_lines="skip", low_memory=False
                    )
    except zipfile.BadZipFile:
        log.warning("corrupt Form345 zip for %dQ%d", year, quarter)
    return out


def parse_insider_transactions(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join submission metadata onto non-derivative transactions."""
    sub, trans = parts.get("submission"), parts.get("nonderiv_trans")
    if sub is None or trans is None or sub.empty or trans.empty:
        return pd.DataFrame()
    sub = sub.rename(columns=str.upper)
    trans = trans.rename(columns=str.upper)
    keep_sub = [c for c in ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "ISSUERCIK",
                            "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"] if c in sub.columns]
    m = trans.merge(sub[keep_sub], on="ACCESSION_NUMBER", how="left")
    out = pd.DataFrame(
        {
            "accession": m.get("ACCESSION_NUMBER"),
            "issuer_cik": pd.to_numeric(m.get("ISSUERCIK"), errors="coerce"),
            "ticker": m.get("ISSUERTRADINGSYMBOL", pd.Series(dtype=str)).astype("string").str.upper(),
            "filing_date": pd.to_datetime(m.get("FILING_DATE"), errors="coerce", format="mixed"),
            "transaction_date": pd.to_datetime(m.get("TRANS_DATE"), errors="coerce", format="mixed"),
            "transaction_code": m.get("TRANS_CODE"),
            "acquired_disposed": m.get("TRANS_ACQUIRED_DISP_CD"),
            "shares": pd.to_numeric(m.get("TRANS_SHARES"), errors="coerce"),
            "price": pd.to_numeric(m.get("TRANS_PRICEPERSHARE"), errors="coerce"),
            "shares_after": pd.to_numeric(m.get("SHRS_OWND_FOLWNG_TRANS"), errors="coerce"),
            "direct_indirect": m.get("DIRECT_INDIRECT_OWNERSHIP"),
            "document_type": m.get("DOCUMENT_TYPE"),
        }
    )
    out["value"] = out["shares"].fillna(0) * out["price"].fillna(0)
    # P = open-market purchase, S = open-market sale. A/M/F/G are grants,
    # option exercises, tax withholding and gifts -- far less informative.
    out["signed_value"] = out["value"] * out["acquired_disposed"].map({"A": 1.0, "D": -1.0}).fillna(0)
    out["is_open_market"] = out["transaction_code"].isin(["P", "S"])
    return out


# --------------------------------------------------------------------------- #
# Filing documents (for text-novelty features)
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def extract_text(html_or_text: bytes, max_chars: int = 400_000) -> str:
    """Cheap, dependency-light HTML-to-text. Good enough for novelty metrics."""
    s = html_or_text.decode("utf-8", errors="replace")[: max_chars * 4]
    s = re.sub(r"(?is)<(script|style|table)[^>]*>.*?</\1>", " ", s)
    s = _TAG_RE.sub(" ", s)
    s = (
        s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#160;", " ")
        .replace("&#8217;", "'").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
    )
    return _WS_RE.sub(" ", s).strip()[:max_chars]


def fetch_filing_text(fetcher: Fetcher, cik: int, accession_nodash: str, document: str) -> str:
    if not document:
        return ""
    url = ARCHIVE_DOC_URL.format(cik=cik, acc_nodash=accession_nodash, doc=document)
    res = fetcher.get(url)
    if res.status != 200 or not res.content:
        return ""
    return extract_text(res.content)
