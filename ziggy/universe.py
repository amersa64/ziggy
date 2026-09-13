"""Point-in-time tradability universe.

The cardinal sin available here is to take today's index membership and pretend
it was known in 2021. Instead the universe is *recomputed at every snapshot*
from trailing market data only:

    eligible on session D  <=>  price, liquidity and history thresholds are all
                                satisfied using bars up to and including D.

A company that IPO'd in 2023 is absent before 2023. A company that was delisted
in 2022 is present until its last bar and then disappears -- provided the market
data vendor kept its history at all. That proviso is the survivorship
limitation, and :func:`survivorship_report` measures it instead of assuming it
away.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_price_panel(prices: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Sort, restrict to calendar sessions and derive per-bar quantities."""
    df = prices[prices["date"].isin(sessions)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["dollar_volume"] = df["close"] * df["volume"]
    g = df.groupby("ticker", sort=False, observed=True)
    df["bar_index"] = g.cumcount()
    df["adv20"] = g["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=10).median())
    return df


def build_universe(
    prices_panel: pd.DataFrame, cfg, sessions: pd.DatetimeIndex,
    study_sessions: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """One row per (session, ticker) that is tradable *as of that session*.

    ``prices_panel`` should span the warmup period as well, so that the history
    and liquidity requirements are evaluated against real trailing bars rather
    than eating the first year of the study window. ``study_sessions`` then
    restricts the returned membership to the period actually under test.
    """
    u = cfg.universe
    df = prices_panel

    eligible = (
        (df["close"] >= float(u["min_price"]))
        & (df["adv20"] >= float(u["min_median_dollar_volume_20d"]))
        & (df["bar_index"] >= int(u["min_history_sessions"]))
    )
    sel = df.loc[eligible, ["date", "ticker", "close", "adv20", "bar_index"]].copy()

    # Cap breadth by liquidity so the daily ranking problem stays a fixed size.
    max_names = int(u.get("max_names") or 0)
    if max_names:
        sel["liq_rank"] = sel.groupby("date", observed=True)["adv20"].rank(ascending=False, method="first")
        sel = sel[sel["liq_rank"] <= max_names]
    else:
        sel["liq_rank"] = sel.groupby("date", observed=True)["adv20"].rank(ascending=False, method="first")

    sel = sel.rename(columns={"date": "session"})
    if study_sessions is not None:
        sel = sel[sel["session"].isin(study_sessions)]
    log.info(
        "universe: %d (session,ticker) rows, %d sessions, median %.0f names/session",
        len(sel), sel["session"].nunique(),
        sel.groupby("session").size().median() if len(sel) else 0,
    )
    return sel.reset_index(drop=True)


def survivorship_report(prices_panel: pd.DataFrame, universe: pd.DataFrame,
                        sessions: pd.DatetimeIndex) -> dict:
    """Quantify how much delisting the price source actually carries.

    A vendor that drops delisted names shows up here as a suspiciously low
    disappearance rate: real US equity markets retire several percent of listed
    names per year through mergers, bankruptcies and exchange delistings.
    """
    if universe.empty:
        return {"status": "empty"}
    last_session = sessions.max()
    last_bar = prices_panel.groupby("ticker", observed=True)["date"].max()
    first_bar = prices_panel.groupby("ticker", observed=True)["date"].min()
    ever = universe["ticker"].unique()
    last_bar = last_bar.reindex(ever)
    first_bar = first_bar.reindex(ever)

    # "Disappeared" = last bar more than 10 sessions before the end of the study.
    cutoff = sessions[max(0, len(sessions) - 11)]
    disappeared = last_bar[last_bar < cutoff]
    years = max((last_session - sessions.min()).days / 365.25, 1e-9)

    per_session = universe.groupby("session")["ticker"].nunique()
    churn = []
    prev: set[str] | None = None
    for s, grp in universe.groupby("session", observed=True):
        cur = set(grp["ticker"])
        if prev is not None:
            churn.append(len(cur - prev) + len(prev - cur))
        prev = cur

    return {
        "tickers_ever_in_universe": int(len(ever)),
        "tickers_disappeared_before_end": int(len(disappeared)),
        "annual_disappearance_rate": float(len(disappeared) / len(ever) / years),
        "expected_annual_delisting_rate_us_equities": 0.04,
        "tickers_first_seen_after_start": int((first_bar > sessions.min() + pd.Timedelta(days=30)).sum()),
        "names_per_session_median": float(per_session.median()),
        "names_per_session_min": int(per_session.min()),
        "names_per_session_max": int(per_session.max()),
        "mean_daily_membership_churn": float(np.mean(churn)) if churn else 0.0,
    }
