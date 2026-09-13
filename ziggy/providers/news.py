"""News provider interface -- deliberately unimplemented.

Historical news is genuinely useful for this problem and equally genuinely
dangerous. The failure mode is subtle: most free "historical" news sources
either (a) carry the crawl/index time rather than the publication time, or
(b) have coverage that is itself a function of what later turned out to matter,
which is survivorship bias wearing a disguise. Either one silently inflates
every number this experiment produces.

So the interface exists and nothing implements it against a source we cannot
vouch for. A future operator with a licensed point-in-time archive (Ravenpack,
Dow Jones DNA, Refinitiv, a self-maintained crawl with captured HTTP dates)
implements :class:`NewsProvider` and sets ``news.provider`` in the config.
The requirements in :meth:`NewsProvider.integrity_requirements` are the bar.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class NewsProvider(ABC):
    name = "abstract"

    @abstractmethod
    def get_articles(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Columns: ``ticker, published_at (tz-aware UTC), headline, body, source, url``.

        ``published_at`` MUST be the publisher's timestamp, not a crawl time.
        """

    @staticmethod
    def integrity_requirements() -> list[str]:
        return [
            "publication timestamp is the publisher's, captured at publication",
            "coverage is complete for the universe over the whole study window, "
            "not conditioned on later outcomes",
            "no back-filled or retroactively edited article bodies",
            "ticker tagging is done from the article text, not from a later database",
            "delisted and renamed issuers are present",
        ]


class NullNewsProvider(NewsProvider):
    """The honest default: no news, and the report says so."""

    name = "null"

    def get_articles(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["ticker", "published_at", "headline", "body", "source", "url"]
        )


def build_news_provider(cfg) -> NewsProvider:
    which = cfg.raw.get("news", {}).get("provider", "null")
    if which in (None, "null", "none"):
        return NullNewsProvider()
    raise NotImplementedError(
        f"news provider {which!r} is configured but not implemented; see "
        "NewsProvider.integrity_requirements() before wiring one in"
    )
