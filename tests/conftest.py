from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ziggy.calendar import build_calendar
from ziggy.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def cal(cfg):
    return build_calendar(cfg)


def make_panel(tickers, sessions, seed=0, stop: dict | None = None):
    """Deterministic synthetic OHLCV panel."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        p = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(sessions))))
        o = p * (1 + rng.normal(0, 0.003, len(sessions)))
        d = pd.DataFrame(
            {
                "date": sessions, "ticker": t,
                "open": o, "high": np.maximum(p, o) * 1.005, "low": np.minimum(p, o) * 0.995,
                "close": p, "adj_close": p,
                "volume": rng.integers(2_000_000, 9_000_000, len(sessions)).astype(float),
            }
        )
        if stop and t in stop:
            d = d.iloc[: stop[t]]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def sessions():
    return pd.DatetimeIndex(pd.bdate_range("2020-01-02", "2022-06-30"))


@pytest.fixture
def panel(sessions):
    return make_panel([f"T{i:02d}" for i in range(12)], sessions, seed=11)


@pytest.fixture
def bench(sessions):
    return make_panel(["SPY", "QQQ", "IWM"], sessions, seed=99)
