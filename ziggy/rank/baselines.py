"""Naive baselines.

The experiment's claim is comparative, so the baselines have to be the ones a
sceptic would actually propose:

``random``           pick uniformly from the tradable universe
``liquidity``        rank by dollar volume (the "just look at the big names" prior)
``inverse_liquidity`` rank by *illiquidity* (small names move more -- a strong,
                     cheap baseline that a naive model can mistake for skill)
``prior_abs_move``   yesterday's biggest absolute movers (pure attention chasing)
``volume_spike``     today's biggest volume anomalies
``trailing_vol``     the highest-volatility names (they will move again)
``any_disclosure``   random among names that filed anything tonight
``n_filings``        rank by how much was filed tonight
``earnings_only``    names whose 8-K carried a results item
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rand(df: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(df)), index=df.index)


def score_baseline(name: str, df: pd.DataFrame, seed: int = 0) -> pd.Series:
    if name == "random":
        return _rand(df, seed)
    if name == "liquidity":
        return df["log_adv20_universe"].astype(float)
    if name == "inverse_liquidity":
        return -df["log_adv20_universe"].astype(float)
    if name == "prior_abs_move":
        return df["abs_ret_1d"].astype(float)
    if name == "volume_spike":
        return df["volume_z_20d"].astype(float)
    if name == "trailing_vol":
        return df["vol_21d"].astype(float)
    if name == "any_disclosure":
        return df["has_disclosure"].astype(float) + 1e-6 * _rand(df, seed)
    if name == "n_filings":
        return df["sec_n_filings"].astype(float) + 1e-6 * _rand(df, seed)
    if name == "earnings_only":
        col = df["sec_item_results"] if "sec_item_results" in df.columns else pd.Series(0.0, index=df.index)
        return col.astype(float) + 1e-6 * _rand(df, seed)
    raise ValueError(f"unknown baseline {name!r}")


BASELINES = [
    "random", "liquidity", "inverse_liquidity", "prior_abs_move", "volume_spike",
    "trailing_vol", "any_disclosure", "n_filings", "earnings_only",
]
