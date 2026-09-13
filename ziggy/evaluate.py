"""Evaluation of a frozen daily ranking.

Every metric is computed *per snapshot* and then averaged across snapshots, so a
handful of huge days cannot carry the result. Uncertainty comes from a
stationary block bootstrap over snapshots (block length ~10 sessions), because
adjacent days are not independent: overlapping label windows and shared market
regimes induce serial correlation that an i.i.d. bootstrap would ignore, giving
intervals that are far too narrow.

Metrics
-------
``precision@k``  share of the k selected names that were consequential
``lift@k``       precision@k divided by the base rate over the same candidate
                 set -- "how many times better than picking at random"
``capture@k``    share of the day's *total* absolute excess movement that landed
                 inside the k selected names
``mean_mag@k``   average |excess move| of the selected names
``ndcg@k``       rank quality using |excess move| as the gain
``hit_any@k``    share of days where at least one selected name was consequential
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-session metrics
# --------------------------------------------------------------------------- #
def _dcg(gains: np.ndarray) -> float:
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def per_session_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    magnitude_col: str,
    ks: list[int],
    session_col: str = "session",
    seed: int = 0,
) -> pd.DataFrame:
    """One row per session per k."""
    rng = np.random.default_rng(seed)
    need = [session_col, score_col, label_col, magnitude_col]
    d = df[need].dropna(subset=[score_col, label_col, magnitude_col])
    if d.empty:
        return pd.DataFrame()
    # Break score ties at random rather than alphabetically by ticker, which
    # would silently favour whatever "AAPL" happens to be that day.
    d = d.assign(_tie=rng.random(len(d)))
    d = d.sort_values([session_col, score_col, "_tie"], ascending=[True, False, True])
    d["_rank"] = d.groupby(session_col, observed=True).cumcount() + 1

    rows = []
    for sess, g in d.groupby(session_col, observed=True, sort=True):
        n = len(g)
        lab = g[label_col].to_numpy()
        mag = g[magnitude_col].to_numpy()
        total_mag = mag.sum()
        base = lab.mean()
        ideal = np.sort(mag)[::-1]
        for k in ks:
            kk = min(k, n)
            sel_lab = lab[:kk]
            sel_mag = mag[:kk]
            idcg = _dcg(ideal[:kk])
            rows.append(
                {
                    "session": sess, "k": k, "n_candidates": n, "base_rate": base,
                    "precision": sel_lab.mean(),
                    "expected_hits": sel_lab.sum(),
                    "capture": (sel_mag.sum() / total_mag) if total_mag > 0 else np.nan,
                    "mean_mag": sel_mag.mean(),
                    "universe_mean_mag": mag.mean(),
                    "ndcg": (_dcg(sel_mag) / idcg) if idcg > 0 else np.nan,
                    "hit_any": float(sel_lab.sum() > 0),
                }
            )
    out = pd.DataFrame(rows)
    out["lift"] = out["precision"] / out["base_rate"].replace(0, np.nan)
    out["mag_ratio"] = out["mean_mag"] / out["universe_mean_mag"].replace(0, np.nan)
    out["capture_lift"] = out["capture"] / (out["k"].clip(upper=out["n_candidates"]) / out["n_candidates"])
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, block: int = 10, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) from a stationary block bootstrap over an ordered series."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) < block * 2:
        block = max(1, len(v) // 4)
    rng = np.random.default_rng(seed)
    n = len(v)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx_offsets = np.arange(block)
    for b in range(n_boot):
        idx = (starts[b][:, None] + idx_offsets[None, :]).ravel()[:n] % n
        means[b] = v[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_block_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, block: int = 10, seed: int = 0
) -> dict:
    """Difference ``a - b`` on the same sessions, with a bootstrap p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_value": np.nan, "n": 0}
    d = a - b
    mean, lo, hi = block_bootstrap_ci(d, n_boot=n_boot, block=block, seed=seed)
    # Two-sided bootstrap p-value: how often a recentred resample reaches 0.
    rng = np.random.default_rng(seed + 1)
    n, blk = len(d), min(block, max(1, len(d) // 4))
    n_blocks = int(np.ceil(n / blk))
    centred = d - d.mean()
    hits = 0
    offs = np.arange(blk)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    for i in range(n_boot):
        idx = (starts[i][:, None] + offs[None, :]).ravel()[:n] % n
        if abs(centred[idx].mean()) >= abs(d.mean()):
            hits += 1
    return {"diff": mean, "lo": lo, "hi": hi, "p_value": (hits + 1) / (n_boot + 1), "n": int(n)}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
METRIC_COLS = ["precision", "lift", "capture", "capture_lift", "mean_mag",
               "mag_ratio", "ndcg", "hit_any", "expected_hits"]


def summarise(
    per_session: pd.DataFrame, n_boot: int = 2000, block: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap mean and 95% CI for each metric at each k."""
    rows = []
    for k, g in per_session.groupby("k", observed=True, sort=True):
        g = g.sort_values("session")
        row = {"k": int(k), "n_sessions": int(len(g)),
               "base_rate": float(g["base_rate"].mean()),
               "mean_candidates": float(g["n_candidates"].mean())}
        for m in METRIC_COLS:
            if m not in g.columns:
                continue
            mean, lo, hi = block_bootstrap_ci(g[m].to_numpy(), n_boot, block, seed=seed)
            row[m] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(
    per_session_by_model: dict[str, pd.DataFrame],
    reference: str,
    k: int,
    metric: str = "lift",
    n_boot: int = 2000,
    block: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired comparison of every model against ``reference`` at one k."""
    ref = per_session_by_model[reference]
    ref = ref[ref["k"] == k].set_index("session")[metric]
    rows = []
    for name, ps in per_session_by_model.items():
        if name == reference:
            continue
        cur = ps[ps["k"] == k].set_index("session")[metric]
        common = ref.index.intersection(cur.index)
        res = paired_block_bootstrap(
            cur.reindex(common).to_numpy(), ref.reindex(common).to_numpy(),
            n_boot=n_boot, block=block, seed=seed,
        )
        rows.append({"model": name, "reference": reference, "k": k, "metric": metric, **res})
    return pd.DataFrame(rows).sort_values("diff", ascending=False).reset_index(drop=True)


def permutation_null(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    magnitude_col: str,
    k: int,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    """Within-day label shuffle.

    Under the null the ranking carries no information about which names moved,
    so lift must collapse to ~1.0. If it does not, something is leaking.

    The shortlist itself never changes under the null -- only which labels sit
    where -- so the top-k selection is computed once and each permutation is
    three vectorised passes rather than a re-sort of the whole panel.
    """
    rng = np.random.default_rng(seed)
    d = df[["session", score_col, label_col, magnitude_col]].dropna()
    if d.empty:
        return {"observed_lift": np.nan, "null_mean": np.nan, "null_std": np.nan,
                "null_p95": np.nan, "p_value": np.nan, "n_permutations": 0}

    d = d.assign(_tie=rng.random(len(d)))
    d = d.sort_values(["session", score_col, "_tie"], ascending=[True, False, True])
    codes, _ = pd.factorize(d["session"], sort=True)
    within = np.arange(len(d)) - np.repeat(
        np.concatenate([[0], np.cumsum(np.bincount(codes))[:-1]]), np.bincount(codes)
    )
    sel = within < k
    lab = d[label_col].to_numpy(dtype=float)

    n_per_session = np.bincount(codes).astype(float)
    base = np.bincount(codes, weights=lab) / n_per_session
    sel_counts = np.bincount(codes[sel], minlength=len(n_per_session)).astype(float)

    def mean_lift(labels: np.ndarray) -> float:
        hits = np.bincount(codes[sel], weights=labels[sel], minlength=len(n_per_session))
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (hits / sel_counts) / base
        return float(np.nanmean(lift))

    obs_lift = mean_lift(lab)
    null = np.empty(n_perm)
    for i in range(n_perm):
        order = np.lexsort((rng.random(len(d)), codes))   # shuffle within session
        null[i] = mean_lift(lab[order])
    return {
        "observed_lift": obs_lift,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float(((null >= obs_lift).sum() + 1) / (n_perm + 1)),
        "n_permutations": int(n_perm),
    }


def _fast_lift_machinery(d: pd.DataFrame, label_col: str, k: int, seed: int):
    """Precompute the per-session bookkeeping shared by every candidate ranking.

    Ranking 150 features in three orientations each means ~450 rankings of the
    same panel. Sorting the panel 450 times with pandas is minutes; one lexsort
    and two bincounts per ranking is seconds.
    """
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(d["session"], sort=True)
    n_sessions = codes.max() + 1 if len(codes) else 0
    counts = np.bincount(codes, minlength=n_sessions)
    offsets = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    lab = d[label_col].to_numpy(dtype=float)
    base = np.bincount(codes, weights=lab, minlength=n_sessions) / np.maximum(counts, 1)
    jitter = rng.random(len(d))

    def lift_of(values: np.ndarray) -> float:
        order = np.lexsort((jitter, -np.asarray(values, dtype=float), codes))
        pos = np.arange(len(d)) - offsets
        rows = order[pos < k]
        rc = codes[rows]
        hits = np.bincount(rc, weights=lab[rows], minlength=n_sessions)
        cnt = np.bincount(rc, minlength=n_sessions).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nanmean((hits / cnt) / base))

    return lift_of, codes


def univariate_lift(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    magnitude_col: str,
    k: int,
    seed: int = 0,
    min_coverage: float = 0.02,
) -> pd.DataFrame:
    """Lift of ranking on each single feature, in its most favourable direction.

    Diagnostic, not a model: it says which individual signals carry information
    about consequential activity, which is exactly what the downstream reasoning
    agent needs to know when it decides what evidence to ask for. Three
    orientations are tried per feature (high, low, and distance-from-median), so
    the reported lift is optimistically biased -- it is read as a ranking of
    features, never as a significance test.
    """
    base = df[["session", label_col, magnitude_col]].dropna()
    if base.empty:
        return pd.DataFrame()
    base = base.sort_values("session", kind="stable")
    idx = base.index
    lift_of, codes = _fast_lift_machinery(base, label_col, k, seed)

    rows = []
    for c in feature_cols:
        col = df.loc[idx, c]
        coverage = float(col.notna().mean())
        if coverage < min_coverage or col.nunique(dropna=True) < 3:
            continue
        filled = col.fillna(col.median())
        v = filled.to_numpy(dtype=float)
        hi, lo = lift_of(v), lift_of(-v)
        # The label is an *absolute* move, so a feature can matter through its
        # extremeness rather than its sign -- both tails of a valuation or flow
        # measure can precede a large move. Ranking on within-day distance from
        # the median catches that, and sign-directional features lose nothing.
        # "median" is a cython groupby path; a Python lambda here costs more
        # than every lexsort in this function combined.
        med = pd.Series(v).groupby(codes).transform("median").to_numpy()
        ex = lift_of(np.abs(v - med))
        scores = {"high": hi, "low": lo, "extreme": ex}
        direction = max(scores, key=scores.get)
        rows.append({"feature": c, "best_lift": scores[direction], "direction": direction,
                     "lift_high": hi, "lift_low": lo, "lift_extreme": ex, "coverage": coverage})
    out = pd.DataFrame(rows)
    return out.sort_values("best_lift", ascending=False).reset_index(drop=True) if len(out) else out


def by_regime(
    per_session: pd.DataFrame, regime: pd.DataFrame, k: int, column: str = "vix_level"
) -> pd.DataFrame:
    """Break the headline metric down by market regime terciles."""
    if column not in regime.columns:
        return pd.DataFrame()
    r = regime[["session", column]].dropna()
    g = per_session[per_session["k"] == k].merge(r, on="session", how="inner")
    if g.empty:
        return pd.DataFrame()
    g["regime"] = pd.qcut(g[column], 3, labels=["calm", "normal", "stressed"])
    out = (
        g.groupby("regime", observed=True)
        .agg(n_sessions=("session", "nunique"), base_rate=("base_rate", "mean"),
             precision=("precision", "mean"), lift=("lift", "mean"),
             capture=("capture", "mean"), mag_ratio=("mag_ratio", "mean"))
        .reset_index()
    )
    return out


def by_year(per_session: pd.DataFrame, k: int) -> pd.DataFrame:
    g = per_session[per_session["k"] == k].copy()
    g["year"] = pd.to_datetime(g["session"]).dt.year
    return (
        g.groupby("year", observed=True)
        .agg(n_sessions=("session", "nunique"), base_rate=("base_rate", "mean"),
             precision=("precision", "mean"), lift=("lift", "mean"),
             capture=("capture", "mean"), mag_ratio=("mag_ratio", "mean"),
             ndcg=("ndcg", "mean"))
        .reset_index()
    )
