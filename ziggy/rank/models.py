"""Ranking models.

Three rankers, deliberately spanning the transparency/flexibility axis:

``DeterministicScorer``
    A fixed, hand-specified linear combination of within-day percentile ranks.
    No fitting at all, so it cannot overfit and it is fully legible. Its weights
    encode priors about what makes a situation worth researching -- salient
    disclosure, unusual disclosure, unusual attention, structural instability.
    They were written down before any evaluation was run and are not tuned.

``LogisticRanker`` / ``GBMRanker``
    Fitted on the training split only. Features are converted to within-day
    percentile ranks first, which makes the models regime-robust and removes the
    need for any global scaler that could carry information across time.

Both fitted rankers can run in ``walk_forward`` mode, where the model used to
rank year Y is fitted only on data strictly before year Y. That is the honest
deployment simulation; "frozen-train" is reported alongside it for comparison.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ziggy.features.assemble import cross_sectional_rank

log = logging.getLogger(__name__)


# Prior weights, specified in advance. Positive = "more worth researching".
DETERMINISTIC_WEIGHTS: dict[str, float] = {
    # -- salient disclosure tonight -------------------------------------------
    "sec_n_high_salience": 1.10,
    "sec_item_results": 0.90,
    "sec_item_ma_assets": 0.85,
    "sec_item_non_reliance": 0.70,
    "sec_item_bankruptcy": 0.70,
    "sec_item_auditor_change": 0.50,
    "sec_item_control_change": 0.50,
    "sec_item_impairment": 0.45,
    "sec_item_restructuring_costs": 0.40,
    "sec_item_listing_compliance": 0.40,
    "sec_item_executive_change": 0.35,
    "sec_item_debt_acceleration": 0.35,
    "sec_item_material_agreement": 0.25,
    "sec_item_unregistered_sale": 0.25,
    "sec_n_8k": 0.35,
    "sec_n_periodic": 0.30,
    "sec_n_13d": 0.30,
    "has_disclosure": 0.40,
    # -- how unusual the disclosure is ----------------------------------------
    "sec_size_log_ratio": 0.30,
    "text_cosine_prior": -0.45,        # low similarity to the last one = novel
    "text_new_token_mass": 0.30,
    "text_neg_delta": 0.25,
    "sec_sessions_since_8k": 0.20,     # a long-silent issuer suddenly speaking
    # -- unusual attention ------------------------------------------------------
    "volume_z_20d": 0.60,
    "dollar_volume_ratio": 0.35,
    "vol_ratio_5_21": 0.40,
    "abs_ret_1d": 0.30,
    "range_pct_1d": 0.20,
    "adv_trend_20_60": 0.15,
    # -- structural propensity to move -----------------------------------------
    "vol_21d": 0.35,
    "idio_vol_63d": 0.30,
    "amihud_21d": 0.20,
    "drawdown_63d": -0.20,
    "log_adv20_universe": -0.45,       # mega caps rarely move idiosyncratically
    # -- insider behaviour ------------------------------------------------------
    "insider_buy_ratio_21d": 0.15,
}


class Ranker(ABC):
    name = "abstract"
    needs_fit = False

    @abstractmethod
    def score(self, df: pd.DataFrame, feature_cols: list[str],
              ranked: pd.DataFrame | None = None) -> pd.Series:
        ...

    def fit(self, df: pd.DataFrame, feature_cols: list[str], label_col: str,
            ranked: pd.DataFrame | None = None) -> "Ranker":
        return self

    def importances(self) -> pd.DataFrame | None:
        return None


class DeterministicScorer(Ranker):
    name = "deterministic"

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or DETERMINISTIC_WEIGHTS)

    def score(self, df: pd.DataFrame, feature_cols: list[str],
              ranked: pd.DataFrame | None = None) -> pd.Series:
        cols = [c for c in self.weights if c in df.columns]
        missing = [c for c in self.weights if c not in df.columns]
        if missing:
            log.info("deterministic scorer: %d weighted features absent (%s...)",
                     len(missing), ", ".join(missing[:4]))
        if ranked is not None and all(c in ranked.columns for c in cols):
            ranks = ranked.loc[df.index, cols] - 0.5
        else:
            ranks = cross_sectional_rank(df, cols) - 0.5
        w = pd.Series({c: self.weights[c] for c in cols})
        return (ranks[cols] * w).sum(axis=1)

    def importances(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": list(self.weights), "weight": list(self.weights.values())})
            .sort_values("weight", key=abs, ascending=False)
            .reset_index(drop=True)
        )


class _SkRanker(Ranker):
    needs_fit = True

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = None
        self.cols: list[str] = []

    def _prep(self, df: pd.DataFrame, cols: list[str],
              ranked: pd.DataFrame | None = None) -> pd.DataFrame:
        """Within-day percentile ranks, reusing a precomputed view when given.

        Ranking 150 columns over a couple of million rows is the single most
        expensive operation in the experiment, and every ranker wants the same
        answer -- so the caller computes it once and passes it down.
        """
        if ranked is not None and all(c in ranked.columns for c in cols):
            return ranked.loc[df.index, cols]
        return cross_sectional_rank(df, cols)

    def fit(self, df: pd.DataFrame, feature_cols: list[str], label_col: str,
            ranked: pd.DataFrame | None = None) -> "_SkRanker":
        sub = df[df[label_col].notna()]
        if sub.empty:
            raise ValueError("no labelled training rows")
        self.cols = list(feature_cols)
        X = self._prep(sub, self.cols, ranked)
        y = sub[label_col].astype(int).values
        self.model = self._make()
        self.model.fit(X.values, y)
        log.info("%s fitted on %d rows, %d features, base rate %.4f",
                 self.name, len(sub), len(self.cols), y.mean())
        return self

    def score(self, df: pd.DataFrame, feature_cols: list[str],
              ranked: pd.DataFrame | None = None) -> pd.Series:
        if self.model is None:
            raise RuntimeError(f"{self.name} used before fit")
        X = self._prep(df, self.cols, ranked)
        return pd.Series(self.model.predict_proba(X.values)[:, 1], index=df.index)

    def _make(self):
        raise NotImplementedError


class LogisticRanker(_SkRanker):
    name = "logistic"

    def _make(self):
        return LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs", random_state=self.seed
        )

    def importances(self) -> pd.DataFrame | None:
        if self.model is None:
            return None
        return (
            pd.DataFrame({"feature": self.cols, "weight": self.model.coef_[0]})
            .sort_values("weight", key=abs, ascending=False)
            .reset_index(drop=True)
        )


class GBMRanker(_SkRanker):
    name = "gbm"

    def _make(self):
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=self.seed,
        )


def build_ranker(name: str, seed: int = 0) -> Ranker:
    return {"deterministic": DeterministicScorer, "logistic": LogisticRanker,
            "gbm": GBMRanker}[name](**({} if name == "deterministic" else {"seed": seed}))


def walk_forward_scores(
    ranker_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    eval_sessions: pd.DatetimeIndex,
    min_train_sessions: int = 250,
    seed: int = 0,
    ranked: pd.DataFrame | None = None,
) -> tuple[pd.Series, list[dict]]:
    """Refit once per calendar year on strictly prior data, then score that year.

    This is the deployment-realistic setting: on 2 January of each year the model
    knows everything up to 31 December and nothing after.
    """
    scores = pd.Series(np.nan, index=df.index, dtype="float64")
    log_rows = []
    eval_df = df[df["session"].isin(eval_sessions)]
    for year, chunk in eval_df.groupby(eval_df["session"].dt.year, observed=True):
        cutoff = pd.Timestamp(year=int(year), month=1, day=1)
        train = df[(df["session"] < cutoff) & df[label_col].notna()]
        if train["session"].nunique() < min_train_sessions:
            log_rows.append({"year": int(year), "status": "insufficient_history",
                             "train_sessions": int(train["session"].nunique())})
            continue
        r = build_ranker(ranker_name, seed=seed)
        r.fit(train, feature_cols, label_col, ranked=ranked)
        scores.loc[chunk.index] = r.score(chunk, feature_cols, ranked=ranked).values
        log_rows.append({
            "year": int(year), "status": "fitted",
            "train_sessions": int(train["session"].nunique()),
            "train_rows": int(len(train)),
            "train_end": str(train["session"].max().date()),
            "scored_rows": int(len(chunk)),
        })
    return scores, log_rows
