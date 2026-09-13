"""Chronological splits with an embargo.

Random shuffling would be fatal here for two reasons: adjacent days share
overlapping label windows, and the whole claim under test is about *forecasting*,
which only means anything out of sample in time. Between consecutive splits we
discard ``embargo_sessions`` so that no training label window can overlap the
first validation snapshot (and likewise validation/holdout).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    holdout: pd.DatetimeIndex
    embargoed: pd.DatetimeIndex

    def to_dict(self) -> dict:
        def rng(ix):
            return {"n_sessions": int(len(ix)),
                    "start": str(ix.min().date()) if len(ix) else None,
                    "end": str(ix.max().date()) if len(ix) else None}
        return {"train": rng(self.train), "validation": rng(self.validation),
                "holdout": rng(self.holdout), "embargoed_sessions": int(len(self.embargoed))}

    def label_of(self, sessions: pd.Series) -> pd.Series:
        s = pd.Series("embargo", index=sessions.index, dtype="object")
        s[sessions.isin(self.train)] = "train"
        s[sessions.isin(self.validation)] = "validation"
        s[sessions.isin(self.holdout)] = "holdout"
        return s


def make_splits(sessions: pd.DatetimeIndex, cfg) -> Splits:
    sp = cfg.splits
    sessions = pd.DatetimeIndex(sorted(pd.unique(sessions)))
    n = len(sessions)
    emb = int(sp.get("embargo_sessions", 0))
    n_tr = int(n * float(sp["train_frac"]))
    n_va = int(n * float(sp["validation_frac"]))

    train = sessions[: max(n_tr - emb, 0)]
    gap1 = sessions[max(n_tr - emb, 0): n_tr]
    validation = sessions[n_tr: n_tr + max(n_va - emb, 0)]
    gap2 = sessions[n_tr + max(n_va - emb, 0): n_tr + n_va]
    holdout = sessions[n_tr + n_va:]
    embargoed = gap1.append(gap2)

    log.info(
        "splits: train %s..%s (%d) | val %s..%s (%d) | holdout %s..%s (%d) | embargo %d",
        train.min().date(), train.max().date(), len(train),
        validation.min().date(), validation.max().date(), len(validation),
        holdout.min().date(), holdout.max().date(), len(holdout), len(embargoed),
    )
    return Splits(train, validation, holdout, embargoed)
