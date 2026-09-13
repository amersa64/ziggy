"""Parquet-backed dataset store.

Tables carry an explicit ``available_at`` column wherever they describe facts
that became knowable at a point in time. ``assert_pit`` is the guard that makes
the point-in-time claim checkable rather than aspirational.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class Store:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.ensure_dirs()

    # -- locations -------------------------------------------------------------
    def path(self, layer: str, name: str) -> Path:
        base = {"raw": self.cfg.raw_dir, "interim": self.cfg.interim_dir,
                "processed": self.cfg.processed_dir, "artifacts": self.cfg.artifacts_dir}[layer]
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{name}.parquet"

    def exists(self, layer: str, name: str) -> bool:
        return self.path(layer, name).exists()

    # -- io --------------------------------------------------------------------
    def write(self, df: pd.DataFrame, layer: str, name: str, **kw) -> Path:
        p = self.path(layer, name)
        tmp = p.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False, compression="zstd", **kw)
        tmp.replace(p)
        log.info("wrote %s (%d rows, %.1f MB)", p.name, len(df), p.stat().st_size / 1e6)
        return p

    def read(self, layer: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.path(layer, name), columns=columns)

    def write_json(self, obj, layer: str, name: str) -> Path:
        base = {"raw": self.cfg.raw_dir, "interim": self.cfg.interim_dir,
                "processed": self.cfg.processed_dir, "artifacts": self.cfg.artifacts_dir}[layer]
        base.mkdir(parents=True, exist_ok=True)
        p = base / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def read_json(self, layer: str, name: str):
        base = {"raw": self.cfg.raw_dir, "interim": self.cfg.interim_dir,
                "processed": self.cfg.processed_dir, "artifacts": self.cfg.artifacts_dir}[layer]
        return json.loads((base / f"{name}.json").read_text())


def assert_pit(
    df: pd.DataFrame,
    snapshot_col: str = "snapshot_ts",
    available_col: str = "available_at",
    label: str = "table",
) -> None:
    """Fail loudly if any row was used before it was knowable."""
    if available_col not in df.columns or snapshot_col not in df.columns:
        raise KeyError(f"{label}: needs both {snapshot_col} and {available_col}")
    a = pd.to_datetime(df[available_col], utc=True)
    s = pd.to_datetime(df[snapshot_col], utc=True)
    bad = (a > s) & a.notna() & s.notna()
    if bad.any():
        n = int(bad.sum())
        worst = (a[bad] - s[bad]).max()
        raise AssertionError(
            f"{label}: {n} rows use information from the future "
            f"(worst offender {worst} after its snapshot)"
        )
