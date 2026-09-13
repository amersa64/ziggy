"""Configuration loading.

The config object is intentionally a thin, typed view over ``configs/default.yaml``
so that every modelling knob lives in one auditable file rather than scattered
through the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    """Dict-backed config with attribute access to top-level sections."""

    raw: dict[str, Any]
    path: Path = DEFAULT_CONFIG_PATH

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- convenience accessors -------------------------------------------------
    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]

    @property
    def http(self) -> dict[str, Any]:
        return self.raw["http"]

    @property
    def sec(self) -> dict[str, Any]:
        return self.raw["sec"]

    @property
    def market(self) -> dict[str, Any]:
        return self.raw["market"]

    @property
    def macro(self) -> dict[str, Any]:
        return self.raw["macro"]

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw["universe"]

    @property
    def labels(self) -> dict[str, Any]:
        return self.raw["labels"]

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw["splits"]

    @property
    def ranking(self) -> dict[str, Any]:
        return self.raw["ranking"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def seed(self) -> int:
        return int(self.experiment.get("seed", 0))

    # -- paths -----------------------------------------------------------------
    def _path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def raw_dir(self) -> Path:
        return self._path("raw")

    @property
    def interim_dir(self) -> Path:
        return self._path("interim")

    @property
    def processed_dir(self) -> Path:
        return self._path("processed")

    @property
    def artifacts_dir(self) -> Path:
        return self._path("artifacts")

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw, path=path)
    # Environment overrides for things that differ per operator.
    ua = os.environ.get("ZIGGY_USER_AGENT")
    if ua:
        cfg.raw["http"]["user_agent"] = ua
    return cfg
