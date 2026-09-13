#!/usr/bin/env python3
"""Derive configs/simulation.yaml from configs/default.yaml.

Keeps the two in lockstep: the simulation must use the *same* modelling choices
as the real run, differing only in where data lives and that it never touches
the network.
"""
from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

HEADER = (
    "# Simulation config: identical modelling choices, separate data root so the\n"
    "# synthetic corpus can never be confused with real data.\n"
    "# Generated from configs/default.yaml by scripts/make_sim_config.py\n"
)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["experiment"]["name"] = "exp1_simulation_control"
    cfg["paths"] = {
        "raw": "data/sim/raw", "interim": "data/sim/interim",
        "processed": "data/sim/processed", "artifacts": "artifacts/simulation",
    }
    cfg["sec"]["fetch_documents"] = False
    cfg["market"]["provider"] = "csv"
    cfg["evaluation"]["bootstrap_samples"] = 1000
    cfg["evaluation"]["permutation_samples"] = 60
    (ROOT / "configs" / "simulation.yaml").write_text(HEADER + yaml.safe_dump(cfg, sort_keys=False))
    print("wrote configs/simulation.yaml")


if __name__ == "__main__":
    main()
