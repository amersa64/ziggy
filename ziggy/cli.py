"""Command-line entry points.

    python -m ziggy.cli ingest          # acquire raw data (network required)
    python -m ziggy.cli build           # features + labels -> candidate matrix
    python -m ziggy.cli experiment      # fit, rank, evaluate, audit
    python -m ziggy.cli report          # render the evidence package
    python -m ziggy.cli all             # ingest -> build -> experiment -> report
    python -m ziggy.cli simulate        # generate a synthetic PIT corpus
    python -m ziggy.cli check-sources   # probe data-source reachability
"""
from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ziggy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["ingest", "build", "experiment", "report", "all",
                                       "simulate", "check-sources"])
    p.add_argument("--config", default=None, help="path to a YAML config")
    p.add_argument("--force", action="store_true", help="re-fetch / rebuild instead of reusing cache")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    from ziggy.config import load_config

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    log = logging.getLogger("ziggy")

    if args.command == "check-sources":
        from ziggy.sources import check_sources

        for row in check_sources(cfg):
            print(f"{row['status']:>8}  {row['name']:<28} {row['url']}")
        return 0

    if args.command == "simulate":
        from ziggy.simulate import generate

        stats = generate(cfg)
        log.info("simulated corpus: %s", stats)
        return 0

    from ziggy.pipeline import build_dataset, ingest_all

    if args.command in ("ingest", "all"):
        log.info("ingest: %s", ingest_all(cfg, force=args.force))
    if args.command in ("build", "all"):
        build_dataset(cfg, force=args.force)
    if args.command in ("experiment", "all"):
        from ziggy.experiment.run import run_experiment

        m = run_experiment(cfg)
        log.info("selected model: %s | holdout lift@%s: %.3f", m["selected_model"],
                 m["primary_k"], m["holdout_headline"].get("lift", float("nan")))
    if args.command in ("report", "all"):
        from ziggy.report.build_report import build_report

        path = build_report(cfg)
        log.info("report written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
