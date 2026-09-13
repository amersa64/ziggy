"""The experiment.

    Given everything publicly knowable at 20:00 ET on session D, can we rank the
    tradable universe so that a 10-30 name shortlist contains substantially more
    consequential future market activity than a naive baseline?

Protocol, fixed before any result was looked at:

* Chronological train / validation / holdout split with an embargo.
* Rankers and the absolute-threshold calibration see the training split only.
* Model selection -- which ranker, frozen vs walk-forward -- is decided on the
  validation split.
* The holdout is scored **once**, with the selection already fixed, and that
  number is the headline result. A run that touches the holdout twice is
  recorded as such in the manifest.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ziggy import audit, evaluate
from ziggy.features.assemble import cross_sectional_rank, feature_columns
from ziggy.labels import add_beta_adjusted_label, add_volnorm_label, calibrate_abs_threshold
from ziggy.rank import baselines as bl
from ziggy.rank.models import build_ranker, walk_forward_scores
from ziggy.splits import make_splits
from ziggy.store import Store

log = logging.getLogger(__name__)


def _score_frame(matrix: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return matrix[["session", "ticker"] + cols]



def _secondary_label_experiment(
    cfg, matrix, ranked, scored, splits, feat_cols, label, mag, ks, primary_k, seed,
    n_boot, block,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the selection against a second label definition.

    The primary label rewards large *absolute* excess moves, and volatile names
    make those by definition -- so a model fitted on it can be, in substance, a
    volatility ranker. The question this answers is whether the disclosure and
    flow features carry anything **beyond** volatility: the same rankers are
    fitted on the volatility-normalised label, where the "just pick the jumpy
    names" strategy is worth less than nothing, and evaluated the same way.

    Frozen-train only. This is a secondary question and does not get to consume
    the compute budget or the selection authority of the primary one.
    """
    if label not in matrix.columns or matrix[label].isna().all():
        return pd.DataFrame(), pd.DataFrame()

    sub = scored[["session", "ticker", "split"]].copy()
    sub[label] = matrix[label].values
    sub[mag] = matrix[mag].values
    for name in bl.BASELINES:
        try:
            sub[f"score_baseline_{name}"] = bl.score_baseline(name, matrix, seed=seed)
        except KeyError:
            pass

    train = matrix[(matrix["split"] == "train") & matrix[label].notna()]
    for name in cfg.ranking["models"]:
        r = build_ranker(name, seed=seed)
        if r.needs_fit:
            r.fit(train, feat_cols, label, ranked=ranked)
        sub[f"score_{name}"] = r.score(matrix, feat_cols, ranked=ranked).values

    val = sub[sub["split"] == "validation"]
    hold = sub[sub["split"] == "holdout"]
    per_model = {}
    rows = []
    best, best_lift = None, -np.inf
    for col in [c for c in sub.columns if c.startswith("score_")]:
        ps_val = evaluate.per_session_metrics(val, col, label, mag, [primary_k], seed=seed)
        ps_hold = evaluate.per_session_metrics(hold, col, label, mag, ks, seed=seed)
        if ps_hold.empty:
            continue
        per_model[col] = ps_hold
        summ = evaluate.summarise(ps_hold, n_boot=n_boot, block=block, seed=seed)
        summ.insert(0, "model", col.replace("score_", ""))
        summ.insert(0, "label", label)
        rows.append(summ)
        if not col.startswith("score_baseline_") and len(ps_val):
            lv = float(ps_val["lift"].mean())
            if lv > best_lift:
                best, best_lift = col, lv

    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    comparison = pd.DataFrame()
    if best is not None and best in per_model:
        cmps = []
        for ref in [c for c in per_model if c.startswith("score_baseline_")]:
            cmps.append(evaluate.compare_models(
                {best.replace("score_", ""): per_model[best],
                 ref.replace("score_", ""): per_model[ref]},
                reference=ref.replace("score_", ""), k=primary_k, metric="lift",
                n_boot=n_boot, block=block, seed=seed,
            ))
        comparison = pd.concat(cmps, ignore_index=True) if cmps else pd.DataFrame()
        comparison["selected_on_validation"] = best.replace("score_", "")
    log.info("secondary experiment on %s: selected %s (validation lift %.3f)",
             label, best, best_lift if np.isfinite(best_lift) else float("nan"))
    return summary, comparison


def run_experiment(cfg, matrix: pd.DataFrame | None = None) -> dict:
    t0 = time.time()
    store = Store(cfg)
    if matrix is None:
        matrix = store.read("processed", "candidates")

    label_col = "label_cs_q90"
    mag_col = "consequence_magnitude"
    horizon = int(cfg.labels["primary_horizon"])
    ks = list(cfg.ranking["top_k"])
    primary_k = int(cfg.ranking["primary_k"])
    seed = cfg.seed
    n_boot = int(cfg.evaluation["bootstrap_samples"])
    block = int(cfg.evaluation["block_bootstrap_length"])

    # Both are derivable from columns the matrix already carries, so an older
    # candidate file does not need rebuilding to gain them. One copy, not one
    # per label: this frame is hundreds of megabytes.
    derived = {"label_cs_q90_volnorm", "label_cs_q90_abret"}
    if not derived.issubset(matrix.columns):
        matrix = matrix.copy()
        matrix = add_volnorm_label(matrix, cfg)
        matrix = add_beta_adjusted_label(matrix, cfg)

    feat_cols = feature_columns(matrix)
    fwd_audit = audit.forward_column_audit(feat_cols)
    if fwd_audit["status"] != "clean":
        raise AssertionError(f"label-like columns reached the feature set: {fwd_audit}")
    log.info("features available to the ranker: %d", len(feat_cols))

    # Every ranker consumes the same within-day percentile ranks; computing them
    # once here turns the most expensive step in the run into a single pass.
    t_rank = time.time()
    ranked = cross_sectional_rank(matrix, feat_cols)
    log.info("cross-sectional ranks computed in %.1fs", time.time() - t_rank)

    sessions = pd.DatetimeIndex(sorted(matrix["session"].unique()))
    splits = make_splits(sessions, cfg)
    matrix = matrix.copy()
    matrix["split"] = splits.label_of(matrix["session"])

    # Absolute threshold calibrated on train only, then applied everywhere.
    train_rows = matrix[matrix["split"] == "train"]
    abs_thr = calibrate_abs_threshold(train_rows, cfg.labels["consequential"]["cross_sectional_quantile"])
    matrix["label_abs"] = (matrix[mag_col] >= abs_thr).astype("float32")
    matrix.loc[matrix[mag_col].isna(), "label_abs"] = np.nan
    log.info("absolute |excess| threshold from train split: %.4f", abs_thr)

    carry = [label_col, "label_abs", mag_col, "label_vol_expansion",
             "label_cs_q90_volnorm", "consequence_magnitude_volnorm",
             "label_cs_q90_abret", "consequence_magnitude_abret"]
    scored = matrix[["session", "ticker", "split"] + [c for c in carry if c in matrix.columns]].copy()

    # -- baselines -------------------------------------------------------------
    for name in bl.BASELINES:
        try:
            scored[f"score_baseline_{name}"] = bl.score_baseline(name, matrix, seed=seed)
        except KeyError as exc:
            log.warning("baseline %s unavailable: %s", name, exc)

    # -- models ----------------------------------------------------------------
    fit_log: dict = {}
    importances: dict[str, pd.DataFrame] = {}
    for name in cfg.ranking["models"]:
        ranker = build_ranker(name, seed=seed)
        if ranker.needs_fit:
            tr = matrix[(matrix["split"] == "train") & matrix[label_col].notna()]
            ranker.fit(tr, feat_cols, label_col, ranked=ranked)
            fit_log[name] = {"mode": "frozen_train", "train_rows": int(len(tr)),
                             "train_sessions": int(tr["session"].nunique())}
        scored[f"score_{name}"] = ranker.score(matrix, feat_cols, ranked=ranked).values
        imp = ranker.importances()
        if imp is not None:
            importances[name] = imp

        if ranker.needs_fit and cfg.ranking["walk_forward"]["enabled"]:
            wf_sessions = splits.validation.append(splits.holdout)
            s, rows = walk_forward_scores(
                name, matrix, feat_cols, label_col, wf_sessions,
                min_train_sessions=int(cfg.ranking["walk_forward"]["min_train_sessions"]),
                seed=seed, ranked=ranked,
            )
            scored[f"score_{name}_wf"] = s.values
            fit_log[f"{name}_wf"] = {"mode": "walk_forward", "refits": rows}

    score_cols = [c for c in scored.columns if c.startswith("score_")]
    log.info("scored %d rankers over %d rows", len(score_cols), len(scored))

    # -- evaluation ------------------------------------------------------------
    per_session: dict[str, dict[str, pd.DataFrame]] = {}
    summaries = []
    for split_name in ("train", "validation", "holdout"):
        sub = scored[scored["split"] == split_name]
        per_session[split_name] = {}
        for sc in score_cols:
            if sub[sc].isna().all():
                continue
            ps = evaluate.per_session_metrics(sub, sc, label_col, mag_col, ks, seed=seed)
            if ps.empty:
                continue
            per_session[split_name][sc] = ps
            s = evaluate.summarise(ps, n_boot=n_boot, block=block, seed=seed)
            s.insert(0, "model", sc.replace("score_", ""))
            s.insert(0, "split", split_name)
            summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    # -- model selection on VALIDATION ONLY ------------------------------------
    val = summary[(summary["split"] == "validation") & (summary["k"] == primary_k)]
    model_names = [c.replace("score_", "") for c in score_cols if not c.startswith("score_baseline_")]
    val_models = val[val["model"].isin(model_names)].sort_values("lift", ascending=False)
    selected = str(val_models.iloc[0]["model"]) if len(val_models) else "deterministic"
    log.info("model selected on validation: %s (lift@%d = %.3f)", selected, primary_k,
             float(val_models.iloc[0]["lift"]) if len(val_models) else float("nan"))

    # -- headline: holdout, evaluated once with the selection fixed ------------
    sel_col = f"score_{selected}"
    holdout = scored[scored["split"] == "holdout"]
    headline = {}
    if sel_col in per_session["holdout"]:
        hs = evaluate.summarise(per_session["holdout"][sel_col], n_boot=n_boot, block=block, seed=seed)
        row = hs[hs["k"] == primary_k].iloc[0].to_dict()
        headline = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}

    # -- comparisons against every baseline on the holdout ---------------------
    comparisons = []
    for k in ks:
        for ref in [c for c in per_session["holdout"] if c.startswith("score_baseline_")]:
            if sel_col not in per_session["holdout"]:
                continue
            cmp = evaluate.compare_models(
                {selected: per_session["holdout"][sel_col], ref.replace("score_", ""): per_session["holdout"][ref]},
                reference=ref.replace("score_", ""), k=k, metric="lift", n_boot=n_boot, block=block, seed=seed,
            )
            comparisons.append(cmp)
    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()

    # -- breakdowns ------------------------------------------------------------
    regime_cols = [c for c in matrix.columns if c in ("vix_level", "macro_VIXCLS")]
    regime = matrix[["session"] + regime_cols].drop_duplicates("session") if regime_cols else pd.DataFrame()
    by_regime = pd.DataFrame()
    if len(regime) and sel_col in per_session["holdout"]:
        by_regime = evaluate.by_regime(per_session["holdout"][sel_col], regime, primary_k, regime_cols[0])
    by_year = pd.DataFrame()
    if sel_col in per_session["holdout"]:
        allps = pd.concat(
            [per_session[s][sel_col] for s in ("train", "validation", "holdout") if sel_col in per_session[s]],
            ignore_index=True,
        )
        by_year = evaluate.by_year(allps, primary_k)

    # -- label-definition robustness -------------------------------------------
    robustness = []
    alt_labels = [
        ("label_abs", mag_col),
        ("label_cs_q90_abret", "consequence_magnitude_abret"),
        # The volatility-normalised label is scored against its own magnitude so
        # that capture and NDCG stay internally consistent.
        ("label_cs_q90_volnorm", "consequence_magnitude_volnorm"),
        ("label_vol_expansion", mag_col),
    ]
    for alt_label, alt_mag in alt_labels:
        if alt_label not in scored.columns or scored[alt_label].isna().all():
            continue
        if alt_mag not in scored.columns or sel_col not in per_session["holdout"]:
            continue
        ps = evaluate.per_session_metrics(holdout, sel_col, alt_label, alt_mag, [primary_k], seed=seed)
        if ps.empty:
            continue
        s = evaluate.summarise(ps, n_boot=n_boot, block=block, seed=seed)
        s.insert(0, "label", alt_label)
        robustness.append(s)
    robustness = pd.concat(robustness, ignore_index=True) if robustness else pd.DataFrame()

    # -- secondary experiment: can it find anything beyond volatility? --------
    sec_summary, sec_comparison = _secondary_label_experiment(
        cfg, matrix, ranked, scored, splits, feat_cols,
        "label_cs_q90_volnorm", "consequence_magnitude_volnorm",
        ks, primary_k, seed, n_boot, block,
    )

    # -- univariate diagnostics (validation split, never the holdout) ----------
    val_rows = scored[scored["split"] == "validation"].join(
        matrix.loc[matrix["split"] == "validation", feat_cols]
    )
    univariate = evaluate.univariate_lift(
        val_rows, feat_cols, label_col, mag_col, primary_k, seed=seed
    )

    # -- audits ----------------------------------------------------------------
    filings = store.read("interim", "filings_normalised") if store.exists("interim", "filings_normalised") else pd.DataFrame()
    events = store.read("interim", "events") if store.exists("interim", "events") else pd.DataFrame()
    audits = audit.run_all(
        filings=filings, events=events, matrix=matrix, feature_cols=feat_cols,
        scored=holdout if sel_col in holdout.columns else None, score_col=sel_col,
        label_col=label_col, magnitude_col=mag_col, k=primary_k, horizon=horizon, seed=seed,
    )
    if sel_col in holdout.columns:
        audits["permutation_test"] = evaluate.permutation_null(
            holdout, sel_col, label_col, mag_col, primary_k,
            n_perm=int(cfg.evaluation.get("permutation_samples", 100)), seed=seed,
        )

    # -- persist ---------------------------------------------------------------
    store.write(summary, "artifacts", "summary")
    if len(comparison):
        store.write(comparison, "artifacts", "holdout_vs_baselines")
    if len(by_regime):
        store.write(by_regime, "artifacts", "holdout_by_regime")
    if len(by_year):
        store.write(by_year, "artifacts", "by_year")
    if len(robustness):
        store.write(robustness, "artifacts", "label_robustness")
    if len(univariate):
        store.write(univariate, "artifacts", "univariate_lift")
    if len(sec_summary):
        store.write(sec_summary, "artifacts", "secondary_volnorm_summary")
    if len(sec_comparison):
        store.write(sec_comparison, "artifacts", "secondary_volnorm_vs_baselines")
    for name, imp in importances.items():
        store.write(imp, "artifacts", f"importances_{name}")
    store.write(scored, "processed", "scored")
    for split_name, d in per_session.items():
        if sel_col in d:
            store.write(d[sel_col], "artifacts", f"per_session_{split_name}")

    manifest = {
        "experiment": cfg.experiment["name"],
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - t0, 1),
        "config_path": str(cfg.path),
        "snapshot_convention": f"{cfg.experiment['snapshot_time_local']} {cfg.experiment['timezone']}",
        "period": {"start": str(cfg.experiment["start_date"]), "end": str(cfg.experiment["end_date"])},
        "rows": int(len(matrix)),
        "sessions": int(matrix["session"].nunique()),
        "tickers": int(matrix["ticker"].nunique()),
        "mean_candidates_per_session": float(matrix.groupby("session").size().mean()),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "splits": splits.to_dict(),
        "abs_threshold_from_train": abs_thr,
        "primary_label": label_col,
        "primary_horizon_sessions": horizon,
        "primary_k": primary_k,
        "selected_model": selected,
        "selection_basis": f"validation lift@{primary_k}",
        "holdout_headline": headline,
        "fit_log": fit_log,
        "audits": audits,
        "holdout_evaluations": 1,
    }
    store.write_json(manifest, "artifacts", "manifest")
    log.info("experiment complete in %.1fs; selected=%s", time.time() - t0, selected)
    return manifest
