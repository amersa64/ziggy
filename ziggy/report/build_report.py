"""Render the evidence package: figures plus a markdown report.

Everything here reads from ``artifacts/`` -- it computes no new statistics, so
the report cannot disagree with the experiment that produced it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ziggy.store import Store  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = {
    "model": "#1f4e79", "baseline": "#9aa5b1", "highlight": "#c0392b",
    "grid": "#e5e7eb", "text": "#1f2933",
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=PALETTE["text"], loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def figure_lift_vs_k(summary: pd.DataFrame, selected: str, out: Path) -> Path | None:
    h = summary[summary["split"] == "holdout"]
    if h.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for model, g in h.groupby("model"):
        g = g.sort_values("k")
        is_sel = model == selected
        ax.plot(g["k"], g["lift"], marker="o", markersize=4,
                color=PALETTE["model"] if is_sel else PALETTE["baseline"],
                linewidth=2.2 if is_sel else 1.0, alpha=1.0 if is_sel else 0.7,
                label=model, zorder=3 if is_sel else 1)
        if is_sel:
            ax.fill_between(g["k"], g["lift_lo"], g["lift_hi"], color=PALETTE["model"], alpha=0.18, zorder=2)
    ax.axhline(1.0, color=PALETTE["highlight"], linestyle="--", linewidth=1.2)
    ax.text(h["k"].max(), 1.02, "random baseline", color=PALETTE["highlight"], fontsize=8, ha="right")
    _style(ax, f"Holdout lift@k — selected model: {selected}", "shortlist size k",
           "lift over base rate")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    p = out / "fig_lift_vs_k.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def figure_metric_bars(summary: pd.DataFrame, k: int, selected: str, out: Path) -> Path | None:
    h = summary[(summary["split"] == "holdout") & (summary["k"] == k)].copy()
    if h.empty:
        return None
    h = h.sort_values("lift")
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.32 * len(h))), dpi=150)
    colors = [PALETTE["model"] if m == selected else PALETTE["baseline"] for m in h["model"]]
    err = [(h["lift"] - h["lift_lo"]).clip(lower=0), (h["lift_hi"] - h["lift"]).clip(lower=0)]
    ax.barh(h["model"], h["lift"], color=colors, xerr=err, error_kw={"elinewidth": 0.9, "ecolor": "#4b5563"})
    ax.axvline(1.0, color=PALETTE["highlight"], linestyle="--", linewidth=1.2)
    _style(ax, f"Holdout lift@{k}, every ranker, 95% block-bootstrap CI", "lift", "")
    fig.tight_layout()
    p = out / "fig_lift_bars.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def figure_stability(per_session: dict[str, pd.DataFrame], k: int, out: Path) -> Path | None:
    frames = []
    for split, df in per_session.items():
        if df is None or df.empty:
            continue
        g = df[df["k"] == k].copy()
        g["split"] = split
        frames.append(g)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["session"] = pd.to_datetime(d["session"])
    d = d.sort_values("session")
    monthly = d.set_index("session").groupby([pd.Grouper(freq="ME"), "split"])["lift"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=150)
    colors = {"train": "#9aa5b1", "validation": "#5b8db8", "holdout": PALETTE["model"]}
    for split, g in monthly.groupby("split"):
        ax.plot(g["session"], g["lift"], color=colors.get(split, "#888"), linewidth=1.6, label=split)
    ax.axhline(1.0, color=PALETTE["highlight"], linestyle="--", linewidth=1.2)
    _style(ax, f"Monthly mean lift@{k} through time", "", "lift")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / "fig_stability.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def figure_importances(imp: pd.DataFrame, out: Path, title: str) -> Path | None:
    if imp is None or imp.empty:
        return None
    d = imp.reindex(imp["weight"].abs().sort_values(ascending=False).index).head(22).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.28 * len(d))), dpi=150)
    colors = [PALETTE["model"] if w > 0 else PALETTE["highlight"] for w in d["weight"]]
    ax.barh(d["feature"], d["weight"], color=colors)
    _style(ax, title, "weight", "")
    fig.tight_layout()
    p = out / "fig_importances.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    d = df[cols] if cols else df

    def fmt(v):
        # numpy scalars are not instances of the builtin float.
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else floatfmt.format(float(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows) + "\n"


def verdict(head: dict, hardest: pd.Series | None, k: int) -> tuple[str, str]:
    """A rule-based answer to the question the repository exists to ask.

    Stated before any result was seen, so the report cannot be talked into a
    conclusion its own numbers do not support:

    * the lift CI must exclude 1.0 -- otherwise the ranking is not
      distinguishable from picking at random over the same universe;
    * the margin over the *hardest* naive baseline must exclude 0 -- otherwise
      the layer is real but not worth building, because something trivial does
      the same job.
    """
    lift_lo = head.get("lift_lo", float("nan"))
    beats_random = lift_lo > 1.0
    if hardest is None:
        return ("INCONCLUSIVE", "no baseline comparison was available")
    beats_best = hardest["lo"] > 0.0 and hardest["p_value"] < 0.05
    name = str(hardest["reference"]).replace("baseline_", "")

    if beats_random and beats_best:
        return ("YES", (
            f"The ranking finds substantially more consequential activity than chance "
            f"(lift@{k} CI lower bound {lift_lo:.2f} > 1.0), and it beats the hardest "
            f"naive alternative (`{name}`) by {hardest['diff']:+.3f} lift "
            f"[{hardest['lo']:.3f}, {hardest['hi']:.3f}], p={hardest['p_value']:.4f}. "
            f"A shortlist of {k} is worth the reasoning budget."))
    if beats_random and not beats_best:
        return ("QUALIFIED NO", (
            f"The ranking beats chance (lift@{k} CI lower bound {lift_lo:.2f} > 1.0) but is "
            f"not distinguishable from the naive `{name}` baseline "
            f"({hardest['diff']:+.3f} lift [{hardest['lo']:.3f}, {hardest['hi']:.3f}], "
            f"p={hardest['p_value']:.4f}). The information layer is real but is not yet "
            f"earning its keep: something trivial does the same job."))
    return ("NO", (
        f"The ranking is not distinguishable from picking at random over the same "
        f"universe (lift@{k} CI lower bound {lift_lo:.2f}). On this data, with these "
        f"features, the answer to the central question is no."))


def build_report(cfg) -> Path:
    store = Store(cfg)
    art = cfg.artifacts_dir
    art.mkdir(parents=True, exist_ok=True)
    manifest = store.read_json("artifacts", "manifest")
    summary = store.read("artifacts", "summary")
    selected = manifest["selected_model"]
    k = int(manifest["primary_k"])
    simulated = bool(store.read_json("raw", "prices_provenance").get("SIMULATED", False)) \
        if (cfg.raw_dir / "prices_provenance.json").exists() else False

    per_session = {}
    for split in ("train", "validation", "holdout"):
        if store.exists("artifacts", f"per_session_{split}"):
            per_session[split] = store.read("artifacts", f"per_session_{split}")

    figs = {
        "lift_vs_k": figure_lift_vs_k(summary, selected, art),
        "lift_bars": figure_metric_bars(summary, k, selected, art),
        "stability": figure_stability(per_session, k, art),
    }
    imp_name = f"importances_{selected.replace('_wf','')}"
    if store.exists("artifacts", imp_name):
        figs["importances"] = figure_importances(
            store.read("artifacts", imp_name), art, f"{selected}: feature weights"
        )

    head = manifest.get("holdout_headline", {})
    audits = manifest.get("audits", {})
    surv = store.read_json("artifacts", "survivorship_report") if (art / "survivorship_report.json").exists() else {}

    lines: list[str] = []
    A = lines.append
    A(f"# {manifest['experiment']} — evidence package\n")
    if simulated:
        A("> **SIMULATED DATA.** This run was produced from the synthetic corpus in\n"
          "> `ziggy/simulate.py`. It validates the pipeline and the evaluation harness\n"
          "> end to end and acts as a positive control. **It is not evidence about real\n"
          "> markets.** See `REPORT.md` for the real-data run.\n")
    A(f"_Generated {manifest['generated_at']} · config `{manifest['config_path']}` · "
      f"runtime {manifest['runtime_seconds']}s_\n")

    A("## The question\n")
    A("> Given everything publicly knowable at the snapshot, can we automatically\n"
      "> produce a short ranked list of situations that contains substantially more\n"
      "> consequential future market activity than a naive baseline?\n")

    hardest_row = None
    if store.exists("artifacts", "holdout_vs_baselines"):
        _c = store.read("artifacts", "holdout_vs_baselines")
        _c = _c[_c["k"] == k]
        if len(_c):
            hardest_row = _c.loc[_c["diff"].idxmin()]

    if head:
        tag, text = verdict(head, hardest_row, k)
        A(f"## Verdict: {tag}\n")
        A(f"{text}\n")

    A("## Headline result\n")
    if head:
        base = head.get("base_rate", float("nan"))
        A(f"On the **holdout** period ({manifest['splits']['holdout']['start']} → "
          f"{manifest['splits']['holdout']['end']}, {manifest['splits']['holdout']['n_sessions']} sessions), "
          f"ranking with **{selected}** and taking the top **{k}** names per snapshot:\n")
        A(f"- **precision@{k} = {head.get('precision', float('nan')):.3f}** "
          f"[{head.get('precision_lo', float('nan')):.3f}, {head.get('precision_hi', float('nan')):.3f}] "
          f"against a base rate of {base:.3f}\n")
        A(f"- **lift@{k} = {head.get('lift', float('nan')):.2f}×** "
          f"[{head.get('lift_lo', float('nan')):.2f}, {head.get('lift_hi', float('nan')):.2f}]\n")
        A(f"- **{head.get('expected_hits', float('nan')):.1f}** of the {k} shortlisted names were "
          f"consequential on an average day\n")
        A(f"- the shortlist captured **{head.get('capture', float('nan')) * 100:.1f}%** of the day's total "
          f"absolute excess movement while being "
          f"{k / head.get('mean_candidates', 1) * 100:.1f}% of the universe "
          f"(**{head.get('capture_lift', float('nan')):.2f}×** its share)\n")
        A(f"- selected names moved **{head.get('mag_ratio', float('nan')):.2f}×** as far as the average name\n")

    # Beating `random` is table stakes. The number that decides whether this
    # layer is worth building is the margin over the best naive alternative.
    if hardest_row is not None:
        if True:
            hardest = hardest_row
            name = str(hardest["reference"]).replace("baseline_", "")
            sig = "significant" if hardest["p_value"] < 0.05 else "NOT significant"
            clears = "clears" if hardest["diff"] > 0 else "**fails to clear**"
            A(f"\nAgainst the **hardest** of the nine naive baselines (`{name}`), the "
              f"selected model {clears} it by **{hardest['diff']:+.3f}** lift "
              f"[{hardest['lo']:.3f}, {hardest['hi']:.3f}], p={hardest['p_value']:.4f} "
              f"({sig}, paired block bootstrap over the same {int(hardest['n'])} sessions). "
              f"Beating `random` is table stakes; this is the comparison that decides "
              f"whether the layer is worth building.\n")
    A("")
    if figs.get("lift_bars"):
        A(f"![lift bars]({figs['lift_bars'].name})\n")
    if figs.get("lift_vs_k"):
        A(f"![lift vs k]({figs['lift_vs_k'].name})\n")

    A("## Protocol\n")
    s = manifest["splits"]
    A(f"- Snapshot convention: **{manifest['snapshot_convention']}**; the earliest action is the next session's open.\n")
    A(f"- Period {manifest['period']['start']} → {manifest['period']['end']}; "
      f"{manifest['sessions']} snapshots, {manifest['rows']:,} candidate rows, "
      f"{manifest['mean_candidates_per_session']:.0f} names ranked per snapshot.\n")
    A(f"- Chronological splits with a {cfg.splits['embargo_sessions']}-session embargo: "
      f"train {s['train']['start']}→{s['train']['end']} ({s['train']['n_sessions']}), "
      f"validation {s['validation']['start']}→{s['validation']['end']} ({s['validation']['n_sessions']}), "
      f"holdout {s['holdout']['start']}→{s['holdout']['end']} ({s['holdout']['n_sessions']}).\n")
    A(f"- Label: `{manifest['primary_label']}` — |excess move| over {manifest['primary_horizon_sessions']} "
      f"sessions in the top decile of that snapshot's own cross-section.\n")
    A(f"- Model chosen on **{manifest['selection_basis']}**; the holdout was scored "
      f"{manifest['holdout_evaluations']} time with the choice already fixed.\n")
    A(f"- {manifest['n_features']} features available to the ranker.\n")

    A("\n## All rankers on the holdout\n")
    h = summary[(summary["split"] == "holdout") & (summary["k"] == k)].copy()
    if not h.empty:
        h = h.sort_values("lift", ascending=False)
        h["precision_ci"] = h.apply(lambda r: f"[{r['precision_lo']:.3f}, {r['precision_hi']:.3f}]", axis=1)
        A(_md_table(h, ["model", "precision", "precision_ci", "lift", "capture", "mag_ratio", "ndcg", "hit_any"]))

    if store.exists("artifacts", "holdout_vs_baselines"):
        cmp = store.read("artifacts", "holdout_vs_baselines")
        cmp = cmp[cmp["k"] == k].sort_values("diff", ascending=False)
        A(f"\n### Paired comparisons at k={k} (selected model minus baseline, same sessions)\n")
        A(_md_table(cmp, ["reference", "diff", "lo", "hi", "p_value", "n"]))

    if store.exists("artifacts", "by_year"):
        A("\n## Stability through time\n")
        A(_md_table(store.read("artifacts", "by_year")))
    if figs.get("stability"):
        A(f"\n![stability]({figs['stability'].name})\n")
    if store.exists("artifacts", "holdout_by_regime"):
        A("\n## By volatility regime (holdout)\n")
        A(_md_table(store.read("artifacts", "holdout_by_regime")))

    if store.exists("artifacts", "label_robustness"):
        A("\n## Robustness to the definition of 'consequential'\n")
        A(_md_table(store.read("artifacts", "label_robustness"),
                    ["label", "k", "precision", "lift", "base_rate"]))

    if figs.get("importances"):
        A("\n## What the ranker is using\n")
        A(f"![importances]({figs['importances'].name})\n")

    if store.exists("artifacts", "univariate_lift"):
        uni = store.read("artifacts", "univariate_lift")
        A("\n## Which single signals carry information\n")
        A(f"Lift@{k} from ranking on one feature alone, measured on the **validation** "
          "split. Both directions are tried, so these numbers are optimistically "
          "biased — read them as a ranking of signals, not as significance tests.\n")
        A(_md_table(uni.head(20), ["feature", "direction", "best_lift", "coverage"]))
        A("\nWeakest signals in the same set:\n")
        A(_md_table(uni.tail(8), ["feature", "direction", "best_lift", "coverage"]))

    A("\n## Point-in-time and leakage audits\n")
    for name, res in audits.items():
        if not isinstance(res, dict):
            continue
        status = res.get("status") or res.get("verdict") or ""
        A(f"**{name}** — `{status}`\n")
        A("```json\n" + json.dumps({k2: v for k2, v in res.items() if k2 != "hour_histogram"}, indent=2, default=str) + "\n```\n")

    if surv:
        A("\n## Survivorship\n")
        A("```json\n" + json.dumps(surv, indent=2, default=str) + "\n```\n")

    A("\n## Reproduce\n")
    A("```bash\npip install -r requirements.txt\n"
      f"python -m ziggy.cli all --config {manifest['config_path']}\n```\n")

    path = art / "REPORT.md"
    path.write_text("\n".join(lines))
    log.info("report written: %s", path)
    return path
