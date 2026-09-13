"""The product: a ranked shortlist with its evidence attached.

Everything else in this repository exists to justify this function. Given a
session, it returns the ``k`` situations most worth spending expensive reasoning
tokens on that evening, each carrying the disclosure that triggered it, a link
to the source document, the market context, and a plain statement of which
signals put it there.

The shortlist deliberately stops short of an opinion. It does not say the event
is good or bad, priced or mispriced, or that anything should be traded. That is
Experiment 2's job; this is the desk research pack it starts from.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ziggy.features.assemble import cross_sectional_rank
from ziggy.rank.models import DETERMINISTIC_WEIGHTS
from ziggy.store import Store

log = logging.getLogger(__name__)

# Context carried alongside every shortlisted name, in the order a human reads it.
EVIDENCE_COLUMNS = [
    "sec_n_filings", "sec_n_8k", "sec_n_periodic", "sec_n_form4",
    "sec_minutes_past_close", "sec_log_total_size", "sec_size_log_ratio",
    "text_cosine_prior", "ret_5d", "exret_5d", "volume_z_20d", "vol_21d",
    "dist_52w_high", "insider_log_net_21d", "log_adv20_universe",
]

HUMAN_LABELS = {
    "sec_n_high_salience": "high-salience 8-K item",
    "sec_item_results": "earnings/results item",
    "sec_item_ma_assets": "acquisition or disposition",
    "sec_item_executive_change": "executive change",
    "sec_item_non_reliance": "non-reliance on prior financials",
    "sec_item_auditor_change": "auditor change",
    "sec_item_impairment": "impairment",
    "sec_item_restructuring_costs": "restructuring costs",
    "sec_item_listing_compliance": "listing-compliance notice",
    "sec_item_bankruptcy": "bankruptcy item",
    "sec_item_control_change": "change of control",
    "sec_item_material_agreement": "material agreement",
    "sec_item_debt_obligation": "new debt obligation",
    "sec_item_debt_acceleration": "debt acceleration",
    "sec_item_unregistered_sale": "unregistered share sale",
    "sec_n_8k": "8-K filed",
    "sec_n_periodic": "periodic report filed",
    "sec_n_13d": "13D/13G filed",
    "sec_size_log_ratio": "filing much longer than the last of its kind",
    "text_cosine_prior": "text unlike the issuer's previous filing",
    "text_new_token_mass": "substantial new language",
    "text_neg_delta": "more negative language than last time",
    "sec_sessions_since_8k": "first 8-K in a long while",
    "volume_z_20d": "unusual volume",
    "dollar_volume_ratio": "turnover well above its own average",
    "vol_ratio_5_21": "volatility expanding",
    "abs_ret_1d": "large move today",
    "range_pct_1d": "wide intraday range",
    "vol_21d": "high trailing volatility",
    "idio_vol_63d": "high idiosyncratic volatility",
    "amihud_21d": "thin, impact-sensitive",
    "drawdown_63d": "in drawdown",
    "log_adv20_universe": "smaller / less liquid name",
    "insider_buy_ratio_21d": "insider buying",
    "has_disclosure": "filed something tonight",
    "adv_trend_20_60": "volume trending up",
    "accel_5_21": "price accelerating",
}


def _reasons(ranks: pd.Series, top_n: int = 4) -> str:
    """Plain-language statement of why a name is on the list.

    Contributions come from the transparent deterministic weights even when the
    ranking itself is a fitted model: a legible account of the evidence is worth
    more to a downstream reasoner than an exact attribution of a boosted tree.
    """
    contrib = {}
    for feat, w in DETERMINISTIC_WEIGHTS.items():
        if feat not in ranks.index or pd.isna(ranks[feat]):
            continue
        c = w * (float(ranks[feat]) - 0.5)
        if c > 0.04:
            contrib[feat] = c
    if not contrib:
        return "no single standout signal; ranked on the combination"
    top = sorted(contrib, key=contrib.get, reverse=True)[:top_n]
    return "; ".join(HUMAN_LABELS.get(f, f.replace("_", " ")) for f in top)


def build_shortlist(
    cfg,
    session: str | pd.Timestamp | None = None,
    k: int | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Top-``k`` situations for one snapshot, with evidence."""
    store = Store(cfg)
    scored = store.read("processed", "scored")
    matrix = store.read("processed", "candidates")
    events = store.read("interim", "events") if store.exists("interim", "events") else pd.DataFrame()

    if model is None:
        try:
            model = store.read_json("artifacts", "manifest")["selected_model"]
        except (FileNotFoundError, KeyError):
            model = "deterministic"
    score_col = f"score_{model}"
    if score_col not in scored.columns:
        raise KeyError(f"{score_col} not present; available: "
                       f"{[c for c in scored.columns if c.startswith('score_')]}")

    usable = scored.loc[scored[score_col].notna(), "session"]
    session = pd.Timestamp(session) if session is not None else usable.max()
    k = int(k or cfg.ranking["primary_k"])

    day = scored[(scored["session"] == session) & scored[score_col].notna()]
    if day.empty:
        raise ValueError(f"no scored candidates for {session.date()}")

    day = day.nlargest(k, score_col)
    ctx = matrix[matrix["session"] == session].set_index("ticker")
    ranks = cross_sectional_rank(
        matrix[matrix["session"] == session],
        [c for c in DETERMINISTIC_WEIGHTS if c in matrix.columns],
    )
    ranks.index = matrix[matrix["session"] == session]["ticker"].values

    rows = []
    for i, (_, r) in enumerate(day.iterrows(), start=1):
        t = r["ticker"]
        c = ctx.loc[t] if t in ctx.index else pd.Series(dtype=float)
        row = {
            "rank": i, "session": session.date().isoformat(), "ticker": t,
            "model": model, "score": float(r[score_col]),
            "why": _reasons(ranks.loc[t]) if t in ranks.index else "",
        }
        if len(events):
            ev = events[(events["snapshot_session"] == session) & (events["ticker"] == t)]
            if len(ev):
                e = ev.iloc[0]
                row.update({
                    "forms": e.get("forms", ""), "item_codes": e.get("item_codes", ""),
                    "item_families": e.get("item_families", ""),
                    "accepted_at": str(e.get("last_accepted_at", "")),
                    "source_url": e.get("primary_url", ""),
                })
        for col in EVIDENCE_COLUMNS:
            if col in c.index:
                v = c[col]
                row[col] = None if pd.isna(v) else round(float(v), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    log.info("shortlist for %s: %d names by %s", session.date(), len(out), model)
    return out


def format_shortlist(df: pd.DataFrame, max_why: int = 88) -> str:
    """Terminal rendering."""
    if df.empty:
        return "(empty shortlist)"
    lines = [f"Shortlist for {df['session'].iloc[0]} — ranked by {df['model'].iloc[0]}", ""]
    def text_of(row, field: str) -> str:
        v = getattr(row, field, None)
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    for r in df.itertuples(index=False):
        forms = text_of(r, "forms") or "no filing"
        fams = text_of(r, "item_families")
        why = (r.why[: max_why - 1] + "\u2026") if len(r.why) > max_why else r.why
        lines.append(f"{r.rank:>3}. {r.ticker:<8} score {r.score:+.4f}   {forms}"
                     + (f"  [{fams}]" if fams else ""))
        lines.append(f"     why: {why}")
        url = text_of(r, "source_url")
        if url:
            lines.append(f"     {url}")
    return "\n".join(lines)
