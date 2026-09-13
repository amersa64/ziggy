# Design notes

## Why a retrieval problem rather than a prediction problem

The downstream agent has a fixed, expensive reasoning budget. It cannot read
every filing from every issuer every night. The question that actually matters
to it is not "what will this stock do" but **"where should I look first"** — and
that is a ranking problem with a well-understood evaluation vocabulary
(precision@k, capture, NDCG, lift over base rate).

Framing it this way has a second benefit: it makes the null hypothesis concrete
and cheap to state. Random selection from the same universe has lift 1.0 by
construction. Any claim the pipeline makes has to beat that, and beat the
baselines a sceptic would propose.

## Why rank the whole universe, not just the filers

A tempting shortcut is to treat "companies that filed something tonight" as the
candidate set. That quietly assumes the answer — it bakes in the belief that
disclosure is what matters, and it makes the base rate incomparable across days
because the candidate set changes size and composition.

Instead **every tradable name is a candidate every session**, with zeroed event
features when nothing was filed. Names with no disclosure have to earn their
rank from price and flow behaviour. `any_disclosure` and `n_filings` then exist
as *baselines*, so "does disclosure help" becomes a measured comparison rather
than an assumption.

## Why cross-sectional percentile ranks everywhere

Raw feature levels are not comparable across regimes: a volume z-score of 3 in
March 2020 means something different from a 3 in August 2021, and a model fitted
on one regime extrapolates badly into another.

Converting features to **within-day percentile ranks** fixes this and is
point-in-time safe by construction — every value in the comparison set was
knowable at that same snapshot. It also removes the need for any global scaler,
which is a classic vector for leaking distributional information across the
train/test boundary.

The same logic drives the primary label: a *cross-sectional* top-decile
definition keeps the base rate pinned at 10% in calm and stressed markets alike,
so lift is comparable through time.

## Why the deterministic scorer exists

A fitted model that beats the baselines is a weaker result than it looks unless
you can also show that a transparent, unfitted scorer built from stated priors
does something sensible. The deterministic scorer:

- cannot overfit (there is nothing to fit),
- makes the hypothesis explicit and falsifiable — the weights *are* the theory,
- provides a floor: if the learned models cannot beat a hand-written linear
  combination of percentile ranks, they have not learned anything worth having.

Its weights were written down before any evaluation was run and are not tuned.

## Why walk-forward as well as frozen-train

"Fit once on the first 60%, evaluate on the last 20%" answers a slightly
artificial question: by the end of the holdout the model is two years stale.
Walk-forward refitting — the model ranking year *Y* sees only data before *Y* —
is what deployment actually looks like. Both are reported. A large gap between
them is itself informative: it says the relationship is drifting.

## Why a block bootstrap

Daily metrics are serially correlated for two reasons: 5-day label windows
overlap on consecutive days, and market regimes persist for months. An i.i.d.
bootstrap over sessions would treat 1,400 snapshots as 1,400 independent
observations and produce confidence intervals roughly `sqrt(block)` times too
narrow. The stationary block bootstrap resamples runs of ~10 consecutive
sessions, which preserves that dependence.

## Why events are clustered

A company that files an 8-K, a 10-Q and six Form 4s on the same evening is one
situation, not eight. Clustering to `(snapshot, ticker)` also makes the
candidate grid rectangular, which is what lets the whole thing be evaluated as a
clean daily ranking.

## Where the expensive work deliberately is not

No sentiment model, no embedding model, no LLM. Textual novelty is measured with
hashed token counts and fixed lexicons — deterministic, cheap, auditable, and
reproducible from the cache. The expensive reasoning is Experiment 2's job; if
Experiment 1 needed a language model to be useful, it would not be a filter, it
would be the thing it is supposed to be filtering *for*.

## Handoff to Experiment 2

`artifacts/` carries everything the downstream agent needs to consume a
shortlist without re-deriving it:

- `processed/scored.parquet` — every candidate, its score under every ranker,
  its split, and its realised outcome.
- `interim/events.parquet` — the underlying disclosure cluster for each
  `(snapshot, ticker)`, including item codes, item families, accession numbers
  and a URL to the primary document, so a reasoning agent can go read the source.
- `artifacts/manifest.json` — the provenance record: config, splits, feature
  list, audit results, selection basis, and how many times the holdout was
  touched.

The contract is intentionally narrow: Experiment 1 hands over *a ranked list
with evidence attached*, and makes no claim about what should be done with it.
