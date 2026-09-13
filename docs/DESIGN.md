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

## Why there is a volatility-normalised label as well

An absolute-magnitude label has a free lunch buried in it: volatile names move
more, by definition. A ranker that does nothing but sort on trailing sigma
therefore scores well without knowing anything about *situations*. On a
controlled panel with a 10x spread in per-name volatility, ranking purely on
`vol_21d` reaches **lift 2.3** against the absolute top-decile label.

So `trailing_vol` is one of the baselines — the comparison that matters is
against it, not against `random` — and a second label is reported alongside the
primary one:

    consequence_magnitude_volnorm = |excess move| / (the name's own expected
                                    h-session sigma, estimated at the snapshot)

On the same controlled panel, ranking on `vol_21d` against this label reaches
**lift 0.28** — worse than random. That is not a bug: estimated volatility mean
-reverts, so the highest-sigma names systematically move *less* than their own
recent volatility implied. The normalised label therefore asks the sharper
question — *which names will move far more than their volatility already
implied?* — and it actively punishes the shortcut.

Reading the two together is the point. A ranker that wins on the absolute label
and not the normalised one has found volatility. A ranker that wins on both has
found something about situations.

And reporting that comparison is necessary but not sufficient. If the primary
model scores poorly on the normalised label, that says the *model* failed the
harder test — not that the *features* would. Those are different claims, and
conflating them would let the report imply there is nothing beyond volatility
in the data when nobody ever looked. So the experiment includes a secondary
run: the same rankers refit **on** the normalised label, evaluated against the
same nine baselines. Frozen-train only — it is a secondary question and does
not get the compute budget or the selection authority of the primary one.

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

## The snapshot window spans a whole trading session

A 20:00 ET snapshot sees everything published since 20:00 ET the previous day —
which includes one complete trading session. That window contains two very
different kinds of disclosure:

- accepted **after** today's 16:00 close: the market has not traded on it, and
  the next session's open is the first opportunity to react;
- accepted **before or during** today's session: the market has already had
  hours to price it, so the snapshot is partly looking at a *reaction*, not at
  news.

`sec_minutes_past_close` is signed precisely so a ranker can separate these,
and `sec_after_hours` / `sec_before_open` make the split explicit. Collapsing
them — treating "filed today" as one thing — would blur the most actionable
distinction in the whole feature set.

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
  and a URL to the primary document.
- `interim/filing_text.parquet` — the extracted text of every document fetched,
  keyed by accession and carrying its acceptance instant and source URL. The
  downstream agent's entire job is to read these; handing it a cosine distance
  and a link would make it re-fetch everything this run already has.
- `artifacts/shortlist_<date>.json` — the finished product for one snapshot:
  rank, ticker, score, a plain-language reason, the disclosure that triggered
  it, its acceptance instant, the source URL, and the market context. This is
  the object Experiment 2 actually consumes.
- `artifacts/manifest.json` — the provenance record: config, splits, feature
  list, audit results, selection basis, and how many times the holdout was
  touched.

The contract is intentionally narrow: Experiment 1 hands over *a ranked list
with evidence attached*, and makes no claim about what should be done with it.
