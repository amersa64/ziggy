# ziggy — Experiment 1: point-in-time market intelligence pipeline

> **Given everything that was publicly knowable at time _T_, can we automatically
> identify a small ranked set of stocks/events that contains substantially more
> consequential future market activity than a naive baseline?**

This repository answers that question with a historical, point-in-time-clean
experiment. It is an **information retrieval and ranking system**, not a trading
agent. It does not decide what to buy, it does not estimate direction, it does
not size positions, and it calls no LLM API. Its only job is to turn a large
universe of raw public information into a short, evidence-carrying shortlist:

> _"These are the 10–30 situations most worth spending expensive reasoning
> tokens investigating tonight, and here is why."_

A separate downstream agent (Experiment 2) decides whether an event matters,
whether the market has mispriced it, direction, confidence, and whether to trade
at all. This repository deliberately solves none of that.

---

## The one invariant

    A snapshot for session D is taken at 20:00 America/New_York on D.
    It may use information publicly released at or before that instant, and
    nothing else. The earliest possible action is the open of session D+1.

An SEC document accepted at 21:00 ET on Tuesday belongs to **Wednesday's**
snapshot, not Tuesday's. Original publication timestamps are persisted
everywhere; dates are never substituted for timestamps where timestamps exist.

`tests/test_calendar.py` pins the convention at the boundaries — 19:59, exactly
20:00, 20:01, weekends, market holidays, DST transitions, and the end of the
study window.

## How the point-in-time claim is made checkable

A point-in-time claim that is not tested is a point-in-time hope. Five
independent mechanisms, all of which run as part of the experiment and report
into the evidence package whether or not the result is flattering:

| Mechanism | What it would catch | Where |
|---|---|---|
| **Truncation property test** — recompute every feature on a panel truncated at session D and require bit-comparable values | centred windows, global normalisation, backfill from the future | `tests/test_no_lookahead.py` |
| **Availability audit** — every disclosure feeding a snapshot must have an acceptance instant at or before it | timestamp/timezone mistakes, off-by-one snapshot mapping | `ziggy/audit.py::feature_availability_audit` |
| **Acceptance-timezone audit** — EDGAR writes Eastern wall-clock with a spurious `Z` | a 5-hour lookahead on every filing | `ziggy/audit.py::acceptance_timezone_audit` |
| **Shuffle control + permutation null** — destroy the score, keep everything else; lift must collapse to ~1.0 | leakage hiding anywhere in the stack | `ziggy/audit.py`, `ziggy/evaluate.py` |
| **Leakage canary** — a deliberately cheating ranker that _does_ see the outcome | a leak detector that never fires and proves nothing | `ziggy/audit.py::leakage_canary` |

The truncation test has a control of its own: an obviously leaky centred-window
feature that the harness must reject. A test that cannot fail is not a test.

---

## Pipeline

```
prices ──► point-in-time universe ──► candidate grid (snapshot × ticker)
                                            │
SEC EDGAR submissions ──► events ───────────┤
Form 345 insider datasets ──────────────────┤──► features ──► ranking ──► top-k
FRED/ALFRED vintages ───────────────────────┤                                │
market/regime context ──────────────────────┘                                │
                                                                             ▼
                                          labels (the only forward-looking step)
                                                                             │
                                                            evaluation + audits
```

**Ordering is deliberate.** Prices come first because they define the universe;
the universe then tells us which CIKs are worth an EDGAR request, turning "every
filer on EDGAR" into a few thousand companies. Labels are computed last, in a
module nothing upstream imports.

### Data sources

| Source | Used for | Point-in-time handling |
|---|---|---|
| **SEC EDGAR submissions API** | form type, 8-K item codes, submission size, primary document, **`acceptanceDateTime`** | acceptance instant localised to ET and mapped to the first snapshot that could see it |
| **SEC Form 345 datasets** | insider transactions (open-market buys/sells) | Form 4 *filing* date drives availability, not the transaction date |
| **SEC filing documents** | textual novelty vs the issuer's own previous same-form filing | compared only against chronologically earlier filings |
| **Market data** (stooq / Yahoo, pluggable) | OHLCV, benchmarks, VIX | session D's own bar is legitimate at a 20:00 snapshot |
| **FRED / ALFRED** | macro context | publication lag modelled explicitly; revisable series pulled as dated **vintages**, so a 2021 snapshot sees the 2021 print of payrolls, not the 2024 revision |
| **News** | — | **interface only, deliberately unimplemented** (see below) |

### On news

Historical news is genuinely useful here and equally genuinely dangerous. Most
free "historical" news sources carry crawl/index time rather than publication
time, or have coverage that is itself a function of what later turned out to
matter — survivorship bias in disguise. Either one silently inflates every
number this experiment produces.

So `ziggy/providers/news.py` defines the interface and
`NewsProvider.integrity_requirements()` states the bar a source must clear.
Nothing implements it against a feed we cannot vouch for, and the limitation is
reported rather than papered over.

### Universe

Recomputed at **every** snapshot from trailing market data only: minimum price,
minimum trailing median dollar volume, minimum history, capped by liquidity
rank. A company that IPO'd in 2023 is absent before 2023; one delisted in 2022
disappears after its last bar. Today's index membership is never used.

`survivorship_report` measures the delisting coverage the price vendor actually
provides — a suspiciously low disappearance rate is the signature of a vendor
that drops dead companies, and it is reported as a number, not assumed away.

### Labels — "consequential activity"

Entry is the **open of the session after the snapshot**; exit is the close of
`t+h`. Four definitions are reported; the cross-sectional one is primary:

- `cs_q90` — `|excess move|` in the top decile of that snapshot's own
  cross-section. Base rate is exactly 10% by construction, so "lift" means
  precisely "how many times better than a coin flip over the same candidate
  set", with no regime drift in the denominator.
- `abs` — `|excess move|` above a threshold **calibrated on the training split
  only**.
- `cs_q90_abret` — the same rule on the **beta-adjusted** excess move. Plain
  excess (stock minus benchmark) leaves a low-beta name carrying `(1 - beta)`
  times the market move as apparent idiosyncratic activity, which in a volatile
  stretch is enough to push large, low-beta names up the `|excess|` ranking —
  visible as the "just rank the biggest names" baseline scoring above chance
  when it should not.
- `cs_q90_volnorm` — the same rule applied to `|excess move| / the name's own
  expected move`. An absolute-magnitude label is partly satisfiable by ranking on
  trailing volatility (volatile names move more, by definition); normalising
  removes that free lunch. On a controlled panel, ranking purely on `vol_21d`
  scores **lift 2.3** against the absolute label and **0.28** against this one.
  A ranker that wins on the absolute label alone has found volatility; one that
  wins on both has found something about situations.
- `vol_expansion`, `volume_shock` — consequence without requiring a directional
  move.

Names that stop trading mid-window are **truncated, not dropped**: the exit uses
the last observed close and the row is flagged. Dropping them would discard
exactly the most consequential outcomes — bankruptcies and cash acquisitions.

### Protocol

Chronological splits (60/20/20) with a 21-session embargo between them, so no
training label window can overlap the first validation snapshot. Rankers and the
absolute-threshold calibration see the training split only. Model selection
happens on **validation**. The holdout is scored **once**, with the selection
already fixed, and the manifest records how many times it was touched.

Three rankers span the transparency/flexibility axis:

- `deterministic` — a fixed, hand-specified linear combination of within-day
  percentile ranks. No fitting, cannot overfit, fully legible. The weights were
  written down before any evaluation was run.
- `logistic`, `gbm` — fitted on train only, additionally run in **walk-forward**
  mode where the model ranking year _Y_ is fitted only on data strictly before
  _Y_. That is the deployment-realistic setting.

Nine naive baselines, chosen to be the ones a sceptic would actually propose —
including `inverse_liquidity` and `prior_abs_move`, which are strong enough that
a weak model can easily lose to them.

### Two questions, not one

The primary experiment asks whether the ranking finds more consequential
activity than a naive baseline. But an absolute-magnitude label can be answered
by a volatility ranker, so the evidence package also runs a **secondary
experiment**: the same rankers refit on the volatility-normalised label, where
"just pick the jumpy names" is worth less than nothing.

The pairing is what makes the result interpretable. A model that wins the
primary and loses the secondary has found volatility. One that wins both has
found something about situations. Reporting only the first would let a
volatility ranker pass as a disclosure model.

### Metrics

Computed per snapshot then averaged, so a handful of huge days cannot carry the
result. Uncertainty comes from a **stationary block bootstrap over snapshots**
(block ≈ 10 sessions) because adjacent days are not independent: overlapping
label windows and shared regimes induce serial correlation an i.i.d. bootstrap
would ignore, giving intervals far too narrow.

`precision@k` · `lift@k` · `capture@k` (share of the day's total absolute excess
movement inside the shortlist) · `mean_mag@k` · `ndcg@k` · `hit_any@k`, plus
paired bootstrap comparisons against every baseline on the same sessions.

---

## Running it

```bash
pip install -r requirements.txt

python -m ziggy.cli check-sources          # confirm the data hosts are reachable
python -m ziggy.cli all                    # ingest → build → experiment → report
```

Individual stages: `ingest`, `build`, `experiment`, `report`. Each caches its
output, so a stage re-run reuses what is already on disk unless you pass
`--force`.

If the data hosts are behind a network policy that has not opened yet,
`scripts/autostart_real_run.sh` polls until CONNECT succeeds and then runs the
whole pipeline unattended, so nobody has to be watching at that moment.

Set a contact address for SEC fair-access compliance:

```bash
export ZIGGY_USER_AGENT="your-project (you@example.com)"
```

Outputs land in `artifacts/`: `REPORT.md`, figures, `manifest.json` (the full
provenance record — config, splits, features, audits, selection basis),
`summary.parquet`, per-session metrics and baseline comparisons.

### The product

```bash
python -m ziggy.cli shortlist --k 20            # latest snapshot
python -m ziggy.cli shortlist --date 2025-11-04 --k 30 --json
```

```
Shortlist for 2026-08-31 — ranked by gbm_wf

  1. SIM1160  score +0.5130   8-K  [exhibits|results]
     why: high-salience 8-K item; earnings/results item; unusual volume; filed something tonight
     https://www.sec.gov/Archives/edgar/data/1001160/.../doc.htm
  2. SIM0348  score +0.3713   8-K  [exhibits|results]
     why: high-salience 8-K item; earnings/results item; unusual volume; smaller / less liquid name
     https://www.sec.gov/Archives/edgar/data/1000348/.../doc.htm
  3. SIM0162  score +0.3685   no filing
     why: unusual volume; high trailing volatility; turnover well above its own average
```

Each row carries the disclosure that triggered it, a link to the source
document, the market context, and a plain statement of which signals put it
there. It stops short of an opinion: nothing here says the event is good or
bad, priced or mispriced, or that anything should be traded. That is
Experiment 2's job — this is the desk research pack it starts from.

### The simulation control

```bash
CFG=configs/simulation.yaml
python -m ziggy.cli simulate   --config $CFG      # synthetic corpus, no network
python -m ziggy.cli build      --config $CFG
python -m ziggy.cli experiment --config $CFG
python -m ziggy.cli report     --config $CFG
```

(`all` is not used here: its first stage is `ingest`, and the whole point of the
simulation is that it never touches the network.)

The corpus has a **known** planted relationship: a salient 8-K raises the
*scale* (not the direction) of the next few days' idiosyncratic move, more so
for smaller and more volatile names.

This is a positive control. It validates the pipeline and the evaluation harness
end to end and distinguishes "the pipeline reports no signal" from "the pipeline
is broken". Artifacts from it are stamped `simulated` and are **not evidence
about real markets**.

## Layout

```
ziggy/
  calendar.py       snapshot convention (the one invariant)
  universe.py       point-in-time tradability screen + survivorship measurement
  events.py         filings → (snapshot, ticker) event clusters
  labels.py         the only module allowed to look forward
  splits.py         chronological splits with embargo
  evaluate.py       metrics, block bootstrap, permutation null, univariate lift
  audit.py          leakage / timestamp / survivorship audits
  shortlist.py      the product: top-k for one snapshot, with evidence
  simulate.py       synthetic point-in-time corpus (positive control)
  sources.py        reachability probe for the external data hosts
  pipeline.py       ingestion and dataset construction
  cli.py            ingest · build · experiment · report · shortlist · simulate
  providers/        sec_edgar · market · macro · news (interface only)
  features/         price · sec · text · macro · assemble
  rank/             baselines · models
  experiment/run.py the protocol
  report/           evidence package
configs/            default.yaml · simulation.yaml
tests/              calendar · no-lookahead · labels · evaluation · PIT integrity
                    · providers · pipeline · text · shortlist · splits & models
docs/               DESIGN.md · LIMITATIONS.md
```

## What this repository does not claim

- It does not claim the shortlist is profitable. "Consequential" means *large
  absolute excess movement*, which is not the same as *predictable direction*,
  and no trading result is implied.
- It does not model transaction costs, borrow, capacity or market impact.
- Coverage limitations (delisted-security coverage, absent news, dividend
  adjustment) are quantified in the report's limitations section rather than
  omitted.

See `docs/LIMITATIONS.md` for the complete list.
