# Limitations

Stated plainly, because a result whose caveats are buried is not a result.

## 1. Survivorship and delisted-security coverage

The universe is reconstructed point-in-time from trailing market data, so index
membership is never back-projected. But that only removes the *selection* half
of survivorship bias. The other half lives in the price vendor: **if the vendor
does not retain history for companies that were delisted, those companies are
absent from the panel entirely** and no amount of careful universe construction
brings them back.

`ziggy/universe.py::survivorship_report` measures this directly and the number
goes into the report. US equity markets retire roughly 3–5% of listed names per
year through mergers, bankruptcies and exchange delistings; a panel showing far
less than that is a panel of survivors.

Direction of the bias: **optimistic for the naive baselines, ambiguous for the
ranker.** Delisted names are disproportionately the consequential ones (the
bankruptcy, the cash acquisition), and they are also disproportionately the ones
a disclosure-aware ranker should catch. Removing them removes hard-to-find
winners from everybody, but it removes them from the *upside* of the ranker more
than from the baselines.

Mitigations in place: names that stop trading mid-window are truncated rather
than dropped (`fwd_truncated_*`), and the market-data provider interface exposes
`covers_delisted` so the report states the position for whichever vendor was
used rather than guessing.

## 2. No news

See the "On news" section of the README. The interface exists; nothing
implements it against a source whose publication timestamps and coverage we
cannot vouch for. This is the single largest information gap in the pipeline: a
great deal of what makes a situation worth researching at 20:00 on a Tuesday was
reported by a journalist, not filed with the SEC.

Consequence: the measured lift is a **lower bound** on what the information
layer could achieve with a licensed point-in-time archive — and, separately, the
relative importance of SEC features is overstated versus a world that includes
news.

## 3. Ticker ↔ CIK mapping is not point-in-time

SEC publishes the *current* ticker mapping. A company that changed its symbol is
mapped under today's symbol for its whole history. The mapping is used only to
attach filings to price series, never as a universe filter, so the effect is
missing or misattached filings for renamed issuers rather than lookahead. It is
still a defect, and the honest fix is a historical symbology source.

## 4. Dividend adjustment

Split adjustment is applied; dividend reinvestment depends on the vendor
(`MarketDataProvider.adjustment` records which). Over 1–21 session horizons the
dividend drift is small relative to the effects being measured, but it is a
known bias in the excess-return calculation and it is not zero.

## 5. "Consequential" is not "profitable"

The primary label is **absolute** excess movement. A name that moves 12% is
consequential whichever way it moved. This is deliberate — Experiment 1's job is
retrieval, and direction is Experiment 2's problem — but it must not be read as
a trading result. Nothing here models transaction costs, borrow availability,
short-sale constraints, capacity or market impact.

## 6. Macro vintages are sampled, not exhaustive

Revisable series are pulled as dated ALFRED vintages at a fixed cadence
(`macro.vintage_cadence_days`). Between vintage dates the snapshot sees the most
recent vintage it could have seen, which is correct, but a release landing
mid-cadence is picked up slightly late. The direction of this error is
conservative — the pipeline knows *less* than it could have, never more.

## 7. Acceptance-time fallback for old filings

Filings without an `acceptanceDateTime` fall back to 17:30 ET on the filing
date. That is the conservative (latest plausible) assumption and it is flagged
per row via `accepted_at_imputed`, but it is an assumption.

Form 345 structured datasets carry a filing *date* rather than an acceptance
instant, so insider features use the same conservative 17:30 ET convention.

## 8. Document-text coverage is budgeted

Textual novelty requires downloading filing documents. The budget
(`sec.document_budget`) is spent on the largest filings of each form, on the
reasoning that a 400-byte 8-K has no text to be novel about. Text features are
therefore missing for many small filings and the missingness is not random.
Models see this as absent values rather than zeros; the deterministic scorer
simply drops the weighted features it cannot find.

## 9. Multiple comparisons

Several rankers, several `k`, several label definitions and several splits are
reported. The holdout headline is a **single** pre-committed combination (model
selected on validation, primary `k`, primary label); everything else is reported
as supporting detail, not as a family of independent tests. Readers should treat
the secondary numbers as descriptive.

## 10. One market, one era

US equities, 2021 onwards. That window contains a meme-stock episode, a rate
shock, a bear market and a recovery — several regimes, but one market and one
decade. The per-regime and per-year breakdowns in the report exist so that a
result driven by a single episode is visible rather than averaged away.
