# Glean Utility Bill Parser — Confidence Scoring Criteria

_Last updated: 2026-06-26_

This document explains the **Confidence** score shown after each bill is parsed. It is
written so a non-developer can review it. Share/review with the team as needed.

---

## 1. What the score is — and what it is NOT

The number is a **Confidence** estimate, **not a measured accuracy**.

For most months the parser has **no source of truth** to check itself against, so it
cannot *know* how accurate it was. Instead it computes a confidence score from real
signals about how the data was obtained and how internally consistent it is.

**It is honest about two different things:**

- **Verified** data — a few values the parser can actually check against the bill
  (see §3). These are trustworthy.
- **Estimated** data — values read by measuring chart bars in pixels. These are good
  but cannot be independently confirmed.

**Key limitation (important):** the score flags *messy* extractions (missing bars,
disagreements between passes, fallbacks). It **cannot** catch a *confident-but-wrong*
read — e.g. misreading a printed "45" as "46." Those look perfectly clean and will
score high. So treat the score as **"how much should I trust this / which months
should I double-check,"** not a guarantee of correctness.

---

## 2. Labels

| Label | Score | Color | Meaning |
|-------|-------|-------|---------|
| **High** | 90–98 | green | Trust it; spot-check only |
| **Medium** | 75–89 | amber | Likely fine; glance at it |
| **Review** | below 75 | red | Double-check before using |

Both the **percentage** and the **label** are shown — overall for the bill, and per
month in the usage table. Review-flagged months are highlighted so they're easy to spot.

---

## 3. How a value is obtained (the biggest factor)

Confidence starts from **how** the usage history was read. Higher = more reliable source.

| Source | Base score | Notes |
|--------|-----------|-------|
| **Printed numbers** (Toronto, Ottawa) | 93 | The kWh values are printed on the bill; the parser reads them directly. |
| **Printed daily-average** (Alectra) | 90 | Daily numbers + day counts are printed; the parser multiplies them. One extra step. |
| **Anchor-calibrated pixels** (Tillsonburg, Milton, Enova) | 85 | Bar heights measured in pixels, then scaled so the newest bar equals the bill's known current-period total. |
| **Gridline-calibrated pixels** (Elexicon, Guelph, Halton) | 80 | Bar heights measured against the chart's printed gridlines. No known total to anchor to. |

The single most reliable check the parser has: the **newest month** is the current
billing period, whose exact total the bill prints. That month is treated as **verified**.

---

## 4. Adjustments to the overall score

Starting from the source base above, the overall score is adjusted by these checks:

| Check | Effect |
|-------|--------|
| Both extraction passes **agree** on the charges (total + TOU) | no penalty |
| The two passes **disagree** on a key charge | −8 |
| Newest bar **matches** the bill's printed current-period total (within ~3%) | no penalty (confirms calibration) |
| Newest bar **misses** the bill's total by more than ~3% | −7 |
| On-Peak + Mid-Peak + Off-Peak **sum to** the total kWh | no penalty |
| TOU components **don't** sum to the total (>3% off) | −5 |
| Each **missing / empty** month in the 12-month window | −3 (max −15) |

Final overall score is clamped to a sensible range (**55–98**) — we never show 100%
(nothing is certain) and never below 55 if a usable result was produced.

---

## 5. Per-month confidence

Each month in the usage table gets its own score:

- **Newest month** (anchored to the bill's exact total) → **verified**, ~96, High.
- **Printed-number month** → the source base (≈93), High.
- **Estimated (pixel) month** → the source base (80–85), usually Medium.
- **Day-count guardrail fired** (an implausible billing-day count was replaced with the
  calendar month length — Alectra-style bills only) → base − 12, flagged Medium.
- **Missing / empty month** (no bar found) → ~40, **Review**.

---

## 6. The reasons line

Alongside the number, the parser shows 2–3 short reasons so the score isn't a black box,
e.g.:

- ✓ Newest month matches the bill's total
- ✓ Both extraction passes agreed on the charges
- ⚠ 11 months estimated from chart pixels (not independently verified)
- ⚠ April's billing-day count looked wrong and was estimated

---

## 7. Where it lives in the code (for developers)

- Scoring is computed in `app.py` (`compute_confidence(...)`), using signals gathered
  during `extract_with_claude` (extraction method, two-pass agreement, anchor match) and
  finalized in the `/upload` handler after the 12-month history is assembled.
- The result is returned per bill as a `confidence` object (overall %, label, reasons)
  plus per-month `conf` / `conf_label` fields, and rendered by `static/index.html`.

The numeric weights in §3–§5 are deliberately simple and conservative; they can be tuned
without changing the overall design.
