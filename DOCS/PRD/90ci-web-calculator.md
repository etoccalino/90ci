# PRD — 90ci web calculator (v1)

## Problem & goal

People estimating an uncertain quantity (a cost, a revenue, an exposure) reason in ranges, not point values. 90ci already computes the 90% confidence interval of a model from the command line, but the TUI is inaccessible to non-technical users and hard to share. The goal is a **zero-backend web app** that lets anyone build a model in the browser, run it, and read off the 90% confidence interval and the shape of the output distribution. All computation runs client-side via the existing Rust engine compiled to WASM — there is no server.

## Primary UX

Recreated from the "Spreadsheet look" mock (`design/version-1/project/90ci Calculator - Spreadsheet.html`), the single-screen app has two columns.

**Left — build the model:**
- An **editable model name** (e.g. "Exchange exposure") that also drives the breadcrumb.
- A **formula bar** (`ƒx`) showing the equation with light syntax highlighting, and a **Run** button.
- A **Random variables** table — one row per variable, columns: Name, Distribution (a tagged pill: uniform / normal / range), 5th percentile, 95th percentile, and a Shape sparkline. A **+ New variable** affordance adds rows.

**Right — read the result:**
- An **output-distribution** area chart: the simulated histogram, with the 90% interval band shaded and dashed markers at the lower/upper bounds.
- A **90% confidence interval hero**: the range (lower – upper) in large type, plus its width, with the caption "the middle 90% of outcomes land in this range."

Running re-simulates and updates both the chart and the hero.

## In scope (v1)
- Build a model: name, equation, and N random variables (uniform / normal / range).
- Run the simulation client-side (WASM) and render the histogram + 90% CI.
- Fixed sample count: 5,000

## Out of scope (v1)
- Adjustable sample count (e.g. 1,000 / 10,000 / 100,000).
- The mock's **"Tweaks" panel** — it is a design-tool artifact for exploring visuals, not a product feature.
- Persistence, saving, sharing, and the "Shared" / "Comments" / "Run all" top-bar actions.
- Managing multiple models / a model library.
- Auth, accounts, any server-side anything.

## About variable bounds

The mock labels the variable bounds "5th / 95th percentile". The engine (`src/lib.rs`, `Distro::new`) interprets `lower`/`upper` differently per shape:
- **normal**: `mean = (upper + lower) / 2`, `sd = (upper - lower) / 3.29` — so `lower`/`upper` genuinely are the ~5th/95th percentiles (3.29 ≈ the 90% z-span).
- **uniform** and **range**: `lower`/`upper` are the **full min/max**, not percentiles — the middle 90% of a uniform over `[1000, 1200]` is `[1010, 1190]`, not `[1000, 1200]`.

So the "5th / 95th" label is accurate for normal but loose for uniform/range. For v1 we keep the current engine semantics and allow the labels/tooltips to be "loose" is meaning.
