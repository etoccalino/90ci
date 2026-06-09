# v1 implementation plan (DRAFT)

Derived from `DOCS/PRD/90ci-web-calculator.md` and `DOCS/PRD/v1-acceptance.md`. The organizing principle is **build the feedback loop first, then drive every stage with tests that encode the §4 error matrix and the §1/§3/§5/§6/§7 acceptance criteria.** Each stage ends with a runnable milestone and closes specific Definition-of-Done boxes (§9).

## Strategy

- **Feedback loop before features.** Stage 0 makes all three test runners (Rust, front-end/jsdom, WASM boundary) runnable and green so every later stage is test-first. Today only `cargo test -p ninety_ci_core` runs (21 unit + 2 integration, confirmed green 2026-06-08); the front-end has no test harness and there are no boundary tests.
- **Tests encode acceptance, not the other way round.** For each stage the acceptance check from the matrix is written as a failing test first, then the code is changed until it passes. Test IDs below map 1:1 to the §4 rows (E-01…E-12) and §-numbered criteria.
- **Correctness gates first.** The panic (E-07) is the top gate per §4 and is sequenced immediately after the harness. Sample-count (§1) follows because it is small and unblocks honest "5,000 samples" assertions everywhere else.
- **Engine changes before the UI that depends on them.** Honest, token-naming error messages (E-01/E-02/E-05) and the computed bucket `step` (§6) are engine work; the front-end validation/markers and chart stages consume them.

## Toolchain baseline (verified 2026-06-08)

Present and working: `cargo 1.96`, `wasm-pack 0.15`, `node v22.22`, `pnpm 11.5`, `wasm32-unknown-unknown` target, Chromium/chromedriver per toolchain doc. `web/dist/` and `web/src/wasm/` are already built. **Missing:** `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`; no `web/src/**/*.test.*` files; no `crates/wasm/tests/`.

---

## Stage 0 — Feedback loop bootstrap - DONE

**Goal:** every test runner is runnable and green-on-trivial, so all later work is test-first.

- Add dev deps to `web`: `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`.
- Configure Vitest in `web/vite.config.ts` (`test.environment = 'jsdom'`, a `setupTests.ts` importing `@testing-library/jest-dom`). Keep `"test": "vitest run"`.
- Write one smoke component test: `App` renders the prefilled model name and a Run button (no assertions on simulation yet).
- Stand up the WASM boundary harness: one passing round-trip test of `simulate` over the built glue.
    - Decision: `wasm-pack test --headless --chrome` (true browser, matches `getrandom` js backend; acceptable even if it is slower than other alternatives).
- Document the three commands in `DOCS/architecture/toolchain.md` if they drift from what's written.

**Milestone / deliverable:** `cargo test -p ninety_ci_core`, `pnpm -C web test`, and the boundary harness all run green. The inner loop exists.

**Tests:** harness smoke only (`App` renders; one boundary round-trip). **DoD:** unblocks §8.

---

## Stage 1 — Kill the panic (E-07, top correctness gate)

**Goal:** no input can make the engine trap; non-finite output becomes a clean `Err`.

- `crates/core`: in `evaluate`/`compute_histogram`, reject or skip non-finite sample outputs; if the resulting series is empty or non-finite, return `Err("The equation produced an undefined result (e.g. division by zero) — check the formula/bounds.")` rather than indexing an empty `counts` vec (`lib.rs:199`).
- Confirm the boundary maps this `Err` to a readable JS exception (it already does via `e.to_string()`); assert it is never `"unreachable"`.

**Milestone / deliverable:** an `X / 0` model returns a human-readable error end-to-end with no trap.

**Tests (write first, must fail):**
- Rust unit: `simulate` on a divide-by-zero / NaN / ∞ model returns `Err`, does **not** panic. (§8 named gap.)
- Boundary: the thrown JS value is a readable string, never `"unreachable"` (E-07/E-10).

**DoD:** ☑ E-07 cannot panic.

---

## Stage 2 — Fix sample count at 5,000 (§1)

**Goal:** honor the PRD; remove the adjustable selector; hard-wire 5,000.

- Drop `SAMPLE_OPTIONS` (`model.ts:27`) and the `<select>` (`ModelEditor.tsx:127-133`).
- Set the initial model and the hook call to pass `5_000` (`App.tsx:11`, `useNinetyCi.ts:60`). Consider removing `samples` from `Model` (keep it on `SimResult`, echoed back by the engine).
- Card header reads "5,000 samples".

**Milestone / deliverable:** the built UI exposes no sample control; every Run reports "5,000 samples".

**Tests (write first):**
- Component: no sample-count control is present in `ModelEditor`.
- Hook/integration: a Run reports `samples === 5000`.

**DoD:** ☑ §1 sample count fixed at 5,000.

---

## Stage 3 — Engine-side validation & honest messages (E-01, E-02, E-03, E-05, E-11, E-12) - DONE

**Goal:** every engine-sourced error names the offending field and the fix; no silent merges.

- **E-01:** when equation names exceed declared vars, return an `Err` naming the undeclared token(s): "`C` is used in the equation but not defined — add a variable row for it (or remove it)." (replaces terse `"Variables missing"`, `lib.rs:317`).
- **E-02:** keep the named message (`lib.rs:135`) but align copy to "defined but not used — use it or remove the row."
- **E-03:** name the row and which bound is inverted: "`X`: 5th (…) must be below 95th (…)." (replaces `"Bad distribution parameters"`).
- **E-05:** split the conflated message — empty equation → "Enter an equation."; zero variables → "Add at least one variable." (today both collapse to `"No variables for the equation"`).
- **E-12 (real gap):** detect duplicate variable names before the `HashMap` silently merges them (`lib.rs:60,137`); return "Two variables are named `X` — names must be unique." Never silently drop a row.
- **E-11:** keep the existing clean shape-guard message (defense-in-depth).
- Note the **validation ordering** (§4): v1 reports one violation at a time as the engine does today; document this in the boundary tests' expectations.

**Milestone / deliverable:** the engine returns precise, field-naming errors for every authored-model failure; duplicate names can no longer silently merge.

**Tests (write first, one per row):** Rust unit + boundary tests asserting each message names the offending token/row and (for E-01/E-02) offers the corrective action; E-12 asserts a blocked run, not a merge.

**DoD:** contributes to ☑ "every §4 error row has the specified behavior" (engine half) and ☑ no silent coercions (E-12).

---

## Stage 4 — Front-end validation & error UX (§3; E-04, E-09, E-10)

**Goal:** the user sees precise, non-silent errors in the UI; no stale chart beside an error; no blank screen on init failure.

- **E-04:** treat an empty/blank bound as invalid, not `0` — stop the silent `Number('')===0` coercion (`ModelEditor.tsx:100,108`); mark the cell and block Run with "`X`: enter a number for the 5th/95th bound." The engine is **not** called with a coerced 0.
- **E-10:** on error, clear the previous `result` so the banner is not shown beside a stale chart (`App.tsx:26` currently keeps it).
- **E-09:** if `init()` rejects, surface it through `useNinetyCi` (today only a `run`-time failure surfaces); show a persistent "Couldn't load the calculator engine — reload the page." and disable Run. No blank/half-rendered screen.
- **§3 surfacing decision:** Run stays always-on; show the engine's field-naming message in the summary banner (`ResultPanel.tsx:17`, add `role="alert"`).

**Milestone / deliverable:** every §3 rule produces a precise, non-silent UI response; errors never coexist with a stale chart; engine-load failure degrades gracefully.

**Tests (write first):** component/hook tests for each §3 rule (block/allow + message names the field), E-04 (no engine call with 0), E-10 (stale chart cleared after a prior success), E-09 (init-reject → fallback + Run disabled, via a stubbed `init`).

**DoD:** ☑ all §3 rules implemented and tested; completes ☑ §4 rows (front-end half: E-04, E-09, E-10).

---

## Stage 5 — The graph (§6)

**Goal:** the hero chart is correct across magnitudes and degenerate inputs.

- **Computed bucket `step` (replaces `DEFAULT_STEP=1`, `useNinetyCi.ts:16`):** `core` derives `step` from output magnitude targeting ~50–100 buckets across the observed range, instead of a front-end absolute constant. This also protects the §7 perf bar (no million-bucket blowups).
- **Degenerate / zero-width (E-03b, `buckets.length < 2`):** render a visible single-point/spike with its value instead of an empty `<svg/>` (`OutputChart.tsx:12-14`); hero width = 0.
- **Marker labels:** add numeric labels at the `ciLow`/`ciHigh` dashed markers.
- **Very narrow spread:** ensure both markers stay visible (min separation / combined marker).
- **Negative values:** confirm axis labels render negatives; keep 3 x-ticks (lo/mid/hi), axis-less Y, no gridlines.

**Milestone / deliverable:** a normal-spread model renders area + outline + shaded band + two labeled dashed markers; a zero-width model renders a visible point; bucket count stays ~50–100 for both a ~10 and a ~200,000 magnitude model.

**Tests (write first):** core test for bucket-count sanity across magnitudes after the `step` change; `OutputChart` degenerate render (visible point, not blank); marker position matches hero `ciLow`/`ciHigh`.

**DoD:** ☑ §6 graph spec met, `step` placeholder replaced, degenerate + empty states.

---

## Stage 6 — Distribution semantics & accessibility (§5; §7 a11y)

**Goal:** honest labels and the a11y minimum bar.

- **§5 tooltip (exact copy):** "For **normal**, these are the ~5th/95th percentiles. For **uniform** and **range**, they are the full minimum and maximum — the middle 90% falls inside them." Reachable by hover/focus on each variable row; column headers stay "5th"/"95th"; no engine change.
- **§7 a11y minimums (the four gates):** `aria-label`s on number cells and the formula bar; Run keyboard-operable; `role="alert"` on the error banner (from Stage 4); text/contrast meets WCAG AA against the design tokens.

**Milestone / deliverable:** tooltip present on `normal` and `uniform` rows with no percentile claim for uniform/range; all inputs have accessible names; Run is keyboard-operable.

**Tests (write first):** tooltip text present/reachable on a `normal` and a `uniform` row; copy review asserts no percentile claim for uniform/range; component tests assert accessible names exist and Run is focusable/activatable by keyboard.

**DoD:** ☑ §5 tooltip in place; ☑ §7 a11y minimums.

---

## Stage 7 — Non-functional gates & release (§7 perf/build; §9)

**Goal:** the final gates — performance, clean static build, end-to-end release smoke.

- **Perf:** measure Run-to-render on the default model; assert < 100 ms at 5,000 samples (the Stage 5 `step` fix must not regress this).
- **Build:** `pnpm -C web build` is clean (no TS errors) to a fully static `web/dist/`.
- **Release smoke (§9, decision #2):** the **built** bundle, served statically, runs a simulation end-to-end via Playwright against `vite preview`.

**Milestone / deliverable:** full §9 DoD checklist green; the shipped bundle runs a real simulation.

**Tests (write first):** a perf assertion on the default-model Run; CI build step proves no TS errors; release-smoke spec drives one Run against the built bundle.

**DoD:** ☑ §7 non-functional bars; ☑ §8 test commitment satisfied (`cargo test` + `pnpm -C web test` green, jsdom/testing-library added); ☑ §9 release smoke.

---

## Definition-of-Done coverage map (§9)

| DoD box | Stage |
|---|---|
| Sample count fixed at 5,000 | 2 |
| E-07 cannot panic | 1 |
| All §3 validation rules | 3 (engine) + 4 (UI) |
| Every §4 error row behavior; no silent coercions | 1, 3, 4 |
| §5 distribution-label tooltip | 6 |
| §6 graph spec (step, degenerate, empty) | 5 |
| §7 non-functional bars (perf, a11y, clean build) | 6 (a11y) + 7 (perf/build) |
| §8 test commitment; jsdom/testing-library added | 0 (harness) + every stage |
| §9 release smoke | 7 |
