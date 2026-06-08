# v1 acceptance — error behavior, edge cases & acceptance criteria

## 1. Scope check (carried from PRD)

Restate, in one line each, what v1 must do — so acceptance is measured against intent, not vibes.

**In scope:**
- Build a model: editable name, one equation, and N random variables (uniform / normal / range) each with a 5th/95th bound pair.
- Run the Monte-Carlo simulation client-side in WASM (no backend, no network) and render the output histogram with the 90% CI band shaded and dashed lower/upper markers.
- Read the 90% CI hero: the `lower – upper` range in large type plus its width.

**Out of scope (do NOT let creep in):**
- Adjustable sample count — fixed at 5,000 (see assumption below; **the current code violates this**).
- The mock's "Tweaks" panel (a design-tool artifact, not a product feature).
- Persistence, saving, sharing; the "Shared" / "Comments" / "Run all" top-bar actions.
- Multiple models / a model library.
- Auth, accounts, any server-side anything.

**Fixed assumptions:**
- **Sample count = 5,000.** This is `core`'s built-in default (`lib.rs:84`), but v1 wires it from the front-end, which currently passes `model.samples`. Not user-adjustable in v1.
- **Bucket step is engine-side, not user-facing.** Currently a `DEFAULT_STEP = 1` placeholder in `useNinetyCi.ts:16` — an unresolved correctness/perf issue (large outputs → millions of buckets), tracked in §6, not a user control.
- **"5th / 95th percentile" bounds are interpreted per-shape by the engine** and the label is honest only for `normal`; for `uniform`/`range` they are full min/max (see §5).
- **Simulation runs synchronously on the main thread** — no Web Worker in v1 (acceptable at 5,000 samples).

> **⚠️ Scope conflict to resolve (implementation vs. PRD).** The PRD fixes the sample count at 5,000 and lists "adjustable sample count" as out of scope, but the implemented front-end ships an adjustable selector: `SAMPLE_OPTIONS = [1_000, 10_000, 100_000]` (`model.ts:27`), a `<select>` in `ModelEditor.tsx:127`, and `samples: 10_000` as the initial model (`App.tsx:11`) — and **5,000 is not even one of the options**. v1 cannot be both. This must be decided before §2 use-cases are written (it changes what the UI exposes). See decision below.

- **DECISION (resolved 2026-06-07):** **Honor the PRD — fix at 5,000.** Remove the sample-count selector and hard-wire 5,000. Concretely: drop `SAMPLE_OPTIONS` (`model.ts:27`) and the `<select>` (`ModelEditor.tsx:127-133`), set the model/hook to pass `5_000` (initial model `App.tsx:11`), and consider dropping `samples` from `Model` (keep it only on `SimResult`, echoed back by the engine). Acceptance: the built UI exposes no sample control and every Run reports "5,000 samples".

---

## 2. Use-cases (thin — a handful, not a suite)

A few concrete user journeys, used here as the vehicle to surface edge cases and acceptance criteria. Aim for 3–5.

### UC-1: First-time user runs the prefilled example
- **Actor / context:** A non-technical user who wants a 90% confidence interval for an uncertain cost. They land on the app cold and see the prefilled "Exchange exposure" model (`App.tsx:8` — `200 * EXCHANGE_RATE + BASE_FEE`, a `uniform` and a `normal` variable). They do not need to author anything to get a first result.
- **Precondition:** Module loaded, model prefilled, no result yet — the result panel shows the empty state "Press **Run** to simulate the model." (`ResultPanel.tsx:18`).
- **Main flow:**
  1. User reads the prefilled equation and the two variable rows (name, distribution pill, 5th/95th bounds, shape sparkline).
  2. (Optional) User edits a bound — e.g. raises `EXCHANGE_RATE` 95th from 1200 to 1300.
  3. User clicks **Run**.
  4. The engine simulates 5,000 samples client-side in WASM and returns `{ciLow, ciHigh, buckets, counts, samples}`.
  5. The chart and the CI hero render.
- **Expected result:** The output chart draws the histogram area with the 90% band shaded and dashed lower/upper markers (`OutputChart.tsx`); the CI hero shows `ciLow – ciHigh` in large type with its width and the caption "the middle 90% of outcomes land in this range." (`CIHero.tsx`); the card header reads "5,000 samples" (per §1 decision).
- **Surfaced edge cases:** empty pre-Run state (§6); "5th/95th" label is loose for the `uniform` variable (§5); large-number rounding/formatting in axes and hero (`fmt` = `Math.round` + `en-US`, §6); a very narrow output range can make the chart/markers degenerate (§6); the unresolved bucket `step` (`DEFAULT_STEP=1`) affects bucket count and chart smoothness (§6).

### UC-2: User references a variable they never declared (error path)
- **Actor / context:** A user authoring their own model. They write an equation that references a name with no matching variable row — e.g. they type `A + B + C` but the table only declares `A` and `B` (or they deleted `C`'s row after writing the equation). All *declared* variables do appear in the equation.
- **Precondition:** Equation references an undeclared name; the table's declared rows are all used by the equation; a previous successful Run may or may not be on screen.
- **Main flow:**
  1. User edits the equation to reference the extra name `C`.
  2. User clicks **Run** (Run is always-on today — there is no pre-Run validation gating; `ModelEditor.tsx:56`).
  3. The engine adds `A` and `B` (both present in the equation), then finds fewer declared vars than equation names → returns `Partial` → `simulate` bails (`lib.rs:317`).
  4. The error crosses the WASM boundary as a thrown exception, caught by `useNinetyCi` into its `error` state (`useNinetyCi.ts:68`).
  5. The model does **not** run; no new chart is produced.
- **Expected result:** The app does not simulate. It shows an error that (a) names the offending variable(s) — `C` — and (b) tells the user how to correct it: declare `C` as a variable row (or remove it from the equation). **Gap vs. today:** the current engine message is the terse, non-specific `"Variables missing"` (`lib.rs:317`) which names nothing — the target behavior must name the undeclared token(s) and give the corrective action. See E-01 in §4.
- **Surfaced edge cases:**
  - E-01 terse message that names no variable (§4) — primary.
  - **Stale-chart-on-error:** `App.tsx:26` swallows the throw without clearing `result`, so after a prior successful Run the old chart stays visible *beside* the error banner (`ResultPanel.tsx:17,24`). Decide: clear the result on error, or keep it? (§4 E-10.)
  - Validation **ordering**: if the user *also* has a declared-but-unused row, the engine reports E-02 first and masks this E-01 (§4 ordering note).
  - The inverse — a declared variable absent from the equation (E-02) — is a sibling of this case and shares the "tell the user precisely what to fix" target.
---

## 3. Input validation rules

Define what makes a model *valid enough to Run*, and what the UI does about invalid state (disable Run? inline marker? both?).

> **Current state:** there is **no client-side validation at all**. Run is always-on (`ModelEditor.tsx:56` disables only while a run is in flight), and every rule below is enforced *only* by the engine, surfaced as a thrown exception after click. The table's **Rule** and **Enforced when** columns below are the v1 *target*; the rightmost column states the target UI behavior. Rows marked *(engine-only today)* have no front-end check yet.

| Field / object | Rule (target) | Enforced when | UI behavior on violation (target) |
|---|---|---|---|
| Model name | Non-empty after trim; soft max ~80 chars. Drives breadcrumb (`TopBar`). | On edit | Empty → fall back to "Untitled model" placeholder (already the `placeholder`, `ModelEditor.tsx:39`); never blocks Run. |
| Equation | Non-empty; every `[[:alpha:]]\w*` token is a declared variable; parseable by `meval`. | On edit (warn) + on Run (block) | Inline marker on the formula bar + a blocking message naming the bad token(s). Today: engine throws `"Variables missing"` (E-01) or an eval error (E-06). |
| Variable name | Non-empty; **unique**; appears in the equation; identifier charset `[A-Za-z]\w*`. | On edit (warn) + on Run (block) | Mark the offending row. Today: a name not in the equation throws E-02 (`"Variable <name> not mentioned…"`); duplicate names are **silently merged** by the engine's `HashMap` — *flag as a real gap, no check exists.* |
| `p5` / `p95` | Both numeric; **`p5 < p95`** for `uniform`/`range`; `p5 == p95` allowed only as a documented degenerate (§6, E-03b); `p5 > p95` invalid. | On edit (warn) + on Run (block) | Mark the two cells. Today: `p5 > p95` throws E-03 (`"Bad distribution parameters"`); empty cell is **silently coerced to 0** (`Number('')===0`, `ModelEditor.tsx:100,108`) — *flag: silent coercion, E-04.* |
| Distribution (shape) | One of `uniform` / `normal` / `range`. | Structurally guaranteed | The UI only offers the three via a `<select>` (`ModelEditor.tsx:88`), so an invalid shape is unreachable from the UI; the engine still guards it (E-11). |
| Variable count | **Min 1.** No hard max. | On Run (block) | Zero variables → block with a clear message. Today: engine bails `"No variables for the equation"` (E-05). |

- **DECISION (resolved 2026-06-07):** Keep **Run always-on** (validate-then-error on click), matching today's code, **but** add lightweight inline markers so the user sees problems before clicking. Rationale: gating Run behind full validity hides *why* it's disabled; an always-on Run plus a precise post-click message (and optional pre-click hints) is more teachable.
- **DECISION (resolved 2026-06-07):** Show errors **both** ways — a per-field/row inline marker for the specific offender *and* the existing summary banner (`ResultPanel.tsx:17`) for the human-readable explanation. v1 minimum if effort-bound: the summary banner alone, but it **must name the offending field** (today it does not).
- **Validation completeness (resolved 2026-06-07):** for v1 we collect violations one-at-a-time as the engine does today (see §4 ordering note). Eventually we will shift to collect-and-report in the front-end so the user fixes everything in one pass.

---

## 4. Error & edge-case matrix (the load-bearing section)

One row per failure mode. The **Observed** column records the *current* engine behavior as measured by a headless WASM smoke run on 2026-06-07 (`tmp/smoke.mjs`, `tmp/errs.mjs`) — these are facts about today's code, not the target. The **Target user-facing behavior** and **Acceptance check** columns are tomorrow's decisions: every row needs a user-facing behavior (not a console error / trap) and a testable check.

| ID | Condition | Observed now — current engine (smoke 2026-06-07) | Target user-facing behavior | Acceptance check |
|---|---|---|---|---|
| E-01 | Equation references a variable not in the table | Throws `"Variables missing"` (terse) — but only reached once all declared vars are present in the equation; otherwise E-02 fires first and masks it | Block Run; message **names the undeclared token(s)** and the fix: "`C` is used in the equation but not defined — add a variable row for it (or remove it)." Highlight the token in the formula bar. | Run a model with an equation name that has no row → no chart produced; error text contains the offending name and the word "add". (UC-2) |
| E-02 | Variable in table but absent from equation | Throws `"Variable <name> not mentioned in the equation"`. **This check runs FIRST**, before E-01/E-08 — see ordering note below | Block Run; message names the unused variable and the fix: "`X` is defined but not used in the equation — use it or remove the row." Mark the row. | Declare a variable absent from the equation → no chart; error names the variable and offers both fixes. |
| E-03 | `p5 >= p95` | Throws `"Bad distribution parameters"` | Block Run; message names the row and which bound is inverted: "`X`: 5th (…) must be below 95th (…)." Mark both cells. | Set p5 > p95 on a row → no chart; error names the variable; both cells flagged. |
| E-04 | Empty / non-numeric bound entered | **Partially resolved:** the `type=number` inputs coerce an empty field to `0` via `Number('')` (`ModelEditor.tsx:100,108`) — a **silent** coercion, not an error; truly non-numeric text can't be typed into a number input but would fail serde at the `crates/wasm` boundary if injected | Treat an empty bound as **invalid input**, not 0: mark the cell, block Run with "`X`: enter a number for the 5th/95th bound." Do not silently simulate with 0. | Clear a bound and Run → blocked with a per-cell message; the engine is **not** called with a coerced 0. |
| E-05 | Empty equation / constant equation on Run | Throws `"No variables for the equation"` (tested with 0 vars; the empty-equation-*with-vars* case not isolated — and the message talks about variables, not the empty equation, so it's misleading) | Block Run with an equation-specific message: empty equation → "Enter an equation."; no variables → "Add at least one variable." (split the two cases — today's single message conflates them). | Empty equation → blocked with "equation" copy; zero variables → blocked with "variable" copy; messages differ. |
| E-06 | Unparseable equation (syntax error, e.g. `X +* Y`) | **Not yet probed** — needs a dedicated probe. Engine path: `meval::eval_str_with_context` returns `Err` → `lib.rs:217` bails `"Error evaluating the equation: …"` | Block Run with "That equation can't be parsed — check the syntax." Optionally surface near the formula bar. Do not leak the raw `meval` debug string as primary copy. | Run `X +* Y` (with `X`,`Y` declared) → no chart; user-grade parse-error message (not a Rust `Debug` dump). **Probe needed to confirm exact engine output.** |
| E-07 | Division by zero / NaN / ∞ in simulation output | **⚠️ PANICS.** `X / 0` → ∞ output → bucket loop yields zero buckets → `crates/core/src/lib.rs:199` indexes an empty `counts` vec → Rust panic, surfaced to JS as an opaque `"unreachable"` trap. **Production crash path; highest priority.** Any equation evaluating to NaN/∞ for the sampled inputs hits this | **Must not panic.** `core` guards non-finite outputs: skip/reject NaN/∞ samples and, if the series is empty or non-finite, return a clean `Err` ("The equation produced an undefined result (e.g. division by zero) — check the formula/bounds."). Front-end shows it as a normal error. | A `X / 0` model Runs **without a trap** — `useNinetyCi.error` holds a human-readable string, never `"unreachable"`. Add a Rust unit test asserting `simulate` returns `Err` (not panic) for a non-finite output. **Engine fix required — this is the top correctness gate.** |
| E-08 | Math function in equation (`sin`, `exp`, `cos`) — **known engine limitation** | Throws `"Variables missing"` (the fn name is extracted as an undeclared variable, so it fails loudly). Only silently-wrong if a user names a *real* variable `sin`/`exp`/etc. — narrower risk than feared, but real | v1 scope is plain variable identifiers; don't add function support. But since `meval` *does* support these, a user typing `sin(X)` gets the misleading "Variables missing". Resolution: document "variables only, no functions" near the formula bar | The limitation is documented in-UI. |
| E-09 | WASM module fails to load / init | **Not probed** (module loads fine under Node with bytes; browser load-failure path untested) | No blank screen: if `init()` rejects, show a persistent "Couldn't load the calculator engine — reload the page." and disable Run. `useNinetyCi` must surface the init rejection (today `ensureReady` failure would only reject inside `run`). | Force `init()` to reject (test stub) → app shows the fallback message and Run is disabled; no blank/half-rendered screen. |
| E-10 | Simulation throws / panics (generic) | Errors surface to JS as thrown exceptions caught by `useNinetyCi`'s `error` state; **but panics (E-07) cross the boundary as `"unreachable"`**, which is useless as user copy | All engine errors reach the user as human-readable copy via `useNinetyCi.error`. On error, clear the previous chart so the banner isn't shown beside a stale result (`App.tsx:26` currently keeps it). | After a successful Run, trigger an error Run → the error banner shows and the **stale chart is cleared** (per decision); no `"unreachable"` ever reaches the user (depends on E-07 fix). |
| E-11 | Unknown / invalid distribution shape | Throws `"Unsupported distribution. Use either 'normal', 'range' or 'uniform'."` (clean, user-grade message) — overlaps §3 shape validation | Unreachable from the UI (shape is a 3-option `<select>`); keep the engine guard as defense-in-depth. No special UI work. | The `<select>` offers exactly uniform/normal/range; the engine message is retained but not normally reachable. |
| E-12 | Duplicate variable names | **Not in original matrix — found during review.** Two rows with the same `name` are silently merged by the engine's `vars: HashMap<&str, …>` (`lib.rs:60,137`); the second row's distribution **silently wins**, the first is lost with no error | Block Run (or de-dupe with a visible warning): "Two variables are named `X` — names must be unique." Mark both rows. | Two rows named `X` → blocked with a uniqueness message; never a silent merge. |

- **Validation ordering (observed):** the engine checks **declared-but-unused vars (E-02) first**. So a model with both an unused declared var *and* an undeclared name in the equation reports only E-02; the user fixes that, re-runs, then hits E-01. Decide whether v1 validates all conditions up-front (collect-and-report) or one-at-a-time as today.
- **Message quality (observed):** several messages are terse or opaque — `"Variables missing"`, `"No variables for the equation"`, `"Bad distribution parameters"`, and especially `"unreachable"` (the panic). None are user-facing-grade as-is; the Target column must specify the real copy.
- **Principle to confirm:** no failure path is silent — and **no failure path may panic/trap** (E-07 violates this today). Every swallowed error or skipped code path must be called out here with its user-visible consequence.

---

## 5. Distribution semantics decision

The PRD flags that the "5th / 95th percentile" label is accurate for **normal** but loose/misleading for **uniform** and **range** (there `lower`/`upper` are full min/max, so the middle 90% of `[1000,1200]` is `[1010,1190]`, not `[1000,1200]`).

- **DECISION:** **Keep current engine semantics + label/tooltip honestly.** This matches the PRD ("for v1 we keep the current engine semantics and allow the labels/tooltips to be loose"). No engine change; no per-shape relabeling of the columns.
- **Rationale:** Correctness-vs-effort trade for v1. For `normal`, `lower`/`upper` genuinely are ~5th/95th (`sd = (upper-lower)/3.29`, `lib.rs:31`). For `uniform`/`range` they are the **full min/max**, so the middle 90% is narrower than `[lower, upper]` (the middle 90% of uniform `[1000,1200]` is `[1010,1190]`). Changing the engine or per-shape labels is out of scope; an honest tooltip closes the trust gap cheaply. Revisit in a later version if users misread it.
- **Resulting copy / tooltip text:**
  - Column headers stay "5th" / "95th".
  - Existing hint stays (`ModelEditor.tsx:135`): "5th / 95th are the percentile bounds for each variable."
  - **Add a tooltip on the distribution pill / bounds** (exact string): "For **normal**, these are the ~5th/95th percentiles. For **uniform** and **range**, they are the full minimum and maximum — the middle 90% falls inside them."
- **Acceptance check:** The tooltip text above is present and reachable (hover/focus) on each variable row; a `normal` and a `uniform` variable both show it; copy review confirms no claim that `uniform`/`range` bounds are percentiles.

---

## 6. The graph (the hero — pin it)

The output chart is the product's centerpiece. Current implementation: `OutputChart.tsx`, fixed `520×188` viewBox, `preserveAspectRatio="none"`.

- **Bucketing:** Buckets come from the **engine**, not the chart — `(buckets, counts)` where `buckets` are lower bounds spaced by `step`. 
  - **Replace fixed step with computed one:** `step` is a front-end constant `DEFAULT_STEP = 1` (`useNinetyCi.ts:16`), an absolute width — so bucket count ≈ output-range / 1. For large-magnitude outputs this is too coarse (few buckets, blocky chart) or, with a small step, explosively many buckets. **v1 must replace the placeholder** with a step derived from output magnitude (e.g. target ~50–100 buckets across the observed range), computed by `core`.
- **Axes & labels:** X is the output value; three ticks — `lo`, midpoint, `hi` — formatted `Math.round` + `en-US` thousands (`OutputChart.tsx:5,48-50`). Y is sample frequency, **unlabeled and unscaled** (normalized to chart height via `maxCount`). v1 minimum: keep the 3 x-ticks; confirm Y is axis-less. No gridlines in v1.
- **CI band & markers:** Full histogram drawn as a filled area (`#eef2f6`) with a blue outline (`var(--blue)`, width 2). The `[ciLow, ciHigh]` band is re-filled blue at `fillOpacity 0.18`, and two dashed vertical markers (`strokeDasharray="3 3"`) sit at `ciLow`/`ciHigh` (`OutputChart.tsx:41-45`). Add numeric labels at the two markers to show the edge values of the band.
- **Degenerate states:**
  - **Zero-width / `buckets.length < 2`** (E-03b, all-identical samples): currently renders an **empty `<svg/>`** (`OutputChart.tsx:12-14`) — a blank box. **Must change:** show a visible single spike / point marker with the value, so a fixed-value model still reads as a result, not a bug.
  - **Very narrow spread:** markers `x5`/`x95` can collapse onto each other; ensure both remain visible (min 1px separation or a combined marker).
  - **Very wide spread / few buckets:** tied to the `step` decision above; avoid a blocky 2–3 bar chart.
  - **Negative values:** supported by the engine (`compute_histogram` handles negative buckets) and by `x()` mapping; confirm axis labels render negative numbers correctly.
- **Empty / pre-Run state:** The chart area is not shown before the first Run; `ResultPanel.tsx:18` shows "Press **Run** to simulate the model." On error with no prior result, the empty state is replaced by the error banner. (See E-10 stale-chart decision for the post-error case.)
- **Acceptance checks:**
  - A normal-spread model renders area + blue outline + shaded band + two dashed markers; x-axis shows lo/mid/hi rounded with thousands separators.
  - A zero-width model (E-03b) renders a **visible** single-point state, not a blank `<svg/>`; hero width = 0.
  - Bucket count stays in a sane range (~50–100) across small- and large-magnitude outputs once the `step` placeholder is replaced — test with a ~10 and a ~200,000 magnitude model.
  - CI markers' on-chart position matches the hero's `ciLow`/`ciHigh` values.

---

## 7. Non-functional acceptance

- **Performance:** Run-to-render budget **< 100 ms** at 5,000 samples (the architecture note says 5k–100k evaluate in milliseconds). Synchronous main-thread compute is **acceptable** at 5,000 samples — no Web Worker in v1 (§1). The `step` fix (§6) must not blow up bucket count and regress this. Acceptance: a Run on the default model returns and paints in < 100 ms on a mid-range laptop.
- **Browser/device support:** Evergreen desktop browsers with WASM + Web Crypto (`thread_rng` uses the `js` getrandom backend) + `crypto.randomUUID` (used in `ModelEditor.tsx:21`). Target: current Chrome, Firefox, Safari, Edge. No IE; mobile is best-effort (layout is a two-column desktop design).
- **Accessibility:** Minimum bar — every input has a label or accessible name (the number cells and the formula bar currently rely on placeholders/visuals; add `aria-label`s); Run reachable and operable by keyboard; the error banner announced (e.g. `role="alert"` on `ResultPanel.tsx:17`); text/contrast meets WCAG AA against the design tokens. Not a full audit; these four are the gate.
- **Build/deploy:** `pnpm -C web build` runs `build:wasm` then `vite build` to a fully static `web/dist/` with no backend; deploy to any static host (GitHub Pages / Netlify). Acceptance: clean build (no TS errors), and the built bundle Runs a simulation when served statically (release smoke, §9).

---

## 8. Test coverage commitment

- **Unit (engine/core):** **Already covered** (`crates/core`): `extract_variable_names`, `sample_variable` (type/bounds/size), `add_variable(s)` partial/full, `compute_histogram` (sizes, negatives, sub-unit), `ninety_ci` (`lib.rs` `#[cfg(test)]`), plus 2 integration tests (`tests/simple.rs`: single normal, single uniform). **Gaps to add:** the **E-07 non-finite/divide-by-zero** case must assert `simulate` returns `Err`, **not panic** (top gate); the E-03b `p5 == p95` zero-width path; an end-to-end `simulate()` assertion that `buckets.len() == counts.len()` and `samples == 5000`.
- **Boundary (wasm marshalling):** Currently **none**. Add at least one a `wasm-pack test` (or a Node harness over the built glue) covering: a valid `simulate` round-trip (vars `[{name,shape,lower,upper}]` → `SimOutput`), and another one that a thrown engine error crosses as a JS exception with a readable message (E-10). Probe E-04 non-numeric at the serde boundary.
- **Front-end component / hook:** Currently **none**. Minimum v1: tests for every §3 validation rule (block/allow + message names the field) and every §4 row's target behavior — especially E-01/E-02 (named variable), E-04 (no silent-0), E-07 (no `"unreachable"` reaches `error`), E-10 stale-chart clearing. Plus `OutputChart` degenerate render (E-03b → visible point, not blank `<svg/>`) and `CIHero` width formatting. **Tooling gap:** `vitest` + `@vitest/coverage-v8` are installed, but **`jsdom` and `@testing-library/react` are not** — add them for component/hook tests.
- **Integration smoke (one full Run path):** One test driving the default model through `useNinetyCi.run` (real WASM) → asserts a `SimResult` with non-empty `buckets`/`counts` and `samples === 5000`. Also a release smoke against the **built** `web/dist/` bundle (§9).
- **Decided runner:** Test runner for `web/` is **Vitest** (`web/package.json` `"test": "vitest run"`, already wired) — consistent with the toolchain doc. Rust uses `cargo test -p ninety_ci_core`; boundary uses `wasm-pack test`.

---

## 9. Definition of Done for v1

A short, checkable list. v1 ships only when every box is true.

- [ ] **§1 sample count fixed at 5,000** — selector removed (`SAMPLE_OPTIONS`, the `<select>`), `5_000` hard-wired; UI exposes no sample control and Runs report "5,000 samples"
- [ ] **E-07 cannot panic** — `simulate` returns a clean `Err` for non-finite/÷0 output; no `"unreachable"` can reach the user (top correctness gate)
- [ ] All §3 validation rules implemented and tested
- [ ] Every §4 error row has the specified user-facing behavior, verified (messages name the offending field; no silent coercions — E-04, E-12)
- [ ] §5 distribution-label decision implemented; tooltip copy in place
- [ ] §6 graph spec met, including the `step` placeholder replaced and degenerate (E-03b) + empty states
- [ ] §7 non-functional bars met (< 100 ms Run-to-render; a11y minimums; clean static build)
- [ ] §8 test commitment satisfied; `cargo test` + `pnpm -C web test` green; jsdom/testing-library added
- [ ] Release smoke: the **built** `web/dist/` bundle, served statically, runs a simulation end-to-end
