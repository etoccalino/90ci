# v1 acceptance — error behavior, edge cases & acceptance criteria

> **Status: TEMPLATE — to fill in.** Companion to [90ci-web-calculator.md](90ci-web-calculator.md). The PRD covers the happy path and visuals; this doc pins the edges that matter for a production ship to thousands of users. Replace every `<…>` and `TODO` before calling v1 done. Delete this banner when filled.

## How to use this doc

Each section is a decision or a checklist, not prose for its own sake. Fill the tables, resolve every `DECISION:` marker, and make sure every row has an observable, testable outcome. If a row can't be made testable, it's not done.

---

## 1. Scope check (carried from PRD)

Restate, in one line each, what v1 must do — so acceptance is measured against intent, not vibes.

- In scope: <…>
- Out of scope (do NOT let creep in): <…>
- Fixed assumptions (e.g. sample count = 5,000): <…>

---

## 2. Use-cases (thin — a handful, not a suite)

A few concrete user journeys, used here as the vehicle to surface edge cases and acceptance criteria. Aim for 3–5.

### UC-1: <short title, e.g. "First-time user models a single cost">
- **Actor / context:** <who, what they know, why they're here>
- **Precondition:** <state of the app before>
- **Main flow:** <numbered happy-path steps>
- **Expected result:** <what they read off; the CI + chart state>
- **Surfaced edge cases:** <list — feed these into §4>

### UC-2: <…>
<repeat the UC-1 shape>

### UC-3: <…>
<repeat>

---

## 3. Input validation rules

Define what makes a model *valid enough to Run*, and what the UI does about invalid state (disable Run? inline marker? both?).

| Field / object | Rule | Enforced when | UI behavior on violation |
|---|---|---|---|
| Model name | <e.g. non-empty? max length?> | <on edit / on Run> | <…> |
| Equation | <non-empty; references only declared vars; allowed tokens> | <…> | <…> |
| Variable name | <non-empty; unique; appears in equation; identifier charset> | <…> | <…> |
| `p5` / `p95` | <numeric; `p5 < p95`? `p5 == p95` allowed?> | <…> | <…> |
| Distribution (shape) | <one of uniform / normal / range> | <…> | <…> |
| Variable count | <min 1? max?> | <…> | <…> |

- **DECISION:** Is **Run** gated (disabled until valid) or always-on (validate-then-error on click)? <…>
- **DECISION:** Are validation errors shown inline per-field, in a summary, or both? <…>

---

## 4. Error & edge-case matrix (the load-bearing section)

One row per failure mode. The **Observed** column records the *current* engine behavior as measured by a headless WASM smoke run on 2026-06-07 (`tmp/smoke.mjs`, `tmp/errs.mjs`) — these are facts about today's code, not the target. The **Target user-facing behavior** and **Acceptance check** columns are tomorrow's decisions: every row needs a user-facing behavior (not a console error / trap) and a testable check.

| ID | Condition | Observed now — current engine (smoke 2026-06-07) | Target user-facing behavior | Acceptance check |
|---|---|---|---|---|
| E-01 | Equation references a variable not in the table | Throws `"Variables missing"` (terse) — but only reached once all declared vars are present in the equation; otherwise E-02 fires first and masks it | <message? highlight the token?> | <…> |
| E-02 | Variable in table but absent from equation | Throws `"Variable <name> not mentioned in the equation"`. **This check runs FIRST**, before E-01/E-08 — see ordering note below | <warn? ignore? block?> | <…> |
| E-03 | `p5 > p95` | Throws `"Bad distribution parameters"` | <…> | <…> |
| E-03b | `p5 == p95` (zero-width bound) | **No throw** — returns a zero-width CI `[v, v]` (degenerate single-point distribution). Chart must handle this — see §6 degenerate states | <allow? warn?> | <…> |
| E-04 | Empty / non-numeric bound entered | **Not yet probed** — non-numeric would fail at the serde marshalling boundary in `crates/wasm`; needs a dedicated probe | <…> | <…> |
| E-05 | Empty equation / constant equation on Run | Throws `"No variables for the equation"` (tested with 0 vars; the empty-equation-*with-vars* case not isolated — and the message talks about variables, not the empty equation, so it's misleading) | <…> | <…> |
| E-06 | Unparseable equation (syntax error, e.g. `X +* Y`) | **Not yet probed** — needs a dedicated probe | <…> | <…> |
| E-07 | Division by zero / NaN / ∞ in simulation output | **⚠️ PANICS.** `X / 0` → ∞ output → bucket loop yields zero buckets → `crates/core/src/lib.rs:199` indexes an empty `counts` vec → Rust panic, surfaced to JS as an opaque `"unreachable"` trap. **Production crash path; highest priority.** Any equation evaluating to NaN/∞ for the sampled inputs hits this | <…> | <…> |
| E-08 | Math function in equation (`sin`, `exp`, `cos`) — **known engine limitation** | Throws `"Variables missing"` (the fn name is extracted as an undeclared variable, so it fails loudly). Only silently-wrong if a user names a *real* variable `sin`/`exp`/etc. — narrower risk than feared, but real | <DECISION: block-with-message, or document-and-allow?> | <…> |
| E-09 | WASM module fails to load / init | **Not probed** (module loads fine under Node with bytes; browser load-failure path untested) | <fallback message; no blank screen> | <…> |
| E-10 | Simulation throws / panics (generic) | Errors surface to JS as thrown exceptions caught by `useNinetyCi`'s `error` state; **but panics (E-07) cross the boundary as `"unreachable"`**, which is useless as user copy | <surfaced via `useNinetyCi` error state, human-readable> | <…> |
| E-11 | Unknown / invalid distribution shape | Throws `"Unsupported distribution. Use either 'normal', 'range' or 'uniform'."` (clean, user-grade message) — overlaps §3 shape validation | <…> | <…> |
| E-NN | <add as found> | | | |

- **Validation ordering (observed):** the engine checks **declared-but-unused vars (E-02) first**. So a model with both an unused declared var *and* an undeclared name in the equation reports only E-02; the user fixes that, re-runs, then hits E-01. Decide whether v1 validates all conditions up-front (collect-and-report) or one-at-a-time as today.
- **Message quality (observed):** several messages are terse or opaque — `"Variables missing"`, `"No variables for the equation"`, `"Bad distribution parameters"`, and especially `"unreachable"` (the panic). None are user-facing-grade as-is; the Target column must specify the real copy.
- **Principle to confirm:** no failure path is silent — and **no failure path may panic/trap** (E-07 violates this today). Every swallowed error or skipped code path must be called out here with its user-visible consequence.

---

## 5. Distribution semantics decision

The PRD flags that the "5th / 95th percentile" label is accurate for **normal** but loose/misleading for **uniform** and **range** (there `lower`/`upper` are full min/max, so the middle 90% of `[1000,1200]` is `[1010,1190]`, not `[1000,1200]`).

- **DECISION:** For v1, do we (a) keep current engine semantics + relabel/tooltip honestly, (b) change per-shape labels, or (c) adjust the engine? <…>
- **Rationale:** <why — trust vs. effort vs. correctness>
- **Resulting copy / tooltip text:** <exact strings>
- **Acceptance check:** <…>

---

## 6. The graph (the hero — pin it)

The output chart is the product's centerpiece; specify beyond "an area chart".

- **Bucketing:** <count / width source; fixed or derived from data?>
- **Axes & labels:** <x units, y meaning, tick strategy, number formatting>
- **CI band & markers:** <shading, dashed markers, value labels — exact behavior>
- **Degenerate states:** <single point; all-identical samples; very narrow vs. very wide spread; negative values>
- **Empty / pre-Run state:** <what the chart area shows before the first Run>
- **Acceptance checks:** <…>

---

## 7. Non-functional acceptance

- **Performance:** <Run-to-render budget at 5,000 samples; main-thread blocking acceptable?>
- **Browser/device support:** <target matrix>
- **Accessibility:** <minimum bar — keyboard, contrast, labels>
- **Build/deploy:** <static bundle builds clean; deploy target>

---

## 8. Test coverage commitment

The front-end currently has no tests. State what v1 must have before ship.

- **Unit (engine/core):** <what's already covered + gaps>
- **Boundary (wasm marshalling):** <…>
- **Front-end component / hook:** <minimum — at least the validation rules §3 and error matrix §4>
- **Integration smoke (one full Run path):** <…>
- **DECISION:** test runner / framework for `web/` <…>

---

## 9. Definition of Done for v1

A short, checkable list. v1 ships only when every box is true.

- [ ] All §3 validation rules implemented and tested
- [ ] Every §4 error row has the specified user-facing behavior, verified
- [ ] §5 distribution-label decision implemented; copy in place
- [ ] §6 graph spec met, including degenerate + empty states
- [ ] §7 non-functional bars met
- [ ] §8 test commitment satisfied; suite green
- [ ] <add release gates: smoke on the built bundle, etc.>
