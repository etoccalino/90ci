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

One row per failure mode. Every row needs a **user-facing behavior** (not a console error) and an **acceptance check**. Cover at least the cases below; add rows as use-cases surface more.

| ID | Condition | Trigger | User-facing behavior | Acceptance check |
|---|---|---|---|---|
| E-01 | Equation references a variable not in the table | <…> | <message? highlight?> | <…> |
| E-02 | Variable in table but absent from equation | <…> | <warn? ignore? block?> | <…> |
| E-03 | `p5 > p95` (or `p5 == p95`) | <…> | <…> | <…> |
| E-04 | Empty / non-numeric bound entered | <…> | <…> | <…> |
| E-05 | Empty equation on Run | <…> | <…> | <…> |
| E-06 | Unparseable equation (syntax error) | <…> | <…> | <…> |
| E-07 | Division by zero / NaN / ∞ in simulation output | <…> | <…> | <…> |
| E-08 | Math function in equation (`sin`, `exp`, `cos`) — **known engine limitation**: parsed as a variable, silently wrong | <…> | <DECISION: block-with-message, or document-and-allow?> | <…> |
| E-09 | WASM module fails to load / init | <…> | <fallback message; no blank screen> | <…> |
| E-10 | Simulation throws / panics | <…> | <surfaced via `useNinetyCi` error state> | <…> |
| E-NN | <add as found> | | | |

- **Principle to confirm:** no failure path is silent. Every swallowed error or skipped code path must be called out here with its user-visible consequence.

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
