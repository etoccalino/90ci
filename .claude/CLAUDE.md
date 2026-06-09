## Rules

### General
1. NEVER introduce unnecessary line-breaks mid-line when writing documentation BECAUSE it breaks the reading flow when user uses a paginator.
2. ALWAYS ignore `TODO.md` at the top level of the repo BECAUSE the file contains actions solely for the user.
3. ALWAYS run the test of component being worked on before the work is considered "done" BECAUSE the tests provide early feedback and allow to catch errors early.
4. ALWAYS fetch web page content with `curl -sL <raw-url>` via Bash rather than `WebFetch` for verbatim raw content BECAUSE WebFetch's summarizer model refuses to reproduce "system-prompt-like" content verbatim and returns a refusal instead of the file.
5. ALWAYS re-audit downstream invariants when you start dropping/filtering elements from a series BECAUSE a guard that silently shrinks a collection can invalidate a count assumed elsewhere.
6. ALWAYS make a new input-validation guard use the strictest predicate that still admits existing degenerate-but-valid inputs (E-03 used `lower > upper`, not `>=`) BECAUSE it guards against unforeseen edge cases.
7. ALWAYS `git status --porcelain` before committing an agent's implementation pass and reject unexpected files BECAUSE agents and users may accidentally add changes that piggy-back on the commit.
8. NEVER let a test assert only the negative ("the error is NOT message X") without also asserting the positive outcome (`is_ok()` or the exact expected message) BECAUSE a negative-only assertion passes even when the code returns a different error.

### Rust related rules
R1. ALWAYS treat trading a panic for a silently-wrong result as a regression, not a fix BECAUSE silent errors can't be found by unit tests.
R2. ALWAYS prefer `f64::total_cmp` over `partial_cmp(..).unwrap()` when sorting floats BECAUSE `total_cmp` is a total order defined for NaN too, removing an unwrap-panic that is uncatchable across the wasm boundary.
R3. ALWAYS clamp histogram bucket indices derived from float arithmetic BECAUSE `div_euclid`-based indexing and the bucket-building `+= step` loop accumulate rounding differently, so a value at the range extreme can compute one index past the end and panic on `counts[i]`.
R4. ALWAYS place a duplicate/uniqueness scan as a read-only pass _before_ the mutating insert loop, not interleaved with it BECAUSE detecting the collision only after a `HashMap::insert` has already overwritten the first entry is a silent merge, which unit tests cannot catch.
R5. ALWAYS add a boundary (`wasm-bindgen-test`) case for every engine error variant the stage introduces, not only unit tests BECAUSE the `e.to_string()` marshalling across the WASM boundary is a separate code path, and an error that is unit-tested can still be malformed or swallowed when it crosses to JS.
R6. ALWAYS guard a float-accumulator loop (`while x <= hi { x += step }`) with a no-progress check (`let next = x + step; if next == x { break; }`) BECAUSE when `step` is smaller than `ULP(x)` the addition is a no-op and the loop never terminates.
R7. ALWAYS re-audit a downstream consumer's numerical assumptions when you replace a hardcoded constant with a value derived from observed data BECAUSE the dynamic value can reach degenerate extremes.
R8. ALWAYS clamp a data-derived divisor/step to a positive, finite minimum tied to the operand magnitude (`x.abs() * f64::EPSILON * k`) BECAUSE `range / n` can underflow to a step that cannot advance an accumulator at that magnitude.

### Front-end related rules
F1. ALWAYS keep Vitest `globals: true` set when using `@testing-library/react` BECAUSE its auto-cleanup accumulates prior renders in the jsdom DOM across a file, so any DOM-count assertion (e.g. "exactly N comboboxes") silently counts stale nodes.
F2. ALWAYS pin mock call arguments with `toHaveBeenCalledWith(...)` (using `expect.any(...)` for the others) rather than reading a positional index like `mock.calls[0][2]` BECAUSE the positional read still passes if the argument order changes, hiding the regression the test was written to catch.
F3. ALWAYS assert the complete expected element count rather than filtering by known magic values BECAUSE a value-list filter lets a re-introduced control with a novel value slip through undetected.
F4. ALWAYS represent an empty numeric input as a distinct sentinel (e.g. `null`), never let `Number('') === 0` stand BECAUSE the coerced `0` is a silently-wrong *valid* value the engine will happily run on.
F5. ALWAYS give an inline affordance that signals one specific failure class (e.g. a "blank bound" cell marker) its own state channel, separate from the generic error banner BECAUSE threading the shared error string into the marker makes it fire under unrelated failures.
F6. NEVER collapse two distinct data values into a single on-screen element (e.g. overlapping CI markers into one label) BECAUSE the collapse silently discards one value.
F7. ALWAYS account for the SVG `<text>` baseline (not top edge) when placing labels near a viewBox edge BECAUSE `y` is the baseline, so a naive `y = top - 2` pushes the glyph's ascenders above `y=0` where default SVG overflow clips them.
F8. ALWAYS assert tooltip/hint reachability through the accessibility tree (`toHaveAccessibleDescription`, `toHaveAccessibleName`), never via `getByText` on a CSS-hidden element BECAUSE jsdom applies no CSS, so the always-present node is found regardless of `:hover`/`:focus` visibility.