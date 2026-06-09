# Candidate rules

## From stage 3 (engine-side validation & honest messages)

- ALWAYS `git status --porcelain` before committing an agent's implementation pass and reject unexpected files BECAUSE a dev agent committed a stray 0-byte `lib.rs.probe` sibling that survived into the first-pass commit and was only caught at review.
- ALWAYS make a new input-validation guard use the strictest predicate that still admits existing degenerate-but-valid inputs (E-03 used `lower > upper`, not `>=`) BECAUSE the E-07 non-finite tests depend on `range(0,0)` being valid, and a `>=` guard would have silently broken them.
- NEVER let a test assert only the negative ("the error is NOT message X") without also asserting the positive outcome (`is_ok()` or the exact expected message) BECAUSE a negative-only assertion passes even when the code returns a different, wrong error — it gives false confidence (caught on `simulate_range_equal_bounds_is_valid`).
- ALWAYS place a duplicate/uniqueness scan as a read-only pass BEFORE the mutating insert loop, not interleaved with it BECAUSE detecting the collision only after a `HashMap::insert` has already overwritten the first entry is a silent merge (E-12), which unit tests cannot catch.
- ALWAYS add a boundary (`wasm-bindgen-test`) case for every engine error variant the stage introduces, not only unit tests BECAUSE the `e.to_string()` marshalling across the WASM boundary is a separate code path, and an error that is unit-tested can still be malformed or swallowed when it crosses to JS (E-11 was nearly shipped with a unit test but no boundary test).
