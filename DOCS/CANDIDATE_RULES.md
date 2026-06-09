# Candidate rules

- ALWAYS re-audit downstream invariants when you start dropping/filtering elements from a series BECAUSE a guard that silently shrinks a collection (Stage 1 dropped non-finite samples) can invalidate a count assumed elsewhere — `ninety_ci` divided by configured `resolution` while the surviving sample count was now smaller, biasing the CI with no error.
- ALWAYS treat trading a panic for a silently-wrong result as a regression, not a fix BECAUSE "kill the panic" (E-07) must produce an honest `Err` or a correct value, never an `Ok` carrying a biased number that looks plausible.
- ALWAYS prefer `f64::total_cmp` over `partial_cmp(..).unwrap()` when sorting floats BECAUSE `total_cmp` is a total order defined for NaN too, removing an unwrap-panic that is uncatchable across the wasm boundary.
- ALWAYS clamp histogram bucket indices derived from float arithmetic BECAUSE `div_euclid`-based indexing and the bucket-building `+= step` loop accumulate rounding differently, so a value at the range extreme can compute one index past the end and panic on `counts[i]`.
