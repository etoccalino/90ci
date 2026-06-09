## Rules

### General
1. NEVER introduce unnecessary line-breaks mid-line when writing documentation BECAUSE it breaks the reading flow when user uses a paginator.
2. ALWAYS ignore `TODO.md` at the top level of the repo BECAUSE the file contains actions solely for the user.
3. ALWAYS run the test of component being worked on before the work is considered "done" BECAUSE the tests provide early feedback and allow to catch errors early.
4. ALWAYS fetch web page content with `curl -sL <raw-url>` via Bash rather than `WebFetch` for verbatim raw content BECAUSE WebFetch's summarizer model refuses to reproduce "system-prompt-like" content verbatim and returns a refusal instead of the file.
5. ALWAYS re-audit downstream invariants when you start dropping/filtering elements from a series BECAUSE a guard that silently shrinks a collection can invalidate a count assumed elsewhere.

### Rust related rules
R1. ALWAYS treat trading a panic for a silently-wrong result as a regression, not a fix BECAUSE silent errors can't be found by unit tests.
R2. ALWAYS prefer `f64::total_cmp` over `partial_cmp(..).unwrap()` when sorting floats BECAUSE `total_cmp` is a total order defined for NaN too, removing an unwrap-panic that is uncatchable across the wasm boundary.
R3. ALWAYS clamp histogram bucket indices derived from float arithmetic BECAUSE `div_euclid`-based indexing and the bucket-building `+= step` loop accumulate rounding differently, so a value at the range extreme can compute one index past the end and panic on `counts[i]`.
