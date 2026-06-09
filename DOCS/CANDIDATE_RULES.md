# Candidate rules

C1. ALWAYS keep Vitest `globals: true` set when using `@testing-library/react` BECAUSE its auto-cleanup registers via a global `afterEach`; without it prior renders accumulate in the jsdom DOM across a file, so any DOM-count assertion (e.g. "exactly N comboboxes") silently counts stale nodes and stops being meaningful.

C2. ALWAYS pin mock call arguments with `toHaveBeenCalledWith(...)` (using `expect.any(...)` for the others) rather than reading a positional index like `mock.calls[0][2]` BECAUSE the positional read still passes if the argument order changes, hiding the regression the test was written to catch.

C3. ALWAYS assert the complete expected element count (e.g. "one distribution select per variable, nothing more") rather than filtering by known magic values BECAUSE a value-list filter lets a re-introduced control with a novel value slip through undetected.

