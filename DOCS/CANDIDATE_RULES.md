# Candidate rules

Rules proposed from stage work, awaiting promotion into `.claude/CLAUDE.md`.

### Front-end related
F4. ALWAYS represent an empty numeric input as a distinct sentinel (e.g. `null`), never let `Number('') === 0` stand BECAUSE the coerced `0` is a silently-wrong *valid* value the engine will happily run on (Stage 4 E-04: a blank percentile bound silently became `0`).

F5. ALWAYS give an inline affordance that signals one specific failure class (e.g. a "blank bound" cell marker) its OWN state channel, separate from the generic error banner BECAUSE threading the shared error string into the marker makes it fire under unrelated failures (Stage 4 SF1: an engine/init load error eagerly marked blank cells with no Run attempted).
