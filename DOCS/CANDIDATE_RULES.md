# Candidate rules

Learnings staged for promotion into `.claude/CLAUDE.md`. Phrased in the house `ALWAYS/NEVER … BECAUSE …` style.

## Rust

- ALWAYS make a Monte-Carlo engine's RNG injectable (`&mut dyn RngCore` threaded through the sampling path, public entrypoints delegating with `thread_rng()`) so tests can drive it from a seeded `StdRng`, rather than asserting a fixed tolerance against an unseeded `thread_rng` BECAUSE only a fixed seed makes the recovered CI reproducible — any tolerance against live entropy merely lowers, never eliminates, the flake probability (observed: `integration_single_variable_normal` flaked ~10% at tolerance 6).
- ALWAYS verify whether a Monte-Carlo estimator's deviation from the theoretical value is a *deterministic bias* (systematic across seeds) before sizing a test tolerance to "sampling variance" BECAUSE the histogram CI estimator here sits a deterministic ~3–7 below the true 5th percentile, so a tolerance derived from variance alone (≈1.9) is wrong and the test flakes; pin the numeric check with a seed and keep the unseeded integration test to seed-independent invariants only.
