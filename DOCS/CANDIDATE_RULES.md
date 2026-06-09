# Candidate rules

Promotion candidates harvested from stage work. When a rule has proven its worth, move it into `.claude/CLAUDE.md`.

## From Stage 5 (the graph, §6)

### Rust / numerical
C1. ALWAYS guard a float-accumulator loop (`while x <= hi { x += step }`) with a no-progress check (`let next = x + step; if next == x { break; }`) BECAUSE when `step` is smaller than `ULP(x)` the addition is a no-op and the loop never terminates — a hang is worse than a panic: it is uncatchable across the WASM boundary and surfaces no error to the user.
C2. ALWAYS re-audit a downstream consumer's numerical assumptions when you replace a hardcoded constant with a value derived from observed data (e.g. bucket `step` from output magnitude) BECAUSE the dynamic value can reach degenerate extremes — subnormal, zero, or huge — that the constant never exercised, so latent loops/index math that were "fine in practice" become reachable.
C3. ALWAYS clamp a data-derived divisor/step to a positive, finite minimum tied to the operand magnitude (`x.abs() * f64::EPSILON * k`) BECAUSE `range / n` can underflow to a step that cannot advance an accumulator at that magnitude.

### Front-end / UX
C4. NEVER collapse two distinct data values into a single on-screen element (e.g. overlapping CI markers into one label) without rendering both — show a range (`lo–hi`) when they differ BECAUSE the collapse silently discards one value, the exact silent-wrong class the project forbids; only merge the display when the values are actually equal.
C5. ALWAYS account for the SVG `<text>` baseline (not top edge) when placing labels near a viewBox edge BECAUSE `y` is the baseline, so a naive `y = top - 2` pushes the glyph's ascenders above `y=0` where default SVG overflow clips them.

### Reviewing / comments
C6. NEVER leave a comment asserting an invariant ("guaranteed non-empty here") that the adjacent fallback code (`reduce(..).unwrap_or(0.0)`) actually contradicts BECAUSE a false invariant comment trains readers to trust a guarantee the code does not provide, hiding the real edge case the fallback handles.
