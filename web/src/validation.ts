/**
 * Client-side validation for the 90ci model.
 *
 * Reports one violation at a time, mirroring the engine's one-at-a-time
 * validation precedence (§4, §3 resolution). The hook uses `firstValidationError`
 * to block engine calls; `ModelEditor` uses the per-field helpers to mark cells.
 */

import type { Variable, Model } from './model';

/**
 * Parse a bound input string to a number or null.
 * Empty string → null (honest "blank"), not 0 (silent coercion).
 * A non-finite string → null.
 */
export function parseBound(raw: string): number | null {
  if (raw === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

/** Returns true when a p5 value is absent or not a finite number. */
export function hasBlankP5(v: Variable): boolean {
  return v.p5 === null || v.p5 === undefined || !Number.isFinite(v.p5);
}

/** Returns true when a p95 value is absent or not a finite number. */
export function hasBlankP95(v: Variable): boolean {
  return v.p95 === null || v.p95 === undefined || !Number.isFinite(v.p95);
}

/**
 * Returns the first validation error message for the model, or null if valid.
 *
 * E-04: blank/non-numeric p5 or p95 bound.
 *
 * One violation is reported at a time. The hook uses this to block `wasmSimulate`;
 * the result is shown verbatim in the error banner.
 */
export function firstValidationError(model: Model): string | null {
  for (const v of model.variables) {
    if (hasBlankP5(v)) {
      return `\`${v.name}\`: enter a number for the 5th bound.`;
    }
    if (hasBlankP95(v)) {
      return `\`${v.name}\`: enter a number for the 95th bound.`;
    }
  }
  return null;
}
