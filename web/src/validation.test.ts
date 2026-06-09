import { describe, it, expect } from 'vitest';
import { firstValidationError, hasBlankP5, hasBlankP95 } from './validation';
import type { Model } from './model';

const VALID_MODEL: Model = {
  name: 'Test',
  equation: 'A + B',
  variables: [
    { id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: 10 },
    { id: 'v2', name: 'B', shape: 'normal', p5: 5, p95: 15 },
  ],
};

describe('firstValidationError', () => {
  it('returns null for a valid model', () => {
    expect(firstValidationError(VALID_MODEL)).toBeNull();
  });

  it('E-04: blank p5 (null) produces an error naming the variable and bound', () => {
    const model: Model = {
      ...VALID_MODEL,
      variables: [
        { id: 'v1', name: 'A', shape: 'uniform', p5: null as unknown as number, p95: 10 },
        { id: 'v2', name: 'B', shape: 'normal', p5: 5, p95: 15 },
      ],
    };
    const err = firstValidationError(model);
    expect(err).not.toBeNull();
    expect(err).toContain('A');
    expect(err).toContain('5th');
  });

  it('E-04: blank p95 (null) produces an error naming the variable and bound', () => {
    const model: Model = {
      ...VALID_MODEL,
      variables: [
        { id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: null as unknown as number },
        { id: 'v2', name: 'B', shape: 'normal', p5: 5, p95: 15 },
      ],
    };
    const err = firstValidationError(model);
    expect(err).not.toBeNull();
    expect(err).toContain('A');
    expect(err).toContain('95th');
  });

  it('E-04: NaN p5 (from Number("")) produces an error', () => {
    const model: Model = {
      ...VALID_MODEL,
      variables: [
        { id: 'v1', name: 'A', shape: 'uniform', p5: NaN, p95: 10 },
        { id: 'v2', name: 'B', shape: 'normal', p5: 5, p95: 15 },
      ],
    };
    const err = firstValidationError(model);
    expect(err).not.toBeNull();
    expect(err).toContain('A');
    expect(err).toContain('5th');
  });

  it('E-04: reports the first violating variable, not all', () => {
    // Both variables have blank p5; only the first should be reported.
    const model: Model = {
      ...VALID_MODEL,
      variables: [
        { id: 'v1', name: 'A', shape: 'uniform', p5: null as unknown as number, p95: 10 },
        { id: 'v2', name: 'B', shape: 'normal', p5: null as unknown as number, p95: 15 },
      ],
    };
    const err = firstValidationError(model);
    expect(err).not.toBeNull();
    // Reports A (first), not necessarily B
    expect(err).toContain('A');
  });
});

describe('hasBlankP5 / hasBlankP95 per-field helpers', () => {
  it('hasBlankP5 returns true for null', () => {
    expect(hasBlankP5({ id: 'v1', name: 'A', shape: 'uniform', p5: null as unknown as number, p95: 10 })).toBe(true);
  });

  it('hasBlankP5 returns true for NaN', () => {
    expect(hasBlankP5({ id: 'v1', name: 'A', shape: 'uniform', p5: NaN, p95: 10 })).toBe(true);
  });

  it('hasBlankP5 returns false for 0 (valid number)', () => {
    expect(hasBlankP5({ id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: 10 })).toBe(false);
  });

  it('hasBlankP95 returns true for null', () => {
    expect(hasBlankP95({ id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: null as unknown as number })).toBe(true);
  });

  it('hasBlankP95 returns true for NaN', () => {
    expect(hasBlankP95({ id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: NaN })).toBe(true);
  });

  it('hasBlankP95 returns false for 0 (valid number)', () => {
    expect(hasBlankP95({ id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: 0 })).toBe(false);
  });
});
