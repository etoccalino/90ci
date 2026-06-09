/**
 * Stage 5 — OutputChart tests.
 *
 * Covers:
 *   - Normal-spread render: area path, blue outline, shaded band, exactly 2 dashed
 *     markers, exactly 2 numeric marker labels, exactly 3 x-axis ticks.
 *   - Degenerate (buckets.length < 2): visible spike and value label, NOT blank svg.
 *   - Marker x-position matches ciLow/ciHigh mapping.
 *   - Negative values: axis ticks render negative numbers correctly.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { OutputChart } from './OutputChart';
import type { SimResult } from '../model';

// ─── Fixtures ────────────────────────────────────────────────────────────────

const NORMAL: SimResult = {
  ciLow: 10,
  ciHigh: 90,
  buckets: [0, 50, 100],
  counts: [5, 80, 5],
  samples: 5000,
};

// Degenerate: single bucket — fixed-value model.
const DEGENERATE: SimResult = {
  ciLow: 42,
  ciHigh: 42,
  buckets: [42],
  counts: [5000],
  samples: 5000,
};

// Negative-value span.
const NEGATIVE: SimResult = {
  ciLow: -90,
  ciHigh: -10,
  buckets: [-100, -50, 0],
  counts: [5, 80, 5],
  samples: 5000,
};

// Very narrow spread: ciLow and ciHigh map to essentially the same pixel.
const NARROW: SimResult = {
  ciLow: 50,
  ciHigh: 50,
  buckets: [0, 50, 100],
  counts: [5, 80, 5],
  samples: 5000,
};

// Chart constants (must mirror OutputChart.tsx).
const W = 520;

function xOf(val: number, lo: number, hi: number): number {
  const span = hi - lo || 1;
  return ((val - lo) / span) * W;
}

// ─── Normal-spread render ─────────────────────────────────────────────────────

describe('OutputChart — normal spread', () => {
  it('renders the filled area path', () => {
    const { container } = render(<OutputChart result={NORMAL} />);
    const paths = container.querySelectorAll('path');
    // area path starts at "M 0 <base>"
    const areaPaths = Array.from(paths).filter((p) => p.getAttribute('d')?.startsWith('M 0'));
    expect(areaPaths).toHaveLength(1);
    expect(areaPaths[0].getAttribute('fill')).toBe('#eef2f6');
  });

  it('renders the blue outline path', () => {
    const { container } = render(<OutputChart result={NORMAL} />);
    const paths = container.querySelectorAll('path');
    const outlines = Array.from(paths).filter(
      (p) => p.getAttribute('fill') === 'none' && p.getAttribute('stroke') === 'var(--blue)',
    );
    expect(outlines).toHaveLength(1);
    expect(outlines[0].getAttribute('stroke-width')).toBe('2');
  });

  it('renders the shaded CI band', () => {
    const { container } = render(<OutputChart result={NORMAL} />);
    const paths = container.querySelectorAll('path');
    const bands = Array.from(paths).filter((p) => p.getAttribute('fill-opacity') === '0.18');
    expect(bands).toHaveLength(1);
    expect(bands[0].getAttribute('fill')).toBe('var(--blue)');
  });

  it('renders exactly 2 dashed marker lines', () => {
    const { container } = render(<OutputChart result={NORMAL} />);
    const lines = container.querySelectorAll('line');
    const dashed = Array.from(lines).filter((l) => l.getAttribute('stroke-dasharray') === '3 3');
    expect(dashed).toHaveLength(2);
  });

  it('renders exactly 2 numeric marker labels matching fmt(ciLow) and fmt(ciHigh)', () => {
    render(<OutputChart result={NORMAL} />);
    // fmt(10) = "10", fmt(90) = "90"
    const labelLow = screen.getByTestId('marker-label-low');
    const labelHigh = screen.getByTestId('marker-label-high');
    expect(labelLow.textContent).toBe('10');
    expect(labelHigh.textContent).toBe('90');
  });

  it('renders exactly 3 x-axis ticks with correct lo/mid/hi text', () => {
    render(<OutputChart result={NORMAL} />);
    // lo=0, mid=50, hi=100; fmt rounds to integer and uses en-US locale.
    const axis = screen.getByText('0').closest('.axis');
    expect(axis).not.toBeNull();
    const spans = axis!.querySelectorAll('span');
    // Assert complete count — rule F3.
    expect(spans).toHaveLength(3);
    expect(spans[0].textContent).toBe('0');
    expect(spans[1].textContent).toBe('50');
    expect(spans[2].textContent).toBe('100');
  });
});

// ─── Degenerate render ───────────────────────────────────────────────────────

describe('OutputChart — degenerate (buckets.length < 2)', () => {
  it('renders a visible spike element (NOT an empty svg)', () => {
    const { container } = render(<OutputChart result={DEGENERATE} />);
    const spike = container.querySelector('[data-testid="degenerate-spike"]');
    // Positive assertion: the spike is present (rule 8 — not just "no empty svg").
    expect(spike).not.toBeNull();
  });

  it('renders the value label with the correct text', () => {
    render(<OutputChart result={DEGENERATE} />);
    const label = screen.getByTestId('degenerate-label');
    // fmt(42) = "42"
    expect(label.textContent).toBe('42');
  });

  it('does NOT render an svg with no children (blank svg guard)', () => {
    const { container } = render(<OutputChart result={DEGENERATE} />);
    const svg = container.querySelector('svg');
    expect(svg).not.toBeNull();
    // The svg has content (spike line + label), not zero children.
    expect(svg!.children.length).toBeGreaterThan(0);
  });
});

// ─── Marker position matches data ────────────────────────────────────────────

describe('OutputChart — marker x-position matches ciLow/ciHigh', () => {
  it('marker-low x1 corresponds to x(ciLow)', () => {
    const { container } = render(<OutputChart result={NORMAL} />);
    const markerLow = container.querySelector('[data-testid="marker-low"]');
    expect(markerLow).not.toBeNull();
    const x1 = parseFloat(markerLow!.getAttribute('x1')!);
    const expected = xOf(NORMAL.ciLow, NORMAL.buckets[0], NORMAL.buckets[NORMAL.buckets.length - 1]);
    // Normal spread: no adjustment applied.
    expect(x1).toBeCloseTo(expected, 3);
  });

  it('marker-high x1 corresponds to x(ciHigh)', () => {
    const { container } = render(<OutputChart result={NORMAL} />);
    const markerHigh = container.querySelector('[data-testid="marker-high"]');
    expect(markerHigh).not.toBeNull();
    const x1 = parseFloat(markerHigh!.getAttribute('x1')!);
    const expected = xOf(NORMAL.ciHigh, NORMAL.buckets[0], NORMAL.buckets[NORMAL.buckets.length - 1]);
    expect(x1).toBeCloseTo(expected, 3);
  });
});

// ─── Negative values ─────────────────────────────────────────────────────────

describe('OutputChart — negative values', () => {
  it('renders x-axis ticks with negative text', () => {
    render(<OutputChart result={NEGATIVE} />);
    // lo=-100, mid=-50, hi=0
    // fmt(-100) = "-100" in en-US locale (Math.round(-100).toLocaleString)
    const axis = screen.getByText('-100').closest('.axis');
    expect(axis).not.toBeNull();
    const spans = axis!.querySelectorAll('span');
    expect(spans).toHaveLength(3);
    expect(spans[0].textContent).toBe('-100');
    expect(spans[1].textContent).toBe('-50');
    expect(spans[2].textContent).toBe('0');
  });

  it('renders marker labels with negative values', () => {
    render(<OutputChart result={NEGATIVE} />);
    const labelLow = screen.getByTestId('marker-label-low');
    const labelHigh = screen.getByTestId('marker-label-high');
    expect(labelLow.textContent).toBe('-90');
    expect(labelHigh.textContent).toBe('-10');
  });
});

// ─── Narrow spread ───────────────────────────────────────────────────────────

describe('OutputChart — narrow spread (x5 ≈ x95)', () => {
  it('still renders both marker line elements', () => {
    const { container } = render(<OutputChart result={NARROW} />);
    const lines = container.querySelectorAll('line');
    const dashed = Array.from(lines).filter((l) => l.getAttribute('stroke-dasharray') === '3 3');
    expect(dashed).toHaveLength(2);
  });

  it('renders a combined label (not two overlapping labels) when spread collapses', () => {
    const { container } = render(<OutputChart result={NARROW} />);
    // On collapse, only marker-label-low is rendered (combined), not marker-label-high.
    const labelLow = container.querySelector('[data-testid="marker-label-low"]');
    const labelHigh = container.querySelector('[data-testid="marker-label-high"]');
    expect(labelLow).not.toBeNull();
    // narrow spread: high label collapsed into combined, so marker-label-high absent.
    expect(labelHigh).toBeNull();
  });
});
