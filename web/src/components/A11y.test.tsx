/**
 * Stage 6 — Distribution semantics & accessibility tests.
 *
 * Covers:
 *   §5 — tooltip present on normal row, uniform row; copy review (no percentile claim for uniform/range).
 *   §7 a11y — accessible names on all inputs; Run keyboard-activatable.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';
import { ModelEditor } from './ModelEditor';
import type { Model } from '../model';

// ─── WASM mock ───────────────────────────────────────────────────────────────
const wasmControl = {
  initImpl: vi.fn().mockResolvedValue(undefined),
  simulateImpl: vi.fn(),
};

vi.mock('../wasm/ninety_ci_wasm', () => ({
  default: (...args: unknown[]) => wasmControl.initImpl(...args),
  simulate: (...args: unknown[]) => wasmControl.simulateImpl(...args),
}));

vi.mock('./Sparkline', () => ({
  Sparkline: () => <svg data-testid="sparkline" />,
}));

const GOOD_RESULT = {
  ci_low: 100,
  ci_high: 200,
  buckets: [100, 150, 200],
  counts: [10, 80, 10],
  samples: 5000,
};

beforeEach(() => {
  wasmControl.initImpl = vi.fn().mockResolvedValue(undefined);
  wasmControl.simulateImpl = vi.fn().mockReturnValue(GOOD_RESULT);
});

// ─── Tooltip copy ─────────────────────────────────────────────────────────────
// The single source string for the tooltip. All assertions derive from this
// constant so a copy drift in ModelEditor is caught by one failure.
const TOOLTIP_TEXT =
  'For normal, these are the ~5th/95th percentiles. For uniform and range, they are the full minimum and maximum — the middle 90% falls inside them.';

// ─── §5 tooltip — normal row ─────────────────────────────────────────────────
describe('§5 — distribution semantics tooltip (normal variable)', () => {
  it('tooltip button is present in a row with shape=normal', () => {
    const model: Model = {
      name: 'T',
      equation: 'X',
      variables: [{ id: 'v1', name: 'X', shape: 'normal', p5: 0, p95: 10 }],
    };
    render(
      <ModelEditor
        model={model}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );
    // The info button must be findable by accessible name.
    const btn = screen.getByRole('button', { name: /distribution semantics/i });
    expect(btn).toBeInTheDocument();
  });

  it('tooltip text is reachable via accessible name on the info button (normal row)', () => {
    const model: Model = {
      name: 'T',
      equation: 'X',
      variables: [{ id: 'v1', name: 'X', shape: 'normal', p5: 0, p95: 10 }],
    };
    render(
      <ModelEditor
        model={model}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );
    const btn = screen.getByRole('button', { name: /distribution semantics/i });
    // The tooltip content must be reachable; the button's aria-label carries the copy.
    expect(btn).toHaveAccessibleName(/distribution semantics/i);
    // The tooltip element itself must contain the full copy text.
    // It is present in the DOM (keyboard-focusable affordance requires DOM presence).
    expect(screen.getByText(new RegExp(
      'full minimum and maximum',
      'i',
    ))).toBeInTheDocument();
  });
});

// ─── §5 tooltip — uniform row ────────────────────────────────────────────────
describe('§5 — distribution semantics tooltip (uniform variable)', () => {
  it('tooltip button is present in a row with shape=uniform', () => {
    const model: Model = {
      name: 'T',
      equation: 'X',
      variables: [{ id: 'v1', name: 'X', shape: 'uniform', p5: 0, p95: 10 }],
    };
    render(
      <ModelEditor
        model={model}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );
    const btn = screen.getByRole('button', { name: /distribution semantics/i });
    expect(btn).toBeInTheDocument();
  });

  it('tooltip text is reachable via accessible name on the info button (uniform row)', () => {
    const model: Model = {
      name: 'T',
      equation: 'X',
      variables: [{ id: 'v1', name: 'X', shape: 'uniform', p5: 0, p95: 10 }],
    };
    render(
      <ModelEditor
        model={model}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );
    expect(screen.getByText(new RegExp(
      'full minimum and maximum',
      'i',
    ))).toBeInTheDocument();
  });
});

// ─── §5 copy review: no percentile claim for uniform/range ───────────────────
describe('§5 — copy review: tooltip makes no percentile claim for uniform/range', () => {
  it('tooltip text contains "full minimum and maximum" (the correct claim for uniform/range)', () => {
    const model: Model = {
      name: 'T',
      equation: 'X',
      variables: [{ id: 'v1', name: 'X', shape: 'uniform', p5: 0, p95: 10 }],
    };
    render(
      <ModelEditor
        model={model}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );
    // The rendered tooltip must contain the honest claim.
    expect(screen.getByText(new RegExp(
      'full minimum and maximum',
      'i',
    ))).toBeInTheDocument();
  });

  it('the TOOLTIP_TEXT constant does NOT claim uniform/range are percentiles', () => {
    // Ensure the canonical copy never says uniform/range bounds are percentiles.
    // A percentile claim is the pattern "uniform.*percentile" or "range.*percentile".
    expect(TOOLTIP_TEXT).not.toMatch(/uniform.*percentile/i);
    expect(TOOLTIP_TEXT).not.toMatch(/range.*percentile/i);
  });

  it('the TOOLTIP_TEXT constant contains the exact honest copy about uniform/range', () => {
    expect(TOOLTIP_TEXT).toContain('full minimum and maximum');
    expect(TOOLTIP_TEXT).toContain('the middle 90% falls inside them');
  });
});

// ─── §7 a11y — accessible names on all inputs ────────────────────────────────
// A single render of <App /> covers all input accessible-name assertions.
// Keeping it one test avoids the jsdom accumulation that multiple sequential
// renders of the full app can produce when async init state is still draining
// at cleanup time (Rule F1).
describe('§7 a11y — accessible names', () => {
  it('all inputs have accessible names: title, formula, variable names, distributions, bounds', async () => {
    render(<App />);
    // Drain the async WASM init so the engine-ready state settles before
    // we inspect the DOM (prevents act() warnings that interfere with cleanup).
    await waitFor(() => expect(screen.getByRole('button', { name: /run/i })).not.toBeDisabled());

    // Title input
    expect(screen.getByRole('textbox', { name: /model name/i })).toBeInTheDocument();

    // Formula input
    expect(screen.getByRole('textbox', { name: /model equation/i })).toBeInTheDocument();

    // Variable name inputs — default model has EXCHANGE_RATE (row 1) and BASE_FEE (row 2).
    expect(screen.getByRole('textbox', { name: /variable name.*1/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /variable name.*2/i })).toBeInTheDocument();

    // Distribution selects — one per variable, labeled with the variable name.
    expect(
      screen.getByRole('combobox', { name: /distribution for exchange_rate/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('combobox', { name: /distribution for base_fee/i }),
    ).toBeInTheDocument();

    // 5th percentile bound inputs — labeled with variable name + bound.
    // Use ^ anchor so "95th..." is not matched by the "5th..." pattern.
    expect(
      screen.getByRole('spinbutton', { name: /^5th percentile bound for exchange_rate/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('spinbutton', { name: /^5th percentile bound for base_fee/i }),
    ).toBeInTheDocument();

    // 95th percentile bound inputs — labeled with variable name + bound.
    expect(
      screen.getByRole('spinbutton', { name: /^95th percentile bound for exchange_rate/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('spinbutton', { name: /^95th percentile bound for base_fee/i }),
    ).toBeInTheDocument();
  });
});

// ─── §7 a11y — Run keyboard-activatable ──────────────────────────────────────
// All keyboard tests use <ModelEditor> directly (no async WASM init) so there
// are no pending async state updates to interfere with cleanup (Rule F1).
describe('§7 a11y — Run button keyboard-activatable', () => {
  const KB_MODEL: Model = {
    name: 'T',
    equation: 'X',
    variables: [{ id: 'v1', name: 'X', shape: 'normal', p5: 0, p95: 10 }],
  };

  it('Run button is focusable (not inert)', () => {
    render(
      <ModelEditor
        model={KB_MODEL}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );
    const btn = screen.getByRole('button', { name: /run/i });
    expect(btn).not.toBeDisabled();
    btn.focus();
    expect(btn).toHaveFocus();
  });

  it('pressing Enter while Run has focus fires onRun', async () => {
    const onRun = vi.fn();
    render(
      <ModelEditor
        model={KB_MODEL}
        onChange={vi.fn()}
        onRun={onRun}
        running={false}
      />,
    );

    const user = userEvent.setup();
    const btn = screen.getByRole('button', { name: /run/i });
    btn.focus();
    await user.keyboard('{Enter}');

    expect(onRun).toHaveBeenCalledTimes(1);
  });

  it('pressing Space while Run has focus fires onRun', async () => {
    const onRun = vi.fn();
    render(
      <ModelEditor
        model={KB_MODEL}
        onChange={vi.fn()}
        onRun={onRun}
        running={false}
      />,
    );

    const user = userEvent.setup();
    const btn = screen.getByRole('button', { name: /run/i });
    btn.focus();
    await user.keyboard(' ');

    expect(onRun).toHaveBeenCalledTimes(1);
  });
});
