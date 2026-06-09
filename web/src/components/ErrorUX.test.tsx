/**
 * Stage 4 — Front-end validation & error UX tests.
 *
 * Covers:
 *   E-04 — blank bound blocks Run; WASM simulate NOT called; inline cell marker; named error.
 *   E-09 — init() rejection → persistent banner + Run disabled.
 *   E-10 — after a prior successful Run, an erroring Run clears the chart.
 *   §3   — error banner has role="alert".
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';

// ─── WASM mock ──────────────────────────────────────────────────────────────
// Hoisted so vi.mock can reference them. We use a mutable object so individual
// tests can override `initImpl` and `simulateImpl` without re-importing.
const wasmControl = {
  initImpl: vi.fn().mockResolvedValue(undefined),
  simulateImpl: vi.fn(),
};

vi.mock('../wasm/ninety_ci_wasm', () => ({
  default: (...args: unknown[]) => wasmControl.initImpl(...args),
  simulate: (...args: unknown[]) => wasmControl.simulateImpl(...args),
}));

// A realistic SimResult fixture.
const GOOD_RESULT = {
  ci_low: 100,
  ci_high: 200,
  buckets: [100, 150, 200],
  counts: [10, 80, 10],
  samples: 5000,
};

beforeEach(() => {
  // Reset to a healthy default before each test.
  wasmControl.initImpl = vi.fn().mockResolvedValue(undefined);
  wasmControl.simulateImpl = vi.fn().mockReturnValue(GOOD_RESULT);
});

// ─── §3 role="alert" ────────────────────────────────────────────────────────

describe('§3 — error banner accessibility', () => {
  it('error banner has role="alert" and shows the error message', async () => {
    // Force simulate to throw so an error banner appears.
    wasmControl.simulateImpl = vi.fn().mockImplementation(() => {
      throw new Error('Engine error');
    });

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /run/i }));

    await waitFor(() => {
      const alerts = screen.getAllByRole('alert');
      expect(alerts).toHaveLength(1);
      expect(alerts[0].textContent).toMatch(/engine error/i);
    });
  });
});

// ─── E-04 — blank bound ─────────────────────────────────────────────────────

describe('E-04 — blank bound blocks Run', () => {
  it('blocks Run, names the variable and bound, and does NOT call simulate', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Clear the p5 bound of the first variable (EXCHANGE_RATE).
    // The spinbuttons (type=number inputs) are p5 and p95 of each variable row.
    // spinbuttons[0] = EXCHANGE_RATE p5 (value=1000), spinbuttons[1] = EXCHANGE_RATE p95.
    const spinbuttons = screen.getAllByRole('spinbutton');
    await user.tripleClick(spinbuttons[0]);
    await user.keyboard('{Delete}');

    await user.click(screen.getByRole('button', { name: /run/i }));

    await waitFor(() => {
      // Error banner must appear — exactly one.
      const alerts = screen.getAllByRole('alert');
      expect(alerts).toHaveLength(1);
      // The message must name the variable.
      expect(alerts[0].textContent).toContain('EXCHANGE_RATE');
    });

    // simulate must NOT have been called (engine not invoked with coerced 0).
    expect(wasmControl.simulateImpl).toHaveBeenCalledTimes(0);
  });

  it('marks the offending cell with an invalid class', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Clear p5 of the first variable row.
    const spinbuttons = screen.getAllByRole('spinbutton');
    await user.tripleClick(spinbuttons[0]);
    await user.keyboard('{Delete}');

    await user.click(screen.getByRole('button', { name: /run/i }));

    await waitFor(() => {
      // After validation error, exactly one input carries the invalid marker
      // (only the blank p5 cell of EXCHANGE_RATE is blank).
      const invalidCells = document.querySelectorAll('.num-input--invalid');
      expect(invalidCells).toHaveLength(1);
    });
  });
});

// ─── E-09 — init() rejection ────────────────────────────────────────────────

describe('E-09 — init() rejection', () => {
  it('shows persistent fallback message when init rejects', async () => {
    wasmControl.initImpl = vi.fn().mockRejectedValue(new Error('WASM load failed'));

    render(<App />);

    await waitFor(() => {
      const alerts = screen.getAllByRole('alert');
      const alertText = alerts.map((a) => a.textContent).join(' ');
      expect(alertText).toMatch(/reload/i);
    });
  });

  it('disables the Run button when init rejects', async () => {
    wasmControl.initImpl = vi.fn().mockRejectedValue(new Error('WASM load failed'));

    render(<App />);

    await waitFor(() => {
      const runButton = screen.getByRole('button', { name: /run/i });
      expect(runButton).toBeDisabled();
    });
  });

  it('SF1 — E-09 init failure with a blank bound does NOT mark cells invalid', async () => {
    // Engine fails to load.
    wasmControl.initImpl = vi.fn().mockRejectedValue(new Error('WASM load failed'));

    const user = userEvent.setup();
    render(<App />);

    // Wait for the init-failure banner (engine error present, Run disabled).
    await waitFor(() => {
      const alerts = screen.getAllByRole('alert');
      expect(alerts).toHaveLength(1);
      expect(screen.getByRole('button', { name: /run/i })).toBeDisabled();
    });

    // Clear p5 of the first variable — now a bound is blank.
    const spinbuttons = screen.getAllByRole('spinbutton');
    await user.tripleClick(spinbuttons[0]);
    await user.keyboard('{Delete}');

    // No Run was attempted — cell marking must NOT appear.
    expect(document.querySelectorAll('.num-input--invalid')).toHaveLength(0);
  });

  it('SF1 — actual E-04 Run with a blank bound still marks the cell', async () => {
    // Engine loads fine.
    const user = userEvent.setup();
    render(<App />);

    // Clear p5 of the first variable.
    const spinbuttons = screen.getAllByRole('spinbutton');
    await user.tripleClick(spinbuttons[0]);
    await user.keyboard('{Delete}');

    // Attempt a Run — validation fires.
    await user.click(screen.getByRole('button', { name: /run/i }));

    await waitFor(() => {
      // Exactly one invalid cell marked.
      expect(document.querySelectorAll('.num-input--invalid')).toHaveLength(1);
    });
  });
});

// ─── E-10 — stale chart cleared on error ────────────────────────────────────

describe('E-10 — stale chart cleared on subsequent error', () => {
  it('clears the chart when a subsequent Run errors after a prior successful Run', async () => {
    const user = userEvent.setup();
    render(<App />);

    // First Run succeeds — chart should appear.
    await user.click(screen.getByRole('button', { name: /run/i }));

    await waitFor(() => {
      // After a successful run, the result card appears (check for sample count text).
      expect(screen.getByText(/5,000 samples/i)).toBeInTheDocument();
    });

    // Now make simulate throw on the next call.
    wasmControl.simulateImpl = vi.fn().mockImplementation(() => {
      throw new Error('Engine error on second run');
    });

    await user.click(screen.getByRole('button', { name: /run/i }));

    await waitFor(() => {
      // Error banner must be present — exactly one.
      const alerts = screen.getAllByRole('alert');
      expect(alerts).toHaveLength(1);

      // The stale chart must be gone — "5,000 samples" text should no longer appear.
      expect(screen.queryByText(/5,000 samples/i)).not.toBeInTheDocument();
    });
  });
});
