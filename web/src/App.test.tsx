import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

// The hook calls `init()` on mount and `wasmSimulate()` on run. Mock the entire
// WASM module so tests don't fail on missing native binary.
vi.mock('./wasm/ninety_ci_wasm', () => ({
  default: vi.fn().mockResolvedValue(undefined),
  simulate: vi.fn(),
}));

describe('App smoke test', () => {
  it('renders the prefilled model name and a Run button', async () => {
    render(<App />);

    // The prefilled model name appears in the TopBar breadcrumb.
    expect(screen.getByText('Exchange exposure')).toBeInTheDocument();
    // The title input carries the same value.
    expect(screen.getByDisplayValue('Exchange exposure')).toBeInTheDocument();

    // The Run button is present and enabled. Wait for the async init to resolve
    // (the hook sets engineReady on mount) so no act() warning is emitted.
    await waitFor(() => {
      const runButton = screen.getByRole('button', { name: /run/i });
      expect(runButton).toBeInTheDocument();
      expect(runButton).not.toBeDisabled();
    });
  });
});

describe('Run — hard-wired sample count', () => {
  it('calls simulate with exactly 5000 samples', async () => {
    // Re-import the mock so we can inspect the call.
    const { simulate } = await import('./wasm/ninety_ci_wasm');
    const simulateMock = vi.mocked(simulate);
    simulateMock.mockClear();
    simulateMock.mockReturnValue({
      ci_low: 100,
      ci_high: 200,
      buckets: [150],
      counts: [5000],
      samples: 5000,
    });

    render(<App />);

    const user = userEvent.setup();
    // Use getAllByRole in case cleanup left a prior render in the DOM; take [0].
    const runButtons = screen.getAllByRole('button', { name: /run/i });
    await user.click(runButtons[0]);

    expect(simulateMock).toHaveBeenCalled();
    expect(simulateMock).toHaveBeenCalledWith(
      expect.any(String),   // equation
      expect.any(Array),    // vars
      5000,                 // SAMPLES — the invariant under test
      expect.any(Number),   // step
    );
  });
});
