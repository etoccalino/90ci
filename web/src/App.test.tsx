import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

// The hook calls `init()` on mount and `wasmSimulate()` on run. Mock the entire
// WASM module so tests don't fail on missing native binary.
vi.mock('./wasm/ninety_ci_wasm', () => ({
  default: vi.fn().mockResolvedValue(undefined),
  simulate: vi.fn(),
}));

describe('App smoke test', () => {
  it('renders the prefilled model name and a Run button', () => {
    render(<App />);

    // The prefilled model name appears in the TopBar breadcrumb and the title input.
    expect(screen.getAllByText('Exchange exposure').length).toBeGreaterThan(0);

    // The Run button is present and enabled on initial render.
    const runButton = screen.getByRole('button', { name: /run/i });
    expect(runButton).toBeInTheDocument();
    expect(runButton).not.toBeDisabled();
  });
});
