import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ModelEditor } from './ModelEditor';
import type { Model } from '../model';

// Sparkline renders an SVG via the WASM-free path; no mock needed.
// ModelEditor itself has no wasm dep — only stub what jsdom can't handle.
vi.mock('./Sparkline', () => ({
  Sparkline: () => <svg data-testid="sparkline" />,
}));

const BASE_MODEL: Model = {
  name: 'Test model',
  equation: 'A + B',
  variables: [
    { id: 'v1', name: 'A', shape: 'uniform', p5: 0, p95: 10 },
    { id: 'v2', name: 'B', shape: 'normal', p5: 5, p95: 15 },
  ],
};

describe('ModelEditor — sample count control', () => {
  it('renders NO "Samples" label', () => {
    render(
      <ModelEditor
        model={BASE_MODEL}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );

    // There must be no label/text that exposes sample-count selection.
    expect(screen.queryByText(/^samples$/i)).not.toBeInTheDocument();
  });

  it('renders NO combobox (select) for choosing a sample count', () => {
    render(
      <ModelEditor
        model={BASE_MODEL}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );

    const allComboboxes = screen.queryAllByRole('combobox');
    // One distribution select per variable, nothing more (no sample-count select).
    expect(allComboboxes).toHaveLength(BASE_MODEL.variables.length);
  });

  it('still renders the hint note about percentile bounds', () => {
    render(
      <ModelEditor
        model={BASE_MODEL}
        onChange={vi.fn()}
        onRun={vi.fn()}
        running={false}
      />,
    );

    // getAllByText is used because prior renders in the same file share jsdom;
    // we just need at least one matching element.
    const notes = screen.getAllByText(/5th \/ 95th are the percentile bounds for each variable/i);
    expect(notes.length).toBeGreaterThanOrEqual(1);
  });
});
