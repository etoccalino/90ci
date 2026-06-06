# Architecture — front-end (React + TypeScript)

## Stack

Vite + React + TypeScript, package-managed with pnpm. Output is a fully static bundle (`vite build` → `web/dist/`) deployable to any static host (GitHub Pages, Netlify, …). No backend. All computation happens in the WASM module; React is presentation only.

## State model

```ts
type Shape = 'uniform' | 'normal' | 'range';

interface Variable {
  id: string;        // stable key for the row
  name: string;      // must appear in the equation
  shape: Shape;
  p5: number;        // "5th" column — maps to core's `lower`
  p95: number;       // "95th" column — maps to core's `upper`
}

interface Model {
  name: string;
  equation: string;
  variables: Variable[];
  samples: number;   // 1_000 | 10_000 | 100_000
}

interface SimResult {
  ciLow: number;
  ciHigh: number;
  buckets: number[]; // bucket lower bounds
  counts: number[];  // samples per bucket
  samples: number;
}
```

`p5`/`p95` map directly to the engine's `lower`/`upper`. (See the PRD open item on uniform/range percentile semantics — a labelling concern, not a state-shape concern.)

## Component tree

Mapped from the Spreadsheet mock; the mock's "Tweaks" panel is dropped.

```
App
├─ TopBar                 # breadcrumb; reflects the model name
└─ (two-column layout)
   ├─ ModelEditor         # left column — build
   │  ├─ EditableTitle    # model name; syncs breadcrumb
   │  ├─ FormulaBar       # equation (highlighted) + Run button
   │  └─ VariablesTable
   │     ├─ VariableRow*  # name, distribution pill, p5, p95, Sparkline
   │     └─ AddVariableRow
   └─ ResultPanel         # right column — read
      ├─ OutputChart       # histogram area chart + CI band + markers
      └─ CIHero            # range (low – high) + width
```

`Sparkline` renders the per-distribution shape preview (uniform plateau vs. normal bell), ported from the mock's `sparkPath`.

## Compute boundary — `useNinetyCi()`

A hook owns the WASM lifecycle:
- Initializes the module **once** — `await init()` from the `wasm-pack --target web` glue (imported from `src/wasm/`).
- Exposes `run(model: Model): Promise<SimResult>`, which maps the `Model` into the `simulate(equation, vars, iterations, step)` call (`vars` = `[{name, shape, lower: p5, upper: p95}]`), and normalizes the returned object into `SimResult`.
- Surfaces loading/error state for the Run button.

Flow: user edits state → clicks **Run** → `run(model)` → store `SimResult` → `OutputChart` + `CIHero` re-render. Presentation stays pure React; all math is in WASM.

## Charting

Reuse the mock's hand-rolled SVG approach (`renderChart` in the Spreadsheet HTML) but drive it from the **real** `(buckets, counts)` instead of the hardcoded `plateau()` function:
- Normalize `counts` to the chart height to draw the distribution area/line.
- Shade the `[ciLow, ciHigh]` band.
- Draw the two dashed CI markers and their value labels.

No charting dependency is needed for v1 — the SVG is simple and already prototyped.

## Styling

Port the mock's design tokens (CSS custom properties `--ink`, `--ink-2`, `--blue`, `--grn`, …, and the Inter Tight + JetBrains Mono fonts) into the app's stylesheet. Recreate the **visual output**, not the prototype's DOM structure — the React component tree above is the real structure.
