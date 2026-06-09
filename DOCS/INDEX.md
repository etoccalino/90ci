# DOCS index

## PRD
- [PRD/90ci-web-calculator.md](PRD/90ci-web-calculator.md) — Product requirements: zero-backend web calculator, UX spec, v1 scope, variable-bounds semantics.
- [PRD/v1-acceptance.md](PRD/v1-acceptance.md) — v1 acceptance criteria: scope check, use-cases (UC-1 happy path, UC-2 undeclared-variable error; UC-3 TODO), input validation, error/edge-case matrix (E-01–E-12), distribution-label decision, graph spec, non-functional bars, test commitment, Definition of Done. Mostly filled; a few recommended DECISIONs await confirmation and the E-07 panic is the top correctness gate.

## Deployment
- [deploy-github-pages.md](deploy-github-pages.md) — Feasibility study and step-by-step plan for deploying the app as a static site to `etoccalino.github.io/90ci` via GitHub Pages.

## Architecture
- [architecture/overview.md](architecture/overview.md) — Workspace layout, crate split rationale, migration map from the old flat repo, `.gitignore` additions.
- [architecture/wasm-module.md](architecture/wasm-module.md) — Rust → WASM boundary: `core::simulate`, `wasm` wrapper crate, JS↔Rust marshalling, lifetime notes, known engine limitations.
- [architecture/frontend.md](architecture/frontend.md) — React + TypeScript front-end: state model, component tree, `useNinetyCi` hook, SVG charting approach, styling tokens.
- [architecture/toolchain.md](architecture/toolchain.md) — Build & integration: prerequisites, `build-wasm.sh`, Vite config, dev loop, test commands, CI sketch.
