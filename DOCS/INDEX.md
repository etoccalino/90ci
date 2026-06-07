# DOCS index

## PRD
- [PRD/90ci-web-calculator.md](PRD/90ci-web-calculator.md) — Product requirements: zero-backend web calculator, UX spec, v1 scope, variable-bounds semantics.

## Architecture
- [architecture/overview.md](architecture/overview.md) — Workspace layout, crate split rationale, migration map from the old flat repo, `.gitignore` additions.
- [architecture/wasm-module.md](architecture/wasm-module.md) — Rust → WASM boundary: `core::simulate`, `wasm` wrapper crate, JS↔Rust marshalling, lifetime notes, known engine limitations.
- [architecture/frontend.md](architecture/frontend.md) — React + TypeScript front-end: state model, component tree, `useNinetyCi` hook, SVG charting approach, styling tokens.
- [architecture/toolchain.md](architecture/toolchain.md) — Build & integration: prerequisites, `build-wasm.sh`, Vite config, dev loop, test commands, CI sketch.
