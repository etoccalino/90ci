# Architecture — build & integration toolchain

## Prerequisites

- **Rust** via rustup, plus the wasm target: `rustup target add wasm32-unknown-unknown`.
- **wasm-pack**: `cargo install wasm-pack` (or via cargo-binstall).
- **Node + pnpm** — Node **≥ 22.13** (pnpm 11 imports `node:sqlite`, absent before Node 22). Install via Volta: `volta install node@22`.
- **Chromium + chromedriver** — required for `wasm-pack test --headless --chrome`. On Arch Linux: `sudo pacman -S chromium` (installs both `/usr/bin/chromium` and `/usr/bin/chromedriver`).

> Note: no Rust toolchain is currently installed in this environment (`cargo`/`rustup` are absent). Install the above before the first wasm build.

## Integration: how the two components meet

The WASM module is built by `wasm-pack` directly into the front-end's source tree, where Vite imports it as an ES module.

`scripts/build-wasm.sh`:
```sh
#!/usr/bin/env sh
set -eu
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm
```

`web/package.json` scripts:
```json
{
  "scripts": {
    "build:wasm": "sh ../scripts/build-wasm.sh",
    "dev":        "pnpm build:wasm && vite",
    "build":      "pnpm build:wasm && vite build",
    "preview":    "vite preview"
  }
}
```

`web/vite.config.ts`:
```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [react(), wasm(), topLevelAwait()],
  build: { target: 'esnext' },
});
```

The app imports the generated glue from `src/wasm/` and calls `init()` once (see `frontend.md`, `useNinetyCi`).

## Dev loop

1. Edit Rust in `crates/core` / `crates/wasm`.
2. `pnpm build:wasm` regenerates `web/src/wasm/`.
3. Vite HMR picks up the change.

The wasm rebuild is a **manual step** in v1 (not file-watched). A `cargo watch`-based auto-rebuild is a possible later convenience.

## Tests

- **Rust**: `cargo test -p ninety_ci_core` (the `core` crate; covers the existing unit + integration tests in `tests/simple.rs`).
- **Front-end**: `pnpm -C web test` — runs Vitest (`vitest run`) across `web/src/**/*.{test,spec}.{ts,tsx}`. No test files exist yet; write them alongside components.

## CI sketch (documented, not wired)

A single pipeline:
1. Install Rust + `wasm32-unknown-unknown` + `wasm-pack`; install pnpm.
2. `cargo test -p ninety_ci_core`.
3. `pnpm install && pnpm build` (which runs `build:wasm` then `vite build`).
4. Deploy `web/dist/` to the static host.
