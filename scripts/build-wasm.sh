#!/usr/bin/env sh
# Build the WASM module straight into the front-end source tree, where Vite
# imports it as an ES module. Output goes to web/src/wasm/ (git-ignored).
#
# Anchored to the repo root via this script's own location, so it works no matter
# the caller's cwd (pnpm runs `build:wasm` from web/, not the repo root).
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

# --out-dir is resolved by wasm-pack relative to the crate dir (crates/wasm),
# so ../../web/src/wasm lands at <root>/web/src/wasm.
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm
