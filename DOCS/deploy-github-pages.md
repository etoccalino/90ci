# GitHub Pages deployment exploration

## Verdict: feasible, minimal changes required

The app (Vite + React + WASM) can be deployed to `https://etoccalino.github.io/90ci/` as a fully static site. No backend, no CDN, no custom server needed. GitHub Pages correctly serves `.wasm` files with `Content-Type: application/wasm`, so `WebAssembly.instantiateStreaming()` works without a fallback.

---

## What needs to change

### 1. `vite.config.ts` — base path

GitHub Pages serves the app at `https://etoccalino.github.io/90ci/`, not at the root. Vite must embed this sub-path into asset URLs at build time:

```ts
base: process.env.GITHUB_ACTIONS === 'true' ? '/90ci/' : '/',
```

Add this to `vite.config.ts` alongside the existing `build: { target: 'esnext' }`. Local `vite dev` keeps working at `http://localhost:5173/` unaffected.

### 2. `web/public/.nojekyll` — disable Jekyll

An empty `.nojekyll` file placed in `web/public/` is copied to `dist/` by Vite and tells GitHub Pages not to run Jekyll. Without it, Jekyll can corrupt binary files, including `.wasm`.

```sh
touch web/public/.nojekyll
```

### 3. `.github/workflows/deploy.yml` — CI/CD pipeline

The full pipeline: install Rust + wasm-pack + pnpm → `pnpm build` (which already runs `build-wasm.sh` then `vite build`) → deploy `web/dist/`.

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown

      - name: Install wasm-pack
        run: cargo install wasm-pack --version 0.15.0 --locked

      - name: Setup pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 9

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'pnpm'
          cache-dependency-path: web/pnpm-lock.yaml

      - name: Install dependencies
        run: pnpm install
        working-directory: web

      - name: Build
        run: pnpm build
        working-directory: web

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: web/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

### 4. GitHub repo settings (one-time, manual)

In **Settings → Pages → Build and deployment**, set Source to **"GitHub Actions"**. No `docs/` directory or `gh-pages` branch is needed.

---

## What does NOT need to change

| Item | Status |
|---|---|
| `web/src/wasm/` gitignored | Already done — wasm-pack output is not committed |
| WASM MIME type | GitHub Pages serves `application/wasm` correctly |
| `vite-plugin-wasm` / `vite-plugin-top-level-await` | No changes; they work with GitHub Pages |
| CORS for WASM fetch | GitHub Pages sets `Access-Control-Allow-Origin: *`; WASM init works |
| WASM threads / SharedArrayBuffer | Not used (v1 runs simulate synchronously) — no COOP/COEP needed |

---

## Known limitations

- **No custom HTTP headers.** GitHub Pages does not allow setting COOP/COEP headers. This blocks `SharedArrayBuffer`-based WASM threads if ever needed in the future. Workaround: `coi-serviceworker` (service worker that injects the headers), or migrate to Cloudflare Pages / Netlify which support a `_headers` file.
- **Bandwidth soft cap.** GitHub Pages has a ~100 GB/month bandwidth soft limit. Not a concern for this app at v1.
- **Build time.** The Rust compilation step adds ~2–3 min to CI on a cold cache. `dtolnay/rust-toolchain` + `sccache` can improve this if it becomes a pain.
