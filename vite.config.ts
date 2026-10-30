import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

/**
 * The Content-Security-Policy and Permissions-Policy that nginx sends in production, read from the
 * one file that defines them, so `vite preview` (and the e2e tests running on it) see exactly the
 * same policy and catch CSP violations before Docker does.
 */
function productionHeaders(): Record<string, string> {
  const file = fileURLToPath(new URL('./docker/security-headers.conf', import.meta.url));
  const conf = readFileSync(file, 'utf8');
  const out: Record<string, string> = {};
  for (const name of ['Content-Security-Policy', 'Permissions-Policy']) {
    const m = new RegExp(`^\\s*add_header\\s+${name}\\s+"([^"]+)"`, 'm').exec(conf);
    if (!m) throw new Error(`vite.config.ts: ${name} not found in docker/security-headers.conf`);
    out[name] = m[1]!;
  }
  return out;
}

export default defineConfig(({ mode }) => ({
  // The worklet is bundled as an ES module chunk: AudioWorkletGlobalScope has no `self`,
  // so the default IIFE worker format would break in production builds.
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    outDir: mode === 'e2e' ? 'dist-e2e' : 'dist',
  },
  server: {
    // No CSP in dev: Vite's HMR client injects inline scripts.
    headers: { 'Permissions-Policy': 'microphone=(self)' },
  },
  preview: {
    headers: productionHeaders(),
  },
}));
