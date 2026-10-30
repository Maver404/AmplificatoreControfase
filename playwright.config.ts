/**
 * End-to-end tests on the e2e build (`vite build --mode e2e`, which installs window.__anc) served by
 * `vite preview` with the production Content-Security-Policy. The locally installed Google Chrome
 * plays a generated WAV as its fake microphone; the file cannot change while Chrome runs, so each
 * fixture gets its own project, which runs only the spec written for it (FIXTURE_SPECS). Output is muted
 * (--mute-audio): nothing reaches the speakers.
 */
import { defineConfig } from '@playwright/test';
import { FIXTURES, FIXTURE_SPECS, fixturePath } from './e2e/helpers.ts';

const PORT = 4173;

export default defineConfig({
  testDir: 'e2e',
  globalSetup: './e2e/global-setup.ts',
  // Real-time audio: parallel Chrome instances would compete for the audio thread (underruns
  // auto-mute the output and make the timing assertions meaningless).
  workers: 1,
  fullyParallel: false,
  forbidOnly: !!process.env['CI'],
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${PORT}`,
    channel: 'chrome',
    permissions: ['microphone'],
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `npm run build:e2e && npx vite preview --mode e2e --outDir dist-e2e --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/`,
    reuseExistingServer: false,
    timeout: 180_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
  projects: FIXTURES.map((name) => ({
    name,
    testMatch: `**/${FIXTURE_SPECS[name]}`,
    use: {
      launchOptions: {
        args: [
          '--use-fake-device-for-media-stream',
          '--use-fake-ui-for-media-stream',
          `--use-file-for-fake-audio-capture=${fixturePath(name)}`,
          '--autoplay-policy=no-user-gesture-required',
          '--mute-audio',
        ],
      },
    },
  })),
});
