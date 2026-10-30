/**
 * The guarded `test` and the page helpers shared by the specs. Every test fails on console errors, CSP
 * violations and on any fault, processorerror or howl event. Actions go through the real buttons and keys;
 * levels and values are read from the text readouts, samples from the tap blocks of window.__anc.
 */
import { expect, test as base } from '@playwright/test';
import type { Locator, Page } from '@playwright/test';
import type { E2eEvent, E2eTap } from '../src/app/e2e-hooks.ts';
import { TAP_BLOCK_FRAMES } from '../src/audio/protocol.ts';

export { expect };

// ------------------------------------------------------------------------------ guard fixture

function isBadEvent(e: E2eEvent): boolean {
  if (e.type === 'fault' || e.type === 'howl') return true;
  if (e.type !== 'lifecycle') return false;
  if (e.source === 'engine') return (e.detail as { cause?: unknown } | undefined)?.cause === 'processorerror';
  return typeof e.detail === 'string' && e.detail.startsWith('processorerror');
}

export const test = base.extend<{ guard: void }>({
  guard: [
    async ({ page }, use, testInfo) => {
      const consoleErrors: string[] = [];
      page.on('console', (m) => {
        if (m.type() !== 'error') return;
        consoleErrors.push(`${m.text()} (${m.location().url})`);
      });
      page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
      // Runs before the app's scripts: CSP reports and the exact time of every Escape keydown.
      await page.addInitScript(() => {
        const probe = { csp: [] as string[], esc: [] as number[] };
        window.__e2e = probe;
        document.addEventListener(
          'securitypolicyviolation',
          (e) => probe.csp.push(`${e.effectiveDirective} ${e.blockedURI}`),
          true,
        );
        window.addEventListener(
          'keydown',
          (e) => {
            if (e.key === 'Escape') probe.esc.push(performance.now());
          },
          true,
        );
      });

      await use();

      const after = await page
        .evaluate(() => ({ csp: window.__e2e?.csp ?? [], events: window.__anc?.events ?? [] }))
        .catch((err: unknown) => ({ csp: [`page not readable: ${String(err)}`], events: [] as E2eEvent[] }));
      if (testInfo.status !== testInfo.expectedStatus) {
        await testInfo.attach('events.json', { body: JSON.stringify(after.events, null, 2), contentType: 'application/json' });
      }
      const bad = after.events.filter(isBadEvent).map((e) => `${e.source}:${e.type} ${JSON.stringify(e.detail)}`);
      expect(consoleErrors, 'console errors').toEqual([]);
      expect(after.csp, 'CSP violations').toEqual([]);
      expect(bad, 'fault / processorerror / howl events').toEqual([]);
    },
    { auto: true },
  ],
});

// ------------------------------------------------------------------------------ page helpers

export const statusChip = (page: Page): Locator => page.locator('.status-bar .sb-state');
export const statusMessage = (page: Page): Locator => page.locator('.status-bar .sb-message-text');
export const strip = (page: Page, ch: 'L' | 'R'): Locator => page.getByRole('article', { name: `Canale ${ch}` });

export async function openApp(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(() => window.__anc !== undefined);
}

/** getUserMedia with the fake device normally resolves in well under a second. */
const MIC_OPEN_TIMEOUT_MS = 20_000;

export async function allowMicrophone(page: Page): Promise<void> {
  await page.locator('#devices').getByRole('button', { name: 'Consenti microfono' }).click();
  try {
    await page.waitForFunction(
      () => {
        const s = window.__anc!.controller.state;
        return s.micOpen && !s.opening;
      },
      undefined,
      { timeout: MIC_OPEN_TIMEOUT_MS },
    );
  } catch {
    const s = await page.evaluate(() => {
      const a = window.__anc!;
      return { opening: a.controller.state.opening, micError: a.controller.state.micError, engine: a.engine.state, ctx: a.engine.contextState };
    });
    throw new Error(
      `«Consenti microfono» did not open the fake microphone within ${MIC_OPEN_TIMEOUT_MS / 1000} s ` +
        `(${JSON.stringify(s)}). If getUserMedia never settles, Chrome's audio service is blocked by the OS ` +
        '(on macOS: microphone permission of the app that runs Playwright, or CoreAudio not answering).',
    );
  }
}

export async function sampleRate(page: Page): Promise<number> {
  const fs = await page.evaluate(() => window.__anc!.controller.sampleRate);
  if (fs === null) throw new Error('sample rate unknown');
  return fs;
}

/** «Attiva uscita» once validation passes; resolves when the 1.5 s soft-start has completed. */
export async function enableOutput(page: Page): Promise<void> {
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const s = window.__anc!.controller.state;
          if (s.canArm) return 'ok';
          const open = s.validation.issues.filter((i) => i.level === 'block' || (i.level === 'error' && !i.overridden));
          return open.map((i) => i.text).join(' | ') || 'not armable yet';
        }),
      { message: 'validation must pass with the fake microphone' },
    )
    .toBe('ok');
  await page.locator('#output-gate').getByRole('button', { name: 'Attiva uscita' }).click();
  await expect(statusChip(page)).toHaveText('Attivo');
}

/** The last `blocks` tap blocks, all delivered after this call started. */
export async function freshTaps(page: Page, blocks: number): Promise<E2eTap> {
  const seq0 = await page.evaluate(() => window.__anc!.lastTap()?.seq ?? 0);
  await page.waitForFunction((target) => (window.__anc!.lastTap()?.seq ?? 0) >= target, seq0 + blocks, { polling: 50 });
  const tap = await page.evaluate((n) => window.__anc!.lastTap(n), blocks);
  if (tap === null) throw new Error('no tap blocks');
  expect(tap.frames).toBe(blocks * TAP_BLOCK_FRAMES);
  return tap;
}

/** Planar tap channels as Float32Array (TAP_CHANNELS order, 'err' left out). */
export interface PackedTap {
  /** window.__anc.lastTap().seq of the newest block in the window. */
  seq: number;
  frames: number;
  ref: Float32Array;
  cond: Float32Array;
  outL: Float32Array;
  outR: Float32Array;
}

/**
 * The last `blocks` tap blocks (≤ 16), once the newest one has seq ≥ `minSeq`. The channels cross the protocol as
 * base64 Float32 bytes: 16 blocks of number arrays (~130 k values, each serialised on its own) take about a second.
 */
export async function packedTaps(page: Page, blocks: number, minSeq: number): Promise<PackedTap> {
  await page.waitForFunction((target) => (window.__anc!.lastTap()?.seq ?? 0) >= target, minSeq, { polling: 50 });
  const raw = await page.evaluate((n) => {
    const tap = window.__anc!.lastTap(n);
    if (tap === null) return null;
    const pack = (v: number[]): string => {
      const bytes = new Uint8Array(Float32Array.from(v).buffer);
      let s = '';
      for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      return btoa(s);
    };
    return { seq: tap.seq, frames: tap.frames, ref: pack(tap.ref), cond: pack(tap.cond), outL: pack(tap.outL), outR: pack(tap.outR) };
  }, blocks);
  if (raw === null) throw new Error('no tap blocks');
  expect(raw.frames).toBe(blocks * TAP_BLOCK_FRAMES);
  // A copy into a new buffer: Node may hand out a pooled Buffer whose offset is not a multiple of 4.
  const unpack = (s: string): Float32Array => new Float32Array(new Uint8Array(Buffer.from(s, 'base64')).buffer);
  return { seq: raw.seq, frames: raw.frames, ref: unpack(raw.ref), cond: unpack(raw.cond), outL: unpack(raw.outL), outR: unpack(raw.outR) };
}
