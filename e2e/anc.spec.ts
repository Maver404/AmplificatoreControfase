/**
 * End-to-end checks of the safety-critical paths in real Chrome with a fake microphone (see
 * playwright.config.ts): page load under the production CSP, exact zeros before «Attiva uscita»,
 * the DSP identity out = −g·cond[n − D] after the soft-start, PANIC timing, lifecycle auto-mutes and
 * preset sanitising. Every test also fails on console errors, CSP violations and on any fault,
 * processorerror or howl event (the guard in app-page.ts). Actions go through the real buttons and keys;
 * levels and values are read from the text readouts, samples from the tap blocks of window.__anc.
 */
import type { Page } from '@playwright/test';
import type { E2eTap } from '../src/app/e2e-hooks.ts';
import { TAP_BLOCK_FRAMES, TAP_CHANNELS } from '../src/audio/protocol.ts';
import { DEFAULT_PARAMS, D_MIN_SAMPLES } from '../src/dsp/params.ts';
import type { EngineParams } from '../src/dsp/params.ts';
import { formatNumber, parseNumber } from '../src/ui/units.ts';
import {
  allowMicrophone,
  enableOutput,
  expect,
  freshTaps,
  openApp,
  sampleRate,
  statusChip,
  statusMessage,
  strip,
  test,
} from './app-page.ts';
import { bestIntegerLag, dbToLin, fitAlignment, silenceAfterEscape, splitBlocks } from './helpers.ts';
import type { FixtureName, RecordedBlock } from './helpers.ts';

/** "Silent" output level and the PANIC deadline from the plan. */
const SILENCE = 1e-6;
const PANIC_DEADLINE_S = 0.05;
const MIN_CORRELATION = 0.99;

// ------------------------------------------------------------------------------ page helpers

const maxAbs = (v: readonly number[]): number => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);

/** Both outputs exactly zero in tap blocks rendered after the mute ramp. */
async function expectSilentOutputs(page: Page, liveReference: boolean): Promise<void> {
  await page.waitForTimeout(200);
  const tap = await freshTaps(page, 3);
  if (liveReference) expect(maxAbs(tap.ref), 'reference still live').toBeGreaterThan(1e-3);
  expect(maxAbs(tap.outL), 'outL').toBe(0);
  expect(maxAbs(tap.outR), 'outR').toBe(0);
}

interface AlignCase {
  ch: 'outL' | 'outR';
  delay: number;
  gain: number;
}

/**
 * Per tap block (blocks are analysed separately, so a block the processor dropped cannot fake a
 * mismatch): out[n] = −g·cond[n − D] with correlation > 0.99, the least-squares gain within 0.1 dB of
 * g and a residual below −40 dB. A one-sample delay error already leaves about −32 dB at 200 Hz.
 * With broadband noise the correlation peak must also sit at D (the tone's is periodic).
 */
function expectAlignment(tap: E2eTap, c: AlignCase, fixture: FixtureName): void {
  const conds = splitBlocks(tap.cond, TAP_BLOCK_FRAMES);
  const outs = splitBlocks(tap[c.ch], TAP_BLOCK_FRAMES);
  expect(outs.length).toBeGreaterThan(0);
  outs.forEach((out, i) => {
    const cond = conds[i]!;
    const fit = fitAlignment(cond, out, c.delay, -1);
    const label = `${c.ch}, block ${i}, D = ${c.delay.toFixed(3)} samples`;
    expect(fit.outRms, `${label}: output level`).toBeGreaterThan(5 * SILENCE);
    expect(fit.corr, `${label}: correlation`).toBeGreaterThan(MIN_CORRELATION);
    expect(Math.abs(20 * Math.log10(fit.gain / c.gain)), `${label}: gain error (dB)`).toBeLessThan(0.1);
    expect(fit.residualDb, `${label}: residual (dB)`).toBeLessThan(-40);
    if (fixture === 'noise') {
      const lag = bestIntegerLag(cond, out, 600, -1);
      expect(Math.abs(lag - c.delay), `${label}: correlation peak at lag ${lag}`).toBeLessThanOrEqual(1);
    }
  });
}

// ------------------------------------------------------------------------------ tests

test('the e2e build loads under the production CSP without console errors', async ({ page }) => {
  const res = await page.goto('/');
  expect(res?.status()).toBe(200);
  const csp = res?.headers()['content-security-policy'] ?? '';
  expect(csp).toContain("script-src 'self'");
  expect(csp).toContain("worker-src 'self'");
  // window.__anc exists only in the e2e build.
  await page.waitForFunction(() => window.__anc !== undefined);
  await expect(statusChip(page)).toHaveText('Fermo');
  // The guided setup does not open by itself here: no anti-noise lock.
  expect(await page.evaluate(() => window.__anc!.controller.state.ancLock)).toBeNull();
  await expect(page.locator('#output-gate').getByRole('button', { name: 'Attiva uscita' })).toBeDisabled();
  // Let the deferred start-up work (permission query, hooks, first frames) report any error.
  await page.waitForTimeout(500);
  // The AudioWorklet module (also under the CSP) loads with the microphone: covered by the tests below,
  // whose guard fails on any CSP violation.
});

test('«Consenti microfono» starts the audio context and the reference reads > −40 dBFS', async ({ page }) => {
  await openApp(page);
  expect(await page.evaluate(() => window.__anc!.engine.contextState)).toBe('none');
  await allowMicrophone(page);
  await expect(statusChip(page)).toHaveText('In ascolto · uscita bloccata');
  await expect.poll(() => page.evaluate(() => window.__anc!.engine.contextState)).toBe('running');
  const meter = page.locator('.meters-view .meter').filter({ hasText: 'Riferimento' });
  await expect.poll(async () => Number(await meter.getByRole('meter').getAttribute('aria-valuenow'))).toBeGreaterThan(-40);
  // The readout next to the bar: "RMS / picco" in dBFS.
  const text = (await meter.locator('.meter-value').textContent()) ?? '';
  const rms = parseNumber(text.split('/')[0] ?? '');
  expect(rms, `reference readout «${text}»`).not.toBeNull();
  expect(rms!).toBeGreaterThan(-40);
});

test('before «Attiva uscita» both outputs are exact zeros, and M does not arm', async ({ page }) => {
  await openApp(page);
  await allowMicrophone(page);
  let tap = await freshTaps(page, 4);
  expect(maxAbs(tap.ref), 'reference').toBeGreaterThan(1e-3);
  expect(maxAbs(tap.outL), 'outL').toBe(0);
  expect(maxAbs(tap.outR), 'outR').toBe(0);

  // Un-muting is not arming: the output stays at exact zeros.
  await page.keyboard.press('m');
  await expect(statusMessage(page)).toContainText('uscita non è attiva');
  expect(await page.evaluate(() => window.__anc!.controller.state.armed)).toBe(false);
  await expect(statusChip(page)).toHaveText('In ascolto · uscita bloccata');
  await page.waitForTimeout(100);
  tap = await freshTaps(page, 4);
  expect(maxAbs(tap.outL), 'outL after M').toBe(0);
  expect(maxAbs(tap.outR), 'outR after M').toBe(0);
});

test('after the soft-start each output is −g·cond[n − D], for D = D_MIN and 5.37 ms', async ({ page }) => {
  const fixture = test.info().project.name as FixtureName;
  await openApp(page);
  await allowMicrophone(page);
  await enableOutput(page);
  const fs = await sampleRate(page);
  const params: EngineParams = await page.evaluate(() => window.__anc!.controller.state.params);
  expect(params.channels.map((c) => c.polarityInvert)).toEqual([true, true]);
  const gainL = dbToLin(params.channels[0].gainDb + params.masterDb);
  const gainR = dbToLin(params.channels[1].gainDb + params.masterDb);
  // Channel-gain smoothing settles well within this.
  await page.waitForTimeout(300);

  let tap = await freshTaps(page, 3);
  expectAlignment(tap, { ch: 'outL', delay: D_MIN_SAMPLES, gain: gainL }, fixture);
  expectAlignment(tap, { ch: 'outR', delay: D_MIN_SAMPLES, gain: gainR }, fixture);

  // 5.37 ms typed in channel L's delay field (Italian decimal comma); R stays at D_MIN.
  const field = strip(page, 'L').getByRole('textbox', { name: 'Ritardo, valore in ms' });
  await field.fill('5,37');
  await field.press('Enter');
  const delayL = D_MIN_SAMPLES + (5.37 / 1000) * fs;
  await expect
    .poll(() => page.evaluate(() => window.__anc!.controller.state.meters?.delaySamples[0] ?? 0))
    .toBeCloseTo(delayL, 6);
  await expect(strip(page, 'L').locator('.strip-actual')).toContainText(`${formatNumber(delayL, 3)} campioni`);
  // 30 ms crossfade to the new read position, plus the block in flight.
  await page.waitForTimeout(300);

  tap = await freshTaps(page, 3);
  expectAlignment(tap, { ch: 'outL', delay: delayL, gain: gainL }, fixture);
  expectAlignment(tap, { ch: 'outR', delay: D_MIN_SAMPLES, gain: gainR }, fixture);
});

test('Esc silences both outputs (< 1e-6) within 50 ms and shows PANIC', async ({ page }) => {
  await openApp(page);
  await allowMicrophone(page);
  await enableOutput(page);
  const fs = await sampleRate(page);

  await page.evaluate(
    ([l, r]) => {
      const rec: RecordedBlock[] = [];
      window.__rec = rec;
      window.__anc!.controller.onTap((block, frames) => {
        rec.push({
          t: performance.now(),
          frames,
          outL: Array.from(block.subarray(l * frames, (l + 1) * frames)),
          outR: Array.from(block.subarray(r * frames, (r + 1) * frames)),
        });
      });
    },
    [TAP_CHANNELS.indexOf('outL'), TAP_CHANNELS.indexOf('outR')] as const,
  );
  await page.waitForFunction(() => (window.__rec?.length ?? 0) >= 8);

  await page.keyboard.press('Escape');
  await expect(statusChip(page)).toHaveText('PANIC · silenziato');
  await expect(statusMessage(page)).toContainText('PANIC');

  // PANIC suspends the audio context after the 10 ms ramp. Resuming it (still muted) completes and
  // delivers the tap block that was being filled, so the stream covers the whole deadline.
  await expect.poll(() => page.evaluate(() => window.__anc!.engine.contextState)).toBe('suspended');
  const delivered = await page.evaluate(() => window.__rec!.length);
  await page.evaluate(() => window.__anc!.engine.resume());
  await page.waitForFunction((n) => window.__rec!.length >= n, delivered + 4);

  const { blocks, esc } = await page.evaluate(() => ({ blocks: window.__rec!, esc: window.__e2e!.esc }));
  expect(esc).toHaveLength(1);
  const r = silenceAfterEscape(blocks, esc[0]!, fs, SILENCE);
  const deadline = r.escSample + PANIC_DEADLINE_S * fs + TAP_BLOCK_FRAMES;
  expect(r.peakBefore, 'output playing before Esc').toBeGreaterThan(5 * SILENCE);
  expect(r.total, 'recording covers the deadline').toBeGreaterThan(deadline + TAP_BLOCK_FRAMES);
  const silentAfterMs = ((r.lastLoud + 1 - r.escSample) / fs) * 1000;
  test.info().annotations.push({ type: 'panic', description: `outputs < 1e-6 ${silentAfterMs.toFixed(1)} ms after Esc` });
  expect(r.lastLoud, `outputs silent ${silentAfterMs.toFixed(1)} ms after Esc`).toBeLessThan(deadline);

  // Still PANIC (muted) after the resume: only «Attiva uscita», Avvio or M leave it.
  await expect(statusChip(page)).toHaveText('PANIC · silenziato');
  expect(await page.evaluate(() => window.__anc!.controller.state.muted)).toBe(true);
});

test('a devicechange event auto-mutes the output with a message', async ({ page }) => {
  await openApp(page);
  await allowMicrophone(page);
  await enableOutput(page);
  await page.evaluate(() => navigator.mediaDevices.dispatchEvent(new Event('devicechange')));
  await expect(statusChip(page)).toHaveText('Muto');
  await expect(statusMessage(page)).toContainText('I dispositivi audio sono cambiati: uscita silenziata');
  expect(await page.evaluate(() => window.__anc!.controller.state.muted)).toBe(true);
  await expectSilentOutputs(page, true);
});

test('stopping the microphone track auto-mutes the output with a message', async ({ page }) => {
  await openApp(page);
  await allowMicrophone(page);
  await enableOutput(page);
  await page.evaluate(() => window.__anc!.engine.mediaTrack?.stop());
  await expect(statusMessage(page)).toContainText('Il microfono si è scollegato o è stato chiuso: uscita silenziata');
  await expect(statusChip(page)).toHaveText('Errore');
  const s = await page.evaluate(() => {
    const st = window.__anc!.controller.state;
    return { muted: st.muted, armed: st.armed };
  });
  expect(s).toEqual({ muted: true, armed: false });
  await expectSilentOutputs(page, false);
});

test('an imported out-of-range preset is clamped when loaded (0 dB, +12 dB unlock off)', async ({ page }) => {
  await openApp(page);
  await allowMicrophone(page);
  await enableOutput(page);

  const hot = structuredClone(DEFAULT_PARAMS);
  hot.channels[0].gainDb = 12;
  hot.channels[1].gainDb = 12;
  hot.ceilingDb = -1;
  hot.gainUnlocked = true;
  const file = {
    format: 'controfase-presets',
    version: 1,
    presets: [{ id: 'e2e-fuori-limiti', name: 'E2E fuori limiti', createdAt: Date.now(), deviceKey: '', sampleRate: 48000, params: hot }],
  };
  const presets = page.locator('#presets');
  const chooser = page.waitForEvent('filechooser');
  await presets.getByRole('button', { name: 'Importa da file…' }).click();
  await (await chooser).setFiles({
    name: 'preset-fuori-limiti.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(file)),
  });
  await expect(presets.locator('.pp-message')).toContainText('Importati 1 preset');
  await presets.getByRole('button', { name: 'Carica il preset E2E fuori limiti' }).click();
  await expect(presets.locator('.pp-message')).toContainText('Preset «E2E fuori limiti» caricato');

  // Readouts: channel gains at 0 dB, ceiling −1 dBFS (in range), unlock off.
  for (const ch of ['L', 'R'] as const) {
    const gain = strip(page, ch).getByRole('textbox', { name: 'Guadagno, valore in dB' });
    await expect.poll(async () => parseNumber(await gain.inputValue()), { message: `gain ${ch}` }).toBe(0);
  }
  const gate = page.locator('#output-gate');
  const ceiling = gate.getByRole('textbox', { name: 'Limite d’uscita (ceiling del limiter), valore in dBFS' });
  await expect.poll(async () => parseNumber(await ceiling.inputValue())).toBe(-1);
  await expect(
    gate.getByRole('switch', { name: 'Sblocca il guadagno dei canali fino a +12 dB', includeHidden: true }),
  ).toHaveAttribute('aria-checked', 'false');
  const p: EngineParams = await page.evaluate(() => window.__anc!.controller.state.params);
  expect(p.gainUnlocked).toBe(false);
  expect(p.channels.map((c) => c.gainDb)).toEqual([0, 0]);
  expect(p.ceilingDb).toBe(-1);

  // The processor really runs at 0 dB: after the soft-start and the ≤ 6 dB/s rise the output gain
  // converges to master + 0 dB (never to +12 dB).
  await expect(statusChip(page)).toHaveText('Attivo');
  const expected = dbToLin(p.masterDb);
  await expect
    .poll(
      async () => {
        const tap = await freshTaps(page, 1);
        return 20 * Math.log10(fitAlignment(tap.cond, tap.outL, D_MIN_SAMPLES, -1).gain / expected);
      },
      { message: 'output gain relative to master + 0 dB', timeout: 15_000, intervals: [250] },
    )
    .toBeCloseTo(0, 1);
});
