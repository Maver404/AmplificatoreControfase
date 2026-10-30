/**
 * Harmonic trims (ChannelParams.harmonicTrims) in real Chrome, through the AudioWorklet and the strip's
 * «Armoniche» controls, on harmonic-100hz.wav (100 Hz + harmonics 2–4; playwright.config.ts runs this spec on
 * that fixture only). The output is armed in tonal mode the way a user does it («Consenti microfono», «Tonale»,
 * «Attiva uscita»; the defaults are f0 100 Hz, 4 harmonics), trims are typed in the number fields or stepped with
 * the keyboard on the sliders.
 *
 * Measurement: per harmonic h, the lock-in at h·f0 over whole periods of the tap channels gives each output's
 * transfer Out(h·f0)/Cond(h·f0) and the conditioning's Cond(h·f0)/Ref(h·f0). A trim must multiply its harmonic's
 * transfer by 10^(g/20)·e^{−jθπ/180} and leave every other harmonic, the other channel and the 'cond' tap alone
 * (and, outside tonal mode, act nowhere).
 * A window counts only once its two halves agree (≤ STEADY_DB / STEADY_DEG): that waits out the soft-start,
 * trim glides and 6 dB/s rises instead of fixed sleeps, and only windows that start SETTLE_BLOCKS after a change
 * count. Chrome's fake device zero-fills a few ms where it loops the WAV, and the bank rings for a second after
 * that: the fixture is 30 s long (global-setup.ts), so no test reaches its loop.
 *
 * f0 following is switched off through the controller (it has no UI): it retunes the bank to every 2 s f0
 * estimate, and an estimate over a loop glitch (or the noise of a real one) moves every transfer by ~0.005 dB.
 */
import type { Locator, Page } from '@playwright/test';
import { TAP_BLOCK_FRAMES } from '../src/audio/protocol.ts';
import { D_MIN_SAMPLES, MAX_HARMONICS } from '../src/dsp/params.ts';
import type { EngineParams, HarmonicTrim } from '../src/dsp/params.ts';
import { formatAdaptive, formatHz, parseNumber } from '../src/ui/units.ts';
import { allowMicrophone, enableOutput, expect, openApp, packedTaps, sampleRate, statusChip, strip, test } from './app-page.ts';
import type { PackedTap } from './app-page.ts';
import { HARMONIC_F0, HARMONIC_PARTIALS, asTrim, divide, lockIn, magnitudeDb, wholePeriods, wrapDeg } from './helpers.ts';
import type { Phasor } from './helpers.ts';

const HARMONICS = HARMONIC_PARTIALS.length;
const HS = Array.from({ length: HARMONICS }, (_, i) => i + 1);

/**
 * Transfer tolerance. Measured errors against the expected factors: ≤ 2e-7 dB / 1e-6° in clean windows (the
 * worklet is deterministic and the fake device's tone exactly periodic), ≤ 9e-4 dB / 3e-3° in the worst window
 * the halves test admitted (bank ringing after a loop glitch), which bounds residual transients at 0.002 dB /
 * 0.01°. 0.01 dB / 0.05° is 5× that bound; a 1 % gain (0.09 dB) or a 0.5° phase error in a trim fails.
 */
const TOL_DB = 0.01;
const TOL_DEG = 0.05;
/** Largest difference between the two halves of a window that still counts as steady. */
const STEADY_DB = 0.002;
const STEADY_DEG = 0.01;
/** Covers the slowest case here: an 18 dB trim rise, 12 dB of it at 6 dB/s. */
const STEADY_TIMEOUT_MS = 15_000;
/** 16 blocks of 2048 frames: 74 periods of 100 Hz at 44.1 kHz. */
const WINDOW_BLOCKS = 16;
/** Blocks left out after a change: the one being rendered and one possibly rendered but not yet delivered. */
const SETTLE_BLOCKS = 2;
/**
 * Rise test: 4-block windows (18 periods, 0.19 s) watched for 6.5 s. Windows whose Cond/Ref halves disagree (a
 * glitch of the fake device: the fixture's loop is out of reach, see the file comment) are skipped. Time is
 * counted in tap blocks, so a block the engine dropped would make the rise look 46 ms early: the level bound has
 * 6 dB/s × 46 ms of slack. +6 dB must arrive by RISE_TOP_BY_S: 30 dB at 6 dB/s is 5.02 s, and block counting
 * adds at most two blocks; a trim that climbed its last 6 dB at 6 dB/s too would arrive at ≈ 6 s.
 */
const RISE_BLOCKS = 4;
const RISE_WATCH_S = 6.5;
const RISE_SLACK_DB = 0.3;
const RISE_TOP_BY_S = 5.5;
/**
 * Drag test levels: master 0 dB and channel L at −6 dB put the output (≈ −26 dBFS peak) above the howl detector's
 * −40 dBFS floor, so «no howl» is a real check (at the defaults, −50 dB from cond, it could not fail).
 */
const LOUD_MASTER_DB = 0;
const LOUD_GAIN_DB = -6;
/** The fast budget rise of the drag: PageUp (1 dB) this many times, this far apart (+6 dB in 0.3 s, 20 dB/s). */
const BUDGET_STEPS = 6;
const BUDGET_STEP_MS = 50;

type Out = 'L' | 'R';

interface Transfer {
  seq: number;
  /** Out(h·f0)/Cond(h·f0) of each output, index h − 1. */
  L: Phasor[];
  R: Phasor[];
  /** Cond(h·f0)/Ref(h·f0): the neutral conditioning chain, index h − 1. */
  cond: Phasor[];
}

// ------------------------------------------------------------------------------ measurement

const trimFactor = (gainDb: number, phaseDeg: number): Phasor => {
  const a = Math.pow(10, gainDb / 20);
  const ph = (-phaseDeg * Math.PI) / 180;
  return { re: a * Math.cos(ph), im: a * Math.sin(ph) };
};

const ONE: Phasor = { re: 1, im: 0 };

/** a/b as a gain (dB) and a phase difference (degrees): 0 / 0 when a = b. */
function deviation(a: Phasor, b: Phasor): { db: number; deg: number } {
  const r = divide(a, b);
  return { db: magnitudeDb(r), deg: wrapDeg((Math.atan2(r.im, r.re) * 180) / Math.PI) };
}

function transferOf(tap: PackedTap, fs: number, start: number, n: number): Omit<Transfer, 'seq'> {
  const at = (x: Float32Array, h: number): Phasor => lockIn(x, h * HARMONIC_F0, fs, start, n);
  return {
    L: HS.map((h) => divide(at(tap.outL, h), at(tap.cond, h))),
    R: HS.map((h) => divide(at(tap.outR, h), at(tap.cond, h))),
    cond: HS.map((h) => divide(at(tap.cond, h), at(tap.ref, h))),
  };
}

/** Tap seq a window must reach to start SETTLE_BLOCKS after whatever the page has just sent to the engine. */
async function windowAfterNow(page: Page): Promise<number> {
  const seq = await page.evaluate(() => window.__anc!.lastTap()?.seq ?? 0);
  return seq + SETTLE_BLOCKS + WINDOW_BLOCKS;
}

/**
 * Largest difference between the `keys` transfers of the first and the second half of the window (whole periods
 * each). Unlike a sample-by-sample periodicity test it does not depend on the fake device's clock matching the
 * context's exactly.
 */
function halvesDeviation(
  tap: PackedTap,
  fs: number,
  w: { start: number; n: number },
  keys: ReadonlyArray<'L' | 'R' | 'cond'>,
): { db: number; deg: number } {
  const half = wholePeriods(Math.floor(w.n / 2), fs, HARMONIC_F0).n;
  const a = transferOf(tap, fs, w.start, half);
  const b = transferOf(tap, fs, w.start + w.n - half, half);
  const worst = { db: 0, deg: 0 };
  for (const k of keys) {
    a[k].forEach((z, i) => {
      const d = deviation(z, b[k][i]!);
      worst.db = Math.max(worst.db, Math.abs(d.db));
      worst.deg = Math.max(worst.deg, Math.abs(d.deg));
    });
  }
  return worst;
}

const isSteady = (d: { db: number; deg: number }): boolean => d.db <= STEADY_DB && d.deg <= STEADY_DEG;

/** The transfers of the first steady window ending at tap seq ≥ minSeq (see the file comment). */
async function steadyTransfer(page: Page, fs: number, minSeq: number, what: string): Promise<Transfer> {
  const deadline = Date.now() + STEADY_TIMEOUT_MS;
  let next = minSeq;
  for (;;) {
    const tap = await packedTaps(page, WINDOW_BLOCKS, next);
    const w = wholePeriods(tap.frames, fs, HARMONIC_F0);
    const { db, deg } = halvesDeviation(tap, fs, w, ['L', 'R', 'cond']);
    if (isSteady({ db, deg })) return { seq: tap.seq, ...transferOf(tap, fs, w.start, w.n) };
    // A silenced output never settles: say why at once (an 'underruns' auto-mute means a loaded machine).
    const stopped = await playState(page);
    if (stopped.armed !== true || stopped.muted || stopped.stops.length > 0) {
      throw new Error(`${what}: the output stopped while waiting for a steady window: ${JSON.stringify(stopped)}`);
    }
    if (Date.now() > deadline) {
      throw new Error(
        `${what}: no steady window within ${STEADY_TIMEOUT_MS / 1000} s ` +
          `(last one: halves differ by ${db.toFixed(4)} dB / ${deg.toFixed(3)}°)`,
      );
    }
    next = tap.seq + WINDOW_BLOCKS;
  }
}

/** Largest |error| over the checks so far, reported as an annotation (the run's measured accuracy). */
class Worst {
  db = 0;
  deg = 0;
  add(d: { db: number; deg: number }): void {
    this.db = Math.max(this.db, Math.abs(d.db));
    this.deg = Math.max(this.deg, Math.abs(d.deg));
  }
  annotate(): void {
    const description = `≤ ${this.db.toExponential(2)} dB, ≤ ${this.deg.toExponential(2)}°`;
    test.info().annotations.push({ type: 'transfer error', description });
  }
}

/**
 * Each output's transfer changed from `base` exactly by its trims (`trims[ch][h − 1]`, missing = none), and the
 * 'cond' tap not at all.
 */
function expectTrimmed(
  after: Transfer,
  base: Transfer,
  trims: Partial<Record<Out, Record<number, [number, number]>>>,
  worst: Worst,
  what: string,
): void {
  for (const ch of ['L', 'R'] as const) {
    for (const h of HS) {
      const t = trims[ch]?.[h];
      const want = t ? trimFactor(t[0], t[1]) : ONE;
      const d = deviation(divide(after[ch][h - 1]!, base[ch][h - 1]!), want);
      worst.add(d);
      const label = `${what}: out${ch} at ${h * HARMONIC_F0} Hz, ${t ? `trim ${t[0]} dB / ${t[1]}°` : 'untrimmed'}`;
      expect(Math.abs(d.db), `${label}: gain error (dB)`).toBeLessThan(TOL_DB);
      expect(Math.abs(d.deg), `${label}: phase error (°)`).toBeLessThan(TOL_DEG);
    }
  }
  for (const h of HS) {
    const d = deviation(after.cond[h - 1]!, base.cond[h - 1]!);
    worst.add(d);
    expect(Math.abs(d.db), `${what}: cond tap at ${h * HARMONIC_F0} Hz, change (dB)`).toBeLessThan(TOL_DB);
    expect(Math.abs(d.deg), `${what}: cond tap at ${h * HARMONIC_F0} Hz, change (°)`).toBeLessThan(TOL_DEG);
  }
}

/**
 * Both outputs as the untrimmed model: −g·e^{−jω·D_MIN} at every harmonic, g = channel gain + master (no base
 * delay and Δ = 0, so the tap delay is D_MIN in tonal and in delay mode).
 */
function expectUntrimmed(t: Transfer, p: EngineParams, fs: number, worst: Worst, what: string): void {
  for (const ch of ['L', 'R'] as const) {
    const c = p.channels[ch === 'L' ? 0 : 1];
    for (const h of HS) {
      const lag = 180 + (360 * h * HARMONIC_F0 * D_MIN_SAMPLES) / fs;
      const d = deviation(t[ch][h - 1]!, trimFactor(c.gainDb + p.masterDb, lag));
      worst.add(d);
      const label = `${what}: out${ch} at ${h * HARMONIC_F0} Hz vs ${c.gainDb + p.masterDb} dB, ${lag.toFixed(3)}° lag`;
      expect(Math.abs(d.db), `${label}: gain error (dB)`).toBeLessThan(TOL_DB);
      expect(Math.abs(d.deg), `${label}: phase error (°)`).toBeLessThan(TOL_DEG);
    }
  }
}

// ------------------------------------------------------------------------------ page helpers

/**
 * Opens the app, switches to tonal mode and arms the output; returns the sample rate. `before`: done through the
 * UI before «Attiva uscita» (e.g. the levels).
 */
async function armTonal(page: Page, before?: () => Promise<void>): Promise<number> {
  await openApp(page);
  // No UI for it (see the file comment).
  await page.evaluate(() => window.__anc!.controller.setF0Follow(false));
  await allowMicrophone(page);
  await page.locator('#filters').getByRole('radio', { name: 'Tonale' }).check();
  const p = await page.evaluate(() => window.__anc!.controller.state.params);
  expect(p.phaseMode).toBe('tonal');
  expect(p.tonal.f0, 'default f0 = the fixture fundamental').toBe(HARMONIC_F0);
  expect(p.tonal.harmonics, 'default harmonics = the fixture partials').toBe(HARMONICS);
  await before?.();
  await enableOutput(page);
  return sampleRate(page);
}

/** Expands the strip's «Armoniche» section and returns it. */
async function openHarmonics(page: Page, ch: Out): Promise<Locator> {
  const details = strip(page, ch).locator('details.strip-harm');
  await details.locator('summary').click();
  await expect(details).toHaveAttribute('open', '');
  return details;
}

/** Types `text` in a precision control's number field and commits it with Enter. */
async function typeValue(scope: Locator, field: string, text: string): Promise<void> {
  const box = scope.getByRole('textbox', { name: field });
  await box.fill(text);
  await box.press('Enter');
}

/** Gain and phase of harmonic h typed in the strip's fields. */
async function typeTrim(harm: Locator, h: number, gainDb: number, phaseDeg: number): Promise<void> {
  await typeValue(harm, `Guadagno armonica ${h}, valore in dB`, String(gainDb));
  await typeValue(harm, `Fase armonica ${h}, valore in °`, String(phaseDeg));
}

async function pressRepeat(page: Page, key: string, times: number): Promise<void> {
  for (let i = 0; i < times; i++) await page.keyboard.press(key);
}

/** All MAX_HARMONICS trims of channel `ch` as the controller holds them equal `want` (h → [gainDb, phaseDeg]). */
async function expectTrims(page: Page, ch: Out, want: Record<number, [number, number]>): Promise<void> {
  const expected: HarmonicTrim[] = Array.from({ length: MAX_HARMONICS }, (_, i) => {
    const t = want[i + 1];
    return { gainDb: t ? t[0] : 0, phaseDeg: t ? t[1] : 0 };
  });
  const idx = ch === 'L' ? 0 : 1;
  await expect
    .poll(
      async () => {
        const trims = await page.evaluate((c) => window.__anc!.controller.state.params.channels[c].harmonicTrims, idx);
        // Steps are rounded in display space (units.ts cleanFloat); compare at 1e-9.
        return trims.map((t) => ({ gainDb: Math.round(t.gainDb * 1e9) / 1e9, phaseDeg: Math.round(t.phaseDeg * 1e9) / 1e9 }));
      },
      { message: `trims of channel ${ch}` },
    )
    .toEqual(expected);
}

/** Output state and every howl or auto-mute (engine or lifecycle) event so far. */
function playState(page: Page): Promise<{ armed: boolean; muted: boolean; panic: boolean; stops: string[] }> {
  return page.evaluate(() => {
    const a = window.__anc!;
    const st = a.controller.state;
    const autoMuted = (d: unknown): boolean => (d as { autoMuted?: boolean } | undefined)?.autoMuted === true;
    const stops = a.events.filter(
      (e) => e.source === 'engine' && (e.type === 'howl' || e.type === 'autoMute' || (e.type === 'lifecycle' && autoMuted(e.detail))),
    );
    return { armed: st.armed, muted: st.muted, panic: st.panic, stops: stops.map((e) => `${e.type} ${JSON.stringify(e.detail)}`) };
  });
}

/** No howl or auto-mute so far; the output is still armed and playing. */
async function expectStillPlaying(page: Page): Promise<void> {
  const s = await playState(page);
  expect(s.stops, 'howl / auto-mute events').toEqual([]);
  expect({ armed: s.armed, muted: s.muted, panic: s.panic }).toEqual({ armed: true, muted: false, panic: false });
  await expect(statusChip(page)).toHaveText('Attivo');
}

/** The number of a «Fase totale» readout ("… 220° …"). */
function phaseOf(text: string | null): number {
  const m = /(−?[\d,]+)°/.exec(text ?? '');
  const v = m ? parseNumber(m[1]!) : null;
  if (v === null) throw new Error(`no phase in «${text}»`);
  return v;
}

// ------------------------------------------------------------------------------ tests

test('typed and stepped trims change only their harmonic, by 10^(g/20)·e^(−jθ), leave the cond tap alone and act only in tonal mode', async ({ page }) => {
  const fs = await armTonal(page);
  const worst = new Worst();
  const base = await steadyTransfer(page, fs, await windowAfterNow(page), 'untrimmed');

  const p = await page.evaluate(() => window.__anc!.controller.state.params);
  expectUntrimmed(base, p, fs, worst, 'untrimmed');

  // Fundamental typed in the number fields; harmonic 3 stepped on the sliders (PageUp/PageDown = 100 fine steps).
  const harm = await openHarmonics(page, 'L');
  await typeTrim(harm, 1, -6, 30);
  await harm.getByRole('slider', { name: 'Guadagno armonica 3' }).focus();
  await pressRepeat(page, 'PageUp', 3);
  await harm.getByRole('slider', { name: 'Fase armonica 3' }).focus();
  await pressRepeat(page, 'PageDown', 6);
  await expectTrims(page, 'L', { 1: [-6, 30], 3: [3, -60] });
  await expectTrims(page, 'R', {});
  await expect(harm.locator('summary')).toHaveText('Armoniche · 2 corrette');
  expect(parseNumber(await harm.getByRole('textbox', { name: 'Guadagno armonica 3, valore in dB' }).inputValue())).toBe(3);
  expect(parseNumber(await harm.getByRole('textbox', { name: 'Fase armonica 3, valore in °' }).inputValue())).toBe(-60);

  const after = await steadyTransfer(page, fs, await windowAfterNow(page), 'trimmed');
  expectTrimmed(after, base, { L: { 1: [-6, 30], 3: [3, -60] } }, worst, 'h1 −6 dB/+30°, h3 +3 dB/−60° on L');

  // Back in «Ritardo» mode the trims stay in the params but act nowhere: «Armoniche» is hidden and both outputs
  // are −g·cond[n − D_MIN] again at every harmonic.
  await page.locator('#filters').getByRole('radio', { name: 'Ritardo' }).check();
  await expect(harm).toBeHidden();
  await expectTrims(page, 'L', { 1: [-6, 30], 3: [3, -60] });
  const delayMode = await steadyTransfer(page, fs, await windowAfterNow(page), 'delay mode');
  expectUntrimmed(delayMode, await page.evaluate(() => window.__anc!.controller.state.params), fs, worst, 'delay mode, trims set');
  await expectStillPlaying(page);
  worst.annotate();
});

test('a fast keyboard drag of the fundamental\'s trim never howls or auto-mutes, and the level ends on the target', async ({ page }) => {
  // Loud enough for the howl detector to judge it (LOUD_MASTER_DB, LOUD_GAIN_DB), set in the UI before arming.
  const fs = await armTonal(page, async () => {
    await typeValue(page.locator('.status-bar'), 'Volume master, valore in dB', String(LOUD_MASTER_DB));
    await typeValue(strip(page, 'L'), 'Guadagno, valore in dB', String(LOUD_GAIN_DB));
  });
  const p = await page.evaluate(() => window.__anc!.controller.state.params);
  expect([p.masterDb, p.channels[0].gainDb]).toEqual([LOUD_MASTER_DB, LOUD_GAIN_DB]);
  const worst = new Worst();
  const base = await steadyTransfer(page, fs, await windowAfterNow(page), 'untrimmed');
  const harm = await openHarmonics(page, 'L');
  const gain = harm.getByRole('slider', { name: 'Guadagno armonica 1' });
  const phase = harm.getByRole('slider', { name: 'Fase armonica 1' });

  // The fundamental carries most of the output. +6 dB from the fast-rise budget in 0.3 s (PageUp, 1 dB, every
  // 50 ms: the hand-over the engine holds the howl detector's levels for, a 'peak-growth' without it) and back;
  // +6 dB in 0.1 dB steps (Maiusc) over about half a second (the drag that once looked like a 'peak-growth'
  // howl), the phase swung +90°, down to −14 dB, a held-key run of fine steps up and back, the phase to −30°, then
  // up 18 dB to +4 dB at once (12 dB beyond the fast-rise budget: 6 dB/s). Every key press is a parameter change
  // sent to the engine.
  const t0 = Date.now();
  await gain.focus();
  for (let i = 0; i < BUDGET_STEPS; i++) {
    await page.keyboard.press('PageUp');
    await page.waitForTimeout(BUDGET_STEP_MS);
  }
  await pressRepeat(page, 'PageDown', BUDGET_STEPS);
  await pressRepeat(page, 'Shift+ArrowUp', 60);
  await phase.focus();
  await pressRepeat(page, 'PageUp', 9);
  await gain.focus();
  await pressRepeat(page, 'PageDown', 20);
  await pressRepeat(page, 'ArrowUp', 50);
  await pressRepeat(page, 'ArrowDown', 50);
  await phase.focus();
  await pressRepeat(page, 'PageDown', 12);
  await gain.focus();
  await pressRepeat(page, 'PageUp', 18);
  const t1 = Date.now();
  await expectTrims(page, 'L', { 1: [4, -30] });
  await expectStillPlaying(page);

  const after = await steadyTransfer(page, fs, await windowAfterNow(page), 'after the drag');
  test.info().annotations.push({ type: 'drag', description: `231 key presses in ${t1 - t0} ms, steady ${Date.now() - t1} ms later` });
  expectTrimmed(after, base, { L: { 1: [4, -30] } }, worst, 'h1 dragged to +4 dB/−30° on L');
  await expectStillPlaying(page);
  worst.annotate();
});

test('a −30 → +6 dB trim rise reaches the output at 6 dB/s, the last 6 dB from the budget, without a howl', async ({ page }) => {
  const fs = await armTonal(page);
  const worst = new Worst();
  const base = await steadyTransfer(page, fs, await windowAfterNow(page), 'untrimmed');
  const harm = await openHarmonics(page, 'L');
  const box = harm.getByRole('textbox', { name: 'Guadagno armonica 1, valore in dB' });
  await box.fill('-30');
  await box.press('Enter');
  await expectTrims(page, 'L', { 1: [-30, 0] });
  const low = await steadyTransfer(page, fs, await windowAfterNow(page), 'h1 at −30 dB');
  expectTrimmed(low, base, { L: { 1: [-30, 0] } }, worst, 'h1 at −30 dB');

  // Every block delivered so far was rendered before the change: the rise starts after the end of block seq0.
  await box.fill('6');
  const seq0 = await page.evaluate(() => window.__anc!.lastTap()?.seq ?? 0);
  await box.press('Enter');
  await expectTrims(page, 'L', { 1: [6, 0] });
  const blockS = TAP_BLOCK_FRAMES / fs;
  const ref = Math.hypot(base.L[0]!.re, base.L[0]!.im);
  /** The fundamental's trim as played: |Out/Cond| at f0 relative to the untrimmed one, dB. */
  const levels: Array<{ start: number; end: number; db: number }> = [];
  let skipped = 0;
  let next = seq0 + RISE_BLOCKS;
  for (let end = 0; end < RISE_WATCH_S; ) {
    const tap = await packedTaps(page, RISE_BLOCKS, next);
    next = tap.seq + RISE_BLOCKS;
    // Window times in s after the end of block seq0 (no earlier than the change).
    end = (tap.seq - seq0) * blockS;
    const w = wholePeriods(tap.frames, fs, HARMONIC_F0);
    if (!isSteady(halvesDeviation(tap, fs, w, ['cond']))) {
      skipped++;
      continue;
    }
    const out = lockIn(tap.outL, HARMONIC_F0, fs, w.start, w.n);
    const cond = lockIn(tap.cond, HARMONIC_F0, fs, w.start, w.n);
    levels.push({ start: end - w.n / fs, end, db: magnitudeDb(divide(out, cond)) - 20 * Math.log10(ref) });
  }
  const trace = `${levels.map((l) => `${l.end.toFixed(2)} s ${l.db.toFixed(3)} dB`).join(', ')}; ${skipped} skipped`;
  // Never above −30 + 6 dB + 6 dB/s since the change (a window's mean is below its end's bound).
  for (const l of levels) {
    const bound = -30 + 6 + 6 * l.end + RISE_SLACK_DB;
    expect(l.db, `level at ${l.end.toFixed(2)} s vs the BIG_JUMP_DB + 6 dB/s bound (${trace})`).toBeLessThan(bound);
    expect(l.db, `level at ${l.end.toFixed(2)} s (${trace})`).toBeLessThan(6 + TOL_DB);
  }
  levels.forEach((l, i) => {
    if (i > 0) expect(l.db, `level falls at ${l.end.toFixed(2)} s (${trace})`).toBeGreaterThan(levels[i - 1]!.db - 0.02);
  });
  // The slow part: 6 dB/s (block counting can only shorten the time, by a block per dropped tap block).
  const ramp = levels.filter((l) => l.db > -27 && l.db < -3);
  expect(ramp.length, `windows inside the ramp (${trace})`).toBeGreaterThan(5);
  const first = ramp[0]!;
  const last = ramp[ramp.length - 1]!;
  const slope = (last.db - first.db) / (last.end - first.end);
  expect(slope, `rise rate, dB/s (${trace})`).toBeGreaterThan(5.7);
  expect(slope, `rise rate, dB/s (${trace})`).toBeLessThan(6.3);
  // +6 dB is reached once 30 dB have risen at 6 dB/s (5 s), not before.
  const top = levels.find((l) => l.db > 6 - TOL_DB);
  expect(top, `+6 dB reached within ${RISE_WATCH_S} s (${trace})`).toBeDefined();
  expect(top!.start, `first window at +6 dB starts after the change (${trace})`).toBeGreaterThan(5 - 2 * blockS);
  // …and right then: the last 6 dB come from the budget at once, not at 6 dB/s.
  expect(top!.start, `first window at +6 dB starts by ${RISE_TOP_BY_S} s (${trace})`).toBeLessThan(RISE_TOP_BY_S);
  const description = `${slope.toFixed(3)} dB/s; +6 dB in the window from ${top!.start.toFixed(2)} s`;
  test.info().annotations.push({ type: 'rise', description });

  const after = await steadyTransfer(page, fs, await windowAfterNow(page), 'h1 at +6 dB');
  expectTrimmed(after, base, { L: { 1: [6, 0] } }, worst, 'h1 at +6 dB');
  await expectStillPlaying(page);
  worst.annotate();
});

test('«Collega L/R» moves both channels\' trims; «Azzera armoniche» clears both while linked, only its own channel unlinked', async ({ page }) => {
  const fs = await armTonal(page);
  const worst = new Worst();
  const base = await steadyTransfer(page, fs, await windowAfterNow(page), 'untrimmed');
  const harmL = await openHarmonics(page, 'L');
  const harmR = await openHarmonics(page, 'R');
  const linkL = strip(page, 'L').getByRole('button', { name: 'Collega L/R' });
  const linkR = strip(page, 'R').getByRole('button', { name: 'Collega L/R' });
  await expect(linkL).toHaveAttribute('aria-pressed', 'false');

  // Unlinked: each channel gets its own trims.
  await typeTrim(harmL, 1, -6, 30);
  await typeTrim(harmL, 3, 3, -60);
  await typeTrim(harmR, 2, -4, 45);
  await typeTrim(harmR, 4, 2, -20);
  await expectTrims(page, 'L', { 1: [-6, 30], 3: [3, -60] });
  await expectTrims(page, 'R', { 2: [-4, 45], 4: [2, -20] });
  let t = await steadyTransfer(page, fs, await windowAfterNow(page), 'both trimmed');
  expectTrimmed(t, base, { L: { 1: [-6, 30], 3: [3, -60] }, R: { 2: [-4, 45], 4: [2, -20] } }, worst, 'both trimmed');

  // Linked: an edit on L moves R's entry by the same amount (offsets kept); «Azzera armoniche» on L names and
  // clears both channels.
  await linkL.click();
  await expect(linkL).toHaveAttribute('aria-pressed', 'true');
  await expect(linkR).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => window.__anc!.controller.state.linked)).toBe(true);
  await typeValue(harmL, 'Guadagno armonica 2, valore in dB', '-3');
  await expectTrims(page, 'L', { 1: [-6, 30], 2: [-3, 0], 3: [3, -60] });
  await expectTrims(page, 'R', { 2: [-7, 45], 4: [2, -20] });
  t = await steadyTransfer(page, fs, await windowAfterNow(page), 'linked edit');
  expectTrimmed(t, base, { L: { 1: [-6, 30], 2: [-3, 0], 3: [3, -60] }, R: { 2: [-7, 45], 4: [2, -20] } }, worst, 'linked edit');
  const both = 'Azzera le correzioni delle armoniche dei canali L e R (collegati)';
  await expect(harmR.getByRole('button', { name: both })).toBeVisible();
  await harmL.getByRole('button', { name: both }).click();
  await expectTrims(page, 'L', {});
  await expectTrims(page, 'R', {});
  await expect(harmL.getByRole('button', { name: both })).toBeDisabled();
  await expect(harmR.getByRole('button', { name: both })).toBeDisabled();
  t = await steadyTransfer(page, fs, await windowAfterNow(page), 'cleared while linked');
  expectTrimmed(t, base, {}, worst, 'cleared while linked');

  // Unlinked again: «Azzera armoniche» on R clears R only.
  await linkL.click();
  await expect(linkL).toHaveAttribute('aria-pressed', 'false');
  await expect(linkR).toHaveAttribute('aria-pressed', 'false');
  await typeTrim(harmL, 1, -6, 30);
  await typeTrim(harmR, 2, -4, 45);
  await expectTrims(page, 'L', { 1: [-6, 30] });
  await expectTrims(page, 'R', { 2: [-4, 45] });
  await harmR.getByRole('button', { name: 'Azzera le correzioni delle armoniche del canale R' }).click();
  await expectTrims(page, 'R', {});
  await expectTrims(page, 'L', { 1: [-6, 30] });
  t = await steadyTransfer(page, fs, await windowAfterNow(page), 'R cleared while unlinked');
  expectTrimmed(t, base, { L: { 1: [-6, 30] } }, worst, 'R cleared while unlinked');
  await expectStillPlaying(page);
  worst.annotate();
});

test('in tonal mode «Fase totale @ f0» includes θ₁, matches the «Fondamentale» row and the measured phase', async ({ page }) => {
  const fs = await armTonal(page);
  const worst = new Worst();
  const stripL = strip(page, 'L');
  const harm = await openHarmonics(page, 'L');
  const main = stripL.locator('.phase-readout');
  const rowName = (h: number): string => `${h === 1 ? 'Fondamentale' : `Armonica ${h}`} · ${formatHz(h * HARMONIC_F0)}`;
  const row = (h: number): Locator => harm.getByRole('group', { name: rowName(h) }).locator('.strip-harm-total');
  // Readouts show one decimal (units.ts formatAdaptive).
  const mainText = (deg: number): string => `Fase totale: ${formatAdaptive(deg, 1)}° @ ${formatHz(HARMONIC_F0)}`;
  const rowText = (deg: number): string => `Fase totale ${formatAdaptive(deg, 1)}°`;
  await expect(main).toHaveText(mainText(180));
  await expect(row(1)).toHaveText(rowText(180));

  // θ₁ alone moves the main readout (polarity 180° + θ₁).
  await typeValue(harm, 'Fase armonica 1, valore in °', '30');
  await expect(main).toHaveText(mainText(210));
  await expect(row(1)).toHaveText(rowText(210));

  // Δ fase fine 10° and θ₂ −50°: harmonic h reads 180 + h·Δ + θ_h; the main readout stays the fundamental's.
  await typeValue(stripL, 'Δ fase fine, valore in °', '10');
  await typeValue(harm, 'Fase armonica 2, valore in °', '-50');
  await expectTrims(page, 'L', { 1: [0, 30], 2: [0, -50] });
  const totals: Record<number, number> = { 1: 220, 2: 150, 3: 210, 4: 220 };
  await expect(main).toHaveText(mainText(220));
  for (const h of HS) await expect(row(h)).toHaveText(rowText(totals[h]!));
  expect(phaseOf(await main.textContent())).toBe(phaseOf(await row(1).textContent()));
  await expect(strip(page, 'R').locator('.phase-readout')).toHaveText(mainText(180));

  // What plays: out L lags cond at h·f0 by the row's total phase plus the D_MIN sample every tap has.
  const t = await steadyTransfer(page, fs, await windowAfterNow(page), 'Δ 10°, θ₁ 30°, θ₂ −50°');
  for (const h of HS) {
    const shown = phaseOf(await row(h).textContent());
    const lag = asTrim(t.L[h - 1]!).phaseDeg;
    const d = { db: 0, deg: wrapDeg(lag - shown - (360 * h * HARMONIC_F0 * D_MIN_SAMPLES) / fs) };
    worst.add(d);
    const label = `out L at ${h * HARMONIC_F0} Hz: measured lag ${lag.toFixed(3)}° vs «Fase totale ${shown}°» + D_MIN`;
    expect(Math.abs(d.deg), label).toBeLessThan(TOL_DEG);
  }
  await expectStillPlaying(page);
  worst.annotate();
});
