/**
 * Guided-setup fixes from the review: master ramp before the level noise (safety-1), quick check of
 * a reused calibration (calib-9 / plan-4), loop gain of separate zones (calib-4), synchronous
 * detection in the tonal autotune (plan-2), Italian howl alarms (ui-14), loopback in samples
 * (plan-10).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { calibrationFreqs, checkLoop, createCalibration, LOOP_GAIN_DANGER_DB, pathResponse, solveFromCalibration } from '../../src/audio/calibration.ts';
import type { CalibrationData, PathData } from '../../src/audio/calibration.ts';
import { TAP_BLOCK_FRAMES, TAP_CHANNELS } from '../../src/audio/protocol.ts';
import type { AppController } from '../../src/app/controller.ts';
import { loopGainDb } from '../../src/dsp/alignment.ts';
import { GainRamp } from '../../src/dsp/gain-ramp.ts';
import { DEFAULT_PARAMS, clampParams } from '../../src/dsp/params.ts';
import { channelResponse } from '../../src/dsp/response.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';
import {
  compareToBaseline,
  gainLimitedAttenuationDb,
  harmonicLevelDbfs,
  howlAlarmText,
  liveMasterDb,
  MASTER_READY_TOL_DB,
  masterReached,
  masterSettleMs,
  masterWaitTimeoutMs,
  isLoopAdvice,
  mergedZonesLoop,
  mergeZoneSolutions,
  quickVerdict,
  usefulOutcome,
} from '../../src/ui/wizard/logic.ts';
import { NOISE_INTERRUPTED_TEXT, autotuneDriver, errorLevelDbfs, playLevelNoise } from '../../src/ui/wizard/ops.ts';
import { loopOutcomeText } from '../../src/ui/wizard/step-align.ts';
import { masterTimeoutText } from '../../src/ui/wizard/step-amp.ts';
import { loopLevelsText, loopRunText } from '../../src/ui/wizard/step-measure.ts';

const FS = 48000;

/** Time (ms) a real GainRamp needs from `from` to within 0.1 dB of `to` (after optional prior moves). */
function rampTimeMs(from: number, to: number, before: (r: GainRamp) => void = () => {}): number {
  const r = new GainRamp(FS, from);
  before(r);
  r.setTargetDb(to);
  for (let n = 0; n < FS * 30; n++) {
    r.next();
    if (Math.abs(r.currentDb - to) < 0.1) return (n / FS) * 1000;
  }
  return Infinity;
}

describe('master ramp before the amplifier-level noise (safety-1)', () => {
  it('masterSettleMs covers the real rate-limited ramp, without waiting much longer', () => {
    for (const from of [-30, -20, -12, -6, -3, -60, -80, -120]) {
      const real = rampTimeMs(from, 0);
      const bound = masterSettleMs(from, 0);
      expect(bound, `from ${from} dB`).toBeGreaterThanOrEqual(real);
      expect(bound - real, `from ${from} dB`).toBeLessThan(1300);
    }
    // −30 → 0 dB takes ≈ 4 s at 6 dB/s: the noise must not start before.
    expect(masterSettleMs(-30, 0)).toBeGreaterThan(4000);
    expect(masterSettleMs(0, -20)).toBeLessThan(200);
  });

  it('step 2 waits for the live master: within 0.1 dB of the target, silence matches silence', () => {
    expect(MASTER_READY_TOL_DB).toBe(0.1);
    expect(masterReached(-0.09, 0)).toBe(true);
    expect(masterReached(-0.11, 0)).toBe(false);
    expect(masterReached(0.05, 0)).toBe(true);
    expect(masterReached(null, 0)).toBe(false);
    expect(masterReached(-120, -120)).toBe(true);
    expect(masterReached(-119.95, -Infinity)).toBe(false);
    expect(masterReached(-60, -120)).toBe(false);
    // Meters missing, or from an engine that does not report the master: unknown, never "arrived".
    expect(liveMasterDb({ meters: null })).toBeNull();
    expect(liveMasterDb({ meters: {} })).toBeNull();
    expect(liveMasterDb({ meters: { masterDb: Number.NaN } })).toBeNull();
    expect(liveMasterDb({ meters: { masterDb: -12.5 } })).toBe(-12.5);
  });

  it('the README gives the range of seconds the step shows when the master is not confirmed (rounded up)', () => {
    const shown = [null, ...Array.from({ length: 121 }, (_, i) => -i)].map((from) => Math.ceil(masterWaitTimeoutMs(from, 0) / 1000));
    expect(Math.min(...shown)).toBe(10);
    expect(Math.max(...shown)).toBe(13);
    expect(masterTimeoutText(masterWaitTimeoutMs(null, 0))).toContain('entro 13 s');
    const readme = readFileSync(new URL('../../README.md', import.meta.url), 'utf8');
    expect(readme).toContain(`entro ${Math.min(...shown)}–${Math.max(...shown)} s`);
  });

  it('its timeout outlasts the slowest real ramp (from silence, empty rise budget), and is at least 10 s', () => {
    for (const from of [-120, -80, -60, -30, -6, 0]) {
      expect(masterWaitTimeoutMs(from, 0), `from ${from} dB`).toBeGreaterThan(rampTimeMs(from, 0) + 1000);
      expect(masterWaitTimeoutMs(from, 0)).toBeGreaterThanOrEqual(10_000);
    }
    // Unknown start: as from silence.
    expect(masterWaitTimeoutMs(null, 0)).toBe(masterWaitTimeoutMs(-120, 0));
    expect(masterWaitTimeoutMs(null, 0)).toBeLessThan(15_000);
    const afterRise = rampTimeMs(-10, 0, (r) => {
      r.setTargetDb(-4);
      for (let n = 0; n < FS * 0.2; n++) r.next();
    });
    expect(masterWaitTimeoutMs(-10, 0)).toBeGreaterThan(afterRise + 1000);
  });
});

/** Acoustic-looking path: 2nd-order high-pass shape × gain, pure delay of `delay` samples. */
function path(delay: number, gain = 1, sign: 1 | -1 = 1): PathData {
  const freqs = [...calibrationFreqs(FS, [20, 1000])];
  const re: number[] = [];
  const im: number[] = [];
  for (const f of freqs) {
    const mag = (gain * (f / 200) ** 2) / (1 + (f / 200) ** 2);
    const a = (-2 * Math.PI * f * delay) / FS;
    re.push(sign * mag * Math.cos(a));
    im.push(sign * mag * Math.sin(a));
  }
  return { delaySamples: delay, sign, qualityDb: 30, stdSamples: 0.1, freqs, re, im, ir: [], irOffset: 0 };
}

describe('quick check of a reused calibration (calib-9, plan-4)', () => {
  const ch = { ...clampParams(DEFAULT_PARAMS).channels[0], gainDb: -12, delayMs: 4 };
  const band: [number, number] = [20, 1000];

  it('measures the amplifier change and the latency change against the saved path', () => {
    const d = compareToBaseline(path(1300), path(1302.5, 10 ** (3 / 20)), FS, band)!;
    expect(d.delaySamples).toBeCloseTo(2.5, 9);
    expect(d.gainDb).toBeCloseTo(3, 6);
    expect(d.signFlipped).toBe(false);
    expect(compareToBaseline(path(1300), path(1300, 1, -1), FS, band)!.signFlipped).toBe(true);
  });

  it('keeps the saved values within 0.5 samples and 0.5 dB', () => {
    const v = quickVerdict(compareToBaseline(path(1300), path(1300.3, 10 ** (0.3 / 20)), FS, band), ch, FS);
    expect(v.kind).toBe('ok');
  });

  it('corrects gain and delay for an amplifier re-levelled by ear and a slightly different latency', () => {
    // Amplifier 3 dB louder, latency 1 ms (48 samples) longer.
    const v = quickVerdict(compareToBaseline(path(1300), path(1348, 10 ** (3 / 20)), FS, band), ch, FS);
    expect(v.kind).toBe('corrected');
    if (v.kind !== 'corrected') return;
    expect(v.channel.gainDb).toBeCloseTo(-15, 6);
    expect(v.channel.delayMs).toBeCloseTo(3, 6);
    expect(v.text).toBe('Guadagno −12,0 → −15,0 dB (percorso 3,0 dB); ritardo 4,000 → 3,000 ms (latenza 1,000 ms).');
  });

  it('asks for new measurements when the change is too big or cannot be corrected', () => {
    const far = quickVerdict(compareToBaseline(path(1300), path(1300, 10 ** (9 / 20)), FS, band), ch, FS);
    expect(far.kind).toBe('mismatch');
    const late = quickVerdict(compareToBaseline(path(1300), path(1300 + 0.006 * FS), FS, band), { ...ch, delayMs: 8 }, FS);
    expect(late.kind).toBe('mismatch');
    // 2 ms more latency with only 1 ms of delay to give back.
    const neg = quickVerdict(compareToBaseline(path(1300), path(1396), FS, band), { ...ch, delayMs: 1 }, FS);
    expect(neg.kind).toBe('mismatch');
    const flipped = quickVerdict(compareToBaseline(path(1300), path(1300, 1, -1), FS, band), ch, FS);
    expect(flipped.kind).toBe('mismatch');
    expect(quickVerdict(null, ch, FS).kind).toBe('mismatch');
  });
});

/** PathData of integer-lag impulses (as measurePath stores it, with the response on the grid). */
function impulsePath(lag: number, gain: number, band: [number, number]): PathData {
  const ir = new Array<number>(lag + 64).fill(0);
  ir[lag] = gain;
  const p: PathData = { delaySamples: lag, sign: gain >= 0 ? 1 : -1, qualityDb: 40, stdSamples: 0.05, freqs: [], re: [], im: [], ir, irOffset: 0 };
  const freqs = calibrationFreqs(FS, band);
  const r = pathResponse(p, FS, freqs);
  return { ...p, freqs: Array.from(freqs), re: Array.from(r.re), im: Array.from(r.im) };
}

describe('separate zones: loop gain with both speakers (calib-4)', () => {
  it('reports the loop of the merged parameters, not of one speaker at a time', () => {
    const band: [number, number] = [20, 1000];
    const cal: CalibrationData = createCalibration(FS, band, 'separate');
    const freqs = calibrationFreqs(FS, band);
    cal.primary = {
      freqs: Array.from(freqs),
      re: Array.from(freqs, (f) => 0.7 * Math.cos(-2 * Math.PI * f * 0.03)),
      im: Array.from(freqs, (f) => 0.7 * Math.sin(-2 * Math.PI * f * 0.03)),
      coherence: Array.from(freqs, () => 1),
      tonalF0: null,
    };
    cal.secondary = [impulsePath(720, 1, band), impulsePath(724, 1, band)];
    // Both speakers equally coupled to the reference mic.
    cal.feedback = [impulsePath(900, 0.3, band), impulsePath(900, 0.3, band)];
    const p = clampParams(undefined);
    p.masterDb = 0;
    const zones = ([0, 1] as const).map((z) =>
      solveFromCalibration({ ...cal, secondary: z === 0 ? [cal.secondary[0], null] : [null, cal.secondary[1]] }, p, FS),
    );
    const perZone = Math.max(...zones.map((o) => o.loopGainDb ?? -Infinity));
    const merged = mergeZoneSolutions(p, [zones[0]!.params, zones[1]!.params]);
    const actual = loopGainDb(
      [(f) => channelResponse(merged, FS, f, 0), (f) => channelResponse(merged, FS, f, 1)],
      [pathResponse(cal.feedback[0]!, FS, freqs), pathResponse(cal.feedback[1]!, FS, freqs)],
      band,
    ).maxDb;
    // The per-zone figure misses the second speaker (≈ 6 dB with symmetric coupling).
    expect(actual - perZone).toBeGreaterThan(4);
    const advice = zones.flatMap((o, i) => o.advice.map((a) => `Zona ${'LR'[i]}: ${a}`));
    const r = mergedZonesLoop(cal, merged, FS, 256, advice);
    expect(r.loopGainDb).toBeCloseTo(actual, 6);
    // One loop statement, about both speakers; the per-zone ones are gone.
    const loopLines = r.advice.filter((a) => /anello/.test(a));
    expect(loopLines).toHaveLength(1);
    expect(loopLines[0]).toMatch(/entrambe le casse/);
    // The neutralised loop of the merged params, as calibration.ts computes it for the FIRs loaded.
    expect(r.loopGainAfterDb).toBeCloseTo(checkLoop(cal, merged, FS, 256)!.afterDb, 9);
    expect(r.loopGainAfterDb!).toBeLessThan(r.loopGainDb!);
  });
});

describe('separate zones: the stability limit applies to the merged parameters (calib-1, calib-4)', () => {
  /** A speaker → reference path with an echo too late for the neutraliser (beyond 1024 taps). */
  function echoPath(lag: number, echoLag: number, gain: number, band: [number, number]): PathData {
    const ir = new Array<number>(lag + echoLag + 64).fill(0);
    ir[lag] = gain;
    ir[lag + echoLag] = gain;
    const p: PathData = { delaySamples: lag, sign: 1, qualityDb: 40, stdSamples: 0.05, freqs: [], re: [], im: [], ir, irOffset: 0 };
    const freqs = calibrationFreqs(FS, band);
    const r = pathResponse(p, FS, freqs);
    return { ...p, freqs: Array.from(freqs), re: Array.from(r.re), im: Array.from(r.im) };
  }

  it('lowers both channel gains by the excess over −6 dB, and leaves a loop with margin alone', () => {
    const band: [number, number] = [20, 1000];
    const cal: CalibrationData = createCalibration(FS, band, 'separate');
    cal.feedback = [echoPath(900, 3000, 0.3, band), echoPath(900, 3000, 0.3, band)];
    const p = clampParams(undefined);
    p.masterDb = 0;
    p.channels[0].gainDb = 0;
    p.channels[1].gainDb = 0;
    const before = checkLoop(cal, p, FS, 256)!;
    // One speaker alone keeps the margin; both together do not.
    const alone = checkLoop(cal, { ...p, channels: [p.channels[0], { ...p.channels[1], enabled: false }] }, FS, 256)!;
    expect(alone.afterDb).toBeLessThan(LOOP_GAIN_DANGER_DB);
    expect(before.afterDb).toBeGreaterThan(LOOP_GAIN_DANGER_DB);
    const r = mergedZonesLoop(cal, p, FS, 256, []);
    expect(r.limitedDb).toBeCloseTo(before.afterDb - LOOP_GAIN_DANGER_DB, 6);
    expect(r.params.channels.map((c) => c.gainDb)).toEqual([-r.limitedDb, -r.limitedDb]);
    expect(r.loopGainAfterDb!).toBeCloseTo(LOOP_GAIN_DANGER_DB, 6);
    expect(r.advice.some((a) => /Rischio di innesco/.test(a))).toBe(false);
    // The attenuation left with that gain error: 1 dB → 19,3 dB, 6 dB (half the anti-noise) → 6 dB.
    expect(gainLimitedAttenuationDb(1)).toBeCloseTo(19.27, 2);
    expect(gainLimitedAttenuationDb(20 * Math.log10(2))).toBeCloseTo(20 * Math.log10(2), 9);
    expect(gainLimitedAttenuationDb(0)).toBe(Infinity);
    // With margin: unchanged.
    const low = clampParams({ ...p, channels: [{ ...p.channels[0], gainDb: -12 }, { ...p.channels[1], gainDb: -12 }] });
    const ok = mergedZonesLoop(cal, low, FS, 256, []);
    expect(ok.limitedDb).toBe(0);
    expect(ok.params).toBe(low);
  });
});

describe('separate zones: a zone left unsolved for stability (calib-ui-2)', () => {
  const band: [number, number] = [20, 1000];
  function arrivals(list: [number, number][]): PathData {
    const last = Math.max(...list.map((a) => a[0]));
    const ir = new Array<number>(last + 64).fill(0);
    for (const [lag, gain] of list) ir[lag] += gain;
    const freqs = calibrationFreqs(FS, band);
    const base: PathData = { delaySamples: Math.min(...list.map((a) => a[0])), sign: 1, qualityDb: 40, stdSamples: 0.05, freqs: [], re: [], im: [], ir, irOffset: 0 };
    const r = pathResponse(base, FS, freqs);
    return { ...base, freqs: Array.from(freqs), re: Array.from(r.re), im: Array.from(r.im) };
  }

  it('switches that zone’s speaker off instead of playing its unsolved values, and keeps the reason', () => {
    const cal = createCalibration(FS, band, 'separate');
    // Speaker L hears itself through a reflection 40 ms later and 20 dB stronger: no gain reduction
    // leaves useful attenuation, so finishSolve refuses zone L.
    cal.feedback = [arrivals([[768, 1], [768 + 1920, 10]]), arrivals([[800, 0.02]])];
    cal.secondary = [arrivals([[720, 1]]), arrivals([[720, 1]])];
    const freqs = Array.from(calibrationFreqs(FS, band));
    cal.primary = {
      freqs,
      re: freqs.map((f) => 0.7 * Math.cos(-2 * Math.PI * f * 0.03)),
      im: freqs.map((f) => 0.7 * Math.sin(-2 * Math.PI * f * 0.03)),
      coherence: freqs.map(() => 1),
      tonalF0: null,
      psd: freqs.map((f) => (f >= 60 && f <= 400 ? 1 : 1e-4)),
    };
    const p = clampParams(undefined);
    p.masterDb = 0;
    // The step-align pipeline for two zones.
    const outs = ([0, 1] as const).map((z) =>
      solveFromCalibration({ ...cal, secondary: z === 0 ? [cal.secondary[0], null] : [null, cal.secondary[1]] }, p, FS, { feedbackTaps: 256 }),
    );
    expect(usefulOutcome(outs[0])).toBe(false);
    expect(usefulOutcome(outs[1])).toBe(true);
    const advice = outs.flatMap((o, i) => o.advice.map((a) => `Zona ${'LR'[i]}: ${a}`));
    const refusal = advice.find((a) => a.startsWith('Zona L: Rischio di innesco') && a.includes('non sono stati cambiati'));
    expect(refusal).toBeDefined();
    const merged = mergeZoneSolutions(p, [null, outs[1]!.params]);
    expect(merged.channels[0].enabled).toBe(false);
    expect(merged.channels[1]).toEqual(outs[1]!.params.channels[1]);
    const r = mergedZonesLoop(cal, merged, FS, 256, advice);
    // With speaker L off the loop is zone R's alone: no reduction of zone R for speaker L's feedback.
    expect(r.limitedDb).toBe(0);
    expect(r.params.channels[0].enabled).toBe(false);
    expect(r.advice).toContain(refusal);
    // Loop figures are still replaced by the merged statement; refusals are not figures.
    expect(isLoopAdvice("Guadagno d'anello −3,0 dB a 120 Hz: …")).toBe(true);
    expect(isLoopAdvice(refusal!.replace(/^Zona L: /, ''))).toBe(false);
  });
});

describe('solve outcome text: loop gain after neutralisation (calib-1, plan-5)', () => {
  it('reports the neutralised loop and the stability gain reduction', () => {
    expect(loopOutcomeText(-12, -31.04, 0)).toBe(
      ' Guadagno d’anello (senza neutralizzazione) −12,0 dB. Guadagno d’anello dopo la neutralizzazione −31,0 dB.',
    );
    expect(loopOutcomeText(-2, -6, 3.5)).toBe(
      ' Guadagno d’anello (senza neutralizzazione) −2,0 dB. Guadagno d’anello dopo la neutralizzazione −6,0 dB: guadagni ridotti di 3,5 dB per la stabilità.',
    );
    expect(loopOutcomeText(null, null, 0)).toBe('');
  });
});

describe('tonal autotune measures the harmonics of f0 (plan-2)', () => {
  const band: [number, number] = [20, 1000];
  /** −30 dBFS white noise + a tone at 100 Hz of amplitude `amp`. */
  function record(amp: number, seed: number, seconds = 0.2): Float32Array {
    const rnd = mulberry32(seed);
    const x = new Float32Array(Math.round(seconds * FS));
    for (let i = 0; i < x.length; i++) x[i] = 0.03 * Math.sqrt(3) * (2 * rnd() - 1) + amp * Math.sin((2 * Math.PI * 100 * i) / FS);
    return x;
  }

  it('reads the level of a pure tone and ignores the broadband noise around it', () => {
    const tone = new Float32Array(9600).map((_v, i) => 0.1 * Math.sin((2 * Math.PI * 100 * i) / FS));
    expect(harmonicLevelDbfs(tone, FS, 100, 4, band)).toBeCloseTo(20 * Math.log10(0.1 / Math.SQRT2), 1);
    expect(harmonicLevelDbfs(tone, FS, 2000, 1, band)).toBeNull();
  });

  it('sees a 20 dB cancellation of a tone 10 dB below the in-band noise (broadband RMS does not)', () => {
    const p = clampParams({ ...DEFAULT_PARAMS, phaseMode: 'tonal', tonal: { f0: 100, harmonics: 4, q: 30 } });
    const on = record(2.7e-3, 7);
    const off = record(2.7e-4, 7);
    expect(errorLevelDbfs(on, FS, band, p) - errorLevelDbfs(off, FS, band, p)).toBeGreaterThan(6);
    const delay = clampParams(DEFAULT_PARAMS);
    expect(Math.abs(errorLevelDbfs(on, FS, band, delay) - errorLevelDbfs(off, FS, band, delay))).toBeLessThan(1);
  });

  it('the driver uses the current phase mode of the controller, for error and bypass windows', async () => {
    const params = clampParams({ ...DEFAULT_PARAMS, phaseMode: 'tonal', tonal: { f0: 100, harmonics: 4, q: 30 } });
    const taps = new Set<(b: Float32Array, frames: number) => void>();
    const controller = {
      state: { params, fs: FS },
      onTap(cb: (b: Float32Array, frames: number) => void) {
        taps.add(cb);
        return () => taps.delete(cb);
      },
      setBypass() {},
    } as unknown as AppController;
    const drv = autotuneDriver(controller, { fs: FS, band, signal: new AbortController().signal, onMeasure: () => {} });
    const pending = drv.measureErrorDb(200);
    const iErr = TAP_CHANNELS.indexOf('err');
    const x = record(0.05, 3, 0.3);
    for (let b = 0; b * TAP_BLOCK_FRAMES < x.length; b++) {
      const block = new Float32Array(TAP_CHANNELS.length * TAP_BLOCK_FRAMES);
      block.set(x.subarray(b * TAP_BLOCK_FRAMES, (b + 1) * TAP_BLOCK_FRAMES), iErr * TAP_BLOCK_FRAMES);
      for (const cb of [...taps]) cb(block, TAP_BLOCK_FRAMES);
    }
    expect(await pending).toBeCloseTo(20 * Math.log10(0.05 / Math.SQRT2), 0);
  });

  it('reads each harmonic on its own for the per-harmonic trim search, bypassing the anti-noise when asked', async () => {
    vi.useFakeTimers();
    try {
      const params = clampParams({ ...DEFAULT_PARAMS, phaseMode: 'tonal', tonal: { f0: 100, harmonics: 2, q: 30 } });
      const taps = new Set<(b: Float32Array, frames: number) => void>();
      const bypass: boolean[] = [];
      const controller = {
        state: { params, fs: FS },
        onTap(cb: (b: Float32Array, frames: number) => void) {
          taps.add(cb);
          return () => taps.delete(cb);
        },
        setBypass(on: boolean) {
          bypass.push(on);
        },
      } as unknown as AppController;
      let measured = 0;
      const drv = autotuneDriver(controller, { fs: FS, band, signal: new AbortController().signal, onMeasure: () => measured++ });
      expect(drv.measureHarmonicsDb).toBeDefined();
      // −50 dBFS noise, 100 Hz at amplitude 0.05 and 200 Hz at 0.015.
      const rnd = mulberry32(11);
      const x = new Float32Array(Math.round(0.2 * FS)).map(
        (_v, i) => 0.003 * Math.sqrt(3) * (2 * rnd() - 1) + 0.05 * Math.sin((2 * Math.PI * 100 * i) / FS) + 0.015 * Math.sin((2 * Math.PI * 200 * i) / FS),
      );
      const feed = (): void => {
        const iErr = TAP_CHANNELS.indexOf('err');
        for (let b = 0; b * TAP_BLOCK_FRAMES < x.length; b++) {
          const block = new Float32Array(TAP_CHANNELS.length * TAP_BLOCK_FRAMES);
          block.set(x.subarray(b * TAP_BLOCK_FRAMES, (b + 1) * TAP_BLOCK_FRAMES), iErr * TAP_BLOCK_FRAMES);
          for (const cb of [...taps]) cb(block, TAP_BLOCK_FRAMES);
        }
      };
      const on = drv.measureHarmonicsDb!(200, [100, 200], false);
      feed();
      const levels = await on;
      expect(Math.abs(levels[0]! - 20 * Math.log10(0.05 / Math.SQRT2))).toBeLessThan(0.2);
      expect(Math.abs(levels[1]! - 20 * Math.log10(0.015 / Math.SQRT2))).toBeLessThan(0.2);
      expect(bypass).toEqual([]);
      // Bypassed: the anti-noise goes off, the room settles, the window is recorded, the anti-noise comes back.
      const off = drv.measureHarmonicsDb!(200, [100], true);
      expect(bypass).toEqual([true]);
      await vi.advanceTimersByTimeAsync(400);
      feed();
      expect((await off)[0]).toBeCloseTo(20 * Math.log10(0.05 / Math.SQRT2), 0);
      expect(bypass).toEqual([true, false]);
      expect(measured).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('amplifier-level noise interrupted by the engine', () => {
  it('names a master change among the causes (the engine aborts a stimulus when the master moves)', async () => {
    expect(NOISE_INTERRUPTED_TEXT).toMatch(/volume master cambiato/);
    const controller = {
      measure: async () => ({ ref: new Float32Array(0), err: null, masterLin: 0, aborted: true }),
    } as unknown as AppController;
    const played: number[] = [];
    const reason = await playLevelNoise(controller, 0, {
      fs: 8000,
      band: [100, 1000],
      rmsDbfs: -20,
      signal: new AbortController().signal,
      onPlayed: (t) => played.push(t),
    });
    expect(reason).toBe(NOISE_INTERRUPTED_TEXT);
    expect(played).toEqual([]);
  });
});

describe('howl alarm text (ui-14)', () => {
  it('is Italian and calls only a growing peak or a saturated limiter "innesco"', () => {
    expect(howlAlarmText('peak-growth @ 1234 Hz')).toMatch(/^Innesco rilevato \(picco in crescita a 1234 Hz\)/);
    expect(howlAlarmText('limiter-saturated')).toMatch(/^Innesco rilevato \(limiter saturo\)/);
    expect(howlAlarmText('output-overload')).toMatch(/^Uscita in sovraccarico/);
    expect(howlAlarmText('anc-worse')).toMatch(/aumentando il rumore/);
    for (const r of ['peak-growth @ 80 Hz', 'limiter-saturated', 'output-overload', 'anc-worse', undefined, 'boh']) {
      expect(howlAlarmText(r)).not.toMatch(/[a-z]+-[a-z]+/);
    }
  });
});

describe('loopback in samples (plan-10)', () => {
  it('shows the delay with 3 decimals, the spread and the count, and compares the two levels', () => {
    const a = { levelDbfs: -30, delaySamples: 1234.5678, stdSamples: 0.0123, count: 5 };
    expect(loopRunText(a, FS)).toBe('−30 dBFS: 1234,568 campioni (25,720 ms), σ 0,012 campioni su 5 misure');
    expect(loopLevelsText([a])).toBeNull();
    const same = loopLevelsText([a, { ...a, levelDbfs: -20, delaySamples: 1234.9 }])!;
    expect(same.ok).toBe(true);
    expect(same.text).toMatch(/0,332 campioni/);
    const os = loopLevelsText([a, { ...a, levelDbfs: -20, delaySamples: 1236 }])!;
    expect(os.ok).toBe(false);
    expect(os.text).toMatch(/elaborano il segnale/);
  });
});
