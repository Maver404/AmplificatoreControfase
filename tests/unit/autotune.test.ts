import { describe, expect, it } from 'vitest';
import { AUTOTUNE_ABORT_MARGIN_DB, AUTOTUNE_BT, MAX_WINDOW_MS, autotuneTiming, runAutotune } from '../../src/audio/autotune.ts';
import type { AutotuneDriver, AutotuneOptions } from '../../src/audio/autotune.ts';
import { bandRmsDbfs, calibrationFreqs, createCalibration, noiseBandwidthHz, welchTransfer } from '../../src/audio/calibration.ts';
import { Biquad, highpass, lowpass } from '../../src/dsp/biquad.ts';
import {
  DELAY_CROSSFADE_MS,
  GAIN_SMOOTH_MS,
  RANGES,
  clampParams,
  computeChannelDelay,
} from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;
const BAND: [number, number] = [80, 400];

interface Log {
  applied: { p: EngineParams; jump: boolean }[];
  settles: number[];
  errors: number[];
  bypass: number[];
}

/**
 * Error-mic model: residual = 1 + Σ_ch a_ch·(±g_ch)·e^{-jω(τ_ch − τ_opt,ch)}·rot_ch over a set of
 * frequencies, where the optimum parameters cancel exactly (a_ch = share / optimum gain). Bypass = 0 dB,
 * floor −40 dB, optional seeded measurement noise.
 */
function model(optimum: EngineParams, freqs: number[], noiseDb = 0, seed = 1) {
  const active = ([0, 1] as const).filter((ch) => optimum.channels[ch].enabled);
  const tau = (p: EngineParams, ch: 0 | 1): number => computeChannelDelay(p, fs, ch).samples / fs;
  const rnd = mulberry32(seed);
  const noise = (): number => noiseDb * (2 * rnd() - 1);
  const errorDb = (p: EngineParams): number => {
    let e = 0;
    for (const f of freqs) {
      const w = 2 * Math.PI * f;
      let re = 1;
      let im = 0;
      for (const ch of active) {
        const c = p.channels[ch];
        if (!c.enabled) continue;
        const o = optimum.channels[ch];
        const share = 1 / active.length;
        const sign = (c.polarityInvert ? -1 : 1) * (o.polarityInvert ? -1 : 1);
        const g = (share * sign * Math.pow(10, c.gainDb / 20)) / Math.pow(10, o.gainDb / 20);
        let ph = -w * (tau(p, ch) - tau(optimum, ch));
        if (p.phaseMode === 'hilbert') ph -= ((c.phaseDeltaDeg - o.phaseDeltaDeg) * Math.PI) / 180;
        re -= g * Math.cos(ph);
        im -= g * Math.sin(ph);
      }
      e += re * re + im * im;
    }
    return 10 * Math.log10(e / freqs.length + 1e-4);
  };
  return { errorDb, noise };
}

function mockDriver(optimum: EngineParams, freqs: number[], over: Partial<AutotuneDriver> = {}, noiseDb = 0) {
  const m = model(optimum, freqs, noiseDb);
  const log: Log = { applied: [], settles: [], errors: [], bypass: [] };
  let current: EngineParams | null = null;
  const driver: AutotuneDriver = {
    apply: (p, jump) => {
      current = clampParams(p);
      log.applied.push({ p: current, jump });
    },
    settle: async (ms) => {
      log.settles.push(ms);
    },
    measureErrorDb: async () => {
      const db = m.errorDb(current!) + m.noise();
      log.errors.push(db);
      return db;
    },
    measureBypassDb: async () => {
      const db = m.noise();
      log.bypass.push(db);
      return db;
    },
    ...over,
  };
  return { driver, log, current: () => current, errorDb: m.errorDb };
}

function params(over: (p: EngineParams) => void): EngineParams {
  const p = clampParams(undefined);
  p.masterDb = -6;
  p.channels[1].enabled = false;
  over(p);
  return clampParams(p);
}

const broadband = Array.from({ length: 40 }, (_, i) => 80 * Math.pow(5, i / 39));
const opts: AutotuneOptions = { fs, channels: [0], rt60S: 0.2, band: BAND };

describe('runAutotune', () => {
  const optimum = params((p) => {
    p.channels[0].delayMs = 5;
    p.channels[0].gainDb = -8;
  });
  const detuned = params((p) => {
    p.channels[0].delayMs = 5.4;
    p.channels[0].gainDb = -12;
  });

  it('delay mode: finds delay and gain from a detuned start, never worse than bypass + 1 dB', async () => {
    const { driver, log } = mockDriver(optimum, broadband);
    const res = await runAutotune(driver, detuned, opts);
    expect(res.aborted).toBe(false);
    expect(res.params.channels[0].delayMs).toBeCloseTo(5, 1);
    expect(Math.abs(res.params.channels[0].gainDb + 8)).toBeLessThan(0.3);
    expect(res.startDb - res.endDb).toBeGreaterThanOrEqual(6);
    expect(res.bypassDb).toBeCloseTo(0, 6);
    for (const e of log.errors) expect(e).toBeLessThanOrEqual(1);
    // Only the searched coordinates move.
    expect(res.params.channels[0].phaseDeltaDeg).toBe(0);
    expect(res.params.channels[0].polarityInvert).toBe(true);
    expect(res.params.masterDb).toBe(-6);
    expect(res.steps).toBeGreaterThan(6);
    // The search steps are not jumps; the start is.
    expect(log.applied[0].jump).toBe(true);
    expect(log.applied.slice(1).every((a) => !a.jump)).toBe(true);
  });

  it('copes with measurement noise (±0.3 dB)', async () => {
    const { driver, log } = mockDriver(optimum, broadband, {}, 0.3);
    const res = await runAutotune(driver, detuned, opts);
    expect(res.aborted).toBe(false);
    expect(res.startDb - res.endDb).toBeGreaterThanOrEqual(6);
    expect(Math.abs(res.params.channels[0].delayMs - 5)).toBeLessThan(0.1);
    for (const e of log.errors) expect(e).toBeLessThanOrEqual(1.3);
  });

  it('tonal mode searches Δ (and gain), not the base delay', async () => {
    const tonal = (delta: number, gainDb: number): EngineParams =>
      params((p) => {
        p.phaseMode = 'tonal';
        p.tonal.f0 = 100;
        p.channels[0].delayMs = 3;
        p.channels[0].phaseDeltaDeg = delta;
        p.channels[0].gainDb = gainDb;
      });
    const { driver } = mockDriver(tonal(25, -8), [100, 200, 300]);
    const res = await runAutotune(driver, tonal(0, -11), opts);
    expect(res.aborted).toBe(false);
    expect(res.params.channels[0].delayMs).toBe(3);
    expect(res.params.channels[0].phaseDeltaDeg).toBeCloseTo(25, 0);
    expect(res.params.channels[0].gainDb).toBeCloseTo(-8, 0);
  });

  it('hilbert mode searches the rotation Δ', async () => {
    const hil = (delta: number): EngineParams =>
      params((p) => {
        p.phaseMode = 'hilbert';
        p.channels[0].delayMs = 2;
        p.channels[0].phaseDeltaDeg = delta;
        p.channels[0].gainDb = -10;
      });
    const { driver } = mockDriver(hil(-30), broadband);
    const res = await runAutotune(driver, hil(-5), opts);
    expect(res.params.channels[0].phaseDeltaDeg).toBeCloseTo(-30, 0);
    expect(res.params.channels[0].delayMs).toBe(2);
  });

  it('two speakers on one zone: each channel is tuned with the other one active', async () => {
    const two = (d0: number, g0: number, d1: number, g1: number): EngineParams =>
      params((p) => {
        p.channels[1].enabled = true;
        p.channels[0].delayMs = d0;
        p.channels[0].gainDb = g0;
        p.channels[1].delayMs = d1;
        p.channels[1].gainDb = g1;
      });
    const { driver, log } = mockDriver(two(5, -14, 6, -14), broadband);
    const start = two(5.4, -18, 5.6, -10);
    const res = await runAutotune(driver, start, { ...opts, channels: [0, 1] });
    expect(res.aborted).toBe(false);
    // Two speakers on one zone have a valley of equivalent settings: judge the level, not the values.
    expect(res.startDb - res.endDb).toBeGreaterThanOrEqual(6);
    expect(res.endDb).toBeLessThan(-25);
    for (const ch of [0, 1] as const) {
      expect(res.params.channels[ch].delayMs).not.toBe(start.channels[ch].delayMs);
      expect(res.params.channels[ch].gainDb).not.toBe(start.channels[ch].gainDb);
    }
    // Both channels stay enabled in every evaluated setting.
    expect(log.applied.every((a) => a.p.channels[0].enabled && a.p.channels[1].enabled)).toBe(true);
  });

  it('keeps the gain inside the parameter range', async () => {
    const loud = params((p) => {
      p.channels[0].delayMs = 5;
      p.channels[0].gainDb = 3;
      p.gainUnlocked = true;
    });
    const start = params((p) => {
      p.channels[0].delayMs = 5;
      p.channels[0].gainDb = -1;
    });
    const { driver, log } = mockDriver(loud, broadband);
    const res = await runAutotune(driver, start, opts);
    for (const a of log.applied) expect(a.p.channels[0].gainDb).toBeLessThanOrEqual(RANGES.gainDb.max);
    expect(res.params.channels[0].gainDb).toBeCloseTo(0, 6);
  });

  it('measures with B·T ≥ 100, settles for smoothing, crossfade, RT60 and the delay slew, and interleaves bypass windows', async () => {
    const t = autotuneTiming(opts);
    expect(((BAND[1] - BAND[0]) * t.windowMs) / 1000).toBeGreaterThanOrEqual(100);
    expect(((BAND[1] - BAND[0]) * t.windowMs) / 1000).toBeGreaterThanOrEqual(AUTOTUNE_BT - 1);
    expect(t.settleMs).toBeGreaterThanOrEqual(opts.rt60S * 1000);
    expect(t.settleMs).toBeGreaterThanOrEqual(5 * GAIN_SMOOTH_MS);
    expect(t.settleMs).toBeGreaterThanOrEqual(DELAY_CROSSFADE_MS);
    const windows: number[] = [];
    const { driver, log } = mockDriver(optimum, broadband);
    const measure = driver.measureErrorDb;
    driver.measureErrorDb = async (ms) => {
      windows.push(ms);
      return measure(ms);
    };
    await runAutotune(driver, detuned, opts);
    expect(windows.every((w) => w === t.windowMs)).toBe(true);
    expect(Math.min(...log.settles)).toBeGreaterThanOrEqual(t.settleMs);
    // A 0.35 ms delay probe slews at DELAY_SLEW samples/sample: ≈ 117 ms more.
    expect(Math.max(...log.settles)).toBeGreaterThan(t.settleMs + 100);
    expect(log.bypass.length).toBeGreaterThanOrEqual(1 + Math.floor((log.errors.length - 1) / 6));
  });

  it('aborts at once and restores the start when the ANC is worse than bypass + 1 dB', async () => {
    const wrong = params((p) => {
      p.channels[0].delayMs = 5;
      p.channels[0].gainDb = -8;
      p.channels[0].polarityInvert = false;
    });
    const { driver, log } = mockDriver(optimum, broadband);
    const res = await runAutotune(driver, wrong, opts);
    expect(res.aborted).toBe(true);
    expect(res.reason).toMatch(/bypass/);
    expect(res.params).toEqual(wrong);
    expect(res.steps).toBe(0);
    expect(log.errors.length).toBe(1);
    const last = log.applied[log.applied.length - 1];
    expect(last.jump).toBe(true);
    expect(last.p).toEqual(wrong);
  });

  it('restores the start parameters on abort, driver failure or invalid measurements', async () => {
    const ctl = new AbortController();
    const a = mockDriver(optimum, broadband, { signal: ctl.signal });
    const measure = a.driver.measureErrorDb;
    a.driver.measureErrorDb = async (ms) => {
      if (a.log.errors.length === 5) ctl.abort();
      return measure(ms);
    };
    const aborted = await runAutotune(a.driver, detuned, opts);
    expect(aborted.aborted).toBe(true);
    expect(aborted.reason).toMatch(/annullato/);
    expect(aborted.params).toEqual(detuned);
    expect(a.current()).toEqual(detuned);

    const pre = new AbortController();
    pre.abort();
    const b = mockDriver(optimum, broadband, { signal: pre.signal });
    const early = await runAutotune(b.driver, detuned, opts);
    expect(early.aborted).toBe(true);
    expect(b.log.errors.length).toBe(0);
    expect(b.current()).toEqual(detuned);

    let calls = 0;
    const c = mockDriver(optimum, broadband);
    const m2 = c.driver.measureErrorDb;
    c.driver.measureErrorDb = async (ms) => {
      if (++calls === 4) throw new Error('worklet scomparso');
      return m2(ms);
    };
    const failed = await runAutotune(c.driver, detuned, opts);
    expect(failed.aborted).toBe(true);
    expect(failed.reason).toMatch(/worklet scomparso/);
    expect(c.current()).toEqual(detuned);

    const d = mockDriver(optimum, broadband, { measureBypassDb: async () => Number.NaN });
    const invalid = await runAutotune(d.driver, detuned, opts);
    expect(invalid.aborted).toBe(true);
    expect(invalid.reason).toMatch(/non valida/);
    expect(d.current()).toEqual(detuned);
  });

  it('keeps the start when nothing improves it', async () => {
    const { driver, errorDb } = mockDriver(optimum, broadband);
    const res = await runAutotune(driver, optimum, opts);
    expect(res.aborted).toBe(false);
    expect(errorDb(res.params)).toBeLessThanOrEqual(errorDb(optimum) + 1e-9);
  });
});

describe('autotune windows and the effective bandwidth of the noise', () => {
  function rumble(seconds: number, seed: number): Float32Array {
    const u = mulberry32(seed);
    const hp = new Biquad(highpass(fs, 25, Math.SQRT1_2));
    const lp1 = new Biquad(lowpass(fs, 70, Math.SQRT1_2));
    const lp2 = new Biquad(lowpass(fs, 70, Math.SQRT1_2));
    const out = new Float32Array(Math.round(seconds * fs));
    for (let i = 0; i < out.length; i++) {
      const g = Math.sqrt(-2 * Math.log(Math.max(1e-12, u()))) * Math.cos(2 * Math.PI * u());
      out[i] = 0.05 * lp2.process(lp1.process(hp.process(g)));
    }
    return out;
  }

  it('rumble in a nominal [20, 1000] band: windows sized on the measured bandwidth keep σ ≤ 0.45 dB', () => {
    const band: [number, number] = [20, 1000];
    const x = rumble(70, 7);
    // The primary measurement (reference → error = a pure delay here) gives the PSD the autotune needs.
    const cal = createCalibration(fs, band, 'single');
    const freqs = calibrationFreqs(fs, band);
    const y = new Float32Array(x.length);
    y.set(x.subarray(0, x.length - 240), 240);
    const t = welchTransfer(x.subarray(0, 10 * fs), y.subarray(0, 10 * fs), fs, freqs);
    cal.primary = {
      freqs: Array.from(freqs),
      re: Array.from(t.re),
      im: Array.from(t.im),
      coherence: Array.from(t.coherence),
      tonalF0: null,
      psd: Array.from(t.pxx),
    };
    const bEff = noiseBandwidthHz(cal)!;
    expect(bEff).toBeGreaterThan(20);
    expect(bEff).toBeLessThan(150);
    const nominal = autotuneTiming({ fs, channels: [0], rt60S: 0.3, band });
    const sized = autotuneTiming({ fs, channels: [0], rt60S: 0.3, band, effectiveBandHz: bEff });
    expect(nominal.windowMs).toBe(200);
    expect(sized.windowMs).toBeGreaterThan(1000);
    expect(sized.windowMs).toBeLessThanOrEqual(MAX_WINDOW_MS);
    const n = Math.round((sized.windowMs / 1000) * fs);
    const levels: number[] = [];
    for (let s = fs; s + n <= x.length; s += n) levels.push(bandRmsDbfs(x.subarray(s, s + n), fs, band));
    const mean = levels.reduce((a, b) => a + b, 0) / levels.length;
    const sd = Math.sqrt(levels.reduce((a, b) => a + (b - mean) ** 2, 0) / (levels.length - 1));
    expect(sd).toBeLessThanOrEqual(0.45);
    expect(sized.sigmaDb).toBeLessThanOrEqual(0.45);
  });

  it('the abort margin covers the scatter of two windows when they are capped', () => {
    const wide = autotuneTiming({ fs, channels: [0], rt60S: 0.05, band: BAND });
    expect(wide.abortMarginDb).toBe(AUTOTUNE_ABORT_MARGIN_DB);
    // 10 Hz of bandwidth: the 4 s cap gives B·T = 40, σ ≈ 0.69 dB, margin 1·√(150/40) ≈ 1.94 dB.
    const narrow = autotuneTiming({ fs, channels: [0], rt60S: 0.05, band: BAND, effectiveBandHz: 10 });
    expect(narrow.windowMs).toBe(MAX_WINDOW_MS);
    expect(narrow.sigmaDb).toBeCloseTo(10 / Math.LN10 / Math.sqrt(40), 6);
    expect(narrow.abortMarginDb).toBeCloseTo(Math.sqrt(AUTOTUNE_BT / 40), 6);
    expect(narrow.abortMarginDb).toBeGreaterThan(2 * Math.SQRT2 * narrow.sigmaDb - 0.01);
    // An effective bandwidth above the nominal one never shortens the windows.
    expect(autotuneTiming({ fs, channels: [0], rt60S: 0.05, band: BAND, effectiveBandHz: 5000 }).windowMs).toBe(wide.windowMs);
  });

  it('a window 1.5 dB above bypass does not abort when 2 σ of two capped windows exceed it', async () => {
    const o: AutotuneOptions = { fs, channels: [0], rt60S: 0.05, band: BAND, effectiveBandHz: 10 };
    let calls = 0;
    const driver: AutotuneDriver = {
      apply() {},
      settle: async () => undefined,
      measureErrorDb: async () => (++calls === 1 ? 1.5 : -10),
      measureBypassDb: async () => 0,
    };
    const res = await runAutotune(driver, clampParams(undefined), o);
    expect(res.aborted).toBe(false);
    const strict = await runAutotune({ ...driver, measureErrorDb: async () => 1.5 }, clampParams(undefined), { ...o, effectiveBandHz: undefined });
    expect(strict.aborted).toBe(true);
  });
});
