import { describe, expect, it } from 'vitest';
import {
  AUTOTUNE_ABORT_MARGIN_DB,
  AUTOTUNE_BT,
  AUTOTUNE_ROUNDS,
  MAX_WINDOW_MS,
  autotuneTiming,
  harmonicLevelsDb,
  harmonicTolerancesHz,
  runAutotune,
  tunedHarmonics,
} from '../../src/audio/autotune.ts';
import type { AutotuneDriver, AutotuneOptions } from '../../src/audio/autotune.ts';
import { autotuneOutcomeText } from '../../src/ui/wizard/logic.ts';
import { bandRmsDbfs, calibrationFreqs, createCalibration, noiseBandwidthHz, welchTransfer } from '../../src/audio/calibration.ts';
import { Biquad, highpass, lowpass } from '../../src/dsp/biquad.ts';
import { trimFactor } from '../../src/dsp/harmonic-trims.ts';
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

/**
 * Error-mic model of the tonal search: at harmonic h the error is E_h = 1 − Σ_ch share·(T_h/T*_h)·(g/g*)·e^{−jh(Δ−Δ*)}
 * (bypass 0 dB per harmonic, floor −40 dB), T = the channel's trim factor at h; the optimum cancels exactly.
 */
function harmonicModel(optimum: EngineParams, noiseDb = 0, seed = 1) {
  const active = ([0, 1] as const).filter((ch) => optimum.channels[ch].enabled);
  const rnd = mulberry32(seed);
  const levels = (p: EngineParams, freqs: readonly number[]): number[] =>
    freqs.map((f) => {
      const h = Math.round(f / p.tonal.f0);
      let re = 1;
      let im = 0;
      for (const ch of active) {
        const c = p.channels[ch];
        const o = optimum.channels[ch];
        const t = trimFactor(c.harmonicTrims[h - 1]);
        const t0 = trimFactor(o.harmonicTrims[h - 1]);
        const d = t0.re * t0.re + t0.im * t0.im;
        const g = (Math.pow(10, (c.gainDb - o.gainDb) / 20) * (c.polarityInvert === o.polarityInvert ? 1 : -1)) / active.length;
        const ph = (-h * (c.phaseDeltaDeg - o.phaseDeltaDeg) * Math.PI) / 180;
        // share·g·(t/t0)·e^{jph}
        const qr = (t.re * t0.re + t.im * t0.im) / d;
        const qi = (t.im * t0.re - t.re * t0.im) / d;
        re -= g * (qr * Math.cos(ph) - qi * Math.sin(ph));
        im -= g * (qr * Math.sin(ph) + qi * Math.cos(ph));
      }
      return 10 * Math.log10(re * re + im * im + 1e-4) + noiseDb * (2 * rnd() - 1);
    });
  return levels;
}

function harmonicDriver(optimum: EngineParams, noiseDb = 0) {
  const levels = harmonicModel(optimum, noiseDb);
  const log = { applied: [] as { p: EngineParams; jump: boolean }[], windows: [] as { bypass: boolean; levels: number[] }[], bandCalls: 0 };
  let current: EngineParams | null = null;
  const driver: AutotuneDriver = {
    apply: (p, jump) => {
      current = clampParams(p);
      log.applied.push({ p: current, jump });
    },
    settle: async () => {},
    measureErrorDb: async () => {
      log.bandCalls++;
      return 0;
    },
    measureBypassDb: async () => {
      log.bandCalls++;
      return 0;
    },
    measureHarmonicsDb: async (_ms, freqs, bypass) => {
      const v = bypass ? freqs.map(() => 0) : levels(current!, freqs);
      log.windows.push({ bypass, levels: v });
      return v;
    },
  };
  return { driver, log, current: () => current, levels };
}

describe('runAutotune: tonal mode, per-harmonic trims', () => {
  const tonal = (over: (p: EngineParams) => void = () => {}): EngineParams =>
    params((p) => {
      p.phaseMode = 'tonal';
      p.tonal = { f0: 100, harmonics: 4, q: 30 };
      p.channels[0].delayMs = 3;
      p.channels[0].gainDb = -8;
      p.channels[0].harmonicTrims = [
        { gainDb: 1.5, phaseDeg: -12 },
        { gainDb: -2, phaseDeg: 30 },
        { gainDb: 0.5, phaseDeg: 70 },
        { gainDb: 3, phaseDeg: -40 },
        { gainDb: 0, phaseDeg: 0 },
        { gainDb: 0, phaseDeg: 0 },
        { gainDb: 0, phaseDeg: 0 },
        { gainDb: 0, phaseDeg: 0 },
      ];
      over(p);
    });
  const optimum = tonal();
  /** The optimum with +20° on harmonic 3 and −3 dB on harmonic 2 (and a little on the others). */
  const detuned = tonal((p) => {
    const t = p.channels[0].harmonicTrims;
    t[0].phaseDeg += 6;
    t[1].gainDb -= 3;
    t[2].phaseDeg += 20;
    t[3].gainDb += 1.5;
  });
  const fourHarmonics = (p: EngineParams): number[] => [1, 2, 3, 4].map((h) => h * p.tonal.f0);

  it('refines every harmonic at once from per-harmonic levels; Δ, delay and gain stay', async () => {
    const { driver, log, levels } = harmonicDriver(optimum);
    const start = levels(detuned, fourHarmonics(detuned));
    expect(Math.max(...start)).toBeGreaterThan(-12);
    const res = await runAutotune(driver, detuned, opts);
    expect(res.aborted).toBe(false);
    const end = levels(res.params, fourHarmonics(res.params));
    for (const db of end) expect(db).toBeLessThan(-30);
    for (let h = 0; h < 4; h++) {
      const a = res.params.channels[0].harmonicTrims[h];
      const b = optimum.channels[0].harmonicTrims[h];
      expect(Math.abs(a.gainDb - b.gainDb)).toBeLessThan(0.3);
      expect(Math.abs(a.phaseDeg - b.phaseDeg)).toBeLessThan(2);
    }
    expect(res.params.channels[0].harmonicTrims.slice(4)).toEqual(detuned.channels[0].harmonicTrims.slice(4));
    expect(res.params.channels[0].delayMs).toBe(3);
    expect(res.params.channels[0].phaseDeltaDeg).toBe(0);
    expect(res.params.channels[0].gainDb).toBe(-8);
    // The band level is not used; every window reads the four harmonics.
    expect(log.bandCalls).toBe(0);
    expect(log.windows.every((w) => w.levels.length === 4)).toBe(true);
    // Per-harmonic report: start, end (last window) and bypass, and their power sums.
    expect(res.harmonics!.map((x) => x.h)).toEqual([1, 2, 3, 4]);
    expect(res.harmonics!.map((x) => x.freqHz)).toEqual([100, 200, 300, 400]);
    res.harmonics!.forEach((x, i) => {
      expect(x.startDb).toBeCloseTo(start[i], 9);
      expect(x.endDb).toBeCloseTo(end[i], 9);
      expect(x.bypassDb).toBeCloseTo(0, 9);
    });
    const sum = (v: number[]): number => 10 * Math.log10(v.reduce((a, db) => a + Math.pow(10, db / 10), 0));
    expect(res.startDb).toBeCloseTo(sum(start), 9);
    expect(res.endDb).toBeCloseTo(sum(end), 9);
    expect(res.steps).toBeLessThan(25);
  });

  it('a harmonic already cancelled is probed gently (the step follows its residual) and stays cancelled', async () => {
    const { driver, log, levels } = harmonicDriver(optimum);
    const near = tonal((p) => (p.channels[0].harmonicTrims[2].phaseDeg += 3));
    const res = await runAutotune(driver, near, opts);
    expect(res.aborted).toBe(false);
    // Harmonics 1, 2 and 4 sit at the −40 dB floor: their probes stay below −20 dB; harmonic 3 (−25.6 dB) below −14 dB.
    for (const w of log.windows.filter((x) => !x.bypass)) {
      expect(Math.max(w.levels[0], w.levels[1], w.levels[3])).toBeLessThan(-20);
      expect(w.levels[2]).toBeLessThan(-14);
    }
    for (const db of levels(res.params, fourHarmonics(res.params))) expect(db).toBeLessThan(-35);
  });

  it('copes with measurement noise (±0.3 dB) and never plays a window above bypass + 1 dB', async () => {
    const { driver, log } = harmonicDriver(optimum, 0.3);
    const res = await runAutotune(driver, detuned, opts);
    expect(res.aborted).toBe(false);
    for (const x of res.harmonics!) expect(x.bypassDb - x.endDb).toBeGreaterThan(20);
    const sum = (v: number[]): number => 10 * Math.log10(v.reduce((a, db) => a + Math.pow(10, db / 10), 0));
    for (const w of log.windows) if (!w.bypass) expect(sum(w.levels)).toBeLessThanOrEqual(sum([0, 0, 0, 0]) + 1);
  });

  it('two speakers on one zone: one channel at a time with the other active', async () => {
    const both = (p: EngineParams): void => {
      p.channels[1] = { ...p.channels[0], harmonicTrims: p.channels[0].harmonicTrims.map((t) => ({ gainDb: -t.gainDb, phaseDeg: t.phaseDeg / 2 })) };
    };
    const opt2 = tonal(both);
    const start = tonal((p) => {
      both(p);
      p.channels[0].harmonicTrims[1].gainDb -= 3;
      p.channels[1].harmonicTrims[2].phaseDeg += 25;
    });
    const { driver, log, levels } = harmonicDriver(opt2);
    const res = await runAutotune(driver, start, { ...opts, channels: [0, 1] });
    expect(res.aborted).toBe(false);
    for (const db of levels(res.params, fourHarmonics(res.params))) expect(db).toBeLessThan(-25);
    // Within a channel's search only its trims move: both change between two played settings only where the
    // search passes from one channel to the other (at most twice per round).
    const trimsOf = (p: EngineParams, ch: 0 | 1): string => JSON.stringify(p.channels[ch].harmonicTrims);
    let switches = 0;
    for (let i = 1; i < log.applied.length; i++) {
      const [a, b] = [log.applied[i - 1].p, log.applied[i].p];
      if (trimsOf(a, 0) !== trimsOf(b, 0) && trimsOf(a, 1) !== trimsOf(b, 1)) switches++;
      expect(b.channels[0].enabled && b.channels[1].enabled).toBe(true);
    }
    expect(switches).toBeLessThanOrEqual(2 * AUTOTUNE_ROUNDS);
    // Both channels were searched (two speakers on one zone have a valley of equivalent settings: the
    // levels are judged, not the values).
    for (const ch of [0, 1] as const) expect(log.applied.some((x) => trimsOf(x.p, ch) !== trimsOf(start, ch))).toBe(true);
  });

  it('keeps the safety rules: the loop guard, the bypass + 1 dB abort and the restore', async () => {
    // Guard: no trim gain above +3.5 dB is ever played (the start's harmonic 4 is at +4.5).
    const guard = (p: EngineParams): boolean => p.channels[0].harmonicTrims.every((t) => t.gainDb <= 4.5 + 1e-9);
    const a = harmonicDriver(optimum);
    const guarded = await runAutotune(a.driver, detuned, { ...opts, allowed: guard });
    expect(guarded.aborted).toBe(false);
    for (const x of a.log.applied.slice(1)) expect(guard(x.p)).toBe(true);
    // A disturbance: one window 3 dB above bypass aborts and restores the start.
    const b = harmonicDriver(optimum);
    const m = b.driver.measureHarmonicsDb!;
    let n = 0;
    b.driver.measureHarmonicsDb = async (ms, freqs, bypass) => (!bypass && ++n === 5 ? freqs.map(() => 3) : m(ms, freqs, bypass));
    const aborted = await runAutotune(b.driver, detuned, opts);
    expect(aborted.aborted).toBe(true);
    expect(aborted.reason).toMatch(/bypass/);
    expect(aborted.params).toEqual(detuned);
    expect(b.current()).toEqual(detuned);
    expect(b.log.applied[b.log.applied.length - 1].jump).toBe(true);
    expect(aborted.harmonics!.map((x) => x.endDb)).toEqual([3, 3, 3, 3]);
    // Invalid per-harmonic data (wrong length, NaN) stops and restores too.
    const c = harmonicDriver(optimum);
    c.driver.measureHarmonicsDb = async (_ms, freqs) => freqs.slice(1).map(() => -10);
    const invalid = await runAutotune(c.driver, detuned, opts);
    expect(invalid.aborted).toBe(true);
    expect(invalid.reason).toMatch(/non valida/);
    expect(c.current()).toEqual(detuned);
  });

  it('a point assembled from several windows is played only when allowed (else the best allowed combination)', async () => {
    // Harmonic 1's optimum lags 8° more than the start, harmonic 2's 8° less: the search keeps harmonic 1 from
    // one probe window and harmonic 2 from another. The guard (standing for a loop that adds up between two
    // overlapping bands) rejects the trims of the two moving apart, which no probe does by itself.
    const start = tonal();
    const target = tonal((p) => {
      p.channels[0].harmonicTrims[0].phaseDeg += 8;
      p.channels[0].harmonicTrims[1].phaseDeg -= 8;
    });
    const moved = (p: EngineParams, h: number): number => p.channels[0].harmonicTrims[h - 1].phaseDeg - start.channels[0].harmonicTrims[h - 1].phaseDeg;
    const guard = (p: EngineParams): boolean => !(moved(p, 1) * moved(p, 2) < -1);
    const { driver, log, levels } = harmonicDriver(target);
    const res = await runAutotune(driver, start, { ...opts, allowed: guard });
    expect(res.aborted).toBe(false);
    for (const x of log.applied) expect(guard(x.p), JSON.stringify(x.p.channels[0].harmonicTrims.slice(0, 2))).toBe(true);
    expect(guard(res.params)).toBe(true);
    // still better than the start: harmonic 1 moved towards its optimum, harmonic 2 left where the guard keeps it
    const sum = (v: number[]): number => 10 * Math.log10(v.reduce((a, db) => a + Math.pow(10, db / 10), 0));
    expect(sum(levels(res.params, fourHarmonics(res.params)))).toBeLessThan(sum(levels(start, fourHarmonics(start))) - 0.5);
    // without the guard the assembled point is the answer (both harmonics found)
    const free = harmonicDriver(target);
    const unguarded = await runAutotune(free.driver, start, opts);
    expect(moved(unguarded.params, 1)).toBeGreaterThan(6);
    expect(moved(unguarded.params, 2)).toBeLessThan(-6);
  });

  it('without per-harmonic levels (or outside tonal mode) the search is the coordinate one', async () => {
    const { driver, log } = harmonicDriver(optimum);
    const plain: AutotuneDriver = { ...driver, measureHarmonicsDb: undefined };
    await runAutotune(plain, detuned, opts);
    expect(log.bandCalls).toBeGreaterThan(0);
    expect(log.windows.length).toBe(0);
    const delay = harmonicDriver(optimum);
    await runAutotune(delay.driver, params((p) => (p.channels[0].delayMs = 3)), opts);
    expect(delay.log.windows.length).toBe(0);
  });
});

/**
 * The ambient level changes by `dbPerS` (dB per second of the run's clock: the settling and windows asked of
 * the driver): every window, bypassed or not, moves with it. `windows` logs each one (centre time, ms).
 */
function ambientRamp(driver: AutotuneDriver, dbPerS: number) {
  let now = 0;
  const windows: { t: number; bypass: boolean; db: number; levels: number[] | null }[] = [];
  const window = (ms: number): number => {
    const t = now + ms / 2;
    now += ms;
    return t;
  };
  const ramped: AutotuneDriver = {
    ...driver,
    settle: async (ms) => {
      now += ms;
      await driver.settle(ms);
    },
    measureErrorDb: async (ms) => {
      const t = window(ms);
      const db = (await driver.measureErrorDb(ms)) + (dbPerS * t) / 1000;
      windows.push({ t, bypass: false, db, levels: null });
      return db;
    },
    measureBypassDb: async (ms) => {
      const t = window(ms);
      const db = (await driver.measureBypassDb(ms)) + (dbPerS * t) / 1000;
      windows.push({ t, bypass: true, db, levels: null });
      return db;
    },
  };
  if (driver.measureHarmonicsDb !== undefined) {
    const m = driver.measureHarmonicsDb;
    ramped.measureHarmonicsDb = async (ms, freqs, bypass, tolHz) => {
      const t = window(ms);
      const levels = (await m(ms, freqs, bypass, tolHz)).map((db) => db + (dbPerS * t) / 1000);
      windows.push({ t, bypass, db: 10 * Math.log10(levels.reduce((a, db) => a + Math.pow(10, db / 10), 0)), levels });
      return levels;
    };
  }
  return { driver: ramped, windows };
}

/** Largest excess of an error window over the latest bypass window before it (the rule before the interpolation). */
function excessOverLatestBypass(windows: readonly { bypass: boolean; db: number }[]): number {
  let latest = Number.NaN;
  let worst = -Infinity;
  for (const w of windows) {
    if (w.bypass) latest = w.db;
    else worst = Math.max(worst, w.db - latest);
  }
  return worst;
}

describe('runAutotune: the bypass reference follows the ambient level (interpolated in time)', () => {
  const tonal = (over: (p: EngineParams) => void = () => {}): EngineParams =>
    params((p) => {
      p.phaseMode = 'tonal';
      p.tonal = { f0: 100, harmonics: 4, q: 30 };
      p.channels[0].gainDb = -8;
      p.channels[0].harmonicTrims[1] = { gainDb: -2, phaseDeg: 30 };
      p.channels[0].harmonicTrims[2] = { gainDb: 0.5, phaseDeg: 70 };
      over(p);
    });
  const optimum = tonal();
  const detuned = tonal((p) => {
    p.channels[0].harmonicTrims[1].gainDb -= 3;
    p.channels[0].harmonicTrims[2].phaseDeg += 20;
  });
  const sum = (v: readonly number[]): number => 10 * Math.log10(v.reduce((a, db) => a + Math.pow(10, db / 10), 0));

  it('per harmonic: attenuation against the bypass level at the end window, not the mean of the run (+5 dB ramp)', async () => {
    const base = harmonicDriver(optimum);
    const { driver, windows } = ambientRamp(base.driver, 0.4);
    const res = await runAutotune(driver, detuned, opts);
    expect(res.aborted).toBe(false);
    const byp = windows.filter((w) => w.bypass);
    expect(byp[byp.length - 1].db - byp[0].db).toBeGreaterThan(5);
    // The model's attenuation (bypass 0 dB per harmonic) is what is reported, at the end and at the start.
    const freqs = [100, 200, 300, 400];
    const end = base.levels(res.params, freqs);
    const start = base.levels(detuned, freqs);
    res.harmonics!.forEach((x, i) => {
      expect(x.bypassDb - x.endDb).toBeCloseTo(-end[i], 6);
      expect(x.startBypassDb! - x.startDb).toBeCloseTo(-start[i], 6);
    });
    expect(res.bypassDb - res.endDb).toBeCloseTo(sum([0, 0, 0, 0]) - sum(end), 6);
    // The final window is followed by a bypass window (its reference is interpolated, not extrapolated).
    expect(windows[windows.length - 1].bypass).toBe(true);
    // The power mean of all bypass windows (the reference before) understated the attenuation by > 1.5 dB.
    const meanDb = (i: number): number => 10 * Math.log10(byp.reduce((a, w) => a + Math.pow(10, w.levels![i] / 10), 0) / byp.length);
    res.harmonics!.forEach((x, i) => expect(x.bypassDb - meanDb(i)).toBeGreaterThan(1.5));
  });

  it('abort rule: a rising ambient level does not stop a run whose anti-noise attenuates 0.5 dB', async () => {
    const t = autotuneTiming(opts);
    const flat: AutotuneDriver = {
      apply() {},
      settle: async () => undefined,
      measureErrorDb: async () => -0.5,
      measureBypassDb: async () => 0,
    };
    // 2 dB per BYPASS_EVERY (6) windows and their settling.
    const { driver, windows } = ambientRamp(flat, 2000 / (6 * (t.windowMs + 2 * t.settleMs)));
    const res = await runAutotune(driver, clampParams(undefined), opts);
    expect(res.aborted).toBe(false);
    // The latest bypass window alone would have stopped it.
    expect(excessOverLatestBypass(windows)).toBeGreaterThan(t.abortMarginDb);
    // Against the level at its time every window is 0.5 dB below bypass.
    expect(res.bypassDb - res.endDb).toBeCloseTo(0.5, 6);
  });

  it('abort rule: a falling ambient level does not hide an anti-noise that adds 1.3 dB', async () => {
    const t = autotuneTiming(opts);
    let current: EngineParams | null = null;
    const applied: EngineParams[] = [];
    const louder: AutotuneDriver = {
      apply(p) {
        current = clampParams(p);
        applied.push(current);
      },
      settle: async () => undefined,
      measureErrorDb: async () => 1.3,
      measureBypassDb: async () => 0,
    };
    const { driver, windows } = ambientRamp(louder, -500 / (t.windowMs + 2 * t.settleMs));
    const start = clampParams(undefined);
    const res = await runAutotune(driver, start, opts);
    // The latest bypass window alone never saw it…
    expect(excessOverLatestBypass(windows)).toBeLessThan(t.abortMarginDb);
    // …the next bypass window does: stopped there, the start restored.
    expect(res.aborted).toBe(true);
    expect(res.reason).toMatch(/supera il bypass/);
    expect(windows[windows.length - 1].bypass).toBe(true);
    expect(windows.filter((w) => w.bypass)).toHaveLength(2);
    expect(current).toEqual(start);
    expect(res.bypassDb - res.endDb).toBeCloseTo(-1.3, 6);
  });

  /**
   * A driver whose harmonics sit at least `floorDb` under bypass (the residual's floor: little left to gain),
   * with seeded noise; `start` and `p` give the model's own power sum against bypass (the truth).
   */
  function floored(floorDb: number, noiseDb: number) {
    const base = harmonicDriver(optimum, noiseDb);
    const clean = harmonicModel(optimum);
    const pow = (db: number): number => Math.pow(10, db / 10);
    const driver: AutotuneDriver = {
      ...base.driver,
      measureHarmonicsDb: async (ms, f, bypass, tol) =>
        (await base.driver.measureHarmonicsDb!(ms, f, bypass, tol)).map((db) => (bypass ? db : 10 * Math.log10(pow(db) + pow(floorDb)))),
    };
    const truth = (p: EngineParams): number => sum(clean(p, [100, 200, 300, 400]).map((db) => 10 * Math.log10(pow(db) + pow(floorDb))));
    return { driver, truth, current: base.current };
  }

  /** The run's length on the run's clock (ms) at a flat ambient level, for a ramp of `db` over it. */
  async function rampOverRun(make: () => AutotuneDriver, start: EngineParams, db: number) {
    let ms = 0;
    const flat = make();
    await runAutotune({ ...flat, settle: async (x) => void (ms += x), measureHarmonicsDb: async (x, f, b, t) => ((ms += x), flat.measureHarmonicsDb!(x, f, b, t)) }, start, opts);
    return ambientRamp(make(), (1000 * db) / ms);
  }

  it('keep/apply against the bypass level at each window’s time: a falling ambient does not pass the start off as improved (F2)', async () => {
    // Started at the optimum, −6 dB of ambient over the run: the raw end window read 5 dB below the start's, the
    // run kept a point 0.35 dB worse and step 6 applied it as «valori migliorati».
    const { driver, windows } = await rampOverRun(() => floored(-12, 0.2).driver, optimum, -6);
    const f = floored(-12, 0.2);
    const res = await runAutotune(driver, optimum, opts);
    expect(res.aborted).toBe(false);
    const byp = windows.filter((w) => w.bypass);
    expect(byp[byp.length - 1].db - byp[0].db).toBeLessThan(-5);
    // Raw, the final window read far below the start window.
    const errs = windows.filter((w) => !w.bypass);
    expect(errs[errs.length - 1].db).toBeLessThan(errs[0].db - 4);
    expect(res.improved).toBe(false);
    expect(res.params).toEqual(clampParams(optimum));
    expect(f.truth(res.params)).toBeCloseTo(f.truth(optimum), 9);
    expect(autotuneOutcomeText(res)).toMatch(/nessun miglioramento, valori invariati\.$/);
  });

  it('a rising ambient does not throw a real improvement away (F2)', async () => {
    // Harmonic 2 detuned (7 dB instead of 12): +6 dB of ambient over the run made every later window read louder
    // than the start, and the run restored the start although it had played the better point.
    const start = tonal((p) => (p.channels[0].harmonicTrims[1].gainDb -= 4));
    const f = floored(-12, 0.2);
    const { driver } = await rampOverRun(() => floored(-12, 0.2).driver, start, 6);
    const res = await runAutotune(driver, start, opts);
    expect(res.aborted).toBe(false);
    expect(res.endDb).toBeGreaterThan(res.startDb);
    expect(res.improved).toBe(true);
    expect(f.truth(res.params)).toBeLessThan(f.truth(start) - 1);
    expect(res.bypassDb - res.endDb).toBeGreaterThan(res.startBypassDb - res.startDb + 1);
    expect(autotuneOutcomeText(res)).toMatch(/^Attenuazione nella zona da \+\d+,\d dB a \+\d+,\d dB \(rispetto al bypass all’inizio e alla fine\): valori migliorati applicati\.$/);
  });

  it('a final window that does not confirm the search restores the start (improved false, the start reported)', async () => {
    const start = tonal((p) => (p.channels[0].harmonicTrims[1].gainDb -= 4));
    const base = harmonicDriver(optimum);
    let errors = 0;
    let last = 0;
    // The point the search ends on reads 3 dB worse in its final window (a door slammed): not kept.
    const driver: AutotuneDriver = {
      ...base.driver,
      measureHarmonicsDb: async (ms, f, bypass, tol) => {
        const v = await base.driver.measureHarmonicsDb!(ms, f, bypass, tol);
        if (!bypass) errors++;
        return v;
      },
    };
    const probe = await runAutotune(driver, start, opts);
    expect(probe.improved).toBe(true);
    last = errors;
    errors = 0;
    const spoiled: AutotuneDriver = {
      ...driver,
      measureHarmonicsDb: async (ms, f, bypass, tol) => {
        const v = await driver.measureHarmonicsDb!(ms, f, bypass, tol);
        return !bypass && errors === last ? v.map(() => -1) : v;
      },
    };
    const res = await runAutotune(spoiled, start, opts);
    expect(res.aborted).toBe(false);
    expect(res.improved).toBe(false);
    expect(res.params).toEqual(clampParams(start));
    expect(base.current()).toEqual(clampParams(start));
    expect(base.log.applied[base.log.applied.length - 1].jump).toBe(true);
    // Reported: the params in place, i.e. the start window again.
    expect(res.endDb).toBe(res.startDb);
    expect(res.bypassDb).toBe(res.startBypassDb);
  });

  it('a transient in the closing bypass window is measured again, not taken as the end reference (F3)', async () => {
    const start = detuned;
    const count = harmonicDriver(optimum);
    await runAutotune(count.driver, start, opts);
    const closing = count.log.windows.filter((w) => w.bypass).length - 1;
    let k = 0;
    const base = harmonicDriver(optimum);
    const driver: AutotuneDriver = {
      ...base.driver,
      // +6 dB on every harmonic of the closing bypass window only (a passing vehicle).
      measureHarmonicsDb: async (ms, f, bypass, tol) => {
        const v = await base.driver.measureHarmonicsDb!(ms, f, bypass, tol);
        return bypass && k++ === closing ? v.map((db) => db + 6) : v;
      },
    };
    const res = await runAutotune(driver, start, opts);
    expect(res.aborted).toBe(false);
    // One bypass window more than the clean run: the closing one again.
    expect(base.log.windows.filter((w) => w.bypass).length).toBe(closing + 2);
    const truth = base.levels(res.params, [100, 200, 300, 400]);
    // Reported 45.2 dB per harmonic against the model's 40 before (+5.2 dB; the old power mean: +2.4 dB).
    res.harmonics!.forEach((x, i) => expect(x.bypassDb - x.endDb).toBeCloseTo(-truth[i], 6));
  });
});

describe('harmonic levels', () => {
  it('harmonicLevelsDb reads each tone of a mixture by synchronous detection (dBFS RMS)', () => {
    const n = fs / 2;
    const rnd = mulberry32(5);
    const x = Float32Array.from({ length: n }, (_, i) =>
      0.5 * Math.sin((2 * Math.PI * 100 * i) / fs) + 0.05 * Math.sin((2 * Math.PI * 300 * i) / fs + 1) + 0.01 * (2 * rnd() - 1),
    );
    const [a, b, c] = harmonicLevelsDb(x, fs, [100, 300, 700]);
    expect(a).toBeCloseTo(20 * Math.log10(0.5 / Math.SQRT2), 1);
    expect(b).toBeCloseTo(20 * Math.log10(0.05 / Math.SQRT2), 1);
    expect(c).toBeLessThan(-60);
    expect(harmonicLevelsDb(new Float32Array(100), fs, [100])).toEqual([-200]);
  });

  it('harmonicLevelsDb with a drift tolerance reads a tone that moved off its nominal frequency', () => {
    // 4 s windows (the wizard's for tonal noise): a tone 0.8 Hz off reads ≈ 30 dB low at the exact frequency.
    const n = 4 * fs;
    const rnd = mulberry32(9);
    const tone = (f: number, a: number) => Float32Array.from({ length: n }, (_, i) => a * Math.sin((2 * Math.PI * f * i) / fs) + 1e-4 * (2 * rnd() - 1));
    const want = 20 * Math.log10(0.01 / Math.SQRT2);
    const off = tone(400.8, 0.01);
    expect(harmonicLevelsDb(off, fs, [400])[0]).toBeLessThan(want - 20);
    const [got] = harmonicLevelsDb(off, fs, [400], [4]);
    expect(Math.abs(got - want)).toBeLessThan(0.15);
    // on the nominal frequency both agree; outside the tolerance nothing is found
    const on = tone(400, 0.01);
    expect(Math.abs(harmonicLevelsDb(on, fs, [400], 4)[0] - harmonicLevelsDb(on, fs, [400])[0])).toBeLessThan(0.15);
    expect(harmonicLevelsDb(tone(410, 0.01), fs, [400], [4])[0]).toBeLessThan(want - 40);
    // noise alone: the tolerant reading is higher (pessimistic), by a few dB at most
    const noise = Float32Array.from({ length: n }, () => 0.01 * (2 * rnd() - 1));
    const [exact] = harmonicLevelsDb(noise, fs, [400]);
    const [wide] = harmonicLevelsDb(noise, fs, [400], [4]);
    expect(wide).toBeGreaterThan(exact - 3);
    expect(wide).toBeLessThan(exact + 12);
  });

  it('harmonicTolerancesHz: the resonator half bandwidth f/(2Q), at most 1 % of f', () => {
    const p = params((q) => {
      q.phaseMode = 'tonal';
      q.tonal = { f0: 100, harmonics: 4, q: 30 };
    });
    // 100 Hz: 1.67 Hz half bandwidth, 1 Hz cap; 400 Hz: 6.67 Hz, 4 Hz cap
    expect(harmonicTolerancesHz(p, [100, 400])).toEqual([1, 4]);
    p.tonal.q = 80;
    expect(harmonicTolerancesHz(p, [400])[0]).toBeCloseTo(2.5, 12);
    p.tonal.q = 200;
    expect(harmonicTolerancesHz(p, [100])[0]).toBeCloseTo(0.25, 12);
  });

  it('tunedHarmonics: h ≤ tonal.harmonics inside the band and below 0.45·fs', () => {
    const p = params((q) => {
      q.phaseMode = 'tonal';
      q.tonal = { f0: 90, harmonics: 6, q: 30 };
    });
    expect(tunedHarmonics(p, [100, 400], fs).map((x) => x.h)).toEqual([2, 3, 4]);
    expect(tunedHarmonics(p, [20, 2000], 1000).map((x) => x.h)).toEqual([1, 2, 3, 4]);
    expect(tunedHarmonics(p, [20, 2000], fs).map((x) => x.freqHz)).toEqual([90, 180, 270, 360, 450, 540]);
  });
});
