import { describe, expect, it } from 'vitest';
import { Biquad, biquadResponse, highpass, lowpass } from '../../src/dsp/biquad.ts';
import type { BiquadCoeffs } from '../../src/dsp/biquad.ts';
import { cAbs } from '../../src/dsp/complex.ts';
import {
  HOWL_LONG_LOOP_MAX_MS,
  HOWL_NARROW_BAND_HZ,
  HOWL_PERIODIC_RHO,
  HOWL_SAT_GR_DB,
  HOWL_WIDE_BAND_HZ,
  HowlDetector,
  PeriodicityAnalyzer,
  howlBandHz,
} from '../../src/dsp/howl-detector.ts';
import { PeakLimiter } from '../../src/dsp/limiter.ts';
import { clampParams } from '../../src/dsp/params.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const RATES = [44100, 48000, 96000] as const;

function gaussian(seed: number): () => number {
  const rnd = mulberry32(seed);
  return () => Math.sqrt(-2 * Math.log(Math.max(1e-12, rnd()))) * Math.cos(2 * Math.PI * rnd());
}

/** Bessel-like 2nd-order low-pass at 500 Hz, the default conditioning of the engine. */
const lp500 = (fs: number): Biquad => new Biquad(lowpass(fs, 1.272 * 500, 1 / Math.sqrt(3)));

interface Sample {
  x: number;
  gainDb: number;
  grDb: number;
  ceilingDb: number;
}

interface RunResult {
  /** Time of the first trigger in seconds, or null. */
  at: number | null;
  reason: string | null;
  freqHz: number | null;
}

function run(fs: number, seconds: number, gen: (t: number) => Sample): RunResult {
  const det = new HowlDetector(fs);
  const n = Math.round(seconds * fs);
  for (let i = 0; i < n; i++) {
    const s = gen(i / fs);
    det.push(s.x, s.gainDb, s.grDb, s.ceilingDb);
    if (det.state.triggered) return { at: i / fs, reason: det.state.reason, freqHz: det.state.freqHz };
  }
  return { at: null, reason: null, freqHz: null };
}

const db = (lin: number): number => (lin > 0 ? 20 * Math.log10(lin) : -Infinity);

describe('HowlDetector: no false triggers', () => {
  for (const fs of RATES) {
    for (const rampDbPerS of [6, 20]) {
      it(`steady 100 Hz tone through a 1.5 s soft-start and a +20 dB ramp at ${rampDbPerS} dB/s (${fs} Hz)`, () => {
        const g = gaussian(1);
        const f = lp500(fs);
        const rampStart = 2;
        const rampEnd = rampStart + 20 / rampDbPerS;
        const r = run(fs, rampEnd + 1.5, (t) => {
          const env = Math.min(1, t / 1.5);
          const gDb = -26 + 20 * Math.min(1, Math.max(0, (t - rampStart) / (rampEnd - rampStart)));
          const commanded = db(env) + gDb;
          const ref = 0.5 * Math.sin(2 * Math.PI * 100 * t + 0.3) + f.process(0.002 * g());
          return { x: ref * Math.pow(10, commanded / 20), gainDb: commanded, grDb: 0, ceilingDb: -1 };
        });
        expect(r.at).toBeNull();
      });
    }

    it(`broadband noise, white and low-passed, with soft-start (${fs} Hz)`, () => {
      const g = gaussian(2);
      const f = lp500(fs);
      const white = run(fs, 5, () => ({ x: 0.1 * g(), gainDb: 0, grDb: 0, ceilingDb: -1 }));
      expect(white.at).toBeNull();
      const lowPassed = run(fs, 5, (t) => {
        const env = Math.min(1, t / 1.5);
        return { x: env * f.process(0.3 * g()), gainDb: db(env), grDb: 0, ceilingDb: -1 };
      });
      expect(lowPassed.at).toBeNull();
    });

    it(`two steady tones beating 1 Hz apart (25 dB envelope swing) do not trigger (${fs} Hz)`, () => {
      const g = gaussian(8);
      const f = lp500(fs);
      const r = run(fs, 12, (t) => ({
        x: 0.05 * Math.sin(2 * Math.PI * 200 * t) + 0.045 * Math.sin(2 * Math.PI * 201 * t) + f.process(0.01 * g()),
        gainDb: -6,
        grDb: 0,
        ceilingDb: -1,
      }));
      expect(r.at).toBeNull();
    });

    it(`a steady weak tone ~15 dB above the noise does not trigger (${fs} Hz)`, () => {
      const g = gaussian(3);
      const f = lp500(fs);
      const r = run(fs, 10, (t) => ({
        x: 0.003 * Math.sin(2 * Math.PI * 230 * t) + f.process(0.02 * g()),
        gainDb: -6,
        grDb: 0,
        ceilingDb: -1,
      }));
      expect(r.at).toBeNull();
    });
  }
});

describe('HowlDetector: triggers', () => {
  for (const fs of RATES) {
    it(`'peak-growth' within 600 ms for a tone growing 20 dB/s at constant gain (${fs} Hz)`, () => {
      const g = gaussian(4);
      const f = lp500(fs);
      const growStart = 1;
      const r = run(fs, 3, (t) => {
        const levelDb = -36 + 20 * Math.max(0, t - growStart);
        const x = Math.pow(10, levelDb / 20) * Math.sin(2 * Math.PI * 700 * t) + f.process(0.01 * g());
        return { x, gainDb: -10, grDb: 0, ceilingDb: -1 };
      });
      expect(r.reason).toBe('peak-growth');
      expect(r.at! - growStart).toBeGreaterThan(0.2);
      expect(r.at! - growStart).toBeLessThan(0.6);
      expect(Math.abs(r.freqHz! - 700)).toBeLessThan(6);
    });

    it(`'peak-growth' for a howl that starts from nothing (${fs} Hz)`, () => {
      const g = gaussian(5);
      const f = lp500(fs);
      const start = 0.5;
      const r = run(fs, 3, (t) => {
        const tone = t < start ? 0 : Math.pow(10, (-38 + 25 * (t - start)) / 20) * Math.sin(2 * Math.PI * 1234 * t);
        return { x: tone + f.process(0.01 * g()), gainDb: 0, grDb: 0, ceilingDb: -1 };
      });
      expect(r.reason).toBe('peak-growth');
      expect(r.at! - start).toBeLessThan(0.6);
      expect(Math.abs(r.freqHz! - 1234)).toBeLessThan(6);
    });

    it(`'limiter-saturated' for a tone held at the ceiling with 6 dB of GR for 400 ms (${fs} Hz)`, () => {
      const ceilingDb = -12;
      const ceil = Math.pow(10, ceilingDb / 20);
      const loudFrom = 0.5;
      const r = run(fs, loudFrom + 0.4, (t) => {
        const loud = t >= loudFrom;
        const amp = loud ? 2 * ceil : 0.05;
        return { x: amp * Math.sin(2 * Math.PI * 300 * t), gainDb: -6, grDb: loud ? 6 : 0, ceilingDb };
      });
      expect(r.reason).toBe('limiter-saturated');
      expect(r.at! - loudFrom).toBeGreaterThan(0.299);
      expect(r.at! - loudFrom).toBeLessThan(0.4);
      expect(Math.abs(r.freqHz! - 300)).toBeLessThan(6);
    });

    it(`'output-overload' for broadband RMS near the ceiling (${fs} Hz)`, () => {
      const g = gaussian(6);
      const ceilingDb = -12;
      const rms = Math.pow(10, (ceilingDb - 2) / 20);
      const r = run(fs, 1, () => ({ x: rms * g(), gainDb: -6, grDb: 0, ceilingDb }));
      expect(r.reason).toBe('output-overload');
      expect(r.at!).toBeGreaterThan(0.2);
      expect(r.at!).toBeLessThan(0.35);
      expect(r.freqHz).toBeNull();
    });

    it(`'output-overload' for a loud tone that the limiter does not reduce (${fs} Hz)`, () => {
      const ceil = Math.pow(10, -12 / 20);
      const r = run(fs, 1, (t) => ({ x: 0.95 * ceil * Math.sin(2 * Math.PI * 150 * t), gainDb: -6, grDb: 0, ceilingDb: -12 }));
      expect(r.reason).toBe('output-overload');
      expect(r.at!).toBeLessThan(0.35);
    });
  }

  it('latches until reset() and then starts over', () => {
    const fs = 48000;
    const det = new HowlDetector(fs);
    const ceil = Math.pow(10, -12 / 20);
    for (let i = 0; i < fs; i++) det.push(ceil * Math.sin((2 * Math.PI * 150 * i) / fs), 0, 0, -12);
    expect(det.state.triggered).toBe(true);
    expect(det.state.reason).toBe('output-overload');
    for (let i = 0; i < fs; i++) det.push(0, 0, 0, -12);
    expect(det.state.triggered).toBe(true);
    det.reset();
    expect(det.state).toEqual({ triggered: false, reason: null, freqHz: null });
    for (let i = 0; i < fs; i++) det.push(0.001 * Math.sin((2 * Math.PI * 150 * i) / fs), 0, 0, -12);
    expect(det.state.triggered).toBe(false);
  });

  it('holdLevels: a level that changes with the conditioning response is no growth; the tracks resume after it (v-dsp H2)', () => {
    const fs = 48000;
    const g = gaussian(9);
    const f = lp500(fs);
    // A 50 Hz tone rising exponentially by 20 dB in 1 s at constant commanded gain (a filter dragged open).
    const rising = (t: number): number => {
      const levelDb = -40 + 20 * Math.min(1, Math.max(0, t - 1));
      return Math.pow(10, levelDb / 20) * Math.sin(2 * Math.PI * 50 * t) + f.process(0.001 * g());
    };
    const free = run(fs, 3, (t) => ({ x: rising(t), gainDb: -6, grDb: 0, ceilingDb: -1 }));
    expect(free.reason).toBe('peak-growth');
    // The same rise in 60 steps of a live filter change, each held for 40 ms.
    const det = new HowlDetector(fs);
    const step = Math.round(fs / 60);
    for (let i = 0; i < 3 * fs; i++) {
      const t = i / fs;
      if (t >= 1 && t <= 2 && i % step === 0) det.holdLevels(40);
      det.push(rising(t), -6, 0, -1);
    }
    expect(det.state.triggered).toBe(false);
    // Afterwards the level tracks work again: a howl growing 20 dB/s from the new level is caught.
    for (let i = 0; i < 2 * fs; i++) {
      const t = i / fs;
      const levelDb = -20 + 20 * Math.min(1.5, t);
      det.push(Math.pow(10, levelDb / 20) * Math.sin(2 * Math.PI * 50 * (3 + t)) + f.process(0.001 * g()), -6, 0, -1);
      if (det.state.triggered) break;
    }
    expect(det.state.reason).toBe('peak-growth');
    // The full-rate rules never wait for the level tracks.
    const det2 = new HowlDetector(fs);
    const ceil = Math.pow(10, -12 / 20);
    let at = -1;
    for (let i = 0; i < fs && at < 0; i++) {
      if (i % step === 0) det2.holdLevels(40);
      det2.push(ceil * Math.sin((2 * Math.PI * 150 * i) / fs), 0, 0, -12);
      if (det2.state.triggered) at = i / fs;
    }
    expect(det2.state.reason).toBe('output-overload');
    expect(at).toBeLessThan(0.35);
  });

  it('ignores NaN and infinite inputs', () => {
    const det = new HowlDetector(48000);
    for (let i = 0; i < 48000; i++) det.push(Number.NaN, Number.NEGATIVE_INFINITY, Number.NaN, -12);
    expect(det.state.triggered).toBe(false);
  });
});

describe('PeriodicityAnalyzer', () => {
  /** Pushes x (length a multiple of the analysis period), then zeros until the analysis of x is done. */
  function analyse(pa: PeriodicityAnalyzer, x: Float64Array): void {
    expect(x.length % pa.everySamples).toBe(0);
    for (let i = 0; i < x.length; i++) pa.push(x[i]);
    for (let i = 0; i < pa.everySamples - 1; i++) pa.push(0);
  }

  /** Direct evaluation of the same quantity over the window ending at x.length. */
  function direct(pa: PeriodicityAnalyzer, x: Float64Array): { rho: number; lag: number } {
    const end = x.length;
    const W = pa.windowSamples;
    let best = 0;
    let lag = 0;
    let r1 = 0;
    let r2 = 0;
    for (let tau = pa.lagMinSamples - 1; tau <= pa.lagMaxSamples + 1; tau++) {
      let c = 0;
      let ea = 0;
      let eb = 0;
      for (let n = end - W; n < end; n++) {
        c += x[n] * x[n - tau];
        ea += x[n] * x[n];
        eb += x[n - tau] * x[n - tau];
      }
      const r = Math.abs(c) / Math.sqrt(ea * eb);
      if (tau - 1 >= pa.lagMinSamples && tau - 1 <= pa.lagMaxSamples && r1 >= r2 && r1 > r && r1 > best) {
        best = r1;
        lag = tau - 1;
      }
      r2 = r1;
      r1 = r;
    }
    return { rho: best, lag };
  }

  /** x[n] = e[n] + g·x[n − d]: what a loop of delay d and gain g does to noise e (a comb). */
  function comb(n: number, d: number, g: number, seed: number): Float64Array {
    const e = gaussian(seed);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = 0.01 * e() + (i >= d ? g * x[i - d] : 0);
    return x;
  }

  const length = (pa: PeriodicityAnalyzer): number => pa.everySamples * Math.ceil(9000 / pa.everySamples);

  for (const rate of [6000, 5512.5]) {
    it(`matches the direct normalized autocorrelation, lag included (rate ${rate} Hz)`, () => {
      const pa = new PeriodicityAnalyzer(rate);
      const d = Math.round(0.137 * rate);
      const x = comb(length(pa), d, 0.9, 1);
      analyse(pa, x);
      const ref = direct(pa, x);
      expect(pa.rho).toBeCloseTo(ref.rho, 9);
      expect(Math.round((pa.lagMs * rate) / 1000)).toBe(ref.lag);
      // A loop is periodic at its delay (or a multiple of it).
      expect(pa.rho).toBeGreaterThan(HOWL_PERIODIC_RHO);
      expect(ref.lag % d).toBe(0);
    });
  }

  it('white and low-passed noise are not periodic; a tone and a 300 ms comb are', () => {
    const rate = 6000;
    const run = (x: Float64Array): number => {
      const pa = new PeriodicityAnalyzer(rate);
      analyse(pa, x);
      return pa.rho;
    };
    const pa = new PeriodicityAnalyzer(rate);
    const n = length(pa);
    const g = gaussian(2);
    const white = Float64Array.from({ length: n }, () => g());
    const f = new Biquad(lowpass(rate, 500, Math.SQRT1_2));
    const low = Float64Array.from({ length: n }, () => f.process(g()));
    const tone = Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 173 * i) / rate));
    expect(run(white)).toBeLessThan(0.2);
    expect(run(low)).toBeLessThan(0.3);
    expect(run(tone)).toBeGreaterThan(0.99);
    expect(run(comb(n, Math.round(0.3 * rate), 0.95, 3))).toBeGreaterThan(HOWL_PERIODIC_RHO);
  });

  it('the long-loop configuration (3 kHz, 500–1100 ms) is exact and sees a loop the default range cannot (v-dsp H3)', () => {
    const rate = 3000;
    const pa = new PeriodicityAnalyzer(rate, 400, { lagMinMs: 500, lagMaxMs: HOWL_LONG_LOOP_MAX_MS, spreadMs: 120 });
    expect(pa.everySamples).toBe(1200);
    expect(pa.lagMaxSamples).toBe(3300);
    expect(pa.windowSamples).toBe(8192 - 3300 - 2);
    const d = Math.round(0.8 * rate);
    const n = pa.everySamples * Math.ceil(24000 / pa.everySamples);
    const x = comb(n, d, 0.95, 5);
    analyse(pa, x);
    const ref = direct(pa, x);
    expect(pa.rho).toBeCloseTo(ref.rho, 9);
    expect(pa.rho).toBeGreaterThan(HOWL_PERIODIC_RHO);
    expect(Math.round((pa.lagMs * rate) / 1000)).toBe(d);
    // The same 800 ms loop at 6 kHz: no lag of the default 5–600 ms range repeats it.
    const main = new PeriodicityAnalyzer(6000);
    analyse(main, comb(main.everySamples * Math.ceil(48000 / main.everySamples), 2 * d, 0.95, 5));
    expect(main.rho).toBeLessThan(0.5);
    // Its work: ≤ 120 ms of pushes, half the main analysis' slice per pushed sample of the 6 kHz stream.
    const units = 8192 + 13 * 8192 + 2 * 4097 + (pa.lagMaxSamples - pa.lagMinSamples + 3);
    expect(pa.budget * Math.ceil(0.12 * rate)).toBeGreaterThanOrEqual(units);
    expect(pa.budget).toBeLessThan(400);
  });

  it('spreads each analysis over ≤ 60 ms of pushes and starts one every 200 ms', () => {
    const rate = 6000;
    const pa = new PeriodicityAnalyzer(rate);
    expect(pa.everySamples).toBe(1200);
    expect(pa.ageMs).toBe(Infinity);
    const x = comb(length(pa), 500, 0.9, 4);
    for (let i = 0; i < x.length; i++) pa.push(x[i]);
    const done = Math.ceil(0.06 * rate) + 1;
    for (let i = 0; i < done; i++) pa.push(0);
    // The analysis of the window that ended with x is complete (younger than its spread).
    expect(pa.ageMs).toBeLessThanOrEqual(61);
    expect(pa.rho).toBeGreaterThan(HOWL_PERIODIC_RHO);
    expect(pa.lagMs).toBeCloseTo((500 / rate) * 1000, 6);
    // Its work fits the budget: ≤ budget units per push, all of them done within the spread.
    const units = 8192 + 13 * 8192 + 2 * 4097 + (pa.lagMaxSamples - pa.lagMinSamples + 3);
    expect(pa.budget * Math.ceil(0.06 * rate)).toBeGreaterThanOrEqual(units);
    expect(pa.budget).toBeLessThan(400);
    pa.reset();
    expect(pa.rho).toBe(0);
    expect(pa.ageMs).toBe(Infinity);
  });
});

describe('HowlDetector: howls of any spectral shape', () => {
  const ceilingDb = -12;
  const ceil = Math.pow(10, ceilingDb / 20);

  interface LoopRun {
    /** Seconds from closing the loop to the trigger. */
    at: number | null;
    reason: string | null;
    freqHz: number | null;
    /** Seconds from closing the loop to the first GR > 0.5 dB, or null. */
    firstGrAt: number | null;
    /** Seconds from closing the loop to sustained limiting (GR > 0.5 dB in 80 % of 200 ms), or null. */
    limitedAt: number | null;
    maxOut: number;
  }

  /**
   * The chain in miniature: v = LP500(ambient + g·y[n − D]), y = limiter(v), commanded gain 0 dB. The
   * loop (delay D, gain g at low frequencies) is closed at `closeAt`; the detector sees v.
   */
  interface LoopSpec {
    loopMs: number;
    g: number;
    ambientDb: number;
    seconds: number;
    seed?: number;
  }

  function loopRun(fs: number, o: LoopSpec): LoopRun {
    const det = new HowlDetector(fs);
    const lim = new PeakLimiter(fs, ceilingDb);
    const f = lp500(fs);
    const e = gaussian(o.seed ?? 9);
    const d = Math.round((o.loopMs / 1000) * fs);
    const closeAt = Math.round(0.5 * fs);
    const n = Math.round(o.seconds * fs);
    const y = new Float64Array(n);
    const amb = Math.pow(10, o.ambientDb / 20);
    const w = Math.round(0.2 * fs);
    const busy = new Uint8Array(n);
    let count = 0;
    let limitedAt: number | null = null;
    let firstGrAt: number | null = null;
    let maxOut = 0;
    for (let i = 0; i < n; i++) {
      const fb = i >= closeAt ? o.g * y[i - d] : 0;
      const v = f.process(amb * e() + fb);
      const gr = lim.gainReductionDb;
      det.push(v, 0, gr, ceilingDb);
      y[i] = lim.process(v);
      maxOut = Math.max(maxOut, Math.abs(y[i]));
      busy[i] = gr > 0.5 ? 1 : 0;
      if (firstGrAt === null && busy[i] === 1) firstGrAt = (i - closeAt) / fs;
      count += busy[i] - (i >= w ? busy[i - w] : 0);
      if (limitedAt === null && i >= closeAt + w && count >= 0.8 * w) limitedAt = (i - w - closeAt) / fs;
      if (det.state.triggered) {
        return { at: (i - closeAt) / fs, reason: det.state.reason, freqHz: det.state.freqHz, firstGrAt, limitedAt, maxOut };
      }
    }
    return { at: null, reason: null, freqHz: null, firstGrAt, limitedAt, maxOut };
  }

  for (const fs of RATES) {
    it(`'peak-growth' within 1 s for a dense-comb howl (100 ms loop, gain 1.5) growing from −50 dBFS (${fs} Hz)`, () => {
      const r = loopRun(fs, { loopMs: 100, g: 1.5, ambientDb: -50, seconds: 2.5 });
      expect(r.reason).toBe('peak-growth');
      expect(r.at!).toBeLessThan(1);
      expect(r.freqHz).not.toBeNull();
      // Caught while growing, before the limiter had to hold it.
      expect(r.limitedAt).toBeNull();
      expect(r.maxOut).toBeLessThanOrEqual(ceil);
    });

    it(`'limiter-saturated' within 400 ms of the limiter acting on a slow 200 ms howl (${fs} Hz)`, () => {
      // Loop gain 1.15: +1.2 dB per 200 ms (6 dB/s, below the growth rule) from −24 dBFS.
      const r = loopRun(fs, { loopMs: 200, g: 1.15, ambientDb: -24, seconds: 8 });
      expect(r.reason).toBe('limiter-saturated');
      expect(r.firstGrAt).not.toBeNull();
      expect(r.at! - r.firstGrAt!).toBeGreaterThan(0);
      expect(r.at! - r.firstGrAt!).toBeLessThanOrEqual(0.4);
      if (r.limitedAt !== null) expect(r.at! - r.limitedAt).toBeLessThanOrEqual(0.4);
      expect(r.freqHz).not.toBeNull();
      expect(r.maxOut).toBeLessThanOrEqual(ceil);
    });
  }

  it('a stable loop (gain 0.7, 100 ms) never triggers', () => {
    const r = loopRun(48000, { loopMs: 100, g: 0.7, ambientDb: -20, seconds: 6 });
    expect(r.at).toBeNull();
  });

  // v-dsp H3: loops longer than the main analysis' 600 ms (a 500 ms channel delay plus device latency)
  // grow slowly and are held at the ceiling with a few dB of GR: the long-loop analysis sees them.
  for (const fs of [44100, 48000]) {
    for (const loopMs of [650, 900]) {
      it(`'limiter-saturated' for a ${loopMs} ms loop with gain 1.3, within two trips of the limiter acting (${fs} Hz)`, () => {
        const r = loopRun(fs, { loopMs, g: 1.3, ambientDb: -24, seconds: 16 });
        expect(r.reason).toBe('limiter-saturated');
        // The limiter first catches the loudest part of each trip around the loop: the howl is held
        // (and muted) within about two trips of the first gain reduction.
        expect(r.firstGrAt).not.toBeNull();
        expect(r.at! - r.firstGrAt!).toBeLessThanOrEqual((2 * loopMs) / 1000 + 0.3);
        expect(r.maxOut).toBeLessThanOrEqual(ceil);
      });
    }
  }

  it('a stable long loop (gain 0.7, 900 ms) never triggers', () => {
    const r = loopRun(48000, { loopMs: 900, g: 0.7, ambientDb: -14, seconds: 8 });
    expect(r.at).toBeNull();
  });
});

/** RMS of unit Gaussian noise through lp500, from a pilot run. */
function lp500Rms(fs: number): number {
  const f = lp500(fs);
  const g = gaussian(110);
  let e = 0;
  for (let i = 0; i < fs; i++) e += f.process(g()) ** 2;
  return Math.sqrt(e / fs);
}

describe('HowlDetector: loud legitimate signals held by the limiter', () => {
  const ceilingDb = -12;
  const ceil = Math.pow(10, ceilingDb / 20);

  /** Runs the detector on `gen` through a real PeakLimiter (GR as the engine passes it). */
  function limited(fs: number, seconds: number, gen: (t: number) => number): RunResult & { meanGr: number } {
    const det = new HowlDetector(fs);
    const lim = new PeakLimiter(fs, ceilingDb);
    const n = Math.round(seconds * fs);
    let grSum = 0;
    for (let i = 0; i < n; i++) {
      const v = gen(i / fs);
      const gr = lim.gainReductionDb;
      grSum += gr;
      det.push(v, -10, gr, ceilingDb);
      lim.process(v);
      if (det.state.triggered) {
        return { at: i / fs, reason: det.state.reason, freqHz: det.state.freqHz, meanGr: grSum / (i + 1) };
      }
    }
    return { at: null, reason: null, freqHz: null, meanGr: grSum / n };
  }

  for (const fs of RATES) {
    it(`broadband noise 6 dB below the ceiling (≈ 3 dB of GR on average) for 10 s (${fs} Hz)`, () => {
      const g = gaussian(10);
      const f = lp500(fs);
      const k = (ceil * Math.pow(10, -6 / 20)) / lp500Rms(fs);
      const r = limited(fs, 10, () => k * f.process(g()));
      expect(r.at).toBeNull();
      expect(r.meanGr).toBeGreaterThan(1);
    });
  }

  /** 100 Hz hum with 3 harmonics (crest ≈ 8.5 dB) whose peaks exceed the ceiling by `overDb`, a little noise. */
  function hum(overDb: number, seed: number): (t: number) => number {
    const amps = [1, 0.7, 0.5, 0.4];
    const wave = (t: number): number => amps.reduce((a, v, h) => a + v * Math.cos(2 * Math.PI * (h + 1) * 100 * t), 0);
    let peak = 0;
    for (let i = 0; i < 20000; i++) peak = Math.max(peak, Math.abs(wave(i / 2e6)));
    const k = (ceil * Math.pow(10, overDb / 20)) / peak;
    const g = gaussian(seed);
    return (t) => k * wave(t) + 0.001 * ceil * g();
  }

  for (const fs of [44100, 48000]) {
    it(`a steady hum whose peaks touch the ceiling (GR < ${HOWL_SAT_GR_DB} dB) plays; pushed 2 dB into the limiter it is held (${fs} Hz, v-dsp H1)`, () => {
      const touch = limited(fs, 6, hum(0.3, 21));
      expect(touch.at).toBeNull();
      expect(touch.meanGr).toBeGreaterThan(0.1);
      expect(touch.meanGr).toBeLessThan(HOWL_SAT_GR_DB);
      const held = limited(fs, 3, hum(2, 21));
      expect(held.reason).toBe('limiter-saturated');
      expect(held.at!).toBeLessThan(1.5);
      expect(Math.abs(held.freqHz! - 100)).toBeLessThan(10);
    });
  }

  it('identical bursts every 0.5 s whose peaks touch the ceiling (periodic, but the limiter is not held)', () => {
    const fs = 48000;
    const rnd = mulberry32(11);
    const decay = (i: number): number => Math.exp(-i / (0.006 * fs));
    const burst = Float64Array.from({ length: Math.round(0.03 * fs) }, (_, i) => (2 * rnd() - 1) * decay(i));
    let peak = 0;
    for (const b of burst) peak = Math.max(peak, Math.abs(b));
    const g = gaussian(12);
    const r = limited(fs, 8, (t) => {
      const i = Math.round((t % 0.5) * fs);
      return 0.001 * g() + (i < burst.length ? ((1.12 * ceil) / peak) * burst[i] : 0);
    });
    expect(r.at).toBeNull();
    expect(r.meanGr).toBeGreaterThan(0);
  });

  it('a broadband source rising smoothly at 20 dB/s (not periodic) is not a howl', () => {
    const fs = 48000;
    const g = gaussian(13);
    const f = lp500(fs);
    const r = limited(fs, 5, (t) => {
      const db = -50 + 20 * Math.min(2, Math.max(0, t - 1));
      return Math.pow(10, db / 20) * 3 * f.process(g());
    });
    expect(r.at).toBeNull();
  });

  it('broadband noise fading in over a steady tone is not a howl (the growing part is not periodic)', () => {
    // Tone 25 dB below the ceiling; noise fading in linearly over 1 s up to 10 dB below it.
    const fs = 48000;
    const g = gaussian(15);
    const f = lp500(fs);
    const k = (ceil * Math.pow(10, -10 / 20)) / lp500Rms(fs);
    const r = limited(fs, 5, (t) => {
      const fade = Math.max(Math.pow(10, -25 / 20), Math.min(1, t - 2));
      return ceil * Math.pow(10, -25 / 20) * Math.sin(2 * Math.PI * 120 * t) + fade * k * f.process(g());
    });
    expect(r.at).toBeNull();
  });

  it('reset() forgets the periodicity: loud noise after a periodic signal does not trigger', () => {
    const fs = 48000;
    const det = new HowlDetector(fs);
    const lim = new PeakLimiter(fs, ceilingDb);
    for (let i = 0; i < 2 * fs; i++) {
      const v = 0.5 * ceil * Math.sin((2 * Math.PI * 140 * i) / fs);
      det.push(v, -10, lim.gainReductionDb, ceilingDb);
      lim.process(v);
    }
    expect(det.state.triggered).toBe(false);
    det.reset();
    const g = gaussian(14);
    for (let i = 0; i < 0.5 * fs; i++) {
      const v = ceil * 0.7 * g();
      det.push(v, -10, lim.gainReductionDb, ceilingDb);
      lim.process(v);
    }
    expect(det.state.triggered).toBe(false);
  });
});

describe('HowlDetector: the periodicity analysis follows the chain band', () => {
  it('howlBandHz: the LP cutoff, the top tonal band, or unbounded', () => {
    const p = clampParams(undefined);
    expect(howlBandHz(p)).toBe(p.filters.lpFreq);
    p.filters.lpFreq = 4000;
    expect(howlBandHz(p)).toBe(4000);
    p.filters.lpEnabled = false;
    expect(howlBandHz(p)).toBe(Infinity);
    p.phaseMode = 'tonal';
    p.tonal = { f0: 400, harmonics: 8, q: 10 };
    expect(howlBandHz(p)).toBeCloseTo(3200 * 1.05, 9);
    p.tonal = { f0: 100, harmonics: 4, q: 30 };
    expect(howlBandHz(p)).toBeCloseTo(400 * (1 + 1 / 60), 9);
    p.filters.lpEnabled = true;
    p.filters.lpFreq = 300;
    expect(howlBandHz(p)).toBe(300);
    // HP and barrier shape the low end only.
    p.filters.lpEnabled = false;
    p.phaseMode = 'delay';
    p.filters.barrierEnabled = true;
    expect(howlBandHz(p)).toBe(Infinity);
  });

  it('setBandHz: ~12 kHz above HOWL_WIDE_BAND_HZ, ~6 kHz again at or below HOWL_NARROW_BAND_HZ; reset() keeps it', () => {
    const det = new HowlDetector(48000);
    expect(det.wideBand).toBe(false);
    det.setBandHz(HOWL_WIDE_BAND_HZ);
    expect(det.wideBand).toBe(false);
    det.setBandHz(HOWL_WIDE_BAND_HZ + 1);
    expect(det.wideBand).toBe(true);
    det.setBandHz(HOWL_NARROW_BAND_HZ + 1);
    expect(det.wideBand).toBe(true);
    det.reset();
    expect(det.wideBand).toBe(true);
    det.setBandHz(HOWL_NARROW_BAND_HZ);
    expect(det.wideBand).toBe(false);
    det.setBandHz(HOWL_WIDE_BAND_HZ - 1);
    expect(det.wideBand).toBe(false);
    det.setBandHz(Number.NaN);
    expect(det.wideBand).toBe(true);
    det.setBandHz(500);
    det.setBandHz(Infinity);
    expect(det.wideBand).toBe(true);
    // At 8 kHz the main analysis already runs at the decimated rate: nothing to switch.
    const low = new HowlDetector(8000);
    low.setBandHz(4000);
    expect(low.wideBand).toBe(false);
  });

  /** Unit Gaussian noise through `f` (a fresh filter per call), as a generator, and its RMS from a pilot run. */
  function filtered(fs: number, coeffs: BiquadCoeffs[], seed: number): { next: () => number; rms: number } {
    const mk = (s: number): (() => number) => {
      const fl = coeffs.map((c) => new Biquad(c));
      const g = gaussian(s);
      return () => fl.reduce((v, b) => b.process(v), g());
    };
    const pilot = mk(seed + 1000);
    let e = 0;
    for (let i = 0; i < fs; i++) e += pilot() ** 2;
    return { next: mk(seed), rms: Math.sqrt(e / fs) };
  }

  /**
   * The chain in miniature with a speaker → reference path that passes only 2.7–4.3 kHz (4th-order band-pass,
   * loop gain g at its peak, ~3.4 kHz): v = BP(ambient + gain·y[n − D]) + lf, y = limiter(v), the ambient
   * 28 dB below the ceiling. `lf` is low-pass noise (500 Hz) `lfDb` re the ceiling: what the amplifier plays
   * below the howl, which dilutes it in the ~6 kHz analysis (that sees the howl only through its 2.4 kHz
   * anti-alias filter). `switchTo`: setBandHz at that time (s from the start; the loop closes at 0.5 s).
   */
  function bandLoop(
    fs: number,
    bandHz: number,
    o: { loopMs: number; g: number; lfDb: number; seconds: number; switchTo?: [atS: number, bandHz: number] },
  ) {
    const ceilingDb = -12;
    const ceil = Math.pow(10, ceilingDb / 20);
    const det = new HowlDetector(fs);
    det.setBandHz(bandHz);
    const lim = new PeakLimiter(fs, ceilingDb);
    const cs = [highpass(fs, 2700, Math.SQRT1_2), highpass(fs, 2700, Math.SQRT1_2), lowpass(fs, 4300, Math.SQRT1_2), lowpass(fs, 4300, Math.SQRT1_2)];
    let peak = 0;
    for (let f = 2000; f < 5000; f += 5) peak = Math.max(peak, cs.reduce((a, c) => a * cAbs(biquadResponse(c, f, fs)), 1));
    const bp = cs.map((c) => new Biquad(c));
    const lf = filtered(fs, [lowpass(fs, 500, Math.SQRT1_2)], 31);
    const lfGain = (ceil * Math.pow(10, o.lfDb / 20)) / lf.rms;
    const e = gaussian(32);
    const amb = ceil * Math.pow(10, -28 / 20);
    const d = Math.round((o.loopMs / 1000) * fs);
    const closeAt = Math.round(0.5 * fs);
    const n = Math.round(o.seconds * fs);
    const y = new Float64Array(n);
    const w = Math.round(0.2 * fs);
    const busy = new Uint8Array(n);
    let count = 0;
    let limitedAt: number | null = null;
    let maxOut = 0;
    const switchAt = o.switchTo === undefined ? -1 : Math.round(o.switchTo[0] * fs);
    for (let i = 0; i < n; i++) {
      if (i === switchAt) det.setBandHz(o.switchTo![1]);
      let h = amb * e() + (i >= closeAt ? (o.g / peak) * y[i - d] : 0);
      for (const b of bp) h = b.process(h);
      const v = h + lfGain * lf.next();
      const gr = lim.gainReductionDb;
      det.push(v, 0, gr, ceilingDb);
      y[i] = lim.process(v);
      maxOut = Math.max(maxOut, Math.abs(y[i]));
      busy[i] = gr > 0.5 ? 1 : 0;
      count += busy[i] - (i >= w ? busy[i - w] : 0);
      if (limitedAt === null && i >= closeAt + w && count >= 0.8 * w) limitedAt = (i - w - closeAt) / fs;
      if (det.state.triggered) return { at: (i - closeAt) / fs, reason: det.state.reason, freqHz: det.state.freqHz, limitedAt, maxOut, ceil };
    }
    return { at: null, reason: null, freqHz: null, limitedAt, maxOut, ceil };
  }

  for (const fs of RATES) {
    for (const [loopMs, g] of [
      [50, 1.3],
      [100, 1.2],
      [200, 1.15],
    ] as const) {
      it(`a dense-comb howl at 2.7–4.3 kHz (${loopMs} ms, gain ${g}) under low-frequency noise: pinned at ~6 kHz, caught at ~12 kHz (${fs} Hz)`, () => {
        const seconds = loopMs === 200 ? 7 : 3.5;
        const wide = bandLoop(fs, 4000, { loopMs, g, lfDb: -22, seconds });
        // While it grows, or within 400 ms of sustained limiting.
        expect(['peak-growth', 'limiter-saturated']).toContain(wide.reason);
        if (wide.limitedAt !== null) expect(wide.at! - wide.limitedAt).toBeLessThanOrEqual(0.4);
        expect(wide.freqHz!).toBeGreaterThan(2400);
        expect(wide.maxOut).toBeLessThanOrEqual(wide.ceil);
        // The same howl with the main analysis at ~6 kHz (a chain up to 500 Hz would not pass it at all).
        const narrow = bandLoop(fs, 500, { loopMs, g, lfDb: -22, seconds });
        expect(narrow.limitedAt).not.toBeNull();
        expect(narrow.at).toBeNull();
      });
    }
  }

  it('a change of rate starts the main analysis over: widened while the howl is pinned, it is caught ~0.2–0.6 s later', () => {
    // The 100 ms howl above, pinned at the ceiling at ~6 kHz; the chain is widened 3.5 s after closing the loop.
    const r = bandLoop(48000, 500, { loopMs: 100, g: 1.2, lfDb: -22, seconds: 5, switchTo: [4, 4000] });
    expect(r.limitedAt!).toBeLessThan(3);
    expect(r.reason).toBe('limiter-saturated');
    // Not before the first analysis at the new rate (200 ms of history, done within 60 ms).
    expect(r.at! - 3.5).toBeGreaterThan(0.2);
    expect(r.at! - 3.5).toBeLessThan(0.6);
  });

  it('after a change of rate the band-level growth rule waits for the second analysis (noise fading in over a tone)', () => {
    // The 'not falling' test compares two analyses of the same signal path: right after a switch the first
    // analysis has no predecessor, and a tone that keeps |ρ| above 0.3 while broadband noise fades in would
    // pass it (8 of these 8 runs triggered 'peak-growth' without the wait).
    const fs = 48000;
    const ceilingDb = -12;
    const ceil = Math.pow(10, ceilingDb / 20);
    for (const cutHz of [500, 3000]) {
      for (const seed of [15, 18]) {
        for (const switchS of [2, 2.1]) {
          const det = new HowlDetector(fs);
          const lim = new PeakLimiter(fs, ceilingDb);
          const noise = filtered(fs, [lowpass(fs, 1.272 * cutHz, 1 / Math.sqrt(3))], seed);
          const k = (ceil * Math.pow(10, -10 / 20)) / noise.rms;
          let wide = cutHz > HOWL_WIDE_BAND_HZ;
          det.setBandHz(wide ? cutHz : 500);
          for (let i = 0; i < 5 * fs && !det.state.triggered; i++) {
            const t = i / fs;
            if (i === Math.round(switchS * fs)) {
              wide = !wide;
              det.setBandHz(wide ? 4000 : 500);
            }
            // Tone 25 dB below the ceiling; noise fading in over 1 s from t = 2 s up to 10 dB below it.
            const fade = Math.max(Math.pow(10, -25 / 20), Math.min(1, t - 2));
            const v = ceil * Math.pow(10, -25 / 20) * Math.sin(2 * Math.PI * 120 * t) + fade * k * noise.next();
            det.push(v, -10, lim.gainReductionDb, ceilingDb);
            lim.process(v);
          }
          expect(det.state.triggered).toBe(false);
        }
      }
    }
  });
});

describe('PeriodicityAnalyzer: envelope (the ~12 kHz analysis)', () => {
  /** x[n] = BP(e[n]) + g·x[n − d] at `rate`, BP = 2.7–4.3 kHz (4th order), then every `keep`-th sample. */
  function bandComb(rate: number, n: number, d: number, g: number, seed: number, keep = 1): Float64Array {
    const bp = [highpass(rate, 2700, Math.SQRT1_2), highpass(rate, 2700, Math.SQRT1_2), lowpass(rate, 4300, Math.SQRT1_2), lowpass(rate, 4300, Math.SQRT1_2)].map(
      (c) => new Biquad(c),
    );
    const e = gaussian(seed);
    const x = new Float64Array(n * keep);
    for (let i = 0; i < x.length; i++) x[i] = bp.reduce((v, b) => b.process(v), e()) + (i >= d ? g * x[i - d] : 0);
    return Float64Array.from({ length: n }, (_, i) => x[i * keep]);
  }

  it('a band-pass howl whose loop delay falls between two lags: |ρ| loses cos(π·f/rate), the envelope does not', () => {
    // A 3.4 kHz comb (gain 0.9) with a 600.5-sample loop at 12 kHz (made at 24 kHz, 1201 samples), lags up to
    // 80 ms (the loop once).
    const rate = 12000;
    const o = { log2N: 13, lagMaxMs: 80 };
    const plain = new PeriodicityAnalyzer(rate, 200, o);
    const env = new PeriodicityAnalyzer(rate, 200, { ...o, envelope: true });
    expect(env.envelope).toBe(true);
    expect(plain.envelope).toBe(false);
    const x = bandComb(2 * rate, plain.everySamples * 12, 1201, 0.9, 41, 2);
    for (const v of x) {
      plain.push(v);
      env.push(v);
    }
    expect(env.rho).toBeGreaterThan(0.85);
    expect(Math.abs(env.lagMs - 50.04)).toBeLessThan(0.1);
    expect(plain.rho).toBeLessThan(env.rho - 0.15);
    expect(env.analyses).toBe(plain.analyses);
    expect(env.analyses).toBeGreaterThan(5);
    env.reset();
    expect(env.analyses).toBe(0);
  });

  it('matches a direct evaluation (analytic correlation, held to the share of the window with lagged samples)', () => {
    const rate = 1000;
    const pa = new PeriodicityAnalyzer(rate, 100, { log2N: 10, lagMinMs: 20, lagMaxMs: 200, envelope: true });
    const N = 1024;
    const W = pa.windowSamples;
    const g = gaussian(43);
    const f = new Biquad(lowpass(rate, 150, Math.SQRT1_2));
    const xs: number[] = [];
    let last = Infinity;
    let checked = 0;
    for (let i = 0; i < 2500; i++) {
      const v = f.process(g()) + (i >= 137 ? 0.6 * xs[i - 137] : 0);
      xs.push(v);
      pa.push(v);
      if (pa.ageMs < last) {
        // The analysed window ended with the sample pushed ageMs − 1 sample ago.
        const end = xs.length + 1 - Math.round((pa.ageMs * rate) / 1000);
        const b = new Float64Array(N);
        for (let k = 0; k < N; k++) b[k] = end - N + k >= 0 ? xs[end - N + k] : 0;
        const a = Float64Array.from(b, (v2, k) => (k >= N - W ? v2 : 0));
        const c = new Float64Array(N);
        for (let tau = 0; tau < N; tau++) for (let k = N - W; k < N; k++) c[tau] += a[k] * b[(k - tau + N) % N];
        // Analytic signal of c: 2× the positive frequencies, none of the negative ones.
        const cr = new Float64Array(N);
        const ci = new Float64Array(N);
        for (let k = 0; k <= N / 2; k++) {
          let re = 0;
          let im = 0;
          for (let t = 0; t < N; t++) {
            re += c[t] * Math.cos((2 * Math.PI * k * t) / N);
            im -= c[t] * Math.sin((2 * Math.PI * k * t) / N);
          }
          const w = k === 0 || k === N / 2 ? 1 : 2;
          for (let t = 0; t < N; t++) {
            const ph = (2 * Math.PI * k * t) / N;
            cr[t] += (w * (re * Math.cos(ph) - im * Math.sin(ph))) / N;
            ci[t] += (w * (re * Math.sin(ph) + im * Math.cos(ph))) / N;
          }
        }
        const energy = (from: number, to: number): number => {
          let e = 0;
          for (let k = Math.max(0, from); k < to; k++) e += b[k] * b[k];
          return e;
        };
        const firstValid = Math.max(0, N - end);
        const ea = energy(N - W, N);
        let best = 0;
        let r1 = 0;
        let r2 = 0;
        for (let tau = pa.lagMinSamples - 1; tau <= pa.lagMaxSamples + 1; tau++) {
          const eb = energy(N - W - tau, N - tau);
          const share = energy(Math.max(N - W, firstValid + tau), N) / ea;
          const r = Math.min(Math.hypot(cr[tau], ci[tau]) / Math.sqrt(ea * eb), Math.sqrt(share));
          if (tau - 1 >= pa.lagMinSamples && r1 >= r2 && r1 > r && r1 > best) best = r1;
          r2 = r1;
          r1 = r;
        }
        expect(pa.rho).toBeCloseTo(best, 9);
        checked++;
      }
      last = pa.ageMs;
    }
    // Analyses of a history still filling up (the share bound acting) and of a full one.
    expect(checked).toBeGreaterThan(15);
  });

  it('noise right after a reset (a history still filling up) stays far from periodic', () => {
    const rate = 12000;
    for (let seed = 1; seed <= 12; seed++) {
      const pa = new PeriodicityAnalyzer(rate, 200, { log2N: 14, envelope: true });
      const g = gaussian(seed);
      const f = new Biquad(lowpass(rate, 500, Math.SQRT1_2));
      let last = Infinity;
      for (let i = 0; i < 1.5 * rate; i++) {
        pa.push(f.process(g()));
        if (pa.ageMs < last) expect(pa.rho).toBeLessThan(0.3);
        last = pa.ageMs;
      }
      expect(pa.analyses).toBe(7);
    }
  });

  it('the ~12 kHz configuration: 5–600 ms over the same ~0.77 s window, its work spread over ≤ 60 ms', () => {
    const rate = 12000;
    const pa = new PeriodicityAnalyzer(rate, 200, { log2N: 14, envelope: true });
    expect(pa.lagMaxSamples).toBe(7200);
    expect(pa.windowSamples).toBe(16384 - 7200 - 2);
    const units = 16384 + 14 * 16384 + 2 * 8193 + (pa.lagMaxSamples - pa.lagMinSamples + 3);
    expect(pa.budget * Math.ceil(0.06 * rate)).toBeGreaterThanOrEqual(units);
    expect(pa.budget).toBeLessThan(400);
    const x = bandComb(rate, pa.everySamples * 10, 1777, 0.95, 44);
    for (const v of x) pa.push(v);
    for (let i = 0; i < Math.ceil(0.06 * rate) + 1; i++) pa.push(0);
    expect(pa.ageMs).toBeLessThanOrEqual(61);
    expect(pa.rho).toBeGreaterThan(HOWL_PERIODIC_RHO);
    expect(pa.lagMs).toBeCloseTo((1777 / rate) * 1000, 6);
  });
});

