import { describe, expect, it } from 'vitest';
import { FFT, nextPow2 } from '../../src/dsp/fft.ts';
import { bandWindow, deconvolve, exponentialSweep, mulberry32, pinkNoise } from '../../src/dsp/sweep.ts';

const fs = 48000;

describe('exponentialSweep', () => {
  const spec = { fs, f1: 20, f2: 1000, durationS: 1.5, fadeMs: 30, amplitude: 0.3 };
  const x = exponentialSweep(spec);

  it('has the right length, peak and fades', () => {
    expect(x.length).toBe(72000);
    let peak = 0;
    for (const v of x) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeLessThanOrEqual(0.3 + 1e-7);
    expect(peak).toBeGreaterThan(0.29);
    expect(x[0]).toBe(0);
    expect(Math.abs(x[x.length - 1])).toBeLessThan(1e-3);
    // Inside the fade-in the envelope is below the amplitude.
    let early = 0;
    for (let i = 0; i < 100; i++) early = Math.max(early, Math.abs(x[i]));
    expect(early).toBeLessThan(0.01);
  });

  it('sweeps exponentially from f1 to f2', () => {
    // Instantaneous frequency from zero-crossing spacing at the start, middle and end.
    const freqAt = (centre: number, span: number): number => {
      const crossings: number[] = [];
      for (let i = centre - span; i < centre + span; i++) {
        if (x[i - 1] < 0 && x[i] >= 0) crossings.push(i - x[i] / (x[i] - x[i - 1]));
      }
      return (fs * (crossings.length - 1)) / (crossings[crossings.length - 1] - crossings[0]);
    };
    const L = 1.5 / Math.log(1000 / 20);
    // A chirp's average crossing rate over a window is within ~1 % of the centre frequency.
    for (const [centre, span] of [
      [9000, 2000],
      [36000, 1000],
      [70000, 300],
    ]) {
      expect(Math.abs(freqAt(centre, span) / (20 * Math.exp(centre / fs / L)) - 1)).toBeLessThan(0.015);
    }
  });

  it('limits f2 to Nyquist and amplitude to 1, and rejects nonsense', () => {
    const y = exponentialSweep({ fs: 8000, f1: 20, f2: 20000, durationS: 0.2, fadeMs: 5, amplitude: 4 });
    expect(Math.max(...Array.from(y, Math.abs))).toBeLessThanOrEqual(1);
    expect(() => exponentialSweep({ ...spec, f2: 10 })).toThrow(RangeError);
    expect(() => exponentialSweep({ ...spec, durationS: 0 })).toThrow(RangeError);
  });
});

describe('deconvolve', () => {
  it('recovers a delayed, scaled, inverted copy as a band-limited impulse at the delay', () => {
    const stim = exponentialSweep({ fs, f1: 20, f2: 20000, durationS: 1, fadeMs: 10, amplitude: 0.5 });
    const rec = new Float32Array(stim.length + 4000);
    for (let i = 0; i < stim.length; i++) rec[i + 777] = -0.4 * stim[i];
    const ir = deconvolve(rec, stim, fs, 20, 20000);
    expect(ir.length).toBe(nextPow2(rec.length + stim.length));
    let k = 0;
    for (let i = 0; i < ir.length; i++) if (Math.abs(ir[i]) > Math.abs(ir[k])) k = i;
    expect(k).toBe(777);
    expect(ir[k]).toBeLessThan(0);
    // In-band magnitude ≈ 0.4: DTFT of the whole circular IR (negative lags in the upper half).
    for (const f of [50, 500, 5000]) {
      let re = 0;
      let im = 0;
      for (let i = 0; i < ir.length; i++) {
        const lag = i < ir.length / 2 ? i : i - ir.length;
        re += ir[i] * Math.cos((-2 * Math.PI * f * lag) / fs);
        im += ir[i] * Math.sin((-2 * Math.PI * f * lag) / fs);
      }
      expect(Math.hypot(re, im)).toBeCloseTo(0.4, 3);
    }
  });

  it('returns zeros for a silent recording', () => {
    const stim = exponentialSweep({ fs, f1: 20, f2: 1000, durationS: 0.5, fadeMs: 10, amplitude: 0.5 });
    const ir = deconvolve(new Float32Array(stim.length), stim, fs, 20, 1000);
    expect(Math.max(...Array.from(ir, Math.abs))).toBeLessThan(1e-12);
  });
});

describe('bandWindow', () => {
  it('is 0 outside, 1 inside and smooth at the edges', () => {
    expect(bandWindow(19, 20, 1000)).toBe(0);
    expect(bandWindow(1001, 20, 1000)).toBe(0);
    expect(bandWindow(100, 20, 1000)).toBe(1);
    const mid = bandWindow(20 * Math.pow(2, 1 / 24), 20, 1000);
    expect(mid).toBeCloseTo(0.5, 6);
  });
});

describe('pinkNoise', () => {
  it('is deterministic per seed and peaks exactly at the requested amplitude', () => {
    const a = pinkNoise(fs, 1, 20, 1000, 0.1, 42);
    const b = pinkNoise(fs, 1, 20, 1000, 0.1, 42);
    const c = pinkNoise(fs, 1, 20, 1000, 0.1, 43);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    let peak = 0;
    for (const v of a) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeCloseTo(0.1, 6);
    expect(Math.abs(a[0])).toBe(0);
  });

  it('has equal power per octave inside the band and little outside', () => {
    const x = pinkNoise(fs, 4, 20, 2000, 0.5, 7);
    const n = nextPow2(x.length);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    re.set(x);
    new FFT(n).forward(re, im);
    const band = (lo: number, hi: number): number => {
      let p = 0;
      for (let k = Math.ceil((lo * n) / fs); k <= Math.floor((hi * n) / fs); k++) p += re[k] ** 2 + im[k] ** 2;
      return p;
    };
    const octaves = [
      [40, 80],
      [80, 160],
      [160, 320],
      [320, 640],
      [640, 1280],
    ].map(([lo, hi]) => band(lo, hi));
    const ref = octaves[0];
    for (const p of octaves) expect(Math.abs(10 * Math.log10(p / ref))).toBeLessThan(0.5);
    expect(10 * Math.log10(band(3000, 20000) / ref)).toBeLessThan(-40);
  });

  it('uses the shared seeded PRNG', () => {
    const r1 = mulberry32(5);
    const r2 = mulberry32(5);
    for (let i = 0; i < 10; i++) {
      const v = r1();
      expect(v).toBe(r2());
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});
