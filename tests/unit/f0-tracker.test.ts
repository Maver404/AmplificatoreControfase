import { describe, expect, it } from 'vitest';
import { F0Tracker, driftPhaseErrorDeg, estimateF0 } from '../../src/dsp/f0-tracker.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;

function gaussian(seed: number): () => number {
  const rnd = mulberry32(seed);
  return () => Math.sqrt(-2 * Math.log(Math.max(1e-12, rnd()))) * Math.cos(2 * Math.PI * rnd());
}

/** Σ amps[h-1]·sin(2π·h·f0·t + φ_h) + white noise of RMS `noise`. */
function harmonicTone(f0: number, amps: number[], noise: number, seconds: number, seed: number): Float64Array {
  const n = Math.round(seconds * fs);
  const g = gaussian(seed);
  const rnd = mulberry32(seed + 1);
  const phases = amps.map(() => 2 * Math.PI * rnd());
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = noise * g();
    for (let h = 0; h < amps.length; h++) v += amps[h] * Math.sin((2 * Math.PI * (h + 1) * f0 * i) / fs + phases[h]);
    x[i] = v;
  }
  return x;
}

describe('estimateF0', () => {
  it('estimates 100.37 Hz with harmonics and noise to better than 0.02 Hz from 2 s', () => {
    const x = harmonicTone(100.37, [0.2, 0.1, 0.06, 0.03], 0.1, 2, 1);
    const e = estimateF0(x, fs, 20, 2000);
    expect(e).not.toBeNull();
    expect(Math.abs(e!.f0 - 100.37)).toBeLessThan(0.02);
    expect(e!.dominanceDb).toBeGreaterThan(30);
    // Tonal power fraction: (0.2² + 0.1² + 0.06² + 0.03²)/2 ≈ 0.0272 vs 0.01 of noise → ≈ 0.73.
    expect(e!.confidence).toBeGreaterThan(0.65);
    expect(e!.confidence).toBeLessThan(0.8);
  });

  it('reaches ~0.01 % on a clean stable tone of 1 s', () => {
    for (const f0 of [37.3, 100.37, 251.9, 1234.5]) {
      const e = estimateF0(harmonicTone(f0, [0.5], 0.001, 1, 3), fs, 20, 2000);
      expect(e).not.toBeNull();
      expect(Math.abs(e!.f0 - f0) / f0).toBeLessThan(1e-4);
      expect(e!.confidence).toBeGreaterThan(0.99);
    }
  });

  it('prefers the fundamental when the 2nd harmonic is stronger', () => {
    const x = harmonicTone(100.37, [0.1, 0.4], 0.05, 2, 4);
    const e = estimateF0(x, fs, 20, 2000);
    expect(e).not.toBeNull();
    expect(Math.abs(e!.f0 - 100.37)).toBeLessThan(0.02);
    // ... and with a weak fundamental under strong 2nd and 3rd harmonics.
    const y = harmonicTone(80.11, [0.05, 0.3, 0.2], 0.05, 2, 5);
    expect(Math.abs(estimateF0(y, fs, 20, 2000)!.f0 - 80.11)).toBeLessThan(0.02);
  });

  it('does not invent a sub-harmonic for a pure tone', () => {
    const e = estimateF0(harmonicTone(300, [0.3], 0.05, 2, 6), fs, 20, 2000);
    expect(Math.abs(e!.f0 - 300)).toBeLessThan(0.02);
  });

  it('returns null on white noise and on silence', () => {
    for (let seed = 10; seed < 16; seed++) {
      const g = gaussian(seed);
      const x = new Float64Array(2 * fs);
      for (let i = 0; i < x.length; i++) x[i] = 0.1 * g();
      expect(estimateF0(x, fs, 20, 2000)).toBeNull();
    }
    expect(estimateF0(new Float64Array(fs), fs, 20, 2000)).toBeNull();
    expect(estimateF0(new Float64Array(100), fs, 20, 2000)).toBeNull();
  });

  it('returns null on pink-ish (1/f) noise', () => {
    const g = gaussian(77);
    const x = new Float64Array(2 * fs);
    // Brown noise high-passed: strongly sloped spectrum, no tone.
    let b = 0;
    let prev = 0;
    for (let i = 0; i < x.length; i++) {
      b = 0.999 * b + 0.05 * g();
      x[i] = b - prev * 0.5;
      prev = b;
    }
    expect(estimateF0(x, fs, 20, 2000)).toBeNull();
  });

  it('respects the search range', () => {
    const x = harmonicTone(1500, [0.3], 0.01, 1, 8);
    expect(estimateF0(x, fs, 20, 1000)).toBeNull();
    expect(Math.abs(estimateF0(x, fs, 1000, 2000)!.f0 - 1500)).toBeLessThan(0.02);
  });
});

describe('F0Tracker', () => {
  it('keeps the last window of pushed blocks and estimates on it', () => {
    const tr = new F0Tracker(fs, 2);
    expect(tr.estimate()).toBeNull();
    // 1 s of 60 Hz, then 3 s of 100.37 Hz: only the latter must be in the 2 s window.
    const a = harmonicTone(60, [0.3], 0.02, 1, 20);
    const b = harmonicTone(100.37, [0.3, 0.1], 0.02, 3, 21);
    for (const sig of [a, b]) {
      for (let i = 0; i < sig.length; i += 128) tr.push(Float32Array.from(sig.subarray(i, i + 128)));
    }
    const e = tr.estimate();
    expect(e).not.toBeNull();
    expect(Math.abs(e!.f0 - 100.37)).toBeLessThan(0.02);
    tr.reset();
    expect(tr.estimate()).toBeNull();
  });
});

describe('driftPhaseErrorDeg', () => {
  it('is 360·h·df·D', () => {
    expect(driftPhaseErrorDeg(1, 1, 0.01)).toBeCloseTo(3.6, 12);
    expect(driftPhaseErrorDeg(0.01, 3, 0.5)).toBeCloseTo(5.4, 12);
    expect(driftPhaseErrorDeg(-0.02, 2, 0.25)).toBeCloseTo(-3.6, 12);
    expect(driftPhaseErrorDeg(0.5, 4, 0)).toBe(0);
  });
});
