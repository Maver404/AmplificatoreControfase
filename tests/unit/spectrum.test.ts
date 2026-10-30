import { describe, expect, it } from 'vitest';
import { SPECTRUM_FLOOR_DB, SpectrumAccumulator } from '../../src/app/spectrum.ts';

const FS = 48000;

function sine(n: number, f: number, amp: number, phase = 0): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * f * i) / FS + phase);
  return x;
}

function argMax(a: Float32Array): number {
  let k = 0;
  for (let i = 1; i < a.length; i++) if (a[i]! > a[k]!) k = i;
  return k;
}

describe('SpectrumAccumulator', () => {
  it('a bin-centred sine reads its RMS level (amplitude − 3.01 dB) in its bin', () => {
    const n = 8192;
    const s = new SpectrumAccumulator({ fftSize: n });
    const bin = 64;
    const f = (bin * FS) / n;
    const amp = 0.5;
    expect(s.push(sine(n, f, amp))).toBe(true);
    const db = s.db;
    expect(db.length).toBe(n / 2 + 1);
    expect(argMax(db)).toBe(bin);
    expect(db[bin]).toBeCloseTo(20 * Math.log10(amp) - 3.0103, 1);
    // Hann leakage stays within ±1 bin: two bins away the level is far below.
    expect(db[bin + 3]!).toBeLessThan(db[bin]! - 60);
  });

  it('emits the first frame after fftSize samples, then one per hop, whatever the push sizes', () => {
    const s = new SpectrumAccumulator({ fftSize: 1024 });
    const x = sine(4096, 1000, 0.1);
    let frames = 0;
    for (let off = 0; off < x.length; off += 100) if (s.push(x, off, Math.min(100, x.length - off))) frames++;
    // Frames at 1024, 1536, 2048, …, 4096 → 7.
    expect(s.frameCount).toBe(7);
    expect(frames).toBeLessThanOrEqual(7);
    expect(frames).toBeGreaterThan(0);
  });

  it('averages power exponentially and resets cleanly', () => {
    const s = new SpectrumAccumulator({ fftSize: 1024, alpha: 0.5 });
    const f = (32 * FS) / 1024;
    s.push(sine(1024, f, 1));
    const loud = s.db[32]!;
    // A silent stretch halves the averaged power per frame: −3 dB per hop.
    s.push(new Float32Array(512));
    expect(s.db[32]!).toBeLessThan(loud);
    s.reset();
    expect(s.frameCount).toBe(0);
    expect(s.db.every((v) => v === SPECTRUM_FLOOR_DB)).toBe(true);
  });

  it('treats non-finite samples as silence and never produces NaN', () => {
    const s = new SpectrumAccumulator({ fftSize: 256 });
    const x = new Float32Array(256).fill(Number.NaN);
    x[10] = Number.POSITIVE_INFINITY;
    s.push(x);
    expect(s.db.every((v) => Number.isFinite(v))).toBe(true);
    expect(s.db[5]).toBe(SPECTRUM_FLOOR_DB);
  });

  it('snapshot() is a copy', () => {
    const s = new SpectrumAccumulator({ fftSize: 256 });
    s.push(sine(256, 3000, 0.2));
    const snap = s.snapshot();
    s.push(new Float32Array(1024));
    expect(snap).not.toBe(s.db);
    expect(snap[16]).not.toBe(s.db[16]);
  });
});
