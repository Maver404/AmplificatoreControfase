import { describe, expect, it } from 'vitest';
import { FFT, nextPow2, sharedFFT } from '../../src/dsp/fft.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

describe('nextPow2', () => {
  it('rounds up to powers of two', () => {
    expect(nextPow2(0)).toBe(1);
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(1000)).toBe(1024);
    expect(nextPow2(1024)).toBe(1024);
    expect(nextPow2(1025)).toBe(2048);
    expect(nextPow2(3_000_000)).toBe(4_194_304);
  });
});

describe('FFT', () => {
  it('rejects sizes that are not powers of two and mismatched arrays', () => {
    expect(() => new FFT(12)).toThrow(RangeError);
    expect(() => new FFT(0)).toThrow(RangeError);
    const f = new FFT(8);
    expect(() => f.forward(new Float64Array(4), new Float64Array(4))).toThrow(RangeError);
  });

  it('forward/inverse round trip restores the signal', () => {
    for (const n of [1, 2, 8, 1024, 65536]) {
      const rnd = mulberry32(n);
      const re = new Float64Array(n);
      const im = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        re[i] = rnd() - 0.5;
        im[i] = rnd() - 0.5;
      }
      const r0 = re.slice();
      const i0 = im.slice();
      const fft = new FFT(n);
      fft.forward(re, im);
      fft.inverse(re, im);
      let err = 0;
      for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(re[i] - r0[i]), Math.abs(im[i] - i0[i]));
      expect(err).toBeLessThan(1e-12);
    }
  });

  it('satisfies Parseval: Σ|x|² = (1/n)·Σ|X|²', () => {
    const n = 4096;
    const rnd = mulberry32(7);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    let et = 0;
    for (let i = 0; i < n; i++) {
      re[i] = rnd() * 2 - 1;
      et += re[i] * re[i];
    }
    new FFT(n).forward(re, im);
    let ef = 0;
    for (let k = 0; k < n; k++) ef += re[k] * re[k] + im[k] * im[k];
    expect(ef / n).toBeCloseTo(et, 9);
  });

  it('puts a bin-centred tone in the right bin with the right amplitude and phase', () => {
    const n = 1024;
    const bin = 37;
    const a = 0.75;
    const phase = 0.4;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = a * Math.cos((2 * Math.PI * bin * i) / n + phase);
    sharedFFT(n).forward(re, im);
    let kMax = 0;
    let mMax = 0;
    for (let k = 0; k <= n / 2; k++) {
      const m = Math.hypot(re[k], im[k]);
      if (m > mMax) {
        mMax = m;
        kMax = k;
      }
    }
    expect(kMax).toBe(bin);
    expect((2 * mMax) / n).toBeCloseTo(a, 10);
    expect(Math.atan2(im[bin], re[bin])).toBeCloseTo(phase, 10);
    // Everything else is numerically zero.
    let leak = 0;
    for (let k = 0; k <= n / 2; k++) if (k !== bin) leak = Math.max(leak, Math.hypot(re[k], im[k]));
    expect(leak / mMax).toBeLessThan(1e-12);
  });

  it('matches a direct DFT', () => {
    const n = 64;
    const rnd = mulberry32(3);
    const x = Array.from({ length: n }, () => rnd() - 0.5);
    const re = Float64Array.from(x);
    const im = new Float64Array(n);
    new FFT(n).forward(re, im);
    for (let k = 0; k < n; k++) {
      let dr = 0;
      let di = 0;
      for (let m = 0; m < n; m++) {
        dr += x[m] * Math.cos((-2 * Math.PI * k * m) / n);
        di += x[m] * Math.sin((-2 * Math.PI * k * m) / n);
      }
      expect(re[k]).toBeCloseTo(dr, 10);
      expect(im[k]).toBeCloseTo(di, 10);
    }
  });

  it('sharedFFT caches instances by size', () => {
    expect(sharedFFT(256)).toBe(sharedFFT(256));
    expect(sharedFFT(512).n).toBe(512);
  });
});
