import { describe, expect, it } from 'vitest';
import { Biquad, biquadResponse } from '../../src/dsp/biquad.ts';
import type { BiquadCoeffs } from '../../src/dsp/biquad.ts';
import { besselLowpassSections } from '../../src/dsp/bessel-lp.ts';
import { cAbs, cArg, cMul } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';

const cascade = (secs: BiquadCoeffs[], f: number, fs: number): Complex =>
  secs.reduce<Complex>((h, c) => cMul(h, biquadResponse(c, f, fs)), { re: 1, im: 0 });

const db = (h: Complex) => 20 * Math.log10(cAbs(h));

/** Group delay (samples) by central difference of the phase. */
function groupDelay(secs: BiquadCoeffs[], f: number, fs: number): number {
  const df = Math.min(0.05, f * 0.01);
  const lo = cascade(secs, f - df, fs);
  const hi = cascade(secs, f + df, fs);
  const dphi = Math.atan2(hi.im * lo.re - hi.re * lo.im, hi.re * lo.re + hi.im * lo.im);
  return (-dphi / (2 * Math.PI * 2 * df)) * fs;
}

describe('besselLowpassSections', () => {
  it('has the documented structure', () => {
    expect(besselLowpassSections(48000, 500, 2)).toHaveLength(1);
    expect(besselLowpassSections(48000, 500, 4)).toHaveLength(2);
    // order 2 at a low cutoff ≈ RBJ low-pass with Q = 1/√3 at 1.272·cutoff
    const [c] = besselLowpassSections(48000, 100, 2);
    const w0 = (2 * Math.PI * 127.202) / 48000;
    const alpha = Math.sin(w0) / (2 / Math.sqrt(3));
    expect(c.a2).toBeCloseTo((1 - alpha) / (1 + alpha), 6);
  });

  it('is -3 dB at the cutoff (±0.1 dB) for orders 2 and 4 at 44.1 and 48 kHz', () => {
    for (const fs of [44100, 48000]) {
      for (const order of [2, 4] as const) {
        for (const cutoff of [100, 250, 500, 1000, 2000, 4000]) {
          const secs = besselLowpassSections(fs, cutoff, order);
          expect(Math.abs(db(cascade(secs, cutoff, fs)) + 3.0103)).toBeLessThan(0.1);
          expect(Math.abs(db(cascade(secs, 1, fs)))).toBeLessThan(0.001);
        }
      }
    }
  });

  it('behaves as a pure delay below cutoff/3 (flat group delay, linear phase)', () => {
    for (const fs of [44100, 48000]) {
      for (const order of [2, 4] as const) {
        for (const cutoff of [100, 500, 2000, 4000]) {
          const secs = besselLowpassSections(fs, cutoff, order);
          const tau0 = groupDelay(secs, cutoff / 100, fs);
          for (const k of [30, 10, 5, 3]) {
            const f = cutoff / k;
            const tau = groupDelay(secs, f, fs);
            expect(Math.abs(tau - tau0) / tau0).toBeLessThan(0.01);
            // phase deviation from the pure delay tau0, in degrees
            const dev = ((cArg(cascade(secs, f, fs)) + (2 * Math.PI * f * tau0) / fs) * 180) / Math.PI;
            expect(Math.abs(dev)).toBeLessThan(1);
          }
        }
      }
    }
  });

  it('attenuates well above the cutoff', () => {
    const fs = 48000;
    for (const order of [2, 4] as const) {
      const secs = besselLowpassSections(fs, 500, order);
      expect(db(cascade(secs, 5000, fs))).toBeLessThan(order === 2 ? -30 : -50);
    }
  });

  it('time-domain cascade matches the response', () => {
    const fs = 48000;
    const secs = besselLowpassSections(fs, 500, 4);
    const bq = secs.map((c) => new Biquad(c));
    const f = 400;
    const w = (2 * Math.PI * f) / fs;
    let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
    for (let n = 0; n < fs; n++) {
      let y = Math.sin(w * n);
      for (const b of bq) y = b.process(y);
      if (n >= fs / 2) {
        const s = Math.sin(w * n);
        const c = Math.cos(w * n);
        ss += s * s; sc += s * c; cc += c * c; ys += y * s; yc += y * c;
      }
    }
    const det = ss * cc - sc * sc;
    const m = { re: (ys * cc - yc * sc) / det, im: (yc * ss - ys * sc) / det };
    const h = cascade(secs, f, fs);
    expect(Math.abs(cAbs(m) - cAbs(h))).toBeLessThan(1e-9);
    expect(Math.abs(cArg(m) - cArg(h))).toBeLessThan(1e-9);
  });

  it('clamps invalid cutoffs to a usable filter', () => {
    for (const c of [Number.NaN, -5, 0, 1e9]) {
      const secs = besselLowpassSections(48000, c, 2);
      for (const s of secs) for (const v of Object.values(s)) expect(Number.isFinite(v)).toBe(true);
    }
  });
});
