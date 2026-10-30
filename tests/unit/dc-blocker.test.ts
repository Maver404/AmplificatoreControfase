import { describe, expect, it } from 'vitest';
import { cAbs, cArg } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { DC_BLOCKER_HZ, DcBlocker, dcBlockerResponse } from '../../src/dsp/dc-blocker.ts';

/** Least-squares fit of y ≈ a·sin(ωn) + b·cos(ωn) over [start, end): response H = a + jb for input sin(ωn). */
function fitSine(y: Float64Array, w: number, start: number, end: number): Complex {
  let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
  for (let n = start; n < end; n++) {
    const s = Math.sin(w * n);
    const c = Math.cos(w * n);
    ss += s * s;
    sc += s * c;
    cc += c * c;
    ys += y[n] * s;
    yc += y[n] * c;
  }
  const det = ss * cc - sc * sc;
  return { re: (ys * cc - yc * sc) / det, im: (yc * ss - ys * sc) / det };
}

describe('DcBlocker', () => {
  it('removes a constant offset', () => {
    const fs = 48000;
    const dc = new DcBlocker(fs);
    let y = 1;
    for (let n = 0; n < 2 * fs; n++) y = dc.process(1);
    expect(Math.abs(y)).toBeLessThan(1e-6);
  });

  it('matches dcBlockerResponse in steady state', () => {
    for (const fs of [44100, 48000]) {
      for (const f of [DC_BLOCKER_HZ, 20, 1000, 15000]) {
        const dc = new DcBlocker(fs);
        const w = (2 * Math.PI * f) / fs;
        const n = 3 * fs;
        const y = new Float64Array(n);
        for (let i = 0; i < n; i++) y[i] = dc.process(Math.sin(w * i));
        const m = fitSine(y, w, 2 * fs, n);
        const h = dcBlockerResponse(fs, f);
        expect(Math.abs(cAbs(m) - cAbs(h))).toBeLessThan(1e-6);
        expect(Math.abs(cArg(m) - cArg(h))).toBeLessThan(1e-6);
      }
    }
  });

  it('is about -3 dB at its corner and transparent in the audio band', () => {
    const fs = 48000;
    const h3 = 20 * Math.log10(cAbs(dcBlockerResponse(fs, DC_BLOCKER_HZ)));
    expect(h3).toBeGreaterThan(-3.2);
    expect(h3).toBeLessThan(-2.8);
    expect(cAbs(dcBlockerResponse(fs, 0))).toBe(0);
    const h100 = dcBlockerResponse(fs, 100);
    expect(Math.abs(20 * Math.log10(cAbs(h100)))).toBeLessThan(0.01);
    expect(Math.abs((cArg(h100) * 180) / Math.PI)).toBeLessThan(2);
  });

  it('recovers from NaN/Infinity input', () => {
    const dc = new DcBlocker(48000);
    dc.process(0.5);
    expect(dc.process(Number.NaN)).toBe(0);
    expect(dc.process(Number.POSITIVE_INFINITY)).toBe(0);
    for (let i = 0; i < 100; i++) expect(Number.isFinite(dc.process(Math.sin(i)))).toBe(true);
  });

  it('flushes its state to exact zero on silence', () => {
    const dc = new DcBlocker(48000);
    dc.process(1);
    let y = 1;
    for (let i = 0; i < 48000 * 30; i++) y = dc.process(0);
    expect(y).toBe(0);
  });
});
