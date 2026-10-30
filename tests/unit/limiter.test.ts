import { describe, expect, it } from 'vitest';
import { PeakLimiter } from '../../src/dsp/limiter.ts';
import { LIMITER_RELEASE_MS } from '../../src/dsp/params.ts';

const dbToAmp = (db: number) => Math.pow(10, db / 20);

describe('PeakLimiter', () => {
  it('keeps a +12 dBFS sine under the ceiling at all times', () => {
    for (const fs of [44100, 48000]) {
      for (const ceilingDb of [-24, -12, -1]) {
        for (const f of [50, 100, 1000, 7000]) {
          const lim = new PeakLimiter(fs, ceilingDb);
          const ceil = dbToAmp(ceilingDb);
          const amp = dbToAmp(12);
          const w = (2 * Math.PI * f) / fs;
          let maxY = 0;
          let maxGr = 0;
          for (let n = 0; n < fs; n++) {
            const y = lim.process(amp * Math.sin(w * n));
            maxY = Math.max(maxY, Math.abs(y));
            maxGr = Math.max(maxGr, lim.gainReductionDb);
          }
          expect(maxY).toBeLessThanOrEqual(ceil + 1e-9);
          // peaks are reduced by (12 − ceiling) dB
          expect(maxGr).toBeGreaterThan(12 - ceilingDb - 0.5);
          expect(maxGr).toBeLessThan(12 - ceilingDb + 0.01);
        }
      }
    }
  });

  it('passes signals under the ceiling untouched', () => {
    const lim = new PeakLimiter(48000, -6);
    const amp = dbToAmp(-7);
    for (let n = 0; n < 4800; n++) {
      const x = amp * Math.sin(0.01 * n);
      expect(lim.process(x)).toBe(x);
    }
    expect(lim.gainReductionDb).toBe(0);
  });

  it('releases with the configured time constant', () => {
    const fs = 48000;
    const lim = new PeakLimiter(fs, -12);
    lim.process(dbToAmp(0)); // 12 dB of gain reduction
    expect(lim.gainReductionDb).toBeCloseTo(12, 9);
    const n = Math.round((LIMITER_RELEASE_MS / 1000) * fs);
    for (let i = 0; i < n; i++) lim.process(0);
    // linear gain recovers 1 − 1/e of the distance to unity in one time constant
    const g0 = dbToAmp(-12);
    const expected = 1 - (1 - g0) * Math.exp(-1);
    expect(-20 * Math.log10(expected)).toBeCloseTo(lim.gainReductionDb, 4);
    for (let i = 0; i < 20 * n; i++) lim.process(0);
    expect(lim.gainReductionDb).toBeLessThan(0.001);
  });

  it('never outputs NaN or Infinity', () => {
    const lim = new PeakLimiter(48000, -12);
    const ceil = dbToAmp(-12);
    const inputs = [0, 0, Number.NaN, 1e300, -1e300, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 5e-324, 1, Number.NaN];
    for (const x of inputs) {
      const y = lim.process(x);
      expect(Number.isFinite(y)).toBe(true);
      expect(Math.abs(y)).toBeLessThanOrEqual(ceil + 1e-9);
      expect(Number.isFinite(lim.gainReductionDb)).toBe(true);
      expect(lim.gainReductionDb).toBeGreaterThanOrEqual(0);
    }
    expect(lim.process(Number.NaN)).toBe(0);
    // silence after a huge input: output stays finite and the limiter recovers
    for (let i = 0; i < 48000 * 5; i++) expect(Number.isFinite(lim.process(0))).toBe(true);
    expect(lim.gainReductionDb).toBeLessThan(1e-3);
    lim.reset();
    expect(lim.gainReductionDb).toBe(0);
  });

  it('applies a new ceiling immediately and clamps invalid ones', () => {
    const lim = new PeakLimiter(48000, -6);
    lim.setCeilingDb(-20);
    expect(Math.abs(lim.process(1))).toBeLessThanOrEqual(dbToAmp(-20) + 1e-12);
    lim.setCeilingDb(Number.NaN);
    expect(lim.ceiling).toBeCloseTo(dbToAmp(-20), 12);
    lim.setCeilingDb(20);
    expect(lim.ceiling).toBe(1);
  });
});
