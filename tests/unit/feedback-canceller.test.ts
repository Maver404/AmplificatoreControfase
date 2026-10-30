import { describe, expect, it } from 'vitest';
import {
  FEEDBACK_FIR_FADE,
  FEEDBACK_FIR_MARGIN,
  FeedbackCanceller,
  designFeedbackFir,
} from '../../src/dsp/feedback-canceller.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;

/** Short decaying IR (the part after the bulk delay of a speaker → reference path). */
function shortIr(seed: number, len = 40, scale = 0.3): Float64Array {
  const rnd = mulberry32(seed);
  const h = new Float64Array(len);
  for (let k = 0; k < len; k++) h[k] = scale * (2 * rnd() - 1) * Math.exp(-k / 8);
  return h;
}

/** Feedback component at sample n: Σ_k h[k]·y[n − bulk − k]. */
function convolveAt(y: Float64Array, n: number, bulk: number, h: Float64Array): number {
  let acc = 0;
  for (let k = 0; k < h.length; k++) {
    const i = n - bulk - k;
    if (i >= 0) acc += h[k] * y[i];
  }
  return acc;
}

const powerDb = (x: Float64Array, from = 0): number => {
  let p = 0;
  for (let i = from; i < x.length; i++) p += x[i] * x[i];
  return 10 * Math.log10(p / (x.length - from) + 1e-300);
};

describe('FeedbackCanceller', () => {
  it('removes the fed-back component of an open loop (> 30 dB) with the exact FIR, two channels', () => {
    const N = 20000;
    const rnd = mulberry32(1);
    const yL = new Float64Array(N);
    const yR = new Float64Array(N);
    const s = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      yL[i] = 2 * rnd() - 1;
      yR[i] = 2 * rnd() - 1;
      s[i] = 0.1 * (2 * rnd() - 1);
    }
    const hL = shortIr(2);
    const hR = shortIr(3);
    const fc = new FeedbackCanceller(fs, 4096, 256);
    fc.setFir(0, 1300, hL);
    fc.setFir(1, 1450, hR);
    const fed = new Float64Array(N);
    const err = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      fed[n] = convolveAt(yL, n, 1300, hL) + convolveAt(yR, n, 1450, hR);
      const x = s[n] + fed[n];
      const xHat = x - fc.predict();
      err[n] = xHat - s[n];
      fc.pushOutputs(yL[n], yR[n]);
    }
    const reduction = powerDb(fed, 2000) - powerDb(err, 2000);
    expect(reduction).toBeGreaterThan(30);
    expect(reduction).toBeGreaterThan(200); // exact FIR: only rounding remains
  });

  it('keeps a closed loop with loop gain > 1 stable and the reference clean', () => {
    // Output = G·x̂ delayed: without the canceller this loop (|G·h| > 1 at some frequency) diverges.
    const N = 30000;
    const rnd = mulberry32(9);
    const h = shortIr(4, 30, 1);
    const fc = new FeedbackCanceller(fs, 2048, 128);
    fc.setFir(0, 1300, h);
    const y = new Float64Array(N);
    const xh = new Float64Array(N);
    const G = 3;
    let maxErr = 0;
    let maxOut = 0;
    for (let n = 0; n < N; n++) {
      const s = 0.05 * (2 * rnd() - 1);
      const x = s + convolveAt(y, n, 1300, h);
      xh[n] = x - fc.predict();
      y[n] = n >= 10 ? G * xh[n - 10] : 0;
      fc.pushOutputs(y[n], 0);
      maxErr = Math.max(maxErr, Math.abs(xh[n] - s));
      maxOut = Math.max(maxOut, Math.abs(y[n]));
    }
    expect(maxErr).toBeLessThan(1e-9);
    expect(maxOut).toBeLessThan(G * 0.05 + 1e-6);
  });

  it('is transparent without FIRs, sanitises inputs and rejects non-finite taps', () => {
    const fc = new FeedbackCanceller(fs, 100, 16);
    for (let i = 0; i < 50; i++) {
      expect(fc.predict()).toBe(0);
      fc.pushOutputs(Number.NaN, Number.POSITIVE_INFINITY);
    }
    const bad = new Float64Array(4);
    bad[2] = Number.NaN;
    fc.setFir(0, 5, bad);
    fc.pushOutputs(1, 1);
    expect(fc.predict()).toBe(0);
    // Bulk delay is clamped to ≥ 1 and taps beyond maxTaps are dropped.
    fc.reset();
    const taps = new Float64Array(32).fill(1);
    fc.setFir(1, 0, taps);
    fc.pushOutputs(0, 1);
    expect(fc.predict()).toBe(1); // only y[n−1] so far
    for (let i = 0; i < 40; i++) fc.pushOutputs(0, 1);
    expect(fc.predict()).toBe(16);
    fc.setFir(1, 3, null);
    expect(fc.predict()).toBe(0);
  });

  it('honours the bulk delay exactly', () => {
    const fc = new FeedbackCanceller(fs, 1000, 4);
    fc.setFir(0, 7, Float64Array.from([1]));
    const out: number[] = [];
    for (let n = 0; n < 12; n++) {
      out.push(fc.predict());
      fc.pushOutputs(n === 0 ? 1 : 0, 0);
    }
    // An impulse output at n = 0 reappears in the prediction at n = 7.
    expect(out.indexOf(1)).toBe(7);
    expect(out.filter((v) => v !== 0)).toHaveLength(1);
  });
});

describe('designFeedbackFir', () => {
  it('extracts the window that starts a margin before the arrival, with fades', () => {
    const ir = new Float64Array(8192);
    const rnd = mulberry32(5);
    for (let i = 0; i < ir.length; i++) ir[i] = 2 * rnd() - 1;
    const { bulkDelay, taps } = designFeedbackFir(ir, 1300.4, 256);
    expect(bulkDelay).toBe(1300 - FEEDBACK_FIR_MARGIN);
    expect(taps.length).toBe(256);
    const fadeIn = FEEDBACK_FIR_MARGIN / 2;
    for (let k = fadeIn; k < 256 - FEEDBACK_FIR_FADE; k++) expect(taps[k]).toBe(ir[bulkDelay + k]);
    for (let k = 0; k < fadeIn; k++) expect(Math.abs(taps[k])).toBeLessThan(Math.abs(ir[bulkDelay + k]) + 1e-15);
    expect(Math.abs(taps[255])).toBeLessThan(Math.abs(ir[bulkDelay + 255]) * 0.01);
    // Short FIRs shrink the margin; early arrivals clamp the bulk delay to 1.
    expect(designFeedbackFir(ir, 1300, 64).bulkDelay).toBe(1300 - 16);
    expect(designFeedbackFir(ir, 10, 256).bulkDelay).toBe(1);
    // Beyond the end of the IR the taps are zero.
    const tail = designFeedbackFir(ir, 8100, 256);
    expect(tail.taps[200]).toBe(0);
  });

  it('a designed FIR cancels the measured path (> 30 dB) inside the loop', () => {
    // Measured IR = bulk 1300 + short IR, as the sweep analysis would return it.
    const h = shortIr(6);
    const measured = new Float64Array(8192);
    measured.set(h, 1300);
    const { bulkDelay, taps } = designFeedbackFir(measured, 1300, 256);
    const fc = new FeedbackCanceller(fs, 4096, 256);
    fc.setFir(0, bulkDelay, taps);
    const N = 20000;
    const rnd = mulberry32(8);
    const y = new Float64Array(N);
    const fed = new Float64Array(N);
    const err = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      y[n] = 2 * rnd() - 1;
      fed[n] = convolveAt(y, n, 1300, h);
      err[n] = fed[n] - fc.predict();
      fc.pushOutputs(y[n], 0);
    }
    expect(powerDb(fed, 2000) - powerDb(err, 2000)).toBeGreaterThan(30);
  });
});
