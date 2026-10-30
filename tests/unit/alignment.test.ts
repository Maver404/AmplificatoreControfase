import { describe, expect, it } from 'vitest';
import {
  harmonicFactors,
  loopGainDb,
  residualDb,
  sampleResponse,
  solveAlignment,
  type AlignmentInput,
  type FreqResponse,
} from '../../src/dsp/alignment.ts';
import { biquadResponse, lowpass } from '../../src/dsp/biquad.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { D_MIN_SAMPLES } from '../../src/dsp/params.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;
/** 10 … 1000 Hz in 1 Hz steps. */
const FREQS = Float64Array.from({ length: 991 }, (_, i) => 10 + i);

/** g·e^{-jω·d}: a pure delay of d samples with gain g (negative g = inverted). */
function delayGain(d: number, g: number, freqs: Float64Array = FREQS): FreqResponse {
  const re = new Float64Array(freqs.length);
  const im = new Float64Array(freqs.length);
  freqs.forEach((f, i) => {
    const w = (-2 * Math.PI * f * d) / fs;
    re[i] = g * Math.cos(w);
    im[i] = g * Math.sin(w);
  });
  return { freqs, re, im };
}

/** 2nd-order Bessel-like low-pass at 500 Hz (Q = 1/√3, f0 = 1.272·cutoff), built from biquad.ts. */
const LP = lowpass(fs, 1.272 * 500, 1 / Math.sqrt(3));
const conditioning = (f: number): Complex => biquadResponse(LP, f, fs);

/** Group delay of the conditioning in samples (central difference of the phase). */
function groupDelay(f: number): number {
  const df = 0.01;
  const a = conditioning(f - df);
  const b = conditioning(f + df);
  let dp = Math.atan2(b.im, b.re) - Math.atan2(a.im, a.re);
  dp -= 2 * Math.PI * Math.round(dp / (2 * Math.PI));
  return (-dp / (2 * Math.PI * 2 * df)) * fs;
}

const BAND: [number, number] = [20, 200];
const TAU_P = 400.3;
const A = 0.8;
const TAU_S = 150.6;
const S = 0.5;

function input(over: Partial<AlignmentInput> = {}): AlignmentInput {
  return {
    fs,
    band: BAND,
    primary: delayGain(TAU_P, A),
    secondary: [delayGain(TAU_S, S), null],
    conditioning,
    speakers: 'single',
    maxDelaySamples: 0.5 * fs,
    ...over,
  };
}

describe('solveAlignment: single speaker', () => {
  it('delay ≈ τP − τS − LP group delay, gain ≈ a/s, polarity inverted, ≥ 25 dB', () => {
    const gd = groupDelay(100);
    expect(gd).toBeGreaterThan(15);
    const r = solveAlignment(input());
    const ch = r.channels[0];
    expect(ch.use).toBe(true);
    expect(r.channels[1].use).toBe(false);
    expect(Math.abs(ch.delaySamples - (TAU_P - TAU_S - gd))).toBeLessThan(0.5);
    expect(ch.gainLin / (A / S)).toBeGreaterThan(1);
    expect(ch.gainLin / (A / S)).toBeLessThan(1.05);
    expect(ch.polarityInvert).toBe(true);
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
    expect(r.causal).toBe(true);
    expect(r.requiredExtraLeadSeconds).toBe(0);
    expect(residualDb(input(), r.channels)).toBeCloseTo(-r.predictedAttenuationDb, 6);
  });

  it('without conditioning the solution is exact', () => {
    const r = solveAlignment(input({ conditioning: () => ({ re: 1, im: 0 }) }));
    expect(r.channels[0].delaySamples).toBeCloseTo(TAU_P - TAU_S, 4);
    expect(r.channels[0].gainLin).toBeCloseTo(A / S, 9);
    expect(r.predictedAttenuationDb).toBeGreaterThan(100);
  });

  it('flips the polarity for an inverting secondary path', () => {
    const r = solveAlignment(input({ secondary: [delayGain(TAU_S, -S), null] }));
    const ch = r.channels[0];
    expect(ch.polarityInvert).toBe(false);
    expect(Math.abs(ch.delaySamples - (TAU_P - TAU_S - groupDelay(100)))).toBeLessThan(0.5);
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
  });

  it('works when the secondary is measured on a different (coarser) grid', () => {
    const coarse = Float64Array.from({ length: 300 }, (_, i) => 5 + 2 * i);
    const r = solveAlignment(input({ secondary: [delayGain(TAU_S, S, coarse), null] }));
    expect(Math.abs(r.channels[0].delaySamples - (TAU_P - TAU_S - groupDelay(100)))).toBeLessThan(0.5);
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
  });

  it('reports a non-causal setup and the lead it would need', () => {
    const tauP = 100;
    const tauS = 300;
    const r = solveAlignment(input({ primary: delayGain(tauP, A), secondary: [delayGain(tauS, S), null] }));
    expect(r.causal).toBe(false);
    const want = (D_MIN_SAMPLES - (tauP - tauS - groupDelay(100))) / fs;
    expect(Math.abs(r.requiredExtraLeadSeconds - want) * fs).toBeLessThan(1);
    // The returned (causal) solution stays inside the allowed range and is poor.
    expect(r.channels[0].delaySamples).toBeGreaterThanOrEqual(D_MIN_SAMPLES);
    expect(r.predictedAttenuationDb).toBeLessThan(10);
  });

  it('tonal mode resolves the delay modulo the period and is always causal', () => {
    const f0 = 100;
    const period = fs / f0;
    const tauP = 100;
    const tauS = 300;
    const inp = input({
      primary: delayGain(tauP, A),
      secondary: [delayGain(tauS, S), null],
      band: [20, 250],
      tonalF0: f0,
    });
    const r = solveAlignment(inp);
    expect(r.causal).toBe(true);
    expect(r.requiredExtraLeadSeconds).toBe(0);
    const d = r.channels[0].delaySamples;
    expect(d).toBeGreaterThanOrEqual(D_MIN_SAMPLES);
    expect(d).toBeLessThan(D_MIN_SAMPLES + period);
    const ideal = tauP - tauS - groupDelay(150);
    let diff = (d - ideal) % period;
    if (diff > period / 2) diff -= period;
    if (diff < -period / 2) diff += period;
    expect(Math.abs(diff)).toBeLessThan(1);
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
    expect(r.channels[0].polarityInvert).toBe(true);
  });

  it('returns unused channels when there is nothing to solve', () => {
    const none = solveAlignment(input({ secondary: [null, delayGain(TAU_S, S)] }));
    expect(none.channels[0].use).toBe(false);
    expect(none.channels[1].use).toBe(false);
    expect(none.predictedAttenuationDb).toBe(0);
    const outOfBand = solveAlignment(input({ band: [5000, 6000] }));
    expect(outOfBand.channels[0].use).toBe(false);
  });

  it('rejects malformed responses', () => {
    const bad: FreqResponse = { freqs: FREQS, re: new Float64Array(3), im: new Float64Array(3) };
    expect(() => solveAlignment(input({ primary: bad }))).toThrow(RangeError);
    expect(() => solveAlignment(input({ weights: new Float64Array(5) }))).toThrow(RangeError);
  });
});

describe('solveAlignment: two speakers', () => {
  it("'shared' with two equal speakers gives each half the gain; independent solutions would not cancel", () => {
    const single = solveAlignment(input());
    const shared = input({ speakers: 'shared', secondary: [delayGain(TAU_S, S), delayGain(TAU_S, S)] });
    const r = solveAlignment(shared);
    for (const ch of r.channels) {
      expect(ch.use).toBe(true);
      expect(ch.polarityInvert).toBe(true);
      expect(Math.abs(ch.gainLin / single.channels[0].gainLin - 0.5)).toBeLessThan(0.01);
      expect(Math.abs(ch.delaySamples - single.channels[0].delaySamples)).toBeLessThan(0.1);
    }
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
    expect(residualDb(shared, r.channels)).toBeLessThan(-25);
    // Both full single-speaker solutions together over-cancel: residual |P − 2P| = |P|.
    const both = [single.channels[0], { ...single.channels[0] }] as const;
    expect(residualDb(shared, [both[0], both[1]])).toBeGreaterThan(-1);
  });

  it("'shared' with different speakers still cancels jointly", () => {
    const shared = input({ speakers: 'shared', secondary: [delayGain(TAU_S, S), delayGain(180.2, 0.3)] });
    const r = solveAlignment(shared);
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
    expect(r.channels[0].gainLin).toBeGreaterThan(0);
    expect(r.channels[1].gainLin).toBeGreaterThan(0);
    expect(r.channels[0].polarityInvert).toBe(true);
    expect(r.channels[1].polarityInvert).toBe(true);
    expect(Math.abs(r.channels[1].delaySamples - (TAU_P - 180.2 - groupDelay(100)))).toBeLessThan(1);
  });

  it("'separate' solves each channel against P and reports the worst", () => {
    const sep = input({ speakers: 'separate', secondary: [delayGain(TAU_S, S), delayGain(170, -0.25)] });
    const r = solveAlignment(sep);
    expect(r.channels[0].gainLin / (A / S)).toBeCloseTo(1, 1);
    expect(r.channels[1].gainLin / (A / 0.25)).toBeCloseTo(1, 1);
    expect(r.channels[1].polarityInvert).toBe(false);
    expect(Math.abs(r.channels[1].delaySamples - (TAU_P - 170 - groupDelay(100)))).toBeLessThan(0.5);
    expect(r.predictedAttenuationDb).toBeGreaterThanOrEqual(25);
    expect(residualDb(sep, r.channels)).toBeCloseTo(-r.predictedAttenuationDb, 6);
  });
});

describe('sampleResponse', () => {
  it('interpolates magnitude and unwrapped phase', () => {
    const coarse = Float64Array.from({ length: 100 }, (_, i) => 10 + 10 * i);
    const fine = Float64Array.from([15, 333.3, 777.7]);
    const s = sampleResponse(delayGain(37.5, 0.7, coarse), fine);
    const ref = delayGain(37.5, 0.7, fine);
    for (let i = 0; i < fine.length; i++) {
      expect(s.re[i]).toBeCloseTo(ref.re[i], 9);
      expect(s.im[i]).toBeCloseTo(ref.im[i], 9);
    }
  });
});

describe('loopGainDb', () => {
  const g = (k: number) => (): Complex => ({ re: k, im: 0 });
  it('computes |Σ G·F| on a known case', () => {
    const fb = delayGain(100, 0.4);
    const one = loopGainDb([g(0.5), g(0.5)], [fb, null], [20, 500]);
    expect(one.maxDb).toBeCloseTo(20 * Math.log10(0.2), 9);
    const two = loopGainDb([g(0.5), g(0.5)], [fb, delayGain(100, 0.4)], [20, 500]);
    expect(two.maxDb).toBeCloseTo(20 * Math.log10(0.4), 9);
    // Opposite channels cancel exactly.
    const cancel = loopGainDb([g(0.5), g(-0.5)], [fb, delayGain(100, 0.4)], [20, 500]);
    expect(cancel.maxDb).toBe(-Infinity);
  });

  it('finds the frequency of the maximum and honours the band', () => {
    const peaky: FreqResponse = { freqs: FREQS, re: Float64Array.from(FREQS, (f) => (f === 300 ? 2 : f === 900 ? 5 : 0.1)), im: new Float64Array(FREQS.length) };
    const r = loopGainDb([g(1), g(0)], [peaky, null], [20, 500]);
    expect(r.atHz).toBe(300);
    expect(r.maxDb).toBeCloseTo(20 * Math.log10(2), 9);
    expect(loopGainDb([g(1), g(1)], [null, null], [20, 500]).maxDb).toBe(-Infinity);
  });
});

describe('harmonicFactors (tonal trims of one harmonic)', () => {
  const c = (re: number, im: number): Complex => ({ re, im });
  const mul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
  const add = (...v: Complex[]): Complex => v.reduce((s, x) => ({ re: s.re + x.re, im: s.im + x.im }), c(0, 0));
  const abs = (a: Complex): number => Math.hypot(a.re, a.im);
  const WIDE = { min: 0, max: Infinity };
  const TRIM = { min: Math.pow(10, -30 / 20), max: Math.pow(10, 6 / 20) };

  it("one speaker: T = −p/b exactly; outside the bounds the magnitude is clamped, the phase kept", () => {
    const p = c(0.3, -0.4);
    const b = c(-0.2, 0.25);
    const r = harmonicFactors(p, [b, null], 'single', WIDE);
    expect(abs(add(p, mul(r.t[0]!, b)))).toBeLessThan(1e-15);
    expect(r.t[1]).toBeNull();
    expect(r.capped).toEqual([false, false]);
    // |T| = 0.5/0.32 = 1.56 > 2 dB: clamped to 2 dB with the same phase.
    const k = harmonicFactors(p, [b, null], 'single', { min: 0.1, max: Math.pow(10, 2 / 20) });
    expect(k.capped[0]).toBe(true);
    expect(abs(k.t[0]!)).toBeCloseTo(Math.pow(10, 2 / 20), 12);
    expect(Math.atan2(k.t[0]!.im, k.t[0]!.re)).toBeCloseTo(Math.atan2(r.t[0]!.im, r.t[0]!.re), 12);
    // 'single' ignores channel 1; a channel that does not reach the harmonic gets no factor.
    expect(harmonicFactors(p, [b, b], 'single', WIDE).t[1]).toBeNull();
    expect(harmonicFactors(p, [c(0, 0), null], 'single', WIDE).t[0]).toBeNull();
  });

  it("'separate': each speaker against p alone (its own zone)", () => {
    const p = c(0.3, -0.4);
    const b: [Complex, Complex] = [c(-0.2, 0.25), c(0.1, 0.5)];
    const r = harmonicFactors(p, b, 'separate', WIDE);
    for (const ch of [0, 1] as const) expect(abs(add(p, mul(r.t[ch]!, b[ch])))).toBeLessThan(1e-15);
  });

  it("'shared': the minimum change Σ|T − 1|² that nulls the residual (an exact broadband split stays 1)", () => {
    const p = c(0.3, -0.4);
    const b: [Complex, Complex] = [c(-0.2, 0.25), c(-0.05, 0.1)];
    const r = harmonicFactors(p, b, 'shared', WIDE);
    const t0 = r.t[0]!;
    const t1 = r.t[1]!;
    expect(abs(add(p, mul(t0, b[0]), mul(t1, b[1])))).toBeLessThan(1e-15);
    const norm = (a: Complex, d: Complex): number => abs(add(a, c(-1, 0))) ** 2 + abs(add(d, c(-1, 0))) ** 2;
    // Every other null moves along (b1, −b0)·ε and is farther from (1, 1).
    for (const e of [c(0.01, 0), c(0, 0.01), c(-0.2, 0.1)]) {
      const a = add(t0, mul(e, b[1]));
      const d = add(t1, mul(e, c(-b[0].re, -b[0].im)));
      expect(abs(add(p, mul(a, b[0]), mul(d, b[1])))).toBeLessThan(1e-14);
      expect(norm(a, d)).toBeGreaterThan(norm(t0, t1));
    }
    // The broadband split already cancels this harmonic: nothing changes.
    const exact = harmonicFactors(c(-b[0].re - b[1].re, -b[0].im - b[1].im), b, 'shared', WIDE);
    expect(abs(add(exact.t[0]!, c(-1, 0)))).toBeLessThan(1e-15);
    expect(abs(add(exact.t[1]!, c(-1, 0)))).toBeLessThan(1e-15);
  });

  it("'shared': a channel clamped at the trims' range leaves the rest to the other one", () => {
    const rnd = mulberry32(42);
    const draw = (): Complex => c(rnd() - 0.5, rnd() - 0.5);
    const seen = { none: 0, oneExact: 0, both: 0 };
    for (let i = 0; i < 4000; i++) {
      const p = draw();
      const b: [Complex, Complex] = [mul(draw(), c(0.6, 0)), mul(draw(), c(0.6, 0))];
      const r = harmonicFactors(p, b, 'shared', TRIM);
      const res = abs(add(p, mul(r.t[0]!, b[0]), mul(r.t[1]!, b[1])));
      for (const ch of [0, 1] as const) {
        expect(abs(r.t[ch]!)).toBeLessThanOrEqual(TRIM.max * (1 + 1e-9));
        expect(abs(r.t[ch]!)).toBeGreaterThanOrEqual(TRIM.min * (1 - 1e-9));
      }
      if (!r.capped[0] && !r.capped[1]) {
        expect(res).toBeLessThan(1e-12);
        seen.none++;
      } else if (r.capped[0] !== r.capped[1]) {
        // The clamped one is at a bound; the other was re-solved against what is left and still nulls it.
        const c0 = r.capped[0] ? 0 : 1;
        const m = abs(r.t[c0]!);
        expect(Math.min(Math.abs(m - TRIM.max), Math.abs(m - TRIM.min))).toBeLessThan(1e-12);
        expect(res).toBeLessThan(1e-12);
        seen.oneExact++;
      } else seen.both++;
    }
    expect(seen.none).toBeGreaterThan(100);
    expect(seen.oneExact).toBeGreaterThan(10);
    expect(seen.both).toBeGreaterThan(10);
  });
});
