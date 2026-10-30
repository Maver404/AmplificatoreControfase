import { describe, expect, it } from 'vitest';
import { FFT, nextPow2 } from '../../src/dsp/fft.ts';
import {
  analyzeImpulseResponse,
  combineEstimates,
  estimateDelay,
  irToResponse,
  type DelayEstimate,
} from '../../src/dsp/latency-estimator.ts';
import { exponentialSweep, mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;

function gaussian(seed: number): () => number {
  const rnd = mulberry32(seed);
  return () => {
    const u = Math.max(1e-12, rnd());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  };
}

interface SystemSpec {
  delay: number;
  sign: 1 | -1;
  /** Reflection level relative to the direct sound (dB) and its extra delay (ms). */
  reflDb: number;
  reflMs: number;
  /** Exponential reverb tail: peak amplitude, RT60, start after the direct sound (ms). */
  tailAmp: number;
  rt60: number;
  tailStartMs: number;
  snrDb: number;
  seed: number;
}

const DEFAULT_SYSTEM: SystemSpec = {
  delay: 1234.37,
  sign: 1,
  reflDb: -4,
  reflMs: 3,
  tailAmp: 0.015,
  rt60: 0.3,
  tailStartMs: 5,
  snrDb: 10,
  seed: 11,
};

/** Plays `stim` through: sign·(δ(n-d) + g_r·δ(n-d-r)) + exponential noise tail, then adds white noise at snrDb. */
function simulate(stim: Float32Array, spec: SystemSpec, postS = 0.6): Float32Array {
  const recLen = stim.length + Math.round(postS * fs);
  const n = nextPow2(recLen + Math.round(0.5 * fs)) * 2;
  const fft = new FFT(n);
  const xr = new Float64Array(n);
  const xi = new Float64Array(n);
  for (let i = 0; i < stim.length; i++) xr[i] = stim[i];
  fft.forward(xr, xi);

  const tr = new Float64Array(n);
  const ti = new Float64Array(n);
  const g = gaussian(spec.seed);
  const t0 = Math.floor(spec.delay + (spec.tailStartMs / 1000) * fs);
  const tau = spec.rt60 / Math.log(1000);
  for (let k = 0; k < Math.round(0.45 * fs); k++) tr[t0 + k] = spec.tailAmp * g() * Math.exp(-k / fs / tau);
  fft.forward(tr, ti);

  const gr = Math.pow(10, spec.reflDb / 20);
  const dr = spec.delay + (spec.reflMs / 1000) * fs;
  for (let k = 0; k < n; k++) {
    const kk = k <= n / 2 ? k : k - n;
    const w = (2 * Math.PI * kk) / n;
    let hr = Math.cos(w * spec.delay) + gr * Math.cos(w * dr);
    let hi = -Math.sin(w * spec.delay) - gr * Math.sin(w * dr);
    if (k === n / 2) hi = 0;
    hr = spec.sign * hr + tr[k];
    hi = spec.sign * hi + ti[k];
    const yr = xr[k] * hr - xi[k] * hi;
    const yi = xr[k] * hi + xi[k] * hr;
    xr[k] = yr;
    xi[k] = yi;
  }
  fft.inverse(xr, xi);
  const out = new Float32Array(recLen);
  let p = 0;
  for (let i = 0; i < recLen; i++) {
    out[i] = xr[i];
    p += xr[i] * xr[i];
  }
  const noiseRms = Math.sqrt(p / recLen) * Math.pow(10, -spec.snrDb / 20);
  const gn = gaussian(spec.seed + 1000);
  for (let i = 0; i < recLen; i++) out[i] += noiseRms * gn();
  return out;
}

const sweepFull = exponentialSweep({ fs, f1: 20, f2: 20000, durationS: 2, fadeMs: 20, amplitude: 0.25 });
const sweepLow = exponentialSweep({ fs, f1: 20, f2: 1000, durationS: 2, fadeMs: 20, amplitude: 0.25 });

describe('estimateDelay on a sweep through delay + reflection + reverb + noise', () => {
  for (const [name, stim, f2] of [
    ['20 Hz – 20 kHz', sweepFull, 20000],
    ['20 Hz – 1 kHz', sweepLow, 1000],
  ] as const) {
    it(`finds the first arrival within ±0.2 samples (${name})`, () => {
      const rec = simulate(stim, DEFAULT_SYSTEM);
      const est = estimateDelay(rec, stim, fs, 20, f2, 'acoustic');
      expect(est.ok).toBe(true);
      expect(est.reason).toBeUndefined();
      expect(est.sign).toBe(1);
      expect(est.peak).toBeGreaterThan(0);
      expect(Math.abs(est.delaySamples - 1234.37)).toBeLessThan(0.2);
      expect(est.qualityDb).toBeGreaterThan(20);
      expect(est.ir.length).toBe(nextPow2(rec.length + stim.length));
    });

    it(`reports sign -1 for an inverted system (${name})`, () => {
      const rec = simulate(stim, { ...DEFAULT_SYSTEM, sign: -1, seed: 5 });
      const est = estimateDelay(rec, stim, fs, 20, f2, 'acoustic');
      expect(est.ok).toBe(true);
      expect(est.sign).toBe(-1);
      expect(est.peak).toBeLessThan(0);
      expect(Math.abs(est.delaySamples - 1234.37)).toBeLessThan(0.2);
    });
  }

  it('picks the direct sound even when a later reflection is stronger (first arrival within -6 dB)', () => {
    const rec = simulate(sweepFull, { ...DEFAULT_SYSTEM, reflDb: 3, seed: 21 });
    const est = estimateDelay(rec, sweepFull, fs, 20, 20000, 'acoustic');
    expect(est.ok).toBe(true);
    expect(Math.abs(est.delaySamples - 1234.37)).toBeLessThan(0.2);
  });

  it('works for other fractional delays and polarities', () => {
    for (const [delay, sign] of [
      [517.5, 1],
      [2049.91, -1],
      [88.02, 1],
    ] as const) {
      const rec = simulate(sweepLow, { ...DEFAULT_SYSTEM, delay, sign, seed: Math.round(delay) });
      const est = estimateDelay(rec, sweepLow, fs, 20, 1000, 'acoustic');
      expect(est.ok).toBe(true);
      expect(est.sign).toBe(sign);
      expect(Math.abs(est.delaySamples - delay)).toBeLessThan(0.2);
    }
  });

  it('narrow band (20–200 Hz): a pure delay passes the loopback gate despite the band-edge skirt', () => {
    const stim = exponentialSweep({ fs, f1: 20, f2: 200, durationS: 2, fadeMs: 20, amplitude: 0.25 });
    const rec = simulate(stim, { ...DEFAULT_SYSTEM, reflDb: -200, tailAmp: 0, snrDb: 30, seed: 31 });
    const est = estimateDelay(rec, stim, fs, 20, 200, 'loopback');
    expect(est.ok).toBe(true);
    expect(est.qualityDb).toBeGreaterThan(30);
    expect(Math.abs(est.delaySamples - 1234.37)).toBeLessThan(0.2);
  });

  it('narrow band (20–200 Hz) with a reverberant tail: picks the arrival, not the opposite-sign skirt before it', () => {
    const stim = exponentialSweep({ fs, f1: 20, f2: 200, durationS: 2, fadeMs: 20, amplitude: 0.25 });
    const rec = simulate(stim, { ...DEFAULT_SYSTEM, reflDb: -200 });
    const est = estimateDelay(rec, stim, fs, 20, 200, 'acoustic');
    expect(est.ok).toBe(true);
    expect(est.sign).toBe(1);
    // A 200 Hz band cannot separate the direct sound from a tail starting 5 ms later: a few samples.
    expect(Math.abs(est.delaySamples - 1234.37)).toBeLessThan(8);
  });

  it('rejects an all-noise recording with an Italian reason', () => {
    const g = gaussian(99);
    const rec = new Float32Array(sweepFull.length + Math.round(0.6 * fs));
    for (let i = 0; i < rec.length; i++) rec[i] = 0.05 * g();
    for (const mode of ['acoustic', 'loopback'] as const) {
      const est = estimateDelay(rec, sweepFull, fs, 20, 20000, mode);
      expect(est.ok).toBe(false);
      expect(est.qualityDb).toBeLessThan(10);
      expect(est.reason).toMatch(/qualità|misura/);
    }
  });

  it('rejects a silent recording and a clipped recording', () => {
    const silent = new Float32Array(sweepLow.length + 1000);
    const s = estimateDelay(silent, sweepLow, fs, 20, 1000, 'acoustic');
    expect(s.ok).toBe(false);
    expect(s.reason).toMatch(/segnale/);

    const rec = simulate(sweepLow, DEFAULT_SYSTEM);
    for (let i = 0; i < rec.length; i++) rec[i] = Math.max(-1, Math.min(1, rec[i] * 20));
    const c = estimateDelay(rec, sweepLow, fs, 20, 1000, 'acoustic');
    expect(c.ok).toBe(false);
    expect(c.reason).toMatch(/saturazione/);
  });
});

describe('analyzeImpulseResponse', () => {
  /** Unit impulse at `at` plus uniform noise of peak `noise` everywhere. */
  function syntheticIr(at: number, noise: number, seed: number): Float64Array {
    const rnd = mulberry32(seed);
    const ir = new Float64Array(16384);
    for (let i = 0; i < ir.length; i++) ir[i] = noise * (2 * rnd() - 1);
    ir[at] = 1;
    return ir;
  }

  it('loopback mode is stricter than acoustic mode', () => {
    // Noise peaks at ≈ -14 dB: good enough for acoustic (≥ 10 dB), not for loopback (≥ 18 dB).
    const ir = syntheticIr(3000, 0.2, 4);
    const ac = analyzeImpulseResponse(ir, fs, 'acoustic');
    const lb = analyzeImpulseResponse(ir, fs, 'loopback');
    expect(ac.qualityDb).toBeGreaterThan(10);
    expect(ac.qualityDb).toBeLessThan(18);
    expect(ac.ok).toBe(true);
    expect(lb.ok).toBe(false);
    expect(lb.reason).toMatch(/Loopback/);
    expect(Math.abs(ac.delaySamples - 3000)).toBeLessThan(0.05);
    // A clean impulse passes both.
    const clean = syntheticIr(3000, 0.01, 5);
    expect(analyzeImpulseResponse(clean, fs, 'loopback').ok).toBe(true);
  });

  it('refines a band-limited fractional impulse to sub-sample precision', () => {
    const n = 8192;
    const d = 700.3;
    const ir = new Float64Array(n);
    // Hann-windowed sinc low-passed at 0.4·fs, centred at d.
    for (let i = 0; i < n; i++) {
      const x = i - d;
      if (Math.abs(x) > 64) continue;
      const s = x === 0 ? 1 : Math.sin(Math.PI * 0.8 * x) / (Math.PI * 0.8 * x);
      ir[i] = -s * (0.5 + 0.5 * Math.cos((Math.PI * x) / 64));
    }
    const est = analyzeImpulseResponse(ir, fs, 'loopback');
    expect(est.sign).toBe(-1);
    expect(Math.abs(est.delaySamples - d)).toBeLessThan(0.02);
  });

  it('fails cleanly on empty, zero or non-finite input', () => {
    expect(analyzeImpulseResponse(new Float64Array(4), fs, 'acoustic').ok).toBe(false);
    expect(analyzeImpulseResponse(new Float64Array(1024), fs, 'acoustic').ok).toBe(false);
    const bad = new Float64Array(1024);
    bad[10] = Number.NaN;
    expect(analyzeImpulseResponse(bad, fs, 'acoustic').reason).toMatch(/valori non validi/);
  });
});

describe('combineEstimates', () => {
  const e = (delaySamples: number, sign: 1 | -1 = 1, ok = true): DelayEstimate => ({
    delaySamples,
    sign,
    peak: sign * 0.5,
    qualityDb: 30,
    ok,
  });

  it('takes the median and reports the spread', () => {
    const r = combineEstimates([e(1000.1), e(1000.3), e(1000.15), e(1000.2), e(1000.12)]);
    expect(r.ok).toBe(true);
    expect(r.delaySamples).toBeCloseTo(1000.15, 10);
    expect(r.count).toBe(5);
    expect(r.stdSamples).toBeGreaterThan(0.05);
    expect(r.stdSamples).toBeLessThan(0.1);
    expect(combineEstimates([e(10), e(12)]).delaySamples).toBeCloseTo(11, 10);
  });

  it('is robust to one outlier in the median but rejects a large spread', () => {
    const spread = combineEstimates([e(1000), e(1000.1), e(1001.5)]);
    expect(spread.delaySamples).toBeCloseTo(1000.1, 10);
    expect(spread.stdSamples).toBeGreaterThan(0.5);
    expect(spread.ok).toBe(false);
    expect(spread.reason).toMatch(/non ripetibili/);
  });

  it('rejects disagreeing signs', () => {
    const r = combineEstimates([e(1000), e(1000.1, -1), e(1000.05)]);
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/polarità/);
  });

  it('ignores invalid repetitions but needs a valid majority', () => {
    const one = combineEstimates([e(1000), e(5000, 1, false), e(1000.2)]);
    expect(one.ok).toBe(true);
    expect(one.count).toBe(2);
    expect(one.delaySamples).toBeCloseTo(1000.1, 10);
    const most = combineEstimates([e(1000), e(5000, 1, false), e(5000, 1, false)]);
    expect(most.ok).toBe(false);
    expect(combineEstimates([]).ok).toBe(false);
    expect(combineEstimates([e(1, 1, false)]).ok).toBe(false);
  });
});

describe('irToResponse', () => {
  it('returns the DTFT of the IR', () => {
    const ir = new Float64Array(200);
    ir[100] = 0.5;
    ir[150] = -0.25;
    const freqs = Float64Array.from([0, 100, 1000, 7777.7]);
    const { re, im } = irToResponse(ir, fs, freqs);
    for (let j = 0; j < freqs.length; j++) {
      const w = (2 * Math.PI * freqs[j]) / fs;
      expect(re[j]).toBeCloseTo(0.5 * Math.cos(-w * 100) - 0.25 * Math.cos(-w * 150), 10);
      expect(im[j]).toBeCloseTo(0.5 * Math.sin(-w * 100) - 0.25 * Math.sin(-w * 150), 10);
    }
  });
});
