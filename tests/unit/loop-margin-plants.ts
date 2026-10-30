// Seeded random tonal plants for the stability-margin tests (loop-margin.test.ts) and the tonal solve's invariants
// (solver-invariants.test.ts), and evaluators independent of the solver: the neutralised loop on a fine grid and the
// closed-loop attenuation R = P + Σ G·S/(1 − Σ G·F̃), per harmonic and as the solver's broadband prediction.
import { calibrationFreqs, createCalibration, feedbackFirs, pathResponse } from '../../src/audio/calibration.ts';
import type { CalibrationData, PathData } from '../../src/audio/calibration.ts';
import type { SpeakerConfig } from '../../src/dsp/alignment.ts';
import { irToResponse } from '../../src/dsp/latency-estimator.ts';
import { clampParams, minTonalQ } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { channelResponse, channelResponseAt } from '../../src/dsp/response.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

export const fs = 48000;
export type Taps = [number, number][];

/** A path of integer-lag impulses stored like measurePath (IR from lag 0, `pad` zeros after the last). */
export function impulsePath(arrivals: Taps, band: [number, number], pad = 64): PathData {
  const last = Math.max(...arrivals.map((a) => a[0]));
  const ir = new Array<number>(last + pad).fill(0);
  for (const [lag, gain] of arrivals) ir[lag] += gain;
  const freqs = calibrationFreqs(fs, band);
  const peak = Math.max(...arrivals.map((a) => Math.abs(a[1])));
  const strongest = arrivals.filter((a) => Math.abs(a[1]) >= 0.5 * peak).reduce((a, b) => (b[0] < a[0] ? b : a));
  const base: PathData = {
    delaySamples: strongest[0],
    sign: strongest[1] >= 0 ? 1 : -1,
    qualityDb: 40,
    stdSamples: 0.05,
    freqs: [],
    re: [],
    im: [],
    ir,
    irOffset: 0,
  };
  const r = pathResponse(base, fs, freqs);
  return { ...base, freqs: Array.from(freqs), re: Array.from(r.re), im: Array.from(r.im) };
}

/**
 * A tonal plant: band, f0, harmonics and Q; one speaker, two on one zone ('shared') or two zones; a pure-delay
 * primary path; per harmonic a PSD and (40 % of plants) a coherence; integer-lag secondary and feedback paths,
 * the feedback often with a comb of late reflections at one harmonic's period (a strong loop there) or scattered
 * reflections beyond the neutraliser's reach.
 */
export interface Plant {
  seed: number;
  band: [number, number];
  f0: number;
  harmonics: number;
  q: number;
  speakers: SpeakerConfig;
  primaryGain: number;
  primaryDelayS: number;
  /** PSD per harmonic (index h − 1, up to the band), 1e-3 elsewhere. */
  psdH: number[];
  cohH: number[] | null;
  secondary: [Taps | null, Taps | null];
  feedback: [Taps | null, Taps | null];
  fbPad: [number, number];
}

export function randomPlant(seed: number, over: Partial<Plant> = {}): Plant {
  const r = mulberry32(seed);
  const u = (a: number, b: number): number => a + (b - a) * r();
  const pick = <T>(v: readonly T[]): T => v[Math.floor(r() * v.length)];
  const band: [number, number] = pick([[20, 1000], [20, 1000], [20, 2000], [40, 600]] as [number, number][]);
  const f0 = u(40, 220);
  const harmonics = 1 + Math.floor(r() * 8);
  const qs = [minTonalQ(harmonics), 5, 10, 20, 30, 30, 60, 120, 200].filter((q) => q >= minTonalQ(harmonics));
  const q = pick(qs);
  const speakers: SpeakerConfig = pick(['single', 'single', 'shared', 'separate'] as SpeakerConfig[]);
  const secPath = (): Taps => {
    const d = Math.round(u(300, 1400));
    const t: Taps = [[d, u(0.5, 1.5) * (r() < 0.5 ? -1 : 1)]];
    const n = Math.floor(r() * 4);
    for (let i = 0; i < n; i++) t.push([d + Math.round(u(5, 1500)), u(-0.7, 0.7)]);
    return t;
  };
  const fbPath = (): Taps => {
    const d = Math.round(u(150, 1000));
    const t: Taps = [[d, u(0.05, 0.6) * (r() < 0.5 ? -1 : 1)]];
    const kind = r();
    if (kind < 0.35) {
      // A comb of late reflections at the period of one harmonic (strong loop there).
      const h = 1 + Math.floor(r() * harmonics);
      const per = fs / (h * f0);
      const off = Math.round(u(1100, 2600));
      const g = u(0.08, 0.5);
      const n = 2 + Math.floor(r() * 6);
      for (let i = 0; i < n; i++) t.push([d + off + Math.round(per * i), g]);
    } else if (kind < 0.7) {
      const n = 1 + Math.floor(r() * 6);
      for (let i = 0; i < n; i++) t.push([d + Math.round(u(200, 4000)), u(-0.5, 0.5)]);
    } else {
      const n = Math.floor(r() * 3);
      for (let i = 0; i < n; i++) t.push([d + Math.round(u(5, 900)), u(-0.4, 0.4)]);
    }
    return t;
  };
  const nH = Math.floor(band[1] / f0);
  const psdH = Array.from({ length: Math.max(nH, 8) }, (_, i) => (i < harmonics + 1 ? Math.pow(10, -u(0, 20) / 10) : 1e-3));
  const cohH = r() < 0.6 ? null : psdH.map(() => (r() < 0.25 ? u(0.1, 0.5) : u(0.6, 1)));
  const two = speakers !== 'single';
  const plant: Plant = {
    seed,
    band,
    f0,
    harmonics,
    q,
    speakers,
    primaryGain: u(0.3, 1.2),
    primaryDelayS: u(0.02, 0.05),
    psdH,
    cohH,
    secondary: [secPath(), two ? secPath() : null],
    feedback: [r() < 0.9 ? fbPath() : null, two && r() < 0.9 ? fbPath() : null],
    fbPad: [r() < 0.5 ? 64 : Math.round(u(64, 3000)), r() < 0.5 ? 64 : Math.round(u(64, 3000))],
  };
  return { ...plant, ...over };
}

export const psdAt = (pl: Plant, f: number): number => {
  const h = Math.round(f / pl.f0);
  return h >= 1 && h <= pl.psdH.length && Math.abs(f - h * pl.f0) < 6 ? pl.psdH[h - 1] : 1e-3;
};
const cohAt = (pl: Plant, f: number): number => {
  if (pl.cohH === null) return 1;
  const h = Math.round(f / pl.f0);
  return h >= 1 && h <= pl.cohH.length && Math.abs(f - h * pl.f0) < 6 ? pl.cohH[h - 1] : 1;
};

export function primaryAtF(pl: Plant, f: number): { re: number; im: number } {
  const a = -2 * Math.PI * f * pl.primaryDelayS;
  return { re: pl.primaryGain * Math.cos(a), im: pl.primaryGain * Math.sin(a) };
}

export function plantCal(pl: Plant, zone: 0 | 1 | null = null): CalibrationData {
  const cal = createCalibration(fs, pl.band, pl.speakers);
  const freqs = Array.from(calibrationFreqs(fs, pl.band));
  cal.primary = {
    freqs,
    re: freqs.map((f) => primaryAtF(pl, f).re),
    im: freqs.map((f) => primaryAtF(pl, f).im),
    coherence: freqs.map((f) => cohAt(pl, f)),
    tonalF0: null,
    psd: freqs.map((f) => psdAt(pl, f)),
  };
  for (const ch of [0, 1] as const) {
    const s = pl.secondary[ch];
    cal.secondary[ch] = s !== null && (zone === null || zone === ch) ? impulsePath(s, pl.band) : null;
    const fb = pl.feedback[ch];
    cal.feedback[ch] = fb !== null ? impulsePath(fb, pl.band, pl.fbPad[ch]) : null;
  }
  return cal;
}

export function plantParams(pl: Plant): EngineParams {
  const p = clampParams(undefined);
  p.masterDb = 0;
  p.channels[1].enabled = pl.speakers !== 'single';
  p.phaseMode = 'tonal';
  p.tonal = { f0: pl.f0, harmonics: pl.harmonics, q: pl.q };
  return clampParams(p);
}

/**
 * A stored path's response at `freqs`: the DTFT of its IR window (pathResponse) over the nonzero samples only — the
 * plants' paths are a few impulses in thousands of zeros, and the fine grid has thousands of frequencies.
 */
function sparseResponse(path: PathData, freqs: Float64Array): { re: Float64Array; im: Float64Array } {
  const taps: Taps = [];
  path.ir.forEach((v, i) => {
    if (v !== 0) taps.push([i + path.irOffset, v]);
  });
  const re = new Float64Array(freqs.length);
  const im = new Float64Array(freqs.length);
  freqs.forEach((f, k) => {
    for (const [lag, gain] of taps) {
      const a = (-2 * Math.PI * f * lag) / fs;
      re[k] += gain * Math.cos(a);
      im[k] += gain * Math.sin(a);
    }
  });
  return { re, im };
}

/** Residual feedback paths F − F̂ (the neutraliser feedbackFirs(cal, taps)) at `freqs`. */
export function residualFeedback(cal: CalibrationData, freqs: Float64Array, taps = 256): ({ re: Float64Array; im: Float64Array } | null)[] {
  const firs = feedbackFirs(cal, taps);
  return ([0, 1] as const).map((ch) => {
    const path = cal.feedback[ch];
    if (path === null) return null;
    const f = sparseResponse(path, freqs);
    const fir = firs[ch];
    if (fir === null) return { re: f.re, im: f.im };
    const h = irToResponse(fir.taps, fs, freqs);
    const re = new Float64Array(freqs.length);
    const im = new Float64Array(freqs.length);
    for (let k = 0; k < freqs.length; k++) {
      const a = (-2 * Math.PI * freqs[k] * fir.bulkDelay) / fs;
      re[k] = f.re[k] - (h.re[k] * Math.cos(a) - h.im[k] * Math.sin(a));
      im[k] = f.im[k] - (h.re[k] * Math.sin(a) + h.im[k] * Math.cos(a));
    }
    return { re, im };
  });
}

/** |Σ G·(F − F̂)| of `p` at `freqs`, dB. */
export function loopOn(cal: CalibrationData, p: EngineParams, freqs: Float64Array, taps = 256): Float64Array {
  const res = residualFeedback(cal, freqs, taps);
  const out = new Float64Array(freqs.length);
  for (let k = 0; k < freqs.length; k++) {
    let re = 0;
    let im = 0;
    for (const ch of [0, 1] as const) {
      const r = res[ch];
      if (r === null) continue;
      const g = channelResponse(p, fs, freqs[k], ch);
      re += g.re * r.re[k] - g.im * r.im[k];
      im += g.re * r.im[k] + g.im * r.re[k];
    }
    out[k] = 20 * Math.log10(Math.hypot(re, im));
  }
  return out;
}

/** The fine loop-check grid: 0.25 Hz across the band, and ±3 half-bandwidths around each harmonic centre at bw/100. */
export function fineGrid(pl: Plant, p: EngineParams): Float64Array {
  const [lo, hi] = pl.band;
  const xs: number[] = [];
  for (let f = lo; f <= hi; f += 0.25) xs.push(f);
  const q = p.tonal.q;
  for (let h = 1; h <= p.tonal.harmonics; h++) {
    const c = h * p.tonal.f0;
    if (c < lo || c > hi) continue;
    xs.push(c);
    const bw = c / q;
    const step = bw / 100;
    for (let f = Math.max(lo, c - 1.5 * bw); f <= Math.min(hi, c + 1.5 * bw); f += step) xs.push(f);
  }
  return Float64Array.from(xs.sort((a, b) => a - b));
}

export function maxLoop(cal: CalibrationData, p: EngineParams, freqs: Float64Array, taps = 256): { db: number; at: number } {
  const v = loopOn(cal, p, freqs, taps);
  let k = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i;
  return { db: v[k], at: freqs[k] };
}

/**
 * Closed-loop coherent attenuation at f (dB): R = P + Σ G·S/(1 − Σ G·F̃), clamped to [−∞, 80]; S from the
 * stored paths `secondary` (the zone's error mic).
 */
export function closedLoopDb(pl: Plant, cal: CalibrationData, p: EngineParams, f: number, secondary: [Taps | null, Taps | null], taps = 256): number {
  return closedLoopDbAt(pl, cal, p, Float64Array.of(f), secondary, taps)[0];
}

/** closedLoopDb at each of `freqs`, the neutraliser designed and the residual feedback paths evaluated once. */
export function closedLoopDbAt(
  pl: Plant,
  cal: CalibrationData,
  p: EngineParams,
  freqs: Float64Array,
  secondary: [Taps | null, Taps | null],
  taps = 256,
): Float64Array {
  const res = residualFeedback(cal, freqs, taps);
  const resp = ([0, 1] as const).map((ch) => channelResponseAt(p, fs, ch));
  return Float64Array.from(freqs, (f, k) => {
    let ar = 0;
    let ai = 0;
    let dr = 1;
    let di = 0;
    for (const ch of [0, 1] as const) {
      const g = resp[ch](f);
      const s = secondary[ch];
      if (s !== null) {
        let sr = 0;
        let si = 0;
        for (const [lag, gain] of s) {
          sr += gain * Math.cos((-2 * Math.PI * f * lag) / fs);
          si += gain * Math.sin((-2 * Math.PI * f * lag) / fs);
        }
        ar += g.re * sr - g.im * si;
        ai += g.re * si + g.im * sr;
      }
      const r = res[ch];
      if (r !== null) {
        dr -= g.re * r.re[k] - g.im * r.im[k];
        di -= g.re * r.im[k] + g.im * r.re[k];
      }
    }
    const d2 = dr * dr + di * di;
    const P = primaryAtF(pl, f);
    const re = P.re + (ar * dr + ai * di) / d2;
    const im = P.im + (ai * dr - ar * di) / d2;
    return Math.min(80, -20 * Math.log10(Math.hypot(re, im) / Math.hypot(P.re, P.im)));
  });
}

/** The frequencies a tonal solve predicts on: every harmonic h·f0 (h ≤ 64) of the band below 0.45·fs. */
export function harmonicGrid(pl: Plant): Float64Array {
  const xs: number[] = [];
  for (let h = 1; h <= 64 && h * pl.f0 <= Math.min(pl.band[1], 0.45 * fs); h++) if (h * pl.f0 >= pl.band[0]) xs.push(h * pl.f0);
  return Float64Array.from(xs);
}

/**
 * The broadband prediction of the solver's model for `p`, from the plant itself (closedLoopDbAt on harmonicGrid):
 * the error-mic power Gyy = |P|²·Gxx/γ² at each harmonic, of which the coherent part is left at the closed-loop
 * residual, −10·log10(Σ (|R|²·Gxx + (1 − γ²)·Gyy) / Σ Gyy), with the plant's PSD and coherence (γ² within
 * [1e-4, 1 − 1e-6], as the solver takes it).
 */
export function closedLoopTotalDb(pl: Plant, cal: CalibrationData, p: EngineParams, secondary: [Taps | null, Taps | null], taps = 256): number {
  const freqs = harmonicGrid(pl);
  const att = closedLoopDbAt(pl, cal, p, freqs, secondary, taps);
  let num = 0;
  let den = 0;
  freqs.forEach((f, k) => {
    const c = Math.min(1 - 1e-6, Math.max(1e-4, cohAt(pl, f)));
    const w = psdAt(pl, f) * pl.primaryGain * pl.primaryGain;
    num += w * Math.pow(10, -att[k] / 10) + ((1 - c) * w) / c;
    den += w / c;
  });
  return -10 * Math.log10(num / den);
}

/**
 * maxLoop on fineGrid for the params of one plant (same f0, harmonics and Q): the grid and the residual feedback
 * paths on it computed once.
 */
export function fineLoopMax(pl: Plant, cal: CalibrationData, p: EngineParams, taps = 256): (q: EngineParams) => { db: number; at: number } {
  const freqs = fineGrid(pl, p);
  const res = residualFeedback(cal, freqs, taps);
  return (q) => {
    const resp = ([0, 1] as const).map((ch) => channelResponseAt(q, fs, ch));
    let best = { db: -Infinity, at: 0 };
    freqs.forEach((f, k) => {
      let re = 0;
      let im = 0;
      for (const ch of [0, 1] as const) {
        const r = res[ch];
        if (r === null) continue;
        const g = resp[ch](f);
        re += g.re * r.re[k] - g.im * r.im[k];
        im += g.re * r.im[k] + g.im * r.re[k];
      }
      const db = 20 * Math.log10(Math.hypot(re, im));
      if (db > best.db) best = { db, at: f };
    });
    return best;
  };
}

/** The model's own per-harmonic formula (coherence ceiling included) from a coherent attenuation. */
export function withCoherence(pl: Plant, f: number, coherentDb: number): number {
  const c = Math.min(1 - 1e-6, Math.max(1e-4, cohAt(pl, f)));
  const r2 = Math.pow(10, -coherentDb / 10);
  // num/den = (r2·w + (1 − c)·w/c) / (w/c) = c·r2 + 1 − c
  return -10 * Math.log10(c * r2 + 1 - c);
}
