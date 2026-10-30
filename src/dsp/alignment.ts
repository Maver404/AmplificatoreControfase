/**
 * Alignment solver: chooses per-channel delay, gain and polarity that minimise the weighted residual
 * at the zone, including the conditioning response H_cond. Runs in the analysis worker.
 *
 * Model: residual R(f) = P(f) + Σ_ch c_ch·H_cond(f)·S_ch(f)·e^{-jωD_ch}, where c_ch = ±gain is the
 * signed linear gain (c < 0 ⇔ polarity inverted) and D_ch the total tap delay in samples.
 * For a fixed D the optimal c is −Re⟨P, A_D⟩/‖A‖² (A = H·S, weighted inner product), so the delay
 * search maximises |Re Σ w·P·conj(A)·e^{jωD}| — a weighted cross-correlation evaluated exactly.
 */
import type { Complex } from './complex.ts';
import { goldenMax } from './latency-estimator.ts';
import { D_MIN_SAMPLES } from './params.ts';

export interface FreqResponse {
  freqs: Float64Array;
  re: Float64Array;
  im: Float64Array;
}

export type SpeakerConfig = 'single' | 'shared' | 'separate';

export interface AlignmentInput {
  fs: number;
  band: [number, number];
  /** Primary path reference→zone (measured with the error mic, or geometric: pure delay·gain). */
  primary: FreqResponse;
  /** Secondary path final output→zone per channel (null = channel unused). */
  secondary: [FreqResponse | null, FreqResponse | null];
  /** Conditioning response H_cond(f) (no delay/gain/polarity), from response.ts. */
  conditioning: (f: number) => Complex;
  speakers: SpeakerConfig;
  /** Optional weights per frequency of `primary.freqs` (e.g. noise PSD); default flat in band. */
  weights?: Float64Array;
  /** In tonal mode only the harmonics of f0 matter and delay is free modulo 1/f0. */
  tonalF0?: number;
  /** Allowed tap delay above D_MIN, in samples: delaySamples ∈ [D_MIN, D_MIN + maxDelaySamples]. */
  maxDelaySamples: number;
}

export interface ChannelSolution {
  use: boolean;
  /** Total tap delay incl. D_MIN. */
  delaySamples: number;
  /** Total linear gain incl. master. */
  gainLin: number;
  polarityInvert: boolean;
}

export interface AlignmentResult {
  channels: [ChannelSolution, ChannelSolution];
  /** Weighted, in band (harmonics only in tonal mode). */
  predictedAttenuationDb: number;
  /** False if the best broadband delay would be < D_MIN. */
  causal: boolean;
  /** How much earlier the noise must arrive to be causal (0 if causal). */
  requiredExtraLeadSeconds: number;
}

const unused = (): ChannelSolution => ({ use: false, delaySamples: D_MIN_SAMPLES, gainLin: 0, polarityInvert: true });

/** Highest harmonic considered in tonal mode. */
const MAX_TONAL_HARMONICS = 64;
/** Delay scan step: 1/16 of the period of the highest frequency in the grid. */
const SCAN_STEPS_PER_PERIOD = 16;
/** Above this many (steps × frequencies) the coarse scan uses a sub-sampled frequency grid. */
const SCAN_WORK_CAP = 3e7;
/** Ridge regularisation of the joint two-speaker gains, relative to the mean path energy. */
const JOINT_RIDGE = 1e-3;
const JOINT_ITERATIONS = 4;
/** A non-causal optimum only matters if the causal constraint costs more than this. */
const CAUSAL_TOLERANCE_DB = 1;
/** Search range below D_MIN for the unconstrained (causality) optimum: at least 100 ms. */
const MIN_NEGATIVE_SEARCH_S = 0.1;

function validateResponse(fr: FreqResponse, name: string): void {
  const n = fr.freqs.length;
  if (fr.re.length !== n || fr.im.length !== n) throw new RangeError(`${name}: freqs/re/im lengths differ`);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(fr.freqs[i]) || !Number.isFinite(fr.re[i]) || !Number.isFinite(fr.im[i])) {
      throw new RangeError(`${name}: non-finite value at index ${i}`);
    }
    if (i > 0 && !(fr.freqs[i] > fr.freqs[i - 1])) throw new RangeError(`${name}: freqs must be strictly increasing`);
  }
}

const sameGrid = (a: Float64Array, b: Float64Array): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-9 * Math.max(1, Math.abs(a[i]))) return false;
  return true;
};

/**
 * Samples a response at arbitrary frequencies: exact copy on an identical grid, otherwise linear
 * interpolation of magnitude and unwrapped phase (constant magnitude and linear phase extrapolation
 * outside the measured range). Interpolation is only trustworthy when the grid is fine enough that
 * the phase moves less than π between points (i.e. spacing ≪ 1/delay).
 */
export function sampleResponse(fr: FreqResponse, at: Float64Array): { re: Float64Array; im: Float64Array } {
  const n = fr.freqs.length;
  const re = new Float64Array(at.length);
  const im = new Float64Array(at.length);
  if (n === 0) return { re, im };
  if (sameGrid(fr.freqs, at)) {
    re.set(fr.re);
    im.set(fr.im);
    return { re, im };
  }
  const mag = new Float64Array(n);
  const ph = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    mag[i] = Math.hypot(fr.re[i], fr.im[i]);
    const p = Math.atan2(fr.im[i], fr.re[i]);
    if (i === 0) ph[i] = p;
    else {
      // Unwrap: step to the representative of p closest to the previous phase.
      let d = p - ph[i - 1];
      d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      ph[i] = ph[i - 1] + d;
    }
  }
  for (let j = 0; j < at.length; j++) {
    const f = at[j];
    let m: number;
    let p: number;
    if (n === 1) {
      m = mag[0];
      p = ph[0];
    } else {
      // Binary search for the segment [k, k+1] containing f (clamped to the end segments).
      let lo = 0;
      let hi = n - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (fr.freqs[mid] <= f) lo = mid;
        else hi = mid;
      }
      const a = (f - fr.freqs[lo]) / (fr.freqs[hi] - fr.freqs[lo]);
      const ac = Math.min(1, Math.max(0, a));
      m = mag[lo] + ac * (mag[hi] - mag[lo]);
      p = ph[lo] + a * (ph[hi] - ph[lo]);
    }
    re[j] = m * Math.cos(p);
    im[j] = m * Math.sin(p);
  }
  return { re, im };
}

/** Linear interpolation of a real per-frequency array (weights). */
function sampleReal(freqs: Float64Array, v: Float64Array, f: number): number {
  const n = freqs.length;
  if (f <= freqs[0]) return v[0];
  if (f >= freqs[n - 1]) return v[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (freqs[mid] <= f) lo = mid;
    else hi = mid;
  }
  const a = (f - freqs[lo]) / (freqs[hi] - freqs[lo]);
  return v[lo] + a * (v[hi] - v[lo]);
}

interface Grid {
  f: Float64Array;
  omega: Float64Array;
  w: Float64Array;
  pRe: Float64Array;
  pIm: Float64Array;
  pEnergy: number;
  /** Tonal period in samples, or 0 in broadband mode. */
  period: number;
  /** Highest angular frequency of the grid (rad/sample). */
  omegaMax: number;
}

interface Path {
  re: Float64Array;
  im: Float64Array;
  energy: number;
}

function buildGrid(input: AlignmentInput): Grid | null {
  const { fs, primary } = input;
  if (!(fs > 0)) throw new RangeError('solveAlignment: fs must be positive');
  validateResponse(primary, 'primary');
  if (input.weights && input.weights.length !== primary.freqs.length) {
    throw new RangeError('solveAlignment: weights must have one value per primary frequency');
  }
  const lo = Math.max(0, Math.min(input.band[0], input.band[1]));
  const hi = Math.min(fs / 2, Math.max(input.band[0], input.band[1]));
  const weights = input.weights;
  const clean = (w: number): number => (Number.isFinite(w) && w > 0 ? w : 0);
  const f0 = input.tonalF0;
  let freqs: number[] = [];
  let ws: number[] = [];
  let pRe: Float64Array;
  let pIm: Float64Array;
  let period = 0;
  if (f0 !== undefined && Number.isFinite(f0) && f0 > 0) {
    period = fs / f0;
    for (let h = 1; h <= MAX_TONAL_HARMONICS && h * f0 <= Math.min(hi, 0.45 * fs); h++) {
      if (h * f0 < lo || primary.freqs.length === 0) continue;
      freqs.push(h * f0);
      ws.push(weights ? clean(sampleReal(primary.freqs, weights, h * f0)) : 1);
    }
    const sampled = sampleResponse(primary, Float64Array.from(freqs));
    pRe = sampled.re;
    pIm = sampled.im;
  } else {
    const idx: number[] = [];
    for (let i = 0; i < primary.freqs.length; i++) {
      const f = primary.freqs[i];
      if (f >= lo && f <= hi && f > 0) idx.push(i);
    }
    freqs = idx.map((i) => primary.freqs[i]);
    ws = idx.map((i) => (weights ? clean(weights[i]) : 1));
    pRe = Float64Array.from(idx, (i) => primary.re[i]);
    pIm = Float64Array.from(idx, (i) => primary.im[i]);
  }
  if (freqs.length === 0) return null;
  const f = Float64Array.from(freqs);
  const w = Float64Array.from(ws);
  let pEnergy = 0;
  for (let k = 0; k < f.length; k++) pEnergy += w[k] * (pRe[k] * pRe[k] + pIm[k] * pIm[k]);
  if (!(pEnergy > 0)) return null;
  const omega = Float64Array.from(f, (x) => (2 * Math.PI * x) / fs);
  let omegaMax = 0;
  for (let k = 0; k < omega.length; k++) omegaMax = Math.max(omegaMax, omega[k]);
  return { f, omega, w, pRe, pIm, pEnergy, period, omegaMax };
}

/** H_cond·S_ch on the grid, or null when the channel has no usable secondary path. */
function buildPath(input: AlignmentInput, grid: Grid, ch: 0 | 1): Path | null {
  const s = input.secondary[ch];
  if (!s) return null;
  validateResponse(s, `secondary[${ch}]`);
  const sampled = sampleResponse(s, grid.f);
  const re = new Float64Array(grid.f.length);
  const im = new Float64Array(grid.f.length);
  let energy = 0;
  for (let k = 0; k < grid.f.length; k++) {
    const h = input.conditioning(grid.f[k]);
    const hr = Number.isFinite(h.re) ? h.re : 0;
    const hi = Number.isFinite(h.im) ? h.im : 0;
    re[k] = hr * sampled.re[k] - hi * sampled.im[k];
    im[k] = hr * sampled.im[k] + hi * sampled.re[k];
    energy += grid.w[k] * (re[k] * re[k] + im[k] * im[k]);
  }
  return energy > 1e-24 * grid.pEnergy ? { re, im, energy } : null;
}

/** b = w·target·conj(A): q(D) = Re Σ b·e^{jωD} is minus the optimal-gain numerator. */
function correlationWeights(grid: Grid, path: Path, tRe: Float64Array, tIm: Float64Array): { re: Float64Array; im: Float64Array } {
  const n = grid.f.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    re[k] = grid.w[k] * (tRe[k] * path.re[k] + tIm[k] * path.im[k]);
    im[k] = grid.w[k] * (tIm[k] * path.re[k] - tRe[k] * path.im[k]);
  }
  return { re, im };
}

function qAt(b: { re: Float64Array; im: Float64Array }, omega: Float64Array, d: number): number {
  let q = 0;
  for (let k = 0; k < omega.length; k++) {
    const a = omega[k] * d;
    q += b.re[k] * Math.cos(a) - b.im[k] * Math.sin(a);
  }
  return q;
}

/** Fixed-size list of the largest local maxima of |q| found by the scan. */
class TopK {
  readonly d: number[] = [];
  readonly v: number[] = [];
  private readonly k: number;
  constructor(k: number) {
    this.k = k;
  }
  add(d: number, v: number): void {
    if (this.v.length === this.k && v <= this.v[this.v.length - 1]) return;
    let i = this.v.length;
    while (i > 0 && this.v[i - 1] < v) i--;
    this.d.splice(i, 0, d);
    this.v.splice(i, 0, v);
    if (this.v.length > this.k) {
      this.d.pop();
      this.v.pop();
    }
  }
}

interface DelaySearch {
  /** Best delay within [lo, hi] (the allowed range). */
  d: number;
  q: number;
  /** Best delay over the extended range [freeLo, hi]. */
  dFree: number;
  qFree: number;
}

/**
 * Global maximisation of |q(D)| over [freeLo, hi], reporting the best overall and the best with
 * D ≥ lo. Exact phasor scan with a step of 1/16 period of the highest frequency, then golden-section
 * refinement of the strongest local maxima on the full grid.
 */
function searchDelay(
  b: { re: Float64Array; im: Float64Array },
  omega: Float64Array,
  lo: number,
  hi: number,
  freeLo: number,
): DelaySearch {
  const n = omega.length;
  let wMax = 0;
  for (let k = 0; k < n; k++) wMax = Math.max(wMax, omega[k]);
  const span = hi - freeLo;
  let step = wMax > 0 ? (2 * Math.PI) / wMax / SCAN_STEPS_PER_PERIOD : span;
  step = Math.max(1e-3, Math.min(step, span / 8 || 1));
  const steps = Math.max(1, Math.ceil(span / step));
  step = span / steps || 1;
  const stride = Math.max(1, Math.ceil((steps * n) / SCAN_WORK_CAP));
  const kk = Math.max(16, 4 * stride);
  const topFree = new TopK(kk);
  const topIn = new TopK(kk);
  // Phasors e^{jω(freeLo + i·step)} for the scanned subset of frequencies.
  const m = Math.ceil(n / stride);
  const zr = new Float64Array(m);
  const zi = new Float64Array(m);
  const rr = new Float64Array(m);
  const ri = new Float64Array(m);
  const br = new Float64Array(m);
  const bi = new Float64Array(m);
  for (let j = 0, k = 0; k < n; j++, k += stride) {
    zr[j] = Math.cos(omega[k] * freeLo);
    zi[j] = Math.sin(omega[k] * freeLo);
    rr[j] = Math.cos(omega[k] * step);
    ri[j] = Math.sin(omega[k] * step);
    br[j] = b.re[k];
    bi[j] = b.im[k];
  }
  let prev2 = -1;
  let prev1 = -1;
  let prevD = freeLo;
  for (let i = 0; i <= steps; i++) {
    const d = freeLo + i * step;
    let q = 0;
    for (let j = 0; j < m; j++) {
      q += br[j] * zr[j] - bi[j] * zi[j];
      const t = zr[j] * rr[j] - zi[j] * ri[j];
      zi[j] = zr[j] * ri[j] + zi[j] * rr[j];
      zr[j] = t;
    }
    if ((i & 1023) === 1023) {
      for (let j = 0; j < m; j++) {
        const g = 1 / Math.hypot(zr[j], zi[j]);
        zr[j] *= g;
        zi[j] *= g;
      }
    }
    const a = Math.abs(q);
    // Local maximum at the previous point (the first point counts against a virtual −1).
    if (i >= 1 && prev1 >= prev2 && prev1 >= a) {
      topFree.add(prevD, prev1);
      if (prevD >= lo) topIn.add(prevD, prev1);
    }
    prev2 = prev1;
    prev1 = a;
    prevD = d;
  }
  if (prev1 >= prev2) {
    topFree.add(prevD, prev1);
    topIn.add(prevD, prev1);
  }
  // The allowed range starts inside the scan: its edge is a candidate too (optimum may be clamped).
  topIn.add(lo, Math.abs(qAt(b, omega, lo)));

  const abs = (d: number): number => Math.abs(qAt(b, omega, d));
  const refine = (cands: TopK, a: number): { d: number; q: number } => {
    let best = { d: Math.max(a, Math.min(hi, cands.d[0] ?? a)), q: -1 };
    for (const c of cands.d) {
      const d = goldenMax(abs, Math.max(a, c - step), Math.min(hi, c + step), 40);
      const v = abs(d);
      if (v > best.q) best = { d, q: v };
    }
    return best;
  };
  const inRange = refine(topIn, lo);
  const free = refine(topFree, freeLo);
  // The free optimum can never be worse than the constrained one.
  return free.q >= inRange.q
    ? { d: inRange.d, q: inRange.q, dFree: free.d, qFree: free.q }
    : { d: inRange.d, q: inRange.q, dFree: inRange.d, qFree: inRange.q };
}

/** Residual energy ratio Σw|P + Σ c·A·e^{-jωD}|² / Σw|P|². */
function residualRatio(grid: Grid, terms: { path: Path; c: number; d: number }[]): number {
  let e = 0;
  for (let k = 0; k < grid.f.length; k++) {
    let rr = grid.pRe[k];
    let ri = grid.pIm[k];
    for (const t of terms) {
      const a = -grid.omega[k] * t.d;
      const cr = Math.cos(a);
      const ci = Math.sin(a);
      rr += t.c * (t.path.re[k] * cr - t.path.im[k] * ci);
      ri += t.c * (t.path.re[k] * ci + t.path.im[k] * cr);
    }
    e += grid.w[k] * (rr * rr + ri * ri);
  }
  return e / grid.pEnergy;
}

const toDb = (ratio: number): number => 10 * Math.log10(Math.max(ratio, 1e-30));

interface SingleSolution {
  d: number;
  c: number;
  causal: boolean;
  extraLeadSamples: number;
}

function delayRange(input: AlignmentInput, grid: Grid): { lo: number; hi: number; freeLo: number } {
  const lo = D_MIN_SAMPLES;
  const maxD = Math.max(0, Number.isFinite(input.maxDelaySamples) ? input.maxDelaySamples : 0);
  if (grid.period > 0) {
    // Tonal: any delay is equivalent to one within a period above D_MIN.
    const hi = lo + Math.min(grid.period, Math.max(maxD, 1e-9));
    return { lo, hi, freeLo: lo };
  }
  const neg = Math.max(maxD, MIN_NEGATIVE_SEARCH_S * input.fs);
  return { lo, hi: lo + maxD, freeLo: lo - neg };
}

function solveSingle(input: AlignmentInput, grid: Grid, path: Path, tRe: Float64Array, tIm: Float64Array): SingleSolution {
  const b = correlationWeights(grid, path, tRe, tIm);
  const r = delayRange(input, grid);
  const s = searchDelay(b, grid.omega, r.lo, r.hi, r.freeLo);
  const c = -qAt(b, grid.omega, s.d) / path.energy;
  let causal = true;
  let extra = 0;
  if (grid.period === 0 && s.dFree < r.lo) {
    // Compare the attenuations reachable with and without the causality constraint.
    const cFree = -qAt(b, grid.omega, s.dFree) / path.energy;
    const target = { pRe: tRe, pIm: tIm };
    const g = { ...grid, ...target, pEnergy: energyOf(grid, tRe, tIm) };
    const attIn = -toDb(residualRatio(g, [{ path, c, d: s.d }]));
    const attFree = -toDb(residualRatio(g, [{ path, c: cFree, d: s.dFree }]));
    if (attFree - attIn > CAUSAL_TOLERANCE_DB) {
      causal = false;
      extra = r.lo - s.dFree;
    }
  }
  return { d: s.d, c, causal, extraLeadSamples: extra };
}

function energyOf(grid: Grid, re: Float64Array, im: Float64Array): number {
  let e = 0;
  for (let k = 0; k < grid.f.length; k++) e += grid.w[k] * (re[k] * re[k] + im[k] * im[k]);
  return e > 0 ? e : 1e-300;
}

/** A·e^{-jωD} on the grid. */
function shifted(grid: Grid, path: Path, d: number): { re: Float64Array; im: Float64Array } {
  const n = grid.f.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = -grid.omega[k] * d;
    const cr = Math.cos(a);
    const ci = Math.sin(a);
    re[k] = path.re[k] * cr - path.im[k] * ci;
    im[k] = path.re[k] * ci + path.im[k] * cr;
  }
  return { re, im };
}

/**
 * Joint real gains of two speakers for fixed delays: ridge-regularised least squares. With two
 * identical paths the problem is singular and the ridge gives the minimum-norm solution (equal
 * split); with nearly identical paths it prevents huge opposite gains that would only cancel at
 * the measurement point.
 */
function jointGains(grid: Grid, paths: [Path, Path], d: [number, number]): [number, number] {
  const a0 = shifted(grid, paths[0], d[0]);
  const a1 = shifted(grid, paths[1], d[1]);
  let m00 = 0;
  let m11 = 0;
  let m01 = 0;
  let v0 = 0;
  let v1 = 0;
  for (let k = 0; k < grid.f.length; k++) {
    const w = grid.w[k];
    m00 += w * (a0.re[k] * a0.re[k] + a0.im[k] * a0.im[k]);
    m11 += w * (a1.re[k] * a1.re[k] + a1.im[k] * a1.im[k]);
    m01 += w * (a0.re[k] * a1.re[k] + a0.im[k] * a1.im[k]);
    v0 -= w * (grid.pRe[k] * a0.re[k] + grid.pIm[k] * a0.im[k]);
    v1 -= w * (grid.pRe[k] * a1.re[k] + grid.pIm[k] * a1.im[k]);
  }
  const lambda = (JOINT_RIDGE * (m00 + m11)) / 2;
  const a = m00 + lambda;
  const c = m11 + lambda;
  const det = a * c - m01 * m01;
  return [(v0 * c - v1 * m01) / det, (v1 * a - v0 * m01) / det];
}

const solution = (d: number, c: number): ChannelSolution => ({
  use: true,
  delaySamples: d,
  gainLin: Math.abs(c),
  polarityInvert: !(c > 0),
});

/**
 * Minimises the weighted residual |P + Σ c_ch·H_cond·S_ch·e^{-jωD_ch}|² in band (c_ch = ±gain).
 *  'single': only channel 0. 'separate': each channel solved against P independently (the caller
 *  measures each zone separately); predictedAttenuationDb is the worst of the two.
 *  'shared': joint solution — individual delays, then alternating refinement of each channel against
 *  the residual of the other, with ridge-regularised joint gains (equal split for equal speakers).
 * In tonal mode (tonalF0) the residual is evaluated at the harmonics only and each delay lies in
 * [D_MIN, D_MIN + 1/f0): whole periods are free, so the result is always causal.
 */
export function solveAlignment(input: AlignmentInput): AlignmentResult {
  const noop: AlignmentResult = {
    channels: [unused(), unused()],
    predictedAttenuationDb: 0,
    causal: true,
    requiredExtraLeadSeconds: 0,
  };
  const grid = buildGrid(input);
  if (!grid) return noop;
  const chans: (0 | 1)[] = input.speakers === 'single' ? [0] : [0, 1];
  const paths: [Path | null, Path | null] = [null, null];
  for (const ch of chans) paths[ch] = buildPath(input, grid, ch);
  const usable = chans.filter((ch) => paths[ch] !== null);
  if (usable.length === 0) return noop;

  const singles: [SingleSolution | null, SingleSolution | null] = [null, null];
  for (const ch of usable) singles[ch] = solveSingle(input, grid, paths[ch]!, grid.pRe, grid.pIm);
  const causal = usable.every((ch) => singles[ch]!.causal);
  const extra = Math.max(0, ...usable.map((ch) => singles[ch]!.extraLeadSamples));
  const result: AlignmentResult = {
    channels: [unused(), unused()],
    predictedAttenuationDb: 0,
    causal,
    requiredExtraLeadSeconds: causal ? 0 : extra / input.fs,
  };

  if (input.speakers === 'shared' && usable.length === 2) {
    const p: [Path, Path] = [paths[0]!, paths[1]!];
    const d: [number, number] = [singles[0]!.d, singles[1]!.d];
    let c = jointGains(grid, p, d);
    const r = delayRange(input, grid);
    for (let it = 0; it < JOINT_ITERATIONS; it++) {
      for (const ch of [0, 1] as const) {
        const o = 1 - ch;
        const ao = shifted(grid, p[o], d[o]);
        const tRe = Float64Array.from(grid.pRe, (v, k) => v + c[o] * ao.re[k]);
        const tIm = Float64Array.from(grid.pIm, (v, k) => v + c[o] * ao.im[k]);
        const b = correlationWeights(grid, p[ch], tRe, tIm);
        const span = (2 * Math.PI) / grid.omegaMax / SCAN_STEPS_PER_PERIOD;
        const abs = (x: number): number => Math.abs(qAt(b, grid.omega, x));
        d[ch] = goldenMax(abs, Math.max(r.lo, d[ch] - 2 * span), Math.min(r.hi, d[ch] + 2 * span), 40);
        c[ch] = -qAt(b, grid.omega, d[ch]) / p[ch].energy;
      }
      c = jointGains(grid, p, d);
    }
    result.channels = [solution(d[0], c[0]), solution(d[1], c[1])];
    result.predictedAttenuationDb = -toDb(residualRatio(grid, [
      { path: p[0], c: c[0], d: d[0] },
      { path: p[1], c: c[1], d: d[1] },
    ]));
    return result;
  }

  let worst = Infinity;
  for (const ch of usable) {
    const s = singles[ch]!;
    result.channels[ch] = solution(s.d, s.c);
    worst = Math.min(worst, -toDb(residualRatio(grid, [{ path: paths[ch]!, c: s.c, d: s.d }])));
  }
  result.predictedAttenuationDb = worst;
  return result;
}

/** Bounds of |T| of harmonicFactors (the trims' gain range as linear amplitudes). */
export interface FactorBounds {
  min: number;
  max: number;
}

/** A contribution this much below the residual reads as "the channel does not reach this harmonic". */
const FACTOR_MIN_REACH = 1e-9;
/** Relative tolerance of the bounds: a factor centred exactly on a bound (rounding) is not a clamp. */
const FACTOR_BOUND_TOL = 1e-9;

/**
 * Tonal mode, one harmonic: the complex factors T_ch, multiplying channel ch's current contribution b_ch
 * (its channel response × secondary path at that frequency), that null the residual p + Σ T_ch·b_ch.
 *  - 'single' / 'separate', or one usable channel: each channel alone against p, T = −p/b (separate zones:
 *    each speaker cancels the noise of its own zone);
 *  - 'shared' with both: the minimum change Σ|T_ch − 1|² that nulls the residual, T_ch = 1 + δ_ch with
 *    δ_ch = −r·conj(b_ch)/Σ|b|², r = p + Σ b: each speaker corrects in proportion to its own contribution, so
 *    a harmonic the broadband split already cancels keeps it.
 * A factor whose magnitude falls outside `bounds` is clamped (phase kept, capped[ch] = true) and, when both
 * channels share the zone, the other one is re-solved against what remains (then clamped in turn). Entries for
 * channels that are unused (null b) or do not reach the harmonic (b ≈ 0) are null.
 */
export function harmonicFactors(
  p: Complex,
  b: [Complex | null, Complex | null],
  speakers: SpeakerConfig,
  bounds: FactorBounds,
): { t: [Complex | null, Complex | null]; capped: [boolean, boolean] } {
  const t: [Complex | null, Complex | null] = [null, null];
  const capped: [boolean, boolean] = [false, false];
  const scale = Math.max(Math.hypot(p.re, p.im), Math.hypot(b[0]?.re ?? 0, b[0]?.im ?? 0), Math.hypot(b[1]?.re ?? 0, b[1]?.im ?? 0));
  const usable = (v: Complex | null): v is Complex =>
    v !== null && Number.isFinite(v.re) && Number.isFinite(v.im) && Math.hypot(v.re, v.im) > FACTOR_MIN_REACH * scale;
  const chans = ([0, 1] as const).filter((ch) => usable(b[ch]) && (speakers !== 'single' || ch === 0));
  if (chans.length === 0 || !Number.isFinite(p.re) || !Number.isFinite(p.im)) return { t, capped };
  /** −(target)/v: the factor that makes T·v cancel `target`. */
  const against = (re: number, im: number, v: Complex): Complex => {
    const d = v.re * v.re + v.im * v.im;
    return { re: -(re * v.re + im * v.im) / d, im: -(im * v.re - re * v.im) / d };
  };
  const outside = (v: Complex): boolean => {
    const m = Math.hypot(v.re, v.im);
    return m > bounds.max * (1 + FACTOR_BOUND_TOL) || m < bounds.min * (1 - FACTOR_BOUND_TOL);
  };
  const clampMag = (ch: 0 | 1): void => {
    const v = t[ch]!;
    if (!outside(v)) return;
    const m = Math.hypot(v.re, v.im);
    const c = Math.min(bounds.max, Math.max(bounds.min, m));
    capped[ch] = true;
    t[ch] = m > 0 ? { re: (v.re * c) / m, im: (v.im * c) / m } : { re: c, im: 0 };
  };
  if (speakers !== 'shared' || chans.length === 1) {
    for (const ch of chans) {
      t[ch] = against(p.re, p.im, b[ch]!);
      clampMag(ch);
    }
    return { t, capped };
  }
  const b0 = b[0]!;
  const b1 = b[1]!;
  const rRe = p.re + b0.re + b1.re;
  const rIm = p.im + b0.im + b1.im;
  const e = b0.re * b0.re + b0.im * b0.im + b1.re * b1.re + b1.im * b1.im;
  for (const ch of [0, 1] as const) {
    const v = b[ch]!;
    // δ = −r·conj(v)/e
    t[ch] = { re: 1 - (rRe * v.re + rIm * v.im) / e, im: -(rIm * v.re - rRe * v.im) / e };
  }
  const o0 = outside(t[0]!);
  const o1 = outside(t[1]!);
  if (o0 !== o1) {
    // Clamp the one outside, re-solve the other against p plus the clamped contribution.
    const c: 0 | 1 = o0 ? 0 : 1;
    const o: 0 | 1 = c === 0 ? 1 : 0;
    clampMag(c);
    const tc = t[c]!;
    const bc = b[c]!;
    t[o] = against(p.re + tc.re * bc.re - tc.im * bc.im, p.im + tc.re * bc.im + tc.im * bc.re, b[o]!);
  }
  clampMag(0);
  clampMag(1);
  return { t, capped };
}

/**
 * Weighted residual in dB (negative = attenuation) for given channel settings, with the same grid
 * and speaker semantics as solveAlignment ('separate': worst channel alone against P).
 */
export function residualDb(input: AlignmentInput, channels: [ChannelSolution, ChannelSolution]): number {
  const grid = buildGrid(input);
  if (!grid) return 0;
  const chans: (0 | 1)[] = input.speakers === 'single' ? [0] : [0, 1];
  const terms: { path: Path; c: number; d: number }[] = [];
  for (const ch of chans) {
    const s = channels[ch];
    if (!s.use) continue;
    const path = buildPath(input, grid, ch);
    if (!path) continue;
    const g = Number.isFinite(s.gainLin) ? s.gainLin : 0;
    terms.push({ path, c: s.polarityInvert ? -g : g, d: s.delaySamples });
  }
  if (input.speakers === 'separate' && terms.length > 0) {
    return Math.max(...terms.map((t) => toDb(residualRatio(grid, [t]))));
  }
  return toDb(residualRatio(grid, terms));
}

/**
 * Open-loop gain |Σ_ch G_ch(f)·F_ch(f)| in dB across band (G = channelResponse), for the howl margin.
 * Evaluated on the frequency grid of the first available feedback response (the other one is
 * sampled onto it). Without feedback data: maxDb = −Infinity, atHz = 0.
 */
export function loopGainDb(
  channelResp: [(f: number) => Complex, (f: number) => Complex],
  feedback: [FreqResponse | null, FreqResponse | null],
  band: [number, number],
): { maxDb: number; atHz: number } {
  const ref = feedback[0] ?? feedback[1];
  if (!ref) return { maxDb: -Infinity, atHz: 0 };
  const lo = Math.min(band[0], band[1]);
  const hi = Math.max(band[0], band[1]);
  const freqs = Float64Array.from(Array.from(ref.freqs).filter((f) => f >= lo && f <= hi));
  if (freqs.length === 0) return { maxDb: -Infinity, atHz: 0 };
  const sampled = feedback.map((fb, i) => {
    if (!fb) return null;
    validateResponse(fb, `feedback[${i}]`);
    return sampleResponse(fb, freqs);
  });
  let best = -1;
  let atHz = freqs[0];
  for (let k = 0; k < freqs.length; k++) {
    let re = 0;
    let im = 0;
    for (const ch of [0, 1] as const) {
      const s = sampled[ch];
      if (!s) continue;
      const g = channelResp[ch](freqs[k]);
      if (!Number.isFinite(g.re) || !Number.isFinite(g.im)) continue;
      re += g.re * s.re[k] - g.im * s.im[k];
      im += g.re * s.im[k] + g.im * s.re[k];
    }
    const m = Math.hypot(re, im);
    if (m > best) {
      best = m;
      atHz = freqs[k];
    }
  }
  return { maxDb: best > 0 ? 20 * Math.log10(best) : -Infinity, atHz };
}
