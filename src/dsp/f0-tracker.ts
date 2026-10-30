/**
 * Precise fundamental-frequency estimation for the tonal mode (analysis worker; allocations allowed
 * in estimateF0, F0Tracker.push is allocation-free).
 */
import { nextPow2, sharedFFT } from './fft.ts';
import { goldenMax } from './latency-estimator.ts';

export interface F0Estimate {
  f0: number;
  /** Fraction (0..1) of the signal power carried by the detected harmonics of f0. */
  confidence: number;
  /** Strongest tonal peak vs the median of the surrounding spectrum (±1/6 octave), dB. */
  dominanceDb: number;
}

/** A tone must stand this far above the local median spectrum to be detected. */
export const F0_DETECT_DB = 15;
/** Harmonics (and a candidate fundamental below the strongest peak) must stand this far above it. */
const HARMONIC_DB = 10;
const MAX_HARMONICS = 8;
/** Fraction of the harmonics 1..h that must be present to accept fp/h as the fundamental. */
const HARMONIC_COVERAGE = 0.75;

interface Peak {
  freq: number;
  power: number;
  excessDb: number;
}

/** Quickselect median (destroys the order of `a`). */
function median(a: Float64Array, n: number): number {
  let lo = 0;
  let hi = n - 1;
  const k = n >> 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k];
}

/**
 * Sub-bin f0 estimate (long windowed FFT + interpolation and/or phase-advance refinement, target
 * ~0.01 % on a clean stable tone of ≥ 1 s), searching fmin..fmax, harmonic-aware (prefers the
 * fundamental over its 2nd harmonic). Returns null if no tonal component.
 *
 * Steps: Hann window, 2× zero-padded FFT; local noise floor = median power over ±1/6 octave; the
 * strongest peak in [fmin, fmax] ≥ 15 dB above it is the dominant tone fp; fp/h is preferred as the
 * fundamental for the largest h ≤ 8 such that fp/h and ≥ 75 % of its harmonics up to fp are present;
 * f0 is then refined by maximising Σ_h |X(h·f)|² (harmonic maximum likelihood) with golden-section
 * search on the exact DTFT.
 */
export function estimateF0(x: Float32Array | Float64Array, fs: number, fmin: number, fmax: number): F0Estimate | null {
  if (!(fs > 0) || !(fmin > 0) || !(fmax > fmin)) return null;
  const fTop = Math.min(fmax, 0.45 * fs);
  if (!(fTop > fmin)) return null;
  const len = x.length;
  if (len < Math.max(256, Math.ceil((4 * fs) / fmin))) return null;

  let mean = 0;
  for (let i = 0; i < len; i++) {
    if (!Number.isFinite(x[i])) return null;
    mean += x[i];
  }
  mean /= len;
  let power = 0;
  const xw = new Float64Array(len);
  let sumW = 0;
  for (let i = 0; i < len; i++) {
    const v = x[i] - mean;
    power += v * v;
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (len - 1));
    xw[i] = v * w;
    sumW += w;
  }
  power /= len;
  if (!(power > 1e-20)) return null;

  const n = 2 * nextPow2(len);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(xw);
  sharedFFT(n).forward(re, im);
  const half = n / 2;
  const p = new Float64Array(half + 1);
  let pMax = 0;
  for (let k = 0; k <= half; k++) {
    p[k] = re[k] * re[k] + im[k] * im[k];
    if (p[k] > pMax) pMax = p[k];
  }
  const binHz = fs / n;

  // Local noise floor: median over ±1/6 octave (at least ±16 bins), sampled every 1/12 octave.
  const floorLo = Math.max(2 * binHz, fmin * 0.8);
  const floorHi = Math.min(0.45 * fs, MAX_HARMONICS * fTop * 1.05);
  const logF: number[] = [];
  const logM: number[] = [];
  const scratch = new Float64Array(half + 1);
  for (let f = floorLo; f <= floorHi * Math.pow(2, 1 / 12); f *= Math.pow(2, 1 / 12)) {
    const k0 = Math.max(1, Math.min(Math.floor((f * Math.pow(2, -1 / 6)) / binHz), Math.round(f / binHz) - 16));
    const k1 = Math.min(half, Math.max(Math.ceil((f * Math.pow(2, 1 / 6)) / binHz), Math.round(f / binHz) + 16));
    let m = 0;
    for (let k = k0; k <= k1; k++) scratch[m++] = p[k];
    logF.push(Math.log(f));
    logM.push(Math.log(Math.max(median(scratch, m), pMax * 1e-20, 1e-300)));
  }
  const floorAt = (f: number): number => {
    const lf = Math.log(f);
    if (lf <= logF[0]) return Math.exp(logM[0]);
    for (let j = 1; j < logF.length; j++) {
      if (lf <= logF[j]) {
        const a = (lf - logF[j - 1]) / (logF[j] - logF[j - 1]);
        return Math.exp(logM[j - 1] + a * (logM[j] - logM[j - 1]));
      }
    }
    return Math.exp(logM[logM.length - 1]);
  };

  /** Parabolic (log-power) refinement of the local maximum at bin k. */
  const refine = (k: number): Peak => {
    const a = Math.log(Math.max(p[k - 1], 1e-300));
    const b = Math.log(Math.max(p[k], 1e-300));
    const c = Math.log(Math.max(p[k + 1], 1e-300));
    const d = a - 2 * b + c;
    const delta = d < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / d)) : 0;
    const freq = (k + delta) * binHz;
    const pw = Math.exp(b - 0.25 * (a - c) * delta);
    return { freq, power: pw, excessDb: 10 * Math.log10(pw / floorAt(freq)) };
  };
  const isLocalMax = (k: number): boolean => k >= 1 && k < half && p[k] > p[k - 1] && p[k] >= p[k + 1];

  /** Strongest significant local maximum within ±tol Hz of `target`. */
  const peakNear = (target: number, tol: number): Peak | null => {
    const k0 = Math.max(1, Math.ceil((target - tol) / binHz));
    const k1 = Math.min(half - 1, Math.floor((target + tol) / binHz));
    let best = -1;
    for (let k = k0; k <= k1; k++) if (isLocalMax(k) && (best < 0 || p[k] > p[best])) best = k;
    if (best < 0) return null;
    const pk = refine(best);
    return pk.excessDb >= HARMONIC_DB ? pk : null;
  };
  const tolAt = (f: number): number => Math.max(2 * binHz, 0.003 * f);

  // Dominant tone in [fmin, fTop].
  let fp: Peak | null = null;
  for (let k = Math.max(1, Math.ceil(fmin / binHz)); k <= Math.min(half - 1, Math.floor(fTop / binHz)); k++) {
    if (!isLocalMax(k) || (fp && p[k] <= fp.power)) continue;
    const pk = refine(k);
    if (pk.excessDb >= F0_DETECT_DB && pk.freq >= fmin && pk.freq <= fTop && (!fp || pk.power > fp.power)) fp = pk;
  }
  if (!fp) return null;

  // Prefer the lowest consistent fundamental below the dominant peak.
  let f0c = fp.freq;
  for (let h = MAX_HARMONICS; h >= 2; h--) {
    const c = fp.freq / h;
    if (c < fmin) continue;
    if (!peakNear(c, tolAt(c))) continue;
    let present = 0;
    for (let m = 1; m <= h; m++) if (m === h || peakNear(m * c, tolAt(m * c))) present++;
    if (present >= Math.ceil(HARMONIC_COVERAGE * h)) {
      f0c = c;
      break;
    }
  }

  // Harmonics present for the refinement; the fundamental is always included.
  const harmonics: number[] = [];
  let strongest: Peak | null = null;
  let mStrong = 1;
  for (let m = 1; m <= MAX_HARMONICS && m * f0c < 0.45 * fs; m++) {
    const pk = peakNear(m * f0c, tolAt(m * f0c));
    if (!pk && m > 1) continue;
    harmonics.push(m);
    const pw = pk ? pk.power : 0;
    if (pk && (!strongest || pw > strongest.power)) {
      strongest = pk;
      mStrong = m;
    }
  }
  const start = strongest ? strongest.freq / mStrong : f0c;
  const hMax = harmonics[harmonics.length - 1];
  const span = Math.min(0.5, 2 / hMax) * binHz;

  // Exact windowed DTFT power at frequency f.
  const dtftPow = (f: number): number => {
    const w = (2 * Math.PI * f) / fs;
    const cr = Math.cos(w);
    const ci = Math.sin(w);
    let zr = 1;
    let zi = 0;
    let ar = 0;
    let ai = 0;
    for (let i = 0; i < len; i++) {
      ar += xw[i] * zr;
      ai -= xw[i] * zi;
      const t = zr * cr - zi * ci;
      zi = zr * ci + zi * cr;
      zr = t;
    }
    return ar * ar + ai * ai;
  };
  const objective = (f: number): number => {
    let s = 0;
    for (const m of harmonics) s += dtftPow(m * f);
    return s;
  };
  const f0 = goldenMax(objective, start - span, start + span, 40);
  if (!(f0 >= fmin && f0 <= fTop)) return null;

  // Tonal power: amplitude of each harmonic is 2|X|/Σw, its power A²/2.
  let tonal = 0;
  for (const m of harmonics) tonal += (2 * dtftPow(m * f0)) / (sumW * sumW);
  return {
    f0,
    confidence: Math.min(1, Math.max(0, tonal / power)),
    dominanceDb: fp.excessDb,
  };
}

/** Keeps the last `windowS` seconds of pushed blocks; estimate() runs estimateF0 on them. */
export class F0Tracker {
  private readonly fs: number;
  private readonly fmin: number;
  private readonly fmax: number;
  private readonly ring: Float64Array;
  private readonly linear: Float64Array;
  private pos = 0;
  private filled = 0;

  constructor(fs: number, windowS = 2, fmin = 20, fmax = 2000) {
    if (!(fs > 0)) throw new RangeError('F0Tracker: fs must be positive');
    this.fs = fs;
    this.fmin = fmin;
    this.fmax = fmax;
    const n = Math.max(256, Math.round(Math.max(0.05, windowS) * fs));
    this.ring = new Float64Array(n);
    this.linear = new Float64Array(n);
  }

  push(block: Float32Array): void {
    const n = this.ring.length;
    for (let i = 0; i < block.length; i++) {
      const v = block[i];
      this.ring[this.pos] = Number.isFinite(v) ? v : 0;
      this.pos = this.pos + 1 === n ? 0 : this.pos + 1;
    }
    this.filled = Math.min(n, this.filled + block.length);
  }

  estimate(): F0Estimate | null {
    const n = this.ring.length;
    const m = this.filled;
    let r = (this.pos - m + n) % n;
    for (let i = 0; i < m; i++) {
      this.linear[i] = this.ring[r];
      r = r + 1 === n ? 0 : r + 1;
    }
    return estimateF0(this.linear.subarray(0, m), this.fs, this.fmin, this.fmax);
  }

  reset(): void {
    this.ring.fill(0);
    this.pos = 0;
    this.filled = 0;
  }
}

/**
 * Phase error in degrees at harmonic h caused by an f0 error df when the total delay is
 * delaySeconds: 360·h·df·D (signed; positive df with positive D gives a positive, i.e. lagging, error).
 */
export function driftPhaseErrorDeg(dfHz: number, harmonic: number, delaySeconds: number): number {
  return 360 * harmonic * dfHz * delaySeconds;
}
