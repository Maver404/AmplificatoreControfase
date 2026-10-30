/**
 * Calibration stimuli (exponential sweep, band-limited pink noise) and sweep deconvolution.
 * Runs in the analysis worker / main thread: allocations are fine here.
 */
import { nextPow2, sharedFFT } from './fft.ts';

export interface SweepSpec {
  fs: number;
  f1: number;
  f2: number;
  durationS: number;
  fadeMs: number;
  amplitude: number;
}

/** Width of the raised-cosine band edges used by deconvolution and pink noise, in octaves. */
export const BAND_EDGE_OCTAVES = 1 / 12;

/** Relative Tikhonov regularisation of the spectral division (fraction of the in-band max |S|²). */
const DECONV_EPSILON = 1e-6;

/** Fade length used by pinkNoise so that playback never starts or stops with a step. */
const PINK_FADE_MS = 20;

const clamp01 = (v: number): number => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

function applyFades(x: Float32Array, fadeSamples: number): void {
  const nf = Math.min(Math.max(0, Math.round(fadeSamples)), Math.floor(x.length / 2));
  for (let i = 0; i < nf; i++) {
    const g = 0.5 - 0.5 * Math.cos((Math.PI * i) / nf);
    x[i] *= g;
    x[x.length - 1 - i] *= g;
  }
}

/**
 * Exponential (Farina) sine sweep with raised-cosine fades, band f1..f2, peak `amplitude`.
 * f2 is limited to fs/2 so that nothing aliases; amplitude is limited to [0, 1].
 */
export function exponentialSweep(spec: SweepSpec): Float32Array {
  const { fs, f1, durationS } = spec;
  if (!(fs > 0) || !(f1 > 0) || !(durationS > 0)) {
    throw new RangeError('exponentialSweep: fs, f1 and durationS must be positive');
  }
  const f2 = Math.min(spec.f2, fs / 2);
  if (!(f2 > f1)) throw new RangeError('exponentialSweep: f2 must be above f1 (and f1 below fs/2)');
  const amplitude = clamp01(spec.amplitude);
  const n = Math.max(2, Math.round(durationS * fs));
  const rate = durationS / Math.log(f2 / f1); // L in seconds: f(t) = f1·e^{t/L}
  const k = 2 * Math.PI * f1 * rate;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amplitude * Math.sin(k * (Math.exp(i / fs / rate) - 1));
  applyFades(out, ((Number.isFinite(spec.fadeMs) ? spec.fadeMs : 0) / 1000) * fs);
  return out;
}

/**
 * Band window with raised-cosine edges (log-frequency) lying inside [f1, f2]: 0 outside, 1 on
 * [f1·r, f2/r], r = 2^BAND_EDGE_OCTAVES. Keeping the taper inside the excited band avoids dividing
 * by a vanishing stimulus spectrum.
 */
export function bandWindow(f: number, f1: number, f2: number): number {
  if (!(f > f1) || !(f < f2)) return 0;
  let lr = BAND_EDGE_OCTAVES * Math.LN2;
  const span = Math.log(f2 / f1);
  if (2 * lr > span) lr = span / 2;
  const lo = Math.log(f / f1);
  if (lo < lr) return 0.5 - 0.5 * Math.cos((Math.PI * lo) / lr);
  const hi = Math.log(f2 / f);
  if (hi < lr) return 0.5 - 0.5 * Math.cos((Math.PI * hi) / lr);
  return 1;
}

/** Half spectrum (bins 0..n/2) of a real signal of length n (power of two). */
export interface HalfSpectrum {
  n: number;
  re: Float64Array;
  im: Float64Array;
}

/**
 * Regularised spectral division H = W·R·conj(S)/(|S|² + ε) on bins 0..n/2,
 * n = nextPow2(recorded.length + stimulus.length) so that the circular deconvolution holds every
 * causal lag up to recorded.length without wrapping.
 */
export function deconvolveHalfSpectrum(
  recorded: Float32Array,
  stimulus: Float32Array,
  fs: number,
  f1: number,
  f2: number,
): HalfSpectrum {
  if (!(fs > 0)) throw new RangeError('deconvolve: fs must be positive');
  const n = Math.max(2, nextPow2(recorded.length + stimulus.length));
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  // Both real signals in one complex FFT: z = r + j·s.
  for (let i = 0; i < recorded.length; i++) re[i] = Number.isFinite(recorded[i]) ? recorded[i] : 0;
  for (let i = 0; i < stimulus.length; i++) im[i] = Number.isFinite(stimulus[i]) ? stimulus[i] : 0;
  sharedFFT(n).forward(re, im);
  const half = n / 2;
  const fHi = Math.min(f2, fs / 2);
  const rRe = new Float64Array(half + 1);
  const rIm = new Float64Array(half + 1);
  const sRe = new Float64Array(half + 1);
  const sIm = new Float64Array(half + 1);
  let sMax = 0;
  for (let k = 0; k <= half; k++) {
    const m = (n - k) & (n - 1);
    const ar = re[k];
    const ai = im[k];
    const br = re[m];
    const bi = im[m];
    // R = (Z[k] + conj Z[n-k]) / 2 ; S = (Z[k] - conj Z[n-k]) / 2j
    rRe[k] = (ar + br) / 2;
    rIm[k] = (ai - bi) / 2;
    sRe[k] = (ai + bi) / 2;
    sIm[k] = (br - ar) / 2;
    const f = (k * fs) / n;
    if (f >= f1 && f <= fHi) sMax = Math.max(sMax, sRe[k] * sRe[k] + sIm[k] * sIm[k]);
  }
  const outRe = new Float64Array(half + 1);
  const outIm = new Float64Array(half + 1);
  if (sMax > 0) {
    const eps = DECONV_EPSILON * sMax;
    for (let k = 1; k < half; k++) {
      const w = bandWindow((k * fs) / n, f1, fHi);
      if (w === 0) continue;
      const s2 = sRe[k] * sRe[k] + sIm[k] * sIm[k];
      const d = w / (s2 + eps);
      // R·conj(S)
      outRe[k] = (rRe[k] * sRe[k] + rIm[k] * sIm[k]) * d;
      outIm[k] = (rIm[k] * sRe[k] - rRe[k] * sIm[k]) * d;
    }
  }
  return { n, re: outRe, im: outIm };
}

/**
 * Real signal from a half spectrum, optionally weighted per bin by `weight(k)` (used for the
 * detection kernel of the latency estimator).
 */
export function halfSpectrumToReal(spec: HalfSpectrum, weight?: (k: number) => number): Float64Array {
  const { n } = spec;
  const half = n / 2;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let k = 0; k <= half; k++) {
    const w = weight ? weight(k) : 1;
    re[k] = spec.re[k] * w;
    im[k] = spec.im[k] * w;
  }
  im[0] = 0;
  im[half] = 0;
  for (let k = 1; k < half; k++) {
    re[n - k] = re[k];
    im[n - k] = -im[k];
  }
  sharedFFT(n).inverse(re, im);
  return re;
}

/** Impulse response of the system that turned `stimulus` into `recorded`, by regularised spectral
 *  division restricted to [f1, f2]; length = nextPow2(recorded.length + stimulus.length).
 *  Index k ≥ 0 is lag k; the upper half of the array holds negative lags (circular). */
export function deconvolve(
  recorded: Float32Array,
  stimulus: Float32Array,
  fs: number,
  f1: number,
  f2: number,
): Float64Array {
  return halfSpectrumToReal(deconvolveHalfSpectrum(recorded, stimulus, fs, f1, f2));
}

/** Deterministic 32-bit PRNG (mulberry32), uniform in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Band-limited pink noise (for the amplifier-level step), seeded.
 * Built in the frequency domain (|X| ∝ 1/√f inside [f1, f2], random phases), so the spectrum is
 * exactly pink with no random magnitude ripple. `amplitude` (limited to [0, 1]) is the PEAK of the
 * returned signal, not its RMS (RMS is typically ≈ amplitude/4, i.e. ~12 dB lower); start and end
 * have 20 ms raised-cosine fades.
 */
export function pinkNoise(
  fs: number,
  durationS: number,
  f1: number,
  f2: number,
  amplitude: number,
  seed = 1,
): Float32Array {
  if (!(fs > 0) || !(durationS > 0) || !(f1 > 0)) throw new RangeError('pinkNoise: fs, durationS, f1 must be positive');
  const fHi = Math.min(f2, fs / 2);
  if (!(fHi > f1)) throw new RangeError('pinkNoise: f2 must be above f1');
  const len = Math.max(2, Math.round(durationS * fs));
  const n = nextPow2(len);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const rnd = mulberry32(seed);
  for (let k = 1; k < n / 2; k++) {
    const f = (k * fs) / n;
    const w = bandWindow(f, f1, fHi);
    const ph = 2 * Math.PI * rnd();
    if (w === 0) continue;
    const mag = w / Math.sqrt(f);
    re[k] = mag * Math.cos(ph);
    im[k] = mag * Math.sin(ph);
    re[n - k] = re[k];
    im[n - k] = -im[k];
  }
  sharedFFT(n).inverse(re, im);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = re[i];
  applyFades(out, Math.min((PINK_FADE_MS / 1000) * fs, len / 4));
  let peak = 0;
  for (let i = 0; i < len; i++) peak = Math.max(peak, Math.abs(out[i]));
  if (peak > 0) {
    const g = clamp01(amplitude) / peak;
    for (let i = 0; i < len; i++) out[i] *= g;
  }
  return out;
}
