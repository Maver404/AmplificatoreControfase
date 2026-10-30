/**
 * Delay / polarity estimation from sweep measurements (analysis worker; allocations allowed).
 */
import { BAND_EDGE_OCTAVES, bandWindow, deconvolveHalfSpectrum, halfSpectrumToReal } from './sweep.ts';

export interface DelayEstimate {
  /** First arrival, sub-sample. */
  delaySamples: number;
  /** Polarity of the first arrival. */
  sign: 1 | -1;
  /** Signed IR value at the arrival. */
  peak: number;
  /** Arrival peak vs pre-arrival noise floor (excluding main lobe). */
  qualityDb: number;
  ok: boolean;
  /** Italian, shown to the user when !ok. */
  reason?: string;
}

/** Minimum arrival quality accepted per measurement kind. */
export const DELAY_QUALITY_MIN_DB = { acoustic: 10, loopback: 18 } as const;
/** Arrivals within this many dB of the strongest one compete for "first arrival". */
export const FIRST_ARRIVAL_DB = -6;
/** combineEstimates rejects repetitions whose spread (std) exceeds this. */
export const MAX_COMBINE_STD_SAMPLES = 0.5;

/** Minimum length of the pre-arrival window used for the noise floor, in seconds and in main lobes. */
const NOISE_WINDOW_S = 0.02;
const NOISE_WINDOW_LOBES = 40;
/** A first-arrival candidate may be at most this much wider than the strongest arrival's lobe. */
const LOBE_WIDTH_RATIO = 1.6;
const QUALITY_CAP_DB = 200;
/** Half length of the windowed-sinc interpolator used for sub-sample refinement. */
const INTERP_HALF = 32;
/** The detection weighting is a Gaussian in frequency whose σ is f2 / DETECTION_SIGMAS. */
const DETECTION_SIGMAS = 4;
/** Recorded samples at or above this magnitude count as clipped. */
const CLIP_LEVEL = 0.999;
const CLIP_MIN_COUNT = 3;

/** Band-limited (Blackman-windowed sinc) evaluation of a circular IR at fractional index t. */
function interpolateIr(ir: Float64Array, t: number): number {
  const n = ir.length;
  const i0 = Math.floor(t);
  let acc = 0;
  for (let k = i0 - INTERP_HALF + 1; k <= i0 + INTERP_HALF; k++) {
    const x = t - k;
    const u = x / INTERP_HALF;
    if (u <= -1 || u >= 1) continue;
    const w = 0.42 + 0.5 * Math.cos(Math.PI * u) + 0.08 * Math.cos(2 * Math.PI * u);
    const s = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    acc += ir[((k % n) + n) % n] * s * w;
  }
  return acc;
}

/** Golden-section search for the maximum of f on [a, b]. */
export function goldenMax(f: (x: number) => number, a: number, b: number, iterations = 48): number {
  const g = (Math.sqrt(5) - 1) / 2;
  let lo = a;
  let hi = b;
  let x1 = hi - g * (hi - lo);
  let x2 = lo + g * (hi - lo);
  let f1 = f(x1);
  let f2 = f(x2);
  for (let i = 0; i < iterations; i++) {
    if (f1 < f2) {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + g * (hi - lo);
      f2 = f(x2);
    } else {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - g * (hi - lo);
      f1 = f(x1);
    }
  }
  const mid = (lo + hi) / 2;
  // Never return something worse than the interval ends (the maximum may sit on a boundary).
  let best = mid;
  let fb = f(mid);
  const fa = f(a);
  if (fa > fb) {
    best = a;
    fb = fa;
  }
  if (f(b) > fb) best = b;
  return best;
}

const failure = (reason: string, partial?: Partial<DelayEstimate>): DelayEstimate => ({
  delaySamples: 0,
  sign: 1,
  peak: 0,
  qualityDb: 0,
  ...partial,
  ok: false,
  reason,
});

/** Noise-floor estimator over the circular window [start, end) of the IR, given the arrival t0. */
type NoiseFloor = (ir: Float64Array, start: number, end: number, t0: number) => number;

const circ = (k: number, n: number): number => ((k % n) + n) % n;

const plainNoiseFloor: NoiseFloor = (ir, start, end) => {
  let m = 0;
  for (let k = start; k < end; k++) m = Math.max(m, Math.abs(ir[circ(k, ir.length)]));
  return m;
};

/**
 * First arrival = earliest local peak of |IR| within -6 dB of the global |IR| maximum (searched in
 * the causal half of the circular IR); sub-sample position by maximising the band-limited
 * interpolation of sign·IR; quality = |arrival| vs the maximum |IR| over a pre-arrival window ending
 * 4 main-lobe widths before the arrival, at least 20 ms and 40 main-lobe widths long (so that the
 * maximum of band-limited noise is estimated from many independent samples; the window wraps into
 * negative lags when needed). Acceptance: acoustic ≥ 10 dB, loopback ≥ 18 dB.
 */
export function analyzeImpulseResponse(ir: Float64Array, fs: number, mode: 'acoustic' | 'loopback'): DelayEstimate {
  return analyzeCore(ir, fs, mode, plainNoiseFloor);
}

function analyzeCore(ir: Float64Array, fs: number, mode: 'acoustic' | 'loopback', noiseFloor: NoiseFloor): DelayEstimate {
  const n = ir.length;
  if (n < 8 || !(fs > 0)) return failure('Risposta all’impulso troppo corta: ripeti la misura.');
  const searchEnd = Math.floor(n / 2);
  let max = 0;
  for (let k = 0; k < n; k++) {
    const v = ir[k];
    if (!Number.isFinite(v)) return failure('La misura contiene valori non validi: ripeti la misura.');
    if (k < searchEnd) max = Math.max(max, Math.abs(v));
  }
  if (max === 0) return failure('Nessun segnale ricevuto: controlla collegamenti, volume e microfono.');

  const thr = max * Math.pow(10, FIRST_ARRIVAL_DB / 20);
  const at = (k: number): number => Math.abs(ir[circ(k, n)]);
  // Half width of the lobe at k: distance to the left until |IR| drops below half of |IR[k]|.
  const lobe = (k: number): number => {
    const half = at(k) / 2;
    let w = 1;
    while (w < n / 16 && at(k - w) > half) w++;
    return w;
  };
  let iMax = 0;
  for (let k = 0; k < searchEnd; k++) if (Math.abs(ir[k]) === max) {
    iMax = k;
    break;
  }
  // A band-limited zero-phase IR has broad band-edge "skirt" lobes of opposite sign before each
  // arrival; they can reach -6 dB in narrow bands with a reverberant tail. A real arrival has the
  // kernel's lobe width (or at least the sign of the strongest arrival).
  const maxLobe = LOBE_WIDTH_RATIO * lobe(iMax) + 2;
  const maxPositive = ir[iMax] >= 0;
  let i = -1;
  for (let k = 0; k <= iMax; k++) {
    const v = Math.abs(ir[k]);
    if (!(v >= thr && v >= at(k - 1) && v >= at(k + 1))) continue;
    if (k === iMax || ir[k] >= 0 === maxPositive || lobe(k) <= maxLobe) {
      i = k;
      break;
    }
  }
  if (i < 0) return failure('Nessun arrivo riconoscibile nella misura.');
  const sign: 1 | -1 = ir[i] >= 0 ? 1 : -1;
  const t = goldenMax((x) => sign * interpolateIr(ir, x), i - 1, i + 1);
  const peak = interpolateIr(ir, t);

  // Noise window ends 4 main-lobe widths before the arrival.
  const w = lobe(i);
  const guard = 4 * w + 4;
  const len = Math.min(Math.floor(n / 4), Math.max(32, Math.round(NOISE_WINDOW_S * fs), NOISE_WINDOW_LOBES * w));
  const noise = noiseFloor(ir, i - guard - len, i - guard, t);
  const qualityDb = noise > 0 ? Math.min(QUALITY_CAP_DB, 20 * Math.log10(Math.abs(peak) / noise)) : QUALITY_CAP_DB;
  const minDb = DELAY_QUALITY_MIN_DB[mode];
  const est: DelayEstimate = { delaySamples: t, sign, peak, qualityDb, ok: qualityDb >= minDb };
  if (!est.ok) {
    est.reason =
      mode === 'loopback'
        ? `Loopback non riconosciuto (qualità ${qualityDb.toFixed(1)} dB, minimo ${minDb} dB): ` +
          'controlla il cavo (sull’ingresso libero, mai su quello del microfono di riferimento) e ripeti la misura.'
        : `Arrivo non riconoscibile (qualità ${qualityDb.toFixed(1)} dB, minimo ${minDb} dB): ` +
          'alza il livello dello stimolo o riduci il rumore ambiente e ripeti la misura.';
  }
  return est;
}

/**
 * Deconvolves the sweep recording and analyses it. The returned `ir` is flat inside [f1, f2]
 * (use it for responses and feedback FIRs); the arrival is searched on a copy additionally weighted
 * by a Gaussian exp(-½(f/σ)²), σ = f2/4: its zero-phase kernel is Gaussian in time too, so a
 * reflection a few ms later or band-edge ringing cannot bias the sub-sample peak (a sinc or Hann
 * kernel leaves ~0.05–0.8 samples of bias for a reflection 3 ms after a 1 kHz-band arrival), and
 * the band-edge skirt is removed before measuring the noise floor. `peak` is read from the flat `ir`.
 */
export function estimateDelay(
  recorded: Float32Array,
  stimulus: Float32Array,
  fs: number,
  f1: number,
  f2: number,
  mode: 'acoustic' | 'loopback',
): DelayEstimate & { ir: Float64Array } {
  if (recorded.length === 0 || stimulus.length === 0) {
    return { ...failure('Registrazione vuota: ripeti la misura.'), ir: new Float64Array(0) };
  }
  let clipped = 0;
  let recMax = 0;
  for (let i = 0; i < recorded.length; i++) {
    const a = Math.abs(recorded[i]);
    if (a >= CLIP_LEVEL) clipped++;
    if (a > recMax) recMax = a;
  }
  const spec = deconvolveHalfSpectrum(recorded, stimulus, fs, f1, f2);
  const ir = halfSpectrumToReal(spec);
  if (!(recMax > 0)) {
    return { ...failure('Nessun segnale ricevuto: controlla collegamenti, volume e microfono.'), ir };
  }
  const fTop = Math.min(f2, fs / 2);
  const sigma = fTop / DETECTION_SIGMAS;
  const gauss = (k: number): number => {
    const u = (k * fs) / spec.n / sigma;
    return Math.exp(-0.5 * u * u);
  };
  const det = halfSpectrumToReal(spec, gauss);
  // Spectrum missing below the band (1 − W)·G: the zero-phase "skirt" every arrival carries.
  const kEdge = Math.min(spec.n / 2, Math.ceil((f1 * Math.pow(2, BAND_EDGE_OCTAVES) * spec.n) / fs));
  const missing = new Float64Array(kEdge + 1);
  for (let k = 0; k <= kEdge; k++) missing[k] = (1 - bandWindow((k * fs) / spec.n, f1, fTop)) * gauss(k);
  const est = analyzeCore(det, fs, mode, skirtFreeNoiseFloor(missing));
  const out = { ...est, ir };
  // peak === 0 only when no arrival was found at all.
  if (est.peak !== 0) out.peak = interpolateIr(ir, est.delaySamples);
  if (clipped >= CLIP_MIN_COUNT) {
    out.ok = false;
    out.reason =
      'Il segnale registrato è in saturazione: abbassa il guadagno d’ingresso o il livello dello stimolo e ripeti la misura.';
  }
  return out;
}

/**
 * Noise floor after removing, by least squares, the band-edge skirt of the arrival at t0.
 * A band-limited zero-phase IR has a slow skirt −m(t − t0) around every arrival, where m is the
 * inverse transform of the missing low band (1 − W)·G; for a real system it is rotated by the phase
 * of the response near f1, so both m and its quadrature m̂ are fitted (2 of ≥ 40 degrees of freedom
 * of the window, so pure noise is not under-estimated). Without this, a 20–200 Hz sweep would cap
 * the quality at ≈ 10 dB even on a perfect loopback.
 */
function skirtFreeNoiseFloor(missing: Float64Array): NoiseFloor {
  return (ir, start, end, t0) => {
    const n = ir.length;
    const len = end - start;
    const y = new Float64Array(len);
    const m = new Float64Array(len);
    const mq = new Float64Array(len);
    let smm = 0;
    let smq = 0;
    let sqq = 0;
    let sym = 0;
    let syq = 0;
    for (let j = 0; j < len; j++) {
      const tau = start + j - t0;
      const a = (2 * Math.PI * tau) / n;
      const c1 = Math.cos(a);
      const s1 = Math.sin(a);
      let zr = 1;
      let zi = 0;
      let sc = 0;
      let ss = 0;
      for (let k = 1; k < missing.length; k++) {
        const t = zr * c1 - zi * s1;
        zi = zr * s1 + zi * c1;
        zr = t;
        sc += missing[k] * zr;
        ss += missing[k] * zi;
      }
      const yy = ir[circ(start + j, n)];
      const mm = (missing[0] + 2 * sc) / n;
      const qq = (2 * ss) / n;
      y[j] = yy;
      m[j] = mm;
      mq[j] = qq;
      smm += mm * mm;
      smq += mm * qq;
      sqq += qq * qq;
      sym += yy * mm;
      syq += yy * qq;
    }
    const det = smm * sqq - smq * smq;
    let alpha = 0;
    let beta = 0;
    if (det > 1e-12 * (smm * sqq) && det > 0) {
      alpha = (sym * sqq - syq * smq) / det;
      beta = (syq * smm - sym * smq) / det;
    } else if (smm > 0) {
      alpha = sym / smm;
    }
    let max = 0;
    for (let j = 0; j < len; j++) max = Math.max(max, Math.abs(y[j] - alpha * m[j] - beta * mq[j]));
    return max;
  };
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Median of repeated estimates (only the valid ones). ok=false if fewer than half are valid, if the
 * spread (sample std) exceeds 0.5 samples, or if any valid sign disagrees.
 */
export function combineEstimates(list: DelayEstimate[]): DelayEstimate & { stdSamples: number; count: number } {
  if (list.length === 0) {
    return { ...failure('Nessuna misura disponibile.'), stdSamples: 0, count: 0 };
  }
  const valid = list.filter((e) => e.ok && Number.isFinite(e.delaySamples));
  if (valid.length === 0) {
    const best = list.reduce((a, b) => (b.qualityDb > a.qualityDb ? b : a));
    return {
      ...failure(best.reason ?? `Nessuna delle ${list.length} misure è valida: ripeti la misura.`, best),
      stdSamples: 0,
      count: 0,
    };
  }
  const delays = valid.map((e) => e.delaySamples);
  const mean = delays.reduce((a, b) => a + b, 0) / delays.length;
  const std =
    delays.length > 1 ? Math.sqrt(delays.reduce((a, d) => a + (d - mean) ** 2, 0) / (delays.length - 1)) : 0;
  const sign = valid[0].sign;
  const result: DelayEstimate & { stdSamples: number; count: number } = {
    delaySamples: median(delays),
    sign,
    peak: median(valid.map((e) => e.peak)),
    qualityDb: median(valid.map((e) => e.qualityDb)),
    ok: true,
    stdSamples: std,
    count: valid.length,
  };
  if (valid.some((e) => e.sign !== sign)) {
    result.ok = false;
    result.reason = 'La polarità cambia tra una ripetizione e l’altra: controlla il posizionamento e ripeti la misura.';
  } else if (std > MAX_COMBINE_STD_SAMPLES) {
    result.ok = false;
    result.reason = `Misure non ripetibili (dispersione ${std.toFixed(2)} campioni, massimo ${MAX_COMBINE_STD_SAMPLES}): riduci il rumore e ripeti.`;
  } else if (valid.length * 2 < list.length) {
    result.ok = false;
    result.reason = `Solo ${valid.length} misure valide su ${list.length}: ripeti la misura.`;
  }
  return result;
}

/**
 * Complex transfer function H(f) = Σ ir[n]·e^{-j2πfn/fs} sampled at `freqs` (exact DTFT, O(len) per
 * frequency). Pass a windowed IR (e.g. bulk delay + a few thousand samples), not the whole circular
 * deconvolution: its upper half holds negative lags.
 */
export function irToResponse(ir: Float64Array, fs: number, freqs: Float64Array): { re: Float64Array; im: Float64Array } {
  const re = new Float64Array(freqs.length);
  const im = new Float64Array(freqs.length);
  for (let j = 0; j < freqs.length; j++) {
    const w = (-2 * Math.PI * freqs[j]) / fs;
    const cr = Math.cos(w);
    const ci = Math.sin(w);
    let zr = 1;
    let zi = 0;
    let ar = 0;
    let aiAcc = 0;
    for (let n = 0; n < ir.length; n++) {
      const v = ir[n];
      ar += v * zr;
      aiAcc += v * zi;
      const t = zr * cr - zi * ci;
      zi = zr * ci + zi * cr;
      zr = t;
      if ((n & 4095) === 4095) {
        // Renormalise the rotating phasor so rounding cannot accumulate over long IRs.
        const m = Math.hypot(zr, zi);
        zr /= m;
        zi /= m;
      }
    }
    re[j] = ar;
    im[j] = aiAcc;
  }
  return { re, im };
}
