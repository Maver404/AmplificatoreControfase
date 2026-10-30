/**
 * Calibration: measurement orchestration (sweeps played by the worklet, analysis by an Analyzer),
 * the plain-JSON calibration record stored in presets, and the conversion of the alignment solution
 * into channel parameters.
 *
 * Pure TypeScript with no DOM access: the browser wizard, the analysis worker (welchTransfer) and the
 * Node closed-loop simulator all run this file unmodified.
 */
import { loopGainDb, sampleResponse, solveAlignment } from '../dsp/alignment.ts';
import type { AlignmentResult, ChannelSolution, FreqResponse, SpeakerConfig } from '../dsp/alignment.ts';
import { FEEDBACK_MAX_TAPS } from '../dsp/antiphase-engine.ts';
import type { Complex } from '../dsp/complex.ts';
import type { F0Estimate } from '../dsp/f0-tracker.ts';
import { designFeedbackFir } from '../dsp/feedback-canceller.ts';
import { nextPow2, sharedFFT } from '../dsp/fft.ts';
import { hilbertResponse } from '../dsp/hilbert-pair.ts';
import { combineEstimates, irToResponse } from '../dsp/latency-estimator.ts';
import type { DelayEstimate } from '../dsp/latency-estimator.ts';
import { D_MIN_SAMPLES, MASTER_MIN_DB, MAX_DELAY_MS, RANGES, clampParams } from '../dsp/params.ts';
import type { EngineParams } from '../dsp/params.ts';
import { geometryDelay, headStartSeconds, speedOfSound } from '../dsp/physics.ts';
import { channelResponse, conditioningResponse, groupDelaySamples } from '../dsp/response.ts';
import { exponentialSweep, mulberry32, pinkNoise } from '../dsp/sweep.ts';

// ------------------------------------------------------------------ types

export type MeasureFn = (
  stimulus: Float32Array,
  channel: 0 | 1,
  postRollFrames: number,
) => Promise<{ ref: Float32Array; err: Float32Array | null; masterLin: number; aborted: boolean }>;

/** Ambient recording with the anti-noise muted. */
export type CaptureFn = (seconds: number) => Promise<{ ref: Float32Array; err: Float32Array | null }>;

/** H1 = Gxy/Gxx (Welch) with the magnitude-squared coherence; `pxx` (one-sided PSD of x) is an extension. */
export interface TransferEstimate {
  re: Float64Array;
  im: Float64Array;
  coherence: Float64Array;
  pxx?: Float64Array;
}

/** Implemented in-process (tests, simulator, fallback) and by the analysis worker client (browser). */
export interface Analyzer {
  estimateDelay(
    recorded: Float32Array,
    stimulus: Float32Array,
    fs: number,
    f1: number,
    f2: number,
    mode: 'acoustic' | 'loopback',
  ): Promise<DelayEstimate & { ir: Float64Array }>;
  transfer(x: Float32Array, y: Float32Array, fs: number, freqs: Float64Array): Promise<TransferEstimate>;
  estimateF0(x: Float32Array, fs: number, fmin: number, fmax: number): Promise<F0Estimate | null>;
}

export interface PathData {
  delaySamples: number;
  sign: 1 | -1;
  qualityDb: number;
  stdSamples: number;
  /** Response of the FINAL output sample → mic (already divided by masterLin). */
  freqs: number[];
  re: number[];
  im: number[];
  /** Impulse response window (final output → mic) whose first sample is lag `irOffset` (may be < 0). */
  ir: number[];
  irOffset: number;
  /** Extension: reverberation time estimated from the full IR (Schroeder), for the autotune settling. */
  rt60S?: number;
}

export interface PrimaryData {
  freqs: number[];
  re: number[];
  im: number[];
  /** Magnitude-squared coherence 0..1. */
  coherence: number[];
  tonalF0: number | null;
  /** Extension: one-sided PSD of the reference during the capture (solver weights). */
  psd?: number[];
}

export interface GeometryData {
  micToZoneM: number;
  sourceAngleDeg: number;
  speakerToZoneM: [number, number];
  tempC: number;
}

/** Plain JSON (number arrays, no typed arrays, no NaN), stored in presets. */
export interface CalibrationData {
  version: 1;
  fs: number;
  band: [number, number];
  tempC: number;
  createdAt: number;
  /** Output channel → reference mic. */
  feedback: [PathData | null, PathData | null];
  /** Output channel → error mic (or zone mic in two-position mode). */
  secondary: [PathData | null, PathData | null];
  primary: PrimaryData | null;
  loopbackLatencyS: number | null;
  geometry: GeometryData | null;
  /** Two-position calibration: noise level in the zone and at the reference (dBFS). */
  levels: { zoneDb: number | null; refDb: number | null };
  speakers: SpeakerConfig;
  /**
   * Input of the reference microphone the paths were measured with (EngineParams.refChannel). Absent in
   * calibrations saved before it was recorded: then the input of the preset's parameters.
   */
  refChannel?: 0 | 1;
}

export interface MeasurePathOptions {
  fs: number;
  channel: 0 | 1;
  band: [number, number];
  /** Peak level of the stimulus in dBFS (before the master volume), e.g. -20. */
  levelDbfs: number;
  repetitions: number;
  mic: 'ref' | 'err';
  /** 1-based input the error mic is on (named when it is missing; default 2, 1 when the reference is on 2). */
  errorInput?: 1 | 2;
  mode: 'acoustic' | 'loopback';
  /** Play one sweep PREVIEW_DB lower first and stop on silence or predicted saturation. */
  previewFirst: boolean;
  signal?: AbortSignal;
  /** Extension: sweep length in seconds (default SWEEP_S). */
  sweepS?: number;
  /** Extension: recording after the sweep in seconds (default POST_ROLL_S). */
  postRollS?: number;
  /** Extension: remove stationary ambient tones from the recordings before the analysis (default true). */
  removeTones?: boolean;
}

export type PathResult = PathData & { ok: boolean; reason?: string; aborted?: boolean };

export interface SolveOutcome {
  params: EngineParams;
  /**
   * The solver's result. predictedAttenuationDb is recomputed for the returned params (after clamping
   * and the loop-gain limit) and, with an error mic, includes the part of the zone noise the
   * reference does not hear (coherence): it is the attenuation the error mic can expect.
   */
  result: AlignmentResult;
  /** Italian, shown to the user. */
  advice: string[];
  /**
   * Open-loop gain of `params` through the feedback paths, without neutralisation (null without
   * feedback data). When the params are left unchanged, the value of the rejected solution.
   */
  loopGainDb: number | null;
  /** Extension: the same after neutralisation with the FIRs feedbackFirs() designs (null without feedback data). */
  loopGainAfterDb?: number | null;
  /**
   * Extension: dB taken off every channel gain so that the neutralised loop stays at LOOP_GAIN_DANGER_DB
   * (−6 dB, the stability margin; 0 = no change). When that leaves less than 1 dB of predicted
   * attenuation the params are left unchanged.
   */
  loopLimitedDb?: number;
}

/** Extension options of the solvers. */
export interface SolveOptions {
  /** Neutraliser length actually loaded (feedbackFirs), for the loop-gain check; default DEFAULT_FEEDBACK_TAPS. */
  feedbackTaps?: number;
}

/** A calibration step failed; `message` is Italian and meant for the user. */
export class CalibrationError extends Error {
  readonly aborted: boolean;
  constructor(message: string, aborted = false) {
    super(message);
    this.name = 'CalibrationError';
    this.aborted = aborted;
  }
}

// -------------------------------------------------------------- constants

/** Default sweep length and recording after it (covers the IR window after any plausible path delay). */
export const SWEEP_S = 1.5;
export const POST_ROLL_S = 0.5;
const SWEEP_FADE_MS = 10;
/** The preview sweep is this much lower than the real ones. */
export const PREVIEW_DB = -20;
/** Predicted recorded peak of the real sweep above which the input would saturate. */
const SATURATION_PEAK = 0.9;
/** Window used to estimate the preview's peak gain: from PEAK_PRE_S before the arrival, PEAK_WINDOW_S long. */
const PEAK_PRE_S = 0.002;
const PEAK_WINDOW_S = 0.025;
const CLIP_PEAK = 0.999;
/** Passes of ambient-tone removal per sweep recording (see withoutTones). */
const TONE_PASSES = 2;
/** A recording whose peak is below −100 dBFS is digital silence: nothing is connected. */
const SILENT_PEAK = 1e-5;
export const MAX_REPETITIONS = 10;
/**
 * Stored IR window: IR_PRE_S before the first arrival to IR_POST_S after it, with raised-cosine tapers
 * of IR_PRE_TAPER_S / IR_POST_TAPER_S at the ends. The deconvolved IR is band-limited by a zero-phase
 * window, so every arrival carries slowly decaying band-edge ringing on BOTH sides; a window that cuts
 * it abruptly just before the arrival leaks the band edge into the band (≈ 7 % error at 100 Hz for a
 * 20 Hz edge). Starting well before the arrival with a long smooth taper keeps the in-band response
 * accurate to well under 1 %. The window may start at a negative lag (irOffset < 0).
 */
export const IR_PRE_S = 0.04;
export const IR_POST_S = 0.1;
const IR_PRE_TAPER_S = 0.03;
const IR_POST_TAPER_S = 0.025;
/** Stored response grid: cells of at least GRID_MIN_STEP_HZ, at most GRID_MAX_POINTS points (see calibrationFreqs). */
const GRID_MIN_STEP_HZ = 2;
const GRID_MAX_POINTS = 800;
const GRID_SEED = 0x6a17;
/** Welch segment length and longest bulk delay searched between reference and error. */
const WELCH_SEGMENT_S = 0.1;
const MAX_BULK_LAG_S = 0.3;
const BULK_LAG_EXCERPT_S = 8;
/** f0 is estimated on at most this much of the primary capture (plenty for 0.01 %). */
const F0_EXCERPT_S = 10;
/**
 * Channel gains outside [min, max] are better fixed at the source of the level: above max on the
 * reference mic's input gain (the amplifier level of wizard step 2 is the safety margin), below min
 * on the amplifier knob; target = middle.
 */
export const AMP_ADVICE_DB = { min: -30, max: -6, target: -18 } as const;
/**
 * Plan: |Σ G·H·F| ≤ −20 dB after neutralisation (the target: above it the advice warns of a less
 * accurate cancellation); above −6 dB the loop has no stability margin and finishSolve lowers the gains.
 */
export const LOOP_GAIN_LIMIT_DB = -20;
export const LOOP_GAIN_DANGER_DB = -6;
export const DEFAULT_FEEDBACK_TAPS = 256;
/** Neutraliser lengths the user can choose ("Avanzate"); feedbackFirs may lengthen a FIR up to the max. */
export const FEEDBACK_TAPS_RANGE = { min: 128, max: FEEDBACK_MAX_TAPS } as const;
/** Arrivals of a feedback IR within this of its strongest one are all covered by the FIR (feedbackFirs). */
const FIR_ARRIVAL_DB = -20;
/** A peak counts as an arrival when its lobe is at most this many times the strongest one's (not a band-edge skirt). */
const FIR_LOBE_RATIO = 2;
/** …and when it stands this many noise σ above the IR's diffuse part (noise, reverberation): not a noise peak. */
const FIR_NOISE_PEAK = 5;
/**
 * Two-level check (plan: OS processing): the preview, PREVIEW_DB lower, and the real sweeps must give
 * the same in-band gain, within LEVEL_GAIN_TOL_DB and LEVEL_GAIN_SIGMAS of the preview's own noise
 * (see levelCheck). Compared only when the preview found a clean arrival.
 */
const LEVEL_GAIN_TOL_DB = 1;
const LEVEL_GAIN_SIGMAS = 5;
const LEVEL_CHECK_MIN_QUALITY_DB = 15;
/** Bulk-delay search of a stored primary path (tonal mode), seconds either way. */
const PRIMARY_BULK_MAX_S = 0.5;
/** A coherence this low reads as "nothing coherent" (keeps Gyy = |H1|²·Gxx/γ² finite). */
const MIN_COHERENCE = 1e-4;
/** Below this predicted attenuation the solution is reported as weak; below MIN_USEFUL_DB it is not applied. */
const WEAK_ATTENUATION_DB = 6;
const MIN_USEFUL_DB = 1;
/** Mean in-band coherence below which the primary measurement is reported as unreliable. */
const LOW_COHERENCE = 0.5;
/** looksLikeLoopback thresholds. */
const LOOPBACK_QUALITY_DB = 18;
const LOOPBACK_FLAT_DB = 3;
const LOOPBACK_PHASE_DEG = 20;
const LOOPBACK_LATENCY_TOL_S = 0.0005;
/** Significant digits kept in stored arrays (keeps presets small; far below measurement noise). */
const STORE_DIGITS = 7;
const MAX_STORED_POINTS = 1 << 16;
const MAX_HARMONICS = 64;
const CH_NAME = ['L', 'R'] as const;

// ---------------------------------------------------------------- helpers

const fmt = (x: number, digits = 1): string => x.toFixed(digits).replace('.', ',').replace('-', '−');
const dbToAmp = (db: number): number => Math.pow(10, db / 20);
const store = (v: ArrayLike<number>): number[] => {
  const out = new Array<number>(v.length);
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    out[i] = Number.isFinite(x) ? Number(x.toPrecision(STORE_DIGITS)) : 0;
  }
  return out;
};

function peakAbs(x: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    if (a > m) m = a;
  }
  return m;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new CalibrationError('Misura annullata.', true);
}

/** [lo, hi] inside (0, fs/2), or null when the band is unusable. */
function normaliseBand(band: [number, number], fs: number): [number, number] | null {
  if (!Array.isArray(band) || band.length !== 2) return null;
  const lo = Math.min(band[0], band[1]);
  const hi = Math.min(Math.max(band[0], band[1]), fs / 2);
  return Number.isFinite(lo) && Number.isFinite(hi) && lo > 0 && hi > lo * 1.05 ? [lo, hi] : null;
}

/**
 * Default measurement band: [max(0.8·hpFreq, 20), min(2·lpFreq, 0.45·fs)] (operating band of the
 * conditioning with an octave of margin above the low-pass).
 */
export function defaultSweepBand(params: EngineParams, fs: number): [number, number] {
  const f = clampParams(params).filters;
  const lo = Math.max(f.hpEnabled ? 0.8 * f.hpFreq : 0, 20);
  const lp = f.lpEnabled ? f.lpFreq : RANGES.lpFreq.max;
  const hi = Math.min(2 * lp, 0.45 * fs);
  return [lo, Math.max(hi, lo * 2)];
}

/**
 * The frequency grid every stored response uses, so that primary and secondary paths share it exactly
 * (the solver then needs no interpolation). One point per cell of width df = max(2 Hz, band/800), at a
 * fixed pseudo-random position inside the cell: with a uniform grid the residual Σ|P + G·S·e^{-jωD}|²
 * is periodic in D with period 1/df (0.5 s at 2 Hz — inside the delay search range), so a broadband
 * solution could "align" 0.5 s off. Stratified jitter averages those aliases out (E e^{j2π·m·u} = 0
 * for u uniform in a cell, |sinc(0.96)| ≈ 0.04 with the margins used) while keeping the points evenly
 * spread, i.e. equal quadrature weights.
 */
export function calibrationFreqs(fs: number, band: [number, number]): Float64Array {
  const b = normaliseBand(band, fs);
  if (!b) return new Float64Array(0);
  const df = Math.max(GRID_MIN_STEP_HZ, (b[1] - b[0]) / GRID_MAX_POINTS);
  const n = Math.max(1, Math.floor((b[1] - b[0]) / df + 1e-9));
  const rnd = mulberry32(GRID_SEED);
  // 2 % margins keep neighbours ≥ 0.04·df apart, so 7-digit storage never merges two points.
  return Float64Array.from({ length: n }, (_, i) => b[0] + (i + 0.02 + 0.96 * rnd()) * df);
}

/** Harmonics h·f0 exactly as solveAlignment builds its tonal grid. */
function harmonicFreqs(f0: number, band: [number, number], fs: number): Float64Array {
  const lo = Math.max(0, Math.min(band[0], band[1]));
  const hi = Math.min(fs / 2, Math.max(band[0], band[1]));
  const out: number[] = [];
  for (let h = 1; h <= MAX_HARMONICS && h * f0 <= Math.min(hi, 0.45 * fs); h++) {
    if (h * f0 >= lo) out.push(h * f0);
  }
  return Float64Array.from(out);
}

/** Response of a stored path at `freqs`: exact DTFT of the stored IR window when present. */
export function pathResponse(path: PathData, fs: number, freqs: Float64Array): FreqResponse {
  if (path.ir.length > 0) {
    const r = irToResponse(Float64Array.from(path.ir), fs, freqs);
    const re = new Float64Array(freqs.length);
    const im = new Float64Array(freqs.length);
    for (let k = 0; k < freqs.length; k++) {
      // The window starts at lag irOffset: H(f) = e^{-jω·irOffset}·DTFT(window).
      const a = (-2 * Math.PI * freqs[k] * path.irOffset) / fs;
      const c = Math.cos(a);
      const s = Math.sin(a);
      re[k] = r.re[k] * c - r.im[k] * s;
      im[k] = r.re[k] * s + r.im[k] * c;
    }
    return { freqs, re, im };
  }
  const s = sampleResponse(storedResponse(path), freqs);
  return { freqs, re: s.re, im: s.im };
}

const storedResponse = (r: { freqs: number[]; re: number[]; im: number[] }): FreqResponse => ({
  freqs: Float64Array.from(r.freqs),
  re: Float64Array.from(r.re),
  im: Float64Array.from(r.im),
});

const emptyPath = (): PathData => ({
  delaySamples: 0,
  sign: 1,
  qualityDb: 0,
  stdSamples: 0,
  freqs: [],
  re: [],
  im: [],
  ir: [],
  irOffset: 0,
});

/** A new, empty calibration record. */
export function createCalibration(fs: number, band: [number, number], speakers: SpeakerConfig, tempC = 20): CalibrationData {
  return {
    version: 1,
    fs,
    band: [band[0], band[1]],
    tempC,
    createdAt: Date.now(),
    feedback: [null, null],
    secondary: [null, null],
    primary: null,
    loopbackLatencyS: null,
    geometry: null,
    levels: { zoneDb: null, refDb: null },
    speakers,
  };
}

// ------------------------------------------------------------ Welch / H1

/**
 * Integer lag of y relative to x (y[n] ≈ h * x[n − lag]) by GCC-PHAT restricted to [fLo, fHi], searched
 * within ±MAX_BULK_LAG_S. For a periodic signal the peak is ambiguous by whole periods, which does not
 * change H1 at its harmonics; for broadband noise it removes the Welch bias of a long path delay.
 */
function bulkLag(x: Float64Array, y: Float64Array, fs: number, fLo: number, fHi: number): number {
  // A few seconds give a stable lag; longer captures would only make the FFT huge.
  const n = Math.min(x.length, Math.round(BULK_LAG_EXCERPT_S * fs));
  const maxLag = Math.min(Math.round(MAX_BULK_LAG_S * fs), Math.floor(n / 4));
  if (maxLag < 1) return 0;
  const size = nextPow2(n + maxLag);
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  re.set(x.subarray(0, n));
  im.set(y.subarray(0, n));
  const fft = sharedFFT(size);
  fft.forward(re, im);
  const half = size / 2;
  const cr = new Float64Array(size);
  const ci = new Float64Array(size);
  const kLo = Math.max(1, Math.floor((fLo * size) / fs));
  const kHi = Math.min(half - 1, Math.ceil((fHi * size) / fs));
  for (let k = kLo; k <= kHi; k++) {
    const j = size - k;
    const xr = (re[k] + re[j]) / 2;
    const xi = (im[k] - im[j]) / 2;
    const yr = (im[k] + im[j]) / 2;
    const yi = (re[j] - re[k]) / 2;
    // conj(X)·Y, whitened (PHAT)
    const pr = xr * yr + xi * yi;
    const pi = xr * yi - xi * yr;
    const m = Math.hypot(pr, pi);
    if (!(m > 0)) continue;
    cr[k] = pr / m;
    ci[k] = pi / m;
    cr[j] = cr[k];
    ci[j] = -ci[k];
  }
  fft.inverse(cr, ci);
  let best = 0;
  let bestV = -Infinity;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const v = cr[lag >= 0 ? lag : size + lag];
    if (v > bestV) {
      bestV = v;
      best = lag;
    }
  }
  return best;
}

/**
 * H1 = Gxy/Gxx by Welch's method (Hann, 50 % overlap, ~WELCH_SEGMENT_S segments zero-padded 2×, linear
 * interpolation between bins), after removing the bulk delay between x and y so that a long path delay
 * does not bias the estimate. Also returns the magnitude-squared coherence and the one-sided PSD of x.
 * Frequencies outside (0, fs/2) or without excitation give zeros.
 */
export function welchTransfer(x: Float32Array, y: Float32Array, fs: number, freqs: Float64Array): Required<TransferEstimate> {
  const m = freqs.length;
  const out = { re: new Float64Array(m), im: new Float64Array(m), coherence: new Float64Array(m), pxx: new Float64Array(m) };
  const n = Math.min(x.length, y.length);
  if (!(fs > 0) || m === 0 || n < 512) return out;
  let fLo = Infinity;
  let fHi = 0;
  for (let k = 0; k < m; k++) {
    const f = freqs[k];
    if (!(f > 0 && f < fs / 2)) continue;
    fLo = Math.min(fLo, f);
    fHi = Math.max(fHi, f);
  }
  if (!(fHi > 0)) return out;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = Number.isFinite(x[i]) ? x[i] : 0;
    ys[i] = Number.isFinite(y[i]) ? y[i] : 0;
  }
  const lag = bulkLag(xs, ys, fs, fLo, fHi);
  const ax = lag < 0 ? -lag : 0;
  const ay = lag > 0 ? lag : 0;
  const len = n - Math.abs(lag);
  let nfft = nextPow2(Math.round(fs * WELCH_SEGMENT_S));
  while (nfft > 256 && 2 * nfft > len) nfft /= 2;
  if (len < nfft) return out;
  const size = 2 * nfft;
  const hop = nfft / 2;
  const fft = sharedFFT(size);
  const w = new Float64Array(nfft);
  let sumW2 = 0;
  for (let i = 0; i < nfft; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft);
    sumW2 += w[i] * w[i];
  }
  const kLo = Math.max(0, Math.floor((fLo * size) / fs));
  const kHi = Math.min(size / 2, Math.ceil((fHi * size) / fs) + 1);
  const sxx = new Float64Array(kHi + 1);
  const syy = new Float64Array(kHi + 1);
  const sxr = new Float64Array(kHi + 1);
  const sxi = new Float64Array(kHi + 1);
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  let count = 0;
  for (let s = 0; s + nfft <= len; s += hop) {
    let mx = 0;
    let my = 0;
    for (let i = 0; i < nfft; i++) {
      mx += xs[ax + s + i];
      my += ys[ay + s + i];
    }
    mx /= nfft;
    my /= nfft;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < nfft; i++) {
      re[i] = (xs[ax + s + i] - mx) * w[i];
      im[i] = (ys[ay + s + i] - my) * w[i];
    }
    fft.forward(re, im);
    for (let k = kLo; k <= kHi; k++) {
      const j = (size - k) & (size - 1);
      const xr = (re[k] + re[j]) / 2;
      const xi = (im[k] - im[j]) / 2;
      const yr = (im[k] + im[j]) / 2;
      const yi = (re[j] - re[k]) / 2;
      sxx[k] += xr * xr + xi * xi;
      syy[k] += yr * yr + yi * yi;
      sxr[k] += xr * yr + xi * yi;
      sxi[k] += xr * yi - xi * yr;
    }
    count++;
  }
  const psdScale = 2 / (count * fs * sumW2);
  for (let j = 0; j < m; j++) {
    const f = freqs[j];
    if (!(f > 0 && f < fs / 2)) continue;
    const b = (f * size) / fs;
    const k0 = Math.min(Math.max(kLo, Math.floor(b)), kHi);
    const k1 = Math.min(k0 + 1, kHi);
    const t = k1 > k0 ? Math.min(1, Math.max(0, b - k0)) : 0;
    const gxx = sxx[k0] + t * (sxx[k1] - sxx[k0]);
    const gyy = syy[k0] + t * (syy[k1] - syy[k0]);
    const gr = sxr[k0] + t * (sxr[k1] - sxr[k0]);
    const gi = sxi[k0] + t * (sxi[k1] - sxi[k0]);
    if (!(gxx > 0)) continue;
    const hr = gr / gxx;
    const hi = gi / gxx;
    // H = H_aligned · e^{-jω·lag}
    const a = (-2 * Math.PI * f * lag) / fs;
    const c = Math.cos(a);
    const sn = Math.sin(a);
    out.re[j] = hr * c - hi * sn;
    out.im[j] = hr * sn + hi * c;
    out.coherence[j] = gyy > 0 ? Math.min(1, (gr * gr + gi * gi) / (gxx * gyy)) : 0;
    out.pxx[j] = gxx * psdScale;
  }
  return out;
}

// ------------------------------------------------------------ levels, decay

/**
 * Statistical bandwidth (Σ S·Δf)² / Σ S²·Δf, Hz, of the noise at the error mic inside the calibration
 * band, from the primary measurement: Gyy = |H1|²·Gxx/γ² (the reference PSD carried through the path,
 * plus what the reference does not hear). The level of a window of T seconds of that noise scatters by
 * 10/ln10/√(B·T) dB (the autotune sizes its windows on it). Null without a primary PSD.
 */
export function noiseBandwidthHz(cal: CalibrationData): number | null {
  const prim = cal.primary;
  if (prim === null || prim.psd === undefined || prim.freqs.length < 2) return null;
  const f = prim.freqs;
  const n = f.length;
  let s1 = 0;
  let s2 = 0;
  for (let k = 0; k < n; k++) {
    if (f[k] < cal.band[0] || f[k] > cal.band[1]) continue;
    // Each point stands for the cell between the midpoints to its neighbours.
    const df = ((k + 1 < n ? f[k + 1] : f[k]) - (k > 0 ? f[k - 1] : f[k])) / (k > 0 && k + 1 < n ? 2 : 1);
    const c = Math.max(MIN_COHERENCE, Math.min(1, prim.coherence[k]));
    const gyy = ((prim.re[k] * prim.re[k] + prim.im[k] * prim.im[k]) * prim.psd[k]) / c;
    if (!(gyy > 0) || !(df > 0)) continue;
    s1 += gyy * df;
    s2 += gyy * gyy * df;
  }
  return s2 > 0 ? (s1 * s1) / s2 : null;
}

/**
 * Reverberation time from an impulse response: Schroeder backward integration of the energy after the
 * arrival (5 ms frames, noise floor from the last fifth subtracted), T20 (−5…−25 dB) × 3, or T10 × 6
 * when the decay does not reach −25 dB above the noise. Null when no decay can be measured.
 */
export function estimateRt60(ir: Float64Array, arrival: number, fs: number): number | null {
  const half = ir.length >> 1;
  const start = Math.max(0, Math.floor(arrival));
  const frame = Math.max(1, Math.round(0.005 * fs));
  const frames = Math.floor((half - start) / frame);
  if (frames < 20) return null;
  const e = new Float64Array(frames);
  for (let j = 0; j < frames; j++) {
    let acc = 0;
    for (let i = start + j * frame, end = i + frame; i < end; i++) acc += ir[i] * ir[i];
    e[j] = acc;
  }
  const tail = Math.floor(frames / 5);
  let noise = 0;
  for (let j = frames - tail; j < frames; j++) noise += e[j];
  noise /= tail;
  // Integrate only while the decay is above the noise floor.
  let last = 0;
  for (let j = 0; j < frames - tail; j++) if (e[j] > 2 * noise) last = j;
  const edc = new Float64Array(last + 1);
  let acc = 0;
  for (let j = last; j >= 0; j--) {
    acc += Math.max(0, e[j] - noise);
    edc[j] = acc;
  }
  if (!(edc[0] > 0)) return null;
  const crossing = (db: number): number | null => {
    const lim = edc[0] * Math.pow(10, db / 10);
    for (let j = 1; j <= last; j++) if (edc[j] <= lim) return j;
    return null;
  };
  const t5 = crossing(-5);
  const t25 = crossing(-25);
  const t15 = crossing(-15);
  if (t5 !== null && t25 !== null && t25 > t5) return (3 * (t25 - t5) * frame) / fs;
  if (t5 !== null && t15 !== null && t15 > t5) return (6 * (t15 - t5) * frame) / fs;
  return null;
}

/** RMS level (dBFS, −120 floor) of the part of x inside [lo, hi] Hz (Welch, Hann, 4096-point segments). */
export function bandRmsDbfs(x: Float32Array, fs: number, band: [number, number]): number {
  const n = Math.min(4096, nextPow2(x.length) / 2);
  if (!(fs > 0) || n < 64) return -120;
  const fft = sharedFFT(n);
  const w = new Float64Array(n);
  let sw2 = 0;
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    sw2 += w[i] * w[i];
  }
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const k0 = Math.max(1, Math.ceil((Math.min(band[0], band[1]) * n) / fs));
  const k1 = Math.min(n / 2 - 1, Math.floor((Math.max(band[0], band[1]) * n) / fs));
  let acc = 0;
  let count = 0;
  for (let s = 0; s + n <= x.length; s += n / 2) {
    for (let i = 0; i < n; i++) {
      re[i] = (Number.isFinite(x[s + i]) ? x[s + i] : 0) * w[i];
      im[i] = 0;
    }
    fft.forward(re, im);
    for (let k = k0; k <= k1; k++) acc += re[k] * re[k] + im[k] * im[k];
    count++;
  }
  // Parseval with the window: mean square in band = 2·Σ|X_k|² / (n·Σw²) per segment.
  const ms = count > 0 ? (2 * acc) / (count * n * sw2) : 0;
  return ms > 0 ? Math.max(-120, 10 * Math.log10(ms)) : -120;
}

/**
 * Band-limited pink noise for the amplifier-level step at `rmsDbfs` RMS (default −20 dBFS; pinkNoise's
 * amplitude is a peak, so the level is set on the measured RMS, with the peak limited to 1).
 * It is a stimulus: it goes through the master volume and the limiter like the sweeps.
 */
export function amplifierLevelNoise(fs: number, band: [number, number], seconds: number, rmsDbfs = -20, seed = 1): Float32Array {
  const x = pinkNoise(fs, seconds, band[0], band[1], 0.5, seed);
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const rms = Math.sqrt(ss / Math.max(1, x.length));
  if (!(rms > 0)) return x;
  let g = dbToAmp(Math.min(0, rmsDbfs)) / rms;
  g = Math.min(g, 1 / peakAbs(x));
  for (let i = 0; i < x.length; i++) x[i] *= g;
  return x;
}

// ------------------------------------------------------------ measurement

class MeasureFailure extends Error {
  readonly aborted: boolean;
  constructor(message: string, aborted = false) {
    super(message);
    this.aborted = aborted;
  }
}

/** Tapered window of a circular deconvolved IR around `arrival` (see IR_PRE_S); offset = first lag. */
function irWindow(ir: Float64Array, arrival: number, fs: number): { win: Float64Array; offset: number } {
  const n = ir.length;
  const pre = Math.round(IR_PRE_S * fs);
  const offset = Math.floor(arrival) - pre;
  // Never reach into the negative lags of the arrival's own wrap-around nor past the causal half.
  const len = Math.max(0, Math.min(pre + Math.round(IR_POST_S * fs), (n >> 1) - offset, n >> 1));
  const win = new Float64Array(len);
  for (let i = 0; i < len; i++) win[i] = ir[(((offset + i) % n) + n) % n];
  const tin = Math.min(Math.round(IR_PRE_TAPER_S * fs), len >> 1);
  for (let i = 0; i < tin; i++) win[i] *= 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / tin);
  const tout = Math.min(Math.round(IR_POST_TAPER_S * fs), len >> 1);
  for (let i = 0; i < tout; i++) win[len - 1 - i] *= 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / tout);
  return { win, offset };
}

/**
 * Peak |H| of the stimulus → mic response held in a deconvolved IR, for the saturation check of the
 * preview. A short window (direct sound and early reflections) smooths |H| and keeps stationary ambient
 * tones, which leave long narrowband ripples in the IR of a quiet sweep, from inflating the estimate;
 * the band edges (where the short window leaks) are skipped.
 */
function peakGain(ir: Float64Array, arrival: number, fs: number, band: [number, number]): number {
  const n = ir.length;
  let at = arrival;
  if (!(arrival > 0)) {
    let iMax = 0;
    for (let k = 1; k < n >> 1; k++) if (Math.abs(ir[k]) > Math.abs(ir[iMax])) iMax = k;
    at = iMax;
  }
  const pre = Math.round(PEAK_PRE_S * fs);
  const len = Math.round(PEAK_WINDOW_S * fs);
  const start = Math.floor(at) - pre;
  const win = new Float64Array(len);
  for (let i = 0; i < len; i++) win[i] = ir[(((start + i) % n) + n) % n];
  const tin = pre >> 1;
  for (let i = 0; i < tin; i++) win[i] *= 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / tin);
  const tout = len >> 2;
  for (let i = 0; i < tout; i++) win[len - 1 - i] *= 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / tout);
  const lo = band[0] * Math.pow(2, 1 / 3);
  const hi = Math.max(lo * 1.01, band[1] * Math.pow(2, -1 / 6));
  const freqs = Float64Array.from({ length: 48 }, (_, i) => lo * Math.pow(hi / lo, i / 47));
  const r = irToResponse(win, fs, freqs);
  let m = 0;
  for (let k = 0; k < freqs.length; k++) m = Math.max(m, Math.hypot(r.re[k], r.im[k]));
  return m;
}

/** y[n] = Σ_k h[k]·x[n − offset − k] for n ∈ [0, length) (FFT, no wrap-around). */
function convolve(h: Float64Array, offset: number, x: Float32Array, length: number): Float64Array {
  const size = nextPow2(length + h.length + x.length + Math.abs(offset));
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  // h at its lags (negative lags wrap to the end), x in the imaginary part: one complex FFT for both.
  for (let k = 0; k < h.length; k++) re[(((offset + k) % size) + size) % size] = h[k];
  for (let i = 0; i < x.length; i++) im[i] = x[i];
  const fft = sharedFFT(size);
  fft.forward(re, im);
  const pr = new Float64Array(size);
  const pi = new Float64Array(size);
  for (let k = 0; k <= size / 2; k++) {
    const j = (size - k) & (size - 1);
    const hr = (re[k] + re[j]) / 2;
    const hi = (im[k] - im[j]) / 2;
    const xr = (im[k] + im[j]) / 2;
    const xi = (re[j] - re[k]) / 2;
    pr[k] = hr * xr - hi * xi;
    pi[k] = hr * xi + hi * xr;
    pr[j] = pr[k];
    pi[j] = -pi[k];
  }
  fft.inverse(pr, pi);
  return pr.subarray(0, length);
}

/**
 * Removes a stationary ambient harmonic series (e.g. a machine's hum) from a sweep recording. A
 * steady tone leaves a narrowband ripple across the whole deconvolved IR, i.e. an error in the
 * measured response exactly at its harmonics — where the tonal mode needs the response most. The
 * sweep's own contribution (first IR estimate convolved with the stimulus) is subtracted, the tone is
 * found in what is left (estimateF0, sub-bin), each harmonic in the band is fitted (Hann-weighted
 * projection) and subtracted from the recording. Returns null when there is no clear tone or no valid
 * first estimate.
 */
async function withoutTones(
  rec: Float32Array,
  stim: Float32Array,
  est: DelayEstimate & { ir: Float64Array },
  analyzer: Analyzer,
  fs: number,
  band: [number, number],
): Promise<Float32Array | null> {
  // Without a valid first estimate the sweep cannot be subtracted, and fitting tones on the raw
  // recording would take part of the sweep with them.
  if (!est.ok || est.ir.length === 0) return null;
  const { win, offset } = irWindow(est.ir, est.delaySamples, fs);
  const pred = convolve(win, offset, stim, rec.length);
  const resid = Float32Array.from(rec, (v, i) => v - pred[i]);
  const t = await analyzer.estimateF0(resid, fs, RANGES.f0.min, Math.min(RANGES.f0.max, band[1]));
  if (t === null || !(t.f0 > 0)) return null;
  const n = rec.length;
  const w = new Float64Array(n);
  let sw = 0;
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    sw += w[i];
  }
  const out = Float64Array.from(rec);
  const lo = band[0] / Math.SQRT2;
  const hi = Math.min(band[1] * Math.SQRT2, 0.45 * fs);
  for (let m = 1; m * t.f0 <= hi; m++) {
    const f = m * t.f0;
    if (f < lo) continue;
    const dw = (2 * Math.PI * f) / fs;
    const rc = Math.cos(dw);
    const rs = Math.sin(dw);
    // Rotating phasor e^{jω·i} (renormalised now and then) instead of a cos/sin per sample.
    let zr = 1;
    let zi = 0;
    let ar = 0;
    let ai = 0;
    for (let i = 0; i < n; i++) {
      ar += w[i] * resid[i] * zr;
      ai -= w[i] * resid[i] * zi;
      const tr = zr * rc - zi * rs;
      zi = zr * rs + zi * rc;
      zr = tr;
      if ((i & 4095) === 4095) {
        const g = 1 / Math.hypot(zr, zi);
        zr *= g;
        zi *= g;
      }
    }
    // The stationary component: x[i] ≈ (2/Σw)·Re{(ar + j·ai)·e^{jω·i}}.
    const cr = (2 * ar) / sw;
    const ci = (2 * ai) / sw;
    zr = 1;
    zi = 0;
    for (let i = 0; i < n; i++) {
      out[i] -= cr * zr - ci * zi;
      const tr = zr * rc - zi * rs;
      zi = zr * rs + zi * rc;
      zr = tr;
      if ((i & 4095) === 4095) {
        const g = 1 / Math.hypot(zr, zi);
        zr *= g;
        zi *= g;
      }
    }
  }
  return Float32Array.from(out);
}

/**
 * Measures one output channel → mic path with exponential sweeps: optional preview 20 dB lower (stops
 * on a silent input or when the real sweep would saturate it), `repetitions` sweeps analysed by the
 * analyzer and combined (median delay, sign agreement, spread ≤ 0.5 samples), the valid IRs averaged
 * and divided by the master gain, then windowed around the first arrival (stored IR) and turned into
 * the response on the calibration grid. Never throws: failures come back with ok=false and an Italian reason.
 */
export async function measurePath(measure: MeasureFn, analyzer: Analyzer, o: MeasurePathOptions): Promise<PathResult> {
  const fail = (reason: string, aborted = false): PathResult =>
    aborted ? { ...emptyPath(), ok: false, reason, aborted } : { ...emptyPath(), ok: false, reason };
  const fs = o.fs;
  if (!(fs > 0) || !Number.isFinite(fs)) return fail('Frequenza di campionamento non valida.');
  const band = normaliseBand(o.band, fs);
  if (!band) return fail('Banda di misura non valida.');
  const level = Number.isFinite(o.levelDbfs) ? Math.min(0, o.levelDbfs) : -20;
  const amp = dbToAmp(level);
  const reps = Math.round(Math.min(MAX_REPETITIONS, Math.max(1, Number.isFinite(o.repetitions) ? o.repetitions : 3)));
  const sweepS = o.sweepS !== undefined && o.sweepS > 0 && o.sweepS <= 30 ? o.sweepS : SWEEP_S;
  const postS = o.postRollS !== undefined && o.postRollS >= 0 && o.postRollS <= 10 ? o.postRollS : POST_ROLL_S;
  const post = Math.round(postS * fs);
  const mode = o.mode === 'loopback' ? 'loopback' : 'acoustic';
  const micName = o.mic === 'err' ? "d'errore" : 'di riferimento';

  const take = async (stimulus: Float32Array): Promise<{ rec: Float32Array; masterLin: number }> => {
    if (o.signal?.aborted) throw new MeasureFailure('Misura annullata.', true);
    // A copy: a MeasureFn may transfer the buffer it is given, and the sweep is reused.
    const r = await measure(stimulus.slice(), o.channel, post);
    if (r.aborted || o.signal?.aborted) throw new MeasureFailure('Misura annullata.', true);
    const rec = o.mic === 'err' ? r.err : r.ref;
    if (rec === null) {
      throw new MeasureFailure(
        `Microfono d'errore assente: collegalo all'ingresso ${o.errorInput ?? 2} o usa la calibrazione a due posizioni.`,
      );
    }
    if (!(r.masterLin > 0) || !Number.isFinite(r.masterLin)) {
      throw new MeasureFailure('Il volume master è a zero: alzalo prima di misurare.');
    }
    const peak = peakAbs(rec);
    if (peak < SILENT_PEAK) {
      throw new MeasureFailure(`Nessun segnale dal microfono ${micName}: controlla collegamenti e guadagno d'ingresso.`);
    }
    if (peak >= CLIP_PEAK) {
      throw new MeasureFailure(
        `Il microfono ${micName} è in saturazione: abbassa il guadagno d'ingresso o il livello dello stimolo.`,
      );
    }
    return { rec, masterLin: r.masterLin };
  };

  try {
    const sweep = exponentialSweep({ fs, f1: band[0], f2: band[1], durationS: sweepS, fadeMs: SWEEP_FADE_MS, amplitude: amp });
    /** The preview's IR (output → mic, divided by the master gain) for the two-level check. */
    let quiet: Float64Array | null = null;
    if (o.previewFirst) {
      const k = dbToAmp(PREVIEW_DB);
      const preview = Float32Array.from(sweep, (v) => v * k);
      const t = await take(preview);
      const est = await analyzer.estimateDelay(t.rec, preview, fs, band[0], band[1], mode);
      // The real sweep is 20 dB louder: its contribution scales, the ambient part does not.
      const predicted = peakGain(est.ir, est.peak !== 0 ? est.delaySamples : -1, fs, band) * amp + peakAbs(t.rec);
      if (predicted >= SATURATION_PEAK) {
        throw new MeasureFailure(
          `Con il livello scelto il microfono ${micName} andrebbe in saturazione: abbassa il livello dello ` +
            "stimolo, il volume dell'amplificatore o il guadagno d'ingresso.",
        );
      }
      // The deconvolution is by the preview itself: its IR is the path at the quieter level (cleaned of
      // ambient tones like the real sweeps, which a 20 dB quieter sweep would otherwise mistake for gain).
      let q = est;
      for (let pass = 0; pass < TONE_PASSES && o.removeTones !== false && q.ok; pass++) {
        const clean = await withoutTones(t.rec, preview, q, analyzer, fs, band);
        if (clean === null) break;
        q = await analyzer.estimateDelay(clean, preview, fs, band[0], band[1], mode);
      }
      if (q.ok && q.qualityDb >= LEVEL_CHECK_MIN_QUALITY_DB && q.ir.length > 0) quiet = scaled(q.ir, 1 / t.masterLin);
    }
    const estimates: DelayEstimate[] = [];
    const irs: Float64Array[] = [];
    for (let r = 0; r < reps; r++) {
      const t = await take(sweep);
      let est = await analyzer.estimateDelay(t.rec, sweep, fs, band[0], band[1], mode);
      // Each pass fits the tones on a residual computed with a cleaner IR, so the fit improves.
      for (let pass = 0; pass < TONE_PASSES && o.removeTones !== false; pass++) {
        const clean = await withoutTones(t.rec, sweep, est, analyzer, fs, band);
        if (clean === null) break;
        est = await analyzer.estimateDelay(clean, sweep, fs, band[0], band[1], mode);
      }
      estimates.push(est);
      if (est.ok) irs.push(scaled(est.ir, 1 / t.masterLin));
    }
    const comb = combineEstimates(estimates);
    if (irs.length === 0 || comb.count === 0) return fail(comb.reason ?? 'Nessuna misura valida: ripeti la misura.');
    const ir = average(irs);
    const { win, offset: irOffset } = irWindow(ir, comb.delaySamples, fs);
    const freqs = calibrationFreqs(fs, band);
    const path: PathData = {
      delaySamples: comb.delaySamples,
      sign: comb.sign,
      qualityDb: comb.qualityDb,
      stdSamples: comb.stdSamples,
      freqs: [],
      re: [],
      im: [],
      ir: store(win),
      irOffset,
    };
    const rt60 = estimateRt60(ir, comb.delaySamples, fs);
    if (rt60 !== null) path.rt60S = Number(rt60.toPrecision(3));
    const resp = pathResponse(path, fs, freqs);
    path.freqs = store(freqs);
    path.re = store(resp.re);
    path.im = store(resp.im);
    if (!comb.ok) return { ...path, ok: false, reason: comb.reason };
    const levelDependence =
      quiet !== null && quiet.length === ir.length ? levelCheck(quiet, irs, ir, comb.delaySamples, fs, band) : null;
    return levelDependence !== null ? { ...path, ok: false, reason: levelDependence } : { ...path, ok: true };
  } catch (e) {
    if (e instanceof MeasureFailure) return fail(e.message, e.aborted);
    return fail(`Misura non riuscita: ${e instanceof Error ? e.message : String(e)}`);
  }
}

/**
 * Two-level check: the in-band gain of the quiet preview against the real sweeps (their average `ir`),
 * as the least-squares complex ratio over the inner band. A linear path gives 0 dB; OS "enhancements",
 * loudness equalisation or an input AGC do not. The preview is 20 dB quieter, so its ambient noise is
 * 10× larger relative to the path: the spread of the same ratio across the real repetitions, scaled by
 * that, is its uncertainty, and the difference must exceed LEVEL_GAIN_SIGMAS of it as well as
 * LEVEL_GAIN_TOL_DB. Needs ≥ 2 repetitions. Italian reason, or null.
 */
function levelCheck(quiet: Float64Array, reps: Float64Array[], ir: Float64Array, arrival: number, fs: number, band: [number, number]): string | null {
  if (reps.length < 2) return null;
  const lo = band[0] * Math.pow(2, 1 / 6);
  const hi = Math.max(lo * 1.01, band[1] * Math.pow(2, -1 / 6));
  const freqs = Float64Array.from({ length: 32 }, (_, i) => lo * Math.pow(hi / lo, i / 31));
  const resp = (x: Float64Array): { re: Float64Array; im: Float64Array } => irToResponse(irWindow(x, arrival, fs).win, fs, freqs);
  const ref = resp(ir);
  let den = 0;
  for (let k = 0; k < freqs.length; k++) den += ref.re[k] * ref.re[k] + ref.im[k] * ref.im[k];
  if (!(den > 0)) return null;
  /** |Σ X·conj(ref)| / Σ|ref|². */
  const ratio = (x: Float64Array): number => {
    const r = resp(x);
    let nr = 0;
    let ni = 0;
    for (let k = 0; k < freqs.length; k++) {
      nr += r.re[k] * ref.re[k] + r.im[k] * ref.im[k];
      ni += r.im[k] * ref.re[k] - r.re[k] * ref.im[k];
    }
    return Math.hypot(nr, ni) / den;
  };
  const g = ratio(quiet);
  if (!(g > 0)) return null;
  const gs = reps.map(ratio);
  const mean = gs.reduce((u, v) => u + v, 0) / gs.length;
  const sd = Math.sqrt(gs.reduce((u, v) => u + (v - mean) ** 2, 0) / (gs.length - 1));
  const gainDb = 20 * Math.log10(g);
  const sigmaDb = (20 / Math.LN10) * ((sd * dbToAmp(-PREVIEW_DB)) / g);
  if (Math.abs(gainDb) <= Math.max(LEVEL_GAIN_TOL_DB, LEVEL_GAIN_SIGMAS * sigmaDb)) return null;
  return (
    `Il percorso dipende dal livello (il guadagno cambia di ${fmt(gainDb)} dB tra l'anteprima, ${fmt(-PREVIEW_DB, 0)} dB ` +
    "più bassa, e la misura): il sistema operativo o l'interfaccia stanno elaborando l'audio. Su Windows disattiva " +
    "«Miglioramenti audio» ed «Equalizzazione volume» (o usa la modalità esclusiva) e spegni ogni controllo " +
    'automatico del guadagno, poi ripeti la misura.'
  );
}

function scaled(x: Float64Array, k: number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * k;
  return out;
}

function average(list: Float64Array[]): Float64Array {
  const n = Math.min(...list.map((a) => a.length));
  const out = new Float64Array(n);
  for (const a of list) for (let i = 0; i < n; i++) out[i] += a[i];
  for (let i = 0; i < n; i++) out[i] /= list.length;
  return out;
}

/**
 * Primary path reference → error mic from ambient noise (anti-noise muted): Welch H1 with coherence and
 * the reference PSD on the calibration grid, plus the dominant tone's f0 (null without a clear tone).
 * Throws CalibrationError (Italian message) when the recording is unusable.
 */
export async function measurePrimary(
  capture: CaptureFn,
  analyzer: Analyzer,
  o: { fs: number; seconds: number; band: [number, number]; signal?: AbortSignal },
): Promise<PrimaryData> {
  const band = normaliseBand(o.band, o.fs);
  if (!band || !(o.fs > 0)) throw new CalibrationError('Banda di misura non valida.');
  const seconds = Number.isFinite(o.seconds) ? Math.min(60, Math.max(1, o.seconds)) : 4;
  throwIfAborted(o.signal);
  const { ref, err } = await capture(seconds);
  throwIfAborted(o.signal);
  if (err === null) {
    throw new CalibrationError("Microfono d'errore assente: serve per misurare il rumore nella zona.");
  }
  if (peakAbs(ref) < SILENT_PEAK) throw new CalibrationError('Il microfono di riferimento non riceve segnale.');
  if (peakAbs(err) < SILENT_PEAK) throw new CalibrationError("Il microfono d'errore non riceve segnale.");
  if (peakAbs(ref) >= CLIP_PEAK || peakAbs(err) >= CLIP_PEAK) {
    throw new CalibrationError("Un microfono è in saturazione durante la misura del rumore: abbassa il guadagno d'ingresso.");
  }
  const freqs = calibrationFreqs(o.fs, band);
  const t = await analyzer.transfer(ref, err, o.fs, freqs);
  const f0 = await analyzer.estimateF0(
    ref.length > F0_EXCERPT_S * o.fs ? ref.subarray(ref.length - Math.round(F0_EXCERPT_S * o.fs)) : ref,
    o.fs,
    Math.max(RANGES.f0.min, band[0]),
    Math.min(RANGES.f0.max, band[1]),
  );
  throwIfAborted(o.signal);
  const primary: PrimaryData = {
    freqs: store(freqs),
    re: store(t.re),
    im: store(t.im),
    coherence: store(t.coherence),
    tonalF0: f0 !== null && Number.isFinite(f0.f0) ? f0.f0 : null,
  };
  if (t.pxx !== undefined && t.pxx.length === freqs.length) primary.psd = store(t.pxx);
  return primary;
}

/**
 * Signature of an electrical loopback (cable still connected): a clean, repeatable arrival (quality
 * ≥ 18 dB, spread ≤ 0.5 samples), |H| flat within 3 dB and a phase that is a pure delay within ±20°
 * over the inner band (1/6 octave in from each edge) and — when `latency` is given — a delay equal to
 * the loopback latency within 0.5 ms. A speaker in a room fails the flatness and phase tests
 * (low-frequency roll-off, reflections).
 */
export function looksLikeLoopback(path: PathData, latency?: { seconds: number; fs: number }): boolean {
  if (!(path.qualityDb >= LOOPBACK_QUALITY_DB) || !(path.stdSamples <= 0.5)) return false;
  if (latency !== undefined && latency.fs > 0 && Number.isFinite(latency.seconds)) {
    if (Math.abs(path.delaySamples / latency.fs - latency.seconds) > LOOPBACK_LATENCY_TOL_S) return false;
  }
  const n = path.freqs.length;
  if (n < 8 || path.re.length !== n || path.im.length !== n) return false;
  // With fs known the measured delay is removed before unwrapping, so long latencies unwrap safely
  // (without it the grid spacing limits the unwrap to delays below ~0.12 s).
  const knownDelayS = latency !== undefined && latency.fs > 0 ? path.delaySamples / latency.fs : 0;
  const lo = path.freqs[0] * Math.pow(2, 1 / 6);
  const hi = path.freqs[n - 1] * Math.pow(2, -1 / 6);
  const f: number[] = [];
  const ph: number[] = [];
  let minDb = Infinity;
  let maxDb = -Infinity;
  for (let k = 0; k < n; k++) {
    if (path.freqs[k] < lo || path.freqs[k] > hi) continue;
    const m = Math.hypot(path.re[k], path.im[k]);
    if (!(m > 0)) return false;
    const db = 20 * Math.log10(m);
    minDb = Math.min(minDb, db);
    maxDb = Math.max(maxDb, db);
    let p = Math.atan2(path.im[k], path.re[k]) + 2 * Math.PI * path.freqs[k] * knownDelayS;
    if (ph.length > 0) p -= 2 * Math.PI * Math.round((p - ph[ph.length - 1]) / (2 * Math.PI));
    f.push(path.freqs[k]);
    ph.push(p);
  }
  if (f.length < 8 || maxDb - minDb > LOOPBACK_FLAT_DB) return false;
  // Least-squares line through the unwrapped phase: a pure delay leaves no residual.
  const mf = f.reduce((a, b) => a + b, 0) / f.length;
  const mp = ph.reduce((a, b) => a + b, 0) / ph.length;
  let sff = 0;
  let sfp = 0;
  for (let i = 0; i < f.length; i++) {
    sff += (f[i] - mf) ** 2;
    sfp += (f[i] - mf) * (ph[i] - mp);
  }
  const slope = sff > 0 ? sfp / sff : 0;
  const tol = (LOOPBACK_PHASE_DEG * Math.PI) / 180;
  for (let i = 0; i < f.length; i++) if (Math.abs(ph[i] - mp - slope * (f[i] - mf)) > tol) return false;
  return true;
}

/**
 * Stored paths that look like an electrical loopback (cable still connected) rather than an acoustic
 * path, as Italian names ("cassa L → microfono di riferimento"); empty for null. The plan's check
 * before every un-mute (the controller refuses it while this is not empty).
 */
export function loopbackSuspectPaths(cal: CalibrationData | null): string[] {
  if (cal === null) return [];
  const lat = cal.loopbackLatencyS !== null ? { seconds: cal.loopbackLatencyS, fs: cal.fs } : undefined;
  const out: string[] = [];
  for (const ch of [0, 1] as const) {
    const fb = cal.feedback[ch];
    if (fb !== null && looksLikeLoopback(fb, lat)) out.push(`cassa ${CH_NAME[ch]} → microfono di riferimento`);
    const sec = cal.secondary[ch];
    if (sec !== null && looksLikeLoopback(sec, lat)) out.push(`cassa ${CH_NAME[ch]} → microfono nella zona`);
  }
  return out;
}

// --------------------------------------------------------------- solution

const noResult = (): AlignmentResult => {
  const unused = (): ChannelSolution => ({ use: false, delaySamples: D_MIN_SAMPLES, gainLin: 0, polarityInvert: true });
  return { channels: [unused(), unused()], predictedAttenuationDb: 0, causal: true, requiredExtraLeadSeconds: 0 };
};

/**
 * Italian advice when a channel needs a gain outside AMP_ADVICE_DB, else null. Too much gain means the
 * reference mic delivers little signal for the noise in the zone: the fix is its input gain (preamp)
 * on the interface, never the amplifier, which wizard step 2 set so that −20 dBFS matches the noise in
 * the zone (raising it would eat that safety margin). Too little gain: the amplifier is too loud.
 */
export function amplifierAdvice(ch: 0 | 1, gainDb: number): string | null {
  if (!Number.isFinite(gainDb)) return null;
  const name = CH_NAME[ch];
  if (gainDb > AMP_ADVICE_DB.max) {
    const by = Math.max(1, Math.round(gainDb - AMP_ADVICE_DB.target));
    return (
      `Canale ${name}: serve un guadagno di ${fmt(gainDb)} dB, sopra ${fmt(AMP_ADVICE_DB.max, 0)} dB (poco margine): ` +
      `il microfono di riferimento riceve poco segnale. Alza di circa ${by} dB il guadagno d'ingresso ` +
      `(preamplificatore) del microfono di riferimento sull'interfaccia, lasciando i picchi sotto −6 dBFS, poi ` +
      `ripeti la misura dei percorsi e l'allineamento. Non alzare l'amplificatore: il livello impostato al ` +
      `passo 2 è il margine di sicurezza.`
    );
  }
  if (gainDb < AMP_ADVICE_DB.min) {
    const by = Math.max(1, Math.round(AMP_ADVICE_DB.target - gainDb));
    return (
      `Canale ${name}: serve un guadagno di soli ${fmt(gainDb)} dB, sotto ${fmt(AMP_ADVICE_DB.min, 0)} dB: ` +
      `l'amplificatore è troppo alto. Abbassalo di circa ${by} dB e ripeti la misura dei percorsi.`
    );
  }
  return null;
}

/**
 * Converts the solver's totals into channel parameters: delayMs = (delay − D_MIN)/fs with Δ = 0,
 * gainDb = 20·log10(total gain) − master, polarity from the sign; channels the solution does not use
 * are disabled. Values are clamped to the parameter ranges (with advice when that changes them) and
 * amplifierAdvice() applies when the channel gain falls outside −30…−6 dB (above: the reference mic's
 * input gain; below: the amplifier knob).
 */
export function solutionToParams(
  params: EngineParams,
  result: AlignmentResult,
  fs: number,
): { params: EngineParams; advice: string[] } {
  const p = clampParams(params);
  const advice: string[] = [];
  const gainMax = p.gainUnlocked ? RANGES.gainDb.maxUnlocked : RANGES.gainDb.max;
  let masterWarned = false;
  for (const ch of [0, 1] as const) {
    const s = result.channels[ch];
    const c = p.channels[ch];
    const name = CH_NAME[ch];
    if (!s.use || !(s.gainLin > 0) || !Number.isFinite(s.gainLin) || !Number.isFinite(s.delaySamples)) {
      c.enabled = false;
      continue;
    }
    let delayMs = ((s.delaySamples - D_MIN_SAMPLES) / fs) * 1000;
    if (delayMs > MAX_DELAY_MS) {
      advice.push(`Canale ${name}: servirebbe un ritardo di ${fmt(delayMs, 0)} ms, oltre il massimo di ${MAX_DELAY_MS} ms.`);
      delayMs = MAX_DELAY_MS;
    }
    c.delayMs = Math.max(0, delayMs);
    c.phaseDeltaDeg = 0;
    c.polarityInvert = s.polarityInvert;
    c.enabled = true;
    if (p.masterDb <= MASTER_MIN_DB) {
      // The channel gain is total − master: with the master at −∞ it is undefined, keep the old one.
      if (!masterWarned) advice.push("Il volume master è a zero: alzalo e ripeti l'allineamento per calcolare i guadagni.");
      masterWarned = true;
      continue;
    }
    let gainDb = 20 * Math.log10(s.gainLin) - p.masterDb;
    const amp = amplifierAdvice(ch, gainDb);
    if (amp !== null) advice.push(amp);
    if (gainDb > gainMax) {
      advice.push(
        `Canale ${name}: guadagno limitato a ${fmt(gainMax, 0)} dB (ne servirebbero ${fmt(gainDb)}): la cancellazione sarà parziale.`,
      );
      gainDb = gainMax;
    } else if (gainDb < RANGES.gainDb.min) {
      advice.push(`Canale ${name}: guadagno limitato a ${fmt(RANGES.gainDb.min, 0)} dB (ne servirebbero ${fmt(gainDb)}).`);
      gainDb = RANGES.gainDb.min;
    }
    c.gainDb = gainDb;
  }
  return { params: clampParams(p), advice };
}

/** H_cond as the solver must see it: in Hilbert mode Δ = 0 leaves path A of the rotator in the chain. */
function conditioningFn(p: EngineParams, fs: number): (f: number) => Complex {
  if (p.phaseMode !== 'hilbert') return (f) => conditioningResponse(p, fs, f);
  return (f) => {
    const h = conditioningResponse(p, fs, f);
    const a = hilbertResponse(f, fs).a;
    return { re: h.re * a.re - h.im * a.im, im: h.re * a.im + h.im * a.re };
  };
}

function interpReal(freqs: Float64Array, v: Float64Array, f: number): number {
  const n = freqs.length;
  if (n === 0) return 0;
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

/** Feedback response left after neutralisation with a FIR: F − F̂. */
function minusFir(fb: FreqResponse, fir: { bulkDelay: number; taps: Float64Array }, fs: number): FreqResponse {
  const r = irToResponse(fir.taps, fs, fb.freqs);
  const re = new Float64Array(fb.freqs.length);
  const im = new Float64Array(fb.freqs.length);
  for (let k = 0; k < fb.freqs.length; k++) {
    const a = (-2 * Math.PI * fb.freqs[k] * fir.bulkDelay) / fs;
    const c = Math.cos(a);
    const s = Math.sin(a);
    re[k] = fb.re[k] - (r.re[k] * c - r.im[k] * s);
    im[k] = fb.im[k] - (r.re[k] * s + r.im[k] * c);
  }
  return { freqs: fb.freqs, re, im };
}

export interface LoopGains {
  /** Without neutralisation, its maximum frequency. */
  rawDb: number;
  atHz: number;
  /** After neutralisation with the FIRs feedbackFirs(cal, taps) designs (the ones the app loads). */
  afterDb: number;
}

/** Open-loop gain of `p` through the measured feedback paths, raw and neutralised; null without data. */
function loopGains(cal: CalibrationData, p: EngineParams, fs: number, taps: number): LoopGains | null {
  if (cal.fs !== fs || (cal.feedback[0] === null && cal.feedback[1] === null)) return null;
  const freqs = calibrationFreqs(fs, cal.band);
  if (freqs.length === 0) return null;
  const fb: [FreqResponse | null, FreqResponse | null] = [
    cal.feedback[0] ? pathResponse(cal.feedback[0], fs, freqs) : null,
    cal.feedback[1] ? pathResponse(cal.feedback[1], fs, freqs) : null,
  ];
  const g: [(f: number) => Complex, (f: number) => Complex] = [
    (f) => channelResponse(p, fs, f, 0),
    (f) => channelResponse(p, fs, f, 1),
  ];
  const raw = loopGainDb(g, fb, cal.band);
  const firs = feedbackFirs(cal, taps);
  const resid: [FreqResponse | null, FreqResponse | null] = [
    fb[0] && firs[0] ? minusFir(fb[0], firs[0], fs) : fb[0],
    fb[1] && firs[1] ? minusFir(fb[1], firs[1], fs) : fb[1],
  ];
  return { rawDb: raw.maxDb, atHz: raw.atHz, afterDb: loopGainDb(g, resid, cal.band).maxDb };
}

/**
 * Open-loop gain of `p` with both channels through the measured feedback paths (they add at the
 * reference mic), raw and after neutralisation with `taps`-tap FIRs; null without feedback data,
 * at another sample rate or when it cannot be computed. The check the solvers make.
 */
export function checkLoop(cal: CalibrationData, p: EngineParams, fs: number, taps: number): LoopGains | null {
  const l = loopGains(cal, p, fs, taps);
  return l !== null && Number.isFinite(l.rawDb) ? l : null;
}

/**
 * Rounding allowance of capLoopGain, dB: a loop that a solve (or an earlier cap) left at the margin is
 * within it, and a cap lands half of this below the margin, never a rounding error above it. Capping
 * twice therefore changes nothing the second time.
 */
const LOOP_CAP_TOL_DB = 1e-7;

/**
 * The stability margin the solvers enforce, for parameters that did not come from a solve (autotune
 * result, shorter neutraliser, loaded preset): when the neutralised loop of `p` (checkLoop with `taps`)
 * exceeds LOOP_GAIN_DANGER_DB, every channel gain is lowered by the excess (the loop scales with them,
 * clamped to the gain range). Returns the params, the dB taken off (0: unchanged) and the loop of the
 * params returned (null without feedback data or at another rate: nothing to check).
 */
export function capLoopGain(
  cal: CalibrationData,
  p: EngineParams,
  fs: number,
  taps: number,
): { params: EngineParams; limitedDb: number; loop: LoopGains | null } {
  const loop = checkLoop(cal, p, fs, taps);
  if (loop === null || !(loop.afterDb > LOOP_GAIN_DANGER_DB + LOOP_CAP_TOL_DB)) return { params: p, limitedDb: 0, loop };
  const limitedDb = loop.afterDb - LOOP_GAIN_DANGER_DB + LOOP_CAP_TOL_DB / 2;
  const q = clampParams(p);
  for (const ch of [0, 1] as const) q.channels[ch].gainDb -= limitedDb;
  const params = clampParams(q);
  return { params, limitedDb, loop: checkLoop(cal, params, fs, taps) };
}

const tapsAdvice = (taps: number): string =>
  taps < FEEDBACK_TAPS_RANGE.max ? ` o aumenta i tap del neutralizzatore (Avanzate, ora ${taps})` : '';

/** Italian advice on a loop that is left as it is (no gain change). */
function loopAdvice(l: LoopGains, taps: number, advice: string[]): void {
  if (!(l.afterDb > LOOP_GAIN_LIMIT_DB) && !(l.rawDb > LOOP_GAIN_LIMIT_DB)) return;
  const at = `${fmt(l.atHz, 0)} Hz`;
  if (l.afterDb > LOOP_GAIN_DANGER_DB) {
    advice.push(
      `Rischio di innesco: il guadagno d'anello cassa → microfono di riferimento è ${fmt(l.rawDb)} dB a ${at} e ` +
        `resta ${fmt(l.afterDb)} dB anche con la neutralizzazione del feedback. Allontana o orienta la cassa ` +
        'lontano dal microfono di riferimento, oppure riduci il guadagno.',
    );
  } else if (l.afterDb > LOOP_GAIN_LIMIT_DB) {
    advice.push(
      `Guadagno d'anello ${fmt(l.rawDb)} dB a ${at}: con la neutralizzazione del feedback scende a ` +
        `${fmt(l.afterDb)} dB, sopra il limite di ${fmt(LOOP_GAIN_LIMIT_DB, 0)} dB. Margine di stabilità ridotto: ` +
        `allontana la cassa dal microfono di riferimento${tapsAdvice(taps)}.`,
    );
  } else {
    advice.push(
      `Guadagno d'anello ${fmt(l.rawDb)} dB a ${at}: la neutralizzazione del feedback, applicata automaticamente ` +
        `dopo la misura dei percorsi (${taps} tap), lo porta a ${fmt(l.afterDb)} dB.`,
    );
  }
}

function causalityAdvice(result: AlignmentResult, extraS: number, tempC: number, advice: string[]): void {
  const s = Math.max(extraS, result.requiredExtraLeadSeconds);
  if (!(s > 0)) return;
  advice.push(
    `Il rumore arriva troppo tardi al microfono di riferimento: servirebbero altri ${fmt(s * 1000)} ms di ` +
      `anticipo (≈ ${fmt(s * speedOfSound(tempC), 2)} m più a monte, verso la sorgente). Sposta il microfono, ` +
      'riduci la latenza oppure, se il rumore è periodico, usa la modalità tonale.',
  );
}

/**
 * What the solver aligned (on its own frequencies), for the attenuation the returned params give:
 * `weights` are the reference PSD (error mic) or |H_cond|² (geometry), `coherence` the primary
 * coherence (null: the model is exact).
 */
interface SolveModel {
  freqs: Float64Array;
  primary: FreqResponse;
  secondary: [FreqResponse | null, FreqResponse | null];
  weights: Float64Array;
  coherence: Float64Array | null;
  band: [number, number];
}

/**
 * Attenuation at the error mic with the channel responses of `p`. Residual R = P + Σ G_ch·S_ch per
 * frequency. With the coherence γ² of the primary measurement, the error-mic power per frequency is
 * Gyy = |P|²·Gxx/γ², of which only the coherent part γ²·Gyy can be cancelled:
 * A = −10·log10(Σ (|R|²·Gxx + (1 − γ²)·Gyy) / Σ Gyy). Without coherence: Σ w|R|² / Σ w|P|².
 */
function predictedAttenuationDb(m: SolveModel, p: EngineParams, fs: number): number {
  let num = 0;
  let den = 0;
  for (let k = 0; k < m.freqs.length; k++) {
    const f = m.freqs[k];
    const w = m.weights[k];
    if (f < m.band[0] || f > m.band[1] || !(w > 0)) continue;
    const pr = m.primary.re[k];
    const pi = m.primary.im[k];
    let rr = pr;
    let ri = pi;
    for (const ch of [0, 1] as const) {
      const s = m.secondary[ch];
      if (s === null) continue;
      const g = channelResponse(p, fs, f, ch);
      rr += g.re * s.re[k] - g.im * s.im[k];
      ri += g.re * s.im[k] + g.im * s.re[k];
    }
    const pp = pr * pr + pi * pi;
    const r2 = rr * rr + ri * ri;
    if (m.coherence === null) {
      num += w * r2;
      den += w * pp;
    } else {
      const c = Math.min(1, Math.max(MIN_COHERENCE, m.coherence[k]));
      const gyy = (pp * w) / c;
      num += r2 * w + (1 - c) * gyy;
      den += gyy;
    }
  }
  return den > 0 && num > 0 ? -10 * Math.log10(num / den) : 0;
}

/** Every enabled channel gain lowered by `db`; null when one would fall below the gain range. */
function lowerGains(p: EngineParams, db: number): EngineParams | null {
  const q = clampParams(p);
  for (const ch of [0, 1] as const) {
    const c = q.channels[ch];
    if (!c.enabled) continue;
    if (c.gainDb - db < RANGES.gainDb.min) return null;
    c.gainDb -= db;
  }
  return clampParams(q);
}

/**
 * Converts the solution into params, then enforces the loop's stability margin: the loop through the
 * measured feedback paths, after the neutralisation FIRs the app loads, must stay at or below
 * LOOP_GAIN_DANGER_DB (6 dB of margin) everywhere in band. Above it every channel gain is lowered by
 * the excess (the loop scales with them); when that leaves no useful attenuation the params are not
 * changed. Between LOOP_GAIN_LIMIT_DB (the plan's −20 dB target) and the margin the params stay and the
 * advice says the cancellation is less accurate: near the band edges a FIR of a few ms cannot model a
 * band-limited path, so enforcing −20 dB there would throw away working solutions. The predicted
 * attenuation is recomputed for the params returned (see predictedAttenuationDb).
 */
function finishSolve(
  cal: CalibrationData,
  p: EngineParams,
  fs: number,
  result: AlignmentResult,
  advice: string[],
  model: SolveModel,
  taps: number,
): SolveOutcome {
  const res: AlignmentResult = { ...result, channels: [{ ...result.channels[0] }, { ...result.channels[1] }] };
  const unchanged = (loop: LoopGains | null, limited = 0): SolveOutcome => ({
    params: p,
    result: res,
    advice,
    loopGainDb: loop?.rawDb ?? null,
    loopGainAfterDb: loop?.afterDb ?? null,
    loopLimitedDb: limited,
  });
  if (!(res.predictedAttenuationDb >= MIN_USEFUL_DB)) {
    // Nothing to gain: do not load a meaningless delay/gain into the engine.
    advice.push(
      `Con queste misure non si ottiene attenuazione (prevista ${fmt(res.predictedAttenuationDb)} dB): ` +
        'i parametri non sono stati cambiati.',
    );
    const loop = loopGains(cal, p, fs, taps);
    if (loop !== null) loopAdvice(loop, taps, advice);
    return unchanged(loop);
  }
  const conv = solutionToParams(p, res, fs);
  advice.push(...conv.advice);
  let params = conv.params;
  const loop = loopGains(cal, params, fs, taps);
  let limited = 0;
  if (loop !== null && loop.afterDb > LOOP_GAIN_DANGER_DB) {
    limited = loop.afterDb - LOOP_GAIN_DANGER_DB;
    const lowered = lowerGains(params, limited);
    const predicted = lowered !== null ? predictedAttenuationDb(model, lowered, fs) : 0;
    if (lowered === null || !(predicted >= MIN_USEFUL_DB)) {
      res.predictedAttenuationDb = predicted;
      advice.push(
        `Rischio di innesco: il guadagno d'anello cassa → microfono di riferimento è ${fmt(loop.rawDb)} dB a ` +
          `${fmt(loop.atHz, 0)} Hz e resta ${fmt(loop.afterDb)} dB anche con la neutralizzazione del feedback. Per ` +
          `scendere a ${fmt(LOOP_GAIN_DANGER_DB, 0)} dB i guadagni andrebbero ridotti di ${fmt(limited)} dB e l'anti-rumore ` +
          'non attenuerebbe più: i parametri non sono stati cambiati. Allontana o orienta la cassa lontano dal ' +
          `microfono di riferimento${tapsAdvice(taps)}, poi ripeti la misura dei percorsi.`,
      );
      return unchanged(loop, limited);
    }
    params = lowered;
    const k = dbToAmp(-limited);
    for (const c of res.channels) c.gainLin *= k;
    advice.push(
      `Guadagno d'anello cassa → microfono di riferimento ${fmt(loop.rawDb)} dB a ${fmt(loop.atHz, 0)} Hz, ` +
        `${fmt(loop.afterDb)} dB dopo la neutralizzazione del feedback: sopra ${fmt(LOOP_GAIN_DANGER_DB, 0)} dB non resta ` +
        `margine di stabilità. Guadagni dei canali ridotti di ${fmt(limited)} dB; per recuperare attenuazione allontana o ` +
        `orienta la cassa lontano dal microfono di riferimento${tapsAdvice(taps)}.`,
    );
  } else if (loop !== null) {
    loopAdvice(loop, taps, advice);
  }
  res.predictedAttenuationDb = predictedAttenuationDb(model, params, fs);
  if (!(res.predictedAttenuationDb >= MIN_USEFUL_DB)) {
    advice.push(
      `Con i valori applicabili non si ottiene attenuazione (prevista ${fmt(res.predictedAttenuationDb)} dB): ` +
        'i parametri non sono stati cambiati.',
    );
    return unchanged(loop, limited);
  }
  if (res.channels.some((c) => c.use) && res.predictedAttenuationDb < WEAK_ATTENUATION_DB) {
    advice.push(
      `Attenuazione prevista solo ${fmt(res.predictedAttenuationDb)} dB: controlla posizionamento, ` +
        'banda dei filtri e causalità prima di attivare l\'uscita.',
    );
  }
  // The loop scales with the gains: the values of the params returned.
  return {
    params,
    result: res,
    advice,
    loopGainDb: loop !== null ? loop.rawDb - limited : null,
    loopGainAfterDb: loop !== null ? loop.afterDb - limited : null,
    loopLimitedDb: limited,
  };
}

const unusable = (p: EngineParams, msg: string): SolveOutcome => ({ params: p, result: noResult(), advice: [msg], loopGainDb: null });

const activeChannels = (s: SpeakerConfig): (0 | 1)[] => (s === 'single' ? [0] : [0, 1]);

const tapsOf = (o: SolveOptions | undefined): number =>
  o?.feedbackTaps !== undefined && Number.isFinite(o.feedbackTaps) ? o.feedbackTaps : DEFAULT_FEEDBACK_TAPS;

/**
 * Bulk delay (s) of a stored response: argmax_τ |Σ w·H·e^{+jωτ}| over ±PRIMARY_BULK_MAX_S, on a grid
 * of 1/(4·f_max) refined by golden section. The calibration grid's jitter keeps aliases down.
 */
function bulkDelayS(h: FreqResponse, w: Float64Array, band: [number, number]): number {
  const idx: number[] = [];
  let fMax = 0;
  for (let k = 0; k < h.freqs.length; k++) {
    if (h.freqs[k] < band[0] || h.freqs[k] > band[1] || !(w[k] > 0)) continue;
    idx.push(k);
    fMax = Math.max(fMax, h.freqs[k]);
  }
  if (idx.length < 2 || !(fMax > 0)) return 0;
  const step = 1 / (4 * fMax);
  const m = Math.ceil(PRIMARY_BULK_MAX_S / step);
  const accRe = new Float64Array(2 * m + 1);
  const accIm = new Float64Array(2 * m + 1);
  for (const k of idx) {
    const om = 2 * Math.PI * h.freqs[k];
    // v = w·H·e^{jω·τ0} at τ0 = −m·step, rotated by e^{jω·step} per grid point.
    const a0 = -om * m * step;
    let vr = w[k] * (h.re[k] * Math.cos(a0) - h.im[k] * Math.sin(a0));
    let vi = w[k] * (h.re[k] * Math.sin(a0) + h.im[k] * Math.cos(a0));
    const zr = Math.cos(om * step);
    const zi = Math.sin(om * step);
    for (let j = 0; j <= 2 * m; j++) {
      accRe[j] += vr;
      accIm[j] += vi;
      const t = vr * zr - vi * zi;
      vi = vr * zi + vi * zr;
      vr = t;
    }
  }
  let best = 0;
  let bestV = -1;
  for (let j = 0; j <= 2 * m; j++) {
    const v = accRe[j] * accRe[j] + accIm[j] * accIm[j];
    if (v > bestV) {
      bestV = v;
      best = j;
    }
  }
  const score = (tau: number): number => {
    let re = 0;
    let im = 0;
    for (const k of idx) {
      const a = 2 * Math.PI * h.freqs[k] * tau;
      const c = Math.cos(a);
      const s = Math.sin(a);
      re += w[k] * (h.re[k] * c - h.im[k] * s);
      im += w[k] * (h.re[k] * s + h.im[k] * c);
    }
    return re * re + im * im;
  };
  // Golden-section refinement inside the neighbouring grid cells.
  let lo = (best - m - 1) * step;
  let hi = (best - m + 1) * step;
  const g = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - g * (hi - lo);
  let x2 = lo + g * (hi - lo);
  let f1 = score(x1);
  let f2 = score(x2);
  for (let i = 0; i < 40; i++) {
    if (f1 > f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - g * (hi - lo);
      f1 = score(x1);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + g * (hi - lo);
      f2 = score(x2);
    }
  }
  return (lo + hi) / 2;
}

/**
 * A stored primary path at other frequencies (the harmonics in tonal mode). Interpolating the stored
 * phase directly fails once the path delay exceeds ~1/(2·grid step) ≈ 0.13 s (the phase turns by
 * more than π between neighbours, at any distance of the source the tonal mode must work): the bulk
 * delay is removed first, the remainder interpolated, and the delay re-applied at the new frequencies.
 */
function primaryAt(pf: FreqResponse, w: Float64Array, band: [number, number], at: Float64Array): FreqResponse {
  const tau = bulkDelayS(pf, w, band);
  const rot = (fr: { re: Float64Array; im: Float64Array }, freqs: Float64Array, sign: 1 | -1): FreqResponse => {
    const re = new Float64Array(freqs.length);
    const im = new Float64Array(freqs.length);
    for (let k = 0; k < freqs.length; k++) {
      const a = sign * 2 * Math.PI * freqs[k] * tau;
      const c = Math.cos(a);
      const s = Math.sin(a);
      re[k] = fr.re[k] * c - fr.im[k] * s;
      im[k] = fr.re[k] * s + fr.im[k] * c;
    }
    return { freqs, re, im };
  };
  return rot(sampleResponse(rot(pf, pf.freqs, 1), at), at, -1);
}

/**
 * Error-mic (or two-position) solution: the measured primary path (weighted by the reference PSD ×
 * coherence) against the measured secondary paths, including H_cond of the current filters (and of the
 * resonator bank at the harmonics of tonal.f0 in tonal mode); converts the totals into ChannelParams
 * (see solutionToParams), enforces the loop-gain limit through the measured feedback paths (see
 * finishSolve) and predicts the attenuation including the incoherent part of the zone noise. In tonal
 * mode, a tone measured off tonal.f0 by more than the resonators' half bandwidth makes it unusable.
 */
export function solveFromCalibration(cal: CalibrationData, params: EngineParams, fs: number, opts?: SolveOptions): SolveOutcome {
  const p = clampParams(params);
  if (cal.fs !== fs) {
    return unusable(
      p,
      `La calibrazione è stata fatta a ${fmt(cal.fs / 1000, 1)} kHz, ora l'audio funziona a ${fmt(fs / 1000, 1)} kHz: ripeti la misura.`,
    );
  }
  const prim = cal.primary;
  if (prim === null || prim.freqs.length === 0) {
    return unusable(p, "Manca la misura del rumore al microfono d'errore (percorso primario).");
  }
  const chans = activeChannels(cal.speakers);
  if (chans.every((ch) => cal.secondary[ch] === null)) {
    return unusable(p, "Manca la misura del percorso cassa → microfono d'errore.");
  }
  const advice: string[] = [];
  const pf = storedResponse(prim);
  const psd = Float64Array.from(prim.freqs, (_, k) => {
    const v = prim.psd !== undefined ? prim.psd[k] : 1;
    return Number.isFinite(v) && v > 0 ? v : 0;
  });
  const coh = Float64Array.from(prim.coherence, (c) => (Number.isFinite(c) && c > 0 ? Math.min(1, c) : 0));
  const w0 = Float64Array.from(coh, (c, k) => c * psd[k]);
  const tonalF0 = p.phaseMode === 'tonal' ? p.tonal.f0 : undefined;
  let primary: FreqResponse = pf;
  let weights = w0;
  let gxx = psd;
  let coherence = coh;
  if (tonalF0 !== undefined) {
    if (prim.tonalF0 !== null) {
      // The resonator at harmonic h passes h·f0 ± h·f0/(2Q): a tone outside it is not reproduced.
      const h = Math.max(1, Math.round(prim.tonalF0 / tonalF0));
      if (Math.abs(prim.tonalF0 - h * tonalF0) > (h * tonalF0) / (2 * p.tonal.q)) {
        return unusable(
          p,
          `Il tono misurato ora è a ${fmt(prim.tonalF0, 2)} Hz, ma la modalità tonale è su ${fmt(tonalF0, 2)} Hz: i ` +
            "risonatori non lo riproducono. Aggiorna f0 («Aggancia al picco») e ripeti l'allineamento.",
        );
      }
    }
    const freqs = harmonicFreqs(tonalF0, cal.band, fs);
    if (freqs.length === 0) return unusable(p, `Nessuna armonica di ${fmt(tonalF0, 2)} Hz nella banda misurata.`);
    primary = primaryAt(pf, w0, cal.band, freqs);
    weights = Float64Array.from(freqs, (f) => interpReal(pf.freqs, w0, f));
    gxx = Float64Array.from(freqs, (f) => interpReal(pf.freqs, psd, f));
    coherence = Float64Array.from(freqs, (f) => interpReal(pf.freqs, coh, f));
  }
  let wSum = 0;
  let cSum = 0;
  let cW = 0;
  for (let k = 0; k < pf.freqs.length; k++) {
    if (pf.freqs[k] < cal.band[0] || pf.freqs[k] > cal.band[1]) continue;
    cSum += coh[k] * psd[k];
    cW += psd[k];
  }
  for (const w of weights) wSum += w;
  if (!(wSum > 0)) return unusable(p, "Il rumore al microfono d'errore non è correlato con il riferimento: nulla da cancellare.");
  if (cW > 0 && cSum / cW < LOW_COHERENCE) {
    advice.push(
      `Coerenza media ${fmt(cSum / cW, 2)} tra riferimento ed errore: buona parte del rumore nella zona non ` +
        'arriva dal microfono di riferimento e non può essere cancellata.',
    );
  }
  const secondary: [FreqResponse | null, FreqResponse | null] = [null, null];
  for (const ch of chans) {
    const s = cal.secondary[ch];
    if (s !== null) secondary[ch] = pathResponse(s, fs, primary.freqs);
  }
  let result: AlignmentResult;
  try {
    result = solveAlignment({
      fs,
      band: cal.band,
      primary,
      secondary,
      conditioning: conditioningFn(p, fs),
      speakers: cal.speakers,
      weights,
      tonalF0,
      maxDelaySamples: (MAX_DELAY_MS / 1000) * fs,
    });
  } catch (e) {
    return unusable(p, `Allineamento non riuscito: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (tonalF0 === undefined && !result.causal) causalityAdvice(result, 0, cal.tempC, advice);
  if (prim.tonalF0 !== null && p.phaseMode !== 'tonal') {
    advice.push(`Il rumore ha un tono dominante a ${fmt(prim.tonalF0, 2)} Hz: la modalità tonale può attenuarlo di più.`);
  }
  const model: SolveModel = { freqs: primary.freqs, primary, secondary, weights: gxx, coherence, band: cal.band };
  return finishSolve(cal, p, fs, result, advice, model, tapsOf(opts));
}

/**
 * Geometry-only solution (no error mic): primary = pure delay from headStartSeconds (far source) with
 * the gain from the two-position levels when known, secondary = the measured speaker → zone path or
 * latency + speaker distance with the chain polarity measured on the speaker → reference path (the
 * virtual zone mic has the reference mic's polarity). The solver weights the band by |H_cond|². A source
 * more than ~45° off the mic → zone line (cos θ < 0.7) makes the head start unreliable: the result is
 * then reported as not causal, with advice, instead of claiming causality.
 */
export function solveFromGeometry(cal: CalibrationData, params: EngineParams, fs: number, latencyS: number, opts?: SolveOptions): SolveOutcome {
  const p = clampParams(params);
  const g = cal.geometry;
  if (g === null) return unusable(p, 'Inserisci le distanze e la direzione della sorgente (assistente geometria).');
  const advice: string[] = [];
  const tempC = Number.isFinite(g.tempC) ? g.tempC : cal.tempC;
  const c = speedOfSound(tempC);
  const head = headStartSeconds({ micToZoneM: g.micToZoneM, sourceAngleDeg: g.sourceAngleDeg, tempC });
  const band = normaliseBand(cal.band, fs);
  if (!band) return unusable(p, 'Banda di calibrazione non valida.');
  const { zoneDb, refDb } = cal.levels;
  let gainP = 1;
  if (zoneDb !== null && refDb !== null && Number.isFinite(zoneDb) && Number.isFinite(refDb)) {
    gainP = Math.pow(10, (zoneDb - refDb) / 20);
  } else {
    advice.push(
      'Livelli del rumore nella zona e al riferimento non misurati: il guadagno è solo stimato ' +
        '(usa la calibrazione a due posizioni).',
    );
  }
  const tonalF0 = p.phaseMode === 'tonal' ? p.tonal.f0 : undefined;
  const freqs = tonalF0 !== undefined ? harmonicFreqs(tonalF0, band, fs) : calibrationFreqs(fs, band);
  if (freqs.length === 0) return unusable(p, 'Nessuna frequenza utile nella banda di calibrazione.');
  const dP = head * fs;
  const primary: FreqResponse = {
    freqs,
    re: Float64Array.from(freqs, (f) => gainP * Math.cos((-2 * Math.PI * f * dP) / fs)),
    im: Float64Array.from(freqs, (f) => gainP * Math.sin((-2 * Math.PI * f * dP) / fs)),
  };
  const chans = activeChannels(cal.speakers);
  const sameFs = cal.fs === fs;
  const lat = Number.isFinite(latencyS) && latencyS > 0 ? latencyS : 0;
  const secondary: [FreqResponse | null, FreqResponse | null] = [null, null];
  let secDelayS = 0;
  let synthetic = false;
  let polarityUnknown = false;
  for (const ch of chans) {
    const s = cal.secondary[ch];
    if (s !== null && sameFs) {
      secondary[ch] = pathResponse(s, fs, freqs);
      secDelayS = Math.max(secDelayS, s.delaySamples / fs);
    } else {
      const d = Number.isFinite(g.speakerToZoneM[ch]) ? Math.max(0, g.speakerToZoneM[ch]) : 0;
      const ds = (lat + d / c) * fs;
      // An inverting amplifier, swapped speaker wires or a pin-3-hot mic invert the whole chain: the
      // first arrival on the speaker → reference path carries that sign.
      const fb = cal.feedback[ch];
      const sign = fb !== null ? fb.sign : 1;
      if (fb === null) polarityUnknown = true;
      secondary[ch] = {
        freqs,
        re: Float64Array.from(freqs, (f) => sign * Math.cos((-2 * Math.PI * f * ds) / fs)),
        im: Float64Array.from(freqs, (f) => sign * Math.sin((-2 * Math.PI * f * ds) / fs)),
      };
      secDelayS = Math.max(secDelayS, ds / fs);
      synthetic = true;
    }
  }
  if (synthetic) {
    advice.push(
      "Percorso cassa → zona non misurato: uso latenza + distanza della cassa e un guadagno nominale; regola il " +
        "guadagno all'ascolto.",
    );
  }
  if (polarityUnknown) {
    advice.push(
      'Polarità della catena non verificata (manca la misura cassa → microfono di riferimento): se l\'amplificatore o ' +
        'i cavi invertono il segnale, l\'anti-rumore si somma al rumore. Misura i percorsi (passo 5) prima di attivarlo.',
    );
  }
  const cond = conditioningFn(p, fs);
  const weights = Float64Array.from(freqs, (f) => {
    const h = cond(f);
    return h.re * h.re + h.im * h.im;
  });
  let result: AlignmentResult;
  try {
    result = solveAlignment({
      fs,
      band,
      primary,
      secondary,
      conditioning: cond,
      speakers: cal.speakers,
      weights,
      tonalF0,
      maxDelaySamples: (MAX_DELAY_MS / 1000) * fs,
    });
  } catch (e) {
    return unusable(p, `Allineamento non riuscito: ${e instanceof Error ? e.message : String(e)}`);
  }
  const fc = tonalF0 ?? Math.sqrt(band[0] * Math.min(band[1], p.filters.lpEnabled ? p.filters.lpFreq : band[1]));
  const gdS = groupDelaySamples(cond, fc, fs) / fs;
  const geo = geometryDelay(head, secDelayS, gdS, tempC, g.sourceAngleDeg);
  if (geo.warnAngle) {
    result.causal = false;
    advice.push(
      `La sorgente è a ${fmt(g.sourceAngleDeg, 0)}° dalla linea microfono → zona (cos θ = ${fmt(geo.cosTheta, 2)}, ` +
        'sotto 0,7): il vantaggio acustico calcolato dipende troppo dalla direzione e la causalità non è garantita. ' +
        "Metti il microfono di riferimento sulla linea sorgente → zona o verifica con il microfono d'errore.",
    );
  }
  if (tonalF0 === undefined && (!geo.causal || !result.causal) && !geo.warnAngle) {
    result.causal = false;
    causalityAdvice(result, geo.causal ? 0 : geo.requiredExtraLeadM / speedOfSound(tempC), tempC, advice);
  }
  advice.push(
    "Senza microfono d'errore il risultato non è misurabile: verifica all'ascolto e, se puoi, rifinisci con il " +
      "microfono d'errore e l'autotune.",
  );
  const model: SolveModel = { freqs, primary, secondary, weights, coherence: null, band };
  return finishSolve(cal, p, fs, result, advice, model, tapsOf(opts));
}

/**
 * Lags of the earliest and latest arrivals of a feedback IR (lags ≥ 0): local peaks of |IR| within
 * FIR_ARRIVAL_DB of the strongest one, FIR_NOISE_PEAK σ above the diffuse part of the untapered window
 * [from, to) (σ from the median |IR|: the arrivals are sparse; the path's qualityDb would count an early
 * arrival as noise), whose lobe is the kernel's (a band-edge skirt is much wider); always including
 * `arrival` (the estimator's first arrival within −6 dB).
 */
function arrivalSpan(ir: Float64Array, arrival: number, from: number, to: number): { first: number; last: number } {
  const n = ir.length;
  let max = 0;
  let iMax = 0;
  for (let k = 0; k < n; k++) {
    const v = Math.abs(ir[k]);
    if (v > max) {
      max = v;
      iMax = k;
    }
  }
  const at = Math.max(0, Math.min(n - 1, Math.floor(arrival)));
  if (!(max > 0)) return { first: at, last: at };
  const lobe = (k: number): number => {
    const half = Math.abs(ir[k]) / 2;
    let w = 1;
    while (k - w > 0 && w < n && Math.abs(ir[k - w]) > half) w++;
    return w;
  };
  const maxLobe = FIR_LOBE_RATIO * lobe(iMax) + 2;
  const a = Math.max(0, Math.min(n, from));
  const b = Math.max(a, Math.min(n, to));
  const mags = Float64Array.from(ir.subarray(a, b), Math.abs).sort();
  const sigma = mags.length > 0 ? mags[mags.length >> 1] / 0.6745 : 0;
  const thr = Math.max(max * dbToAmp(FIR_ARRIVAL_DB), FIR_NOISE_PEAK * sigma);
  let first = at;
  let last = at;
  for (let k = 1; k < n - 1; k++) {
    const v = Math.abs(ir[k]);
    if (v < thr || v < Math.abs(ir[k - 1]) || v < Math.abs(ir[k + 1]) || lobe(k) > maxLobe) continue;
    first = Math.min(first, k);
    last = Math.max(last, k);
  }
  return { first, last };
}

/**
 * Feedback-neutralisation FIRs (final output → reference mic) from the stored feedback IRs. A FIR of
 * `taps` taps starts just before the first arrival; when the stored IR has other strong arrivals (a
 * direct sound weaker than a later reflection, a reflection beyond the window) the FIR is lengthened to
 * cover them all — from the earliest to `taps` past the latest — up to FEEDBACK_MAX_TAPS (beyond that
 * it keeps the strongest arrival and the loop-gain check accounts for what is left).
 */
export function feedbackFirs(
  cal: CalibrationData,
  taps: number,
): [{ bulkDelay: number; taps: Float64Array } | null, { bulkDelay: number; taps: Float64Array } | null] {
  const n = Math.round(Math.min(FEEDBACK_MAX_TAPS, Math.max(16, Number.isFinite(taps) ? taps : DEFAULT_FEEDBACK_TAPS)));
  const one = (path: PathData | null): { bulkDelay: number; taps: Float64Array } | null => {
    if (path === null || path.ir.length === 0) return null;
    // designFeedbackFir indexes the IR by lag ≥ 0: rebuild it with the window at its offset (the
    // pre-arrival part at negative lags cannot be predicted causally anyway).
    const end = path.irOffset + path.ir.length;
    if (end <= 0) return null;
    const full = new Float64Array(end);
    for (let i = 0; i < path.ir.length; i++) {
      const lag = path.irOffset + i;
      if (lag >= 0) full[lag] = path.ir[i];
    }
    const span = arrivalSpan(
      full,
      path.delaySamples,
      path.irOffset + Math.round(IR_PRE_TAPER_S * cal.fs),
      end - Math.round(IR_POST_TAPER_S * cal.fs),
    );
    const len = Math.floor(span.last) - Math.floor(span.first) + n;
    if (len <= FEEDBACK_MAX_TAPS) return designFeedbackFir(full, span.first, Math.max(n, len));
    // Too far apart for one FIR: as long as possible, from the earliest arrival that still keeps the
    // strongest one (the estimator's arrival) inside.
    const start = Math.max(Math.floor(span.first), Math.floor(path.delaySamples) + n - FEEDBACK_MAX_TAPS);
    return designFeedbackFir(full, start, FEEDBACK_MAX_TAPS);
  };
  return [one(cal.feedback[0]), one(cal.feedback[1])];
}

// ----------------------------------------------------------- sanitising

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

const numIn = (v: unknown, min: number, max: number): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max ? v : undefined;

/** Plain number array from an array or typed array of finite numbers, else undefined. */
function finiteArray(v: unknown, maxLen = MAX_STORED_POINTS): number[] | undefined {
  let src: ArrayLike<unknown>;
  if (Array.isArray(v)) src = v;
  else if (ArrayBuffer.isView(v) && !(v instanceof DataView)) src = v as unknown as ArrayLike<unknown>;
  else return undefined;
  if (src.length > maxLen) return undefined;
  const out = new Array<number>(src.length);
  for (let i = 0; i < src.length; i++) {
    const x = src[i];
    if (typeof x !== 'number' || !Number.isFinite(x)) return undefined;
    out[i] = x;
  }
  return out;
}

function increasingPositive(f: number[]): boolean {
  for (let i = 0; i < f.length; i++) if (!(f[i] > 0) || (i > 0 && !(f[i] > f[i - 1]))) return false;
  return true;
}

/** A pair [a, b] (array or typed array) mapped by `item`; undefined if anything is invalid. */
function pair<T>(v: unknown, item: (x: unknown) => T | undefined): [T, T] | undefined {
  if (!(Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView)))) return undefined;
  const a = v as unknown as ArrayLike<unknown>;
  if (a.length !== 2) return undefined;
  const x = item(a[0]);
  const y = item(a[1]);
  return x === undefined || y === undefined ? undefined : [x, y];
}

function sanitizePath(v: unknown): PathData | null | undefined {
  if (v === null) return null;
  if (!isRecord(v)) return undefined;
  const delaySamples = numIn(v.delaySamples, 0, 1e8);
  const sign = v.sign === 1 || v.sign === -1 ? v.sign : undefined;
  const qualityDb = numIn(v.qualityDb, -1000, 1000);
  const stdSamples = numIn(v.stdSamples, 0, 1e8);
  const freqs = finiteArray(v.freqs);
  const re = finiteArray(v.re);
  const im = finiteArray(v.im);
  const ir = finiteArray(v.ir);
  const irOffset = numIn(v.irOffset, -1e8, 1e8);
  if (
    delaySamples === undefined ||
    sign === undefined ||
    qualityDb === undefined ||
    stdSamples === undefined ||
    freqs === undefined ||
    re === undefined ||
    im === undefined ||
    ir === undefined ||
    irOffset === undefined ||
    !Number.isInteger(irOffset) ||
    re.length !== freqs.length ||
    im.length !== freqs.length ||
    !increasingPositive(freqs)
  ) {
    return undefined;
  }
  const out: PathData = { delaySamples, sign, qualityDb, stdSamples, freqs, re, im, ir, irOffset };
  if (v.rt60S !== undefined) {
    const rt = numIn(v.rt60S, 0, 20);
    if (rt === undefined) return undefined;
    out.rt60S = rt;
  }
  return out;
}

function sanitizePrimary(v: unknown): PrimaryData | null | undefined {
  if (v === null || v === undefined) return null;
  if (!isRecord(v)) return undefined;
  const freqs = finiteArray(v.freqs);
  const re = finiteArray(v.re);
  const im = finiteArray(v.im);
  const coherence = finiteArray(v.coherence);
  const tonalF0 = v.tonalF0 === null || v.tonalF0 === undefined ? null : numIn(v.tonalF0, 1, 1e5);
  if (freqs === undefined || re === undefined || im === undefined || coherence === undefined || tonalF0 === undefined) {
    return undefined;
  }
  const n = freqs.length;
  if (re.length !== n || im.length !== n || coherence.length !== n || !increasingPositive(freqs)) return undefined;
  const out: PrimaryData = { freqs, re, im, coherence: coherence.map((c) => Math.min(1, Math.max(0, c))), tonalF0 };
  if (v.psd !== undefined) {
    const psd = finiteArray(v.psd);
    if (psd === undefined || psd.length !== n || psd.some((x) => x < 0)) return undefined;
    out.psd = psd;
  }
  return out;
}

function nullableNum(v: unknown, min: number, max: number): number | null | undefined {
  return v === null || v === undefined ? null : numIn(v, min, max);
}

/**
 * Validates an untrusted calibration (e.g. from a preset): returns a deep plain-JSON copy (typed
 * arrays become number arrays) or null if the version is not 1 or anything is missing, inconsistent
 * or not finite. Absent nullable fields (loopback, geometry, levels, primary) read as null.
 */
export function sanitizeCalibration(raw: unknown): CalibrationData | null {
  if (!isRecord(raw) || raw.version !== 1) return null;
  const fs = numIn(raw.fs, 8000, 768000);
  if (fs === undefined) return null;
  const band = pair(raw.band, (x) => numIn(x, 1e-3, fs / 2));
  const tempC = numIn(raw.tempC, -50, 70);
  const createdAt = numIn(raw.createdAt, 0, 1e15);
  const feedback = pair(raw.feedback, sanitizePath);
  const secondary = pair(raw.secondary, sanitizePath);
  const primary = sanitizePrimary(raw.primary);
  const loopbackLatencyS = nullableNum(raw.loopbackLatencyS, 0, 10);
  const speakers = raw.speakers === 'single' || raw.speakers === 'shared' || raw.speakers === 'separate' ? raw.speakers : undefined;
  if (
    band === undefined ||
    !(band[1] > band[0]) ||
    tempC === undefined ||
    createdAt === undefined ||
    feedback === undefined ||
    secondary === undefined ||
    primary === undefined ||
    loopbackLatencyS === undefined ||
    speakers === undefined
  ) {
    return null;
  }
  let geometry: GeometryData | null = null;
  if (raw.geometry !== null && raw.geometry !== undefined) {
    const g = raw.geometry;
    if (!isRecord(g)) return null;
    const micToZoneM = numIn(g.micToZoneM, 0, 1000);
    const sourceAngleDeg = numIn(g.sourceAngleDeg, -360, 360);
    const speakerToZoneM = pair(g.speakerToZoneM, (x) => numIn(x, 0, 1000));
    const gt = numIn(g.tempC, -50, 70);
    if (micToZoneM === undefined || sourceAngleDeg === undefined || speakerToZoneM === undefined || gt === undefined) return null;
    geometry = { micToZoneM, sourceAngleDeg, speakerToZoneM, tempC: gt };
  }
  let levels: CalibrationData['levels'] = { zoneDb: null, refDb: null };
  if (raw.levels !== null && raw.levels !== undefined) {
    if (!isRecord(raw.levels)) return null;
    const zoneDb = nullableNum(raw.levels.zoneDb, -400, 100);
    const refDb = nullableNum(raw.levels.refDb, -400, 100);
    if (zoneDb === undefined || refDb === undefined) return null;
    levels = { zoneDb, refDb };
  }
  return {
    version: 1,
    fs,
    band,
    tempC,
    createdAt,
    feedback,
    secondary,
    primary,
    loopbackLatencyS,
    geometry,
    levels,
    speakers,
    ...(raw.refChannel === 0 || raw.refChannel === 1 ? { refChannel: raw.refChannel } : {}),
  };
}
