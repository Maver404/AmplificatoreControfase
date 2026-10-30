/**
 * Calibration: measurement orchestration (sweeps played by the worklet, analysis by an Analyzer),
 * the plain-JSON calibration record stored in presets, and the conversion of the alignment solution
 * into channel parameters.
 *
 * Pure TypeScript with no DOM access: the browser wizard, the analysis worker (welchTransfer) and the
 * Node closed-loop simulator all run this file unmodified.
 */
import { harmonicFactors, sampleResponse, solveAlignment } from '../dsp/alignment.ts';
import type { AlignmentResult, ChannelSolution, FreqResponse, SpeakerConfig } from '../dsp/alignment.ts';
import { FEEDBACK_MAX_TAPS } from '../dsp/antiphase-engine.ts';
import type { Complex } from '../dsp/complex.ts';
import type { F0Estimate } from '../dsp/f0-tracker.ts';
import { designFeedbackFir } from '../dsp/feedback-canceller.ts';
import { nextPow2, sharedFFT } from '../dsp/fft.ts';
import { trimForTarget } from '../dsp/harmonic-trims.ts';
import { hilbertResponse } from '../dsp/hilbert-pair.ts';
import { combineEstimates, irToResponse } from '../dsp/latency-estimator.ts';
import type { DelayEstimate } from '../dsp/latency-estimator.ts';
import { D_MIN_SAMPLES, MASTER_MIN_DB, MAX_DELAY_MS, MAX_HARMONICS as MAX_TRIMS, RANGES, clampParams, withoutTrims } from '../dsp/params.ts';
import type { EngineParams } from '../dsp/params.ts';
import { geometryDelay, headStartSeconds, speedOfSound } from '../dsp/physics.ts';
import { channelResponse, channelResponseAt, conditioningResponse, groupDelaySamples } from '../dsp/response.ts';
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
   * reference does not hear (coherence): it is the attenuation the error mic can expect. When the params
   * were left unchanged it is the prediction refused (below 1 dB, the figure the advice quotes): such an
   * outcome is never useful (logic.ts usefulOutcome).
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
  /**
   * Extension, tonal mode: the harmonics the trims act on (1 … tonal.harmonics, inside the calibration band and
   * below 0.45·fs), with the prediction for the params returned. Absent outside tonal mode and when the
   * params were left unchanged.
   */
  harmonics?: HarmonicSolution[];
  /**
   * Extension, tonal mode: the delay/gain/polarity solution alone (no trims, no centring raise) under its own
   * stability margin, and its predicted attenuation: what the trims are compared with (result.predictedAttenuationDb
   * is never below it; HarmonicSolution.untrimmedDb is its prediction per harmonic). Absent outside tonal mode, when
   * the params were left unchanged and when the margin refused the delay alone.
   */
  delayOnly?: { params: EngineParams; predictedDb: number };
  /**
   * Extension: result.predictedAttenuationDb is at the ceiling of the solve model (predictionCeiling), not a
   * precise figure. Absent otherwise.
   */
  predictionCeiling?: PredictionCeiling;
}

/** One harmonic of a tonal solution (SolveOutcome.harmonics). */
export interface HarmonicSolution {
  /** Harmonic number (ChannelParams.harmonicTrims[h − 1]) and its frequency h·f0, Hz. */
  h: number;
  freqHz: number;
  /**
   * Predicted attenuation at the error mic at this harmonic with the params returned (trims, clamping and the
   * loop-gain limit included; with an error mic also the zone noise the reference does not hear), dB.
   */
  predictedDb: number;
  /**
   * The same for the delay/gain/polarity solution alone (no trims, no centring raise) under its own stability margin
   * (its own cuts: uniform, or of the loop at this harmonic), dB: the fair comparison (SolveOutcome.delayOnly).
   * predictedDb is below it only where giving this harmonic that response back would take the loop over the margin
   * (and neither lowering this harmonic's trims into the margin predicts more there and in total, nor lowering every
   * gain instead predicts more in total by at least the cut, GIVE_BACK_CUT_GAIN_DB = 0.05 dB at most) or needs a trim
   * outside RANGES.trimGainDb. In separate zones played together (step 6) a zone may play its delay alone instead.
   */
  untrimmedDb: number;
  /** The trim this harmonic needed was outside RANGES.trimGainDb and was clamped: partial cancellation. */
  capped: boolean;
  /**
   * dB taken off this harmonic's trims (on the channels that feed back into the reference mic) so that the
   * neutralised loop stays at LOOP_GAIN_DANGER_DB around its frequency; absent when none. The rest of the
   * margin, taken off every channel gain, is SolveOutcome.loopLimitedDb.
   */
  loopLimitedDb?: number;
  /** predictedDb is at the ceiling of the solve model (predictionCeiling), not a precise figure; absent otherwise. */
  ceiling?: PredictionCeiling;
}

/**
 * The ceiling a prediction reached (predictionCeiling): 'geometry' — without an error mic, the accuracy limit
 * of the model (GEOMETRY_ACCURACY_DB); 'exact' — with one, an exact null of the measured model, which the
 * calculation stops at EXACT_NULL_DB.
 */
export type PredictionCeiling = 'geometry' | 'exact';

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
/**
 * Without an error mic the solve model (geometric primary) has no measured incoherent part: a residual floor
 * of GEOMETRY_ACCURACY_DB below the noise at every frequency (a few % of model error: a couple of cm of
 * distance at 400 Hz) bounds its predictions at about that many dB, instead of the ~300 dB of an exact null
 * of the model (per-harmonic trims null it exactly).
 */
export const GEOMETRY_ACCURACY_DB = 30;
const GEOMETRY_FLOOR = Math.pow(10, -GEOMETRY_ACCURACY_DB / 10);
/**
 * A measured coherence is never taken as exactly 1: an exact null of the measured model (the per-harmonic
 * trims make one) then predicts 60 dB, not an infinite attenuation (or 0 dB, read off a zero residual).
 * Real Welch estimates stay well below it.
 */
const MAX_COHERENCE = 1 - 1e-6;
/** The prediction of an exact null of the measured model: −10·log10(1 − MAX_COHERENCE) ≈ 60 dB. */
export const EXACT_NULL_DB = -10 * Math.log10(1 - MAX_COHERENCE);
/**
 * Within this of its ceiling a prediction is the ceiling: the model's own residual is at most ~12 % of the
 * floor that bounds it, so the figure says how far the model reaches, not how much the room will give.
 */
export const PREDICTION_CEILING_TOL_DB = 0.5;
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
/** A gain cut for the advice: «meno di 0,1» where one decimal would read 0,0 (a give-back's cut of a few 1e-5 dB). */
const fmtCut = (db: number): string => (db < 0.05 ? 'meno di 0,1' : fmt(db));
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
  return pathResponseAt(path, fs)(freqs);
}

/** pathResponse as a function of the frequencies, the stored IR converted once (evaluated again and again by the loop checks). */
function pathResponseAt(path: PathData, fs: number): (freqs: Float64Array) => FreqResponse {
  if (path.ir.length === 0) {
    const stored = storedResponse(path);
    return (freqs) => {
      const s = sampleResponse(stored, freqs);
      return { freqs, re: s.re, im: s.im };
    };
  }
  const ir = Float64Array.from(path.ir);
  return (freqs) => {
    const r = irToResponse(ir, fs, freqs);
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
  };
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

/**
 * Frequencies the loop gain is checked at: the calibration grid and, in tonal mode, the centres of the
 * harmonics the bank reproduces. There the channel response peaks (with its trim) in a band of h·f0/Q, at
 * Q 30 narrower than the grid's 2 Hz cells: the grid alone can miss the top of the fundamental's peak by
 * several dB.
 */
function loopFreqs(cal: CalibrationData, p: EngineParams, fs: number): Float64Array {
  const grid = calibrationFreqs(fs, cal.band);
  const q = clampParams(p);
  if (q.phaseMode !== 'tonal' || grid.length === 0) return grid;
  const band = normaliseBand(cal.band, fs);
  if (band === null) return grid;
  const peaks = Array.from(harmonicFreqs(q.tonal.f0, band, fs)).filter((f) => Math.round(f / q.tonal.f0) <= q.tonal.harmonics);
  const all = [...grid, ...peaks].sort((a, b) => a - b);
  return Float64Array.from(all.filter((f, i) => i === 0 || f - all[i - 1] > 1e-9 * f));
}

type FeedbackPair = [FreqResponse | null, FreqResponse | null];

/**
 * The measured feedback paths at any frequencies, raw and after neutralisation with the FIRs
 * feedbackFirs(cal, taps) designs (F − F̂, designed once); null without feedback data or at another rate.
 */
function feedbackAt(cal: CalibrationData, fs: number, taps: number): ((freqs: Float64Array) => { raw: FeedbackPair; resid: FeedbackPair }) | null {
  if (cal.fs !== fs || (cal.feedback[0] === null && cal.feedback[1] === null)) return null;
  const firs = feedbackFirs(cal, taps);
  const paths = [cal.feedback[0] ? pathResponseAt(cal.feedback[0], fs) : null, cal.feedback[1] ? pathResponseAt(cal.feedback[1], fs) : null];
  return (freqs) => {
    const raw: FeedbackPair = [paths[0]?.(freqs) ?? null, paths[1]?.(freqs) ?? null];
    const resid: FeedbackPair = [
      raw[0] && firs[0] ? minusFir(raw[0], firs[0], fs) : raw[0],
      raw[1] && firs[1] ? minusFir(raw[1], firs[1], fs) : raw[1],
    ];
    return { raw, resid };
  };
}

/** feedbackAt at `freqs`; null without feedback data, at another rate or without frequencies. */
function neutralisedFeedback(cal: CalibrationData, freqs: Float64Array, fs: number, taps: number): { raw: FeedbackPair; resid: FeedbackPair } | null {
  const at = freqs.length > 0 ? feedbackAt(cal, fs, taps) : null;
  return at !== null ? at(freqs) : null;
}

/** Open-loop gain of `p` through the measured feedback paths, raw and neutralised (loopChecker); null without data. */
function loopGains(cal: CalibrationData, p: EngineParams, fs: number, taps: number): LoopGains | null {
  return loopChecker(cal, p, fs, taps)?.gains(p) ?? null;
}

/** Both channels' responses of some params as functions of f (response.ts channelResponseAt). */
type ChannelResponses = [(f: number) => Complex, (f: number) => Complex];

/** channelResponse of both channels of `p`, each prepared once for many frequencies. */
const responsesOf = (p: EngineParams, fs: number): ChannelResponses => [channelResponseAt(p, fs, 0), channelResponseAt(p, fs, 1)];

/** The neutralised loop |Σ G·(F − F̂)| of `p` at `freqs`, dB, from the residual paths `resid` on them. */
function loopDbAt(p: EngineParams | ChannelResponses, fs: number, freqs: Float64Array, resid: FeedbackPair): Float64Array {
  return loopsDbAt(Array.isArray(p) ? p : responsesOf(p, fs), freqs, [resid])[0];
}

/** loopDbAt through several pairs of paths at once (raw and neutralised), each channel response evaluated once per frequency. */
function loopsDbAt(resp: ChannelResponses, freqs: Float64Array, pairs: FeedbackPair[]): Float64Array[] {
  const db = pairs.map(() => new Float64Array(freqs.length).fill(-Infinity));
  const used = ([0, 1] as const).map((ch) => pairs.some((x) => x[ch] !== null));
  for (let k = 0; k < freqs.length; k++) {
    const g = ([0, 1] as const).map((ch) => (used[ch] ? resp[ch](freqs[k]) : null));
    pairs.forEach((pair, i) => {
      let re = 0;
      let im = 0;
      for (const ch of [0, 1] as const) {
        const r = pair[ch];
        const v = g[ch];
        if (r === null || v === null || !Number.isFinite(v.re) || !Number.isFinite(v.im)) continue;
        re += v.re * r.re[k] - v.im * r.im[k];
        im += v.re * r.im[k] + v.im * r.re[k];
      }
      const m = Math.hypot(re, im);
      if (m > 0) db[i][k] = 20 * Math.log10(m);
    });
  }
  return db;
}

/** Points per side of a harmonic's band where enforceLoopMargin samples the loop before refining its peak. */
const TRIM_LOOP_POINTS = 8;
/**
 * A sampled loop peak further than this below the margin (a harmonic's band) or below the highest sample (the
 * loop frequencies) is not refined: between samples ≤ 1/16 of a resonator's bandwidth, or 2.5 Hz on the
 * calibration grid (whose feedback paths of ≤ 0.2 s ripple over ≥ 5 Hz), a peak hides well under 3 dB.
 */
const LOOP_REFINE_BELOW_DB = 3;
/** At most this many of the loop frequencies' sampled peaks are refined (the highest). */
const LOOP_REFINE_PEAKS = 8;

/**
 * Highest value of `one` between a and b (a sampled maximum `peak` between them): golden-section search down to
 * 1/2000 of the interval (on a peak a few samples wide that reads its top within 1e-5 dB).
 */
function refinePeak(one: (f: number) => number, a: number, b: number, peak: number): number {
  const g = (Math.sqrt(5) - 1) / 2;
  const stop = (b - a) / 2000;
  let x1 = b - g * (b - a);
  let x2 = a + g * (b - a);
  let f1 = one(x1);
  let f2 = one(x2);
  let top = Math.max(peak, f1, f2);
  for (let i = 0; i < 30 && b - a > stop; i++) {
    if (f1 > f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = b - g * (b - a);
      f1 = one(x1);
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + g * (b - a);
      f2 = one(x2);
    }
    top = Math.max(top, f1, f2);
  }
  return top;
}

/** The neutralised loop at one frequency, dB, of the params whose channel responses are `resp`. */
function loopDbAtOne(resp: ChannelResponses, fs: number, at: (freqs: Float64Array) => { resid: FeedbackPair }, f: number): number {
  const x = Float64Array.of(f);
  return loopDbAt(resp, fs, x, at(x).resid)[0];
}

/** The 2·TRIM_LOOP_POINTS + 1 frequencies bandLoopPeakDb samples across [lo, hi], `centre` in the middle. */
const bandFreqs = (centre: number, lo: number, hi: number): Float64Array =>
  Float64Array.from({ length: 2 * TRIM_LOOP_POINTS + 1 }, (_, i) => {
    const u = (i - TRIM_LOOP_POINTS) / TRIM_LOOP_POINTS;
    return u < 0 ? centre + u * (centre - lo) : centre + u * (hi - centre);
  });

/**
 * Peak of the neutralised loop of `p` across a harmonic's band (see trimLoopBands), dB: the band's samples
 * (bandFreqs, with the residual feedback paths there), then a golden-section search between the neighbours of
 * the highest one (the peak need not sit on the centre when F̃ slopes there, and the loop grid's 2 Hz cells are
 * wider than the top of a Q 30 peak).
 */
function bandLoopPeakDb(
  p: EngineParams,
  fs: number,
  at: (freqs: Float64Array) => { resid: FeedbackPair },
  samples: { freqs: Float64Array; resid: FeedbackPair },
): number {
  const freqs = samples.freqs;
  const resp = responsesOf(p, fs);
  const db = loopDbAt(resp, fs, freqs, samples.resid);
  let best = 0;
  for (let i = 1; i < db.length; i++) if (db[i] > db[best]) best = i;
  if (!(db[best] > LOOP_GAIN_DANGER_DB - LOOP_REFINE_BELOW_DB)) return db[best];
  const a = freqs[Math.max(0, best - 1)];
  const b = freqs[Math.min(freqs.length - 1, best + 1)];
  return refinePeak((f) => loopDbAtOne(resp, fs, at, f), a, b, db[best]);
}

/** Single frequencies whose feedback paths a loopChecker keeps (a few thousand per solve). */
const LOOP_VISITED_MAX = 16384;

/** The loop of params that share f0 and the harmonics (the loop-check frequencies) with those it was made for. */
interface LoopChecker {
  /** Loop gains of `q`: raw at its maximum on the loop frequencies, neutralised at its maximum refined between them. */
  gains(q: EngineParams): LoopGains;
  /** The neutralised peak of `q` across one harmonic's band (bandLoopPeakDb). */
  bandPeak(q: EngineParams, b: { centre: number; lo: number; hi: number }): number;
}

/**
 * The loop check of params like `p` (same f0 and harmonics: the frequencies loopFreqs gives), with the
 * neutralisation FIRs designed and the feedback paths evaluated once. The neutralised maximum is refined
 * between the loop frequencies around the highest sampled peaks (within LOOP_REFINE_BELOW_DB of the highest,
 * LOOP_REFINE_PEAKS at most; golden section): on the calibration grid alone a peak between two harmonics'
 * bands read up to 0.1 dB low, 0.3 dB once separate zones played together, so a loop capped to the grid's
 * maximum still exceeded the margin there. Relative to the highest sample, the result scales exactly with the
 * gains (a cap lands where it aims). The feedback paths at the single frequencies the refinements visit are kept:
 * the golden-section searches of one solve's many checks revisit the same ones (a few thousand per solve, mostly
 * hits; the solves of the review's heaviest seeded plants took half the time). Null without feedback data, at
 * another rate or without frequencies.
 */
function loopChecker(cal: CalibrationData, p: EngineParams, fs: number, taps: number): LoopChecker | null {
  const all = loopFreqs(cal, p, fs);
  const lo = Math.min(cal.band[0], cal.band[1]);
  const hi = Math.max(cal.band[0], cal.band[1]);
  const freqs = all.filter((f) => f >= lo && f <= hi);
  const paths = freqs.length > 0 ? feedbackAt(cal, fs, taps) : null;
  if (paths === null) return null;
  const fb = paths(freqs);
  const visited = new Map<number, { raw: FeedbackPair; resid: FeedbackPair }>();
  const at = (x: Float64Array): { raw: FeedbackPair; resid: FeedbackPair } => {
    if (x.length !== 1) return paths(x);
    let v = visited.get(x[0]);
    if (v === undefined) {
      if (visited.size >= LOOP_VISITED_MAX) visited.clear();
      v = paths(x);
      visited.set(x[0], v);
    }
    return v;
  };
  // Each band's samples (bandLoopPeakDb) and the feedback paths there, evaluated once.
  const bandSamples = new Map<string, { freqs: Float64Array; resid: FeedbackPair }>();
  return {
    gains(q) {
      const resp = responsesOf(q, fs);
      const [raw, after] = loopsDbAt(resp, freqs, [fb.raw, fb.resid]);
      let r = 0;
      let top = -Infinity;
      for (let k = 0; k < freqs.length; k++) {
        if (raw[k] > raw[r]) r = k;
        top = Math.max(top, after[k]);
      }
      const peaks: number[] = [];
      for (let k = 0; k < freqs.length; k++) {
        const v = after[k];
        if ((k === 0 || v >= after[k - 1]) && (k === freqs.length - 1 || v >= after[k + 1]) && v > top - LOOP_REFINE_BELOW_DB) peaks.push(k);
      }
      let peak = top;
      for (const k of peaks.sort((a, b) => after[b] - after[a]).slice(0, LOOP_REFINE_PEAKS)) {
        const a = freqs[Math.max(0, k - 1)];
        const b = freqs[Math.min(freqs.length - 1, k + 1)];
        if (b > a) peak = Math.max(peak, refinePeak((f) => loopDbAtOne(resp, fs, at, f), a, b, after[k]));
      }
      return { rawDb: raw[r], atHz: freqs[r], afterDb: peak };
    },
    bandPeak(q, b) {
      const key = `${b.centre}|${b.lo}|${b.hi}`;
      let x = bandSamples.get(key);
      if (x === undefined) {
        const f = bandFreqs(b.centre, b.lo, b.hi);
        x = { freqs: f, resid: at(f).resid };
        bandSamples.set(key, x);
      }
      return bandLoopPeakDb(q, fs, at, x);
    },
  };
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
 * coherence (null: the model is exact), `feedback` the feedback paths after neutralisation F̃ = F − F̂
 * (null without feedback data: an open loop).
 */
interface SolveModel {
  freqs: Float64Array;
  primary: FreqResponse;
  secondary: [FreqResponse | null, FreqResponse | null];
  weights: Float64Array;
  coherence: Float64Array | null;
  band: [number, number];
  feedback: FeedbackPair | null;
}

/** `m` with the neutralised feedback paths the app loads (feedbackFirs(cal, taps)) on its frequencies. */
const withFeedback = (m: SolveModel, cal: CalibrationData, fs: number, taps: number): SolveModel => ({
  ...m,
  feedback: neutralisedFeedback(cal, m.freqs, fs, taps)?.resid ?? null,
});

/** Below this |1 − L| the closed loop has no steady state to predict (L: the loop gain): the point counts as uncancelled. */
const LOOP_POLE_EPS = 1e-12;

/** The prediction `db` at the ceiling of its model (with or without an error mic), or null (an ordinary figure). */
export function predictionCeiling(db: number, errorMic: boolean): PredictionCeiling | null {
  if (!Number.isFinite(db)) return null;
  if (errorMic) return db >= EXACT_NULL_DB - PREDICTION_CEILING_TOL_DB ? 'exact' : null;
  return db >= GEOMETRY_ACCURACY_DB - PREDICTION_CEILING_TOL_DB ? 'geometry' : null;
}

/**
 * Attenuation at the error mic with the channel responses of `p`. Residual R = P + Σ G_ch·S_ch/(1 − L) per
 * frequency, L = Σ G_ch·F̃_ch the neutralised loop: the reference mic hears the noise plus what the speakers
 * feed back past the neutraliser, so the anti-noise is the open-loop one × 1/(1 − L) (without feedback data
 * L = 0). At the −6 dB margin that is −3.5…+6 dB, enough to undo a null. With the coherence γ² of the
 * primary measurement, the error-mic power per frequency is Gyy = |P|²·Gxx/γ², of which only the coherent
 * part γ²·Gyy can be cancelled: A = −10·log10(Σ (|R|²·Gxx + (1 − γ²)·Gyy) / Σ Gyy). Without coherence (no
 * error mic): Σ w·(|R|² + 10^(−GEOMETRY_ACCURACY_DB/10)·|P|²) / Σ w|P|², at most GEOMETRY_ACCURACY_DB.
 */
function predictedAttenuationDb(m: SolveModel, p: EngineParams, fs: number): number {
  let num = 0;
  let den = 0;
  const resp = responsesOf(p, fs);
  for (let k = 0; k < m.freqs.length; k++) {
    const t = residualTerms(m, resp, k);
    if (t === null) continue;
    num += t.num;
    den += t.den;
  }
  return den > 0 && num > 0 ? -10 * Math.log10(num / den) : 0;
}

/**
 * Residual (num) and noise (den) power of predictedAttenuationDb at grid point k, with the channel responses
 * `resp` (responsesOf the params); null outside the band or weightless.
 */
function residualTerms(m: SolveModel, resp: ChannelResponses, k: number): { num: number; den: number } | null {
  const f = m.freqs[k];
  const w = m.weights[k];
  if (f < m.band[0] || f > m.band[1] || !(w > 0)) return null;
  const pr = m.primary.re[k];
  const pi = m.primary.im[k];
  // Σ G·S (the open-loop anti-noise at the error mic) and 1 − L.
  let ar = 0;
  let ai = 0;
  let dr = 1;
  let di = 0;
  for (const ch of [0, 1] as const) {
    const s = m.secondary[ch];
    const fb = m.feedback?.[ch] ?? null;
    if (s === null && fb === null) continue;
    const g = resp[ch](f);
    if (s !== null) {
      ar += g.re * s.re[k] - g.im * s.im[k];
      ai += g.re * s.im[k] + g.im * s.re[k];
    }
    if (fb !== null) {
      dr -= g.re * fb.re[k] - g.im * fb.im[k];
      di -= g.re * fb.im[k] + g.im * fb.re[k];
    }
  }
  const d2 = dr * dr + di * di;
  let rr = pr;
  let ri = pi;
  if (d2 > LOOP_POLE_EPS * LOOP_POLE_EPS) {
    rr += (ar * dr + ai * di) / d2;
    ri += (ai * dr - ar * di) / d2;
  }
  const pp = pr * pr + pi * pi;
  const r2 = rr * rr + ri * ri;
  // No error mic: the model (pure-delay primary, measured or geometric secondary) is exact only on paper.
  if (m.coherence === null) return { num: w * (r2 + pp * GEOMETRY_FLOOR), den: w * pp };
  const c = Math.min(MAX_COHERENCE, Math.max(MIN_COHERENCE, m.coherence[k]));
  const gyy = (pp * w) / c;
  return { num: r2 * w + (1 - c) * gyy, den: gyy };
}

/** Attenuation of predictedAttenuationDb at grid point k alone, dB (0 when it cannot be computed). */
function pointAttenuationDb(m: SolveModel, p: EngineParams, fs: number, k: number): number {
  const t = residualTerms(m, responsesOf(p, fs), k);
  return t !== null && t.den > 0 && t.num > 0 ? -10 * Math.log10(t.num / t.den) : 0;
}

/** The harmonics h ≤ tonal.harmonics (and ≤ MAX_TRIMS) below 0.45·fs on the solver's grid, with their grid index. */
function trimmedHarmonics(p: EngineParams, m: SolveModel, fs: number): { h: number; k: number }[] {
  const f0 = p.tonal.f0;
  const out: { h: number; k: number }[] = [];
  for (let h = 1; h <= Math.min(p.tonal.harmonics, MAX_TRIMS) && h * f0 < 0.45 * fs; h++) {
    const f = h * f0;
    let k = -1;
    for (let i = 0; i < m.freqs.length && k < 0; i++) if (Math.abs(m.freqs[i] - f) <= 1e-9 * f) k = i;
    if (k >= 0 && m.freqs[k] >= m.band[0] && m.freqs[k] <= m.band[1]) out.push({ h, k });
  }
  return out;
}

/**
 * Tonal mode: per-harmonic trims of the solved params (delay, gain and polarity already chosen) that null
 * the modelled residual at each harmonic (alignment.ts harmonicFactors: one speaker per zone exactly, two on
 * one zone with the minimum change). The trims of different harmonics do not interact, so each is computed
 * on its own from the untrimmed channel response there. The null is the closed loop's (see residualTerms):
 * p + Σ T·b/(1 − Σ T·ℓ) = 0, ℓ_ch = G_ch·F̃_ch, is p + Σ T·(b − p·ℓ) = 0, so the factors are solved against
 * the contributions b − p·ℓ (b alone without feedback data). With `centre`, a channel whose harmonics need more
 * than RANGES.trimGainDb.max gets its gain raised by the excess and its trims lowered by as much, so the
 * trims stay centred — only for harmonics the raise brings into range (as far as the channel gain range
 * allows): raising the gain for a harmonic that stays out of range anyway (a speaker notch) would buy a
 * fraction of a dB there with the whole raise of loop gain at that harmonic. What is still out of range is
 * clamped (capped). A harmonic whose trims would leave a larger modelled residual than none (possible only
 * when clamped) keeps them at 0, as does one without weight (no coherent noise to cancel). A clamp is not
 * reported (capped) on a harmonic whose primary coherence is below LOW_COHERENCE: little of the noise there
 * comes from the reference, so the trim that nulls the model is no statement about the speaker. Returns the
 * params and the capped harmonics.
 */
function solveHarmonicTrims(
  p: EngineParams,
  m: SolveModel,
  speakers: SpeakerConfig,
  fs: number,
  centre: boolean,
): { params: EngineParams; capped: Set<number> } {
  const q = withoutTrims(p);
  const capped = new Set<number>();
  const rows = trimmedHarmonics(q, m, fs).filter((r) => m.weights[r.k] > 0);
  if (rows.length === 0) return { params: q, capped };
  const r = RANGES.trimGainDb;
  const factorsAt = (bounds: { min: number; max: number }): ReturnType<typeof harmonicFactors>[] =>
    rows.map(({ h, k }) => {
      const f = h * q.tonal.f0;
      const pr = m.primary.re[k];
      const pi = m.primary.im[k];
      const b = ([0, 1] as const).map((ch): Complex | null => {
        const s = m.secondary[ch];
        if (s === null || !q.channels[ch].enabled) return null;
        const g = channelResponse(q, fs, f, ch);
        const v = { re: g.re * s.re[k] - g.im * s.im[k], im: g.re * s.im[k] + g.im * s.re[k] };
        const fb = m.feedback?.[ch] ?? null;
        if (fb === null) return v;
        // − p·ℓ: the share of the anti-noise that comes back through the reference mic.
        const lr = g.re * fb.re[k] - g.im * fb.im[k];
        const li = g.re * fb.im[k] + g.im * fb.re[k];
        return { re: v.re - (pr * lr - pi * li), im: v.im - (pr * li + pi * lr) };
      }) as [Complex | null, Complex | null];
      return harmonicFactors({ re: pr, im: pi }, b, speakers, bounds);
    });
  // Centring: a channel gain low enough that a harmonic needs more than the trims' maximum moves up, by
  // what the harmonics the raise can bring into range need.
  const gainMax = q.gainUnlocked ? RANGES.gainDb.maxUnlocked : RANGES.gainDb.max;
  const free = centre ? factorsAt({ min: 0, max: Infinity }) : [];
  for (const ch of [0, 1] as const) {
    const room = gainMax - q.channels[ch].gainDb;
    let raise = 0;
    for (const fct of free) {
      const t = fct.t[ch];
      if (t === null) continue;
      const over = 20 * Math.log10(Math.hypot(t.re, t.im)) - r.max;
      if (over <= room + 1e-9) raise = Math.max(raise, over);
    }
    if (raise > 0 && Number.isFinite(raise)) q.channels[ch].gainDb += Math.min(raise, room);
  }
  const bounded = factorsAt({ min: dbToAmp(r.min), max: dbToAmp(r.max) });
  const one: Complex = { re: 1, im: 0 };
  rows.forEach(({ h, k }, i) => {
    const clamped = bounded[i].capped[0] || bounded[i].capped[1];
    const before = clamped ? pointAttenuationDb(m, q, fs, k) : 0;
    for (const ch of [0, 1] as const) {
      const t = bounded[i].t[ch];
      if (t !== null) q.channels[ch].harmonicTrims[h - 1] = trimForTarget(t, one);
    }
    if (!clamped) return;
    if (m.coherence === null || m.coherence[k] >= LOW_COHERENCE) capped.add(h);
    if (pointAttenuationDb(m, q, fs, k) < before) {
      // Clamped trims that would leave this harmonic worse than none: none.
      for (const ch of [0, 1] as const) q.channels[ch].harmonicTrims[h - 1] = { gainDb: 0, phaseDeg: 0 };
    }
  });
  return { params: clampParams(q), capped };
}

/** One harmonic whose trims the stability margin lowered: the neutralised loop around it before, and the dB taken off. */
interface TrimLoopLimit {
  loopDb: number;
  loweredDb: number;
}

/**
 * Per-harmonic prediction of the params returned (SolveOutcome.harmonics); untrimmedDb from `delayOnly`, the delay
 * alone under its own margin (null: `p` without trims).
 */
function harmonicSolutions(
  p: EngineParams,
  m: SolveModel,
  fs: number,
  capped: Set<number>,
  lowered: ReadonlyMap<number, TrimLoopLimit>,
  delayOnly: EngineParams | null,
): HarmonicSolution[] {
  const bare = delayOnly ?? withoutTrims(p);
  return trimmedHarmonics(p, m, fs).map(({ h, k }) => {
    const x: HarmonicSolution = {
      h,
      freqHz: m.freqs[k],
      predictedDb: pointAttenuationDb(m, p, fs, k),
      untrimmedDb: pointAttenuationDb(m, bare, fs, k),
      capped: capped.has(h),
    };
    const l = lowered.get(h);
    if (l !== undefined) x.loopLimitedDb = l.loweredDb;
    const c = predictionCeiling(x.predictedDb, m.coherence !== null);
    if (c !== null) x.ceiling = c;
    return x;
  });
}

/** "a", "a e b", "a, b e c". */
const joinList = (v: string[]): string => (v.length > 1 ? `${v.slice(0, -1).join(', ')} e ${v[v.length - 1]}` : (v[0] ?? ''));

/**
 * Italian advice on an applied prediction below WEAK_ATTENUATION_DB, null above it (step 6 restates it for separate
 * zones merged).
 */
export function weakAttenuationAdvice(predictedDb: number): string | null {
  if (!(predictedDb < WEAK_ATTENUATION_DB)) return null;
  return `Attenuazione prevista solo ${fmt(predictedDb)} dB: controlla posizionamento, banda dei filtri e causalità prima di attivare l'uscita.`;
}

/** Italian advice on the harmonics whose trim was clamped, null when none (step 6 restates it for separate zones merged). */
export function cappedAdvice(list: readonly HarmonicSolution[]): string | null {
  const c = list.filter((x) => x.capped);
  if (c.length === 0) return null;
  const many = c.length > 1;
  return (
    `${many ? 'Armoniche' : 'Armonica'} ${joinList(c.map((x) => String(x.h)))} (${joinList(c.map((x) => fmt(x.freqHz, 0)))} Hz): ` +
    `la correzione necessaria esce dal campo del ritocco per armonica (${fmt(RANGES.trimGainDb.min, 0)}…+${fmt(RANGES.trimGainDb.max, 0)} dB), ` +
    `attenuazione prevista ${joinList(c.map((x) => fmt(x.predictedDb)))} dB. Controlla la cassa e la sua posizione a ` +
    `${many ? 'quelle frequenze' : 'quella frequenza'}.`
  );
}

/**
 * Italian advice on the harmonics whose trims the stability margin lowered, null when none. `uniformDb`: what the
 * margin then took off every channel gain as well (the other harmonics lost it too). «le altre armoniche non
 * cambiano» only when that is true: no uniform cut, and some trimmed harmonic left as it was.
 */
function loopTrimAdvice(list: HarmonicSolution[], lowered: ReadonlyMap<number, TrimLoopLimit>, taps: number, uniformDb: number): string | null {
  const c = list.filter((x) => x.loopLimitedDb !== undefined && lowered.has(x.h));
  if (c.length === 0) return null;
  const many = c.length > 1;
  const others = uniformDb > 0
    ? `, oltre alla riduzione di ${fmtCut(uniformDb)} dB dei guadagni dei canali`
    : '';
  const rest = uniformDb === 0 && list.some((x) => !lowered.has(x.h)) ? '; le altre armoniche non cambiano' : '';
  return (
    `${many ? 'Armoniche' : 'Armonica'} ${joinList(c.map((x) => String(x.h)))} (${joinList(c.map((x) => fmt(x.freqHz, 0)))} Hz): ` +
    `guadagno d'anello cassa → microfono di riferimento ${joinList(c.map((x) => fmt(lowered.get(x.h)!.loopDb)))} dB dopo la ` +
    `neutralizzazione del feedback, sopra ${fmt(LOOP_GAIN_DANGER_DB, 0)} dB: per la stabilità ` +
    `${many ? 'le correzioni di queste armoniche sono state ridotte' : "la correzione dell'armonica è stata ridotta"} di ` +
    `${joinList(c.map((x) => fmt(x.loopLimitedDb!)))} dB${others} (attenuazione prevista ${joinList(c.map((x) => fmt(x.predictedDb)))} dB)` +
    `${rest}. Per recuperare attenuazione allontana o orienta la cassa lontano dal microfono di riferimento${tapsAdvice(taps)}.`
  );
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

/** Passes of the per-harmonic loop margin: off its centre a band's loop follows its trim only nearly. */
const TRIM_LOOP_PASSES = 4;

/**
 * Tonal mode: the harmonics whose trims act (h ≤ tonal.harmonics and MAX_TRIMS, inside the calibration band,
 * below 0.45·fs), each with the stretch of the band its resonator dominates: within h·f0/(2Q) (the
 * resonator's half bandwidth) and f0/2 (half way to the neighbours) of h·f0, inside the calibration band.
 */
function trimLoopBands(cal: CalibrationData, q: EngineParams, fs: number): { h: number; centre: number; lo: number; hi: number }[] {
  const band = normaliseBand(cal.band, fs);
  if (q.phaseMode !== 'tonal' || band === null) return [];
  const f0 = q.tonal.f0;
  return Array.from(harmonicFreqs(f0, band, fs)).flatMap((f) => {
    const h = Math.round(f / f0);
    if (h > Math.min(q.tonal.harmonics, MAX_TRIMS) || !(f < 0.45 * fs)) return [];
    const half = Math.min(f / (2 * q.tonal.q), f0 / 2);
    return [{ h, centre: f, lo: Math.max(band[0], f - half), hi: Math.min(band[1], f + half) }];
  });
}

/** finishSolve's stability margin applied to some params (enforceLoopMargin). */
interface LoopMargin {
  /** The params within the margin; null when the channel gains would have to fall below their range. */
  params: EngineParams | null;
  /** Tonal mode: the harmonics whose trims stay lowered (by loweredDb, net of what was given back). */
  trims: Map<number, TrimLoopLimit>;
  /** Loop once the trims were lowered, before the gains are (null without feedback data). */
  rest: LoopGains | null;
  /** dB taken off every channel gain for what the trims could not take (0 = none). */
  limitedDb: number;
  /** Loop of `params` (null without feedback data or params). */
  final: LoopGains | null;
}

/** Halvings tried when a trim cut given back after the uniform cut would put the loop over the margin again. */
const GIVE_BACK_TRIES = 3;
/** Below this uniform cut nothing is given back: a harmonic would get back less than it, a few hundredths of a dB. */
const GIVE_BACK_MIN_DB = 0.1;

/**
 * The solvers' stability margin: the neutralised loop of the params returned stays at or below
 * LOOP_GAIN_DANGER_DB in band (with capLoopGain's rounding allowance, so that capping them again changes
 * nothing; a sampled check: the loop frequencies with the highest peaks refined between them, and each
 * harmonic's band finely, see loopChecker). In tonal mode (`perHarmonic`) a harmonic whose loop is above it
 * first has its own trims lowered, on the channels that feed back into the reference mic, by just the
 * excess (to the trims' floor at most): at h·f0 the loop scales with the trim exactly, and the other
 * harmonics keep their cancellation (a uniform cut of every gain would cost them all of it for one harmonic's
 * feedback). Each harmonic's band is searched for its peak (bandLoopPeakDb), and a few passes cover a peak off
 * the centre, where the loop follows the trim only nearly (the bank's other bands). Through the closed loop a
 * weakened anti-noise can add to the noise (the loop's phase): a harmonic the cut would leave louder than with
 * those trims at their floor gets the floor (a lower loop still). What is left (between the harmonics, a trim
 * at its floor, or outside tonal mode) is taken off every channel gain: the loop scales with them. That cut
 * lowers the harmonics already cut as well, so each gets back as much of its own cut as the margin then
 * allows (largest gain at its harmonic first, verified on the whole loop): no harmonic is cut twice.
 */
function enforceLoopMargin(
  cal: CalibrationData,
  p: EngineParams,
  fs: number,
  taps: number,
  model: SolveModel,
  perHarmonic = true,
  checker?: LoopChecker | null,
): LoopMargin {
  const start = clampParams(p);
  const q = clampParams(p);
  const trims = new Map<number, TrimLoopLimit>();
  const lc = checker !== undefined ? checker : loopChecker(cal, q, fs, taps);
  if (lc === null) return { params: q, trims, rest: null, limitedDb: 0, final: null };
  const bands = perHarmonic ? trimLoopBands(cal, q, fs) : [];
  const fed = ([0, 1] as const).filter((ch) => cal.feedback[ch] !== null && q.channels[ch].enabled);
  const floor = RANGES.trimGainDb.min;
  const above = (db: number): boolean => db > LOOP_GAIN_DANGER_DB + LOOP_CAP_TOL_DB;
  for (let pass = 0; pass < TRIM_LOOP_PASSES && bands.length > 0 && fed.length > 0; pass++) {
    let changed = false;
    for (const b of bands) {
      const peak = lc.bandPeak(q, b);
      if (!above(peak)) continue;
      const step = peak - LOOP_GAIN_DANGER_DB + LOOP_CAP_TOL_DB / 2;
      for (const ch of fed) {
        const t = q.channels[ch].harmonicTrims[b.h - 1];
        const g = Math.max(floor, t.gainDb - step);
        if (g < t.gainDb) changed = true;
        t.gainDb = g;
      }
      if (!trims.has(b.h)) trims.set(b.h, { loopDb: peak, loweredDb: 0 });
    }
    if (!changed) break;
  }
  const rows = new Map(trimmedHarmonics(q, model, fs).map((r) => [r.h, r.k]));
  for (const h of trims.keys()) {
    const k = rows.get(h);
    if (k === undefined) continue;
    const floored = clampParams(q);
    for (const ch of fed) floored.channels[ch].harmonicTrims[h - 1].gainDb = floor;
    if (pointAttenuationDb(model, floored, fs, k) > pointAttenuationDb(model, q, fs, k)) {
      for (const ch of fed) q.channels[ch].harmonicTrims[h - 1].gainDb = floor;
    }
  }
  const settle = (from: EngineParams): void => {
    for (const [h, t] of trims) {
      t.loweredDb = Math.max(...fed.map((ch) => start.channels[ch].harmonicTrims[h - 1].gainDb - from.channels[ch].harmonicTrims[h - 1].gainDb));
      if (!(t.loweredDb > LOOP_CAP_TOL_DB)) trims.delete(h);
    }
  };
  settle(q);
  const rest = lc.gains(q);
  if (!above(rest.afterDb)) return { params: q, trims, rest, limitedDb: 0, final: rest };
  const limitedDb = rest.afterDb - LOOP_GAIN_DANGER_DB + LOOP_CAP_TOL_DB / 2;
  let cur = lowerGains(q, limitedDb);
  if (cur === null) return { params: null, trims, rest, limitedDb, final: null };
  // The uniform cut lowered the harmonics cut before it too: give their own cuts back where the margin allows.
  // Each give-back is checked on its own band and the neighbours' (the rest of the loop barely moves), the
  // result on the whole loop; when that fails, again with every give-back checked on the whole loop.
  if (!(limitedDb >= GIVE_BACK_MIN_DB)) return { params: cur, trims, rest, limitedDb, final: lc.gains(cur) };
  const uniform = cur;
  const within = (x: EngineParams): boolean => !bands.some((b) => above(lc.bandPeak(x, b))) && !above(lc.gains(x).afterDb);
  const near = (x: EngineParams, h: number): boolean => !bands.some((b) => Math.abs(b.h - h) <= 1 && above(lc.bandPeak(x, b)));
  const raised = (x: EngineParams, h: number, db: number): EngineParams => {
    const y = clampParams(x);
    for (const ch of fed) {
      const t = y.channels[ch].harmonicTrims[h - 1];
      t.gainDb = Math.min(start.channels[ch].harmonicTrims[h - 1].gainDb, t.gainDb + db);
    }
    return clampParams(y);
  };
  const gainAt = (x: EngineParams, h: number): number => pointAttenuationDb(model, x, fs, rows.get(h)!);
  const order = [...trims.keys()]
    .filter((h) => rows.has(h) && bands.some((b) => b.h === h))
    .map((h) => ({ h, gain: gainAt(raised(cur!, h, Infinity), h) - gainAt(cur!, h) }))
    .filter((x) => x.gain > 0)
    .sort((a, b) => b.gain - a.gain);
  const giveBack = (check: (x: EngineParams, h: number) => boolean): EngineParams => {
    let x = uniform;
    for (const { h } of order) {
      const b = bands.find((y) => y.h === h)!;
      let up = Math.min(trims.get(h)!.loweredDb, LOOP_GAIN_DANGER_DB - LOOP_CAP_TOL_DB / 2 - lc.bandPeak(x, b));
      for (let i = 0; i < GIVE_BACK_TRIES && up > 1e-3; i++, up /= 2) {
        const next = raised(x, h, up);
        if (!(gainAt(next, h) > gainAt(x, h))) break;
        if (check(next, h)) {
          x = next;
          break;
        }
      }
    }
    return x;
  };
  cur = giveBack(near);
  if (cur !== uniform && !within(cur)) cur = giveBack(within);
  settle(cur);
  return { params: cur, trims, rest, limitedDb, final: lc.gains(cur) };
}

/**
 * The stability margin that keeps more attenuation: enforceLoopMargin per harmonic, or with the uniform cut
 * alone. The per-harmonic cut usually wins by far, but a peak off a harmonic's centre that its neighbours'
 * skirts feed too needs a deeper trim cut than the uniform excess, and that can cost more in total.
 */
function bestLoopMargin(cal: CalibrationData, p: EngineParams, fs: number, taps: number, model: SolveModel, checker?: LoopChecker | null): LoopMargin {
  const lc = checker !== undefined ? checker : loopChecker(cal, p, fs, taps);
  const per = enforceLoopMargin(cal, p, fs, taps, model, true, lc);
  if (per.trims.size === 0) return per;
  const uniform = enforceLoopMargin(cal, p, fs, taps, model, false, lc);
  return marginScore(uniform, model, fs) > marginScore(per, model, fs) ? uniform : per;
}

/** Predicted attenuation once the stability margin `m` is enforced (−∞: refused). */
const marginScore = (m: LoopMargin, model: SolveModel, fs: number): number =>
  m.params !== null ? predictedAttenuationDb(model, m.params, fs) : -Infinity;

/** Below this a difference between two predictions is rounding, dB. */
const PREDICTION_TOL_DB = 1e-6;

/** One tonal solution before and after the stability margin (tonalSolution). */
interface TonalCandidate {
  /** The params the margin was enforced on: the solved delay/gain/polarity, with the trims and any centring raise. */
  pre: EngineParams;
  margin: LoopMargin;
  /** Predicted attenuation once the margin is enforced (marginScore: −∞ refused). */
  score: number;
  /** Harmonics whose trims in effect were clamped (solveHarmonicTrims). */
  capped: Set<number>;
  /** Harmonics reconcileTrims gave the delay alone's response back. */
  given?: Set<number>;
}

/** Rounds of reconcileTrims over the harmonics left worse than the delay alone (a uniform cut can make room for another). */
const RECONCILE_ROUNDS = 3;
/**
 * reconcileTrims lowers every gain to give a harmonic back only for at least as much more in total as the cut, and at
 * most this much, dB: a cut that moves every harmonic for less than it takes off them is not worth reporting, while a
 * cut of a few 1e-5 dB that gives a harmonic back (seed 1105 zone R: 0.14 → 1.97 dB at h1) is.
 */
const GIVE_BACK_CUT_GAIN_DB = 0.05;
/** Steps of reconcileTrims' lowering of a give-back's trims over the margin (secant steps: 2–3 usually reach it). */
const TRIM_LOWER_PASSES = 8;
/** Each secant step aims this much past the margin (a step short of it costs another pass). */
const TRIM_LOWER_OVERSHOOT = 1.02;
/** Least share of a step the loop is assumed to follow (a peak h's trims barely feed ends at the trims' floor). */
const TRIM_LOWER_MIN_SHARE = 0.05;
/** Up to this excess left after the steps is taken off every gain instead, dB (moves the other harmonics by as much). */
const TRIM_LOWER_FINISH_DB = 0.02;

/**
 * Tonal mode: a trimmed candidate after its margin, reconciled harmonic by harmonic with the delay alone under its
 * own margin (`ref`). The trims null each harmonic on its own, but then the margin cuts them (a null cut by a factor
 * α leaves p·(1 − α)/(1 − α·T·ℓ): through the closed loop that can be louder than no trim), a uniform cut or the
 * centring raise moves every harmonic, and the resonators' skirts carry each trim to the harmonics above
 * tonal.harmonics, which the trims are not solved for (at a low Q, trims undid the delay's partial cancellation of
 * the next harmonic: seed 169, 25 → −4 dB on the zone's loudest harmonic). On the params within the margin, then:
 * a harmonic its trims leave worse than the delay alone gets the delay alone's response there back (trims that
 * reproduce it on the candidate's gains, raise and cuts included, when inside the trims' range). When that takes the
 * loop over the margin, those trims are lowered into it (secant steps, TRIM_LOWER_PASSES at most, the last
 * TRIM_LOWER_FINISH_DB off every gain), unless lowering every gain by the excess instead predicts more in total by at
 * least that cut (GIVE_BACK_CUT_GAIN_DB at most); up
 * to RECONCILE_ROUNDS rounds (a uniform cut leaves the harmonics given back before it below the delay alone again).
 * Then, one at a time, the harmonic whose trims cost the total most goes back to the delay alone's response (over the
 * margin: with its trims lowered into it, or a uniform cut, when the total still gains), while that raises the total. A change is checked on its band, the neighbours' and the loop frequencies (the other bands
 * barely move), the result on the whole loop; when that fails, again with every change checked on the whole loop.
 * The harmonics given back take the delay alone's loop cuts (plus any lowering) and lose their capped flag; a
 * uniform cut adds to the candidate's.
 */
function reconcileTrims(
  cal: CalibrationData,
  c: TonalCandidate,
  ref: TonalCandidate,
  model: SolveModel,
  fs: number,
  lc: LoopChecker | null,
): TonalCandidate {
  const start = c.margin.params;
  const base = ref.margin.params;
  if (start === null || base === null) return c;
  const rows = trimmedHarmonics(start, model, fs).filter((r) => model.weights[r.k] > 0);
  const chans = ([0, 1] as const).filter((ch) => start.channels[ch].enabled);
  const { min, max } = RANGES.trimGainDb;
  // The trims that give `x` the delay alone's response at harmonic h (null: outside the trims' range).
  const back = (x: EngineParams, h: number): EngineParams | null => {
    const y = clampParams(x);
    for (const ch of chans) {
      const r = base.channels[ch];
      const gainDb = r.harmonicTrims[h - 1].gainDb + r.gainDb - x.channels[ch].gainDb;
      if (!r.enabled || !(gainDb <= max + 1e-9 && gainDb >= min - 1e-9)) return null;
      y.channels[ch].harmonicTrims[h - 1] = { gainDb: Math.min(max, Math.max(min, gainDb)), phaseDeg: r.harmonicTrims[h - 1].phaseDeg };
    }
    return y;
  };
  const bands = lc !== null ? trimLoopBands(cal, start, fs) : [];
  const above = (db: number): boolean => db > LOOP_GAIN_DANGER_DB + LOOP_CAP_TOL_DB;
  // The loop of x everywhere (bands and loop frequencies), dB above the margin (≤ 0: within it); with its loop gains.
  const excess = (x: EngineParams): { db: number; gains: LoopGains } | null => {
    if (lc === null) return null;
    const gains = lc.gains(x);
    return { db: Math.max(gains.afterDb, ...bands.map((b) => lc.bandPeak(x, b))) - LOOP_GAIN_DANGER_DB, gains };
  };
  const total = (x: EngineParams): number => predictedAttenuationDb(model, x, fs);
  const point = (x: EngineParams, k: number): number => pointAttenuationDb(model, x, fs, k);
  // How far the loop of x is above the margin around harmonic h (its band and the neighbours') and at the loop
  // frequencies (where a uniform cut left the loop's peak at the margin), dB.
  const nearExcess = (x: EngineParams, h: number): number =>
    lc === null
      ? -Infinity
      : Math.max(lc.gains(x).afterDb, ...bands.filter((b) => Math.abs(b.h - h) <= 1).map((b) => lc.bandPeak(x, b))) - LOOP_GAIN_DANGER_DB;
  // Harmonic h's trims of x lowered by db (null below the trims' floor).
  const lowerTrims = (x: EngineParams, h: number, db: number): EngineParams | null => {
    const y = clampParams(x);
    for (const ch of chans) {
      const t = y.channels[ch].harmonicTrims[h - 1];
      if (t.gainDb - db < min) return null;
      t.gainDb -= db;
    }
    return y;
  };
  interface Pass {
    params: EngineParams;
    /** Harmonics given the delay alone's response back, with the dB their trims were lowered further (and the loop before). */
    done: Map<number, { loopDb: number; extraDb: number }>;
    cutDb: number;
  }
  const pass = (whole: boolean): Pass => {
    let x = start;
    let cutDb = 0;
    const done: Pass['done'] = new Map();
    const over = (y: EngineParams, h: number): number => (whole ? (excess(y)?.db ?? -Infinity) : nearExcess(y, h));
    const ok = (y: EngineParams, h: number): boolean => !(over(y, h) > LOOP_CAP_TOL_DB);
    const tryBack = (h: number, k: number, cut: boolean): boolean => {
      const y = back(x, h);
      if (y === null) return false;
      const loopDb = over(y, h);
      if (!(loopDb > LOOP_CAP_TOL_DB)) {
        x = y;
        done.set(h, { loopDb: Number.NaN, extraDb: 0 });
        return true;
      }
      if (!cut) return false;
      // The loop over the margin: the trims given back lowered by the excess (the loop around h follows them, a few
      // passes), or every gain lowered by it when that predicts GIVE_BACK_CUT_GAIN_DB more in total.
      // Each step is the excess over the share of it the last step removed (a secant: when h's trims feed only part of
      // the peak, equal steps would close in on the margin geometrically and stop just above it), a little more.
      // What the steps leave over the margin (a peak h's trims barely feed: hundredths of a dB) comes off every gain.
      let w: EngineParams | null = y;
      let extraDb = 0;
      let share = 1;
      let left = loopDb;
      for (let i = 0; i < TRIM_LOWER_PASSES && w !== null && left > LOOP_CAP_TOL_DB; i++) {
        const step = (left / share) * TRIM_LOWER_OVERSHOOT + LOOP_CAP_TOL_DB / 2;
        w = lowerTrims(w, h, step);
        extraDb += step;
        const now = w !== null ? over(w, h) : Infinity;
        share = Math.min(1, Math.max(TRIM_LOWER_MIN_SHARE, (left - now) / step));
        left = now;
      }
      const finishDb = w !== null && left > LOOP_CAP_TOL_DB && left <= TRIM_LOWER_FINISH_DB ? left + LOOP_CAP_TOL_DB / 2 : 0;
      if (finishDb > 0) w = lowerGains(w!, finishDb);
      // (Lowered, they no longer reproduce the delay alone: kept only when the total does not lose by it, seed 575.)
      const lowered =
        w !== null && ok(w, h) && point(w, k) > point(x, k) + PREDICTION_TOL_DB && total(w) >= total(x) - PREDICTION_TOL_DB ? w : null;
      const best = Math.max(total(x), lowered !== null ? total(lowered) : -Infinity);
      // (A cut that would leave no more than the best before it is not worth the whole-loop check.)
      const all = total(y) > best + PREDICTION_TOL_DB ? excess(y) : null;
      const allDb = all !== null && all.db > 0 ? all.db + LOOP_CAP_TOL_DB / 2 : 0;
      const z = allDb > 0 ? lowerGains(y, allDb) : null;
      if (z !== null && total(z) >= best + Math.min(GIVE_BACK_CUT_GAIN_DB, allDb) && total(z) > best + PREDICTION_TOL_DB && ok(z, h)) {
        x = z;
        cutDb += allDb;
        done.set(h, { loopDb: Number.NaN, extraDb: 0 });
        return true;
      }
      if (lowered === null) return false;
      x = lowered;
      cutDb += finishDb;
      done.set(h, { loopDb: loopDb + LOOP_GAIN_DANGER_DB, extraDb });
      return true;
    };
    const fixWorse = (): void => {
      for (let round = 0, changed = true; changed && round < RECONCILE_ROUNDS; round++) {
        changed = false;
        for (const { h, k } of rows) {
          if (point(x, k) < point(base, k) - PREDICTION_TOL_DB && tryBack(h, k, true)) changed = true;
        }
      }
    };
    fixWorse();
    const tried = new Set(done.keys());
    for (;;) {
      const now = total(x);
      let best: { h: number; k: number; gain: number } | null = null;
      for (const { h, k } of rows) {
        const y = tried.has(h) ? null : back(x, h);
        const gain = y !== null ? total(y) - now : 0;
        if (gain > PREDICTION_TOL_DB && (best === null || gain > best.gain)) best = { h, k, gain };
      }
      if (best === null) break;
      tried.add(best.h);
      if (tryBack(best.h, best.k, false)) continue;
      // Over the margin: with its trims lowered into it or a uniform cut, when the total still gains by it (seed 575:
      // h7 back after h3, 1.81 → 2.13 dB).
      // A cut or lowered trims can leave a harmonic given back before below the delay alone again: those first.
      const before = { x, cutDb, done: new Map(done) };
      if (!tryBack(best.h, best.k, true)) continue;
      if (total(x) > now + PREDICTION_TOL_DB) {
        fixWorse();
        if (total(x) > now + PREDICTION_TOL_DB) continue;
      }
      x = before.x;
      cutDb = before.cutDb;
      done.clear();
      for (const [h, d] of before.done) done.set(h, d);
    }
    return { params: x, done, cutDb };
  };
  let r = pass(false);
  if (r.done.size === 0) return c;
  let e = excess(r.params);
  if (e !== null && above(e.db + LOOP_GAIN_DANGER_DB)) {
    r = pass(true);
    if (r.done.size === 0) return c;
    e = excess(r.params);
  }
  // Loop cuts: the delay alone's on the harmonics given back, plus what their trims were lowered further.
  const trims = new Map<number, TrimLoopLimit>();
  for (const [h, t] of c.margin.trims) if (!r.done.has(h)) trims.set(h, t);
  for (const [h, d] of r.done) {
    const t = ref.margin.trims.get(h);
    const loweredDb = (t?.loweredDb ?? 0) + d.extraDb;
    if (loweredDb > LOOP_CAP_TOL_DB) trims.set(h, { loopDb: t?.loopDb ?? d.loopDb, loweredDb });
  }
  const limitedDb = c.margin.limitedDb + r.cutDb;
  const final = e?.gains ?? null;
  return {
    pre: c.pre,
    margin: { params: r.params, trims, rest: e !== null ? restOf(e, limitedDb) : c.margin.rest, limitedDb, final },
    score: total(r.params),
    capped: new Set([...c.capped].filter((h) => !r.done.has(h))),
    given: new Set([...(c.given ?? []), ...r.done.keys()]),
  };
}

/**
 * The loop a uniform cut of `limitedDb` acts on, from the loop of the params after it (`e`: its whole-loop excess,
 * bands included, and its loop gains): scaled back up by the cut (the loop scales with the gains). With no cut, the
 * loop itself.
 */
function restOf(e: { db: number; gains: LoopGains }, limitedDb: number): LoopGains {
  const up = limitedDb > 0 ? limitedDb : 0;
  return { rawDb: e.gains.rawDb + up, atHz: e.gains.atHz, afterDb: Math.max(e.gains.afterDb, e.db + LOOP_GAIN_DANGER_DB) + up };
}

/**
 * Tonal mode: a candidate's uniform gain cut given back as far as its loop now allows (reconcileTrims' give-backs, a
 * trim at its floor or the per-harmonic margin's give-backs can leave the whole loop below the margin with the cut
 * still in place: seed 211, 10.6 dB where 11.3 dB fit), when that predicts more in total. Every channel gain is raised
 * by the slack (the loop scales with them, so it lands at the margin) or by the whole cut; the harmonics given the
 * delay alone's response back (`given`) keep it when their trims can be lowered by as much (else the plain raise).
 * Checked on the whole loop; with `ref`, the delay alone a candidate was reconciled with, a raise that would leave a
 * harmonic below the delay alone's response (it was not) is not made (seed 123, h3). The loop the cut acts on is then
 * restated from the loop returned (restOf).
 */
function reclaimCut(
  cal: CalibrationData,
  c: TonalCandidate,
  model: SolveModel,
  fs: number,
  lc: LoopChecker | null,
  ref: TonalCandidate | null = null,
): TonalCandidate {
  const x = c.margin.params;
  if (x === null || lc === null || !(c.margin.limitedDb > 0)) return c;
  // With `ref` (the delay alone, for a reconciled candidate): no harmonic at or above its response may fall below it.
  const base = ref?.margin.params ?? null;
  const rows = base !== null ? trimmedHarmonics(x, model, fs).filter((r) => model.weights[r.k] > 0) : [];
  const point = (y: EngineParams, k: number): number => pointAttenuationDb(model, y, fs, k);
  const kept = rows.filter((r) => point(x, r.k) >= point(base!, r.k) - PREDICTION_TOL_DB);
  const fair = (y: EngineParams): boolean => kept.every((r) => point(y, r.k) >= point(base!, r.k) - PREDICTION_TOL_DB);
  const bands = trimLoopBands(cal, x, fs);
  const excess = (y: EngineParams): { db: number; gains: LoopGains } => {
    const gains = lc.gains(y);
    return { db: Math.max(gains.afterDb, ...bands.map((b) => lc.bandPeak(y, b))) - LOOP_GAIN_DANGER_DB, gains };
  };
  const now = excess(x);
  const up = Math.min(c.margin.limitedDb, -now.db - LOOP_CAP_TOL_DB / 2);
  if (!(up > (up === c.margin.limitedDb ? 0 : LOOP_CAP_TOL_DB))) return c;
  const chans = ([0, 1] as const).filter((ch) => x.channels[ch].enabled);
  if (chans.some((ch) => x.channels[ch].gainDb + up > RANGES.gainDb.max)) return c;
  const raised = lowerGains(x, -up)!;
  const variants = [raised];
  if (c.given !== undefined && c.given.size > 0) {
    const same = clampParams(raised);
    let fits = true;
    for (const h of c.given) {
      for (const ch of chans) {
        const t = same.channels[ch].harmonicTrims[h - 1];
        if (t.gainDb - up < RANGES.trimGainDb.min) fits = false;
        else t.gainDb -= up;
      }
    }
    if (fits) variants.unshift(clampParams(same));
  }
  // The whole cut given back is kept unless it predicts less (a cut that buys nothing is not reported as one).
  const whole = up === c.margin.limitedDb;
  let best: { params: EngineParams; score: number; e: { db: number; gains: LoopGains } } | null = null;
  for (const y of variants) {
    const score = predictedAttenuationDb(model, y, fs);
    const enough = whole ? score >= c.score - PREDICTION_TOL_DB : score > c.score + PREDICTION_TOL_DB;
    if (!enough || (best !== null && !(score > best.score + PREDICTION_TOL_DB)) || !fair(y)) continue;
    const e = excess(y);
    if (e.db > LOOP_CAP_TOL_DB) continue;
    best = { params: y, score, e };
  }
  if (best === null) return c;
  const limitedDb = c.margin.limitedDb - up > LOOP_CAP_TOL_DB ? c.margin.limitedDb - up : 0;
  return {
    ...c,
    margin: { ...c.margin, params: best.params, limitedDb, rest: restOf(best.e, limitedDb), final: best.e.gains },
    score: best.score,
  };
}

/**
 * Tonal mode: the solved delay/gain/polarity `bare` (no trims) with per-harmonic trims, under the stability margin.
 * Candidates, each judged after its own best margin (bestLoopMargin): the trims with the centring raise and, when a
 * raise was made, without it (solveHarmonicTrims), each reconciled with the delay alone (reconcileTrims); and the
 * delay alone itself, untrimmed and without a raise. The one that predicts most in total is kept (the raise, then
 * the delay alone, only when strictly better): a tonal solve never predicts less than its delay alone under the
 * same rules. Returns the candidate kept and the delay alone (SolveOutcome.delayOnly, the reference of
 * HarmonicSolution.untrimmedDb).
 */
function tonalSolution(
  cal: CalibrationData,
  bare: EngineParams,
  model: SolveModel,
  fs: number,
  taps: number,
): { chosen: TonalCandidate; delayOnly: TonalCandidate } {
  const lc = loopChecker(cal, bare, fs, taps);
  const judge = (t: { params: EngineParams; capped: Set<number> }): TonalCandidate => {
    const margin = bestLoopMargin(cal, t.params, fs, taps, model, lc);
    return { pre: t.params, margin, score: marginScore(margin, model, fs), capped: t.capped };
  };
  const delayOnly = reclaimCut(cal, judge({ params: bare, capped: new Set() }), model, fs, lc);
  const raised = solveHarmonicTrims(bare, model, cal.speakers, fs, true);
  const trimmed = [raised];
  // The centring raise costs loop gain at the harmonic it serves and lifts the harmonics above the trimmed ones:
  // the trims without it are a candidate too.
  if (([0, 1] as const).some((ch) => raised.params.channels[ch].gainDb !== bare.channels[ch].gainDb)) {
    trimmed.unshift(solveHarmonicTrims(bare, model, cal.speakers, fs, false));
  }
  let chosen: TonalCandidate | null = null;
  for (const t of trimmed) {
    const c = reclaimCut(cal, reconcileTrims(cal, judge(t), delayOnly, model, fs, lc), model, fs, lc, delayOnly);
    if (chosen === null || c.score > chosen.score + PREDICTION_TOL_DB) chosen = c;
  }
  if (delayOnly.score > chosen!.score + PREDICTION_TOL_DB) chosen = delayOnly;
  return { chosen: chosen!, delayOnly };
}

/** Below this a uniform cut is not what leaves a solve useless, dB (it moves the prediction by hundredths at most). */
const NEGLIGIBLE_CUT_DB = 0.05;

/**
 * Whether the uniform cut of a margin is what leaves its params below MIN_USEFUL_DB: the gains would fall below their
 * range, or the cut is at least NEGLIGIBLE_CUT_DB. Otherwise a refusal is about the attenuation («non si ottiene
 * attenuazione»), not a howl risk: seed 21 was refused as «Rischio di innesco … ridotti di 0,0 dB» for a cut of
 * 1e-4 dB with its loop already at the margin. (Without the cut the loop is over the margin: its prediction says
 * nothing.)
 */
const cutRefuses = (m: LoopMargin): boolean => m.params === null || m.limitedDb >= NEGLIGIBLE_CUT_DB;

/**
 * Converts the solution into params, then enforces the loop's stability margin: the loop through the
 * measured feedback paths, after the neutralisation FIRs the app loads, must stay at or below
 * LOOP_GAIN_DANGER_DB (6 dB of margin) everywhere in band (enforceLoopMargin: in tonal mode the trims of the
 * harmonics above it first, then every channel gain by what is left; tonal mode compares the trims with the delay
 * alone, see tonalSolution); when that leaves no useful attenuation the params are not changed, and the outcome's
 * result carries the prediction refused (below 1 dB: never a useful outcome). Between LOOP_GAIN_LIMIT_DB (the plan's
 * −20 dB target) and the margin the params stay and the advice says the cancellation is less accurate: near the band
 * edges a FIR of a few ms cannot model a band-limited path, so enforcing −20 dB there would throw away working
 * solutions. The predicted attenuation is recomputed for the params returned, through the closed loop (see
 * predictedAttenuationDb).
 */
function finishSolve(
  cal: CalibrationData,
  p: EngineParams,
  fs: number,
  result: AlignmentResult,
  advice: string[],
  solved: SolveModel,
  taps: number,
): SolveOutcome {
  // Predictions and trims through the closed loop the app runs (the neutralisers feedbackFirs(cal, taps) designs).
  const model = withFeedback(solved, cal, fs, taps);
  const res: AlignmentResult = { ...result, channels: [{ ...result.channels[0] }, { ...result.channels[1] }] };
  // The params stay as they were: the result carries the prediction that was refused (the advice quotes it).
  const unchanged = (loop: LoopGains | null, predictedDb: number, limited = 0): SolveOutcome => {
    res.predictedAttenuationDb = predictedDb;
    return {
      params: p,
      result: res,
      advice,
      loopGainDb: loop?.rawDb ?? null,
      loopGainAfterDb: loop?.afterDb ?? null,
      loopLimitedDb: limited,
    };
  };
  const nothing = (predictedDb: number): SolveOutcome => {
    // Nothing to gain: do not load a meaningless delay/gain into the engine.
    advice.push(`Con queste misure non si ottiene attenuazione (prevista ${fmt(predictedDb)} dB): i parametri non sono stati cambiati.`);
    const loop = loopGains(cal, p, fs, taps);
    if (loop !== null) loopAdvice(loop, taps, advice);
    return unchanged(loop, predictedDb);
  };
  const tonal = p.phaseMode === 'tonal' && res.channels.some((c) => c.use);
  if (!tonal && !(res.predictedAttenuationDb >= MIN_USEFUL_DB)) return nothing(res.predictedAttenuationDb);
  const conv = solutionToParams(p, res, fs);
  let margin: LoopMargin;
  let capped = new Set<number>();
  // Tonal mode: the delay alone under its own margin, what the trims are compared with.
  let delayOnly: TonalCandidate | null = null;
  if (tonal) {
    // Per-harmonic trims on top of the delay/gain solution; usefulness, margin and prediction are judged on the
    // candidate kept.
    const t = tonalSolution(cal, withoutTrims(conv.params), model, fs, taps);
    if (!(t.chosen.score >= MIN_USEFUL_DB) && !cutRefuses(t.chosen.margin)) return nothing(t.chosen.score);
    for (const ch of [0, 1] as const) {
      const g = t.chosen.pre.channels[ch].gainDb;
      if (g === conv.params.channels[ch].gainDb) continue;
      // The trims raised this gain (centring): the amplifier advice is about the gain actually used.
      const head = `Canale ${CH_NAME[ch]}: serve un guadagno di`;
      conv.advice = conv.advice.filter((a) => !a.startsWith(head));
      const amp = amplifierAdvice(ch, g);
      if (amp !== null) conv.advice.push(amp);
    }
    conv.params = t.chosen.pre;
    margin = t.chosen.margin;
    capped = t.chosen.capped;
    delayOnly = t.delayOnly;
  } else {
    // Trims are relative to the delay/gain they were solved with: a new solution starts without them.
    conv.params = withoutTrims(conv.params);
    margin = bestLoopMargin(cal, conv.params, fs, taps, model);
  }
  advice.push(...conv.advice);
  // The loop the uniform part of the margin acts on: after any per-harmonic trim cut.
  const loop = margin.rest;
  const limited = margin.limitedDb;
  const params = margin.params ?? conv.params;
  if (loop !== null && limited > 0) {
    const predicted = margin.params !== null ? predictedAttenuationDb(model, margin.params, fs) : 0;
    if (!(predicted >= MIN_USEFUL_DB) && !cutRefuses(margin)) return nothing(predicted);
    if (margin.params === null || !(predicted >= MIN_USEFUL_DB)) {
      advice.push(
        `Rischio di innesco: il guadagno d'anello cassa → microfono di riferimento è ${fmt(loop.rawDb)} dB a ` +
          `${fmt(loop.atHz, 0)} Hz e resta ${fmt(loop.afterDb)} dB anche con la neutralizzazione del feedback. Per ` +
          `scendere a ${fmt(LOOP_GAIN_DANGER_DB, 0)} dB i guadagni andrebbero ridotti di ${fmtCut(limited)} dB e l'anti-rumore ` +
          'non attenuerebbe più: i parametri non sono stati cambiati. Allontana o orienta la cassa lontano dal ' +
          `microfono di riferimento${tapsAdvice(taps)}, poi ripeti la misura dei percorsi.`,
      );
      return unchanged(loop, predicted, limited);
    }
    const k = dbToAmp(-limited);
    for (const c of res.channels) c.gainLin *= k;
    advice.push(
      `Guadagno d'anello cassa → microfono di riferimento ${fmt(loop.rawDb)} dB a ${fmt(loop.atHz, 0)} Hz, ` +
        `${fmt(loop.afterDb)} dB dopo la neutralizzazione del feedback: sopra ${fmt(LOOP_GAIN_DANGER_DB, 0)} dB non resta ` +
        `margine di stabilità. Guadagni dei canali ridotti di ${fmtCut(limited)} dB; per recuperare attenuazione allontana o ` +
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
    return unchanged(loop, res.predictedAttenuationDb, limited);
  }
  const weak = res.channels.some((c) => c.use) ? weakAttenuationAdvice(res.predictedAttenuationDb) : null;
  if (weak !== null) advice.push(weak);
  const out: SolveOutcome = {
    params,
    result: res,
    advice,
    // The values of the params returned.
    loopGainDb: margin.final?.rawDb ?? null,
    loopGainAfterDb: margin.final?.afterDb ?? null,
    loopLimitedDb: limited,
  };
  const ceiling = predictionCeiling(res.predictedAttenuationDb, model.coherence !== null);
  if (ceiling !== null) out.predictionCeiling = ceiling;
  const delayParams = delayOnly?.margin.params ?? null;
  if (tonal) {
    if (delayParams !== null) out.delayOnly = { params: clampParams(delayParams), predictedDb: delayOnly!.score };
    out.harmonics = harmonicSolutions(params, model, fs, capped, margin.trims, delayParams);
    for (const a of [cappedAdvice(out.harmonics), loopTrimAdvice(out.harmonics, margin.trims, taps, limited)]) if (a !== null) advice.push(a);
  }
  outcomeModels.set(out, { model, fs, solved: params, delayOnly: delayParams });
  return out;
}

/**
 * The model each applied outcome was solved and predicted with (predictOutcome), with the params returned and, in
 * tonal mode, the delay alone under its own margin (null: refused, or outside tonal mode); outcomes stay plain data.
 */
const outcomeModels = new WeakMap<SolveOutcome, { model: SolveModel; fs: number; solved: EngineParams; delayOnly: EngineParams | null }>();

/**
 * The delay alone of an outcome (`delayOnly`, kept with the outcome's params `solved`) on other params `p` (separate
 * zones merged): each of the outcome's channels gets the delay alone's values with the change of gain `p` made to
 * it (the merged cut); the other channels are `p`'s.
 */
function delayOnlyOn(delayOnly: EngineParams, solved: EngineParams, p: EngineParams): EngineParams {
  const q = clampParams(p);
  const d = clampParams(delayOnly);
  for (const ch of [0, 1] as const) {
    if (!solved.channels[ch].enabled || !q.channels[ch].enabled) continue;
    const moved = q.channels[ch].gainDb - solved.channels[ch].gainDb;
    q.channels[ch] = { ...d.channels[ch], gainDb: moved !== 0 ? d.channels[ch].gainDb + moved : d.channels[ch].gainDb };
  }
  return clampParams(q);
}

/** What predictOutcome gives: an applied outcome's predictions on other params. */
export interface OutcomePrediction {
  predictedDb: number;
  predictionCeiling: PredictionCeiling | null;
  harmonics?: HarmonicSolution[];
  /**
   * Tonal mode: the outcome's delay alone (SolveOutcome.delayOnly) moved onto the params (its channels with their change
   * of gain), and its prediction; absent without one, and when the params predicted are the delay alone already.
   */
  delayOnly?: { params: EngineParams; predictedDb: number };
}

/**
 * An applied outcome's predictions recomputed for other params `p` (same f0 and harmonics) with its own model —
 * separate zones: the merged params, both speakers playing. Each zone is solved with its own speaker alone, but
 * its model holds both neutralised feedback paths: with both channels on, the zone's anti-noise goes through the
 * closed loop 1/(1 − L₀ − L₁), the other speaker's feedback included (its secondary path into this zone is not
 * modelled: none was measured). A zone the other speaker's feedback undoes is then predicted as such, not as
 * the exact null of its own solve. Per harmonic, the outcome's capped and loop-limited flags are kept, and
 * untrimmedDb is the outcome's delay alone moved onto `p` (its channels with p's change of gain): on the outcome's
 * own params, the outcome's figures. `delayAlone`: `p` carries the outcome's delay alone on its channels (the
 * `delayOnly` this returns, capped again): no trims of the outcome's are flagged, a channel's negative trims are the
 * delay alone's own loop cuts, and untrimmedDb is `p`'s. Null for an outcome the solvers did not apply (no model).
 */
export function predictOutcome(o: SolveOutcome, p: EngineParams, delayAlone = false): OutcomePrediction | null {
  const m = outcomeModels.get(o);
  if (m === undefined) return null;
  const predictedDb = predictedAttenuationDb(m.model, p, m.fs);
  const out: OutcomePrediction = { predictedDb, predictionCeiling: predictionCeiling(predictedDb, m.model.coherence !== null) };
  if (o.harmonics !== undefined) {
    const capped = new Set(delayAlone ? [] : o.harmonics.filter((x) => x.capped).map((x) => x.h));
    const lowered = new Map<number, TrimLoopLimit>();
    if (!delayAlone) {
      for (const x of o.harmonics) if (x.loopLimitedDb !== undefined) lowered.set(x.h, { loopDb: Number.NaN, loweredDb: x.loopLimitedDb });
    } else {
      for (const x of o.harmonics) {
        const cut = Math.max(0, ...([0, 1] as const).filter((ch) => m.solved.channels[ch].enabled && p.channels[ch].enabled).map((ch) => -p.channels[ch].harmonicTrims[x.h - 1].gainDb));
        if (cut > LOOP_CAP_TOL_DB) lowered.set(x.h, { loopDb: Number.NaN, loweredDb: cut });
      }
    }
    const delayOnly = delayAlone ? p : m.delayOnly !== null ? delayOnlyOn(m.delayOnly, m.solved, p) : null;
    out.harmonics = harmonicSolutions(p, m.model, m.fs, capped, lowered, delayOnly);
    if (!delayAlone && delayOnly !== null) out.delayOnly = { params: delayOnly, predictedDb: predictedAttenuationDb(m.model, delayOnly, m.fs) };
  }
  return out;
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
  const model: SolveModel = { freqs: primary.freqs, primary, secondary, weights: gxx, coherence, band: cal.band, feedback: null };
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
  const model: SolveModel = { freqs, primary, secondary, weights, coherence: null, band, feedback: null };
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
