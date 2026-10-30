/**
 * Autotune: a coordinate search on the error-mic level around the current settings — delay mode:
 * delay + gain; tonal/Hilbert mode: Δ + gain — one channel at a time with the other one active.
 * The search is pure logic; all audio goes through the AutotuneDriver (the worklet in the browser,
 * the closed-loop simulator in the tests).
 *
 * Each coordinate is probed at x ± h and moved to the vertex of the parabola through the three
 * residual powers (exact for the linear gain, where the residual power is quadratic; a good local
 * model for delay and Δ near the optimum), keeping the best point actually measured. Steps halve
 * every round. Candidates outside `allowed` (the loop-gain margin) are skipped, never played.
 *
 * Reference level: bypass windows interleaved with the search (one every BYPASS_EVERY windows, one
 * before the start and one after the final window). Every error window is compared with the bypass
 * level at its own time, interpolated (dB, linear in time) between the bypass windows before and after
 * it — the ambient level may change by several dB during a run of a minute or two, so the mean of all
 * bypass windows (or the latest alone) would bias both the reported attenuation and the abort rule.
 * Abort rule: a window louder than that reference + 1 dB aborts and restores the start. A window above
 * the latest bypass window + 1 dB is decided at once by a fresh bypass window (the anti-noise is off
 * while it plays: a rising ambient level is no reason to stop); the others are judged when the next
 * bypass window arrives (a falling ambient level would hide an anti-noise that adds noise).
 * The search compares windows by their level relative to that reference too (a point measured a minute
 * earlier, under another ambient level, is not better or worse for it), and so does the final choice: the
 * best point is kept only when it beats the start relative to the bypass level at each one's time, and
 * when its final window does not confirm that, the start is restored (`improved` false). The closing bypass
 * window carries the end reference alone: one that leaves the trend of the bypass windows before it by more
 * than CLOSING_CHECK_MARGINS abort margins (a door, a passing vehicle) is measured again, and the pair is
 * averaged when it agrees, else the one nearer the trend is kept.
 *
 * Tonal mode with a driver that measures each harmonic (measureHarmonicsDb): the search refines the
 * channel's per-harmonic trims instead of Δ + gain. The trim of harmonic h multiplies the channel's
 * response there by T_h, so the error at h is E_h = P_h + T_h·A_h (A_h: the rest of the anti-noise path,
 * the other channel's part in P_h): its power is an isotropic paraboloid in the complex T_h, and the
 * harmonics do not interact (the resonators separate them; synchronous detection reads each alone). Every
 * window therefore probes all harmonics at once: T_h → T_h·(1 + z) for z = ±δ_h, ±jδ_h (at most gain ±≈ 2 dB
 * and phase ±14°, less on a harmonic already cancelled), then each harmonic moves to the vertex of the
 * paraboloid fitted (least squares) to its five powers, and keeps the best point measured. Steps halve every
 * round. The safety rules are those of the coordinate search: the abort test uses the power sum of the
 * harmonics (the tonal error level).
 */
import { FFT, nextPow2 } from '../dsp/fft.ts';
import { trimFactor, trimForTarget } from '../dsp/harmonic-trims.ts';
import {
  DELAY_CROSSFADE_MS,
  DELAY_JUMP_THRESHOLD_MS,
  DELAY_SLEW,
  GAIN_SMOOTH_MS,
  MAX_HARMONICS,
  RANGES,
  clampParams,
  computeChannelDelay,
} from '../dsp/params.ts';
import type { EngineParams, HarmonicTrim } from '../dsp/params.ts';

export interface AutotuneDriver {
  /** Sends the parameters to the engine (`jump`: restore/start, not a search step). */
  apply(p: EngineParams, jump: boolean): void;
  settle(ms: number): Promise<void>;
  /** Error-mic level in the band (dB) with the anti-noise active, averaged over `ms`. */
  measureErrorDb(ms: number): Promise<number>;
  /** Error-mic level in the band (dB) with the anti-noise bypassed; the driver toggles bypass and settles. */
  measureBypassDb(ms: number): Promise<number>;
  /**
   * Optional, tonal mode: the error-mic level at each of `freqs` (dB, one value per frequency, synchronous
   * detection over `ms`: harmonicLevelsDb(x, fs, freqs, tolHz)), with the anti-noise active, or bypassed when
   * `bypass` (the driver toggles the bypass and settles as for measureBypassDb). `tolHz`: per frequency, how
   * far the tone may have drifted from it (harmonicTolerancesHz). When present the tonal search refines the
   * per-harmonic trims (see the file comment) and measureErrorDb / measureBypassDb are not called.
   */
  measureHarmonicsDb?(ms: number, freqs: readonly number[], bypass: boolean, tolHz?: readonly number[]): Promise<number[]>;
  signal?: AbortSignal;
}

/** One harmonic of a per-harmonic tonal search (AutotuneResult.harmonics). */
export interface HarmonicTuneResult {
  /** Harmonic number (ChannelParams.harmonicTrims[h − 1]) and its frequency, Hz. */
  h: number;
  freqHz: number;
  /** Error level at this harmonic (dB): the start window's and that of the window AutotuneResult.endDb comes from. */
  startDb: number;
  endDb: number;
  /**
   * Bypassed level at this harmonic (dB) at the time of the endDb window: interpolated between the bypass
   * windows around it (see the file comment). Attenuation = bypassDb − endDb.
   */
  bypassDb: number;
  /** The same at the time of the startDb window (the attenuation before the run: startBypassDb − startDb). */
  startBypassDb?: number;
}

export interface AutotuneResult {
  /** The params in place at the end: the best point found, or the start when that did not hold (`improved`). */
  params: EngineParams;
  /**
   * Error level of the start window and of the last window measured with `params` (the final window, or the
   * start window again when the start was kept or restored). Tonal mode with measureHarmonicsDb: power sums of
   * the harmonic levels.
   */
  startDb: number;
  endDb: number;
  /** Bypassed level at the time of the endDb window, interpolated between the bypass windows around it. */
  bypassDb: number;
  /** The same at the time of the startDb window: the attenuation before the run is startBypassDb − startDb. */
  startBypassDb: number;
  /**
   * `params` differ from the start and attenuate more than it did, each against the bypass level at its own
   * time ((bypassDb − endDb) > (startBypassDb − startDb), the final window confirming the search). False when
   * aborted.
   */
  improved: boolean;
  /** Search measurements made (start and final check excluded). */
  steps: number;
  aborted: boolean;
  /** Italian, shown to the user when aborted. */
  reason?: string;
  /**
   * Per-harmonic tonal search only (absent otherwise): every tuned harmonic, measured attenuation
   * bypassDb − endDb. On an abort, endDb is the last window measured (the start params are restored).
   */
  harmonics?: HarmonicTuneResult[];
}

export interface AutotuneOptions {
  fs: number;
  channels: (0 | 1)[];
  /** Reverberation time of the room (from the measured IRs): the error needs it to settle. */
  rt60S: number;
  /** Band in which the driver measures the error level. */
  band: [number, number];
  /**
   * Statistical bandwidth (ΣS)²/ΣS² of the noise the error mic hears in `band`, Hz (calibration.ts
   * noiseBandwidthHz). Low-frequency rumble has a few tens of Hz in a nominal [20, 1000]: the windows
   * must be sized on it for B·T ≥ AUTOTUNE_BT. Default: the nominal band width.
   */
  effectiveBandHz?: number;
  /**
   * Candidates for which this is false are never played (the wizard passes the loop-gain stability
   * margin: raising a gain the solver capped would bring the loop back above it). The start is always
   * played.
   */
  allowed?: (p: EngineParams) => boolean;
}

/**
 * Abort (and restore) when an error window exceeds the bypass level at its time by more than this (≈ 2 σ
 * of the difference of two windows at B·T = AUTOTUNE_BT), widened as 1/√(B·T) when the windows are
 * capped below that B·T.
 */
export const AUTOTUNE_ABORT_MARGIN_DB = 1;
/**
 * The closing bypass window is measured again when it leaves the trend of the bypass windows before it (linear
 * through the last two, the last one alone before that) by more than this many abort margins (≈ 4 σ of the
 * difference of two windows: a ramp follows the trend, a transient does not).
 */
const CLOSING_CHECK_MARGINS = 2;
/** Bandwidth × time of every measurement window (≥ 100 ⇒ σ ≤ 0.45 dB; 150 ⇒ ≈ 0.35 dB). */
export const AUTOTUNE_BT = 150;
const MIN_WINDOW_MS = 200;
/** Longest window (a few tens of evaluations must fit in a minute or two). */
export const MAX_WINDOW_MS = 4000;
export const AUTOTUNE_ROUNDS = 3;
/** One bypass window after this many error windows (the ambient level may drift: see the file comment). */
const BYPASS_EVERY = 6;
const GAIN_STEP_DB = 1.5;
const DELTA_STEP_DEG = 12;
/** Initial delay step: this fraction of the period at the band's (geometric) centre. */
const DELAY_STEP_PERIOD = 1 / 16;
const STEP_SHRINK = 0.5;
/** Gain settling: one-pole of GAIN_SMOOTH_MS reaches 99 % in ≈ 5 τ. */
const GAIN_SETTLE_TAU = 5;
const SETTLE_MARGIN_MS = 20;
/** The parabola vertex is only measured when it is not (almost) a point already measured. */
const VERTEX_MIN_FRACTION = 0.1;
/**
 * Largest first relative step |z| of the per-harmonic trim search (T·(1 ± 0.25): +1.9 / −2.5 dB; T·(1 ± 0.25j):
 * ±14°). A harmonic already cancelled starts smaller: TRIM_STEP_PER_RESIDUAL × its residual relative to bypass
 * (|z*| = |E/(T·A)|, up to twice |E/P| with two speakers sharing the zone), so the probes neither drown the fit
 * in a large paraboloid nor undo much of the cancellation while they play.
 */
const TRIM_STEP = 0.25;
const TRIM_STEP_PER_RESIDUAL = 2;
/** Smallest relative trim step: the fit must still see the paraboloid above the measurement noise. */
const TRIM_MIN_STEP = 0.04;
/** The fitted vertex moves a trim by at most this many steps per round (the fit is local). */
const TRIM_MAX_MOVE = 2;

const fmt = (x: number): string => x.toFixed(1).replace('.', ',').replace('-', '−');
const toPow = (db: number): number => Math.pow(10, db / 10);
const toDb = (pow: number): number => 10 * Math.log10(Math.max(pow, 1e-30));
const powerSumDb = (levels: readonly number[]): number => toDb(levels.reduce((a, db) => a + toPow(db), 0));

/**
 * How far each harmonic's tone may sit from its nominal frequency for harmonicLevelsDb: the resonator's half
 * bandwidth f/(2Q) (a tone beyond it is not reproduced anyway), at most HARMONIC_DRIFT_MAX of f. A source's
 * f0 drifts during a run of a minute or two (0.2 % moves harmonic 4 of 100 Hz by 0.8 Hz), while a 4 s window
 * reads ±0.25 Hz at −6 dB: an exact-frequency lock-in would read the drifted harmonic tens of dB low.
 */
export const HARMONIC_DRIFT_MAX = 0.01;
export function harmonicTolerancesHz(p: EngineParams, freqs: readonly number[]): number[] {
  const q = clampParams(p).tonal.q;
  return freqs.map((f) => Math.min(f / (2 * q), HARMONIC_DRIFT_MAX * f));
}

/**
 * Synchronous (lock-in) level of `x` at each frequency: a Hann-windowed DFT over the whole record (bandwidth
 * ≈ 1.5/T, so broadband noise barely moves it), dBFS RMS (a full-scale sine reads −3.01 dB; −200 for
 * silence). What a driver's measureHarmonicsDb returns for its recorded error window.
 * With `tolHz` (one per frequency, or one for all; harmonicTolerancesHz): the highest level within ±tol of
 * each frequency (zero-padded FFT, parabolic peak interpolation), so a drifted tone is read at its own
 * frequency. On broadband noise alone that maximum reads a few dB above the exact-frequency level (the
 * largest of a few dozen bins): a pessimistic floor, never an optimistic one.
 */
export function harmonicLevelsDb(
  x: Float32Array | Float64Array,
  fs: number,
  freqs: readonly number[],
  tolHz?: number | readonly number[],
): number[] {
  const n = x.length;
  let wsum = 0;
  for (let i = 0; i < n; i++) wsum += 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  const tolOf = (i: number): number => {
    const t = typeof tolHz === 'number' ? tolHz : tolHz?.[i];
    return t !== undefined && Number.isFinite(t) && t > 0 ? t : 0;
  };
  let spectrum: { pow: Float64Array; df: number } | null = null;
  return freqs.map((f, idx) => {
    if (!(n > 0) || !(fs > 0) || !Number.isFinite(f)) return -200;
    const tol = tolOf(idx);
    if (tol > 0) {
      spectrum ??= hannSpectrum(x, fs);
      const { pow, df } = spectrum;
      const half = pow.length - 1;
      const lo = Math.max(1, Math.ceil((f - tol) / df));
      const hi = Math.min(half - 1, Math.floor((f + tol) / df));
      if (lo <= hi) {
        let k = lo;
        for (let j = lo + 1; j <= hi; j++) if (pow[j] > pow[k]) k = j;
        // Parabola through the dB values around the largest bin (Hann, 2× zero-padded: < 0.1 dB).
        const [a, b, c] = [pow[k - 1], pow[k], pow[k + 1]].map((v) => 10 * Math.log10(Math.max(v, 1e-300)));
        const d = a - 2 * b + c;
        const top = d < 0 ? b - ((a - c) * (a - c)) / (8 * d) : b;
        const amp2 = (4 * Math.pow(10, top / 10)) / (wsum * wsum);
        const power = amp2 / 2;
        return power > 0 ? 10 * Math.log10(power) : -200;
      }
    }
    const w = (2 * Math.PI * f) / fs;
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const v = (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)) * x[i];
      re += v * Math.cos(w * i);
      im -= v * Math.sin(w * i);
    }
    const amp = (2 * Math.hypot(re, im)) / wsum;
    const power = (amp * amp) / 2;
    return power > 0 ? 10 * Math.log10(power) : -200;
  });
}

/** |DFT|² of the Hann-windowed record, zero-padded to ≥ 2n (bins ≤ 1/(2T) apart), bins 0 … N/2. */
function hannSpectrum(x: Float32Array | Float64Array, fs: number): { pow: Float64Array; df: number } {
  const n = x.length;
  const size = nextPow2(2 * n);
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  for (let i = 0; i < n; i++) re[i] = (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)) * x[i];
  new FFT(size).forward(re, im);
  const pow = new Float64Array(size / 2 + 1);
  for (let k = 0; k < pow.length; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
  return { pow, df: fs / size };
}

/** Harmonics a per-harmonic search tunes: h ≤ tonal.harmonics (and MAX_HARMONICS) inside `band`, below 0.45·fs. */
export function tunedHarmonics(p: EngineParams, band: [number, number], fs: number): { h: number; freqHz: number }[] {
  const q = clampParams(p);
  const lo = Math.min(band[0], band[1]);
  const hi = Math.min(Math.max(band[0], band[1]), 0.45 * fs);
  const out: { h: number; freqHz: number }[] = [];
  for (let h = 1; h <= Math.min(q.tonal.harmonics, MAX_HARMONICS); h++) {
    const f = h * q.tonal.f0;
    if (f >= lo && f <= hi && f < 0.45 * fs) out.push({ h, freqHz: f });
  }
  return out;
}

/**
 * Least-squares fit of y = a·|z|² + b1·Re z + b2·Im z + d (an isotropic paraboloid in the complex z) and its
 * vertex −(b1 + j·b2)/(2a); null with fewer than four points, a singular fit or no minimum (a ≤ 0).
 */
function paraboloidVertex(zr: readonly number[], zi: readonly number[], y: readonly number[]): { re: number; im: number } | null {
  if (y.length < 4) return null;
  // Normal equations M·θ = v, rows [|z|², Re z, Im z, 1].
  const m = new Float64Array(16);
  const v = new Float64Array(4);
  for (let j = 0; j < y.length; j++) {
    const row = [zr[j] * zr[j] + zi[j] * zi[j], zr[j], zi[j], 1];
    for (let r = 0; r < 4; r++) {
      v[r] += row[r] * y[j];
      for (let c = 0; c < 4; c++) m[r * 4 + c] += row[r] * row[c];
    }
  }
  // Gaussian elimination with partial pivoting.
  const scale = Math.max(...m.map(Math.abs));
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let r = col + 1; r < 4; r++) if (Math.abs(m[r * 4 + col]) > Math.abs(m[piv * 4 + col])) piv = r;
    if (!(Math.abs(m[piv * 4 + col]) > 1e-12 * scale)) return null;
    for (let c = 0; c < 4; c++) [m[col * 4 + c], m[piv * 4 + c]] = [m[piv * 4 + c], m[col * 4 + c]];
    [v[col], v[piv]] = [v[piv], v[col]];
    for (let r = col + 1; r < 4; r++) {
      const f = m[r * 4 + col] / m[col * 4 + col];
      for (let c = col; c < 4; c++) m[r * 4 + c] -= f * m[col * 4 + c];
      v[r] -= f * v[col];
    }
  }
  const th = new Float64Array(4);
  for (let r = 3; r >= 0; r--) {
    let s = v[r];
    for (let c = r + 1; c < 4; c++) s -= m[r * 4 + c] * th[c];
    th[r] = s / m[r * 4 + r];
  }
  const a = th[0];
  if (!(a > 0) || !th.every(Number.isFinite)) return null;
  return { re: -th[1] / (2 * a), im: -th[2] / (2 * a) };
}

/**
 * Measurement window, settling time, the standard deviation of one window's level (dB, Gaussian
 * noise: 10/ln10/√(B·T)) and the abort margin for the given options. B is the noise's effective
 * bandwidth when known (never above the nominal band width).
 */
export function autotuneTiming(o: AutotuneOptions): { windowMs: number; settleMs: number; sigmaDb: number; abortMarginDb: number } {
  const nominal = Math.abs(o.band[1] - o.band[0]);
  const eff = o.effectiveBandHz;
  let bw = nominal > 0 ? nominal : 100;
  if (eff !== undefined && Number.isFinite(eff) && eff > 0) bw = Math.min(bw, eff);
  const windowMs = Math.min(MAX_WINDOW_MS, Math.max(MIN_WINDOW_MS, Math.ceil((1000 * AUTOTUNE_BT) / bw)));
  const rt60Ms = Number.isFinite(o.rt60S) && o.rt60S > 0 ? Math.min(o.rt60S, 5) * 1000 : 0;
  const settleMs = Math.max(GAIN_SETTLE_TAU * GAIN_SMOOTH_MS, DELAY_CROSSFADE_MS, rt60Ms) + SETTLE_MARGIN_MS;
  const bt = (bw * windowMs) / 1000;
  const sigmaDb = 10 / Math.LN10 / Math.sqrt(bt);
  // 1 dB ≈ 2 σ of the difference of two windows at B·T = AUTOTUNE_BT: shorter B·T widens it alike.
  const abortMarginDb = AUTOTUNE_ABORT_MARGIN_DB * Math.max(1, Math.sqrt(AUTOTUNE_BT / bt));
  return { windowMs, settleMs: Math.ceil(settleMs), sigmaDb, abortMarginDb };
}

class Stop extends Error {}

/** A point of the coordinate search: its level and the time of its window (on the run's clock). */
interface Point {
  p: EngineParams;
  db: number;
  t: number;
}

/**
 * A measured window: its centre on the run's clock (ms of settling and windows asked of the driver), the
 * level (the power sum in the per-harmonic search) and the harmonics' levels (per-harmonic search only).
 */
interface Window {
  t: number;
  db: number;
  levels: number[] | null;
}

/**
 * `of` of the bypass windows at time t (dB): linear in time between the bypass windows before and after t,
 * the nearest one before the first or after the last (never extrapolated); NaN without any.
 */
function bypassAt(byp: readonly Window[], t: number, of: (w: Window) => number): number {
  if (byp.length === 0) return Number.NaN;
  const i = byp.findIndex((w) => w.t >= t);
  if (i < 0) return of(byp[byp.length - 1]);
  if (i === 0) return of(byp[0]);
  const a = byp[i - 1];
  const b = byp[i];
  const u = (t - a.t) / (b.t - a.t);
  return of(a) + u * (of(b) - of(a));
}

/**
 * A point of the per-harmonic search: the level of each tuned harmonic and the time of the window it came from;
 * `whole`: all from one window.
 */
interface HPoint {
  p: EngineParams;
  levels: number[];
  times: number[];
  whole: boolean;
}

interface Coord {
  get(p: EngineParams): number;
  set(p: EngineParams, v: number): EngineParams;
}

const withChannel = (p: EngineParams, ch: 0 | 1, patch: Partial<EngineParams['channels'][0]>): EngineParams => {
  const q = clampParams(p);
  q.channels[ch] = { ...q.channels[ch], ...patch };
  return clampParams(q);
};

const delayCoord = (ch: 0 | 1): Coord => ({
  get: (p) => p.channels[ch].delayMs,
  set: (p, v) => withChannel(p, ch, { delayMs: v }),
});

const deltaCoord = (ch: 0 | 1): Coord => ({
  get: (p) => p.channels[ch].phaseDeltaDeg,
  set: (p, v) => {
    let d = ((v + 180) % 360 + 360) % 360 - 180;
    if (d === -180) d = 180;
    return withChannel(p, ch, { phaseDeltaDeg: d });
  },
});

/** Gain as a linear amplitude: the residual power is exactly quadratic in it. */
const gainCoord = (ch: 0 | 1): Coord => ({
  get: (p) => Math.pow(10, p.channels[ch].gainDb / 20),
  set: (p, v) => withChannel(p, ch, { gainDb: v > 0 ? 20 * Math.log10(v) : RANGES.gainDb.min }),
});

const ONE = { re: 1, im: 0 };

/** p with channel ch's trims of the tuned harmonics replaced (clamped copy). */
function withTrims(p: EngineParams, ch: 0 | 1, hs: readonly number[], trims: readonly HarmonicTrim[]): EngineParams {
  const q = clampParams(p);
  hs.forEach((h, i) => (q.channels[ch].harmonicTrims[h - 1] = { ...trims[i] }));
  return clampParams(q);
}

/**
 * Coordinate search of the error level (see the file comment). Never throws: driver errors, invalid
 * measurements, the abort signal and the "worse than bypass + 1 dB" guard all restore the start
 * parameters (apply(start, true)) and return aborted=true with an Italian reason.
 */
export async function runAutotune(driver: AutotuneDriver, start: EngineParams, o: AutotuneOptions): Promise<AutotuneResult> {
  const p0 = clampParams(start);
  const fs = o.fs;
  const allowed = o.allowed ?? ((): boolean => true);
  const { windowMs, settleMs, abortMarginDb } = autotuneTiming(o);
  const jumpSamples = (DELAY_JUMP_THRESHOLD_MS / 1000) * fs;
  // Tonal mode with per-harmonic levels: the trim search (see the file comment).
  const tuned = p0.phaseMode === 'tonal' && driver.measureHarmonicsDb !== undefined ? tunedHarmonics(p0, o.band, fs) : [];
  const perHarmonic = tuned.length > 0;
  const freqs = tuned.map((t) => t.freqHz);
  const tolHz = harmonicTolerancesHz(p0, freqs);
  const hs = tuned.map((t) => t.h);
  let applied: EngineParams | null = null;
  let steps = 0;
  let sinceBypass = 0;
  /** The run's clock (ms): settling and windows asked of the driver (the driver's own bypass settling is not counted). */
  let clock = 0;
  const bypassWins: Window[] = [];
  /** Error windows not yet judged against a bypass window after them (the abort rule, file comment). */
  let unjudged: Window[] = [];
  // Set in closures (TypeScript would narrow a plain `= null` to null for good).
  let first = null as Window | null;
  let last = null as Window | null;

  const settle = async (ms: number): Promise<void> => {
    await driver.settle(ms);
    clock += ms;
  };

  /** Bypass level (power sum) and, per tuned harmonic, bypass levels at time t (file comment). */
  const bypassDbAt = (t: number): number => bypassAt(bypassWins, t, (w) => w.db);
  const harmonicBypassAt = (t: number, i: number): number => bypassAt(bypassWins, t, (w) => w.levels?.[i] ?? Number.NaN);

  const harmonicsResult = (end: Window | null): HarmonicTuneResult[] | undefined =>
    perHarmonic
      ? tuned.map((t, i) => {
          const at = end ?? first;
          const x: HarmonicTuneResult = {
            h: t.h,
            freqHz: t.freqHz,
            startDb: first?.levels?.[i] ?? Number.NaN,
            endDb: at?.levels?.[i] ?? Number.NaN,
            bypassDb: at !== null ? harmonicBypassAt(at.t, i) : Number.NaN,
          };
          if (first !== null) x.startBypassDb = harmonicBypassAt(first.t, i);
          return x;
        })
      : undefined;

  const check = (): void => {
    if (driver.signal?.aborted) throw new Stop('Autotune annullato: ripristinati i parametri iniziali.');
  };

  /** Extra time the delay taps need to slew from `a` to `b` (big changes crossfade within settleMs). */
  const slewMs = (a: EngineParams | null, b: EngineParams): number => {
    if (a === null || !(fs > 0)) return 0;
    let ms = 0;
    for (const ch of [0, 1] as const) {
      const d = Math.abs(computeChannelDelay(a, fs, ch).samples - computeChannelDelay(b, fs, ch).samples);
      if (d < jumpSamples) ms = Math.max(ms, (d / DELAY_SLEW / fs) * 1000);
    }
    return ms;
  };

  /** One window: the band level, or (per-harmonic search) the harmonics' levels and their power sum. */
  const measureWindow = async (bypass: boolean): Promise<Window> => {
    const t = clock + windowMs / 2;
    let w: Window;
    if (!perHarmonic) {
      w = { t, db: await (bypass ? driver.measureBypassDb(windowMs) : driver.measureErrorDb(windowMs)), levels: null };
    } else {
      const levels = await driver.measureHarmonicsDb!(windowMs, freqs, bypass, tolHz);
      const ok = Array.isArray(levels) && levels.length === freqs.length && levels.every(Number.isFinite);
      w = ok ? { t, db: powerSumDb(levels), levels: [...levels] } : { t, db: Number.NaN, levels: null };
    }
    clock += windowMs;
    return w;
  };

  /** A window's levels as one list: the level (power sum) and, per-harmonic search, each harmonic's. */
  const levelsOf = (w: Window): number[] => [w.db, ...(w.levels ?? [])];
  /** Largest distance of w's levels from the trend of the bypass windows so far (linear through the last two). */
  const offTrend = (w: Window): number => {
    const n = bypassWins.length;
    const b = levelsOf(bypassWins[n - 1]);
    const a = n > 1 ? levelsOf(bypassWins[n - 2]) : null;
    const u = n > 1 ? (w.t - bypassWins[n - 1].t) / (bypassWins[n - 1].t - bypassWins[n - 2].t) : 0;
    return Math.max(...levelsOf(w).map((v, i) => Math.abs(v - (b[i] + (a !== null ? u * (b[i] - a[i]) : 0)))));
  };
  /**
   * The closing bypass window, checked against the trend (file comment): measured again when it leaves it; the
   * pair's mean (at their mean time) when the two agree — the level really moved — else the one nearer the trend.
   */
  const closingCheck = async (w: Window): Promise<Window> => {
    const limit = CLOSING_CHECK_MARGINS * abortMarginDb;
    if (bypassWins.length === 0 || !(offTrend(w) > limit)) return w;
    check();
    const again = await measureWindow(true);
    if (!Number.isFinite(again.db)) return w;
    const b = levelsOf(again);
    if (Math.max(...levelsOf(w).map((v, i) => Math.abs(v - b[i]))) <= limit) {
      const levels = w.levels !== null && again.levels !== null ? w.levels.map((v, i) => (v + again.levels![i]) / 2) : null;
      return { t: (w.t + again.t) / 2, db: levels !== null ? powerSumDb(levels) : (w.db + again.db) / 2, levels };
    }
    return offTrend(again) < offTrend(w) ? again : w;
  };

  /**
   * A bypass window; the error windows since the previous one now lie between two and are judged against the
   * level interpolated at their time (`settleAfter` false: nothing is measured after it; `closing`: the last
   * one, the end reference, checked by closingCheck).
   */
  const measureBypass = async (settleAfter = true, closing = false): Promise<void> => {
    check();
    let w = await measureWindow(true);
    if (!Number.isFinite(w.db)) throw new Stop('Misura del livello in bypass non valida: autotune interrotto.');
    if (closing) w = await closingCheck(w);
    bypassWins.push(w);
    sinceBypass = 0;
    const due = unjudged;
    unjudged = [];
    for (const e of due) {
      const ref = bypassDbAt(e.t);
      if (e.db > ref + abortMarginDb) {
        last = e;
        throw new Stop(
          `Con l'ANC attivo l'errore (${fmt(e.db)} dB) supera il bypass (${fmt(ref)} dB) di oltre ` +
            `${fmt(abortMarginDb)} dB: autotune interrotto, ripristinati i parametri iniziali.`,
        );
      }
    }
    if (settleAfter) await settle(settleMs);
  };

  const evaluate = async (p: EngineParams, jump = false): Promise<Window> => {
    check();
    if (sinceBypass >= BYPASS_EVERY) await measureBypass();
    const wait = settleMs + slewMs(applied, p);
    driver.apply(p, jump);
    applied = p;
    await settle(wait);
    check();
    const w = await measureWindow(false);
    sinceBypass++;
    if (!Number.isFinite(w.db)) throw new Stop("Misura del livello d'errore non valida: autotune interrotto.");
    last = w;
    unjudged.push(w);
    // Above the latest bypass window: a fresh one decides at once (the abort rule, file comment).
    if (w.db > bypassWins[bypassWins.length - 1].db + abortMarginDb) await measureBypass();
    return w;
  };

  /** A point's level relative to the bypass level at its window's time (file comment). */
  const relDb = (pt: Point): number => pt.db - bypassDbAt(pt.t);

  /** Probes x0 ± h, then the parabola vertex; returns the best point measured (relative to bypass: relDb). */
  const lineSearch = async (cur: Point, coord: Coord, h: number): Promise<Point> => {
    const x0 = coord.get(cur.p);
    const pts: { x: number; pt: Point }[] = [{ x: x0, pt: cur }];
    for (const x of [x0 - h, x0 + h]) {
      const p = coord.set(cur.p, x);
      const xa = coord.get(p);
      if (Math.abs(xa - x0) < 1e-9 * Math.max(1, Math.abs(x0)) || !allowed(p)) continue;
      const w = await evaluate(p);
      steps++;
      pts.push({ x: xa, pt: { p, db: w.db, t: w.t } });
    }
    if (pts.length === 3) {
      const [a, b, c] = [...pts].sort((u, v) => u.x - v.x);
      // Vertex of the parabola through (x, power) for unequal spacing.
      const ya = toPow(relDb(a.pt));
      const yb = toPow(relDb(b.pt));
      const yc = toPow(relDb(c.pt));
      const d1 = (yb - ya) / (b.x - a.x);
      const d2 = (yc - yb) / (c.x - b.x);
      const curv = (d2 - d1) / (c.x - a.x);
      if (curv > 0) {
        // y = ya + d1·(x − a) + curv·(x − a)(x − b)  ⇒  y' = 0 at x = (a + b)/2 − d1/(2·curv)
        const xv = Math.min(x0 + 2 * h, Math.max(x0 - 2 * h, (a.x + b.x) / 2 - d1 / (2 * curv)));
        const near = pts.some((q) => Math.abs(q.x - xv) < VERTEX_MIN_FRACTION * h);
        const p = coord.set(cur.p, xv);
        if (!near && allowed(p)) {
          const w = await evaluate(p);
          steps++;
          pts.push({ x: coord.get(p), pt: { p, db: w.db, t: w.t } });
        }
      }
    }
    let best = pts[0].pt;
    for (const q of pts) if (relDb(q.pt) < relDb(best)) best = q.pt;
    return best;
  };

  /**
   * One round of the per-harmonic trim search on channel ch (file comment): the current point (measured
   * as a whole first if it was assembled from several windows), four probes T·(1 ± δ), T·(1 ± jδ), the
   * fitted vertices (each harmonic moved by at most TRIM_MAX_MOVE·δ), then per harmonic the best point
   * measured. Probes and vertices raise a trim by ≤ 3.5 dB, under BIG_JUMP_DB, a window (≥ 0.3 s) apart:
   * the engine applies them at once (20 ms glide). A point assembled from several windows is returned
   * (and later played) only when `allowed`; otherwise the best allowed combination built greedily from the
   * measured point with the lowest total.
   */
  const trimSearch = async (cur: HPoint, ch: 0 | 1, step: Float64Array): Promise<HPoint> => {
    if (!cur.whole) {
      const m = await evaluate(cur.p);
      steps++;
      cur = { p: cur.p, levels: m.levels!, times: m.levels!.map(() => m.t), whole: true };
    }
    const base = cur.p;
    const t0 = hs.map((h) => trimFactor(base.channels[ch].harmonicTrims[h - 1]));
    const zeros = hs.map(() => 0);
    // lv/tm: each harmonic's level and its window's time; y: its power relative to the bypass level at that time
    // (relative: refreshed before every use, as later bypass windows refine the interpolation).
    const pts: { p: EngineParams; zr: number[]; zi: number[]; lv: number[]; tm: number[]; y: number[] }[] = [
      { p: base, zr: zeros, zi: zeros, lv: cur.levels, tm: cur.times, y: [] },
    ];
    const relative = (): void => {
      for (const q of pts) q.y = q.lv.map((db, i) => toPow(db - harmonicBypassAt(q.tm[i], i)));
    };
    const probe = async (z: { re: number; im: number }[]): Promise<void> => {
      // T·(1 + z), as the trim the engine applies (clamped to its range).
      const trims = t0.map((t, i) => trimForTarget({ re: t.re * (1 + z[i].re) - t.im * z[i].im, im: t.re * z[i].im + t.im * (1 + z[i].re) }, ONE));
      const p = withTrims(base, ch, hs, trims);
      if (!allowed(p)) return;
      const m = await evaluate(p);
      steps++;
      // Where the probe really is (a trim at the end of its range moves less): (T_played / T) − 1.
      const zr: number[] = [];
      const zi: number[] = [];
      hs.forEach((h, i) => {
        const t = trimFactor(p.channels[ch].harmonicTrims[h - 1]);
        const d = t0[i].re * t0[i].re + t0[i].im * t0[i].im;
        zr.push((t.re * t0[i].re + t.im * t0[i].im) / d - 1);
        zi.push((t.im * t0[i].re - t.re * t0[i].im) / d);
      });
      pts.push({ p, zr, zi, lv: m.levels!, tm: m.levels!.map(() => m.t), y: [] });
    };
    for (const [dr, di] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      await probe(hs.map((_, i) => ({ re: dr * step[i], im: di * step[i] })));
    }
    relative();
    const vertex = hs.map((_, i) => {
      const v = paraboloidVertex(
        pts.map((q) => q.zr[i]),
        pts.map((q) => q.zi[i]),
        pts.map((q) => q.y[i]),
      );
      if (v === null) return null;
      const r = Math.hypot(v.re, v.im);
      const lim = TRIM_MAX_MOVE * step[i];
      return r > lim ? { re: (v.re * lim) / r, im: (v.im * lim) / r } : v;
    });
    // Measured only when some harmonic's vertex is not (almost) a point already measured.
    const fresh = vertex.some((v, i) => v !== null && pts.every((q) => Math.hypot(q.zr[i] - v.re, q.zi[i] - v.im) >= VERTEX_MIN_FRACTION * step[i]));
    if (fresh) await probe(vertex.map((v) => v ?? { re: 0, im: 0 }));
    relative();
    // Best point per harmonic: the harmonics do not interact.
    const pick = hs.map((_, i) => {
      let best = 0;
      for (let j = 1; j < pts.length; j++) if (pts[j].y[i] < pts[best].y[i]) best = j;
      return best;
    });
    const from = (src: number[]): Pick<HPoint, 'levels' | 'times'> => ({
      levels: src.map((j, i) => pts[j].lv[i]),
      times: src.map((j, i) => pts[j].tm[i]),
    });
    if (pick.every((j) => j === pick[0])) return { p: pts[pick[0]].p, ...from(pick), whole: true };
    const assembled = withTrims(base, ch, hs, hs.map((h, i) => pts[pick[i]].p.channels[ch].harmonicTrims[h - 1]));
    if (allowed(assembled)) return { p: assembled, ...from(pick), whole: false };
    // The assembled point was never played and is outside `allowed` (overlapping bands: the loop between two
    // harmonics adds up from both): start from the measured point with the lowest total and take the other
    // per-harmonic bests, largest gain first, only while the result stays allowed.
    const total = (j: number): number => pts[j].y.reduce((a, v) => a + v, 0);
    let lowest = 0;
    for (let j = 1; j < pts.length; j++) if (total(j) < total(lowest)) lowest = j;
    const src = hs.map(() => lowest);
    const order = hs.map((_, i) => i).sort((a, b) => pts[pick[b]].y[b] - pts[lowest].y[b] - (pts[pick[a]].y[a] - pts[lowest].y[a]));
    let p = pts[lowest].p;
    for (const i of order) {
      if (pick[i] === lowest) continue;
      const next = withTrims(p, ch, [hs[i]], [pts[pick[i]].p.channels[ch].harmonicTrims[hs[i] - 1]]);
      if (!allowed(next)) continue;
      p = next;
      src[i] = pick[i];
    }
    return { p, ...from(src), whole: src.every((j) => j === lowest) };
  };

  /**
   * The chosen params measured as a whole, then (unless one already followed it) the closing bypass window, and
   * the result: `final` is kept only when that window attenuates more than the start window did, each against
   * the bypass level at its time; otherwise the start is restored (the search's best point owed its lead to the
   * noise or to a change of the ambient level) and reported with its own window.
   */
  const conclude = async (final: EngineParams): Promise<AutotuneResult> => {
    const start = first!;
    const end = await evaluate(final);
    if (unjudged.length > 0) await measureBypass(false, true);
    const attenuation = (w: Window): number => bypassDbAt(w.t) - w.db;
    const improved = final !== p0 && attenuation(end) > attenuation(start);
    if (final !== p0 && !improved) {
      driver.apply(p0, true);
      await driver.settle(settleMs);
    }
    const shown = final === p0 || improved ? end : start;
    const out: AutotuneResult = {
      params: improved ? final : p0,
      startDb: start.db,
      endDb: shown.db,
      bypassDb: bypassDbAt(shown.t),
      startBypassDb: bypassDbAt(start.t),
      steps,
      aborted: false,
      improved,
    };
    const harmonics = harmonicsResult(shown);
    if (harmonics !== undefined) out.harmonics = harmonics;
    return out;
  };

  try {
    await measureBypass();
    const w0 = await evaluate(p0, true);
    first = w0;
    const startDb = w0.db;
    const chans = [...new Set(o.channels)].filter((ch) => (ch === 0 || ch === 1) && p0.channels[ch].enabled);
    if (perHarmonic) {
      let cur: HPoint = { p: p0, levels: w0.levels!, times: w0.levels!.map(() => w0.t), whole: true };
      const residual = w0.levels!.map((db, i) => Math.pow(10, (db - harmonicBypassAt(w0.t, i)) / 20));
      const step0 = residual.map((r) => Math.min(TRIM_STEP, Math.max(TRIM_MIN_STEP, TRIM_STEP_PER_RESIDUAL * r)));
      const step = chans.map(() => Float64Array.from(step0));
      for (let round = 0; round < AUTOTUNE_ROUNDS; round++) {
        for (const [k, ch] of chans.entries()) {
          cur = await trimSearch(cur, ch, step[k]);
          step[k] = step[k].map((h) => Math.max(TRIM_MIN_STEP, h * STEP_SHRINK));
        }
      }
      // Each level against the bypass level at its time, referred to the start window's (file comment).
      const atStart = (x: Pick<HPoint, 'levels' | 'times'>): number =>
        powerSumDb(x.levels.map((db, i) => db - harmonicBypassAt(x.times[i], i) + harmonicBypassAt(w0.t, i)));
      return await conclude(atStart(cur) < atStart({ levels: w0.levels!, times: w0.levels!.map(() => w0.t) }) ? cur.p : p0);
    }
    const startPoint: Point = { p: p0, db: startDb, t: w0.t };
    let best = startPoint;
    const fc =
      p0.phaseMode === 'tonal' ? p0.tonal.f0 : Math.sqrt(Math.max(1, Math.min(o.band[0], o.band[1])) * Math.max(o.band[0], o.band[1]));
    let hDelayMs = (DELAY_STEP_PERIOD * 1000) / fc;
    let hDeltaDeg = DELTA_STEP_DEG;
    let hGainDb = GAIN_STEP_DB;
    for (let round = 0; round < AUTOTUNE_ROUNDS; round++) {
      for (const ch of chans) {
        if (p0.phaseMode === 'delay') best = await lineSearch(best, delayCoord(ch), hDelayMs);
        else best = await lineSearch(best, deltaCoord(ch), hDeltaDeg);
        const g = gainCoord(ch);
        best = await lineSearch(best, g, g.get(best.p) * (Math.pow(10, hGainDb / 20) - 1));
      }
      hDelayMs *= STEP_SHRINK;
      hDeltaDeg *= STEP_SHRINK;
      hGainDb *= STEP_SHRINK;
    }
    return await conclude(relDb(best) < relDb(startPoint) ? best.p : p0);
  } catch (e) {
    const reason =
      e instanceof Stop ? e.message : `Autotune interrotto (${e instanceof Error ? e.message : String(e)}): ripristinati i parametri iniziali.`;
    try {
      driver.apply(p0, true);
      await driver.settle(settleMs);
    } catch {
      // Restoring is best effort: the parameters were sent, nothing else can be done here.
    }
    // endDb: the window that stopped the run (else the last one measured), against the bypass level at its time.
    const out: AutotuneResult = {
      params: p0,
      startDb: first?.db ?? Number.NaN,
      endDb: last?.db ?? Number.NaN,
      bypassDb: bypassDbAt(last?.t ?? clock),
      startBypassDb: first !== null ? bypassDbAt(first.t) : Number.NaN,
      steps,
      aborted: true,
      improved: false,
      reason,
    };
    const harmonics = harmonicsResult(last);
    if (harmonics !== undefined) out.harmonics = harmonics;
    return out;
  }
}
