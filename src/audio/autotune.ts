/**
 * Autotune: a coordinate search on the error-mic level around the current settings — delay mode:
 * delay + gain; tonal/Hilbert mode: Δ + gain — one channel at a time with the other one active.
 * The search is pure logic; all audio goes through the AutotuneDriver (the worklet in the browser,
 * the closed-loop simulator in the tests).
 *
 * Each coordinate is probed at x ± h and moved to the vertex of the parabola through the three
 * residual powers (exact for the linear gain, where the residual power is quadratic; a good local
 * model for delay and Δ near the optimum), keeping the best point actually measured. Steps halve
 * every round. Any window louder than the latest bypass window + 1 dB aborts and restores the start.
 * Candidates outside `allowed` (the loop-gain margin) are skipped, never played.
 */
import {
  DELAY_CROSSFADE_MS,
  DELAY_JUMP_THRESHOLD_MS,
  DELAY_SLEW,
  GAIN_SMOOTH_MS,
  RANGES,
  clampParams,
  computeChannelDelay,
} from '../dsp/params.ts';
import type { EngineParams } from '../dsp/params.ts';

export interface AutotuneDriver {
  /** Sends the parameters to the engine (`jump`: restore/start, not a search step). */
  apply(p: EngineParams, jump: boolean): void;
  settle(ms: number): Promise<void>;
  /** Error-mic level in the band (dB) with the anti-noise active, averaged over `ms`. */
  measureErrorDb(ms: number): Promise<number>;
  /** Error-mic level in the band (dB) with the anti-noise bypassed; the driver toggles bypass and settles. */
  measureBypassDb(ms: number): Promise<number>;
  signal?: AbortSignal;
}

export interface AutotuneResult {
  params: EngineParams;
  startDb: number;
  endDb: number;
  /** Power average of the bypass windows. */
  bypassDb: number;
  /** Search measurements made (start and final check excluded). */
  steps: number;
  aborted: boolean;
  /** Italian, shown to the user when aborted. */
  reason?: string;
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
 * Abort (and restore) when an error window exceeds the latest bypass window by more than this (≈ 2 σ
 * of the difference of two windows at B·T = AUTOTUNE_BT), widened as 1/√(B·T) when the windows are
 * capped below that B·T.
 */
export const AUTOTUNE_ABORT_MARGIN_DB = 1;
/** Bandwidth × time of every measurement window (≥ 100 ⇒ σ ≤ 0.45 dB; 150 ⇒ ≈ 0.35 dB). */
export const AUTOTUNE_BT = 150;
const MIN_WINDOW_MS = 200;
/** Longest window (a few tens of evaluations must fit in a minute or two). */
export const MAX_WINDOW_MS = 4000;
export const AUTOTUNE_ROUNDS = 3;
/** One bypass window after this many error windows (the ambient level may drift). */
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

const fmt = (x: number): string => x.toFixed(1).replace('.', ',').replace('-', '−');
const toPow = (db: number): number => Math.pow(10, db / 10);

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

interface Point {
  p: EngineParams;
  db: number;
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
  let applied: EngineParams | null = null;
  let steps = 0;
  let lastBypass = Number.NaN;
  let sinceBypass = 0;
  const bypassPow: number[] = [];
  let startDb = Number.NaN;
  let lastDb = Number.NaN;

  const bypassDb = (): number =>
    bypassPow.length > 0 ? 10 * Math.log10(bypassPow.reduce((a, b) => a + b, 0) / bypassPow.length) : Number.NaN;

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

  const measureBypass = async (): Promise<void> => {
    check();
    const db = await driver.measureBypassDb(windowMs);
    if (!Number.isFinite(db)) throw new Stop('Misura del livello in bypass non valida: autotune interrotto.');
    lastBypass = db;
    bypassPow.push(toPow(db));
    sinceBypass = 0;
    await driver.settle(settleMs);
  };

  const evaluate = async (p: EngineParams, jump = false): Promise<number> => {
    check();
    if (sinceBypass >= BYPASS_EVERY) await measureBypass();
    const wait = settleMs + slewMs(applied, p);
    driver.apply(p, jump);
    applied = p;
    await driver.settle(wait);
    check();
    const db = await driver.measureErrorDb(windowMs);
    sinceBypass++;
    if (!Number.isFinite(db)) throw new Stop("Misura del livello d'errore non valida: autotune interrotto.");
    lastDb = db;
    if (db > lastBypass + abortMarginDb) {
      throw new Stop(
        `Con l'ANC attivo l'errore (${fmt(db)} dB) supera il bypass (${fmt(lastBypass)} dB) di oltre ` +
          `${fmt(abortMarginDb)} dB: autotune interrotto, ripristinati i parametri iniziali.`,
      );
    }
    return db;
  };

  /** Probes x0 ± h, then the parabola vertex; returns the best point measured. */
  const lineSearch = async (cur: Point, coord: Coord, h: number): Promise<Point> => {
    const x0 = coord.get(cur.p);
    const pts: { x: number; pt: Point }[] = [{ x: x0, pt: cur }];
    for (const x of [x0 - h, x0 + h]) {
      const p = coord.set(cur.p, x);
      const xa = coord.get(p);
      if (Math.abs(xa - x0) < 1e-9 * Math.max(1, Math.abs(x0)) || !allowed(p)) continue;
      const db = await evaluate(p);
      steps++;
      pts.push({ x: xa, pt: { p, db } });
    }
    if (pts.length === 3) {
      const [a, b, c] = [...pts].sort((u, v) => u.x - v.x);
      // Vertex of the parabola through (x, power) for unequal spacing.
      const ya = toPow(a.pt.db);
      const yb = toPow(b.pt.db);
      const yc = toPow(c.pt.db);
      const d1 = (yb - ya) / (b.x - a.x);
      const d2 = (yc - yb) / (c.x - b.x);
      const curv = (d2 - d1) / (c.x - a.x);
      if (curv > 0) {
        // y = ya + d1·(x − a) + curv·(x − a)(x − b)  ⇒  y' = 0 at x = (a + b)/2 − d1/(2·curv)
        const xv = Math.min(x0 + 2 * h, Math.max(x0 - 2 * h, (a.x + b.x) / 2 - d1 / (2 * curv)));
        const near = pts.some((q) => Math.abs(q.x - xv) < VERTEX_MIN_FRACTION * h);
        const p = coord.set(cur.p, xv);
        if (!near && allowed(p)) {
          const db = await evaluate(p);
          steps++;
          pts.push({ x: coord.get(p), pt: { p, db } });
        }
      }
    }
    let best = pts[0].pt;
    for (const q of pts) if (q.pt.db < best.db) best = q.pt;
    return best;
  };

  try {
    await measureBypass();
    startDb = await evaluate(p0, true);
    let best: Point = { p: p0, db: startDb };
    const chans = [...new Set(o.channels)].filter((ch) => (ch === 0 || ch === 1) && p0.channels[ch].enabled);
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
    const final = best.db < startDb ? best.p : p0;
    const endDb = await evaluate(final);
    return { params: final, startDb, endDb, bypassDb: bypassDb(), steps, aborted: false };
  } catch (e) {
    const reason =
      e instanceof Stop ? e.message : `Autotune interrotto (${e instanceof Error ? e.message : String(e)}): ripristinati i parametri iniziali.`;
    try {
      driver.apply(p0, true);
      await driver.settle(settleMs);
    } catch {
      // Restoring is best effort: the parameters were sent, nothing else can be done here.
    }
    return {
      params: p0,
      startDb,
      endDb: Number.isFinite(lastDb) ? lastDb : startDb,
      bypassDb: bypassDb(),
      steps,
      aborted: true,
      reason,
    };
  }
}
