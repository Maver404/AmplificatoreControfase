/**
 * Pure helpers of the guided setup (no DOM, no audio): levels, master ramp timing, causality,
 * geometry input, saved calibrations and their quick re-check, loop gain, loopback signatures,
 * error-mic levels, preferences. Tested in tests/unit/wizard-logic.test.ts and wizard-review.test.ts.
 */
import type { AudioStats } from '../../audio/audio-engine.ts';
import {
  calibrationFreqs,
  capLoopGain,
  checkLoop,
  LOOP_GAIN_DANGER_DB,
  LOOP_GAIN_LIMIT_DB,
  loopbackSuspectPaths,
  pathResponse,
  sanitizeCalibration,
} from '../../audio/calibration.ts';
import type { CalibrationData, LoopGains, PathData, SolveOutcome } from '../../audio/calibration.ts';
import type { HowlReason } from '../../audio/protocol.ts';
import type { Preset } from '../../app/presets.ts';
import type { SpeakerConfig } from '../../dsp/alignment.ts';
import type { Complex } from '../../dsp/complex.ts';
import { RISE_FLOOR_DB } from '../../dsp/gain-ramp.ts';
import { BIG_JUMP_DB, GAIN_SMOOTH_MS, MAX_RISE_DB_PER_S, RANGES } from '../../dsp/params.ts';
import type { ChannelParams, EngineParams } from '../../dsp/params.ts';
import { attenuationDb, geometryDelay, headStartSeconds, speedOfSound } from '../../dsp/physics.ts';
import { conditioningResponse, groupDelaySamples } from '../../dsp/response.ts';
import { formatNumber, parseNumber } from '../units.ts';

export const CH_NAME = ['L', 'R'] as const;

/** Channels that carry anti-noise for a speaker configuration (one speaker = L only). */
export const activeChannels = (s: SpeakerConfig): (0 | 1)[] => (s === 'single' ? [0] : [0, 1]);

// ------------------------------------------------------------------------------------- levels

/** Master volume during the amplifier-level step and the measurements: the stimulus level is the output level. */
export const AMP_STEP_MASTER_DB = 0;
/** Amplifier-level noise RMS and sweep peak, dBFS at the output (plan: −20 dBFS). */
const STIMULUS_DBFS = -20;
/** Pink noise peaks sit ≈ 12 dB above its RMS: the stimulus stays this far below the ceiling. */
const CEILING_HEADROOM_DB = 8;

/**
 * Output levels of the guided stimuli: −20 dBFS, lowered with a low ceiling (headphones −24 dBFS)
 * so that the limiter does not squash them.
 */
export function stimulusLevels(ceilingDb: number): { noiseRmsDbfs: number; sweepPeakDbfs: number } {
  const c = Number.isFinite(ceilingDb) ? ceilingDb : RANGES.ceilingDb.max;
  const v = Math.min(STIMULUS_DBFS, c - CEILING_HEADROOM_DB);
  return { noiseRmsDbfs: v, sweepPeakDbfs: v };
}

/** Raised-cosine fade-in/out of `ms` on both ends (in place), so looped noise chunks do not click. */
export function fadeEdges(x: Float32Array, fs: number, ms: number): Float32Array {
  const n = Math.min(Math.floor(x.length / 2), Math.max(0, Math.round((ms / 1000) * fs)));
  for (let i = 0; i < n; i++) {
    const g = 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / n);
    x[i] *= g;
    x[x.length - 1 - i] *= g;
  }
  return x;
}

/**
 * Worst-case time (ms) the engine's master GainRamp needs, on an armed output, to settle within
 * ~0.1 dB of `toDb` from `fromDb`: below RISE_FLOOR_DB a fast one-pole, then a linear rise at
 * MAX_RISE_DB_PER_S while the gap is above BIG_JUMP_DB, then the last BIG_JUMP_DB paced by the rise
 * budget (empty in the worst case) and the one-pole tail. Decreases only need the one-pole.
 */
export function masterSettleMs(fromDb: number, toDb: number): number {
  const tailMs = 5 * GAIN_SMOOTH_MS;
  if (!Number.isFinite(toDb) || toDb <= RISE_FLOOR_DB) return tailMs;
  const start = Number.isFinite(fromDb) && fromDb > RISE_FLOOR_DB ? fromDb : RISE_FLOOR_DB;
  const floorMs = Number.isFinite(fromDb) && fromDb >= RISE_FLOOR_DB ? 0 : 2 * GAIN_SMOOTH_MS;
  const gap = toDb - start;
  if (gap <= 0) return floorMs + tailMs;
  const linearS = Math.max(0, gap - BIG_JUMP_DB) / MAX_RISE_DB_PER_S;
  const budgetS = Math.min(gap, BIG_JUMP_DB) / MAX_RISE_DB_PER_S;
  return floorMs + (linearS + budgetS) * 1000 + tailMs;
}

/**
 * Follows the master target the controller sends, to know when the engine's rate-limited master
 * has really reached it (a stimulus started earlier plays quieter and then grows by up to tens of
 * dB). Conservative: a change while a rise is still running adds its whole settling time.
 */
export class MasterRampTracker {
  private target: number | null = null;
  private readyAt = 0;

  /** Call with every controller state (master target, armed output) and the current time in ms. */
  observe(masterDb: number, armed: boolean, nowMs: number): void {
    if (this.target === null || !armed) {
      // Disarmed, parameters apply at once (and arming starts from them).
      this.target = masterDb;
      if (!armed) this.readyAt = nowMs;
      return;
    }
    if (masterDb === this.target) return;
    this.readyAt = Math.max(this.readyAt, nowMs) + masterSettleMs(this.target, masterDb);
    this.target = masterDb;
  }

  /** Milliseconds until the master is settled (0 = settled). */
  readyInMs(nowMs: number): number {
    return Math.max(0, this.readyAt - nowMs);
  }
}

export type RefLevel = 'none' | 'low' | 'ok' | 'high';

/** Advice on the reference-mic input gain from its RMS and peak levels (dBFS). */
export function refLevelAdvice(rmsDb: number | null, peakDb: number | null): { level: RefLevel; text: string } {
  if (rmsDb === null || !Number.isFinite(rmsDb) || rmsDb <= -110) {
    return { level: 'none', text: 'Nessun segnale dal microfono di riferimento: controlla il collegamento e il guadagno d’ingresso.' };
  }
  if (peakDb !== null && Number.isFinite(peakDb) && peakDb > -3) {
    return {
      level: 'high',
      text: 'Il microfono di riferimento è vicino alla saturazione: abbassa un po’ il guadagno d’ingresso dell’interfaccia.',
    };
  }
  if (rmsDb < -50) {
    return {
      level: 'low',
      text:
        'Il rumore arriva molto debole al microfono di riferimento: alza il guadagno d’ingresso dell’interfaccia ' +
        '(meglio che alzare l’amplificatore), lasciando i picchi sotto −6 dBFS.',
    };
  }
  return { level: 'ok', text: 'Livello del microfono di riferimento adeguato.' };
}

/** Power mean of dB values (null entries skipped); null when nothing is left. */
export function powerMeanDb(list: readonly (number | null)[]): number | null {
  const v = list.filter((x): x is number => x !== null && Number.isFinite(x));
  if (v.length === 0) return null;
  return 10 * Math.log10(v.reduce((a, x) => a + Math.pow(10, x / 10), 0) / v.length);
}

// ------------------------------------------------------------------------------------- latency and causality

/** Electronic latency from the browser's figures (input + base + output), seconds; null when unknown. */
export function estimatedLatencyS(stats: AudioStats): number | null {
  const out = stats.outputLatencyMs;
  if (out === null || !Number.isFinite(out) || out <= 0) return null;
  const add = (v: number | null): number => (v !== null && Number.isFinite(v) && v > 0 ? v : 0);
  return (out + add(stats.baseLatencyMs) + add(stats.trackLatencyMs)) / 1000;
}

/** Group delay (s) of the conditioning chain at the centre of the working band. */
export function conditioningDelayS(p: EngineParams, fs: number, band: [number, number]): number {
  const hi = p.filters.lpEnabled ? Math.min(band[1], p.filters.lpFreq) : band[1];
  const fc = p.phaseMode === 'tonal' ? p.tonal.f0 : Math.sqrt(Math.max(1, band[0]) * Math.max(band[0] + 1, hi));
  const h = (f: number): Complex => conditioningResponse(p, fs, f);
  const s = groupDelaySamples(h, fc, fs) / fs;
  return Number.isFinite(s) && s > 0 ? s : 0;
}

export interface CausalityInput {
  micToZoneM: number;
  sourceAngleDeg: number;
  /** Farthest speaker → zone distance among the active channels. */
  speakerToZoneM: number;
  tempC: number;
  latencyS: number;
  conditioningS: number;
}

export interface CausalityReport {
  /** Time the noise needs from the reference mic to the zone. */
  headStartS: number;
  /** Time the anti-noise needs: latency + speaker → zone + filters. */
  neededS: number;
  marginS: number;
  causal: boolean;
  /** How much farther upstream the mic should go (m), 0 if causal. */
  extraLeadM: number;
  cosTheta: number;
  warnAngle: boolean;
}

export function causality(i: CausalityInput): CausalityReport {
  const c = speedOfSound(i.tempC);
  const headStartS = headStartSeconds({ micToZoneM: i.micToZoneM, sourceAngleDeg: i.sourceAngleDeg, tempC: i.tempC });
  const secondaryS = i.latencyS + Math.max(0, i.speakerToZoneM) / c;
  const g = geometryDelay(headStartS, secondaryS, i.conditioningS, i.tempC, i.sourceAngleDeg);
  const neededS = secondaryS + i.conditioningS;
  return {
    headStartS,
    neededS,
    marginS: headStartS - neededS,
    causal: g.causal,
    extraLeadM: g.requiredExtraLeadM,
    cosTheta: g.cosTheta,
    warnAngle: g.warnAngle,
  };
}

// ------------------------------------------------------------------------------------- geometry input

export const GEOMETRY_RANGES = {
  micToZoneM: { min: 0.1, max: 200 },
  speakerToZoneM: { min: 0.05, max: 50 },
  sourceAngleDeg: { min: 0, max: 180 },
  tempC: { min: -30, max: 50 },
} as const;

/** What the user typed in the placement step (kept as text so a half-typed value is not lost). */
export interface GeometryDraft {
  micToZoneM: string;
  speakerToZoneM: [string, string];
  sourceAngleDeg: string;
  tempC: string;
}

export interface GeometryValues {
  micToZoneM: number;
  speakerToZoneM: [number, number];
  sourceAngleDeg: number;
  tempC: number;
}

export const DEFAULT_GEOMETRY: GeometryDraft = {
  micToZoneM: '',
  speakerToZoneM: ['', ''],
  sourceAngleDeg: '0',
  tempC: '20',
};

/** A number inside [min, max] parsed from Italian or English text, else null. */
export function parseInRange(text: string, r: { min: number; max: number }): number | null {
  const v = parseNumber(text);
  return v !== null && Number.isFinite(v) && v >= r.min && v <= r.max ? v : null;
}

/** Parsed geometry, or the Italian errors (one per invalid field the active channels need). */
export function parseGeometry(
  d: GeometryDraft,
  channels: readonly (0 | 1)[],
): { ok: true; value: GeometryValues } | { ok: false; errors: string[] } {
  const R = GEOMETRY_RANGES;
  const errors: string[] = [];
  const range = (r: { min: number; max: number }, unit: string): string =>
    `tra ${formatNumber(r.min, r.min < 1 ? 2 : 0)} e ${formatNumber(r.max, 0)} ${unit}`;
  const mic = parseInRange(d.micToZoneM, R.micToZoneM);
  if (mic === null) errors.push(`Distanza microfono → zona: inserisci un valore ${range(R.micToZoneM, 'm')}.`);
  const spk: [number, number] = [0, 0];
  for (const ch of channels) {
    const v = parseInRange(d.speakerToZoneM[ch], R.speakerToZoneM);
    if (v === null) errors.push(`Distanza cassa ${CH_NAME[ch]} → zona: inserisci un valore ${range(R.speakerToZoneM, 'm')}.`);
    else spk[ch] = v;
  }
  const angle = parseInRange(d.sourceAngleDeg, R.sourceAngleDeg);
  if (angle === null) errors.push(`Angolo della sorgente: inserisci un valore ${range(R.sourceAngleDeg, '°')}.`);
  const temp = parseInRange(d.tempC, R.tempC);
  if (temp === null) errors.push(`Temperatura: inserisci un valore tra −30 e 50 °C.`);
  if (errors.length > 0 || mic === null || angle === null || temp === null) return { ok: false, errors };
  // An unused R speaker keeps the L distance (the calibration record needs a finite value).
  if (!channels.includes(1)) spk[1] = spk[0];
  return { ok: true, value: { micToZoneM: mic, speakerToZoneM: spk, sourceAngleDeg: angle, tempC: temp } };
}

/** Draft from stored numbers (calibration geometry), Italian decimals. */
export function geometryToDraft(g: GeometryValues): GeometryDraft {
  const t = (v: number): string => String(Number(v.toPrecision(6))).replace('.', ',');
  return {
    micToZoneM: t(g.micToZoneM),
    speakerToZoneM: [t(g.speakerToZoneM[0]), t(g.speakerToZoneM[1])],
    sourceAngleDeg: t(g.sourceAngleDeg),
    tempC: t(g.tempC),
  };
}

/**
 * The distances a measured path of channel `ch` depends on (mic → zone, speaker ch → zone), as a
 * key; null while one of them is not a valid number. Editing them means the hardware moved.
 */
export function positionsKey(d: GeometryDraft, ch: 0 | 1): string | null {
  const mic = parseInRange(d.micToZoneM, GEOMETRY_RANGES.micToZoneM);
  const spk = parseInRange(d.speakerToZoneM[ch], GEOMETRY_RANGES.speakerToZoneM);
  return mic === null || spk === null ? null : `${mic}|${spk}`;
}

/** Everything a geometry-based solution depends on (distances, angle, temperature); null if invalid. */
export function geometryKey(d: GeometryDraft, channels: readonly (0 | 1)[]): string | null {
  const g = parseGeometry(d, channels);
  if (!g.ok) return null;
  const v = g.value;
  return [v.micToZoneM, ...channels.map((ch) => v.speakerToZoneM[ch]), v.sourceAngleDeg, v.tempC].join('|');
}

// ------------------------------------------------------------------------------------- saved calibrations

/** Presets holding a valid calibration, most recent first. */
export function presetsWithCalibration(list: readonly Preset[]): { preset: Preset; calibration: CalibrationData }[] {
  const out: { preset: Preset; calibration: CalibrationData }[] = [];
  for (const p of list) {
    const c = p.calibration !== undefined ? sanitizeCalibration(p.calibration) : null;
    if (c !== null) out.push({ preset: p, calibration: c });
  }
  return out.sort((a, b) => b.preset.createdAt - a.preset.createdAt);
}

/** "Calibrazione 30/10/2026 14:05". */
export function defaultPresetName(d: Date): string {
  const two = (n: number): string => String(n).padStart(2, '0');
  return `Calibrazione ${two(d.getDate())}/${two(d.getMonth() + 1)}/${d.getFullYear()} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

/** Calibration paths that look like an electrical loopback (cable still connected), as Italian names. */
export function loopbackSuspects(cal: CalibrationData | null): string[] {
  return loopbackSuspectPaths(cal);
}

/** The plain-JSON PathData part of a measurement result (drops ok/reason/aborted). */
export function toPathData(r: PathData): PathData {
  const p: PathData = {
    delaySamples: r.delaySamples,
    sign: r.sign,
    qualityDb: r.qualityDb,
    stdSamples: r.stdSamples,
    freqs: r.freqs,
    re: r.re,
    im: r.im,
    ir: r.ir,
    irOffset: r.irOffset,
  };
  if (r.rt60S !== undefined) p.rt60S = r.rt60S;
  return p;
}

export interface PathSummary {
  delayMs: number;
  inverted: boolean;
  qualityDb: number;
  /** Largest |H| on the stored grid, dB (final output → mic). */
  peakDb: number | null;
}

export function pathSummary(p: PathData, fs: number): PathSummary {
  let m = 0;
  for (let k = 0; k < p.re.length; k++) m = Math.max(m, Math.hypot(p.re[k] ?? 0, p.im[k] ?? 0));
  return {
    delayMs: (p.delaySamples / fs) * 1000,
    inverted: p.sign < 0,
    qualityDb: p.qualityDb,
    peakDb: m > 0 ? 20 * Math.log10(m) : null,
  };
}

/** Room decay for the autotune's settling: the longest measured RT60 of the zone paths (default 0.3 s). */
export function rt60ForAutotune(cal: CalibrationData | null): number {
  const v = cal ? [cal.secondary[0]?.rt60S, cal.secondary[1]?.rt60S].filter((x): x is number => x !== undefined && x > 0) : [];
  const r = v.length > 0 ? Math.max(...v) : 0.3;
  return Math.min(2, Math.max(0.05, r));
}

/** A solve outcome worth applying: some channel is used and the predicted attenuation is ≥ 1 dB. */
export const usefulOutcome = (o: SolveOutcome | null | undefined): o is SolveOutcome =>
  o !== null && o !== undefined && o.result.channels.some((c) => c.use) && o.result.predictedAttenuationDb >= 1;

/**
 * Params whose channel `ch` comes from `perZone[ch]` (separate zones are solved one at a time). A zone
 * without a useful solution (null) has its channel switched off: its speaker would otherwise play the
 * unsolved values (the defaults or an older solution) into a zone it does not cancel.
 */
export function mergeZoneSolutions(base: EngineParams, perZone: [EngineParams | null, EngineParams | null]): EngineParams {
  const pick = (ch: 0 | 1): ChannelParams => {
    const z = perZone[ch];
    return z !== null ? { ...z.channels[ch] } : { ...base.channels[ch], enabled: false };
  };
  return { ...base, channels: [pick(0), pick(1)] };
}

/** Loop gain with both speakers, raw and after neutralisation (calibration.ts checkLoop). */
export type LoopCheck = LoopGains;

/**
 * Per-zone loop-gain figures (computed with one speaker each): replaced by the merged check. A zone the
 * solver refused to change for stability ("… i parametri non sono stati cambiati") keeps that reason.
 */
export const isLoopAdvice = (text: string): boolean =>
  /^(Rischio di innesco|Guadagno d'anello)/.test(text) && !/non sono stati cambiati/.test(text);

/** Italian advice on the loop gain of both speakers playing together (separate zones); null below the limit. */
export function mergedLoopAdvice(c: LoopCheck, taps: number): string | null {
  if (!(c.rawDb > LOOP_GAIN_LIMIT_DB)) return null;
  const at = `${formatNumber(c.atHz, 0)} Hz`;
  const head = `Con entrambe le casse attive il guadagno d'anello cassa → microfono di riferimento è ${formatNumber(c.rawDb, 1)} dB a ${at}`;
  if (c.afterDb > LOOP_GAIN_DANGER_DB) {
    return (
      `Rischio di innesco: ${head.charAt(0).toLowerCase()}${head.slice(1)} e resta ${formatNumber(c.afterDb, 1)} dB anche con la neutralizzazione ` +
      'del feedback. Allontana o orienta le casse lontano dal microfono di riferimento, oppure riduci il guadagno.'
    );
  }
  if (c.afterDb > LOOP_GAIN_LIMIT_DB) {
    return (
      `${head}: con la neutralizzazione del feedback (${taps} tap) scende a ${formatNumber(c.afterDb, 1)} dB, sopra il limite di ` +
      `${formatNumber(LOOP_GAIN_LIMIT_DB, 0)} dB. Margine di stabilità ridotto: aumenta i tap del neutralizzatore (Fase e filtri › Avanzate) ` +
      'o allontana le casse dal microfono di riferimento.'
    );
  }
  return `${head}: la neutralizzazione del feedback (automatica, ${taps} tap) lo porta a ${formatNumber(c.afterDb, 1)} dB.`;
}

/**
 * Separate zones are solved one zone at a time, each with the other speaker off: their loop-gain
 * figures, advice and stability limit see one speaker only. Checks the merged parameters instead (both
 * speakers feed back into the reference mic together): when their neutralised loop exceeds
 * LOOP_GAIN_DANGER_DB, both channel gains are lowered by the excess (the loop scales with them), as the
 * solvers do for one zone. Returns the params to apply, the dB taken off, the loop gains (raw and after
 * neutralisation) and the advice with the per-zone loop statements replaced.
 */
export function mergedZonesLoop(
  cal: CalibrationData,
  merged: EngineParams,
  fs: number,
  taps: number,
  advice: string[],
): { params: EngineParams; limitedDb: number; loopGainDb: number | null; loopGainAfterDb: number | null; advice: string[] } {
  const kept = advice.filter((a) => !isLoopAdvice(a.replace(/^Zona [LR]: /, '')));
  const { params, limitedDb, loop: c } = capLoopGain(cal, merged, fs, taps);
  if (c === null) return { params, limitedDb, loopGainDb: null, loopGainAfterDb: null, advice: kept };
  const a = mergedLoopAdvice(c, taps);
  return { params, limitedDb, loopGainDb: c.rawDb, loopGainAfterDb: c.afterDb, advice: a ? [...kept, a] : kept };
}

/** Rounding allowance of loopGuard (a solve leaves the loop exactly at the margin). */
const LOOP_GUARD_TOL_DB = 0.05;

/**
 * Loop-gain limit of a search around `start` (the step-6 autotune): true when the neutralised loop of
 * `p` (checkLoop with `taps`) stays within LOOP_GAIN_DANGER_DB, or within the start's own loop when
 * that is already higher (never worse than the start). Always true without feedback data.
 */
export function loopGuard(cal: CalibrationData, start: EngineParams, fs: number, taps: number): (p: EngineParams) => boolean {
  const startDb = checkLoop(cal, start, fs, taps)?.afterDb ?? -Infinity;
  const maxDb = Math.max(LOOP_GAIN_DANGER_DB, startDb) + LOOP_GUARD_TOL_DB;
  return (p) => !((checkLoop(cal, p, fs, taps)?.afterDb ?? -Infinity) > maxDb);
}

/**
 * Best attenuation left when every channel gain was lowered by `limitedDb` for stability (a pure gain
 * error of the cancelling signal): −10·log10(1 + g² − 2g), g = 10^(−limitedDb/20).
 */
export const gainLimitedAttenuationDb = (limitedDb: number): number =>
  limitedDb > 0 ? -attenuationDb(Math.pow(10, -limitedDb / 20), 0) : Infinity;

/** Copies one channel of a planar tap block into `dst` from `filled`; returns the new fill. */
export function appendTapChannel(block: Float32Array, frames: number, chIndex: number, dst: Float32Array, filled: number): number {
  const n = Math.min(frames, dst.length - filled);
  if (n <= 0) return filled;
  dst.set(block.subarray(chIndex * frames, chIndex * frames + n), filled);
  return filled + n;
}

// ------------------------------------------------------------------------------------- reuse of a saved calibration


/** Tolerances of the quick check of a saved calibration (plan: L_e repeatable within 0.5 samples). */
export const QUICK_CHECK = {
  delaySamples: 0.5,
  gainDb: 0.5,
  /** Beyond these the setup changed too much to correct: measure again. */
  maxGainDb: 6,
  maxDelayMs: 3,
} as const;

export interface BaselineDiff {
  /** now − saved, samples (positive = later). */
  delaySamples: number;
  /** In-band power ratio now / saved, dB. */
  gainDb: number;
  /** The polarity of the path changed (rewired). */
  signFlipped: boolean;
}

/** Change of a re-measured path against the saved one: arrival delay and in-band level. */
export function compareToBaseline(saved: PathData, now: PathData, fs: number, band: [number, number]): BaselineDiff | null {
  const freqs = calibrationFreqs(fs, band);
  if (freqs.length === 0) return null;
  const a = pathResponse(saved, fs, freqs);
  const b = pathResponse(now, fs, freqs);
  let pa = 0;
  let pb = 0;
  for (let k = 0; k < freqs.length; k++) {
    pa += a.re[k]! * a.re[k]! + a.im[k]! * a.im[k]!;
    pb += b.re[k]! * b.re[k]! + b.im[k]! * b.im[k]!;
  }
  if (!(pa > 0) || !(pb > 0)) return null;
  return { delaySamples: now.delaySamples - saved.delaySamples, gainDb: 10 * Math.log10(pb / pa), signFlipped: now.sign !== saved.sign };
}

export type QuickVerdict =
  | { kind: 'ok'; diff: BaselineDiff }
  | { kind: 'corrected'; diff: BaselineDiff; channel: ChannelParams; text: string }
  | { kind: 'mismatch'; diff: BaselineDiff | null; text: string };

/**
 * What to do with channel `ch` after its quick re-measurement: keep the saved values (within the
 * tolerances), correct gain and delay by the measured change (the amplifier was re-levelled by
 * ear, the latency moved a little), or ask for new measurements (too big a change, polarity
 * flipped, a delay that cannot go further down).
 */
export function quickVerdict(diff: BaselineDiff | null, ch: ChannelParams, fs: number): QuickVerdict {
  if (diff === null) return { kind: 'mismatch', diff, text: 'La misura non è confrontabile con quella salvata.' };
  if (diff.signFlipped) {
    return { kind: 'mismatch', diff, text: 'La polarità del percorso è cambiata (collegamenti invertiti?): rifai le misure.' };
  }
  const dMs = (diff.delaySamples / fs) * 1000;
  if (Math.abs(diff.delaySamples) <= QUICK_CHECK.delaySamples && Math.abs(diff.gainDb) <= QUICK_CHECK.gainDb) return { kind: 'ok', diff };
  if (Math.abs(diff.gainDb) > QUICK_CHECK.maxGainDb || Math.abs(dMs) > QUICK_CHECK.maxDelayMs) {
    return {
      kind: 'mismatch',
      diff,
      text:
        `Il percorso è cambiato troppo (livello ${formatNumber(diff.gainDb, 1)} dB, ritardo ${formatNumber(dMs, 2)} ms): ` +
        'qualcosa è stato spostato o l’amplificatore è molto diverso. Rifai le misure.',
    };
  }
  const next: ChannelParams = { ...ch };
  const parts: string[] = [];
  if (Math.abs(diff.gainDb) > QUICK_CHECK.gainDb) {
    next.gainDb = ch.gainDb - diff.gainDb;
    if (next.gainDb < RANGES.gainDb.min || next.gainDb > RANGES.gainDb.max) {
      return {
        kind: 'mismatch',
        diff,
        text: `Servirebbe un guadagno di ${formatNumber(next.gainDb, 1)} dB, fuori dai limiti: riporta l’amplificatore dov’era o rifai le misure.`,
      };
    }
    parts.push(`guadagno ${formatNumber(ch.gainDb, 1)} → ${formatNumber(next.gainDb, 1)} dB (percorso ${formatNumber(diff.gainDb, 1)} dB)`);
  }
  if (Math.abs(diff.delaySamples) > QUICK_CHECK.delaySamples) {
    // Same shift on every path through the interface: the anti-noise must leave that much earlier.
    next.delayMs = ch.delayMs - dMs;
    if (next.delayMs < RANGES.delayMs.min) {
      return {
        kind: 'mismatch',
        diff,
        text: `La latenza è cresciuta di ${formatNumber(dMs, 2)} ms e il ritardo non può scendere sotto zero: rifai le misure.`,
      };
    }
    parts.push(`ritardo ${formatNumber(ch.delayMs, 3)} → ${formatNumber(next.delayMs, 3)} ms (latenza ${formatNumber(dMs, 3)} ms)`);
  }
  const text = parts.join('; ');
  return { kind: 'corrected', diff, channel: next, text: `${text.charAt(0).toUpperCase()}${text.slice(1)}.` };
}

// ------------------------------------------------------------------------------------- error-mic levels and alarms

/**
 * Synchronous (lock-in) level of the harmonics h·f0 inside `band`, power-summed, dBFS RMS: a
 * Hann-windowed DFT at each harmonic over the whole record (bandwidth ≈ 1.5/T), so broadband
 * noise in the band barely moves it. null when no harmonic falls in the band.
 */
export function harmonicLevelDbfs(x: Float32Array, fs: number, f0: number, harmonics: number, band: [number, number]): number | null {
  const n = x.length;
  if (n < 16 || !(f0 > 0) || !(fs > 0)) return null;
  const lo = Math.min(band[0], band[1]);
  const hi = Math.min(Math.max(band[0], band[1]), 0.45 * fs);
  let wsum = 0;
  for (let i = 0; i < n; i++) wsum += 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  let power = 0;
  let any = false;
  for (let h = 1; h <= Math.max(1, Math.round(harmonics)); h++) {
    const f = h * f0;
    if (f < lo || f > hi) continue;
    any = true;
    const w = (2 * Math.PI * f) / fs;
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const v = (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)) * x[i]!;
      re += v * Math.cos(w * i);
      im -= v * Math.sin(w * i);
    }
    const amp = (2 * Math.hypot(re, im)) / wsum;
    power += (amp * amp) / 2;
  }
  if (!any) return null;
  return power > 0 ? 10 * Math.log10(power) : -200;
}

const HOWL_WHAT: Record<HowlReason, string> = {
  'peak-growth': 'picco in crescita',
  'limiter-saturated': 'limiter saturo',
  'output-overload': 'uscita in sovraccarico',
  'anc-worse': 'l’anti-rumore aumentava il rumore',
};

/** Italian alarm for a howl event of the controller (detail "<reason>[ @ <Hz> Hz]"). */
export function howlAlarmText(detail: string | undefined): string {
  const m = /^([a-z-]+)(?: @ (\d+) Hz)?$/.exec(detail ?? '');
  const reason = m && m[1]! in HOWL_WHAT ? (m[1] as HowlReason) : null;
  const hz = m?.[2] !== undefined ? ` a ${m[2]} Hz` : '';
  if (reason === 'anc-worse') {
    return 'L’ANC stava aumentando il rumore nella zona: uscita silenziata. Controlla posizioni e misure (passi 4–6).';
  }
  if (reason === 'output-overload') {
    return `Uscita in sovraccarico${hz}: uscita silenziata. Abbassa il volume master o i guadagni (o l’amplificatore).`;
  }
  const what = reason ? `${HOWL_WHAT[reason]}${hz}` : 'protezione di sicurezza';
  return `Innesco rilevato (${what}): uscita silenziata. Abbassa l’amplificatore o allontana la cassa dal microfono di riferimento.`;
}

// ------------------------------------------------------------------------------------- preferences

export type LayoutChoice = 'A' | 'B';

/** Remembered wizard choices (per browser), plain JSON. */
export interface WizardPrefs {
  layout: LayoutChoice;
  geometry: GeometryDraft;
}

export const WIZARD_PREFS_KEY = 'controfase.wizard.v1';

export function parseWizardPrefs(raw: string | null): WizardPrefs {
  const d: WizardPrefs = { layout: 'A', geometry: { ...DEFAULT_GEOMETRY, speakerToZoneM: [...DEFAULT_GEOMETRY.speakerToZoneM] } };
  if (!raw) return d;
  try {
    const o = JSON.parse(raw) as Record<string, unknown>;
    const str = (v: unknown, fallback: string): string => (typeof v === 'string' && v.length <= 32 ? v : fallback);
    const g = (typeof o['geometry'] === 'object' && o['geometry'] !== null ? o['geometry'] : {}) as Record<string, unknown>;
    const spk = Array.isArray(g['speakerToZoneM']) ? (g['speakerToZoneM'] as unknown[]) : [];
    return {
      layout: o['layout'] === 'B' ? 'B' : 'A',
      geometry: {
        micToZoneM: str(g['micToZoneM'], d.geometry.micToZoneM),
        speakerToZoneM: [str(spk[0], ''), str(spk[1], '')],
        sourceAngleDeg: str(g['sourceAngleDeg'], d.geometry.sourceAngleDeg),
        tempC: str(g['tempC'], d.geometry.tempC),
      },
    };
  } catch {
    return d;
  }
}

export const serializeWizardPrefs = (p: WizardPrefs): string => JSON.stringify(p);

// ------------------------------------------------------------------------------------- formatting

export const fmtMs = (s: number, decimals = 1): string => `${formatNumber(s * 1000, decimals)} ms`;
export const fmtDb = (db: number, decimals = 1): string => `${formatNumber(db, decimals)} dB`;
export const fmtM = (m: number, decimals = 2): string => `${formatNumber(m, decimals)} m`;
