/**
 * The complete real-time anti-phase chain, shared by the AudioWorklet, the unit tests and the closed-loop
 * simulator. Pure TypeScript: no globals, the sample rate is passed in, double precision throughout.
 *
 * Per sample: x = in[refChannel]; x̂ = x − feedback prediction; cond = conditioning(x̂); delay line;
 * per channel: smooth delay tap → [Hilbert rotation] → polarity → channel gain + master (one ramp, so
 * both share the rise limit) → soft-start · mute · bypass · reconfigure · channel soft-start · measurement
 * gate → (+ calibration stimulus × master) → howl detector (pre-limiter) → limiter and final clip →
 * output (exact zeros while not armed).
 *
 * Rises are never completed unheard: while the bypass or the measurement gate holds the anti-noise at 0
 * (both reopen in 15 ms) gain increases and soft-starts wait, and a channel that is enabled, or whose path
 * gets broader (filter section removed, tonal mode left, other reference input), soft-starts from 0.
 *
 * handleMessage() may allocate; process() allocates nothing except the outgoing message objects
 * (meters, tap blocks, measurement results, events), which drain() hands to the caller.
 */
import type { Ch2Status, HowlReason, LevelReading, WorkletInMsg, WorkletOutMsg } from '../audio/protocol.ts';
import { TAP_BLOCK_FRAMES, TAP_CHANNELS } from '../audio/protocol.ts';
import { COEFF_RAMP_MS } from './biquad.ts';
import { FeedbackCanceller } from './feedback-canceller.ts';
import { DelayLine, SmoothDelayTap } from './fractional-delay.ts';
import { GainRamp, RISE_FLOOR_DB } from './gain-ramp.ts';
import { PhaseRotator } from './hilbert-pair.ts';
import { HowlDetector } from './howl-detector.ts';
import { PeakLimiter } from './limiter.ts';
import { LevelMeter } from './meters.ts';
import {
  DELAY_JUMP_THRESHOLD_MS,
  MASTER_MIN_DB,
  MAX_RISE_DB_PER_S,
  MUTE_RAMP_MS,
  PANIC_RAMP_MS,
  SOFT_START_MS,
  clampParams,
  computeChannelDelay,
  delayCapacity,
} from './params.ts';
import type { EngineParams, PhaseMode } from './params.ts';
import { ConditioningChain } from './response.ts';
import { LinearEnvelope } from './smoother.ts';
import { TapCollector } from './tap-buffer.ts';

export interface EngineOptions {
  softStartMs?: number /* SOFT_START_MS */;
  meterHz?: number /* 30 */;
  /** Tests only: bypass DC blocker/HP/LP/resonators (cond = raw reference). */
  testBypassConditioning?: boolean;
}

/** Longest feedback-neutralisation FIR (taps) and bulk delay the engine accepts. */
export const FEEDBACK_MAX_TAPS = 1024;
export const FEEDBACK_MAX_BULK_MS = 500;
/** Longest measurement (stimulus + post-roll) the engine records. */
export const MAX_MEASURE_S = 120;
/** Interval of the channel-2 check. */
export const CH2_WINDOW_MS = 1000;
const CH2_SILENT_DBFS = -90;
const CH2_DUPLICATE_CORR = 0.999;
const CH2_DUPLICATE_DB = 0.1;
const PROGRESS_HZ = 10;
const TAP_POOL_SIZE = 8;
/** Beyond this many undrained messages, meters/progress/taps are dropped (events are always kept). */
const MAX_QUEUE = 256;
/** The 4-tap Lagrange read at delay d reaches back to sample n − ⌊d⌋ − 2. */
const LAGRANGE_REACH = 3;
/**
 * Backstop for howls the HowlDetector cannot see (above its ~2.4 kHz periodicity band, ~1.2 kHz for loops
 * of 600–1100 ms; loops longer than 1.1 s): a limiter whose gain reduction, smoothed over
 * LIMITER_BACKSTOP_TAU_MS, stays above LIMITER_BACKSTOP_DB for LIMITER_BACKSTOP_MS is muted as
 * 'limiter-saturated', whatever the content.
 * GR alone cannot tell a howl from loud noise (broadband noise 6 dB below the ceiling averages ~3 dB of
 * GR, as much as a howl with loop gain 1.5), so the threshold sits above that: it catches loop gains
 * ≳ 1.8 and gross overloads only.
 */
export const LIMITER_BACKSTOP_DB = 4.5;
export const LIMITER_BACKSTOP_MS = 2000;
export const LIMITER_BACKSTOP_TAU_MS = 300;

const M_IDLE = 0;
const M_GATING = 1;
const M_PLAYING = 2;
const M_FADING = 3;

const EMPTY = new Float32Array(0);

const HOWL_TEXT: Record<HowlReason, string> = {
  'peak-growth': 'il livello cresce più in fretta del guadagno impostato',
  'limiter-saturated': 'il limiter è saturo con un suono che si ripete (fischio o eco)',
  'output-overload': "il livello d'uscita è vicino al ceiling",
  'anc-worse': "l'ANC sta aumentando il rumore",
};

/** Structure of the conditioning chain: changing it needs a chain swap under the reconfigure envelope. */
const chainKey = (p: EngineParams): string => {
  const f = p.filters;
  const tonal = p.phaseMode === 'tonal' ? p.tonal.harmonics : 0;
  return `${f.hpEnabled ? f.hpOrder : 0}|${f.barrierEnabled ? 1 : 0}|${f.lpEnabled ? f.lpOrder : 0}|${tonal}`;
};

/**
 * Largest coefficient changes applied to the running chain: ConditioningChain glides its direct-form I
 * biquads and resonator gains to the new values over COEFF_RAMP_MS (20 ms, smoothstep). On 50/100/400 Hz
 * tones, cutoffs ×2/÷2, tonal Q ×3/÷3 and f0 ±3 % leave nothing above 2 kHz (≤ −100 dB re the tone) and
 * deviate from an ideal 20 ms crossfade less than the reconfigure envelope (30 ms dip) does, except one
 * stop-band tone. Bigger jumps still go through the envelope.
 */
export const LIVE_FREQ_RATIO = 2;
export const LIVE_Q_RATIO = 3;
export const LIVE_F0_RATIO = 1.03;

const within = (a: number, b: number, ratio: number): boolean => Math.max(a, b) <= ratio * Math.min(a, b);

/**
 * After a live change of the conditioning response the howl detectors' level tracks restart
 * (HowlDetector.holdLevels) for the coefficient glide plus as long again for the filters to settle: the
 * level then changes at constant commanded gain (a hum while the HP cutoff is dragged down rises by tens
 * of dB in a second), which is no howl. Larger changes go through the reconfigure envelope, which the
 * commanded gain the detectors compensate for includes.
 */
const LIVE_HOLD_MS = 2 * COEFF_RAMP_MS;

/** True when a live change (same chainKey) moves a filter cutoff or, in tonal mode, f0 or Q. */
const responseChanges = (a: EngineParams, b: EngineParams): boolean => {
  const fa = a.filters;
  const fb = b.filters;
  if (fa.hpEnabled && fa.hpFreq !== fb.hpFreq) return true;
  if (fa.barrierEnabled && fa.barrierFreq !== fb.barrierFreq) return true;
  if (fa.lpEnabled && fa.lpFreq !== fb.lpFreq) return true;
  return a.phaseMode === 'tonal' && (a.tonal.f0 !== b.tonal.f0 || a.tonal.q !== b.tonal.q);
};

/** True when `b` differs from `a` (same chainKey) only by coefficient changes small enough to apply live. */
const liveChange = (a: EngineParams, b: EngineParams): boolean => {
  const fa = a.filters;
  const fb = b.filters;
  if (fa.hpEnabled && !within(fa.hpFreq, fb.hpFreq, LIVE_FREQ_RATIO)) return false;
  if (fa.barrierEnabled && !within(fa.barrierFreq, fb.barrierFreq, LIVE_FREQ_RATIO)) return false;
  if (fa.lpEnabled && !within(fa.lpFreq, fb.lpFreq, LIVE_FREQ_RATIO)) return false;
  if (a.phaseMode !== 'tonal') return true;
  return within(a.tonal.f0, b.tonal.f0, LIVE_F0_RATIO) && within(a.tonal.q, b.tonal.q, LIVE_Q_RATIO);
};

/**
 * True when the chain `b` passes a wider band than `a`: a filter section removed or the tonal mode left
 * (resonators → broadband). The level can then jump by tens of dB, so the channels soft-start.
 */
const broadens = (a: EngineParams, b: EngineParams): boolean => {
  const fa = a.filters;
  const fb = b.filters;
  return (
    (fa.hpEnabled && !fb.hpEnabled) ||
    (fa.barrierEnabled && !fb.barrierEnabled) ||
    (fa.lpEnabled && !fb.lpEnabled) ||
    (a.phaseMode === 'tonal' && b.phaseMode !== 'tonal')
  );
};

/** Commanded anti-noise gain of a channel (channel gain + master); master ≤ MASTER_MIN_DB is silence. */
const totalGainDb = (p: EngineParams, ch: number): number =>
  p.masterDb <= MASTER_MIN_DB ? MASTER_MIN_DB : p.channels[ch].gainDb + p.masterDb;

const reading = (m: LevelMeter): LevelReading => ({ rmsDb: m.rmsDb, peakDb: m.peakDb });

interface Queued {
  msg: WorkletOutMsg;
  transfer?: ArrayBuffer[];
}

interface PendingFir {
  bulk: number;
  taps: Float64Array | null;
}

export class AntiPhaseEngine {
  private readonly fs: number;
  private readonly softStartMs: number;
  private readonly bypassCond: boolean;
  /** Latest requested (clamped) parameters. */
  private target: EngineParams;

  // Reference path.
  private readonly chains: [ConditioningChain, ConditioningChain];
  private live = 0;
  private liveKey: string;
  /** Parameters the live / standby chains are configured with. */
  private liveChain: EngineParams;
  private pendingChain: EngineParams;
  /** The standby chain is configured for the pending structure and warms up in parallel. */
  private swapPending = false;
  private pendingKey = '';
  private liveRef: 0 | 1;
  private errCh: 0 | 1 | -1;
  private readonly fb: FeedbackCanceller;
  private readonly firPending: [PendingFir | null, PendingFir | null] = [null, null];
  private readonly line: DelayLine;

  // Channels.
  private readonly taps: [SmoothDelayTap, SmoothDelayTap];
  private readonly rot: [PhaseRotator, PhaseRotator];
  /** Per-channel total gain, channel gain + master: one rise budget for both (6 dB + 6 dB/s in all). */
  private readonly gains: [GainRamp, GainRamp];
  /** Per-channel reconfigure envelope (structural changes, channel enable). */
  private readonly rec: [LinearEnvelope, LinearEnvelope];
  /** Per-channel soft-start: a channel enabled, or whose path gets broader, rises from 0 like an unmute. */
  private readonly chSoft: [LinearEnvelope, LinearEnvelope];
  private readonly lim: [PeakLimiter, PeakLimiter];
  private readonly howl: [HowlDetector, HowlDetector];
  private readonly pol = new Float64Array([1, 1]);
  private readonly useRot = new Uint8Array(2);
  private readonly enabled = new Uint8Array(2);
  private readonly liveMode: [PhaseMode, PhaseMode] = ['delay', 'delay'];
  private readonly delayTarget = new Float64Array(2);
  private readonly delayClamped: [boolean, boolean] = [false, false];
  private readonly pendingCh = new Uint8Array(2);
  /** Index of the sample being processed (between blocks: of the next one). */
  private clock = 0;
  /**
   * First sample of clean delay-line content: everything before it precedes a discontinuity (chain swap,
   * reference or FIR change) or holds the pickup of a calibration stimulus. A channel never plays a read
   * that reaches before it.
   */
  private cleanFrom = -Infinity;
  /** The bypass or the measurement gate holds the anti-noise at exactly 0 (both reopen in 15 ms). */
  private closed = false;
  private readonly envCache = new Float64Array([Number.NaN, Number.NaN]);
  private readonly envDb = new Float64Array(2);
  /** Limiter GR smoothed over LIMITER_BACKSTOP_TAU_MS, and consecutive samples with it above LIMITER_BACKSTOP_DB. */
  private readonly grSmooth = new Float64Array(2);
  private readonly grCoef: number;
  private readonly satCount = new Int32Array(2);
  private readonly satLimit: number;
  private satTrip = false;
  private pendingGlobal = false;
  private recBusy = false;
  private readonly jumpSamples: number;

  // Master (applied to the stimulus; the anti-noise gets it through `gains`), global envelopes and state.
  private readonly master: GainRamp;
  private readonly soft: LinearEnvelope;
  private readonly muteEnv: LinearEnvelope;
  private readonly bypassEnv: LinearEnvelope;
  /** Mutes the anti-noise during a measurement. */
  private readonly gate: LinearEnvelope;
  /** Fades the stimulus out when a measurement is aborted. */
  private readonly stimGate: LinearEnvelope;
  private armed = false;
  private muted = false;
  private bypass = false;
  private softRestart = false;
  private howlLatched = false;
  private bad = false;

  // Ceiling (rises are rate-limited like gains, falls are immediate).
  private ceilDb: number;
  private ceilTgtDb: number;
  private ceilRamping = false;
  private readonly ceilStep: number;
  private ceil32 = 0;
  private readonly f32 = new Float32Array(1);
  private readonly u32 = new Uint32Array(this.f32.buffer);

  // Measurement.
  private measState = M_IDLE;
  private measId = 0;
  private measCh: 0 | 1 = 0;
  private measStim: Float32Array = EMPTY;
  private measRef: Float32Array = EMPTY;
  private measErr: Float32Array | null = null;
  private measIdx = 0;
  private measTotal = 0;
  /** Master of the stimulus, fixed when it starts: changing the master while it plays aborts the measurement. */
  private measMasterDb = 0;
  private measMasterLin = 0;
  private readonly progressPeriod: number;
  private progressCount = 0;
  /** After a measurement: the gate reopens once both taps read only clean content. */
  private gateRecover = false;

  // Meters, taps, channel-2 check.
  private readonly mRef: LevelMeter;
  private readonly mErr: LevelMeter;
  private readonly mOutL: LevelMeter;
  private readonly mOutR: LevelMeter;
  private readonly meterPeriod: number;
  private meterCount = 0;
  private blockSize = 0;
  private errPresent = false;
  private readonly tapper: TapCollector;
  private readonly tapVals = new Float64Array(TAP_CHANNELS.length);
  private tapOn = false;
  private readonly ch2Window: number;
  private c2s00 = 0;
  private c2s11 = 0;
  private c2s01 = 0;
  private c2n = 0;
  private c2Absent = false;
  private in1Null = true;
  private ch2Status: Ch2Status = 'pending';

  private queue: Queued[] = [];
  private spare: Queued[] = [];

  constructor(fs: number, opts?: EngineOptions) {
    if (!(fs > 0) || !Number.isFinite(fs)) throw new RangeError('AntiPhaseEngine: fs must be positive');
    this.fs = fs;
    const ss = opts?.softStartMs;
    this.softStartMs = ss !== undefined && Number.isFinite(ss) && ss >= 0 ? ss : SOFT_START_MS;
    const mh = opts?.meterHz;
    this.meterPeriod = Math.max(1, Math.round(fs / (mh !== undefined && mh > 0 ? mh : 30)));
    this.bypassCond = opts?.testBypassConditioning === true;
    const p = clampParams(undefined);
    this.target = p;

    this.chains = [new ConditioningChain(fs, p), new ConditioningChain(fs, p)];
    this.liveKey = chainKey(p);
    this.liveChain = p;
    this.pendingChain = p;
    this.liveRef = p.refChannel;
    this.errCh = p.errorChannel;
    this.fb = new FeedbackCanceller(fs, Math.round((FEEDBACK_MAX_BULK_MS / 1000) * fs), FEEDBACK_MAX_TAPS);
    this.line = new DelayLine(delayCapacity(fs));

    this.taps = [new SmoothDelayTap(fs), new SmoothDelayTap(fs)];
    this.rot = [new PhaseRotator(fs), new PhaseRotator(fs)];
    this.gains = [new GainRamp(fs, totalGainDb(p, 0)), new GainRamp(fs, totalGainDb(p, 1))];
    this.gains[0].setRiseFloorDb(p.channels[0].gainDb + RISE_FLOOR_DB);
    this.gains[1].setRiseFloorDb(p.channels[1].gainDb + RISE_FLOOR_DB);
    this.rec = [new LinearEnvelope(fs, 0), new LinearEnvelope(fs, 0)];
    this.chSoft = [new LinearEnvelope(fs, 1), new LinearEnvelope(fs, 1)];
    this.lim = [new PeakLimiter(fs, p.ceilingDb), new PeakLimiter(fs, p.ceilingDb)];
    this.howl = [new HowlDetector(fs), new HowlDetector(fs)];
    this.jumpSamples = (DELAY_JUMP_THRESHOLD_MS / 1000) * fs;
    this.satLimit = Math.round((LIMITER_BACKSTOP_MS / 1000) * fs);
    this.grCoef = 1 - Math.exp(-1000 / (LIMITER_BACKSTOP_TAU_MS * fs));

    this.master = new GainRamp(fs, p.masterDb);
    this.soft = new LinearEnvelope(fs, 0);
    this.muteEnv = new LinearEnvelope(fs, 1);
    this.bypassEnv = new LinearEnvelope(fs, 1);
    this.gate = new LinearEnvelope(fs, 1);
    this.stimGate = new LinearEnvelope(fs, 1);

    this.ceilDb = p.ceilingDb;
    this.ceilTgtDb = p.ceilingDb;
    this.ceilStep = MAX_RISE_DB_PER_S / fs;
    this.applyCeiling();

    this.progressPeriod = Math.max(1, Math.round(fs / PROGRESS_HZ));
    this.mRef = new LevelMeter(fs);
    this.mErr = new LevelMeter(fs);
    this.mOutL = new LevelMeter(fs);
    this.mOutR = new LevelMeter(fs);
    this.tapper = new TapCollector(TAP_CHANNELS.length, TAP_BLOCK_FRAMES, TAP_POOL_SIZE);
    this.ch2Window = Math.max(1, Math.round((CH2_WINDOW_MS / 1000) * fs));

    this.computeDelays(p);
    this.applyAllNow();
  }

  /** A valid copy of the latest requested parameters. */
  get params(): EngineParams {
    return clampParams(this.target);
  }

  // ---------------------------------------------------------------- control

  handleMessage(msg: WorkletInMsg): void {
    if (msg === null || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'params':
        this.setParams(msg.params);
        break;
      case 'arm':
        this.setArmed(msg.armed === true);
        break;
      case 'mute':
        if (msg.muted === true) {
          this.muted = true;
          this.softRestart = false;
          if (this.muteEnv.target !== 0) this.muteEnv.rampTo(0, MUTE_RAMP_MS);
        } else if (msg.muted === false && this.muted) {
          this.muted = false;
          this.startSoft();
        }
        break;
      case 'bypass':
        this.bypass = msg.bypass === true;
        this.bypassEnv.rampTo(this.bypass ? 0 : 1, MUTE_RAMP_MS);
        break;
      case 'panic':
        this.panic(false);
        break;
      case 'softStart':
        if (!this.armed || this.muteEnv.value * this.soft.value === 0) {
          this.softRestart = false;
          this.soft.set(0);
          this.soft.rampTo(1, this.softStartMs);
        } else {
          // Fade out first: restarting from 0 while audible would click.
          this.softRestart = true;
          this.soft.rampTo(0, MUTE_RAMP_MS);
        }
        break;
      case 'measure':
        this.startMeasure(msg.id, msg.stimulus, msg.channel, msg.postRollFrames);
        break;
      case 'abortMeasure':
        this.abortMeasure(PANIC_RAMP_MS);
        break;
      case 'feedbackFir':
        this.setFir(msg.channel === 1 ? 1 : 0, msg.bulkDelay, msg.taps);
        break;
      case 'tap':
        this.tapOn = msg.enabled === true;
        if (!this.tapOn) this.tapper.reset();
        break;
      case 'recycle':
        if (msg.buffer instanceof Float32Array) this.tapper.recycle(msg.buffer);
        break;
      case 'resetHowl':
        this.resetHowl();
        break;
    }
  }

  private setParams(raw: EngineParams): void {
    const p = clampParams(raw);
    this.target = p;
    if (p.errorChannel !== this.errCh) this.mErr.reset();
    this.errCh = p.errorChannel;
    this.computeDelays(p);
    if (!this.armed) {
      this.applyAllNow();
      return;
    }
    this.setGain(0, p, false);
    this.setGain(1, p, false);
    // measureResult reports one masterLin: a stimulus never plays across a master change.
    if (this.measState === M_PLAYING && p.masterDb !== this.measMasterDb) this.abortMeasure(PANIC_RAMP_MS);
    this.master.setTargetDb(p.masterDb);
    this.setCeiling(p.ceilingDb, false);

    let global = false;
    const key = chainKey(p);
    const standby = this.chains[1 - this.live];
    if (this.swapPending && key === this.pendingKey && liveChange(this.pendingChain, p)) {
      standby.configure(p);
      this.pendingChain = p;
    } else if (key === this.liveKey && liveChange(this.liveChain, p)) {
      // Also cancels a pending swap: the running chain is close enough to the request.
      this.swapPending = false;
      if (responseChanges(this.liveChain, p)) for (const h of this.howl) h.holdLevels(LIVE_HOLD_MS);
      this.chains[this.live].configure(p);
      this.liveChain = p;
    } else {
      standby.configure(p);
      standby.reset();
      this.swapPending = true;
      this.pendingKey = key;
      this.pendingChain = p;
      global = true;
    }
    if (p.refChannel !== this.liveRef) global = true;
    if (global) this.pendingGlobal = true;

    for (let ch = 0; ch < 2; ch++) {
      const c = p.channels[ch];
      const structural =
        this.pendingGlobal ||
        (c.polarityInvert ? -1 : 1) !== this.pol[ch] ||
        (c.enabled ? 1 : 0) !== this.enabled[ch] ||
        p.phaseMode !== this.liveMode[ch];
      if (structural) this.markPending(ch);
      else if (this.pendingCh[ch] === 0) this.applyDelay(ch);
    }
  }

  /**
   * Channel gain + master on one ramp. Rises are free only while the master part is below RISE_FLOOR_DB
   * (un-muting a master at −∞), as they were for the master alone.
   */
  private setGain(ch: number, p: EngineParams, jump: boolean): void {
    const g = this.gains[ch];
    g.setRiseFloorDb(p.channels[ch].gainDb + RISE_FLOOR_DB);
    if (jump) g.jumpTo(totalGainDb(p, ch));
    else g.setTargetDb(totalGainDb(p, ch));
  }

  private computeDelays(p: EngineParams): void {
    for (let ch = 0; ch < 2; ch++) {
      const d = computeChannelDelay(p, this.fs, ch as 0 | 1);
      this.delayTarget[ch] = d.samples;
      this.delayClamped[ch] = d.clamped;
    }
  }

  private setArmed(armed: boolean): void {
    if (armed === this.armed) return;
    if (armed) {
      this.armed = true;
      this.soft.set(0);
      this.soft.rampTo(1, this.softStartMs);
      this.softRestart = false;
      this.lim[0].reset();
      this.lim[1].reset();
      this.resetHowl();
    } else {
      this.armed = false;
      this.abortMeasure(0);
      this.soft.set(0);
      this.softRestart = false;
      this.applyAllNow();
    }
  }

  /**
   * Unmute: always a soft-start from 0. If the mute ramp has not reached 0 yet, the remaining level
   * fades out first at the mute slope (jumping to 0 would click).
   */
  private startSoft(): void {
    const cur = this.muteEnv.value * this.soft.value;
    this.muteEnv.set(1);
    this.soft.set(cur);
    if (cur > 0) {
      this.softRestart = true;
      this.soft.rampTo(0, MUTE_RAMP_MS * cur);
    } else {
      this.softRestart = false;
      this.soft.rampTo(1, this.softStartMs);
    }
    this.resetHowl();
  }

  private panic(immediate: boolean): void {
    this.muted = true;
    this.softRestart = false;
    if (immediate) this.muteEnv.set(0);
    else if (this.muteEnv.value !== 0) this.muteEnv.rampTo(0, PANIC_RAMP_MS);
    this.abortMeasure(immediate ? 0 : PANIC_RAMP_MS);
  }

  private resetHowl(): void {
    this.howl[0].reset();
    this.howl[1].reset();
    this.satCount[0] = 0;
    this.satCount[1] = 0;
    this.grSmooth[0] = 0;
    this.grSmooth[1] = 0;
    this.satTrip = false;
    this.howlLatched = false;
  }

  private setFir(ch: 0 | 1, bulkDelay: number, taps: Float64Array | null): void {
    const copy = taps instanceof Float64Array ? Float64Array.from(taps) : null;
    const bulk = Number.isFinite(bulkDelay) ? bulkDelay : 1;
    if (!this.armed) {
      this.fb.setFir(ch, bulk, copy);
      this.firPending[ch] = null;
      return;
    }
    // Changing x̂ abruptly is a discontinuity in the delay line, like a chain swap.
    this.firPending[ch] = { bulk, taps: copy };
    this.pendingGlobal = true;
    this.markPending(0);
    this.markPending(1);
  }

  private setCeiling(db: number, immediate: boolean): void {
    this.ceilTgtDb = db;
    if (immediate || db <= this.ceilDb) {
      this.ceilDb = db;
      this.ceilRamping = false;
      this.applyCeiling();
    } else {
      this.ceilRamping = true;
    }
  }

  private applyCeiling(): void {
    this.lim[0].setCeilingDb(this.ceilDb);
    this.lim[1].setCeilingDb(this.ceilDb);
    // Largest float32 not above the ceiling: the output arrays are float32 and rounding up would exceed it.
    const c = this.lim[0].ceiling;
    this.f32[0] = c;
    if (this.f32[0] > c) this.u32[0] -= 1;
    this.ceil32 = this.f32[0];
  }

  /** Output is silent (not armed): every pending change is applied at once, gains jump. */
  private applyAllNow(): void {
    const p = this.target;
    this.swapPending = false;
    this.chains[this.live].configure(p);
    this.liveKey = chainKey(p);
    this.liveChain = p;
    this.liveRef = p.refChannel;
    for (let ch = 0; ch < 2; ch++) {
      const f = this.firPending[ch];
      if (f !== null) this.fb.setFir(ch as 0 | 1, f.bulk, f.taps);
      this.firPending[ch] = null;
      this.applyChannel(ch);
      this.rec[ch].set(this.enabled[ch]);
      // Arming soft-starts every channel anyway.
      this.chSoft[ch].set(1);
      this.setGain(ch, p, true);
    }
    this.master.jumpTo(p.masterDb);
    this.setCeiling(p.ceilingDb, true);
    this.pendingGlobal = false;
    this.recBusy = false;
  }

  private markPending(ch: number): void {
    this.pendingCh[ch] = 1;
    if (this.rec[ch].target !== 0) this.rec[ch].rampTo(0, MUTE_RAMP_MS);
    this.recBusy = true;
  }

  /** Non-structural delay/Δ update: slew or crossfade, smoothed rotation. */
  private applyDelay(ch: number): void {
    const tap = this.taps[ch];
    const d = this.delayTarget[ch];
    if (!this.readsClean(d)) {
      // The longer delay would replay what precedes the last discontinuity: ramp down, snap, wait.
      this.markPending(ch);
      return;
    }
    const diff = Math.abs(d - tap.target);
    // Tonal mode: a change of more than half a period is a D mod T wrap (equivalent to a small change for
    // periodic noise). Slewed, it would sweep the anti-noise phase the long way round, through in-phase.
    const wrap = this.liveMode[ch] === 'tonal' && diff > this.fs / (2 * this.target.tonal.f0);
    tap.setTarget(d, diff >= this.jumpSamples || wrap);
    this.rot[ch].setDeltaDeg(this.target.channels[ch].phaseDeltaDeg);
  }

  /** True when a read at delay `d` (at the current sample) touches only clean delay-line content. */
  private readsClean(d: number): boolean {
    return this.clock - Math.ceil(d) - LAGRANGE_REACH >= this.cleanFrom;
  }

  /** Soft-start of one channel from 0 (called while the channel is silent). */
  private restartChannel(ch: number): void {
    this.chSoft[ch].set(0);
    this.chSoft[ch].rampTo(1, this.softStartMs);
  }

  /** Structural per-channel update, applied while the channel is silent. */
  private applyChannel(ch: number): void {
    const p = this.target;
    const c = p.channels[ch];
    if (c.enabled && this.enabled[ch] === 0) this.restartChannel(ch);
    this.pol[ch] = c.polarityInvert ? -1 : 1;
    this.enabled[ch] = c.enabled ? 1 : 0;
    this.liveMode[ch] = p.phaseMode;
    this.useRot[ch] = p.phaseMode === 'hilbert' ? 1 : 0;
    const tap = this.taps[ch];
    tap.setTarget(this.delayTarget[ch]);
    tap.snap();
    this.rot[ch].setDeltaDeg(c.phaseDeltaDeg, true);
    this.pendingCh[ch] = 0;
  }

  /** Global update, applied at the start of a sample while both channels are silent. */
  private applyGlobal(): void {
    let changed = false;
    let broader = false;
    if (this.swapPending) {
      broader = broadens(this.liveChain, this.pendingChain);
      this.live = 1 - this.live;
      this.liveKey = this.pendingKey;
      this.liveChain = this.pendingChain;
      this.swapPending = false;
      changed = true;
    }
    if (this.target.refChannel !== this.liveRef) {
      this.liveRef = this.target.refChannel;
      changed = true;
      broader = true;
    }
    for (let ch = 0; ch < 2; ch++) {
      const f = this.firPending[ch];
      if (f === null) continue;
      this.fb.setFir(ch as 0 | 1, f.bulk, f.taps);
      this.firPending[ch] = null;
      changed = true;
    }
    // This sample is the first one written after the discontinuity.
    if (changed) this.cleanFrom = this.clock;
    if (broader) {
      this.restartChannel(0);
      this.restartChannel(1);
    }
    this.pendingGlobal = false;
  }

  /** Reconfigure state machine, run at the start of a sample while something is pending or ramping. */
  private stepReconfigure(): void {
    if (this.pendingGlobal && this.rec[0].value === 0 && this.rec[1].value === 0) this.applyGlobal();
    let busy = this.pendingGlobal;
    for (let ch = 0; ch < 2; ch++) {
      const env = this.rec[ch];
      if (this.pendingCh[ch] === 1) {
        if (this.pendingGlobal || env.value !== 0) {
          busy = true;
          continue;
        }
        this.applyChannel(ch);
      }
      const want = this.enabled[ch];
      if (env.target !== want) {
        // Back up only once the tap reads nothing from before the last discontinuity.
        if (want === 1 && !this.readsClean(this.taps[ch].reach)) {
          busy = true;
          continue;
        }
        env.rampTo(want, MUTE_RAMP_MS);
      }
      if (!env.done) busy = true;
    }
    this.recBusy = busy;
  }

  private stepCeiling(): void {
    let c = this.ceilDb + this.ceilStep;
    if (c >= this.ceilTgtDb) {
      c = this.ceilTgtDb;
      this.ceilRamping = false;
    }
    this.ceilDb = c;
    this.applyCeiling();
  }

  // ------------------------------------------------------------ measurement

  private startMeasure(id: number, stimulus: Float32Array, channel: 0 | 1, postRollFrames: number): void {
    const post = Number.isFinite(postRollFrames) ? Math.max(0, Math.floor(postRollFrames)) : -1;
    const total = stimulus instanceof Float32Array && post >= 0 ? stimulus.length + post : -1;
    const ok =
      this.armed &&
      this.measState === M_IDLE &&
      (channel === 0 || channel === 1) &&
      total > 0 &&
      total <= MAX_MEASURE_S * this.fs;
    if (!ok) {
      this.postAborted(id);
      return;
    }
    this.measId = id;
    this.measCh = channel;
    this.measStim = stimulus;
    this.measTotal = total;
    this.measRef = new Float32Array(total);
    this.measErr = this.errCh >= 0 && this.errPresent ? new Float32Array(total) : null;
    this.measIdx = 0;
    this.progressCount = 0;
    this.stimGate.set(1);
    this.gateRecover = false;
    if (this.gate.target !== 0) this.gate.rampTo(0, MUTE_RAMP_MS);
    this.measState = M_GATING;
    this.resetHowl();
  }

  private postAborted(id: number): void {
    this.post({ type: 'measureResult', id, ref: new Float32Array(0), err: null, masterLin: this.master.currentLin, aborted: true });
  }

  /** Stops the measurement (result aborted=true); the stimulus fades out over `fadeMs` (0 = at once). */
  private abortMeasure(fadeMs: number): void {
    const st = this.measState;
    if (st === M_IDLE || st === M_FADING) {
      if (st === M_FADING && fadeMs === 0) this.endMeasure(false);
      return;
    }
    this.postAborted(this.measId);
    this.measRef = EMPTY;
    this.measErr = null;
    if (st === M_PLAYING && fadeMs > 0) {
      this.measState = M_FADING;
      this.stimGate.rampTo(0, fadeMs);
    } else {
      this.endMeasure(st === M_PLAYING);
    }
  }

  private finishMeasure(): void {
    const ref = this.measRef;
    const err = this.measErr;
    const transfer = [ref.buffer as ArrayBuffer];
    if (err !== null) transfer.push(err.buffer as ArrayBuffer);
    this.post(
      { type: 'measureResult', id: this.measId, ref, err, masterLin: this.measMasterLin, aborted: false },
      transfer,
    );
    this.measRef = EMPTY;
    this.measErr = null;
    this.endMeasure(true);
  }

  /** Back to idle; after a stimulus was played the anti-noise waits until the delay line has no trace of it. */
  private endMeasure(played: boolean): void {
    this.measState = M_IDLE;
    this.measStim = EMPTY;
    this.stimGate.set(1);
    this.resetHowl();
    if (!this.armed) {
      this.gate.set(1);
      this.gateRecover = false;
      return;
    }
    // Everything written up to this sample may hold the stimulus: later reads too (delay increase) stay away.
    if (played && this.clock + 1 > this.cleanFrom) this.cleanFrom = this.clock + 1;
    this.gateRecover = true;
  }

  /** The anti-noise is at 0 and the stimulus has not started: nothing else is played through the master. */
  private startStimulus(): void {
    this.measState = M_PLAYING;
    this.measIdx = 0;
    // A master still ramping would change the level along the stimulus while one masterLin is reported.
    this.measMasterDb = this.target.masterDb;
    this.master.jumpTo(this.measMasterDb);
    // Gain reduction left over from the anti-noise would attenuate the first ~300 ms of the stimulus.
    this.lim[0].reset();
    this.lim[1].reset();
  }

  private stepMeasureControl(): void {
    switch (this.measState) {
      case M_GATING:
        if (this.gate.value === 0) this.startStimulus();
        break;
      case M_FADING:
        if (this.stimGate.value === 0) this.endMeasure(true);
        break;
      case M_IDLE:
        if (this.readsClean(this.taps[0].reach) && this.readsClean(this.taps[1].reach)) {
          this.gateRecover = false;
          this.gate.rampTo(1, MUTE_RAMP_MS);
        }
        break;
    }
  }

  // ---------------------------------------------------------------- audio

  /** in0/in1 = input channels (null when absent); outL/outR same length n (block length is n, never assume 128). */
  process(in0: Float32Array | null, in1: Float32Array | null, outL: Float32Array, outR: Float32Array): void {
    const n = outL.length;
    this.blockSize = n;
    this.in1Null = in1 === null;
    if (this.in1Null) this.c2Absent = true;
    const errCh = this.errCh;
    this.errPresent = errCh === 0 ? in0 !== null : errCh === 1 ? in1 !== null : false;
    const base = this.clock;
    try {
      for (let i = 0; i < n; i++) {
        this.clock = base + i;
        if (this.recBusy) this.stepReconfigure();
        if (this.measState !== M_IDLE || this.gateRecover) this.stepMeasureControl();
        if (this.ceilRamping) this.stepCeiling();
        if (this.softRestart && this.soft.value === 0) {
          this.softRestart = false;
          this.soft.rampTo(1, this.softStartMs);
        }

        // Number.isFinite also rejects undefined, so shorter input arrays read as silence.
        let a0 = in0 !== null ? in0[i] : 0;
        if (!Number.isFinite(a0)) a0 = 0;
        let a1 = in1 !== null ? in1[i] : 0;
        if (!Number.isFinite(a1)) a1 = 0;
        const x = this.liveRef === 0 ? a0 : a1;
        const e = errCh === 0 ? a0 : errCh === 1 ? a1 : 0;

        // While the bypass or the measurement gate is shut, rises (soft-starts, gain increases) wait.
        const closed = this.bypassEnv.value === 0 || this.gate.value === 0;
        this.closed = closed;
        const soft = closed && this.soft.target > this.soft.value ? this.soft.value : this.soft.next();
        const common = soft * this.muteEnv.next() * this.bypassEnv.next() * this.gate.next();
        const master = this.master.next();

        const xh = x - this.fb.predict();
        let cond = xh;
        if (!this.bypassCond) {
          cond = this.chains[this.live].process(xh);
          if (this.swapPending) this.chains[1 - this.live].process(xh);
        }
        if (!Number.isFinite(cond)) {
          cond = 0;
          this.bad = true;
        }
        this.line.write(cond);

        const st = this.measState;
        let stim = 0;
        if (st === M_PLAYING || st === M_FADING) {
          const k = this.measIdx;
          const s = k < this.measStim.length ? this.measStim[k] : 0;
          stim = (Number.isFinite(s) ? s : 0) * this.stimGate.next();
          if (st === M_PLAYING && k === 0) this.measMasterLin = master;
        }
        const howlOn = this.armed && st === M_IDLE && !this.howlLatched;
        const yl = this.channelSample(0, common, master, this.measCh === 0 ? stim : 0, howlOn);
        const yr = this.channelSample(1, common, master, this.measCh === 1 ? stim : 0, howlOn);
        outL[i] = yl;
        outR[i] = yr;
        const ol = Math.fround(yl);
        const or = Math.fround(yr);
        this.fb.pushOutputs(ol, or);
        if (howlOn && (this.howl[0].state.triggered || this.howl[1].state.triggered || this.satTrip)) this.onHowl();
        if (this.bad) this.resetStates();

        this.mRef.push(x);
        if (this.errPresent) this.mErr.push(e);
        this.mOutL.push(ol);
        this.mOutR.push(or);

        if (st === M_PLAYING) {
          const k = this.measIdx;
          this.measRef[k] = x;
          if (this.measErr !== null) this.measErr[k] = e;
          this.measIdx = k + 1;
          if (k + 1 >= this.measTotal) this.finishMeasure();
          else if (++this.progressCount >= this.progressPeriod) {
            this.progressCount = 0;
            this.post({ type: 'measureProgress', id: this.measId, fraction: (k + 1) / this.measTotal });
          }
        } else if (st === M_FADING) {
          this.measIdx++;
        }

        if (this.tapOn) {
          const v = this.tapVals;
          v[0] = x;
          v[1] = cond;
          v[2] = ol;
          v[3] = or;
          v[4] = e;
          const block = this.tapper.push(v);
          if (block !== null) this.post({ type: 'tap', block, frames: TAP_BLOCK_FRAMES }, [block.buffer as ArrayBuffer]);
        }

        this.c2s00 += a0 * a0;
        this.c2s11 += a1 * a1;
        this.c2s01 += a0 * a1;
        if (++this.c2n >= this.ch2Window) this.evaluateCh2();

        if (++this.meterCount >= this.meterPeriod) {
          this.meterCount = 0;
          this.postMeters();
        }
      }
    } catch (err) {
      this.fault(err, outL, outR);
    }
    this.clock = base + n;
  }

  private channelSample(ch: 0 | 1, common: number, master: number, stim: number, howlOn: boolean): number {
    let v = this.taps[ch].next(this.line);
    // The rotators always run so that switching to Hilbert mode finds them settled.
    const r = this.rot[ch].process(v);
    if (this.useRot[ch] === 1) v = r;
    const gain = this.gains[ch].next(this.closed);
    const rec = this.rec[ch].next();
    const cs = this.chSoft[ch];
    // Like the global soft-start, it progresses only while the channel can be heard.
    const soft = (this.closed || rec === 0) && cs.target > cs.value ? cs.value : cs.next();
    const env = common * rec * soft;
    v = v * this.pol[ch] * gain * env + stim * master;
    if (!Number.isFinite(v)) {
      v = 0;
      this.bad = true;
    }
    const lim = this.lim[ch];
    if (howlOn) {
      if (env !== this.envCache[ch]) {
        this.envCache[ch] = env;
        this.envDb[ch] = env > 0 ? 20 * Math.log10(env) : MASTER_MIN_DB;
      }
      let cmd = this.gains[ch].currentDb + this.envDb[ch];
      if (!(cmd > MASTER_MIN_DB)) cmd = MASTER_MIN_DB;
      const gr = lim.gainReductionDb;
      this.howl[ch].push(v, cmd, gr, this.ceilDb);
      const sm = this.grSmooth[ch] + this.grCoef * (gr - this.grSmooth[ch]);
      this.grSmooth[ch] = sm;
      if (sm <= LIMITER_BACKSTOP_DB) this.satCount[ch] = 0;
      else if (++this.satCount[ch] > this.satLimit) this.satTrip = true;
    }
    let y = lim.process(v);
    const c = this.ceil32;
    if (y > c) y = c;
    else if (y < -c) y = -c;
    return this.armed ? y : 0;
  }

  private onHowl(): void {
    const h0 = this.howl[0].state;
    const h1 = this.howl[1].state;
    const s = h0.triggered ? h0 : h1.triggered ? h1 : null;
    const reason: HowlReason = s !== null ? (s.reason ?? 'output-overload') : 'limiter-saturated';
    const freqHz = s !== null ? s.freqHz : null;
    this.howlLatched = true;
    this.panic(false);
    this.post({ type: 'howl', reason, freqHz });
    const backstopDb = String(LIMITER_BACKSTOP_DB).replace('.', ',');
    const text =
      s !== null
        ? `Innesco rilevato${freqHz !== null ? ` a ${Math.round(freqHz)} Hz` : ''}: ${HOWL_TEXT[reason]}`
        : `Innesco probabile: il limiter ha tolto più di ${backstopDb} dB per più di ${LIMITER_BACKSTOP_MS / 1000} s`;
    this.post({ type: 'autoMute', reason: `${text}. Uscita silenziata.` });
  }

  /** NaN/Infinity somewhere in the chain: clear every recursive state (the offending sample was output as 0). */
  private resetStates(): void {
    this.bad = false;
    this.chains[0].reset();
    this.chains[1].reset();
    this.line.reset();
    this.rot[0].reset();
    this.rot[1].reset();
    this.lim[0].reset();
    this.lim[1].reset();
    this.fb.reset();
  }

  private fault(err: unknown, outL: Float32Array, outR: Float32Array): void {
    outL.fill(0);
    outR.fill(0);
    try {
      this.panic(true);
      this.resetStates();
      this.resetHowl();
    } catch {
      // Already failing: the outputs are zero and `muted` is set, nothing else to do.
    }
    const detail = err instanceof Error ? err.message : String(err);
    this.post({ type: 'fault', message: `Errore nel processore audio: ${detail}. Uscita silenziata.` });
  }

  /**
   * Checks the error-mic input: input 2 unless errorChannel says input 1. 'absent' = mono track (no second
   * input at all), 'silent' = the error-mic input is silent, 'duplicate' = both inputs carry the same signal.
   */
  private evaluateCh2(): void {
    const n = this.c2n;
    let status: Ch2Status;
    if (this.c2Absent) status = 'absent';
    else {
      const onFirst = this.errCh === 0;
      const pe = (onFirst ? this.c2s00 : this.c2s11) / n;
      const po = (onFirst ? this.c2s11 : this.c2s00) / n;
      const dbE = pe > 0 ? 10 * Math.log10(pe) : -Infinity;
      if (dbE < CH2_SILENT_DBFS) status = 'silent';
      else {
        const corr = this.c2s01 / Math.sqrt(this.c2s00 * this.c2s11);
        const diffDb = po > 0 ? Math.abs(10 * Math.log10(po / pe)) : Infinity;
        status = corr > CH2_DUPLICATE_CORR && diffDb < CH2_DUPLICATE_DB ? 'duplicate' : 'ok';
      }
    }
    this.c2s00 = 0;
    this.c2s11 = 0;
    this.c2s01 = 0;
    this.c2n = 0;
    this.c2Absent = this.in1Null;
    if (status !== this.ch2Status) {
      this.ch2Status = status;
      this.post({ type: 'channelCheck', status });
    }
  }

  private postMeters(): void {
    this.post({
      type: 'meters',
      ref: reading(this.mRef),
      err: this.errPresent ? reading(this.mErr) : null,
      outL: reading(this.mOutL),
      outR: reading(this.mOutR),
      limiterGrDb: [this.lim[0].gainReductionDb, this.lim[1].gainReductionDb],
      armed: this.armed,
      muted: this.muted,
      bypass: this.bypass,
      softStart: this.soft.value,
      delaySamples: [this.delayTarget[0], this.delayTarget[1]],
      delayClamped: [this.delayClamped[0], this.delayClamped[1]],
      blockSize: this.blockSize,
      ch2: this.ch2Status,
      measuring: this.measState !== M_IDLE,
    });
  }

  // -------------------------------------------------------------- messages

  private post(msg: WorkletOutMsg, transfer?: ArrayBuffer[]): void {
    if (this.queue.length >= MAX_QUEUE) {
      if (msg.type === 'tap') {
        this.tapper.recycle(msg.block);
        return;
      }
      if (msg.type === 'meters' || msg.type === 'measureProgress') return;
    }
    this.queue.push(transfer !== undefined ? { msg, transfer } : { msg });
  }

  /** Hands queued outgoing messages to `post` (with transferable buffers), then clears the queue. */
  drain(post: (msg: WorkletOutMsg, transfer?: ArrayBuffer[]) => void): void {
    const q = this.queue;
    if (q.length === 0) return;
    this.queue = this.spare;
    this.spare = q;
    try {
      for (let i = 0; i < q.length; i++) post(q[i].msg, q[i].transfer);
    } finally {
      q.length = 0;
    }
  }
}
