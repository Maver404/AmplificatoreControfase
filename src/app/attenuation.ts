/**
 * Broadband attenuation at the error mic, A = [L_err − L_ref]_off − [L_err − L_ref]_on, plus the
 * "ANC sta aumentando il rumore" guard (error worse than the bypass reference by more than 3 dB for
 * more than 2 s).
 *
 * While the anti-noise plays, the raw reference mic also hears it (feedback), so L_ref "on" is never
 * taken from the raw reference: either it is interpolated from the most recent ANC-off windows
 * ('interp'), or, when the feedback neutraliser is active, both sides use the conditioned x̂ tap
 * ('cond'), whose level does not depend on the anti-noise. In 'interp' the raw reference still tells
 * when the ambient noise drops while the ANC is on (an ANC that adds noise must not hide behind a
 * quieter source), but only by its change during the ANC-on run: the anti-noise it picks up shifts it
 * from the first window on (by + for broadband noise, by ± for a tone, whose feedback can cancel at the
 * reference), so the raw level is compared with its own level over the first second of the run, not
 * with the bypass one. A drop counts only when the mean over the last `averageS` is below that by more
 * than 1 dB and 3 standard errors (window scatter, measured on the bypass windows): single noisy windows
 * do not bias the readout. The stored bypass level is then lowered by the drop. `referenceAgeS` tells
 * how old the latest bypass reference is.
 *
 * Pure and deterministic: time is the count of pushed frames.
 */

export type AncPhase = 'on' | 'off' | 'transition';
export type RefSource = 'interp' | 'cond';

export interface AttenuationOptions {
  /** Analysis window, s (default 0.25). */
  windowS?: number;
  /** How much on/off history the readout averages, s (default 3). */
  averageS?: number;
  /** An ANC-off reference older than this is not used, s (default 300). */
  maxOffAgeS?: number;
  /** Guard: error above the bypass reference by more than this, dB (default 3)… */
  guardExcessDb?: number;
  /** …for at least this long, s (default 2). */
  guardHoldS?: number;
  /** Windows with error or reference below this level carry no information, dBFS (default −100). */
  minLevelDb?: number;
}

interface Win {
  t: number;
  errDb: number;
  refDb: number;
  condDb: number;
}

const toDb = (meanSquare: number): number => (meanSquare > 0 ? 10 * Math.log10(meanSquare) : -Infinity);

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

const stdDev = (xs: readonly number[]): number => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) * (x - m), 0) / (xs.length - 1));
};

/** ANC-on windows (0.25 s each) whose raw reference is the run's baseline ('interp'). */
const RUN_BASE_WINDOWS = 4;
/** A raw-reference drop smaller than this (or than 3 standard errors) is not taken for an ambient drop, dB. */
const RAW_DROP_MIN_DB = 1;

export class AttenuationEstimator {
  private readonly fs: number;
  private readonly winFrames: number;
  private readonly avgWindows: number;
  private readonly averageS: number;
  private readonly maxOffAgeS: number;
  private readonly guardExcessDb: number;
  private readonly guardFrames: number;
  private readonly minLevelDb: number;
  private source: RefSource = 'interp';

  private frames = 0;
  private phase: AncPhase = 'transition';
  private accFrames = 0;
  private accErr = 0;
  private accRef = 0;
  private accCond = 0;
  private off: Win[] = [];
  /** ANC-on windows: error level relative to the reference estimate, dB. */
  private on: Array<{ t: number; ratio: number }> = [];
  /** Raw reference (dB) of the ANC-on run since the latest bypass window: its first windows, the recent ones. */
  private runBase: number[] = [];
  private runRecent: Array<{ t: number; db: number }> = [];
  private worseFrames = 0;
  private att: number | null = null;
  private excess: number | null = null;

  constructor(fs: number, opts: AttenuationOptions = {}) {
    if (!(fs > 0)) throw new RangeError('AttenuationEstimator: fs must be positive');
    this.fs = fs;
    const windowS = opts.windowS ?? 0.25;
    this.winFrames = Math.max(1, Math.round(windowS * fs));
    this.averageS = opts.averageS ?? 3;
    this.avgWindows = Math.max(1, Math.round(this.averageS / windowS));
    this.maxOffAgeS = opts.maxOffAgeS ?? 300;
    this.guardExcessDb = opts.guardExcessDb ?? 3;
    this.guardFrames = Math.round((opts.guardHoldS ?? 2) * fs);
    this.minLevelDb = opts.minLevelDb ?? -100;
  }

  get refSource(): RefSource {
    return this.source;
  }

  /** Changing the source makes the stored ratios incomparable: history is cleared. */
  setRefSource(s: RefSource): void {
    if (s === this.source) return;
    this.source = s;
    this.reset();
  }

  /** Latest broadband attenuation, dB (positive = quieter with ANC), or null. */
  get attenuationDb(): number | null {
    return this.att;
  }

  /** Latest guard excess (error vs bypass reference, dB), or null. */
  get excessDb(): number | null {
    return this.excess;
  }

  /** Seconds of signal pushed so far. */
  get timeS(): number {
    return this.frames / this.fs;
  }

  /** Age of the newest ANC-off window (s), null without one. */
  get referenceAgeS(): number | null {
    const last = this.off[this.off.length - 1];
    return last ? this.frames / this.fs - last.t : null;
  }

  reset(): void {
    this.off = [];
    this.on = [];
    this.runBase = [];
    this.runRecent = [];
    this.att = null;
    this.excess = null;
    this.worseFrames = 0;
    this.clearAcc(this.phase);
  }

  /**
   * One block of `frames` samples with the mean squares of the error, raw reference and conditioned
   * reference. Returns true when the guard trips (once; it re-arms after `guardHoldS` more).
   */
  push(phase: AncPhase, frames: number, errMs: number, refMs: number, condMs: number): boolean {
    if (!(frames > 0)) return false;
    this.frames += frames;
    if (phase !== this.phase) {
      this.clearAcc(phase);
      if (phase !== 'on') this.worseFrames = 0;
    }
    if (phase === 'transition') {
      this.worseFrames = 0;
      return false;
    }
    if (!Number.isFinite(errMs) || !Number.isFinite(refMs) || !Number.isFinite(condMs)) {
      this.clearAcc(phase);
      return false;
    }
    this.accFrames += frames;
    this.accErr += errMs * frames;
    this.accRef += refMs * frames;
    this.accCond += condMs * frames;
    if (this.accFrames < this.winFrames) return false;
    const n = this.accFrames;
    const w: Win = {
      t: this.frames / this.fs,
      errDb: toDb(this.accErr / n),
      refDb: toDb(this.accRef / n),
      condDb: toDb(this.accCond / n),
    };
    this.clearAcc(phase);
    return phase === 'on' ? this.closeOn(w, n) : this.closeOff(w);
  }

  private clearAcc(phase: AncPhase): void {
    this.phase = phase;
    this.accFrames = 0;
    this.accErr = 0;
    this.accRef = 0;
    this.accCond = 0;
  }

  private base(w: Win): number {
    return this.source === 'cond' ? w.condDb : w.refDb;
  }

  private closeOff(w: Win): boolean {
    this.worseFrames = 0;
    if (!(w.errDb > this.minLevelDb) || !(this.base(w) > this.minLevelDb)) return false;
    this.off.push(w);
    // A new bypass reference: the next ANC-on run starts its own raw-reference baseline.
    this.runBase = [];
    this.runRecent = [];
    const minT = w.t - this.maxOffAgeS;
    while (this.off.length > 0 && (this.off.length > 4 * this.avgWindows || this.off[0]!.t < minT)) this.off.shift();
    return false;
  }

  /** Off windows still recent enough to serve as the bypass reference. */
  private recentOff(t: number): Win[] {
    const minT = t - this.maxOffAgeS;
    return this.off.filter((o) => o.t >= minT).slice(-this.avgWindows);
  }

  private closeOn(w: Win, n: number): boolean {
    const off = this.recentOff(w.t);
    if (off.length === 0 || !(w.errDb > this.minLevelDb)) {
      this.worseFrames = 0;
      return false;
    }
    const offRatio = mean(off.map((o) => o.errDb - this.base(o)));
    let baseOn: number;
    if (this.source === 'cond') {
      if (!(w.condDb > this.minLevelDb)) return false;
      baseOn = w.condDb;
    } else {
      baseOn = mean(off.slice(-4).map((o) => o.refDb)) - this.ambientDrop(w, off);
    }
    const ratio = w.errDb - baseOn;
    this.on.push({ t: w.t, ratio });
    const minT = w.t - this.averageS;
    while (this.on.length > 0 && this.on[0]!.t <= minT) this.on.shift();
    this.att = offRatio - mean(this.on.map((o) => o.ratio));

    this.excess = ratio - offRatio;
    if (this.excess > this.guardExcessDb) this.worseFrames += n;
    else this.worseFrames = 0;
    if (this.worseFrames >= this.guardFrames) {
      this.worseFrames = 0;
      return true;
    }
    return false;
  }

  /**
   * 'interp': how far the ambient noise has dropped during this ANC-on run, dB (0 = no evidence): the raw
   * reference's mean over the last `averageS` below its mean over the run's first RUN_BASE_WINDOWS by
   * more than RAW_DROP_MIN_DB and 3 standard errors (window scatter from the bypass windows `off`).
   */
  private ambientDrop(w: Win, off: readonly Win[]): number {
    if (!(w.refDb > this.minLevelDb)) return 0;
    if (this.runBase.length < RUN_BASE_WINDOWS) {
      this.runBase.push(w.refDb);
      return 0;
    }
    this.runRecent.push({ t: w.t, db: w.refDb });
    const minT = w.t - this.averageS;
    while (this.runRecent.length > 0 && this.runRecent[0]!.t <= minT) this.runRecent.shift();
    const recent = this.runRecent.map((r) => r.db);
    const drop = mean(this.runBase) - mean(recent);
    const se = stdDev(off.map((o) => o.refDb)) * Math.sqrt(1 / this.runBase.length + 1 / recent.length);
    return drop > Math.max(RAW_DROP_MIN_DB, 3 * se) ? drop : 0;
  }
}
