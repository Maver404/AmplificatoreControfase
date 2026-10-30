import { BIG_JUMP_DB, GAIN_SMOOTH_MS, MASTER_MIN_DB, MAX_RISE_DB_PER_S, RANGES } from './params.ts';

/** Highest gain a ramp may ever reach (channel gain with the +12 dB unlock). */
const MAX_DB = RANGES.gainDb.maxUnlocked;
/**
 * When the target is silence the one-pole aims this far below MASTER_MIN_DB, so the gain crosses
 * the floor in finite time (≈ 1.7 τ from -30 dB) and then sits exactly on it (output exactly 0).
 */
const FLOOR_OVERSHOOT_DB = 20;
const SNAP_DB = 1e-9;
/**
 * Rises that start below this level reach it with the plain one-pole and are rate-limited only above
 * it: otherwise un-muting a master at -∞ would take 14 s, 10 of them inaudible. Silence on (un)mute
 * comes from the mute/soft-start envelopes, not from the master at -∞. Default of setRiseFloorDb().
 */
export const RISE_FLOOR_DB = -60;

/**
 * Channel/master gain smoothing with the safety rule: decreases and increases ≤ BIG_JUMP_DB use a
 * one-pole of GAIN_SMOOTH_MS (in dB); while (target − current) > BIG_JUMP_DB the gain rises in dB at
 * no more than MAX_RISE_DB_PER_S. Works on dB internally, outputs linear gain; ≤ MASTER_MIN_DB → exactly 0.
 * Below the rise floor (RISE_FLOOR_DB unless set) rises are not rate-limited: the gain goes up to the
 * floor with the one-pole (crossing it in finite time) and the rules above apply from there.
 *
 * Anti-bypass: every rise above the floor draws from a budget of BIG_JUMP_DB that refills at
 * MAX_RISE_DB_PER_S, so a series of small steps (e.g. +5 dB every 100 ms) cannot rise faster than
 * MAX_RISE_DB_PER_S on average: over any interval T the gain rises above the floor by at most
 * BIG_JUMP_DB + MAX_RISE_DB_PER_S·T.
 */
export class GainRamp {
  private readonly a: number;
  private readonly risePerSample: number;
  private cur: number;
  private tgt: number;
  /** Remaining fast-rise budget in dB, within [0, BIG_JUMP_DB]. */
  private budget = BIG_JUMP_DB;
  private floor = RISE_FLOOR_DB;
  private linDb = Number.NaN;
  private lin = 0;

  constructor(fs: number, initialDb: number) {
    this.a = Math.exp(-1000 / (GAIN_SMOOTH_MS * fs));
    this.risePerSample = MAX_RISE_DB_PER_S / fs;
    const d = GainRamp.clampDb(initialDb, MASTER_MIN_DB);
    this.cur = d;
    this.tgt = d;
  }

  private static clampDb(db: number, fallback: number): number {
    if (Number.isNaN(db)) return fallback;
    return db <= MASTER_MIN_DB ? MASTER_MIN_DB : db > MAX_DB ? MAX_DB : db;
  }

  /** NaN is ignored; −∞ and anything ≤ MASTER_MIN_DB mean silence; capped at +12 dB. */
  setTargetDb(db: number): void {
    this.tgt = GainRamp.clampDb(db, this.tgt);
  }

  /**
   * Level below which rises are free (default RISE_FLOOR_DB). A ramp carrying channel gain + master uses
   * channel gain + RISE_FLOOR_DB: free while the master part is below RISE_FLOOR_DB, as for a master alone.
   * NaN is ignored.
   */
  setRiseFloorDb(db: number): void {
    if (!Number.isNaN(db)) this.floor = GainRamp.clampDb(db, this.floor);
  }

  /** Moves immediately (no ramp) and refills the rise budget: use only while the output is silent. */
  jumpTo(db: number): void {
    const d = GainRamp.clampDb(db, this.cur);
    this.cur = d;
    this.tgt = d;
    this.budget = BIG_JUMP_DB;
  }

  /**
   * Next linear gain sample. `holdRise`: nobody hears this gain right now and the path can come back
   * within milliseconds (bypass, measurement gate), so an increase waits there instead of completing
   * unheard: the gain stays where it was last heard (the budget still refills), decreases go on.
   */
  next(holdRise = false): number {
    let budget = this.budget + this.risePerSample;
    if (budget > BIG_JUMP_DB) budget = BIG_JUMP_DB;
    const gap = this.tgt - this.cur;
    if (gap > 0 && holdRise) {
      // Increase deferred until the gain is audible again.
    } else if (gap > 0 && this.cur < this.floor) {
      // Inaudible range: one-pole, no budget. Aiming past the floor crosses it in finite time (≈ 1.4 τ
      // from silence), then the gain sits on it and the rate-limited rules take over.
      const floor = this.floor;
      if (this.tgt <= floor) {
        const step = gap * (1 - this.a);
        this.cur = gap - step < SNAP_DB ? this.tgt : this.cur + step;
      } else {
        const aim = floor + FLOOR_OVERSHOOT_DB;
        const next = aim + this.a * (this.cur - aim);
        this.cur = next >= floor ? floor : next;
      }
    } else if (gap > BIG_JUMP_DB) {
      // Large increase: linear rise in dB. The budget refill equals this rise, so it stays unchanged.
      const step = this.risePerSample < budget ? this.risePerSample : budget;
      this.cur += step;
      budget -= step;
    } else if (gap > 0) {
      let step = gap * (1 - this.a);
      if (step > budget) step = budget;
      this.cur = gap - step < SNAP_DB ? this.tgt : this.cur + step;
      budget -= gap - step < SNAP_DB ? gap : step;
      if (budget < 0) budget = 0;
    } else if (gap < 0) {
      const aim = this.tgt <= MASTER_MIN_DB ? MASTER_MIN_DB - FLOOR_OVERSHOOT_DB : this.tgt;
      const next = aim + this.a * (this.cur - aim);
      this.cur = next <= this.tgt || this.tgt - next > -SNAP_DB ? this.tgt : next;
    }
    this.budget = budget;
    if (this.cur !== this.linDb) {
      this.linDb = this.cur;
      this.lin = this.cur <= MASTER_MIN_DB ? 0 : Math.pow(10, this.cur / 20);
    }
    return this.lin;
  }

  get currentDb(): number {
    return this.cur;
  }

  get targetDb(): number {
    return this.tgt;
  }

  /** Current linear gain (as returned by the last next()). */
  get currentLin(): number {
    return this.cur <= MASTER_MIN_DB ? 0 : Math.pow(10, this.cur / 20);
  }
}
