import { MASTER_MIN_DB } from './params.ts';

const DENORMAL = 1e-30;
/** Peak fall rate after the hold time has expired. */
const PEAK_FALL_DB_PER_S = 20;

const toDb = (power: number): number =>
  power > 0 ? Math.max(MASTER_MIN_DB, 10 * Math.log10(power)) : MASTER_MIN_DB;

/**
 * Exponentially averaged RMS (time constant `rmsMs`) and peak with hold (`peakHoldMs`, then falls at
 * 20 dB/s). Levels in dBFS, floored at -120. Non-finite samples are treated as silence.
 */
export class LevelMeter {
  private readonly rmsCoeff: number;
  private readonly holdSamples: number;
  private readonly fall: number;
  private ms = 0;
  private peak = 0;
  private hold = 0;

  constructor(fs: number, rmsMs = 300, peakHoldMs = 1500) {
    this.rmsCoeff = rmsMs > 0 ? 1 - Math.exp(-1000 / (rmsMs * fs)) : 1;
    this.holdSamples = peakHoldMs > 0 ? Math.round((peakHoldMs * fs) / 1000) : 0;
    this.fall = Math.pow(10, -PEAK_FALL_DB_PER_S / (20 * fs));
  }

  push(x: number): void {
    const v = Number.isFinite(x) ? x : 0;
    const p = v * v;
    this.ms += this.rmsCoeff * (p - this.ms);
    if (this.ms < DENORMAL) this.ms = 0;
    const a = Math.abs(v);
    if (a >= this.peak) {
      this.peak = a;
      this.hold = this.holdSamples;
    } else if (this.hold > 0) {
      this.hold--;
    } else {
      this.peak = Math.max(a, this.peak * this.fall);
      if (this.peak < DENORMAL) this.peak = 0;
    }
  }

  get rmsDb(): number {
    return toDb(this.ms);
  }

  get peakDb(): number {
    return toDb(this.peak * this.peak);
  }

  reset(): void {
    this.ms = 0;
    this.peak = 0;
    this.hold = 0;
  }
}
