/** Per-sample parameter smoothers. Allocation-free in next(). */

/** Distance below which a smoother snaps exactly onto its target (avoids endless denormal tails). */
const SNAP = 1e-12;

const onePoleCoeff = (fs: number, timeMs: number): number =>
  timeMs > 0 && Number.isFinite(timeMs) ? Math.exp(-1000 / (timeMs * fs)) : 0;

/** Exponential smoother: v ← target + a·(v − target), time constant `timeMs`. */
export class OnePole {
  private readonly a: number;
  private v: number;
  private t: number;

  constructor(fs: number, timeMs: number, initial = 0) {
    this.a = onePoleCoeff(fs, timeMs);
    const v0 = Number.isFinite(initial) ? initial : 0;
    this.v = v0;
    this.t = v0;
  }

  /** New target; non-finite values are ignored. */
  setTarget(v: number): void {
    if (Number.isFinite(v)) this.t = v;
  }

  /** Jump immediately (value and target). */
  set(v: number): void {
    if (!Number.isFinite(v)) return;
    this.v = v;
    this.t = v;
  }

  next(): number {
    const d = this.a * (this.v - this.t);
    this.v = Math.abs(d) < SNAP ? this.t : this.t + d;
    return this.v;
  }

  get value(): number {
    return this.v;
  }

  get target(): number {
    return this.t;
  }
}

/** Linear ramp to a target over a given time (soft-start 1.5 s, mute 15 ms, panic 10 ms). */
export class LinearEnvelope {
  private readonly fs: number;
  private v: number;
  private t: number;
  private inc = 0;
  private remaining = 0;

  constructor(fs: number, initial = 0) {
    this.fs = fs;
    const v0 = Number.isFinite(initial) ? initial : 0;
    this.v = v0;
    this.t = v0;
  }

  /** Ramp linearly from the current value to `target` in `ms` milliseconds (≤ 0 → immediate). */
  rampTo(target: number, ms: number): void {
    if (!Number.isFinite(target)) return;
    const n = Number.isFinite(ms) ? Math.round((ms * this.fs) / 1000) : 0;
    if (n <= 0) {
      this.set(target);
      return;
    }
    this.t = target;
    this.inc = (target - this.v) / n;
    this.remaining = n;
  }

  set(v: number): void {
    if (!Number.isFinite(v)) return;
    this.v = v;
    this.t = v;
    this.inc = 0;
    this.remaining = 0;
  }

  next(): number {
    if (this.remaining > 0) {
      this.remaining--;
      // land exactly on the target on the last step, whatever the rounding of inc
      this.v = this.remaining === 0 ? this.t : this.v + this.inc;
    }
    return this.v;
  }

  get value(): number {
    return this.v;
  }

  get target(): number {
    return this.t;
  }

  get done(): boolean {
    return this.remaining === 0;
  }
}
