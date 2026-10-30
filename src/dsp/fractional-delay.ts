import {
  D_MIN_SAMPLES,
  DELAY_CROSSFADE_MS,
  DELAY_JUMP_THRESHOLD_MS,
  DELAY_SLEW,
  MAX_DELAY_MS,
} from './params.ts';

/** Single-writer ring buffer with 3rd-order Lagrange interpolated reads. */
export class DelayLine {
  private readonly buf: Float64Array;
  private readonly mask: number;
  /** Index where the next sample will be written; the newest sample x[n] sits at w − 1. */
  private w = 0;
  /** Largest readable delay: taps x[n−k..n−k−3] must all still be in the buffer. */
  readonly maxDelay: number;

  /** `capacity` is rounded up to a power of two (use delayCapacity(fs)). */
  constructor(capacity: number) {
    let n = 8;
    while (n < capacity) n <<= 1;
    this.buf = new Float64Array(n);
    this.mask = n - 1;
    this.maxDelay = n - 3;
  }

  get capacity(): number {
    return this.buf.length;
  }

  /** Call once per sample BEFORE the reads of that sample. Non-finite input is stored as 0. */
  write(x: number): void {
    this.buf[this.w] = Number.isFinite(x) ? x : 0;
    this.w = (this.w + 1) & this.mask;
  }

  /**
   * x[n − d], d clamped to [D_MIN_SAMPLES, maxDelay]. With k = ⌊d⌋ − 1 and μ = d − k ∈ [1, 2) the
   * interpolation uses taps x[n−k], x[n−k−1], x[n−k−2], x[n−k−3], so the evaluation point always lies
   * between the two central taps (best Lagrange accuracy) and integer delays are exact.
   */
  read(delaySamples: number): number {
    let d = delaySamples;
    if (!(d >= D_MIN_SAMPLES)) d = D_MIN_SAMPLES; // also catches NaN
    else if (d > this.maxDelay) d = this.maxDelay;
    const k = Math.floor(d) - 1;
    const mu = d - k;
    const m1 = mu - 1;
    const m2 = mu - 2;
    const m3 = mu - 3;
    const h0 = (-m1 * m2 * m3) / 6;
    const h1 = (mu * m2 * m3) / 2;
    const h2 = (-mu * m1 * m3) / 2;
    const h3 = (mu * m1 * m2) / 6;
    const buf = this.buf;
    const mask = this.mask;
    const i0 = (this.w - 1 - k) & mask;
    return (
      h0 * buf[i0] + h1 * buf[(i0 - 1) & mask] + h2 * buf[(i0 - 2) & mask] + h3 * buf[(i0 - 3) & mask]
    );
  }

  reset(): void {
    this.buf.fill(0);
    this.w = 0;
  }
}

/**
 * One channel's read tap: slew-limited (DELAY_SLEW samples/sample) for small changes, 30 ms
 * raised-cosine crossfade between two read positions for jumps ≥ DELAY_JUMP_THRESHOLD_MS (or when
 * setTarget(..., true)). A jump requested during a crossfade starts when the current one ends.
 */
export class SmoothDelayTap {
  private readonly jumpThreshold: number;
  private readonly fadeLen: number;
  private readonly maxDelay: number;
  /** Delay of the main read position (the one being faded in during a crossfade). */
  private pos: number;
  /** Delay of the position being faded out. */
  private from = 0;
  private tgt: number;
  /** Remaining samples of the crossfade (0 = no crossfade). */
  private fadeLeft = 0;
  private jumpRequested = false;

  constructor(fs: number, initialDelay: number = D_MIN_SAMPLES) {
    this.jumpThreshold = (DELAY_JUMP_THRESHOLD_MS / 1000) * fs;
    this.fadeLen = Math.max(1, Math.round((DELAY_CROSSFADE_MS / 1000) * fs));
    // Same headroom as delayCapacity(): the longest delay plus Δ at the lowest f_ref.
    this.maxDelay = ((MAX_DELAY_MS + 30) / 1000) * fs;
    const d0 = this.clamp(initialDelay, D_MIN_SAMPLES);
    this.pos = d0;
    this.tgt = d0;
  }

  private clamp(d: number, fallback: number): number {
    if (!Number.isFinite(d)) return fallback;
    return d < D_MIN_SAMPLES ? D_MIN_SAMPLES : d > this.maxDelay ? this.maxDelay : d;
  }

  /** NaN is ignored. Delays are clamped to [D_MIN_SAMPLES, (MAX_DELAY_MS + 30 ms)·fs]. */
  setTarget(delaySamples: number, forceJump = false): void {
    if (Number.isNaN(delaySamples)) return;
    this.tgt = this.clamp(delaySamples, this.tgt);
    if (forceJump) this.jumpRequested = true;
  }

  /** One output sample from `line` (written for this sample already). */
  next(line: DelayLine): number {
    if (this.fadeLeft === 0) {
      const diff = this.tgt - this.pos;
      const big = diff >= this.jumpThreshold || -diff >= this.jumpThreshold;
      if (this.jumpRequested || big) {
        this.jumpRequested = false;
        if (diff !== 0) {
          this.from = this.pos;
          this.pos = this.tgt;
          this.fadeLeft = this.fadeLen;
        }
      } else if (diff !== 0) {
        this.pos += diff > DELAY_SLEW ? DELAY_SLEW : diff < -DELAY_SLEW ? -DELAY_SLEW : diff;
      }
    }
    if (this.fadeLeft > 0) {
      this.fadeLeft--;
      // raised cosine: g goes 0 → 1, the two weights always sum to 1
      const t = 1 - this.fadeLeft / this.fadeLen;
      const g = 0.5 - 0.5 * Math.cos(Math.PI * t);
      return (1 - g) * line.read(this.from) + g * line.read(this.pos);
    }
    return line.read(this.pos);
  }

  /** Current delay in samples (during a crossfade: the delay being faded in). */
  get current(): number {
    return this.pos;
  }

  get target(): number {
    return this.tgt;
  }

  /** True while crossfading between two read positions. */
  get crossfading(): boolean {
    return this.fadeLeft > 0;
  }

  /**
   * Longest delay this tap reads now or will move to: the read position, the position being faded out
   * and the target. Reads at delay d touch samples back to n − ⌊d⌋ − 2.
   */
  get reach(): number {
    const r = this.fadeLeft > 0 && this.from > this.pos ? this.from : this.pos;
    return this.tgt > r ? this.tgt : r;
  }

  /** Snap to the target with no slew or crossfade (use only while the output is silent). */
  snap(): void {
    this.pos = this.tgt;
    this.fadeLeft = 0;
    this.jumpRequested = false;
  }
}
