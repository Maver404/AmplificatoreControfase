import { DEFAULT_PARAMS, LIMITER_RELEASE_MS } from './params.ts';

/** Smallest gain the envelope may hold; below it the hard clip alone guarantees the ceiling. */
const MIN_GAIN = 1e-30;
const CEILING_MIN_DB = -120;
const CEILING_MAX_DB = 0;

/**
 * Zero-lookahead peak limiter: instant attack, LIMITER_RELEASE_MS release, then hard clip at the ceiling.
 * |output| ≤ ceiling is guaranteed for every input; non-finite input gives 0 and leaves the state intact.
 */
export class PeakLimiter {
  private readonly rel: number;
  private ceil = Math.pow(10, DEFAULT_PARAMS.ceilingDb / 20);
  private g = 1;

  constructor(fs: number, ceilingDb: number) {
    // Per-sample fraction of the remaining distance to unity recovered during release.
    this.rel = 1 - Math.exp(-1000 / (LIMITER_RELEASE_MS * fs));
    this.setCeilingDb(ceilingDb);
  }

  /** NaN is ignored (the previous ceiling stays); the ceiling is kept within [-120, 0] dBFS. */
  setCeilingDb(db: number): void {
    if (Number.isNaN(db)) return;
    const d = Math.min(CEILING_MAX_DB, Math.max(CEILING_MIN_DB, db));
    this.ceil = Math.pow(10, d / 20);
  }

  get ceiling(): number {
    return this.ceil;
  }

  process(x: number): number {
    if (!Number.isFinite(x)) return 0;
    const ceil = this.ceil;
    const a = x < 0 ? -x : x;
    const required = a > ceil ? ceil / a : 1;
    let g = this.g + (1 - this.g) * this.rel;
    if (required < g) g = required;
    if (g < MIN_GAIN) g = MIN_GAIN;
    this.g = g;
    const y = x * g;
    return y > ceil ? ceil : y < -ceil ? -ceil : y;
  }

  /** Current gain reduction in dB (≥ 0). */
  get gainReductionDb(): number {
    return this.g >= 1 ? 0 : -20 * Math.log10(this.g);
  }

  reset(): void {
    this.g = 1;
  }
}
