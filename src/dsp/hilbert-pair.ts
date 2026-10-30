import type { Complex } from './complex.ts';
import { GAIN_SMOOTH_MS } from './params.ts';

/** Olli Niemitalo's 90° phase-difference allpass pair (8 sections, ±0.7° from ~30 Hz to ~20 kHz at 44.1/48 kHz). */
const COEFF_A = [0.6923878, 0.9360654322959, 0.988229522686, 0.9987488452737] as const;
const COEFF_B = [0.4021921162426, 0.856171088242, 0.9722909545651, 0.9952884791278] as const;
const A2 = COEFF_A.map((a) => a * a);
const B2 = COEFF_B.map((b) => b * b);
const SECTIONS = 4;
const DENORMAL = 1e-30;

/**
 * Olli Niemitalo's IIR 90° phase-difference pair. Path A has an extra z⁻¹; each section is
 * out = a²(in + out[n−2]) − in[n−2], i.e. H(z) = (a² − z⁻²)/(1 − a² z⁻²). B leads A by 90°.
 */
export class HilbertPair {
  /** Per section: in[n−1], in[n−2], out[n−1], out[n−2]; sections 0..3 = path A, 4..7 = path B. */
  private readonly s = new Float64Array(4 * 2 * SECTIONS);
  private aDelay = 0;
  private outA = 0;
  private outB = 0;

  process(x: number): void {
    const s = this.s;
    let a = Number.isFinite(x) ? x : 0;
    let b = a;
    for (let i = 0; i < SECTIONS; i++) {
      const o = 4 * i;
      const y = A2[i] * (a + s[o + 3]) - s[o + 1];
      s[o + 1] = s[o];
      s[o] = a;
      s[o + 3] = s[o + 2];
      s[o + 2] = Math.abs(y) < DENORMAL ? 0 : y;
      a = y;
    }
    for (let i = 0; i < SECTIONS; i++) {
      const o = 4 * (SECTIONS + i);
      const y = B2[i] * (b + s[o + 3]) - s[o + 1];
      s[o + 1] = s[o];
      s[o] = b;
      s[o + 3] = s[o + 2];
      s[o + 2] = Math.abs(y) < DENORMAL ? 0 : y;
      b = y;
    }
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      this.reset();
      return;
    }
    this.outA = this.aDelay;
    this.aDelay = a;
    this.outB = b;
  }

  /** Path A output (delayed one sample). */
  get a(): number {
    return this.outA;
  }

  /** Path B output, leading A by 90°. */
  get b(): number {
    return this.outB;
  }

  reset(): void {
    this.s.fill(0);
    this.aDelay = 0;
    this.outA = 0;
    this.outB = 0;
  }
}

/**
 * y = cos Δ·A − sin Δ·B (positive Δ = lag relative to A).
 * Δ changes are smoothed per sample (one-pole of GAIN_SMOOTH_MS along the shortest arc) so that a new
 * phase never produces a click; `fs` only sets that smoothing time (default 48 kHz).
 */
export class PhaseRotator {
  private readonly pair = new HilbertPair();
  private readonly k: number;
  /** Current (smoothed, unwrapped) and target angles in radians. */
  private angle = 0;
  private targetAngle = 0;
  private c = 1;
  private s = 0;

  constructor(fs = 48000) {
    this.k = 1 - Math.exp(-1000 / (GAIN_SMOOTH_MS * fs));
  }

  /** Non-finite values are ignored. `immediate` skips the smoothing (use only while silent). */
  setDeltaDeg(deg: number, immediate = false): void {
    if (!Number.isFinite(deg)) return;
    const want = (deg * Math.PI) / 180;
    // shortest arc from the current angle, keeping the angle unwrapped
    let d = (want - this.angle) % (2 * Math.PI);
    if (d > Math.PI) d -= 2 * Math.PI;
    else if (d < -Math.PI) d += 2 * Math.PI;
    this.targetAngle = this.angle + d;
    if (immediate) this.snap();
  }

  private snap(): void {
    // land on the target and re-wrap so the unwrapped angle never grows without bound
    const a = Math.atan2(Math.sin(this.targetAngle), Math.cos(this.targetAngle));
    this.angle = a;
    this.targetAngle = a;
    this.c = Math.cos(a);
    this.s = Math.sin(a);
  }

  process(x: number): number {
    if (this.angle !== this.targetAngle) {
      const d = this.targetAngle - this.angle;
      if (Math.abs(d) < 1e-9) {
        this.snap();
      } else {
        this.angle += d * this.k;
        this.c = Math.cos(this.angle);
        this.s = Math.sin(this.angle);
      }
    }
    this.pair.process(x);
    return this.c * this.pair.a - this.s * this.pair.b;
  }

  /** Current (smoothed) Δ in degrees, wrapped to (−180, 180]. */
  get currentDeg(): number {
    return (Math.atan2(this.s, this.c) * 180) / Math.PI;
  }

  /** Clears the filter state and lands on the target angle. */
  reset(): void {
    this.pair.reset();
    this.snap();
  }
}

const allpassSection = (a2: number, w: number): Complex => {
  // (a² − e^{-2jω}) / (1 − a² e^{-2jω})
  const zr = Math.cos(2 * w);
  const zi = -Math.sin(2 * w);
  const nr = a2 - zr;
  const ni = -zi;
  const dr = 1 - a2 * zr;
  const di = -a2 * zi;
  const d = dr * dr + di * di;
  return { re: (nr * dr + ni * di) / d, im: (ni * dr - nr * di) / d };
};

/** Complex responses of paths A (including its z⁻¹) and B at frequency f. */
export function hilbertResponse(f: number, fs: number): { a: Complex; b: Complex } {
  const w = (2 * Math.PI * f) / fs;
  let ar = Math.cos(w);
  let ai = -Math.sin(w);
  let br = 1;
  let bi = 0;
  for (let i = 0; i < SECTIONS; i++) {
    const sa = allpassSection(A2[i], w);
    const sb = allpassSection(B2[i], w);
    const nar = ar * sa.re - ai * sa.im;
    ai = ar * sa.im + ai * sa.re;
    ar = nar;
    const nbr = br * sb.re - bi * sb.im;
    bi = br * sb.im + bi * sb.re;
    br = nbr;
  }
  return { a: { re: ar, im: ai }, b: { re: br, im: bi } };
}
