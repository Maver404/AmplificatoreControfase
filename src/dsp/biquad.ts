import type { Complex } from './complex.ts';

/** Biquad coefficients normalised so that a0 = 1: H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²). */
export interface BiquadCoeffs {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

export const IDENTITY: BiquadCoeffs = { b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 };

const DENORMAL = 1e-30;

function rbj(fs: number, f0: number, q: number) {
  const w0 = (2 * Math.PI * f0) / fs;
  return { cw: Math.cos(w0), alpha: Math.sin(w0) / (2 * q) };
}

/** RBJ cookbook 2nd-order low-pass (Q is linear, not dB). */
export function lowpass(fs: number, f0: number, q: number): BiquadCoeffs {
  const { cw, alpha } = rbj(fs, f0, q);
  const a0 = 1 + alpha;
  const b = (1 - cw) / 2 / a0;
  return { b0: b, b1: 2 * b, b2: b, a1: (-2 * cw) / a0, a2: (1 - alpha) / a0 };
}

/** RBJ cookbook 2nd-order high-pass (Q is linear, not dB). */
export function highpass(fs: number, f0: number, q: number): BiquadCoeffs {
  const { cw, alpha } = rbj(fs, f0, q);
  const a0 = 1 + alpha;
  const b = (1 + cw) / 2 / a0;
  return { b0: b, b1: -2 * b, b2: b, a1: (-2 * cw) / a0, a2: (1 - alpha) / a0 };
}

/** RBJ cookbook band-pass with constant 0 dB peak gain and zero phase at f0. */
export function bandpass(fs: number, f0: number, q: number): BiquadCoeffs {
  const { cw, alpha } = rbj(fs, f0, q);
  const a0 = 1 + alpha;
  return { b0: alpha / a0, b1: 0, b2: -alpha / a0, a1: (-2 * cw) / a0, a2: (1 - alpha) / a0 };
}

/** 1st-order low-pass via bilinear transform with pre-warping (b2 = a2 = 0). */
export function onePoleLowpass(fs: number, fc: number): BiquadCoeffs {
  const k = Math.tan((Math.PI * fc) / fs);
  const a0 = 1 + k;
  return { b0: k / a0, b1: k / a0, b2: 0, a1: (k - 1) / a0, a2: 0 };
}

/** 1st-order high-pass via bilinear transform with pre-warping (b2 = a2 = 0). */
export function onePoleHighpass(fs: number, fc: number): BiquadCoeffs {
  const k = Math.tan((Math.PI * fc) / fs);
  const a0 = 1 + k;
  return { b0: 1 / a0, b1: -1 / a0, b2: 0, a1: (k - 1) / a0, a2: 0 };
}

/** Complex frequency response of a biquad at frequency f. */
export function biquadResponse(c: BiquadCoeffs, f: number, fs: number): Complex {
  const w = (2 * Math.PI * f) / fs;
  const c1 = Math.cos(w);
  const s1 = Math.sin(w);
  const c2 = Math.cos(2 * w);
  const s2 = Math.sin(2 * w);
  const nr = c.b0 + c.b1 * c1 + c.b2 * c2;
  const ni = -(c.b1 * s1 + c.b2 * s2);
  const dr = 1 + c.a1 * c1 + c.a2 * c2;
  const di = -(c.a1 * s1 + c.a2 * s2);
  const d = dr * dr + di * di;
  return { re: (nr * dr + ni * di) / d, im: (ni * dr - nr * di) / d };
}

/** Duration of the coefficient glide used for live filter changes (ConditioningChain, ResonatorBank). */
export const COEFF_RAMP_MS = 20;

/** COEFF_RAMP_MS in whole samples at `fs`. */
export const coeffRampSamples = (fs: number): number => Math.max(2, Math.round((COEFF_RAMP_MS / 1000) * fs));

/** Glide shape: smoothstep 3u² − 2u³, zero slope at both ends so the coefficient path has no corner. */
export const rampShape = (u: number): number => u * u * (3 - 2 * u);

/**
 * Direct-form I biquad in double precision, allocation-free per sample. Its state is the last two inputs
 * and outputs, which do not depend on the coefficients, so they can change while it runs:
 * setCoeffs(c, n) glides to c along rampShape over n samples (a click-free live change), setCoeffs(c)
 * switches at once. Every intermediate set is a convex combination of the start and the target, so if
 * both are stable so is each frozen intermediate filter (the stability triangle is convex). Rounding:
 * ≈ 1e-11 of the output RMS for a 10 Hz 2nd-order high-pass at 192 kHz, like the transposed form II.
 */
export class Biquad {
  /** Coefficients in use. */
  private b0 = 1;
  private b1 = 0;
  private b2 = 0;
  private a1 = 0;
  private a2 = 0;
  /** Glide start and target (target = coefficients in use when not gliding). */
  private readonly from: BiquadCoeffs = { ...IDENTITY };
  private readonly to: BiquadCoeffs = { ...IDENTITY };
  private pos = 0;
  /** Glide length in samples, 0 when not gliding. */
  private len = 0;
  private x1 = 0;
  private x2 = 0;
  private y1 = 0;
  private y2 = 0;

  constructor(coeffs: BiquadCoeffs = IDENTITY) {
    this.setCoeffs(coeffs);
  }

  /**
   * rampSamples ≤ 1 (default): the next sample uses `coeffs`. rampSamples > 1: glides from the
   * coefficients in use to `coeffs` over that many samples; a new target mid-glide starts a new glide
   * from where it is, the current target again changes nothing. A filter at rest (zero state: new, reset
   * or after exact silence) has nothing to protect and switches at once.
   */
  setCoeffs(coeffs: BiquadCoeffs, rampSamples = 0): void {
    const t = this.to;
    const n = Number.isFinite(rampSamples) ? Math.floor(rampSamples) : 0;
    if (n > 1 && !this.atRest) {
      if (coeffs.b0 === t.b0 && coeffs.b1 === t.b1 && coeffs.b2 === t.b2 && coeffs.a1 === t.a1 && coeffs.a2 === t.a2) {
        return;
      }
      const f = this.from;
      f.b0 = this.b0;
      f.b1 = this.b1;
      f.b2 = this.b2;
      f.a1 = this.a1;
      f.a2 = this.a2;
      this.pos = 0;
      this.len = n;
    } else {
      this.len = 0;
      this.b0 = coeffs.b0;
      this.b1 = coeffs.b1;
      this.b2 = coeffs.b2;
      this.a1 = coeffs.a1;
      this.a2 = coeffs.a2;
    }
    t.b0 = coeffs.b0;
    t.b1 = coeffs.b1;
    t.b2 = coeffs.b2;
    t.a1 = coeffs.a1;
    t.a2 = coeffs.a2;
  }

  /** Coefficients in use (mid-glide: the interpolated ones), as a copy. */
  get coeffs(): BiquadCoeffs {
    return { b0: this.b0, b1: this.b1, b2: this.b2, a1: this.a1, a2: this.a2 };
  }

  get ramping(): boolean {
    return this.len !== 0;
  }

  /** True when the state is all zero (new, reset, or flushed after silence). */
  get atRest(): boolean {
    return this.x1 === 0 && this.x2 === 0 && this.y1 === 0 && this.y2 === 0;
  }

  /** Clears the state and completes any glide (a cleared filter has no history a jump could disturb). */
  reset(): void {
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
    if (this.len !== 0) this.finish();
  }

  process(x: number): number {
    if (this.len !== 0) this.glide();
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    if (!Number.isFinite(y)) {
      this.reset();
      return 0;
    }
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    // Manual flush-to-zero: JS cannot set FTZ and decaying outputs can go denormal on silence.
    this.y1 = Math.abs(y) < DENORMAL ? 0 : y;
    return y;
  }

  private glide(): void {
    const k = ++this.pos;
    if (k >= this.len) {
      this.finish();
      return;
    }
    const w = rampShape(k / this.len);
    const f = this.from;
    const t = this.to;
    this.b0 = f.b0 + w * (t.b0 - f.b0);
    this.b1 = f.b1 + w * (t.b1 - f.b1);
    this.b2 = f.b2 + w * (t.b2 - f.b2);
    this.a1 = f.a1 + w * (t.a1 - f.a1);
    this.a2 = f.a2 + w * (t.a2 - f.a2);
  }

  private finish(): void {
    const t = this.to;
    this.b0 = t.b0;
    this.b1 = t.b1;
    this.b2 = t.b2;
    this.a1 = t.a1;
    this.a2 = t.a2;
    this.len = 0;
  }
}
