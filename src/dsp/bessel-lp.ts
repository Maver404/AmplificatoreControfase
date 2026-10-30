import { lowpass } from './biquad.ts';
import type { BiquadCoeffs } from './biquad.ts';

/**
 * Second-order sections of the analog Bessel (maximally flat group delay) low-pass prototypes,
 * normalised so that the whole cascade is -3 dB at ω = 1: natural frequency (× cutoff) and Q.
 * Derived from the roots of the Bessel polynomials s²+3s+3 and s⁴+10s³+45s²+105s+105
 * (-3 dB points 1.3616541287 and 2.1139176749 of the delay-normalised forms).
 */
const PROTOTYPES: Record<2 | 4, readonly { r: number; q: number }[]> = {
  2: [{ r: 1.2720196495140694, q: 0.5773502691896257 }],
  4: [
    { r: 1.4301715599939904, q: 0.5219345816689801 },
    { r: 1.6033575162169729, q: 0.8055382818416659 },
  ],
};

const MIN_CUTOFF_HZ = 1;
const MAX_CUTOFF_FRACTION = 0.49;

/**
 * Cascade of biquads approximating a Bessel low-pass whose -3 dB point is `cutoff`.
 * The bilinear transform is pre-warped once at `cutoff` for the whole cascade, so the digital
 * cascade is exactly -3 dB at `cutoff` (each section is an RBJ low-pass whose own pre-warp frequency
 * is chosen so that its analog frequency is r·Ωc). Order 2: Q = 1/√3, f0 ≈ 1.272·cutoff.
 */
export function besselLowpassSections(fs: number, cutoff: number, order: 2 | 4): BiquadCoeffs[] {
  const fc = Math.min(MAX_CUTOFF_FRACTION * fs, Math.max(MIN_CUTOFF_HZ, Number.isFinite(cutoff) ? cutoff : MIN_CUTOFF_HZ));
  const warped = Math.tan((Math.PI * fc) / fs);
  const proto = PROTOTYPES[order === 4 ? 4 : 2];
  return proto.map(({ r, q }) => lowpass(fs, (fs / Math.PI) * Math.atan(r * warped), q));
}
