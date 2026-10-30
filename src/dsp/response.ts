import { Biquad, biquadResponse, coeffRampSamples, highpass, onePoleHighpass, onePoleLowpass } from './biquad.ts';
import type { BiquadCoeffs } from './biquad.ts';
import { besselLowpassSections } from './bessel-lp.ts';
import { cDelay, cMul } from './complex.ts';
import type { Complex } from './complex.ts';
import { DcBlocker, dcBlockerResponse } from './dc-blocker.ts';
import { hilbertResponse } from './hilbert-pair.ts';
import { clampParams, computeChannelDelay, dbToLin } from './params.ts';
import type { EngineParams, FilterParams, TonalParams } from './params.ts';
import { ResonatorBank, resonatorBankResponse } from './resonator-bank.ts';

/** Q of the 2nd-order protective high-pass (Butterworth). */
export const HP_Q = Math.SQRT1_2;

/** The conditioning biquads, in processing order. The engine and the response functions both use this. */
export interface ConditioningSections {
  /** 2nd-order Butterworth or 1st-order high-pass, or null when disabled. */
  hp: BiquadCoeffs | null;
  /** 1st-order barrier low-pass, or null when disabled. */
  barrier: BiquadCoeffs | null;
  /** Bessel low-pass sections (1 or 2), empty when disabled. */
  bessel: BiquadCoeffs[];
}

export function conditioningSections(f: FilterParams, fs: number): ConditioningSections {
  return {
    hp: f.hpEnabled ? (f.hpOrder === 1 ? onePoleHighpass(fs, f.hpFreq) : highpass(fs, f.hpFreq, HP_Q)) : null,
    barrier: f.barrierEnabled ? onePoleLowpass(fs, f.barrierFreq) : null,
    bessel: f.lpEnabled ? besselLowpassSections(fs, f.lpFreq, f.lpOrder) : [],
  };
}

/** Number of biquad slots of ConditioningChain: HP, barrier, two Bessel sections. */
const SLOTS = 4;

/**
 * Time-domain conditioning chain: DC blocker → HP if enabled → barrier LP if enabled → Bessel LP if
 * enabled → resonator bank if phaseMode === 'tonal'. Exactly what conditioningResponse() describes.
 * configure() may allocate; process() does not. Coefficient updates keep the filter states and glide to
 * the new coefficients over COEFF_RAMP_MS (no click, see Biquad.setCoeffs and ResonatorBank.configure);
 * a section that becomes enabled starts from a clean state with its coefficients at once, and reset()
 * completes any glide.
 */
export class ConditioningChain {
  private readonly fs: number;
  private readonly dc: DcBlocker;
  private readonly slots: Biquad[] = [];
  private readonly active = new Uint8Array(SLOTS);
  private readonly bank: ResonatorBank;
  /** Coefficient glide length in samples. */
  private readonly ramp: number;
  private tonal = false;

  constructor(fs: number, params?: EngineParams) {
    this.fs = fs;
    this.ramp = coeffRampSamples(fs);
    this.dc = new DcBlocker(fs);
    for (let i = 0; i < SLOTS; i++) this.slots.push(new Biquad());
    this.bank = new ResonatorBank(fs);
    this.configure(params ?? clampParams(undefined));
  }

  configure(params: EngineParams): void {
    const p = clampParams(params);
    const s = conditioningSections(p.filters, this.fs);
    const wanted: (BiquadCoeffs | null)[] = [s.hp, s.barrier, s.bessel[0] ?? null, s.bessel[1] ?? null];
    for (let i = 0; i < SLOTS; i++) {
      const c = wanted[i];
      if (c === null) {
        this.active[i] = 0;
        continue;
      }
      if (this.active[i] === 0) {
        this.slots[i].reset();
        this.slots[i].setCoeffs(c);
      } else this.slots[i].setCoeffs(c, this.ramp);
      this.active[i] = 1;
    }
    const tonal = p.phaseMode === 'tonal';
    if (tonal) {
      this.bank.configure(p.tonal.f0, p.tonal.harmonics, p.tonal.q);
      if (!this.tonal) this.bank.reset();
    }
    this.tonal = tonal;
  }

  process(x: number): number {
    let y = this.dc.process(Number.isFinite(x) ? x : 0);
    for (let i = 0; i < SLOTS; i++) if (this.active[i] === 1) y = this.slots[i].process(y);
    return this.tonal ? this.bank.process(y) : y;
  }

  reset(): void {
    this.dc.reset();
    for (let i = 0; i < SLOTS; i++) this.slots[i].reset();
    this.bank.reset();
  }
}

const tonalResponse = (t: TonalParams, fs: number, f: number): Complex =>
  resonatorBankResponse(fs, f, t.f0, t.harmonics, t.q);

/**
 * Conditioning chain (DC blocker → HP if enabled → barrier LP if enabled → Bessel LP if enabled →
 * resonator bank if phaseMode === 'tonal'), exactly as the engine implements it (ConditioningChain).
 * `p` is clamped with clampParams first, as the engine does.
 */
export function conditioningResponse(p: EngineParams, fs: number, f: number): Complex {
  const q = clampParams(p);
  const s = conditioningSections(q.filters, fs);
  let h = dcBlockerResponse(fs, f);
  if (s.hp) h = cMul(h, biquadResponse(s.hp, f, fs));
  if (s.barrier) h = cMul(h, biquadResponse(s.barrier, f, fs));
  for (const c of s.bessel) h = cMul(h, biquadResponse(c, f, fs));
  if (q.phaseMode === 'tonal') h = cMul(h, tonalResponse(q.tonal, fs, f));
  return h;
}

/**
 * Full channel response from reference input to final output (before limiter, assuming no limiting):
 * conditioning · e^{-jω·computeChannelDelay(p, fs, ch).samples} · (hilbert rotation if mode hilbert,
 * i.e. cosΔ·A − sinΔ·B) · (−1 if polarityInvert) · channel gain · master gain. 0 if channel disabled.
 * The tap's Lagrange interpolation is modelled as an ideal delay (error < 0.01 dB / 0.001 samples
 * below 1 kHz at 48 kHz).
 */
export function channelResponse(p: EngineParams, fs: number, f: number, ch: 0 | 1): Complex {
  const q = clampParams(p);
  const c = q.channels[ch];
  if (!c.enabled) return { re: 0, im: 0 };
  let h = cMul(conditioningResponse(q, fs, f), cDelay(f, fs, computeChannelDelay(q, fs, ch).samples));
  if (q.phaseMode === 'hilbert') {
    const { a, b } = hilbertResponse(f, fs);
    const d = (c.phaseDeltaDeg * Math.PI) / 180;
    const cs = Math.cos(d);
    const sn = Math.sin(d);
    h = cMul(h, { re: cs * a.re - sn * b.re, im: cs * a.im - sn * b.im });
  }
  const g = (c.polarityInvert ? -1 : 1) * dbToLin(c.gainDb) * dbToLin(q.masterDb);
  return { re: h.re * g, im: h.im * g };
}

/**
 * Group delay in samples of an arbitrary response function, by central difference of unwrapped phase
 * (step ≤ 0.05 Hz, valid for delays up to ~fs/0.4 samples, i.e. well beyond the 0.53 s maximum).
 */
export function groupDelaySamples(h: (f: number) => Complex, f: number, fs: number): number {
  const df = Math.max(1e-4, Math.min(0.05, 0.01 * Math.abs(f)));
  const fc = Math.max(f, 2 * df);
  const lo = h(fc - df);
  const hi = h(fc + df);
  // phase of hi·conj(lo) is the phase increment, already unwrapped to (−π, π]
  const re = hi.re * lo.re + hi.im * lo.im;
  const im = hi.im * lo.re - hi.re * lo.im;
  if (re === 0 && im === 0) return 0;
  return (-Math.atan2(im, re) / (2 * Math.PI * 2 * df)) * fs;
}
