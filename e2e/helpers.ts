/**
 * Shared pieces of the e2e suite that do not depend on the Playwright runner (the config imports
 * this file too): fixture paths, the page-side globals and the signal analysis done in Node on the
 * planar tap blocks the e2e build exposes through window.__anc.
 */
import { fileURLToPath } from 'node:url';
import type { E2eHooks } from '../src/app/e2e-hooks.ts';

export const FIXTURE_FS = 48000;
export const FIXTURES = ['tone-200hz', 'noise', 'harmonic-100hz'] as const;
export type FixtureName = (typeof FIXTURES)[number];

/** The spec each fixture's project runs (anc.spec.ts keeps its two fixtures, trims.spec.ts gets its own). */
export const FIXTURE_SPECS: Record<FixtureName, string> = {
  'tone-200hz': 'anc.spec.ts',
  noise: 'anc.spec.ts',
  'harmonic-100hz': 'trims.spec.ts',
};

/** harmonic-100hz.wav: fundamental and the relative amplitude / phase (degrees, sine) of harmonics 1…4. */
export const HARMONIC_F0 = 100;
export const HARMONIC_PARTIALS: ReadonlyArray<{ amp: number; phaseDeg: number }> = [
  { amp: 1, phaseDeg: 0 },
  { amp: 0.6, phaseDeg: 45 },
  { amp: 0.4, phaseDeg: 110 },
  { amp: 0.3, phaseDeg: 250 },
];

export const fixturePath = (name: FixtureName): string =>
  fileURLToPath(new URL(`./fixtures/${name}.wav`, import.meta.url));

/** Recorded in the page by an init script, before the app registers its own listeners. */
export interface E2eProbe {
  csp: string[];
  /** performance.now() of every Escape keydown (window, capture phase, ahead of the app). */
  esc: number[];
}

/** Tap blocks copied in the page with their arrival time (performance.now()). */
export interface RecordedBlock {
  t: number;
  frames: number;
  outL: number[];
  outR: number[];
}

declare global {
  interface Window {
    __anc?: E2eHooks;
    __e2e?: E2eProbe;
    __rec?: RecordedBlock[];
  }
}

export const dbToLin = (db: number): number => Math.pow(10, db / 20);

/**
 * x[n − d] read exactly like DelayLine.read (3rd-order Lagrange, k = ⌊d⌋ − 1, μ = d − k ∈ [1, 2)),
 * so an integer delay is exact and a fractional one matches the engine to float precision.
 */
export function lagrangeAt(x: ArrayLike<number>, n: number, d: number): number {
  const k = Math.floor(d) - 1;
  const mu = d - k;
  const m1 = mu - 1;
  const m2 = mu - 2;
  const m3 = mu - 3;
  const i0 = n - k;
  const at = (i: number): number => x[i] as number;
  return (
    ((-m1 * m2 * m3) / 6) * at(i0) +
    ((mu * m2 * m3) / 2) * at(i0 - 1) +
    ((-mu * m1 * m3) / 2) * at(i0 - 2) +
    ((mu * m1 * m2) / 6) * at(i0 - 3)
  );
}

export interface AlignmentFit {
  /** Normalised correlation between the output and polarity·cond[n − d]. */
  corr: number;
  /** Least-squares gain from polarity·cond[n − d] to the output. */
  gain: number;
  /** Energy left after subtracting gain·polarity·cond[n − d], relative to the output (dB). */
  residualDb: number;
  /** Output RMS (linear). */
  outRms: number;
  samples: number;
}

/** Compares one tap block's output with the delayed, polarity-flipped conditioned reference. */
export function fitAlignment(cond: ArrayLike<number>, out: ArrayLike<number>, delay: number, polarity: 1 | -1): AlignmentFit {
  const n0 = Math.floor(delay) + 3;
  let so = 0;
  let sp = 0;
  let sop = 0;
  let count = 0;
  for (let n = n0; n < out.length; n++) {
    const p = polarity * lagrangeAt(cond, n, delay);
    const o = out[n] as number;
    so += o * o;
    sp += p * p;
    sop += o * p;
    count++;
  }
  const gain = sp > 0 ? sop / sp : 0;
  // Σ(o − g·p)² = Σo² − 2g·Σop + g²·Σp² = Σo² − Σop²/Σp² for the least-squares g.
  const resid = Math.max(0, so - gain * sop);
  return {
    corr: so > 0 && sp > 0 ? sop / Math.sqrt(so * sp) : 0,
    gain,
    residualDb: so > 0 ? 10 * Math.log10(Math.max(resid / so, 1e-30)) : 0,
    outRms: count > 0 ? Math.sqrt(so / count) : 0,
    samples: count,
  };
}

/** Integer lag in [0, maxLag] where polarity·cond[n − lag] correlates best with the output. */
export function bestIntegerLag(cond: ArrayLike<number>, out: ArrayLike<number>, maxLag: number, polarity: 1 | -1): number {
  let best = -1;
  let bestCorr = -Infinity;
  for (let lag = 0; lag <= maxLag; lag++) {
    let so = 0;
    let sp = 0;
    let sop = 0;
    for (let n = maxLag; n < out.length; n++) {
      const p = polarity * (cond[n - lag] as number);
      const o = out[n] as number;
      so += o * o;
      sp += p * p;
      sop += o * p;
    }
    const c = so > 0 && sp > 0 ? sop / Math.sqrt(so * sp) : -Infinity;
    if (c > bestCorr) {
      bestCorr = c;
      best = lag;
    }
  }
  return best;
}

/** Splits concatenated per-channel samples back into the tap blocks they came from. */
export function splitBlocks(samples: readonly number[], blockFrames: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i + blockFrames <= samples.length; i += blockFrames) out.push(samples.slice(i, i + blockFrames));
  return out;
}

export interface SilenceReport {
  /** Estimated sample index (in the recorded stream) at which Escape was pressed. */
  escSample: number;
  /** Last sample of either output with |y| ≥ threshold (−1 if none). */
  lastLoud: number;
  /** Total recorded samples. */
  total: number;
  /** Largest |y| before Escape (shows the output was actually playing). */
  peakBefore: number;
}

/**
 * Maps the Escape time onto the recorded tap stream and finds where the output went silent.
 * Each block's arrival time t_k is no earlier than the moment its last sample was rendered, so
 * end_k + (t_esc − t_k)·fs never overestimates the sample being rendered at t_esc; the maximum over
 * the blocks received before the key press is the tightest such bound. Underestimating it only makes
 * "silent within X ms of Escape" stricter.
 */
export function silenceAfterEscape(blocks: readonly RecordedBlock[], tEsc: number, fs: number, threshold: number): SilenceReport {
  let end = 0;
  let escSample = -Infinity;
  for (const b of blocks) {
    end += b.frames;
    if (b.t <= tEsc) escSample = Math.max(escSample, end + ((tEsc - b.t) / 1000) * fs);
  }
  let idx = 0;
  let lastLoud = -1;
  let peakBefore = 0;
  for (const b of blocks) {
    for (let i = 0; i < b.frames; i++, idx++) {
      const a = Math.max(Math.abs(b.outL[i] as number), Math.abs(b.outR[i] as number));
      if (a >= threshold) lastLoud = idx;
      if (idx < escSample) peakBefore = Math.max(peakBefore, a);
    }
  }
  return { escSample, lastLoud, total: idx, peakBefore };
}

// ------------------------------------------------------------------------------ lock-in (harmonic-100hz)

export interface Phasor {
  re: number;
  im: number;
}

/**
 * Complex amplitude of x at f over x[start … start + n): (2/n)·Σ x[k]·e^{−j2πf(k − start)/fs}. Every channel
 * analysed on the same window shares the time origin, so ratios between channels are transfer functions. Over a
 * whole number of periods of f0 the other harmonics of f0 cancel exactly (wholePeriods).
 */
export function lockIn(x: ArrayLike<number>, f: number, fs: number, start: number, n: number): Phasor {
  const w = (2 * Math.PI * f) / fs;
  let re = 0;
  let im = 0;
  for (let k = 0; k < n; k++) {
    const v = x[start + k] as number;
    re += v * Math.cos(w * k);
    im -= v * Math.sin(w * k);
  }
  return { re: (2 * re) / n, im: (2 * im) / n };
}

/** The last whole number of periods of f0 in `frames` samples (fs/f0 must be an integer number of samples). */
export function wholePeriods(frames: number, fs: number, f0: number): { start: number; n: number } {
  const period = fs / f0;
  if (!Number.isInteger(period)) throw new Error(`${fs} Hz / ${f0} Hz is not a whole number of samples`);
  const n = Math.floor(frames / period) * period;
  return { start: frames - n, n };
}

export const divide = (a: Phasor, b: Phasor): Phasor => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};

export const magnitudeDb = (z: Phasor): number => 20 * Math.log10(Math.hypot(z.re, z.im));

/** An angle in degrees wrapped into (−180, 180]. */
export const wrapDeg = (d: number): number => d - 360 * Math.ceil((d - 180) / 360);

/**
 * A complex ratio as a harmonic trim: gain in dB and phase LAG in degrees (−arg, wrapped), the convention of
 * HarmonicTrim (factor 10^(gainDb/20)·e^{−j·phaseDeg·π/180}).
 */
export const asTrim = (z: Phasor): { gainDb: number; phaseDeg: number } => ({
  gainDb: magnitudeDb(z),
  phaseDeg: wrapDeg((-Math.atan2(z.im, z.re) * 180) / Math.PI),
});
