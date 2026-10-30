/**
 * Averaged spectrum of a stream of tap samples for the spectrum view: Hann-windowed FFT frames
 * (default 8192 points, 50 % overlap), exponentially averaged power, levels in dBFS where a
 * full-scale sine centred on a bin reads −3.01 dB (its RMS level).
 */
import { FFT } from '../dsp/fft.ts';

export const SPECTRUM_FFT_SIZE = 8192;
export const SPECTRUM_FLOOR_DB = -160;

export interface SpectrumOptions {
  fftSize?: number;
  /** Samples between frames (default fftSize / 2). */
  hop?: number;
  /** Weight of the newest frame in the exponential average, 0 < α ≤ 1 (default 0.3). */
  alpha?: number;
}

export class SpectrumAccumulator {
  readonly fftSize: number;
  private readonly hop: number;
  private readonly alpha: number;
  private readonly fft: FFT;
  private readonly window: Float64Array;
  /** Power normalisation: 2/(Σw)² turns |X|² of a sine into its mean square. */
  private readonly norm: number;
  private readonly ring: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly power: Float64Array;
  private readonly out: Float32Array;
  private pos = 0;
  private filled = 0;
  private sinceFrame = 0;
  private frames = 0;

  constructor(opts: SpectrumOptions = {}) {
    const n = opts.fftSize ?? SPECTRUM_FFT_SIZE;
    this.fft = new FFT(n);
    this.fftSize = n;
    const hop = Math.round(opts.hop ?? n / 2);
    this.hop = Math.min(n, Math.max(1, hop));
    const a = opts.alpha ?? 0.3;
    this.alpha = a > 0 && a <= 1 ? a : 0.3;
    this.window = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      // Periodic Hann: exact 50 % overlap-add, bin-centred sines land on one bin plus two neighbours.
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
      this.window[i] = w;
      sum += w;
    }
    this.norm = 2 / (sum * sum);
    this.ring = new Float64Array(n);
    this.re = new Float64Array(n);
    this.im = new Float64Array(n);
    this.power = new Float64Array(n / 2 + 1);
    this.out = new Float32Array(n / 2 + 1).fill(SPECTRUM_FLOOR_DB);
  }

  /** Number of frames averaged so far (0 = no data yet). */
  get frameCount(): number {
    return this.frames;
  }

  /**
   * Adds `count` samples of `x` starting at `offset` (non-finite values read as 0).
   * Returns true when at least one new frame was averaged in.
   */
  push(x: ArrayLike<number>, offset = 0, count = x.length - offset): boolean {
    const n = this.fftSize;
    const end = Math.min(x.length, offset + Math.max(0, count));
    let updated = false;
    for (let i = offset; i < end; i++) {
      const v = x[i] as number;
      this.ring[this.pos] = Number.isFinite(v) ? v : 0;
      this.pos = this.pos + 1 === n ? 0 : this.pos + 1;
      if (this.filled < n) {
        // First frame as soon as the window is full, then one every `hop` samples.
        if (++this.filled < n) continue;
      } else if (++this.sinceFrame < this.hop) {
        continue;
      }
      this.sinceFrame = 0;
      this.frame();
      updated = true;
    }
    return updated;
  }

  /** Averaged levels in dB, fftSize/2 + 1 bins (bin k = k·fs/fftSize). The array is reused. */
  get db(): Float32Array {
    return this.out;
  }

  /** Copy of the current levels. */
  snapshot(): Float32Array {
    return this.out.slice();
  }

  reset(): void {
    this.ring.fill(0);
    this.power.fill(0);
    this.out.fill(SPECTRUM_FLOOR_DB);
    this.pos = 0;
    this.filled = 0;
    this.sinceFrame = 0;
    this.frames = 0;
  }

  private frame(): void {
    const n = this.fftSize;
    let r = this.pos;
    for (let i = 0; i < n; i++) {
      this.re[i] = this.ring[r] * this.window[i];
      this.im[i] = 0;
      r = r + 1 === n ? 0 : r + 1;
    }
    this.fft.forward(this.re, this.im);
    const a = this.frames === 0 ? 1 : this.alpha;
    const half = n / 2;
    for (let k = 0; k <= half; k++) {
      // DC and Nyquist have no mirrored half: their mean square is |X|²/(Σw)².
      const scale = k === 0 || k === half ? this.norm / 2 : this.norm;
      const p = (this.re[k] * this.re[k] + this.im[k] * this.im[k]) * scale;
      const avg = this.power[k] + a * (p - this.power[k]);
      this.power[k] = avg;
      this.out[k] = avg > 0 ? Math.max(SPECTRUM_FLOOR_DB, 10 * Math.log10(avg)) : SPECTRUM_FLOOR_DB;
    }
    this.frames++;
  }
}

/** Spectrum levels in dB per bin (bin k = k·fs/fftSize), as SpectrumAccumulator.db. */
type DbBins = ArrayLike<number>;

/** Power sum (dB) of bins k − half … k + half; −Infinity when nothing is there. */
function binsDb(db: DbBins, k: number, half: number): number {
  let p = 0;
  for (let i = Math.max(0, k - half); i <= Math.min(db.length - 1, k + half); i++) {
    const v = db[i];
    if (Number.isFinite(v)) p += Math.pow(10, v / 10);
  }
  return p > 0 ? 10 * Math.log10(p) : -Infinity;
}

/**
 * Measured attenuation at each harmonic h·f0 (h = 1 … harmonics; index h − 1) from the controller's ANC on /
 * off spectra (SpectraSnapshot): A_h = [L_err − L_rif]_off − [L_err − L_rif]_on, dB, positive = quieter with
 * the anti-noise, each level the power of the Hann main lobe around the harmonic (± 2 bins, fewer when the
 * harmonics are closer than that). null for a harmonic at or above 0.45·fs or when a spectrum is missing.
 */
export function harmonicAttenuationDb(
  s: { fs: number; fftSize: number; err: DbBins | null; errBypass: DbBins | null; refOn: DbBins | null; refOff: DbBins | null },
  f0: number,
  harmonics: number,
): (number | null)[] {
  const n = Math.max(0, Math.min(64, Math.floor(Number.isFinite(harmonics) ? harmonics : 0)));
  const ok = s.err !== null && s.errBypass !== null && s.refOn !== null && s.refOff !== null && s.fs > 0 && s.fftSize > 0 && f0 > 0;
  const binHz = s.fs / s.fftSize;
  const half = Math.max(0, Math.min(2, Math.floor((f0 / binHz - 1) / 2)));
  return Array.from({ length: n }, (_, i) => {
    const f = (i + 1) * f0;
    if (!ok || f >= 0.45 * s.fs) return null;
    const k = Math.round(f / binHz);
    const v = [s.errBypass!, s.refOff!, s.err!, s.refOn!].map((a) => binsDb(a, k, half));
    if (!v.every(Number.isFinite)) return null;
    return v[0] - v[1] - (v[2] - v[3]);
  });
}
