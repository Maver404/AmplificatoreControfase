/**
 * Acoustic plant for the closed-loop simulator: a noise source, the reference and error mics and two
 * speakers, connected by delay/gain paths, around the real AntiPhaseEngine (see harness.ts).
 *
 *   ref[n] = s[n] + Σ_ch (F_ch * y_ch)[n] + sensor noise
 *   err[n] = (P * s)[n] + Σ_ch (S_ch * y_ch)[n] + z[n] + sensor noise
 *
 * s is the noise as it reaches the reference mic, y_ch the engine's final outputs, z an optional noise
 * at the zone that the reference mic does not hear (coherence < 1). Every path is a sum of
 * (fractional) delays with gains, optionally a diffuse exponentially decaying tail, followed by
 * 1st-order low/high-passes and a polarity inversion. S_ch and F_ch include the electro-acoustic
 * latency L_e, which must be at least one block (plus the interpolation reach) so that a block of mic
 * samples only depends on earlier output blocks.
 */
import { Biquad, highpass, lowpass, onePoleHighpass, onePoleLowpass } from '../../src/dsp/biquad.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

export interface Arrival {
  delayS: number;
  gain: number;
}

export interface PathSpec {
  /** Main arrival in seconds (for S and F: the acoustic part only, L_e is added by the plant). */
  delayS: number;
  gain: number;
  /** Further arrivals: delays relative to the main arrival, gains absolute (not relative to `gain`). */
  reflections?: Arrival[];
  /** 1st-order low-pass on the path (e.g. a wall/window on P), Hz. */
  lowpassHz?: number;
  /** 1st-order high-pass on the path (e.g. a small speaker's roll-off), Hz. */
  highpassHz?: number;
  /** Polarity inversion of the whole path (e.g. a speaker wired backwards). */
  invert?: boolean;
  /**
   * Diffuse reverberation after the main arrival: random taps from +1 ms every `spacingMs` (default
   * 0.25), RMS `gain` at the start, decaying by 60 dB over `rt60S`, cut at −40 dB.
   */
  tail?: { rt60S: number; gain: number; spacingMs?: number; seed?: number };
}

export interface NoiseSpec {
  /** Periodic noise: fundamental and peak amplitudes of harmonics 1, 2, … at the reference mic. */
  tone?: { f0: number; amplitudes: number[] };
  /** Gaussian noise band-passed (2nd-order Butterworth) to [lo, hi] with this RMS at the reference mic. */
  broadband?: { rms: number; lo?: number; hi?: number };
  /** Time-varying gain (t in seconds) on the broadband part: steps, bursts, speech-like modulation. */
  broadbandGain?: (t: number) => number;
  /** Time-varying gain (t in seconds) on harmonic h (1, 2, …) of the tone. */
  toneGain?: (t: number, h: number) => number;
  /** Any further source sample at time t (seconds), added at the reference mic. */
  custom?: (t: number) => number;
}

export interface PlantConfig {
  fs: number;
  noise: NoiseSpec;
  /** Electro-acoustic latency L_e (output sample → DAC → … → ADC → input sample), seconds. */
  latencyS: number;
  /** Reference mic → zone (the head start of the noise). */
  primary: PathSpec;
  /** Speaker → error mic (acoustic part). */
  secondary: [PathSpec | null, PathSpec | null];
  /** Speaker → reference mic (acoustic part). */
  feedback: [PathSpec | null, PathSpec | null];
  /** White noise of each mic, RMS. */
  sensorNoiseRms?: number;
  /** Noise at the zone only (local sources, wind on the error mic): Gaussian band-passed to [lo, hi]. */
  zoneNoise?: { rms: number; lo?: number; hi?: number };
  seed?: number;
}

/** Windowed-sinc fractional delay: taps j ∈ [−HALF + 1, HALF]. */
const HALF = 8;

/** A path as a sparse FIR (integer lags ≥ 0) plus optional filters. */
class PathFilter {
  readonly lags: Int32Array;
  readonly coefs: Float64Array;
  readonly minLag: number;
  readonly maxLag: number;
  private readonly filters: Biquad[] = [];

  constructor(spec: PathSpec, fs: number, extraDelayS: number) {
    const taps = new Map<number, number>();
    const sign = spec.invert === true ? -1 : 1;
    const add = (delayS: number, gain: number): void => {
      const d = delayS * fs;
      const di = Math.floor(d);
      const mu = d - di;
      if (mu < 1e-9) {
        taps.set(di, (taps.get(di) ?? 0) + sign * gain);
        return;
      }
      const k: number[] = [];
      let sum = 0;
      for (let j = -HALF + 1; j <= HALF; j++) {
        const u = j - mu;
        const s = Math.sin(Math.PI * u) / (Math.PI * u);
        const w = 0.42 + 0.5 * Math.cos((Math.PI * u) / HALF) + 0.08 * Math.cos((2 * Math.PI * u) / HALF);
        k.push(s * w);
        sum += s * w;
      }
      for (let j = -HALF + 1, i = 0; j <= HALF; j++, i++) {
        // x(t − d) ≈ Σ_j x[t − di − j]·sinc(j − μ)·w(j − μ), normalised to unit DC gain.
        taps.set(di + j, (taps.get(di + j) ?? 0) + (sign * gain * k[i]) / sum);
      }
    };
    const main = extraDelayS + spec.delayS;
    add(main, spec.gain);
    for (const r of spec.reflections ?? []) add(main + r.delayS, r.gain);
    const tail = spec.tail;
    if (tail !== undefined && tail.rt60S > 0 && tail.gain > 0) {
      const g = gaussian(tail.seed ?? 77);
      const step = Math.max(1, Math.round(((tail.spacingMs ?? 0.25) / 1000) * fs));
      const start = Math.round((main + 0.001) * fs);
      const end = start + Math.round(((tail.rt60S * 40) / 60) * fs);
      for (let lag = start; lag < end; lag += step) {
        const t = (lag - start) / fs;
        taps.set(lag, (taps.get(lag) ?? 0) + sign * tail.gain * g() * Math.pow(10, (-3 * t) / tail.rt60S));
      }
    }
    const lags = [...taps.keys()].sort((a, b) => a - b);
    this.lags = Int32Array.from(lags);
    this.coefs = Float64Array.from(lags, (l) => taps.get(l) ?? 0);
    this.minLag = lags[0];
    this.maxLag = lags[lags.length - 1];
    if (this.minLag < 0) throw new RangeError('PathFilter: negative lag');
    if (spec.lowpassHz !== undefined) this.filters.push(new Biquad(onePoleLowpass(fs, spec.lowpassHz)));
    if (spec.highpassHz !== undefined) this.filters.push(new Biquad(onePoleHighpass(fs, spec.highpassHz)));
  }

  /** Output at absolute time t from a ring history (index & mask). */
  at(hist: Float64Array, mask: number, t: number): number {
    let acc = 0;
    for (let i = 0; i < this.lags.length; i++) acc += this.coefs[i] * hist[(t - this.lags[i]) & mask];
    for (const f of this.filters) acc = f.process(acc);
    return acc;
  }
}

/** Standard normal samples from a seeded uniform generator (Box–Muller). */
function gaussian(seed: number): () => number {
  const u = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let a = u();
    while (a <= 1e-12) a = u();
    const b = u();
    const r = Math.sqrt(-2 * Math.log(a));
    spare = r * Math.sin(2 * Math.PI * b);
    return r * Math.cos(2 * Math.PI * b);
  };
}

/** Band-passed (2nd-order Butterworth) Gaussian noise of a given RMS. */
class BandNoise {
  private readonly g: () => number;
  private readonly bp: Biquad[];
  private readonly scale: number;

  constructor(fs: number, rms: number, lo: number, hi: number, seed: number) {
    const make = (): Biquad[] => [new Biquad(highpass(fs, lo, Math.SQRT1_2)), new Biquad(lowpass(fs, Math.min(hi, 0.45 * fs), Math.SQRT1_2))];
    const pilot = make();
    const pg = gaussian(seed + 7919);
    let e = 0;
    const len = Math.round(2 * fs);
    for (let i = 0; i < len; i++) {
      let v = pg();
      for (const f of pilot) v = f.process(v);
      e += v * v;
    }
    this.scale = rms / Math.sqrt(e / len);
    this.bp = make();
    this.g = gaussian(seed);
  }

  next(): number {
    let v = this.g();
    for (const f of this.bp) v = f.process(v);
    return this.scale * v;
  }
}

class NoiseSource {
  private readonly tone: { w: number[]; a: number[]; ph: number[] } | null = null;
  private readonly g: (() => number) | null = null;
  private readonly bp: Biquad[] = [];
  private readonly scale: number = 0;
  private readonly fs: number;
  private readonly bGain: ((t: number) => number) | null;
  private readonly tGain: ((t: number, h: number) => number) | null;
  private readonly custom: ((t: number) => number) | null;
  private n = 0;

  constructor(spec: NoiseSpec, fs: number, seed: number) {
    this.fs = fs;
    this.bGain = spec.broadbandGain ?? null;
    this.tGain = spec.toneGain ?? null;
    this.custom = spec.custom ?? null;
    const rnd = mulberry32(seed ^ 0x5eed);
    if (spec.tone) {
      const { f0, amplitudes } = spec.tone;
      this.tone = {
        w: amplitudes.map((_, h) => (2 * Math.PI * (h + 1) * f0) / fs),
        a: [...amplitudes],
        ph: amplitudes.map(() => 2 * Math.PI * rnd()),
      };
    }
    const b = spec.broadband;
    if (b && b.rms > 0) {
      const lo = b.lo ?? 20;
      const hi = Math.min(b.hi ?? 2000, 0.45 * fs);
      const make = (): Biquad[] => [new Biquad(highpass(fs, lo, Math.SQRT1_2)), new Biquad(lowpass(fs, hi, Math.SQRT1_2))];
      // RMS of the filtered unit noise, from a pilot run with an unrelated seed.
      const pilot = make();
      const pg = gaussian(seed + 7919);
      let e = 0;
      const len = Math.round(2 * fs);
      for (let i = 0; i < len; i++) {
        let v = pg();
        for (const f of pilot) v = f.process(v);
        e += v * v;
      }
      this.scale = b.rms / Math.sqrt(e / len);
      this.bp = make();
      this.g = gaussian(seed + 104729);
    }
  }

  next(): number {
    let v = 0;
    const sec = this.n / this.fs;
    const t = this.tone;
    if (t !== null) {
      for (let h = 0; h < t.a.length; h++) {
        const a = this.tGain !== null ? t.a[h] * this.tGain(sec, h + 1) : t.a[h];
        v += a * Math.sin(t.w[h] * this.n + t.ph[h]);
      }
    }
    if (this.g !== null) {
      let b = this.g();
      for (const f of this.bp) b = f.process(b);
      v += (this.bGain !== null ? this.bGain(sec) : 1) * this.scale * b;
    }
    if (this.custom !== null) v += this.custom(sec);
    this.n++;
    return v;
  }
}

export class Plant {
  readonly fs: number;
  readonly block: number;
  /** Samples simulated so far. */
  private t = 0;
  private readonly mask: number;
  private readonly sHist: Float64Array;
  private readonly yHist: [Float64Array, Float64Array];
  private readonly p: PathFilter;
  private readonly s: [PathFilter | null, PathFilter | null];
  private readonly f: [PathFilter | null, PathFilter | null];
  private readonly noise: NoiseSource;
  private readonly zone: BandNoise | null;
  private readonly sensor: [() => number, () => number];
  private readonly sensorRms: number;
  private readonly ref: Float32Array;
  private readonly err: Float32Array;
  /** Longest output → mic lag (for settling after the outputs change). */
  readonly maxOutputLagS: number;
  /** Multiplies each speaker → reference path (F_ch); e.g. 0 until the loop is closed. Default 1. */
  readonly feedbackScale: [number, number] = [1, 1];

  constructor(cfg: PlantConfig, block = 128) {
    const fs = cfg.fs;
    this.fs = fs;
    this.block = block;
    const seed = cfg.seed ?? 1;
    this.p = new PathFilter(cfg.primary, fs, 0);
    const mk = (spec: PathSpec | null): PathFilter | null => (spec === null ? null : new PathFilter(spec, fs, cfg.latencyS));
    this.s = [mk(cfg.secondary[0]), mk(cfg.secondary[1])];
    this.f = [mk(cfg.feedback[0]), mk(cfg.feedback[1])];
    let maxLag = this.p.maxLag;
    let maxOut = 0;
    for (const pf of [...this.s, ...this.f]) {
      if (pf === null) continue;
      if (pf.minLag < block) throw new RangeError('Plant: L_e must exceed one block plus the interpolation reach');
      maxLag = Math.max(maxLag, pf.maxLag);
      maxOut = Math.max(maxOut, pf.maxLag);
    }
    this.maxOutputLagS = maxOut / fs;
    let size = 1;
    while (size < maxLag + 2 * block + 2) size <<= 1;
    this.mask = size - 1;
    this.sHist = new Float64Array(size);
    this.yHist = [new Float64Array(size), new Float64Array(size)];
    this.noise = new NoiseSource(cfg.noise, fs, seed);
    const z = cfg.zoneNoise;
    this.zone = z !== undefined && z.rms > 0 ? new BandNoise(fs, z.rms, z.lo ?? 20, z.hi ?? 2000, seed * 31 + 3) : null;
    this.sensor = [gaussian(seed * 31 + 1), gaussian(seed * 31 + 2)];
    this.sensorRms = cfg.sensorNoiseRms ?? 0;
    this.ref = new Float32Array(block);
    this.err = new Float32Array(block);
  }

  /** Mic signals of the next block (depend on outputs of earlier blocks only). Arrays are reused. */
  next(): { ref: Float32Array; err: Float32Array } {
    const m = this.mask;
    for (let i = 0; i < this.block; i++) {
      const t = this.t + i;
      const s = this.noise.next();
      this.sHist[t & m] = s;
      let r = s;
      let e = this.p.at(this.sHist, m, t);
      for (const ch of [0, 1] as const) {
        const f = this.f[ch];
        if (f !== null) r += this.feedbackScale[ch] * f.at(this.yHist[ch], m, t);
        const sp = this.s[ch];
        if (sp !== null) e += sp.at(this.yHist[ch], m, t);
      }
      if (this.zone !== null) e += this.zone.next();
      if (this.sensorRms > 0) {
        r += this.sensorRms * this.sensor[0]();
        e += this.sensorRms * this.sensor[1]();
      }
      this.ref[i] = r;
      this.err[i] = e;
    }
    return { ref: this.ref, err: this.err };
  }

  /** The engine outputs for the block returned by the last next(). */
  push(outL: Float32Array, outR: Float32Array): void {
    const m = this.mask;
    for (let i = 0; i < this.block; i++) {
      this.yHist[0][(this.t + i) & m] = outL[i];
      this.yHist[1][(this.t + i) & m] = outR[i];
    }
    this.t += this.block;
  }

  get time(): number {
    return this.t;
  }
}
