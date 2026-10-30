/**
 * Howl (feedback runaway) detector for one output channel, run inside the AudioWorklet on the
 * pre-limiter signal. Everything is defined in Hz and ms so it behaves the same at any sample rate.
 * push() allocates nothing: the FFTs, spectra, scratch buffers and track histories are preallocated.
 *
 * A howl with a short loop (a few ms) is a narrow spectral peak. With a realistic acoustic loop
 * (≥ ~12 ms) it is a dense comb of close teeth with a high crest factor, which no narrow-peak test sees;
 * but whatever its spectrum, a howl with loop delay L repeats itself every L (the speaker plays back
 * what the mic picked up L earlier), so its waveform is periodic, and while it grows its total level
 * rises faster than the commanded gain explains. The rules below use all three signatures.
 */
import type { HowlReason } from '../audio/protocol.ts';
import { Biquad, lowpass } from './biquad.ts';
import { FFT } from './fft.ts';

export interface HowlState {
  triggered: boolean;
  reason: HowlReason | null;
  freqHz: number | null;
}

export const HOWL_TARGET_RATE = 12000;
export const HOWL_FFT_SIZE = 512;
/** A narrow peak stands at least this far above both the median spectrum and its neighbourhood. */
export const HOWL_PEAK_DB = 12;
export const HOWL_GROWTH_DB_PER_S = 10;
export const HOWL_GROWTH_MS = 300;
export const HOWL_MIN_LEVEL_DBFS = -40;
export const HOWL_LIMITER_GR_DB = 3;
export const HOWL_LIMITER_MS = 300;
export const HOWL_OVERLOAD_MARGIN_DB = 4;
export const HOWL_OVERLOAD_MS = 200;
/** Loop delays the main periodicity analysis looks for (~6 kHz, band ≤ ~2.4 kHz). */
export const HOWL_LOOP_MIN_MS = 5;
export const HOWL_LOOP_MAX_MS = 600;
/**
 * Longest loop delay looked for, by a second, slower periodicity analysis (~3 kHz, band ≤ ~1.2 kHz, lags
 * 500 ms…HOWL_LONG_LOOP_MAX_MS): the 500 ms channel delay plus up to 600 ms of device latency and
 * acoustic delay. Long loops grow slowly (20·log10(G) dB per loop), so they reach the ceiling.
 */
export const HOWL_LONG_LOOP_MAX_MS = 1100;
/** A signal is periodic when its normalized autocorrelation |ρ| reaches this at some loop delay. */
export const HOWL_PERIODIC_RHO = 0.8;
/**
 * Broadband growth also needs this much periodicity (broadband noise stays ≲ 0.2 over the 0.77 s window;
 * a growing howl reaches it while it is still ~30 dB below the ceiling), and a periodicity that is not
 * falling (by more than HOWL_GROWTH_RHO_DROP since the previous analysis): what grows must be the
 * periodic part. So a smoothly rising noise source (a passing vehicle) is not taken for a howl, even
 * with a steady tone in the background (that tone's periodicity fades as the noise rises).
 */
export const HOWL_GROWTH_RHO = 0.3;
export const HOWL_GROWTH_RHO_DROP = 0.01;
/** A new periodicity analysis starts this often (each one takes ~60 ms of spread-out work). */
export const HOWL_PERIODIC_EVERY_MS = 200;
/** An analysis counts for this long after the end of the window it analysed. */
export const HOWL_PERIODIC_VALID_MS = 600;
/** Saturated periodic signal: limiter attacks never more than this far apart, … */
export const HOWL_SAT_GAP_MS = 200;
/** … for this long, … */
export const HOWL_SAT_MS = 300;
/** … with the pre-limiter RMS (100 ms) within this many dB of the ceiling … */
export const HOWL_SAT_LEVEL_DB = 12;
/**
 * … and the limiter taking off at least this much (GR smoothed over 50 ms). A howl held at the ceiling
 * needs 20·log10(G) dB of GR (0.8 dB at loop gain 1.1); a steady hum or tone whose peaks barely touch the
 * ceiling needs almost none, and plays (the former backstop allowed 0.5 dB too).
 */
export const HOWL_SAT_GR_DB = 0.5;

/** 50 % overlap: a new spectrum every ~21 ms. */
const HOP = HOWL_FFT_SIZE / 2;
/** The growth slope must also be this many standard errors above zero (rejects noisy level tracks). */
const GROWTH_MIN_T = 6;
const MIN_GROWTH_POINTS = 6;
/** Short-term slope (last 5 frames, ~100 ms) above the growth rate starts or continues a growth run. */
const SHORT_SLOPE_FRAMES = 5;
const MAX_TRACKS = 4;
/** Index of the track that follows the total (broadband) level of the analysis band. */
const BB = MAX_TRACKS;
/**
 * The band level is averaged (power, over the last 5 frames: ~110 ms) before its growth is tested: a
 * growing long-loop howl repeats the envelope of the noise it started from, a ±3 dB ripple with the loop
 * period that would break the 100 ms growth runs. A level step still rises in < 110 ms, but any rise
 * lasts 4 hops (~85 ms) longer in the average: the band level's growth run must be that much longer.
 */
const BB_AVG_FRAMES = 5;
/**
 * A howl grows exponentially, at a steady rate in dB; the averaged band level of a source switching on
 * (or fading in) rises fast and then decelerates. The band level's latest short-term slope must still
 * be this fraction of the slope over the whole run.
 */
const BB_STEADY_RATIO = 0.4;
/** The periodicity that gates a band-level growth run must come from a window ending this far into the run. */
const BB_RUN_PERIODIC_MS = 100;
/** Frames kept per track (~2.7 s at 12 kHz): the growth window plus history for the new-level test. */
const TRACK_CAP = 128;
/** The growing level must exceed the track's level before the growth run by this much. */
const NEW_LEVEL_DB = 3;
/** Peaks within this many bins of a track continue it. */
const TRACK_MATCH_BINS = 1.5;
/** A peak is "present" for the limiter/overload rules once its track has this many frames. */
const PERSISTENT_FRAMES = 3;
/** Frames whose commanded gain changes more than this ratio (3 dB) cannot be compensated reliably. */
const MAX_FRAME_GAIN_RATIO = 1.4125;
const MIN_FRAME_GAIN = 1e-4;
const RMS_TAU_MS = 25;
const SAT_RMS_TAU_MS = 100;
const SAT_GR_TAU_MS = 50;
/** Neighbourhood for the narrowness test: bins [k−16, k−3] and [k+3, k+16] (outside the Hann main lobe). */
const LOCAL_INNER = 3;
const LOCAL_OUTER = 16;
/** Anti-alias low-pass: 6th-order Butterworth at 0.4·decimated rate; peaks are searched below it. */
const AA_FRACTION = 0.4;
const AA_Q = [0.5176380902, 0.7071067812, 1.9318516526];
/** Periodicity runs at half the decimated rate (~6 kHz, band ≤ ~2.4 kHz) behind a 4th-order Butterworth. */
const PER_AA_Q = [0.5411961001, 1.3065629649];
const PER_MIN_RATE = 9000;
/** Each periodicity analysis completes within about this time (its work is spread evenly over it). */
const PER_SPREAD_MS = 60;
/** The long-loop analysis: lags from here (overlapping the main one) to HOWL_LONG_LOOP_MAX_MS, … */
const LONG_LOOP_MIN_MS = 500;
/** … a new one every 400 ms, spread over 120 ms (half the main analysis' load per sample). */
const LONG_EVERY_MS = 400;
const LONG_SPREAD_MS = 120;

const quickMedian = (a: Float64Array, n: number): number => {
  let lo = 0;
  let hi = n - 1;
  const k = n >> 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k];
};

/** Default analysis size: 8192-point FFT. */
const PER_LOG2 = 13;
const P_IDLE = 0;
const P_COPY = 1;
const P_FWD = 2;
const P_SPLIT = 3;
const P_INV = 4;
const P_SEARCH = 5;

/** Size, lag range and pace of a PeriodicityAnalyzer (defaults: the HowlDetector's main analysis). */
export interface PeriodicityOptions {
  /** log2 of the FFT size (default 13: 8192 points). */
  log2N?: number;
  /** Lags searched, ms (default HOWL_LOOP_MIN_MS…HOWL_LOOP_MAX_MS; at most half the FFT size). */
  lagMinMs?: number;
  lagMaxMs?: number;
  /** Each analysis completes within about this time (default 60 ms). */
  spreadMs?: number;
}

/**
 * Periodicity of a signal: the largest normalized autocorrelation |ρ(τ)| = |Σ x[n]·x[n−τ]| /
 * √(Σ x[n]²·Σ x[n−τ]²) at a local maximum over lags HOWL_LOOP_MIN_MS…HOWL_LOOP_MAX_MS (by default), the
 * sums running over the latest window W (FFT size minus the longest lag: 8192 points, ~0.77 s at 6 kHz).
 * A howl with loop delay L gives |ρ(L)| ≈ 0.9–1 (the sign follows the loop polarity), broadband noise
 * ≲ 0.2, tones ≈ 1. A new analysis starts every `everyMs`; its FFT work (one complex FFT for both real
 * sequences, the cross-spectrum, one inverse) is cut into bounded slices, one per push(), so that it
 * completes in `spreadMs` and no single render quantum carries more than a slice per sample.
 */
export class PeriodicityAnalyzer {
  readonly rate: number;
  /** Samples between the starts of two analyses. */
  readonly everySamples: number;
  /** Lags searched, samples. */
  readonly lagMinSamples: number;
  readonly lagMaxSamples: number;
  /** Window W of the sums, samples (the FFT size minus the longest lag). */
  readonly windowSamples: number;
  /** Work units (copied samples, butterflies, 2 per spectrum pair, lags) done per push() while analysing. */
  readonly budget: number;
  private readonly log2N: number;
  private readonly n: number;
  private readonly mask: number;
  private readonly ring: Float64Array;
  private pos = 0;
  private count = 0;
  private since = 0;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly cum: Float64Array;
  private readonly cosT: Float64Array;
  private readonly sinT: Float64Array;
  private readonly rev: Uint32Array;

  private phase = P_IDLE;
  private cursor = 0;
  private stage = 0;
  private snapPos = 0;
  private snapCount = 0;
  /** Samples pushed since the analysed window ended (while an analysis runs). */
  private running = 0;
  private best = 0;
  private bestLag = 0;
  private r1 = 0;
  private r2 = 0;

  private rhoOut = 0;
  private rhoPrev = 0;
  private lagOut = 0;
  /** Samples since the end of the window of the latest completed analysis. */
  private age = Infinity;

  constructor(rate: number, everyMs = HOWL_PERIODIC_EVERY_MS, o: PeriodicityOptions = {}) {
    if (!(rate > 0)) throw new RangeError('PeriodicityAnalyzer: rate must be positive');
    const log2N = Math.round(o.log2N ?? PER_LOG2);
    if (!(log2N >= 6 && log2N <= 16)) throw new RangeError('PeriodicityAnalyzer: log2N must be 6…16');
    const N = 1 << log2N;
    this.log2N = log2N;
    this.n = N;
    this.mask = N - 1;
    this.ring = new Float64Array(N);
    this.re = new Float64Array(N);
    this.im = new Float64Array(N);
    this.cum = new Float64Array(N + 1);
    this.cosT = new Float64Array(N / 2);
    this.sinT = new Float64Array(N / 2);
    this.rev = new Uint32Array(N);
    this.rate = rate;
    this.everySamples = Math.max(1, Math.round((everyMs / 1000) * rate));
    this.lagMinSamples = Math.max(2, Math.round(((o.lagMinMs ?? HOWL_LOOP_MIN_MS) / 1000) * rate));
    this.lagMaxSamples = Math.max(this.lagMinSamples + 2, Math.min(N >> 1, Math.round(((o.lagMaxMs ?? HOWL_LOOP_MAX_MS) / 1000) * rate)));
    this.windowSamples = N - this.lagMaxSamples - 2;
    for (let k = 0; k < N / 2; k++) {
      const a = (2 * Math.PI * k) / N;
      this.cosT[k] = Math.cos(a);
      this.sinT[k] = Math.sin(a);
    }
    for (let i = 1; i < N; i++) this.rev[i] = (this.rev[i >>> 1] >>> 1) | ((i & 1) << (log2N - 1));
    // Work units: one per copied sample, butterfly and searched lag, two per spectrum pair.
    const units = N + 2 * log2N * (N / 2) + 2 * (N / 2 + 1) + (this.lagMaxSamples - this.lagMinSamples + 3);
    this.budget = Math.ceil(units / Math.max(1, ((o.spreadMs ?? PER_SPREAD_MS) / 1000) * rate));
  }

  /** |ρ| of the latest completed analysis (0 before the first one). */
  get rho(): number {
    return this.rhoOut;
  }

  /** |ρ| of the analysis before it (0 if none). */
  get previousRho(): number {
    return this.rhoPrev;
  }

  /** Lag of that maximum, ms. */
  get lagMs(): number {
    return (this.lagOut * 1000) / this.rate;
  }

  /** Time since the end of the window of the latest completed analysis, ms (Infinity before it). */
  get ageMs(): number {
    return (this.age * 1000) / this.rate;
  }

  reset(): void {
    this.pos = 0;
    this.count = 0;
    this.since = 0;
    this.phase = P_IDLE;
    this.rhoOut = 0;
    this.rhoPrev = 0;
    this.lagOut = 0;
    this.age = Infinity;
  }

  push(x: number): void {
    // Work first: COPY reads the oldest samples before this write can overwrite them (≥ 1 per push).
    if (this.phase !== P_IDLE) {
      this.running++;
      this.work(this.budget);
    }
    this.ring[this.pos] = x;
    this.pos = (this.pos + 1) & this.mask;
    if (this.count < this.n) this.count++;
    this.age++;
    if (++this.since >= this.everySamples && this.phase === P_IDLE) {
      this.since = 0;
      // The window ends here; the history is copied oldest first.
      this.snapPos = this.pos;
      this.snapCount = this.count;
      this.running = 0;
      this.phase = P_COPY;
      this.cursor = 0;
    }
  }

  private work(budget: number): void {
    const N = this.n;
    const LOG2 = this.log2N;
    const MASK = this.mask;
    const re = this.re;
    const im = this.im;
    let left = budget;
    while (left > 0) {
      switch (this.phase) {
        case P_COPY: {
          const end = Math.min(N, this.cursor + left);
          const firstValid = N - this.snapCount;
          const winStart = N - this.windowSamples;
          const cum = this.cum;
          for (let i = this.cursor; i < end; i++) {
            const b = i >= firstValid ? this.ring[(this.snapPos + i) & MASK] : 0;
            im[i] = b;
            re[i] = i >= winStart ? b : 0;
            cum[i + 1] = cum[i] + b * b;
          }
          left -= end - this.cursor;
          this.cursor = end;
          if (end === N) {
            this.phase = P_FWD;
            this.stage = 0;
            this.cursor = 0;
          }
          break;
        }
        case P_FWD: {
          // Decimation in frequency: natural order in, bit-reversed order out, w = e^{−j2πk/N}.
          const s = this.stage;
          const hb = LOG2 - 1 - s;
          const half = 1 << hb;
          const end = Math.min(N >> 1, this.cursor + left);
          for (let q = this.cursor; q < end; q++) {
            const j = q & (half - 1);
            const a = ((q >> hb) << (hb + 1)) + j;
            const b = a + half;
            const k = j << s;
            const c = this.cosT[k];
            const sn = this.sinT[k];
            const ur = re[a];
            const ui = im[a];
            const vr = re[b];
            const vi = im[b];
            re[a] = ur + vr;
            im[a] = ui + vi;
            const dr = ur - vr;
            const di = ui - vi;
            re[b] = dr * c + di * sn;
            im[b] = di * c - dr * sn;
          }
          left -= end - this.cursor;
          this.cursor = end;
          if (end === N >> 1) {
            this.cursor = 0;
            if (++this.stage === LOG2) this.phase = P_SPLIT;
          }
          break;
        }
        case P_SPLIT: {
          // Z = FFT(a + j·b) with a, b real: A = (Z[k] + Z*[N−k])/2, B = (Z[k] − Z*[N−k])/2j.
          // C = A·B* (the cross-spectrum, whose inverse is Σ a[n]·b[n−τ]); C[N−k] = C*[k].
          const end = Math.min((N >> 1) + 1, this.cursor + (left >> 1) + 1);
          for (let k = this.cursor; k < end; k++) {
            const ik = this.rev[k];
            const inn = this.rev[(N - k) & MASK];
            const zr = re[ik];
            const zi = im[ik];
            const nr = re[inn];
            const ni = im[inn];
            const ar = 0.5 * (zr + nr);
            const ai = 0.5 * (zi - ni);
            const br = 0.5 * (zi + ni);
            const bi = -0.5 * (zr - nr);
            const cr = ar * br + ai * bi;
            const ci = ai * br - ar * bi;
            re[ik] = cr;
            im[ik] = ci;
            if (inn !== ik) {
              re[inn] = cr;
              im[inn] = -ci;
            }
          }
          left -= 2 * (end - this.cursor);
          this.cursor = end;
          if (end === (N >> 1) + 1) {
            this.phase = P_INV;
            this.stage = 0;
            this.cursor = 0;
          }
          break;
        }
        case P_INV: {
          // Decimation in time: bit-reversed order in, natural order out, w = e^{+j2πk/N} (unscaled).
          const s = this.stage;
          const half = 1 << s;
          const end = Math.min(N >> 1, this.cursor + left);
          for (let q = this.cursor; q < end; q++) {
            const j = q & (half - 1);
            const a = ((q >> s) << (s + 1)) + j;
            const b = a + half;
            const k = j << (LOG2 - 1 - s);
            const c = this.cosT[k];
            const sn = this.sinT[k];
            const xr = re[b];
            const xi = im[b];
            const tr = xr * c - xi * sn;
            const ti = xr * sn + xi * c;
            re[b] = re[a] - tr;
            im[b] = im[a] - ti;
            re[a] += tr;
            im[a] += ti;
          }
          left -= end - this.cursor;
          this.cursor = end;
          if (end === N >> 1) {
            this.cursor = 0;
            if (++this.stage === LOG2) {
              this.phase = P_SEARCH;
              this.cursor = this.lagMinSamples - 1;
              this.best = 0;
              this.bestLag = 0;
              this.r1 = 0;
              this.r2 = 0;
            }
          }
          break;
        }
        case P_SEARCH: {
          const cum = this.cum;
          const W = this.windowSamples;
          const ea = cum[N] - cum[N - W];
          // Energies below this are rounding residue of the running sum (or silence): no correlation.
          const floor = 1e-9 * cum[N] + 1e-30;
          const last = this.lagMaxSamples + 1;
          const end = Math.min(last + 1, this.cursor + left);
          for (let tau = this.cursor; tau < end; tau++) {
            const eb = cum[N - tau] - cum[N - W - tau];
            let r = ea > floor && eb > floor ? re[tau] / (N * Math.sqrt(ea * eb)) : 0;
            if (r < 0) r = -r;
            // r1 = |ρ(τ−1)|: a local maximum inside [lagMin, lagMax].
            const lag = tau - 1;
            const inside = lag >= this.lagMinSamples && lag <= this.lagMaxSamples;
            if (inside && this.r1 >= this.r2 && this.r1 > r && this.r1 > this.best) {
              this.best = this.r1;
              this.bestLag = lag;
            }
            this.r2 = this.r1;
            this.r1 = r;
          }
          left -= end - this.cursor;
          this.cursor = end;
          if (end === last + 1) {
            this.rhoPrev = this.rhoOut;
            this.rhoOut = Math.min(1, this.best);
            this.lagOut = this.bestLag;
            this.age = this.running;
            this.phase = P_IDLE;
            return;
          }
          break;
        }
        default:
          return;
      }
    }
  }
}

/**
 * Runs on the pre-limiter output of one channel. Decimates to ~12 kHz (factor round(fs/12000)) with
 * an anti-alias low-pass, 512-point Hann FFT every 256 decimated samples, and a periodicity analysis
 * (PeriodicityAnalyzer) of the signal at half that rate every 200 ms. Triggers:
 *  - 'peak-growth': a level that grows > 10 dB/s faster than the commanded gain for ≥ 300 ms (a run of
 *    short-term slopes above 10 dB/s; least-squares slope over the run > 10 dB/s and ≥ 6 standard
 *    errors), reaching 3 dB above the track's earlier levels (so beating tones do not count), while
 *    above −40 dBFS. The level is either a narrow peak (≥ 12 dB above the median spectrum and its
 *    ±16-bin neighbourhood; freqHz = the peak) or, for a howl of any spectral shape, the total level of
 *    the band below ~4.8 kHz averaged over 5 frames (~110 ms), whose run must last 300 ms plus the 4 hops
 *    of that average (the average stretches a rise), must still be growing at ≥ 40 % of the run's
 *    slope (exponential, not a decelerating step) and needs, from a periodicity analysis of
 *    a window ending ≥ 100 ms into the run, |ρ| ≥ 0.3 not down by more than 0.01 since the analysis
 *    before (the growing part is the periodic one; freqHz = the strongest bin, or the narrow peak);
 *  - 'limiter-saturated': limiter GR > 3 dB for > 300 ms while a persistent narrow peak is present; or a
 *    periodic signal held at the ceiling: the latest periodicity analysis (window ending ≤ 600 ms ago)
 *    found |ρ| ≥ 0.8 at a lag of 5–600 ms (or the long-loop analysis at 500–1100 ms), the limiter has
 *    kept attacking (pre-limiter samples above the ceiling never more than 200 ms apart) for > 300 ms,
 *    the pre-limiter RMS (100 ms) is within 12 dB of the ceiling and the GR (50 ms) is ≥ 0.5 dB. A tone
 *    pushed into the limiter by more than that is periodic too, as before;
 *  - 'output-overload': RMS of the limited output (25 ms) > ceiling − 4 dB for > 200 ms. While a
 *    persistent narrow peak or a periodic signal is driving the limiter it waits (≤ 300 ms) so a howl
 *    is reported as such.
 * holdLevels() makes the level tracks restart after a change of the conditioning filters (their glide
 * changes the level at constant commanded gain). The state latches until reset().
 */
export class HowlDetector {
  private readonly decim: number;
  private readonly fsd: number;
  private readonly msPerSample: number;
  private readonly hopMs: number;
  private readonly aa: Biquad[];
  private readonly fft = new FFT(HOWL_FFT_SIZE);
  private readonly win = new Float64Array(HOWL_FFT_SIZE);
  private readonly winSum: number;
  private readonly levelOffsetDb: number;
  private readonly bandOffsetDb: number;
  private readonly kLo = 2;
  private readonly kHi: number;
  private readonly rmsCoef: number;
  private readonly satRmsCoef: number;
  private readonly satGrCoef: number;
  private readonly satGap: number;

  // Decimated ring of samples and of commanded linear gains.
  private readonly xr = new Float64Array(HOWL_FFT_SIZE);
  private readonly gr = new Float64Array(HOWL_FFT_SIZE);
  private wpos = 0;
  private phase = 0;
  private decCount = 0;
  private sinceFrame = 0;

  private readonly re = new Float64Array(HOWL_FFT_SIZE);
  private readonly im = new Float64Array(HOWL_FFT_SIZE);
  private readonly pw = new Float64Array(HOWL_FFT_SIZE / 2 + 1);
  private readonly scratch = new Float64Array(HOWL_FFT_SIZE / 2 + 1);

  // Candidate peaks of the current frame (strongest first).
  private readonly cBin = new Float64Array(MAX_TRACKS);
  private readonly cDb = new Float64Array(MAX_TRACKS);
  private nCand = 0;

  // Tracks: 0…MAX_TRACKS−1 narrow peaks, BB the band level.
  private readonly tActive = new Uint8Array(MAX_TRACKS + 1);
  private readonly tMatched = new Uint8Array(MAX_TRACKS + 1);
  private readonly tBin = new Float64Array(MAX_TRACKS + 1);
  private readonly tCount = new Int32Array(MAX_TRACKS + 1);
  private readonly tHead = new Int32Array(MAX_TRACKS + 1);
  private readonly tLastDb = new Float64Array(MAX_TRACKS + 1);
  /** Start time of the current growth run of each track (NaN when the track is not growing). */
  private readonly tGrowStart = new Float64Array(MAX_TRACKS + 1);
  /** Latest short-term slope of each track, dB/s. */
  private readonly tShort = new Float64Array(MAX_TRACKS + 1);
  private readonly tTime = new Float64Array((MAX_TRACKS + 1) * TRACK_CAP);
  private readonly tLevel = new Float64Array((MAX_TRACKS + 1) * TRACK_CAP);

  private peakPresent = false;
  private peakFreq = 0;
  /** Strongest bin of the latest frame, Hz. */
  private domFreq = 0;
  /** Band power divided by the commanded power gain, last BB_AVG_FRAMES frames. */
  private readonly bbPow = new Float64Array(BB_AVG_FRAMES);
  private bbN = 0;

  // Periodicity at ~6 kHz, and of long loops at ~3 kHz.
  private readonly perDec: number;
  private readonly perAa: Biquad[];
  private perPhase = 0;
  private readonly per: PeriodicityAnalyzer;
  private readonly longAa: Biquad[];
  private longPhase = 0;
  private readonly perLong: PeriodicityAnalyzer;
  /** Decimated samples whose commanded gain is recorded as unknown (holdLevels). */
  private holdLeft = 0;

  // Full-rate rules.
  private rmsPow = 0;
  private overMs = 0;
  private grMs = 0;
  private lastCeilingDb = Number.NaN;
  private overloadPow = 0;
  private ceilLin = 0;
  private satPow = 0;
  private preRmsPow = 0;
  /** Limiter GR smoothed over SAT_GR_TAU_MS, dB. */
  private satGrDb = 0;
  private sinceAttack = Infinity;
  private satMs = 0;
  /** periodic(), refreshed with every sample of the periodicity analysis. */
  private periodicNow = false;

  private readonly st: HowlState = { triggered: false, reason: null, freqHz: null };

  constructor(fs: number) {
    if (!(fs > 0)) throw new RangeError('HowlDetector: fs must be positive');
    this.decim = Math.max(1, Math.round(fs / HOWL_TARGET_RATE));
    this.fsd = fs / this.decim;
    this.msPerSample = 1000 / fs;
    this.hopMs = (HOP * 1000) / this.fsd;
    this.aa = this.decim > 1 ? AA_Q.map((q) => new Biquad(lowpass(fs, AA_FRACTION * this.fsd, q))) : [];
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < HOWL_FFT_SIZE; i++) {
      this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / HOWL_FFT_SIZE);
      s += this.win[i];
      s2 += this.win[i] * this.win[i];
    }
    this.winSum = s;
    // Sine amplitude A → |X| = A·Σw/2, so level (dB re full-scale sine) = 10·log10(|X|²) + this.
    this.levelOffsetDb = 20 * Math.log10(2 / s);
    // Band power: mean square = 2·Σ|X|²/(N·Σw²), in dB re a full-scale sine (mean square ½).
    this.bandOffsetDb = 10 * Math.log10(4 / (HOWL_FFT_SIZE * s2));
    this.kHi = Math.floor(AA_FRACTION * HOWL_FFT_SIZE) - 1;
    this.rmsCoef = 1 - Math.exp(-1 / ((RMS_TAU_MS / 1000) * fs));
    this.satRmsCoef = 1 - Math.exp(-1 / ((SAT_RMS_TAU_MS / 1000) * fs));
    this.satGrCoef = 1 - Math.exp(-1 / ((SAT_GR_TAU_MS / 1000) * fs));
    this.satGap = (HOWL_SAT_GAP_MS / 1000) * fs;
    this.perDec = this.fsd >= PER_MIN_RATE ? 2 : 1;
    const fp = this.fsd / this.perDec;
    this.perAa = this.perDec > 1 ? PER_AA_Q.map((q) => new Biquad(lowpass(this.fsd, AA_FRACTION * fp, q))) : [];
    this.per = new PeriodicityAnalyzer(fp);
    this.longAa = PER_AA_Q.map((q) => new Biquad(lowpass(fp, (AA_FRACTION * fp) / 2, q)));
    this.perLong = new PeriodicityAnalyzer(fp / 2, LONG_EVERY_MS, {
      lagMinMs: LONG_LOOP_MIN_MS,
      lagMaxMs: HOWL_LONG_LOOP_MAX_MS,
      spreadMs: LONG_SPREAD_MS,
    });
  }

  /** Live view of the detector state (the same object is updated in place). */
  get state(): HowlState {
    return this.st;
  }

  reset(): void {
    for (const b of this.aa) b.reset();
    for (const b of this.perAa) b.reset();
    for (const b of this.longAa) b.reset();
    this.xr.fill(0);
    this.gr.fill(0);
    this.wpos = 0;
    this.phase = 0;
    this.decCount = 0;
    this.sinceFrame = 0;
    this.nCand = 0;
    this.tActive.fill(0);
    this.tCount.fill(0);
    this.tGrowStart.fill(Number.NaN);
    this.peakPresent = false;
    this.peakFreq = 0;
    this.domFreq = 0;
    this.bbN = 0;
    this.perPhase = 0;
    this.per.reset();
    this.longPhase = 0;
    this.perLong.reset();
    this.holdLeft = 0;
    this.rmsPow = 0;
    this.overMs = 0;
    this.grMs = 0;
    this.preRmsPow = 0;
    this.satGrDb = 0;
    this.sinceAttack = Infinity;
    this.satMs = 0;
    this.periodicNow = false;
    this.st.triggered = false;
    this.st.reason = null;
    this.st.freqHz = null;
  }

  push(preLimiter: number, commandedGainDb: number, limiterGrDb: number, ceilingDb: number): void {
    if (this.st.triggered) return;
    const x = Number.isFinite(preLimiter) ? preLimiter : 0;
    const gr = Number.isFinite(limiterGrDb) && limiterGrDb > 0 ? limiterGrDb : 0;

    if (ceilingDb !== this.lastCeilingDb) {
      this.lastCeilingDb = ceilingDb;
      const c = Number.isFinite(ceilingDb) ? ceilingDb : 0;
      this.overloadPow = Math.pow(10, (c - HOWL_OVERLOAD_MARGIN_DB) / 10);
      this.ceilLin = Math.pow(10, c / 20);
      this.satPow = Math.pow(10, (c - HOWL_SAT_LEVEL_DB) / 10);
    }
    // Output level after the limiter, for the overload rule.
    const post = gr > 0 ? x * Math.pow(10, -gr / 20) : x;
    this.rmsPow += this.rmsCoef * (post * post - this.rmsPow);
    if (this.rmsPow < 1e-30) this.rmsPow = 0;
    this.overMs = this.rmsPow > this.overloadPow ? this.overMs + this.msPerSample : 0;
    this.grMs = gr > HOWL_LIMITER_GR_DB ? this.grMs + this.msPerSample : 0;
    // The limiter keeps attacking (the signal is held at the ceiling) and the level stays close to it.
    this.preRmsPow += this.satRmsCoef * (x * x - this.preRmsPow);
    if (this.preRmsPow < 1e-30) this.preRmsPow = 0;
    this.satGrDb += this.satGrCoef * (gr - this.satGrDb);
    if (this.satGrDb < 1e-12) this.satGrDb = 0;
    this.sinceAttack = (x < 0 ? -x : x) > this.ceilLin ? 0 : this.sinceAttack + 1;
    this.satMs = this.sinceAttack <= this.satGap ? this.satMs + this.msPerSample : 0;

    if (this.grMs > HOWL_LIMITER_MS && this.peakPresent) {
      this.trigger('limiter-saturated', this.peakFreq);
      return;
    }
    const periodicHeld = this.satMs > 0 && this.periodicNow;
    if (periodicHeld && this.satMs > HOWL_SAT_MS && this.preRmsPow >= this.satPow && this.satGrDb >= HOWL_SAT_GR_DB) {
      this.trigger('limiter-saturated', this.peakPresent ? this.peakFreq : this.domFreq > 0 ? this.domFreq : null);
      return;
    }
    if (this.overMs > HOWL_OVERLOAD_MS) {
      const howlPending = (this.peakPresent && this.grMs > 0) || periodicHeld;
      if (!howlPending) {
        this.trigger('output-overload', this.peakPresent ? this.peakFreq : null);
        return;
      }
    }

    // Anti-alias filter at the full rate, then keep every decim-th sample.
    let y = x;
    for (let i = 0; i < this.aa.length; i++) y = this.aa[i].process(y);
    if (++this.phase < this.decim) return;
    this.phase = 0;
    this.xr[this.wpos] = y;
    // A held sample records an unknown gain (0): the frames that contain it are unreliable.
    if (this.holdLeft > 0) {
      this.holdLeft--;
      this.gr[this.wpos] = 0;
    } else {
      this.gr[this.wpos] =
        Number.isFinite(commandedGainDb) && commandedGainDb > -120 ? Math.pow(10, commandedGainDb / 20) : 0;
    }
    this.wpos = (this.wpos + 1) & (HOWL_FFT_SIZE - 1);
    this.decCount++;
    let z = y;
    for (let i = 0; i < this.perAa.length; i++) z = this.perAa[i].process(z);
    if (++this.perPhase >= this.perDec) {
      this.perPhase = 0;
      this.per.push(z);
      let u = z;
      for (let i = 0; i < this.longAa.length; i++) u = this.longAa[i].process(u);
      if (++this.longPhase >= 2) {
        this.longPhase = 0;
        this.perLong.push(u);
      }
      this.periodicNow = this.periodic();
    }
    if (++this.sinceFrame >= HOP && this.decCount >= HOWL_FFT_SIZE) {
      this.sinceFrame = 0;
      this.analyzeFrame();
    }
  }

  /** |ρ| of the latest periodicity analysis, 0 once it is older than HOWL_PERIODIC_VALID_MS. */
  private recentRho(): number {
    return this.per.ageMs <= HOWL_PERIODIC_VALID_MS ? this.per.rho : 0;
  }

  /** The latest periodicity analysis (or the latest long-loop one) is recent and found a strongly periodic signal. */
  private periodic(): boolean {
    if (this.recentRho() >= HOWL_PERIODIC_RHO) return true;
    return this.perLong.ageMs <= HOWL_PERIODIC_VALID_MS && this.perLong.rho >= HOWL_PERIODIC_RHO;
  }

  /**
   * The conditioning filters are gliding to new coefficients for the next `ms` (a live filter change):
   * the level changes they cause at constant commanded gain are not a howl's. Samples within `ms` are
   * recorded with an unknown commanded gain, so every frame that contains one is unreliable and ends the
   * level tracks (as a commanded-gain jump does); they restart on the new response. The full-rate rules
   * and the periodicity are not affected.
   */
  holdLevels(ms: number): void {
    const n = Math.ceil((Math.max(0, ms) / 1000) * this.fsd);
    if (n > this.holdLeft) this.holdLeft = n;
  }

  private trigger(reason: HowlReason, freq: number | null): void {
    this.st.triggered = true;
    this.st.reason = reason;
    this.st.freqHz = freq;
  }

  private analyzeFrame(): void {
    const N = HOWL_FFT_SIZE;
    const re = this.re;
    const im = this.im;
    let gw = 0;
    let gMin = Infinity;
    let gMax = 0;
    for (let i = 0; i < N; i++) {
      const idx = (this.wpos + i) & (N - 1); // oldest first
      const w = this.win[i];
      re[i] = this.xr[idx] * w;
      im[i] = 0;
      const g = this.gr[idx];
      gw += w * g;
      if (g < gMin) gMin = g;
      if (g > gMax) gMax = g;
    }
    // Window-weighted commanded gain: exact for a tone under a linear gain ramp.
    const gEff = gw / this.winSum;
    const reliable = gEff > MIN_FRAME_GAIN && gMin > 0 && gMax / gMin <= MAX_FRAME_GAIN_RATIO;
    this.fft.forward(re, im);
    const pw = this.pw;
    for (let k = 0; k <= N / 2; k++) pw[k] = re[k] * re[k] + im[k] * im[k];

    const kLo = this.kLo;
    const kHi = this.kHi;
    let m = 0;
    let band = 0;
    let kMax = kLo;
    for (let k = kLo; k <= kHi; k++) {
      const v = pw[k];
      this.scratch[m++] = v;
      band += v;
      if (v > pw[kMax]) kMax = k;
    }
    this.domFreq = band > 0 ? (kMax * this.fsd) / N : 0;
    const globalMed = quickMedian(this.scratch, m);
    const ratio = Math.pow(10, HOWL_PEAK_DB / 10);

    // Narrow peaks, strongest MAX_TRACKS kept (insertion into fixed arrays).
    this.nCand = 0;
    for (let k = kLo; k <= kHi; k++) {
      const v = pw[k];
      if (!(v > pw[k - 1] && v >= pw[k + 1] && v > globalMed * ratio)) continue;
      let n = 0;
      for (let j = k - LOCAL_OUTER; j <= k + LOCAL_OUTER; j++) {
        if (j < 1 || j > N / 2 || (j > k - LOCAL_INNER && j < k + LOCAL_INNER)) continue;
        this.scratch[n++] = pw[j];
      }
      if (n < 4 || v <= quickMedian(this.scratch, n) * ratio) continue;
      // Parabolic interpolation on the dB spectrum: sub-bin frequency and peak level.
      const a = 10 * Math.log10(pw[k - 1] + 1e-300);
      const b = 10 * Math.log10(v);
      const c = 10 * Math.log10(pw[k + 1] + 1e-300);
      const d = a - 2 * b + c;
      const delta = d < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / d)) : 0;
      const levelDb = b - 0.25 * (a - c) * delta + this.levelOffsetDb;
      this.insertCandidate(k + delta, levelDb);
    }

    const now = ((this.decCount - N / 2) * 1000) / this.fsd;
    this.updateTracks(reliable, now, gEff, band);
  }

  private insertCandidate(bin: number, db: number): void {
    let pos = this.nCand;
    while (pos > 0 && this.cDb[pos - 1] < db) pos--;
    if (pos >= MAX_TRACKS) return;
    const last = Math.min(this.nCand, MAX_TRACKS - 1);
    for (let i = last; i > pos; i--) {
      this.cBin[i] = this.cBin[i - 1];
      this.cDb[i] = this.cDb[i - 1];
    }
    this.cBin[pos] = bin;
    this.cDb[pos] = db;
    if (this.nCand < MAX_TRACKS) this.nCand++;
  }

  /** Appends a compensated level to track t. */
  private addPoint(t: number, now: number, levelDb: number, gainDb: number): void {
    this.tMatched[t] = 1;
    this.tLastDb[t] = levelDb;
    const slot = t * TRACK_CAP + this.tHead[t];
    this.tTime[slot] = now;
    this.tLevel[slot] = levelDb - gainDb;
    this.tHead[t] = (this.tHead[t] + 1) % TRACK_CAP;
    if (this.tCount[t] < TRACK_CAP) this.tCount[t]++;
    this.updateGrowthRun(t);
  }

  private startTrack(t: number): void {
    this.tActive[t] = 1;
    this.tCount[t] = 0;
    this.tHead[t] = 0;
    this.tGrowStart[t] = Number.NaN;
  }

  private updateTracks(reliable: boolean, now: number, gEff: number, band: number): void {
    this.tMatched.fill(0);
    if (reliable) {
      const gainDb = 20 * Math.log10(gEff);
      for (let c = 0; c < this.nCand; c++) {
        const bin = this.cBin[c];
        let best = -1;
        let bestDist = TRACK_MATCH_BINS;
        for (let t = 0; t < MAX_TRACKS; t++) {
          if (!this.tActive[t] || this.tMatched[t]) continue;
          const dist = Math.abs(this.tBin[t] - bin);
          if (dist <= bestDist) {
            best = t;
            bestDist = dist;
          }
        }
        if (best < 0) {
          for (let t = 0; t < MAX_TRACKS; t++) {
            if (!this.tActive[t] && !this.tMatched[t]) {
              best = t;
              break;
            }
          }
          if (best < 0) continue;
          this.startTrack(best);
        }
        this.tBin[best] = bin;
        this.addPoint(best, now, this.cDb[c], gainDb);
      }
      if (band > 0) {
        if (!this.tActive[BB]) {
          this.startTrack(BB);
          this.bbN = 0;
        }
        this.bbPow[this.bbN % BB_AVG_FRAMES] = band / (gEff * gEff);
        this.bbN++;
        const n = Math.min(this.bbN, BB_AVG_FRAMES);
        let sum = 0;
        for (let i = 0; i < n; i++) sum += this.bbPow[i];
        this.addPoint(BB, now, 10 * Math.log10(sum / n) + this.bandOffsetDb + gainDb, gainDb);
      }
    }
    // Unmatched tracks end; an unreliable frame ends them all.
    let strongest = -1;
    for (let t = 0; t <= BB; t++) {
      if (!this.tMatched[t]) {
        this.tActive[t] = 0;
        continue;
      }
      if (t === BB) continue;
      if (this.tCount[t] >= PERSISTENT_FRAMES && (strongest < 0 || this.tLastDb[t] > this.tLastDb[strongest])) strongest = t;
    }
    this.peakPresent = strongest >= 0;
    if (strongest >= 0) this.peakFreq = (this.tBin[strongest] * this.fsd) / HOWL_FFT_SIZE;

    for (let t = 0; t < MAX_TRACKS; t++) {
      if (this.tMatched[t] && this.growthDetected(t, now)) {
        this.trigger('peak-growth', (this.tBin[t] * this.fsd) / HOWL_FFT_SIZE);
        return;
      }
    }
    // The latest periodicity analysis covers the growth run, and its periodicity is not falling.
    const rho = this.recentRho();
    const perEnd = (this.decCount * 1000) / this.fsd - this.per.ageMs;
    const periodicGrowth =
      rho >= HOWL_GROWTH_RHO &&
      rho >= this.per.previousRho - HOWL_GROWTH_RHO_DROP &&
      perEnd >= this.tGrowStart[BB] + BB_RUN_PERIODIC_MS;
    if (this.tMatched[BB] && periodicGrowth && this.growthDetected(BB, now)) {
      this.trigger('peak-growth', this.peakPresent ? this.peakFreq : this.domFreq);
    }
  }

  /**
   * A growth run starts when the short-term slope (last 5 frames) of the compensated level exceeds
   * the growth rate, and ends as soon as it does not: a level step or a brief rise cannot sustain it,
   * an exponential howl (linear in dB) can.
   */
  private updateGrowthRun(t: number): void {
    if (this.tCount[t] < SHORT_SLOPE_FRAMES) return;
    let st = 0;
    let sl = 0;
    let stt = 0;
    let stl = 0;
    let first = 0;
    let idx = this.tHead[t];
    for (let i = 0; i < SHORT_SLOPE_FRAMES; i++) {
      idx = (idx - 1 + TRACK_CAP) % TRACK_CAP;
      const slot = t * TRACK_CAP + idx;
      const ts = this.tTime[slot] / 1000;
      const l = this.tLevel[slot];
      st += ts;
      sl += l;
      stt += ts * ts;
      stl += ts * l;
      first = this.tTime[slot];
    }
    const n = SHORT_SLOPE_FRAMES;
    const sxx = stt - (st * st) / n;
    const slope = sxx > 0 ? (stl - (st * sl) / n) / sxx : 0;
    this.tShort[t] = slope;
    if (slope > HOWL_GROWTH_DB_PER_S) {
      if (Number.isNaN(this.tGrowStart[t])) this.tGrowStart[t] = first;
    } else {
      this.tGrowStart[t] = Number.NaN;
    }
  }

  /**
   * Growth run of ≥ 300 ms (the band level: plus its averaging span) whose least-squares slope is
   * > 10 dB/s and ≥ 6 standard errors, reaching ≥ 3 dB above anything the track showed before the run
   * (within its ~2.7 s history).
   */
  private growthDetected(t: number, now: number): boolean {
    const from = this.tGrowStart[t];
    if (Number.isNaN(from)) return false;
    const tol = this.hopMs / 2;
    // The band level is averaged over BB_AVG_FRAMES frames: a rise shows up that many hops longer.
    const runMs = t === BB ? HOWL_GROWTH_MS + (BB_AVG_FRAMES - 1) * this.hopMs : HOWL_GROWTH_MS;
    if (now - from < runMs - tol) return false;
    if (this.tLastDb[t] < HOWL_MIN_LEVEL_DBFS) return false;
    let n = 0;
    let st = 0;
    let sl = 0;
    let stt = 0;
    let stl = 0;
    let sll = 0;
    let latest = Number.NaN;
    let before = -Infinity;
    let idx = this.tHead[t];
    for (let i = 0; i < this.tCount[t]; i++) {
      idx = (idx - 1 + TRACK_CAP) % TRACK_CAP;
      const slot = t * TRACK_CAP + idx;
      const time = this.tTime[slot];
      const l = this.tLevel[slot];
      if (i === 0) latest = l;
      if (time < from) {
        if (l > before) before = l;
        continue;
      }
      const ts = (time - now) / 1000;
      n++;
      st += ts;
      sl += l;
      stt += ts * ts;
      stl += ts * l;
      sll += l * l;
    }
    if (n < MIN_GROWTH_POINTS) return false;
    // A howl reaches levels never seen before on this track; beating or modulated tones do not.
    if (latest < before + NEW_LEVEL_DB) return false;
    const sxx = stt - (st * st) / n;
    if (!(sxx > 0)) return false;
    const sxy = stl - (st * sl) / n;
    const slope = sxy / sxx;
    if (!(slope > HOWL_GROWTH_DB_PER_S)) return false;
    if (t === BB && this.tShort[t] < BB_STEADY_RATIO * slope) return false;
    const syy = sll - (sl * sl) / n;
    const ssr = Math.max(0, syy - slope * sxy);
    const se = Math.sqrt(ssr / (n - 2) / sxx);
    return se === 0 || slope / se >= GROWTH_MIN_T;
  }
}
