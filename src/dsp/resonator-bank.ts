import { Biquad, bandpass, biquadResponse, coeffRampSamples, rampShape } from './biquad.ts';
import type { BiquadCoeffs } from './biquad.ts';
import type { Complex } from './complex.ts';
import {
  TRIM_WEIGHTS,
  anyTrim,
  quadratureCompanion,
  trimBasis,
  trimRampShape,
  trimWeights,
  trimmedBankResponseOf,
} from './harmonic-trims.ts';
import type { TrimBasis } from './harmonic-trims.ts';
import { DEFAULT_PARAMS, MAX_HARMONICS, RANGES, minTonalQ } from './params.ts';
import type { HarmonicTrim } from './params.ts';

/** Harmonics at or above this fraction of fs are skipped. */
const MAX_BAND_FRACTION = 0.45;
const MAX_BANDS = RANGES.harmonics.max;
const NEWTON_MAX_ITER = 30;
const NEWTON_TOL = 1e-9;
const MAX_HALVINGS = 8;
/** Accepted |H − 1| at the centres (≈ 0.00006°, 0.00001 dB); limited by the precision of low-ω biquads. */
const ACCEPT_TOL = 1e-6;

export interface ResonatorBankDesign {
  /** Band centres h·f0 actually used (bands ≥ 0.45·fs skipped). */
  harmonicsHz: Float64Array;
  /** Per band: constant-0-dB-peak band-pass coefficients (centre slightly detuned for overlap correction). */
  coeffs: BiquadCoeffs[];
  /**
   * Per band: its quadrature companion, a first-order allpass run on the band's output (allpass · band-pass is
   * −j·band-pass at the band centre, with the band-pass's magnitude everywhere; harmonic-trims.ts).
   */
  quadCoeffs: BiquadCoeffs[];
  /** Per band real output gain. */
  gains: Float64Array;
  /** Per band centre / (h·f0) (the overlap-correction detuning). */
  ratios: Float64Array;
  /** Q actually used (after clamping). */
  q: number;
  /** Worst |H − 1| over the harmonic centres after the correction. */
  residual: number;
  converged: boolean;
}

interface Clamped {
  f0: number;
  harmonics: number;
  q: number;
}

const clampInputs = (fs: number, f0: number, harmonics: number, q: number): Clamped => {
  const fin = (v: number, d: number) => (Number.isFinite(v) ? v : d);
  const t = DEFAULT_PARAMS.tonal;
  const cf0 = Math.min(RANGES.f0.max, Math.max(RANGES.f0.min, fin(f0, t.f0)), MAX_BAND_FRACTION * fs * 0.999);
  const h = Math.round(Math.min(MAX_BANDS, Math.max(RANGES.harmonics.min, fin(harmonics, t.harmonics))));
  // Like clampParams: below minTonalQ(h) the overlap cannot be corrected.
  const cq = Math.min(RANGES.q.max, Math.max(minTonalQ(h), fin(q, t.q)));
  return { f0: cf0, harmonics: h, q: cq };
};

/** Solves A·x = b in place (Gaussian elimination, partial pivoting). Returns false if singular. */
function solveLinear(a: Float64Array[], b: Float64Array): boolean {
  const n = b.length;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (!(Math.abs(a[piv][col]) > 1e-300)) return false;
    if (piv !== col) {
      const tr = a[piv];
      a[piv] = a[col];
      a[col] = tr;
      const tb = b[piv];
      b[piv] = b[col];
      b[col] = tb;
    }
    const p = a[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = a[r][col] / p;
      if (f === 0) continue;
      for (let c = col; c < n; c++) a[r][c] -= f * a[col][c];
      b[r] -= f * b[col];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= a[r][c] * b[c];
    b[r] = s / a[r][r];
  }
  return b.every(Number.isFinite);
}

interface Solution {
  g: Float64Array;
  nu: Float64Array;
  err: number;
}

/** Damped Newton iteration on (g_h, ν_h) from the given start; returns the best point found. */
function newton(fs: number, q: number, fk: Float64Array, g0: Float64Array, ratio0: Float64Array): Solution {
  const m = fk.length;
  const g = Float64Array.from(g0);
  const nu = Float64Array.from(fk, (f, i) => f * ratio0[i]);
  const nuMin = fk.map((f) => f / 1.5);
  const nuMax = fk.map((f) => Math.min(1.5 * f, 0.499 * fs));

  const resid = (gg: Float64Array, nn: Float64Array, out: Float64Array): number => {
    let worst = 0;
    const cs = Array.from(nn, (v) => bandpass(fs, v, q));
    for (let k = 0; k < m; k++) {
      let re = -1;
      let im = 0;
      for (let i = 0; i < m; i++) {
        const h = biquadResponse(cs[i], fk[k], fs);
        re += gg[i] * h.re;
        im += gg[i] * h.im;
      }
      out[2 * k] = re;
      out[2 * k + 1] = im;
      worst = Math.max(worst, Math.hypot(re, im));
    }
    return worst;
  };

  const r = new Float64Array(2 * m);
  let err = resid(g, nu, r);
  const trialG = new Float64Array(m);
  const trialNu = new Float64Array(m);
  const trialR = new Float64Array(2 * m);
  for (let iter = 0; m > 1 && err > NEWTON_TOL && iter < NEWTON_MAX_ITER; iter++) {
    const jac = Array.from({ length: 2 * m }, () => new Float64Array(2 * m));
    for (let i = 0; i < m; i++) {
      const ci = bandpass(fs, nu[i], q);
      const dnu = nu[i] * 1e-6;
      const cp = bandpass(fs, nu[i] + dnu, q);
      const cm = bandpass(fs, nu[i] - dnu, q);
      for (let k = 0; k < m; k++) {
        const h = biquadResponse(ci, fk[k], fs);
        const hp = biquadResponse(cp, fk[k], fs);
        const hm = biquadResponse(cm, fk[k], fs);
        jac[2 * k][i] = h.re;
        jac[2 * k + 1][i] = h.im;
        jac[2 * k][m + i] = (g[i] * (hp.re - hm.re)) / (2 * dnu);
        jac[2 * k + 1][m + i] = (g[i] * (hp.im - hm.im)) / (2 * dnu);
      }
    }
    const step = Float64Array.from(r, (v) => -v);
    if (!solveLinear(jac, step)) break;
    // Damped update: halve the step until the residual decreases.
    let lambda = 1;
    let improved = false;
    for (let tries = 0; tries < MAX_HALVINGS; tries++) {
      for (let i = 0; i < m; i++) {
        trialG[i] = g[i] + lambda * step[i];
        trialNu[i] = Math.min(nuMax[i], Math.max(nuMin[i], nu[i] + lambda * step[m + i]));
      }
      const e = resid(trialG, trialNu, trialR);
      if (e < err) {
        g.set(trialG);
        nu.set(trialNu);
        r.set(trialR);
        err = e;
        improved = true;
        break;
      }
      lambda /= 2;
    }
    if (!improved) break;
  }
  return { g, nu, err };
}

/**
 * Designs the bank: band-passes at h·f0 with the given Q. Overlapping skirts would shift the level and
 * phase at each centre, so every band gets a real gain g_h and a slightly detuned centre ν_h, solved by
 * Newton's method so that Σ_h g_h·BP(ν_h)(h·f0) = 1 + 0j exactly at every harmonic. Detuning (instead of
 * a complex weight) keeps each band a pure band-pass with zeros at DC and Nyquist, so the broadband
 * rejection is preserved. `warm` (a previous converged design with the same Q and band count) is used
 * as the starting point, which makes small f0 updates converge in one or two iterations. A Q below
 * minTonalQ(harmonics) is raised to it (design.q), where a solution always exists.
 */
export function designResonatorBank(
  fs: number,
  f0: number,
  harmonics: number,
  q: number,
  warm?: ResonatorBankDesign | null,
): ResonatorBankDesign {
  const c = clampInputs(fs, f0, harmonics, q);
  let m = 0;
  while (m < c.harmonics && (m + 1) * c.f0 < MAX_BAND_FRACTION * fs) m++;
  const fk = new Float64Array(m);
  for (let i = 0; i < m; i++) fk[i] = (i + 1) * c.f0;
  const ones = new Float64Array(m).fill(1);
  let sol: Solution | null = null;
  if (warm && warm.converged && warm.q === c.q && warm.gains.length === m) {
    sol = newton(fs, c.q, fk, warm.gains, warm.ratios);
  }
  if (sol === null || sol.err > ACCEPT_TOL) sol = newton(fs, c.q, fk, ones, ones);
  const { g, nu, err } = sol;
  if (err <= ACCEPT_TOL) {
    const coeffs = Array.from(nu, (v) => bandpass(fs, v, c.q));
    return {
      harmonicsHz: fk,
      coeffs,
      quadCoeffs: Array.from(nu, (v) => quadratureCompanion(fs, v)),
      gains: g,
      ratios: Float64Array.from(nu, (v, i) => v / fk[i]),
      q: c.q,
      residual: err,
      converged: true,
    };
  }
  // No exact solution. Not expected with Q ≥ minTonalQ (checked over the whole f0 and fs range), kept as a
  // safety net: plain bands at h·f0 with one common gain so that no centre exceeds 0 dB; the response
  // functions still describe this bank exactly, so calibration stays correct.
  const plain = Array.from(fk, (v) => bandpass(fs, v, c.q));
  let worstMag = 0;
  for (let k = 0; k < m; k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < m; i++) {
      const h = biquadResponse(plain[i], fk[k], fs);
      re += h.re;
      im += h.im;
    }
    worstMag = Math.max(worstMag, Math.hypot(re, im));
  }
  const scale = worstMag > 0 ? 1 / worstMag : 1;
  const gains = new Float64Array(m).fill(scale);
  let res = 0;
  for (let k = 0; k < m; k++) {
    let re = -1;
    let im = 0;
    for (let i = 0; i < m; i++) {
      const h = biquadResponse(plain[i], fk[k], fs);
      re += scale * h.re;
      im += scale * h.im;
    }
    res = Math.max(res, Math.hypot(re, im));
  }
  const quadCoeffs = Array.from(fk, (v) => quadratureCompanion(fs, v));
  return { harmonicsHz: fk, coeffs: plain, quadCoeffs, gains, ratios: ones, q: c.q, residual: res, converged: false };
}

/** Single-entry cache: response plots evaluate the same design at many frequencies. */
let cacheKey = '';
let cacheDesign: ResonatorBankDesign | null = null;
let cacheBasis: TrimBasis | null = null;
const cachedDesign = (fs: number, f0: number, harmonics: number, q: number): ResonatorBankDesign => {
  const key = `${fs}|${f0}|${harmonics}|${q}`;
  if (cacheDesign === null || key !== cacheKey) {
    cacheDesign = designResonatorBank(fs, f0, harmonics, q);
    cacheBasis = null;
    cacheKey = key;
  }
  return cacheDesign;
};
const cachedBasis = (fs: number, d: ResonatorBankDesign): TrimBasis => {
  if (cacheBasis === null) cacheBasis = trimBasis(d, fs);
  return cacheBasis;
};

/** Channels with their own harmonic trims (the engine's L and R). */
const TRIM_CHANNELS = 2;

/**
 * Sum of constant-0-dB-peak band-pass biquads at h·f0 (h = 1..harmonics, skipping ≥ 0.45·fs),
 * each normalised so the bank has 0 dB and 0° at every harmonic centre (correct for overlap).
 * configure() solves a small nonlinear system (allocates, ~sub-millisecond): call it on f0/Q changes,
 * not per block. Band states are kept across reconfiguration and the band coefficients and gains glide
 * to the new design together (COEFF_RAMP_MS, Biquad.setCoeffs), so f0 tracking and Q changes do not click.
 *
 * Harmonic trims (harmonic-trims.ts): every band's output also runs through its quadrature companion (a
 * first-order allpass), and process() leaves in `out[ch]` the output of channel ch's trimmed bank,
 * Σ (g_i + u_i)·band_i + v_i·companion_i. A channel whose
 * trims are all neutral has u = v = 0 and out[ch] is exactly the return value (the neutral output). The
 * weights glide (trimRampShape) with the bank's own glide (configure) or over COEFF_RAMP_MS (setTrims).
 */
export class ResonatorBank {
  private readonly fs: number;
  private readonly bands: Biquad[] = [];
  /** Quadrature companions (allpass) of the band outputs (same glide). */
  private readonly quads: Biquad[] = [];
  /** Per band: output gain in use, at the start of the glide, and target (0 for a band fading out). */
  private readonly gains = new Float64Array(MAX_BANDS);
  private readonly gainsFrom = new Float64Array(MAX_BANDS);
  private readonly gainsTo = new Float64Array(MAX_BANDS);
  /** Bands processed: the design's, plus those fading out during a glide. */
  private count = 0;
  /** Bands of the current design. */
  private designCount = 0;
  private rampPos = 0;
  /** Gain glide length in samples, 0 when not gliding. */
  private rampLen = 0;
  private readonly defaultRamp: number;
  private key = '';
  private lastDesign: ResonatorBankDesign | null = null;
  private basis: TrimBasis | null = null;

  /** Latest band and companion outputs. */
  private readonly bandOut = new Float64Array(MAX_BANDS);
  private readonly quadOut = new Float64Array(MAX_BANDS);
  /** Per channel: trims to realise, weights in use / at the start of their glide / target (TRIM_WEIGHTS layout). */
  private readonly trims: HarmonicTrim[][] = [];
  private readonly w: Float64Array[] = [];
  private readonly wFrom: Float64Array[] = [];
  private readonly wTo: Float64Array[] = [];
  private readonly wPos = new Int32Array(TRIM_CHANNELS);
  /** Weight glide length, 0 when not gliding. */
  private readonly wLen = new Int32Array(TRIM_CHANNELS);
  /** 1 while the channel's weights are not all 0 (or gliding). */
  private readonly trimOn = new Uint8Array(TRIM_CHANNELS);
  private readonly scratchW = new Float64Array(TRIM_WEIGHTS);
  private readonly scratchR = new Float64Array(TRIM_WEIGHTS);
  /** Per-channel outputs of the last process(): the neutral output plus that channel's trims. */
  readonly out = new Float64Array(TRIM_CHANNELS);

  constructor(fs: number) {
    this.fs = fs;
    this.defaultRamp = coeffRampSamples(fs);
    for (let i = 0; i < MAX_BANDS; i++) {
      this.bands.push(new Biquad());
      this.quads.push(new Biquad());
    }
    for (let ch = 0; ch < TRIM_CHANNELS; ch++) {
      this.trims.push(Array.from({ length: MAX_HARMONICS }, () => ({ gainDb: 0, phaseDeg: 0 })));
      this.w.push(new Float64Array(TRIM_WEIGHTS));
      this.wFrom.push(new Float64Array(TRIM_WEIGHTS));
      this.wTo.push(new Float64Array(TRIM_WEIGHTS));
    }
    const t = DEFAULT_PARAMS.tonal;
    this.configure(t.f0, t.harmonics, t.q, 0);
  }

  /**
   * Applies a new design. rampSamples > 1 (default: COEFF_RAMP_MS): running bands glide to their new
   * coefficients and gains, bands that appear fade in from silence and bands that disappear (fewer
   * harmonics, or a harmonic crossing 0.45·fs) fade out. rampSamples ≤ 1, or a bank at rest (new, reset,
   * silent): at once. The trims of both channels are re-solved for the new design and glide along.
   */
  configure(f0: number, harmonics: number, q: number, rampSamples: number = this.defaultRamp): void {
    const key = `${f0}|${harmonics}|${q}`;
    if (key === this.key) return;
    const d = designResonatorBank(this.fs, f0, harmonics, q, this.lastDesign);
    this.key = key;
    this.lastDesign = d;
    this.basis = trimBasis(d, this.fs);
    const n = d.coeffs.length;
    const len = Number.isFinite(rampSamples) && !this.atRest() ? Math.floor(rampSamples) : 0;
    for (let i = 0; i < MAX_BANDS; i++) {
      const running = i < this.count;
      if (i < n) {
        if (running) {
          this.bands[i].setCoeffs(d.coeffs[i], len);
          this.quads[i].setCoeffs(d.quadCoeffs[i], len);
        } else {
          this.bands[i].reset();
          this.bands[i].setCoeffs(d.coeffs[i]);
          this.quads[i].reset();
          this.quads[i].setCoeffs(d.quadCoeffs[i]);
          this.gains[i] = 0;
        }
      }
      this.gainsFrom[i] = this.gains[i];
      this.gainsTo[i] = i < n ? d.gains[i] : 0;
    }
    this.designCount = n;
    if (len > 1) {
      this.rampPos = 0;
      this.rampLen = len;
      this.count = Math.max(this.count, n);
    } else this.finishGlide();
    for (let ch = 0; ch < TRIM_CHANNELS; ch++) this.retarget(ch, len);
  }

  /**
   * Harmonic trims of channel `ch` (entry h − 1 for harmonic h; clamped to RANGES, non-finite → 0; entries
   * beyond the design's bands ignored). rampSamples > 1 (default COEFF_RAMP_MS): the weights glide from where
   * they are along trimRampShape (a new target mid-glide restarts from the current point; the current target again
   * is a no-op); ≤ 1, or a bank at rest: at once. Allocation-free (the engine calls it while a trim rises).
   */
  setTrims(ch: number, trims: readonly HarmonicTrim[], rampSamples: number = this.defaultRamp): void {
    if (ch !== 0 && ch !== 1) return;
    const own = this.trims[ch];
    for (let k = 0; k < MAX_HARMONICS; k++) {
      const t = k < trims.length ? trims[k] : undefined;
      own[k].gainDb = t !== undefined && Number.isFinite(t.gainDb) ? t.gainDb : 0;
      own[k].phaseDeg = t !== undefined && Number.isFinite(t.phaseDeg) ? t.phaseDeg : 0;
    }
    const len = Number.isFinite(rampSamples) && !this.atRest() ? Math.floor(rampSamples) : 0;
    this.retarget(ch, len);
  }

  /** The design currently in use (for diagnostics and tests). */
  get design(): ResonatorBankDesign | null {
    return this.lastDesign;
  }

  /** True while channel ch's trim weights glide. */
  trimGliding(ch: number): boolean {
    return this.wLen[ch] !== 0;
  }

  process(x: number): number {
    if (this.rampLen !== 0) this.glide();
    const v = Number.isFinite(x) ? x : 0;
    const bo = this.bandOut;
    const qo = this.quadOut;
    let y = 0;
    for (let i = 0; i < this.count; i++) {
      const b = this.bands[i].process(v);
      bo[i] = b;
      y += this.gains[i] * b;
      qo[i] = this.quads[i].process(b);
    }
    for (let ch = 0; ch < TRIM_CHANNELS; ch++) {
      if (this.trimOn[ch] === 0) {
        this.out[ch] = y;
        continue;
      }
      if (this.wLen[ch] !== 0) this.glideWeights(ch);
      const w = this.w[ch];
      let dy = 0;
      for (let i = 0; i < this.count; i++) dy += w[i] * bo[i] + w[MAX_HARMONICS + i] * qo[i];
      this.out[ch] = y + dy;
    }
    return y;
  }

  /** Clears the band states and completes any glide (bank and trims). */
  reset(): void {
    for (const b of this.bands) b.reset();
    for (const b of this.quads) b.reset();
    this.finishGlide();
    for (let ch = 0; ch < TRIM_CHANNELS; ch++) if (this.wLen[ch] !== 0) this.finishWeights(ch);
  }

  private atRest(): boolean {
    for (let i = 0; i < this.count; i++) if (!this.bands[i].atRest) return false;
    return true;
  }

  /** Same schedule as the band coefficients (Biquad.setCoeffs): both advance once per processed sample. */
  private glide(): void {
    const k = ++this.rampPos;
    if (k >= this.rampLen) {
      this.finishGlide();
      return;
    }
    const w = rampShape(k / this.rampLen);
    for (let i = 0; i < this.count; i++) this.gains[i] = this.gainsFrom[i] + w * (this.gainsTo[i] - this.gainsFrom[i]);
  }

  private finishGlide(): void {
    this.gains.set(this.gainsTo);
    this.rampLen = 0;
    this.settleCount();
  }

  /**
   * Out of a bank glide, the design's bands, plus any band whose trim weight is still gliding to 0 (trims
   * changed during a fade-out): it keeps running, at gain 0, until its weights are exactly 0.
   */
  private settleCount(): void {
    if (this.rampLen !== 0) return;
    let c = this.designCount;
    for (let ch = 0; ch < TRIM_CHANNELS; ch++) {
      if (this.wLen[ch] === 0) continue;
      const w = this.w[ch];
      const to = this.wTo[ch];
      for (let i = MAX_BANDS - 1; i >= c; i--) {
        if (w[i] !== 0 || w[MAX_HARMONICS + i] !== 0 || to[i] !== 0 || to[MAX_HARMONICS + i] !== 0) {
          c = i + 1;
          break;
        }
      }
    }
    this.count = c;
  }

  /** New weight target for channel ch from its trims and the current basis; glides over `len` samples (≤ 1: at once). */
  private retarget(ch: number, len: number): void {
    const to = this.scratchW;
    const basis = this.basis;
    if (basis === null || !anyTrim(this.trims[ch], basis.m)) to.fill(0);
    else trimWeights(basis, this.trims[ch], to, this.scratchR);
    const cur = this.wTo[ch];
    let same = true;
    for (let i = 0; i < TRIM_WEIGHTS && same; i++) same = to[i] === cur[i];
    if (same) {
      // A glide in progress keeps going; at rest nothing changes.
      if (len <= 1 && this.wLen[ch] !== 0) this.finishWeights(ch);
      return;
    }
    cur.set(to);
    if (len > 1) {
      this.wFrom[ch].set(this.w[ch]);
      this.wPos[ch] = 0;
      this.wLen[ch] = len;
      this.trimOn[ch] = 1;
    } else this.finishWeights(ch);
  }

  private glideWeights(ch: number): void {
    const k = ++this.wPos[ch];
    const len = this.wLen[ch];
    if (k >= len) {
      this.finishWeights(ch);
      return;
    }
    const s = trimRampShape(k / len);
    const w = this.w[ch];
    const a = this.wFrom[ch];
    const b = this.wTo[ch];
    for (let i = 0; i < TRIM_WEIGHTS; i++) w[i] = a[i] + s * (b[i] - a[i]);
  }

  private finishWeights(ch: number): void {
    const to = this.wTo[ch];
    this.w[ch].set(to);
    this.wLen[ch] = 0;
    let on = 0;
    for (let i = 0; i < TRIM_WEIGHTS; i++) if (to[i] !== 0) on = 1;
    this.trimOn[ch] = on;
    this.settleCount();
  }
}

/**
 * Response of the bank at f (at rest: the response of a glide's target). With `trims` (one channel's
 * harmonicTrims): that channel's trimmed bank, exactly as ResonatorBank computes out[ch]; without them, or
 * with every used trim neutral, the neutral bank.
 */
export function resonatorBankResponse(
  fs: number,
  f: number,
  f0: number,
  harmonics: number,
  q: number,
  trims?: readonly HarmonicTrim[],
): Complex {
  return resonatorBankResponseAt(fs, f0, harmonics, q, trims)(f);
}

/**
 * resonatorBankResponse as a function of f, the trimmed bank's weights solved once (a loop check evaluates one
 * bank at hundreds of frequencies). The same values, computed the same way.
 */
export function resonatorBankResponseAt(
  fs: number,
  f0: number,
  harmonics: number,
  q: number,
  trims?: readonly HarmonicTrim[],
): (f: number) => Complex {
  const d = cachedDesign(fs, f0, harmonics, q);
  if (trims === undefined || !anyTrim(trims, d.coeffs.length)) {
    return (f) => {
      let re = 0;
      let im = 0;
      for (let i = 0; i < d.coeffs.length; i++) {
        const h = biquadResponse(d.coeffs[i], f, fs);
        re += d.gains[i] * h.re;
        im += d.gains[i] * h.im;
      }
      return { re, im };
    };
  }
  const w = new Float64Array(TRIM_WEIGHTS);
  trimWeights(cachedBasis(fs, d), trims, w, new Float64Array(TRIM_WEIGHTS));
  return (f) => trimmedBankResponseOf(d, w, f, fs);
}
