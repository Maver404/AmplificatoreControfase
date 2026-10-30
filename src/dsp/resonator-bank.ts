import { Biquad, bandpass, biquadResponse, coeffRampSamples, rampShape } from './biquad.ts';
import type { BiquadCoeffs } from './biquad.ts';
import type { Complex } from './complex.ts';
import { DEFAULT_PARAMS, RANGES, minTonalQ } from './params.ts';

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
    return {
      harmonicsHz: fk,
      coeffs: Array.from(nu, (v) => bandpass(fs, v, c.q)),
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
  return { harmonicsHz: fk, coeffs: plain, gains, ratios: ones, q: c.q, residual: res, converged: false };
}

/** Single-entry cache: response plots evaluate the same design at many frequencies. */
let cacheKey = '';
let cacheDesign: ResonatorBankDesign | null = null;
const cachedDesign = (fs: number, f0: number, harmonics: number, q: number): ResonatorBankDesign => {
  const key = `${fs}|${f0}|${harmonics}|${q}`;
  if (cacheDesign === null || key !== cacheKey) {
    cacheDesign = designResonatorBank(fs, f0, harmonics, q);
    cacheKey = key;
  }
  return cacheDesign;
};

/**
 * Sum of constant-0-dB-peak band-pass biquads at h·f0 (h = 1..harmonics, skipping ≥ 0.45·fs),
 * each normalised so the bank has 0 dB and 0° at every harmonic centre (correct for overlap).
 * configure() solves a small nonlinear system (allocates, ~sub-millisecond): call it on f0/Q changes,
 * not per block. Band states are kept across reconfiguration and the band coefficients and gains glide
 * to the new design together (COEFF_RAMP_MS, Biquad.setCoeffs), so f0 tracking and Q changes do not click.
 */
export class ResonatorBank {
  private readonly fs: number;
  private readonly bands: Biquad[] = [];
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

  constructor(fs: number) {
    this.fs = fs;
    this.defaultRamp = coeffRampSamples(fs);
    for (let i = 0; i < MAX_BANDS; i++) this.bands.push(new Biquad());
    const t = DEFAULT_PARAMS.tonal;
    this.configure(t.f0, t.harmonics, t.q, 0);
  }

  /**
   * Applies a new design. rampSamples > 1 (default: COEFF_RAMP_MS): running bands glide to their new
   * coefficients and gains, bands that appear fade in from silence and bands that disappear (fewer
   * harmonics, or a harmonic crossing 0.45·fs) fade out. rampSamples ≤ 1, or a bank at rest (new, reset,
   * silent): at once.
   */
  configure(f0: number, harmonics: number, q: number, rampSamples: number = this.defaultRamp): void {
    const key = `${f0}|${harmonics}|${q}`;
    if (key === this.key) return;
    const d = designResonatorBank(this.fs, f0, harmonics, q, this.lastDesign);
    this.key = key;
    this.lastDesign = d;
    const n = d.coeffs.length;
    const len = Number.isFinite(rampSamples) && !this.atRest() ? Math.floor(rampSamples) : 0;
    for (let i = 0; i < MAX_BANDS; i++) {
      const running = i < this.count;
      if (i < n) {
        if (running) this.bands[i].setCoeffs(d.coeffs[i], len);
        else {
          this.bands[i].reset();
          this.bands[i].setCoeffs(d.coeffs[i]);
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
  }

  /** The design currently in use (for diagnostics and tests). */
  get design(): ResonatorBankDesign | null {
    return this.lastDesign;
  }

  process(x: number): number {
    if (this.rampLen !== 0) this.glide();
    const v = Number.isFinite(x) ? x : 0;
    let y = 0;
    for (let i = 0; i < this.count; i++) y += this.gains[i] * this.bands[i].process(v);
    return y;
  }

  /** Clears the band states and completes any glide. */
  reset(): void {
    for (const b of this.bands) b.reset();
    this.finishGlide();
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
    this.count = this.designCount;
  }
}

export function resonatorBankResponse(fs: number, f: number, f0: number, harmonics: number, q: number): Complex {
  const d = cachedDesign(fs, f0, harmonics, q);
  let re = 0;
  let im = 0;
  for (let i = 0; i < d.coeffs.length; i++) {
    const h = biquadResponse(d.coeffs[i], f, fs);
    re += d.gains[i] * h.re;
    im += d.gains[i] * h.im;
  }
  return { re, im };
}
