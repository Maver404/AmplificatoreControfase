/**
 * Per-channel complex trims of the harmonics in tonal mode (ChannelParams.harmonicTrims).
 *
 * Why: a single delay per channel sets the phase of every harmonic h at once (h·Δ), which is right only when
 * the chain and the acoustic paths have linear phase. The default 20 Hz 2nd-order HP, the Bessel LP and room
 * reflections are not, so a delay that cancels the fundamental leaves the higher harmonics partly in phase
 * (simulator: −17 dB at 100 Hz, only ~−10 dB at 400 Hz). A trim multiplies the channel's response at h·f0 by
 * T_h = 10^(gainDb/20)·e^{−jθ_h} (θ_h = phaseDeg, positive = lag), on top of the delay, harmonic by harmonic.
 *
 * Structure: the resonator bank's band-passes stay shared (one set for both channels) and its neutral output
 * Σ g_i·BP_i (the 'cond' signal) is unchanged. Each band's output also feeds a QUADRATURE COMPANION: the
 * first-order allpass AP_i(z) = (a_i + z⁻¹)/(1 + a_i·z⁻¹), a_i = (t_i − 1)/(t_i + 1), t_i = tan(π·ν_i/fs),
 * which lags 90° at the band centre ν_i, so Q_i = AP_i·BP_i is exactly −j·BP_i there and |Q_i| = |BP_i| at
 * every frequency. A channel's trimmed bank is
 *   B_ch = Σ_i (g_i + u_i)·BP_i + v_i·Q_i = Σ_i BP_i·(g_i + u_i + v_i·AP_i),
 * and its 2m real weights (u, v) solve the 2m × 2m real linear system "B_ch(h·f0) = N_h·T_h for every used
 * harmonic h", N_h being the neutral bank's response there (1 + 0j to 1e-6, so θ_h is the extra lag at h·f0
 * exactly and the trims of different harmonics do not interact). The matrix depends only on the bank design
 * (f0, harmonics, Q, fs): its inverse is computed when the design changes (configure, message handler) and
 * the weights are a matrix-vector product of it, so applying new trims never solves anything.
 * Choices:
 * - Each band contributes an in-phase and a quadrature component, so the system is diagonally dominant
 *   (≈ [1, 0; 0, −1] blocks, coupled by the skirts) and the weights are O(trim). Weighting band[n] and
 *   band[n−1] instead (the obvious alternative) is nearly singular at low ω: a 30° trim at 100 Hz / 48 kHz needs
 *   weights ≈ ±38 whose difference is a differentiator, lifting the skirts to ~1/Q up to Nyquist (≈ −35 dB at
 *   10 kHz instead of −70 dB).
 * - The companion has the band-pass's own skirts (|AP| = 1): each band of a trimmed bank stays a band-pass
 *   whose gain changes by |g + u + v·AP|/|g| ≤ (|Re T| + |Im T|)·(1 + the overlap terms) ≈ √2·|T| at most, far
 *   from its centre, so the tonal chain still rejects what lies between and below the harmonics (wind and
 *   rumble at the reference mic). An earlier companion with the band-pass's poles and an RBJ low-pass
 *   numerator was −j·BP at the centre too, but flat at ~1/Q below it: a +6 dB / 90° trim on the fundamental
 *   raised 20 Hz by 13–15 dB, a 180° phase-only trim by 15 dB. White noise through a bank with the same T on
 *   every harmonic carries |T|² times the neutral bank's power within 1 dB at Q 30 (phases up to ±150°, gains
 *   −6…+6 dB), within 3 dB with 8 harmonics at Q 6, where AP's phase (−90° only at each centre) changes how
 *   the overlapping bands add between the centres.
 * - With every trim of a channel at 0 its weights are exactly 0 and its output IS the neutral output
 *   (sample-identical); the engine then reproduces the untrimmed engine bit for bit.
 * - Trim changes glide the weights over COEFF_RAMP_MS, starting and ending with the bank's own coefficient
 *   and gain glides, so an f0 or Q glide carries the trims along. The shape is trimRampShape (C2), not the
 *   bank's rampShape (C1): the weights scale the band outputs directly and one trim can move a harmonic by
 *   3× its level (+6 dB at 180°). A 50 / 100 / 400 Hz tone whose gain glides from 0 dB to +6 dB/90° leaves
 *   −120 / −119 / −104.5 dB above 2 kHz (rampShape: −92 / −88 dB, above the −100 dB of the other live changes).
 *   The safety rule of gains (rises above BIG_JUMP_DB at ≤ MAX_RISE_DB_PER_S, with the same budget) applies
 *   to each harmonic's level, channel gain + master + trim: TrimRamp, run by the engine, shares the channel's
 *   GainRamp budget and turns a big increase into back-to-back 20 ms glides of ≤ 0.12 dB (C2 joins: no click
 *   either); the engine starts a new glide only when the previous one has ended.
 * The weights are per channel, so the two channels' tonal signals differ: the engine keeps one delay line per
 * channel (written with the neutral signal when the channel is untrimmed or not in tonal mode).
 */
import { biquadResponse, coeffRampSamples } from './biquad.ts';
import type { BiquadCoeffs } from './biquad.ts';
import type { Complex } from './complex.ts';
import { BIG_JUMP_DB, MAX_HARMONICS, MAX_RISE_DB_PER_S, RANGES } from './params.ts';
import type { HarmonicTrim } from './params.ts';
import type { ResonatorBankDesign } from './resonator-bank.ts';

/** Weight vector layout: [u_0 … u_{MAX−1}, v_0 … v_{MAX−1}] (in-phase, quadrature). */
export const TRIM_WEIGHTS = 2 * MAX_HARMONICS;

/** Glide shape of the trim weights: smootherstep 6u⁵ − 15u⁴ + 10u³ (zero slope and curvature at both ends). */
export const trimRampShape = (u: number): number => u * u * u * (u * (u * 6 - 15) + 10);

/**
 * Quadrature companion of a band centred on `centreHz`, applied to the band's OUTPUT: the first-order allpass
 * (a + z⁻¹)/(1 + a·z⁻¹) with a 90° lag at the centre, so that companion · band-pass = −j·band-pass there and
 * |companion · band-pass| = |band-pass| everywhere.
 */
export function quadratureCompanion(fs: number, centreHz: number): BiquadCoeffs {
  const t = Math.tan((Math.PI * centreHz) / fs);
  const a = (t - 1) / (t + 1);
  return { b0: a, b1: 1, b2: 0, a1: a, a2: 0 };
}

/** Response of band i's companion branch, quadCoeffs[i] after coeffs[i] (AP_i·BP_i), at f. */
function companionResponse(d: ResonatorBankDesign, i: number, f: number, fs: number, bp: Complex): Complex {
  const ap = biquadResponse(d.quadCoeffs[i], f, fs);
  return { re: ap.re * bp.re - ap.im * bp.im, im: ap.re * bp.im + ap.im * bp.re };
}

/** What the weights of a bank design are computed from (see the file header). */
export interface TrimBasis {
  /** Bands of the design (= harmonics used). */
  m: number;
  /** Neutral bank response Σ g_i·BP_i at each harmonic centre (≈ 1 + 0j). */
  nRe: Float64Array;
  nIm: Float64Array;
  /** Row-major 2m × 2m inverse of the response matrix (columns u_i, v_i; rows Re, Im at each centre); null if singular. */
  inv: Float64Array | null;
}

/** Gauss-Jordan inverse with partial pivoting (row-major n × n); null if singular or not finite. */
function invert(a: Float64Array, n: number): Float64Array | null {
  const m = Float64Array.from(a);
  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) inv[i * n + i] = 1;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r * n + col]) > Math.abs(m[piv * n + col])) piv = r;
    if (!(Math.abs(m[piv * n + col]) > 1e-300)) return null;
    if (piv !== col) {
      for (let c = 0; c < n; c++) {
        let t = m[piv * n + c];
        m[piv * n + c] = m[col * n + c];
        m[col * n + c] = t;
        t = inv[piv * n + c];
        inv[piv * n + c] = inv[col * n + c];
        inv[col * n + c] = t;
      }
    }
    const p = m[col * n + col];
    for (let c = 0; c < n; c++) {
      m[col * n + c] /= p;
      inv[col * n + c] /= p;
    }
    for (let r = 0; r < n; r++) {
      const f = m[r * n + col];
      if (r === col || f === 0) continue;
      for (let c = 0; c < n; c++) {
        m[r * n + c] -= f * m[col * n + c];
        inv[r * n + c] -= f * inv[col * n + c];
      }
    }
  }
  return inv.every(Number.isFinite) ? inv : null;
}

/** Builds the basis of a bank design (allocates; call when the design changes, not per sample). */
export function trimBasis(d: ResonatorBankDesign, fs: number): TrimBasis {
  const m = d.coeffs.length;
  const n2 = 2 * m;
  const a = new Float64Array(n2 * n2);
  const nRe = new Float64Array(m);
  const nIm = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const f = d.harmonicsHz[k];
    for (let i = 0; i < m; i++) {
      const b = biquadResponse(d.coeffs[i], f, fs);
      const q = companionResponse(d, i, f, fs, b);
      a[2 * k * n2 + i] = b.re;
      a[(2 * k + 1) * n2 + i] = b.im;
      a[2 * k * n2 + m + i] = q.re;
      a[(2 * k + 1) * n2 + m + i] = q.im;
      nRe[k] += d.gains[i] * b.re;
      nIm[k] += d.gains[i] * b.im;
    }
  }
  return { m, nRe, nIm, inv: m > 0 ? invert(a, n2) : null };
}

/** Clamped value of a trim field; anything not finite is 0 (untrimmed). */
const field = (v: number | undefined, r: { min: number; max: number }): number =>
  v !== undefined && Number.isFinite(v) ? Math.min(r.max, Math.max(r.min, v)) : 0;

/** True when any of the first `m` trims is not neutral (non-finite values count as 0). */
export function anyTrim(trims: readonly HarmonicTrim[], m = MAX_HARMONICS): boolean {
  for (let k = 0; k < m && k < trims.length; k++) {
    const t = trims[k];
    if (field(t?.gainDb, RANGES.trimGainDb) !== 0 || field(t?.phaseDeg, RANGES.trimPhaseDeg) !== 0) return true;
  }
  return false;
}

/**
 * Weights realising `trims` on a design: out[i] = u_i, out[MAX_HARMONICS + i] = v_i (TRIM_WEIGHTS entries,
 * zero beyond the design's bands). `r` is a scratch array of ≥ TRIM_WEIGHTS entries. Allocation-free: the
 * engine calls it while a trim rises. Trims are clamped to RANGES (non-finite → 0); entries beyond the
 * design's bands are ignored. Returns false (all weights 0) when every used trim is neutral or the basis is
 * singular.
 */
export function trimWeights(b: TrimBasis, trims: readonly HarmonicTrim[], out: Float64Array, r: Float64Array): boolean {
  out.fill(0);
  const m = b.m;
  const inv = b.inv;
  if (inv === null) return false;
  let any = false;
  for (let k = 0; k < m; k++) {
    const t = k < trims.length ? trims[k] : undefined;
    const g = field(t?.gainDb, RANGES.trimGainDb);
    const ph = field(t?.phaseDeg, RANGES.trimPhaseDeg);
    if (g === 0 && ph === 0) {
      r[2 * k] = 0;
      r[2 * k + 1] = 0;
      continue;
    }
    any = true;
    // c = N·(T − 1), T = γ·e^{−jθ}
    const amp = Math.pow(10, g / 20);
    const th = (ph * Math.PI) / 180;
    const tr = amp * Math.cos(th) - 1;
    const ti = -amp * Math.sin(th);
    r[2 * k] = b.nRe[k] * tr - b.nIm[k] * ti;
    r[2 * k + 1] = b.nRe[k] * ti + b.nIm[k] * tr;
  }
  if (!any) return false;
  const n2 = 2 * m;
  for (let row = 0; row < n2; row++) {
    let s = 0;
    const o = row * n2;
    for (let c = 0; c < n2; c++) s += inv[o + c] * r[c];
    out[row < m ? row : MAX_HARMONICS + row - m] = s;
  }
  return true;
}

/** Response of a trimmed bank at f: Σ (g_i + u_i)·BP_i(f) + v_i·Q_i(f), weights as trimWeights() returns them. */
export function trimmedBankResponseOf(d: ResonatorBankDesign, w: Float64Array | null, f: number, fs: number): Complex {
  let re = 0;
  let im = 0;
  for (let i = 0; i < d.coeffs.length; i++) {
    const h = biquadResponse(d.coeffs[i], f, fs);
    const u = d.gains[i] + (w !== null ? w[i] : 0);
    re += u * h.re;
    im += u * h.im;
    const v = w !== null ? w[MAX_HARMONICS + i] : 0;
    if (v !== 0) {
      const q = companionResponse(d, i, f, fs, h);
      re += v * q.re;
      im += v * q.im;
    }
  }
  return { re, im };
}

/** The complex factor a trim applies at its harmonic: 10^(gainDb/20)·e^{−j·phaseDeg·π/180} (clamped like the engine). */
export function trimFactor(t: HarmonicTrim): Complex {
  const amp = Math.pow(10, field(t.gainDb, RANGES.trimGainDb) / 20);
  const th = (field(t.phaseDeg, RANGES.trimPhaseDeg) * Math.PI) / 180;
  return { re: amp * Math.cos(th), im: -amp * Math.sin(th) };
}

/**
 * The trim that turns `current` (the channel's response at h·f0 with this harmonic's trim at 0, e.g.
 * channelResponse(withoutTrims(p), fs, h·f0, ch)) into `desired`: T = desired / current, gainDb = 20·log10|T|
 * and phaseDeg = −arg T in degrees (lag positive), both clamped to RANGES (a gain outside −30…+6 dB must come
 * from the channel gain). The trims of different harmonics do not interact, so each can be computed alone.
 * Returns the neutral trim when either response is 0 or not finite.
 */
export function trimForTarget(desired: Complex, current: Complex): HarmonicTrim {
  const d = current.re * current.re + current.im * current.im;
  const re = (desired.re * current.re + desired.im * current.im) / d;
  const im = (desired.im * current.re - desired.re * current.im) / d;
  const mag = Math.hypot(re, im);
  if (!(mag > 0) || !Number.isFinite(mag)) return { gainDb: 0, phaseDeg: 0 };
  const { trimGainDb: g, trimPhaseDeg: p } = RANGES;
  return {
    gainDb: Math.min(g.max, Math.max(g.min, 20 * Math.log10(mag))),
    phaseDeg: Math.min(p.max, Math.max(p.min, (-Math.atan2(im, re) * 180) / Math.PI)),
  };
}

/**
 * The rise budget of the gain in front of a channel's trims (gain-ramp.ts GainRamp: channel gain + master).
 * A harmonic's level is that gain plus its trim, so both rises draw from one budget per harmonic.
 */
export interface RiseBudget {
  /** Remaining fast-rise budget, dB (0 … BIG_JUMP_DB). */
  readonly budgetDb: number;
  /** Highest budget since the previous call (refills included); the record restarts from the current budget. */
  takeBudgetPeak(): number;
  /** Lowers the budget to `db` (spent by a trim rise) and restarts the record. */
  spendBudgetTo(db: number): void;
}

/** A RiseBudget refilled by the TrimRamp's own clock: a TrimRamp without a gain in front (tests, standalone). */
class ClockBudget implements RiseBudget {
  budgetDb = BIG_JUMP_DB;
  private peak = BIG_JUMP_DB;

  refill(db: number): void {
    this.budgetDb = Math.min(BIG_JUMP_DB, this.budgetDb + db);
    if (this.budgetDb > this.peak) this.peak = this.budgetDb;
  }

  fill(): void {
    this.budgetDb = BIG_JUMP_DB;
    this.peak = BIG_JUMP_DB;
  }

  takeBudgetPeak(): number {
    const p = Math.max(this.peak, this.budgetDb);
    this.peak = this.budgetDb;
    return p;
  }

  spendBudgetTo(db: number): void {
    if (db < this.budgetDb) this.budgetDb = Math.max(0, db);
    this.peak = this.budgetDb;
  }
}

/**
 * The engine's rate limiter of one channel's trims (the gain safety rule of GainRamp, per harmonic):
 * decreases, phase changes and increases ≤ BIG_JUMP_DB apply at once (the bank glides them over
 * COEFF_RAMP_MS); while (target − applied) > BIG_JUMP_DB the gain rises at MAX_RISE_DB_PER_S, and every
 * rise draws from a BIG_JUMP_DB budget refilled at MAX_RISE_DB_PER_S, so over any interval T a trim rises by
 * at most BIG_JUMP_DB + MAX_RISE_DB_PER_S·T (−30 → +6 dB: 30 dB at 6 dB/s, then the last 6 dB).
 *
 * Joint budget: harmonic h's LEVEL is the channel gain (with the master: `gain`, the engine's GainRamp) plus
 * its trim, and it obeys that rule as a whole. Its budget is c_h = gain.budgetDb + extra_h: a gain rise
 * spends the gain's budget, so every c_h; a trim rise spends c_h, and the gain's budget is lowered to the
 * lowest c_h (a later gain rise also raises that harmonic). extra_h ≥ 0 is what harmonic h has beyond the
 * gain's budget; since every c_h refills with it and stops at BIG_JUMP_DB, extra_h shrinks to
 * BIG_JUMP_DB − (highest gain budget since the last advance). Exact, and allocation-free. Without `gain` the
 * ramp keeps its own budget, refilled by the sample clock (the trims alone).
 *
 * While `hold` (nobody hears the anti-noise: bypass, measurement gate) increases wait and the budget still
 * refills. Time is the engine's sample clock; a pending rise advances in steps of COEFF_RAMP_MS (`step()`,
 * ≤ 0.12 dB each, glided back to back by the bank). Allocation-free after construction.
 */
export class TrimRamp {
  /** Trims in effect (mutated in place, never reallocated): what the bank is told to apply. */
  readonly applied: HarmonicTrim[];
  private readonly tgtDb = new Float64Array(MAX_HARMONICS);
  private readonly tgtDeg = new Float64Array(MAX_HARMONICS);
  /** Per harmonic: budget beyond the shared one (harmonic h's budget = shared.budgetDb + extra[h − 1]). */
  private readonly extra = new Float64Array(MAX_HARMONICS);
  private readonly shared: RiseBudget;
  private readonly own: ClockBudget | null;
  private readonly perSample: number;
  /** Length of a rise step (and of the bank's glide), samples. */
  readonly stepSamples: number;
  /** Sample clock of the last advance (the pace of the slow part). */
  private last = 0;
  /** Clock at which the next rise step is due. */
  private dueAt = 0;
  private rising = false;
  /** Applied gains at the last setMark() (the engine marks what it handed to the live chain). */
  private readonly mark = new Float64Array(MAX_HARMONICS);

  /** `gain`: the channel's gain ramp, whose budget the trims share (see the class comment). */
  constructor(fs: number, gain?: RiseBudget) {
    this.perSample = MAX_RISE_DB_PER_S / fs;
    this.stepSamples = coeffRampSamples(fs);
    this.applied = Array.from({ length: MAX_HARMONICS }, () => ({ gainDb: 0, phaseDeg: 0 }));
    this.own = gain === undefined ? new ClockBudget() : null;
    this.shared = gain ?? this.own!;
  }

  /** A rise is still under way. */
  get pending(): boolean {
    return this.rising;
  }

  /** True when a step is due at `clock`. */
  due(clock: number): boolean {
    return this.rising && clock >= this.dueAt;
  }

  /** Largest gain increase of any harmonic since the last setMark(), dB (0 when none rose). */
  riseSinceMark(): number {
    let m = 0;
    for (let k = 0; k < MAX_HARMONICS; k++) m = Math.max(m, this.applied[k].gainDb - this.mark[k]);
    return m;
  }

  /** Marks the applied gains (riseSinceMark() restarts from them). */
  setMark(): void {
    for (let k = 0; k < MAX_HARMONICS; k++) this.mark[k] = this.applied[k].gainDb;
  }

  /**
   * Target and applied trims both set to `trims` at once, budgets full (output silent: not armed; the engine
   * jumps the gain ramp first, which fills its budget).
   */
  jump(trims: readonly HarmonicTrim[], clock: number): void {
    for (let k = 0; k < MAX_HARMONICS; k++) {
      const t = trims[k];
      this.tgtDb[k] = field(t?.gainDb, RANGES.trimGainDb);
      this.tgtDeg[k] = field(t?.phaseDeg, RANGES.trimPhaseDeg);
      this.applied[k].gainDb = this.tgtDb[k];
      this.applied[k].phaseDeg = this.tgtDeg[k];
    }
    this.own?.fill();
    this.extra.fill(0);
    this.shared.takeBudgetPeak();
    this.last = clock;
    this.rising = false;
    this.setMark();
  }

  /**
   * New requested trims. Unchanged targets: no-op (false). Otherwise the rise in progress is first brought
   * to `clock`, then decreases, phase changes and (unless `hold`) increases within the rules apply at once.
   * Returns true when `applied` changed.
   */
  setTarget(trims: readonly HarmonicTrim[], clock: number, hold: boolean): boolean {
    let same = true;
    for (let k = 0; k < MAX_HARMONICS && same; k++) {
      const t = trims[k];
      same = field(t?.gainDb, RANGES.trimGainDb) === this.tgtDb[k] && field(t?.phaseDeg, RANGES.trimPhaseDeg) === this.tgtDeg[k];
    }
    if (same) return false;
    let changed = this.advance(clock, hold);
    for (let k = 0; k < MAX_HARMONICS; k++) {
      const t = trims[k];
      this.tgtDb[k] = field(t?.gainDb, RANGES.trimGainDb);
      this.tgtDeg[k] = field(t?.phaseDeg, RANGES.trimPhaseDeg);
    }
    // No elapsed time left: only the immediate parts (decreases, phase, fast rises from the budget).
    if (this.advance(clock, hold)) changed = true;
    this.dueAt = clock + this.stepSamples;
    return changed;
  }

  /** One scheduled step of a pending rise (call when due()); true when `applied` changed. */
  step(clock: number, hold: boolean): boolean {
    const changed = this.advance(clock, hold);
    this.dueAt = clock + this.stepSamples;
    return changed;
  }

  /** Brings every trim to `clock` under the rules. */
  private advance(clock: number, hold: boolean): boolean {
    const pace = Math.max(0, clock - this.last) * this.perSample;
    this.last = clock;
    this.own?.refill(pace);
    const shared = this.shared.budgetDb;
    // Every harmonic's budget stops at BIG_JUMP_DB while the shared one refills.
    const top = BIG_JUMP_DB - this.shared.takeBudgetPeak();
    const c = this.extra;
    let lowest = shared;
    let changed = false;
    let rising = false;
    for (let k = 0; k < MAX_HARMONICS; k++) {
      const a = this.applied[k];
      if (a.phaseDeg !== this.tgtDeg[k]) {
        a.phaseDeg = this.tgtDeg[k];
        changed = true;
      }
      // Harmonic k's budget now (c[k] holds it until the end of the loop).
      let budget = shared + Math.max(0, Math.min(c[k], top));
      let cur = a.gainDb;
      const tgt = this.tgtDb[k];
      let gap = tgt - cur;
      if (gap < 0) cur = tgt;
      else if (gap > 0 && !hold) {
        if (gap > BIG_JUMP_DB) {
          // Slow part, at MAX_RISE_DB_PER_S: the rest of the budget is kept for the last BIG_JUMP_DB.
          const slow = Math.min(pace, gap - BIG_JUMP_DB, budget);
          cur += slow;
          gap -= slow;
          budget -= slow;
        }
        if (gap <= BIG_JUMP_DB) {
          const fast = Math.min(gap, budget);
          cur += fast;
          budget -= fast;
          // Rounding must not leave a pending rise of a few ulps.
          if (tgt - cur < 1e-9) cur = tgt;
        }
      }
      c[k] = budget;
      if (budget < lowest) lowest = budget;
      if (cur !== a.gainDb) {
        a.gainDb = cur;
        changed = true;
      }
      if (cur < tgt) rising = true;
    }
    // A later gain rise raises every harmonic: the shared budget is the lowest harmonic's.
    if (lowest < shared) this.shared.spendBudgetTo(lowest);
    const base = this.shared.budgetDb;
    for (let k = 0; k < MAX_HARMONICS; k++) c[k] = Math.max(0, c[k] - base);
    this.rising = rising;
    return changed;
  }
}
