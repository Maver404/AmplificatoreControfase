# Module contracts

Internal API spec that every module must implement exactly. The approved design is in
`/Users/live/.claude/plans/dobbiamo-creare-un-app-una-majestic-crescent.md`.
Already written (do not change signatures without updating every user): `src/dsp/complex.ts`,
`src/dsp/biquad.ts`, `src/dsp/params.ts`, `src/audio/protocol.ts`.

## Global rules
- `src/dsp/**` is pure TypeScript: no `window`, `self`, `document`, `sampleRate` global, `AudioContext`,
  `import.meta`, or npm imports. It must compile under the app (DOM), worklet (ES2022 + audioworklet) and
  worker (WebWorker) tsconfigs. Pass the sample rate explicitly.
- Per-sample methods (`process`, `next`, `push`, `read`) allocate nothing: no arrays, objects, closures or
  tuples. Use preallocated `Float64Array`s and fields. Constructors and `configure`/`set*` may allocate.
- Double precision state, NaN/Infinity reset to 0, manual denormal flush (|v| < 1e-30 → 0) on recursive states.
- Never assume a 128-frame block: block length comes from the arrays passed in.
- Import paths use the `.ts` extension (`import { Biquad } from './biquad.ts'`), `import type` for types.
- Code comments in English, user-facing strings in Italian.
- Tests: Vitest, files in `tests/unit/<module>.test.ts`, `environment: node`, deterministic (seeded PRNG,
  no `Math.random` without a seed).

## DSP primitives (`src/dsp/`)

### biquad.ts
```ts
/** a0 = 1: H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²). Designers: lowpass/highpass/bandpass (RBJ,
 *  linear Q; bandpass has a 0 dB peak and 0° at f0), onePoleLowpass/onePoleHighpass (bilinear, pre-warped). */
export interface BiquadCoeffs { b0: number; b1: number; b2: number; a1: number; a2: number }
export const COEFF_RAMP_MS = 20;                          // glide used for live filter changes
export const coeffRampSamples = (fs: number) => number    // max(2, round(COEFF_RAMP_MS·fs/1000))
export const rampShape = (u: number) => number             // smoothstep 3u² − 2u³ (zero slope at both ends)
/** Direct-form I, double precision. State = last 2 inputs/outputs (independent of the coefficients). */
export class Biquad {
  constructor(coeffs?: BiquadCoeffs)
  /** rampSamples ≤ 1 (default): the next sample uses coeffs (unchanged legacy behaviour).
   *  rampSamples > 1: glides from the coefficients in use along rampShape and is exactly on target after
   *  rampSamples processed samples. A new target mid-glide restarts from the current point; the same
   *  target again is a no-op. A filter at rest (zero state: new, reset, silent) switches at once. */
  setCoeffs(coeffs: BiquadCoeffs, rampSamples?: number): void
  get coeffs(): BiquadCoeffs  // COPY of the coefficients in use (mid-glide: interpolated); was the live internal object
  get ramping(): boolean
  get atRest(): boolean       // state all zero
  reset(): void               // zero state AND completes any glide
  process(x: number): number  // NaN/Inf → reset, returns 0; |y| < 1e-30 flushed in the state
}
/** Closed form (of the given coefficients, i.e. a glide's target). */
export function biquadResponse(c: BiquadCoeffs, f: number, fs: number): Complex
```
Steady-state output equals the former transposed direct-form II within 1e-10 of the output RMS (10 Hz at 192 kHz
included). Rounding error is about 1e-11 RMS against exact arithmetic. Each intermediate filter of a glide is a
convex combination of the start and the target, so it is stable when both are.

### params.ts (additions)
```ts
export const TONAL_MIN_Q: readonly number[] = [2, 2, 2, 2, 2, 3, 4, 5, 6]  // index = number of harmonics
/** Lowest allowed tonal Q for `harmonics` (rounded and clamped to the harmonics range; NaN → strictest). */
export const minTonalQ = (harmonics: number) => number
```
clampParams raises `tonal.q` to minTonalQ(tonal.harmonics) (the resonator bank does the same with its inputs).

Harmonic trims (tonal mode only, see harmonic-trims.ts):
```ts
/** The channel's response at h·f0 is multiplied by 10^(gainDb/20)·e^{−j·phaseDeg·π/180} (positive phase = lag). */
export interface HarmonicTrim { gainDb: number; phaseDeg: number }
export const MAX_HARMONICS = 8                      // = RANGES.harmonics.max; entry h − 1 trims harmonic h
ChannelParams.harmonicTrims: HarmonicTrim[]         // exactly MAX_HARMONICS entries, default all { 0, 0 }
RANGES.trimGainDb = { min: -30, max: 6 }; RANGES.trimPhaseDeg = { min: -180, max: 180 }
export const neutralTrims = () => HarmonicTrim[]    // MAX_HARMONICS new { gainDb: 0, phaseDeg: 0 }
export function withoutTrims(p: EngineParams): EngineParams  // clampParams(p) with every trim neutral
```
- clampParams returns a deep copy: new trim arrays and objects, each field clamped to its range, missing / NaN /
  non-finite → 0, non-array → all neutral, padded or truncated to MAX_HARMONICS. `DEFAULT_PARAMS` channels have
  their own arrays. Code that copies a ChannelParams with a spread shares the trims array: copy it too
  (`harmonicTrims: c.harmonicTrims.map((t) => ({ ...t }))`) or go through clampParams.
- Trims act only in tonal mode; entries above `tonal.harmonics`, or whose h·f0 ≥ 0.45·fs (bands the bank skips),
  are ignored. All 0 → the channel is sample-identical to the untrimmed engine (every mode).
- Phase convention, tonal mode, at harmonic h·f0: total phase = polarity + h·Δ (through the wrapped delay
  D mod T) + θ_h, θ_h = harmonicTrims[h − 1].phaseDeg (lag positive, on top of the delay); gainDb scales that
  harmonic alone. Trims of different harmonics do not interact (the response at h·f0 is the untrimmed one × T_h
  exactly, to ~1e-12).
- Engine safety rule (as for gains): a harmonic's level is channel gain + master + its trim, and it rises by at
  most BIG_JUMP_DB + MAX_RISE_DB_PER_S·T over any interval T, whichever of them rises: the trims share the
  channel's GainRamp budget (TrimRamp). A trim gain increase larger than BIG_JUMP_DB rises at ≤ MAX_RISE_DB_PER_S
  (−30 → +6 dB takes 5 s, then the last 6 dB); decreases, phase changes and rises within the budget glide over
  COEFF_RAMP_MS.
On fs 44.1–192 kHz, f0 20–2000 Hz and every Q from the minimum to 200 the bank design converges; the exact
limit is ≈ 2.4 / 3.2 / 4.0 / 4.9 for 5 / 6 / 7 / 8 harmonics, so the table keeps ≥ 0.6 of margin. The UI's
Q control uses minTonalQ(harmonics) as its minimum.

### dc-blocker.ts
```ts
export const DC_BLOCKER_HZ = 3;
export class DcBlocker { constructor(fs: number, fc?: number); reset(): void; process(x: number): number }
export function dcBlockerResponse(fs: number, f: number, fc?: number): Complex  // (1 - z⁻¹)/(1 - R z⁻¹)
```

### bessel-lp.ts
```ts
/** Cascade of biquads approximating a Bessel low-pass whose -3 dB point is `cutoff`.
 *  order 2: one section, Q = 1/√3, f0 = 1.272·cutoff. order 4: two sections with the
 *  correct Bessel Q values and frequency factors, normalised so the cascade is -3 dB at `cutoff`. */
export function besselLowpassSections(fs: number, cutoff: number, order: 2 | 4): BiquadCoeffs[]
```

### fractional-delay.ts
```ts
/** Single-writer ring buffer with 3rd-order Lagrange interpolated reads. */
export class DelayLine {
  constructor(capacity: number)            // power of two (use delayCapacity(fs))
  write(x: number): void                    // call once per sample BEFORE reads of that sample
  /** x[n - d] with d ≥ D_MIN_SAMPLES (=1); uses taps x[n-k..n-k-3] with fractional part in [1,2). */
  read(delaySamples: number): number
  reset(): void
}
/** One channel's read tap: slew-limited for small changes, 30 ms crossfade between two read
 *  positions for jumps ≥ DELAY_JUMP_THRESHOLD_MS (or when setTarget(..., true)). */
export class SmoothDelayTap {
  constructor(fs: number, initialDelay?: number)
  setTarget(delaySamples: number, forceJump?: boolean): void
  next(line: DelayLine): number             // one output sample
  get current(): number                     // current effective delay (samples)
  get target(): number
  /** Largest delay this tap reads now or will move to: the read position, the crossfade 'from' position
   *  and the target (a read at delay d touches samples back to n − ⌊d⌋ − 2). */
  get reach(): number
  snap(): void                              // to the target, no slew/crossfade (output silent only)
}
```

### smoother.ts
```ts
export class OnePole { constructor(fs: number, timeMs: number, initial?: number); setTarget(v): void; set(v): void; next(): number; get value(): number }
/** Linear ramp to a target over a given time (soft-start 1.5 s, mute 15 ms, panic 10 ms). */
export class LinearEnvelope { constructor(fs: number, initial?: number); rampTo(target: number, ms: number): void; set(v: number): void; next(): number; get value(): number; get done(): boolean }
```

### gain-ramp.ts
```ts
export const RISE_FLOOR_DB = -60   // default free-rise level (see the phase-2 decisions)
/** Channel/master gain smoothing with the safety rule: decreases and increases ≤ BIG_JUMP_DB use a
 *  one-pole of GAIN_SMOOTH_MS; while (target - current) > BIG_JUMP_DB the gain rises in dB at no more
 *  than MAX_RISE_DB_PER_S. Works on dB internally, outputs linear gain; ≤ MASTER_MIN_DB → exactly 0.
 *  Rises below the rise floor are free (one-pole up to the floor). Every rise above it draws from a
 *  BIG_JUMP_DB budget refilled at MAX_RISE_DB_PER_S (a series of small steps cannot go faster). */
export class GainRamp { constructor(fs: number, initialDb: number); setTargetDb(db: number): void; jumpTo(db: number): void;
  /** holdRise: nobody hears this gain now (bypass, measurement gate) — increases are deferred (the budget
   *  still refills), decreases go on. */
  next(holdRise?: boolean): number;
  /** Free-rise level, default RISE_FLOOR_DB. The engine's channel ramp (channel gain + master) uses
   *  channel gain + RISE_FLOOR_DB: rises are free only while the master part is below −60 dB. */
  setRiseFloorDb(db: number): void;
  get currentDb(): number; get targetDb(): number; get currentLin(): number
  /** Joint budget with a TrimRamp (harmonic-trims.ts RiseBudget): the remaining fast-rise budget; the highest
   *  budget since the previous call (each sample's refill included), the record restarting from the current one;
   *  lowering it to `db` (a trim rise spent it) and restarting the record. One compare per sample. */
  get budgetDb(): number; takeBudgetPeak(): number; spendBudgetTo(db: number): void }
```

### limiter.ts
```ts
/** Zero-lookahead peak limiter: instant attack, LIMITER_RELEASE_MS release, then hard clip at the ceiling. */
export class PeakLimiter { constructor(fs: number, ceilingDb: number); setCeilingDb(db: number): void; process(x: number): number; get gainReductionDb(): number; reset(): void }
```

### hilbert-pair.ts
```ts
/** Olli Niemitalo's IIR 90° phase-difference pair. Path A (with the extra z⁻¹) a = 0.6923878,
 *  0.9360654322959, 0.9882295226860, 0.9987488452737; path B a = 0.4021921162426, 0.8561710882420,
 *  0.9722909545651, 0.9952884791278; each section out = a²(in + out[n-2]) - in[n-2]. B leads A by 90°. */
export class HilbertPair { process(x: number): void; get a(): number; get b(): number; reset(): void }
/** y = cos Δ·A − sin Δ·B (positive Δ = lag relative to A). */
export class PhaseRotator { constructor(); setDeltaDeg(deg: number): void; process(x: number): number; reset(): void }
export function hilbertResponse(f: number, fs: number): { a: Complex; b: Complex }
```

### resonator-bank.ts
```ts
/** Sum of constant-0-dB-peak band-pass biquads at h·f0 (h = 1..harmonics, skipping ≥ 0.45·fs),
 *  each normalised so the bank has 0 dB and 0° at every harmonic centre (correct for overlap).
 *  A Q below minTonalQ(harmonics) (params.ts TONAL_MIN_Q = [2,2,2,2,2,3,4,5,6], index = harmonics) is
 *  raised to it: below it the skirts overlap so much that no design is 0 dB / 0° at every harmonic.
 *  clampParams raises the tonal Q the same way, so the params report the Q actually used.
 *  configure(f0, harmonics, q, rampSamples = coeffRampSamples(fs)): running bands glide their coefficients,
 *  and the band gains glide on the same schedule. Bands that appear fade in from 0; bands that disappear
 *  (fewer harmonics, or a harmonic crossing 0.45·fs) fade out. With rampSamples ≤ 1, or a bank at rest
 *  (new or reset), the design applies at once. reset() also completes the glide. Unchanged parameters:
 *  configure() is still a no-op. */
export class ResonatorBank { constructor(fs: number); configure(f0: number, harmonics: number, q: number, rampSamples?: number): void; process(x: number): number; reset(): void
  /** Channel ch's harmonic trims (0 | 1; clamped, non-finite → 0). rampSamples > 1 (default COEFF_RAMP_MS): the
   *  weights glide along trimRampShape from where they are; the current target again is a no-op; ≤ 1 or a bank
   *  at rest: at once. Allocation-free. configure() re-solves both channels' weights for the new design and
   *  glides them with the bank. reset() also completes the weight glides. */
  setTrims(ch: number, trims: readonly HarmonicTrim[], rampSamples?: number): void
  trimGliding(ch: number): boolean
  /** After process(): channel ch's trimmed bank output Σ (g_i + u_i)·band_i + v_i·companion_i; exactly the
   *  return value (the neutral output) while ch's trims are all neutral. */
  readonly out: Float64Array /* 2 */ }
/** Neutral bank; with `trims` (one channel's harmonicTrims) that channel's trimmed bank, as ResonatorBank.out. */
export function resonatorBankResponse(fs: number, f: number, f0: number, harmonics: number, q: number, trims?: readonly HarmonicTrim[]): Complex
/** The same as a function of f, the trimmed bank's weights solved once (identical values). */
export function resonatorBankResponseAt(fs: number, f0: number, harmonics: number, q: number, trims?: readonly HarmonicTrim[]): (f: number) => Complex
```
`ResonatorBankDesign.quadCoeffs`: per band its quadrature companion, a first-order allpass run on the band's
OUTPUT (see harmonic-trims.ts; biquad form b0 = a1 = a, b1 = 1, b2 = a2 = 0). The companions run with the bands
(same glide, always in tonal mode) whether trims are set or not. A band whose trim weight is still gliding to 0
when a fade-out ends keeps running at gain 0 until the weight is 0. setTrims with a new target mid-glide restarts
the glide from the current point (a slope break): the engine hands trims to the live bank at most once per glide.

### harmonic-trims.ts
```ts
export const TRIM_WEIGHTS = 2 * MAX_HARMONICS     // weight layout [u_0…u_7, v_0…v_7]
export const trimRampShape = (u: number) => number  // smootherstep 6u⁵ − 15u⁴ + 10u³ (weight glides)
/** First-order allpass (a + z⁻¹)/(1 + a·z⁻¹), a = (t − 1)/(t + 1), t = tan(π·centre/fs), run on the band's output:
 *  90° lag at the centre, so companion·band-pass = −j·band-pass there and |companion·band-pass| = |band-pass| everywhere. */
export function quadratureCompanion(fs: number, centreHz: number): BiquadCoeffs
export interface TrimBasis { m: number; nRe: Float64Array; nIm: Float64Array; inv: Float64Array | null }
/** Response matrix at the harmonic centres (columns band_i, companion_i) and its 2m × 2m inverse (allocates). */
export function trimBasis(d: ResonatorBankDesign, fs: number): TrimBasis
/** Weights realising B(h·f0) = N_h·T_h at every used harmonic; allocation-free (inverse · vector). False = all 0. */
export function trimWeights(b: TrimBasis, trims: readonly HarmonicTrim[], out: Float64Array, r: Float64Array): boolean
export function trimmedBankResponseOf(d: ResonatorBankDesign, w: Float64Array | null, f: number, fs: number): Complex
export function anyTrim(trims: readonly HarmonicTrim[], m?: number): boolean
export function trimFactor(t: HarmonicTrim): Complex   // 10^(gainDb/20)·e^{−j·phaseDeg·π/180}, clamped
/** T = desired / current: the trim that turns `current` (the response at h·f0 with this trim at 0) into
 *  `desired`; gainDb = 20·log10|T|, phaseDeg = −arg T (lag positive), both clamped; neutral if 0/non-finite. */
export function trimForTarget(desired: Complex, current: Complex): HarmonicTrim
/** The gain in front of a channel's trims (GainRamp implements it). */
export interface RiseBudget { readonly budgetDb: number; takeBudgetPeak(): number; spendBudgetTo(db: number): void }
/** The engine's per-channel rate limiter (rules above). applied: the trims to hand to the chain (mutated in
 *  place). setTarget (false = unchanged), step (when due), jump (silent output; the gain jumped first). hold: rises
 *  wait. `gain`: the channel's GainRamp, whose budget the trims share; without it a budget of its own (clock). */
export class TrimRamp { constructor(fs: number, gain?: RiseBudget); readonly applied: HarmonicTrim[]; readonly stepSamples: number; get pending(): boolean;
  due(clock: number): boolean; jump(trims, clock): void; setTarget(trims, clock, hold): boolean; step(clock, hold): boolean
  /** Largest gain increase of any harmonic since setMark() (the engine marks each hand-over to the live chain). */
  riseSinceMark(): number; setMark(): void }
```
Joint budget: harmonic h's level budget is c_h = gain.budgetDb + extra_h. A gain rise spends the gain's budget, so
every c_h; a trim rise spends c_h and the gain's budget is lowered to the lowest c_h (a later gain rise raises that
harmonic too); extra_h ≥ 0 shrinks to BIG_JUMP_DB − (highest gain budget since the last advance), as every c_h refills
with the gain's and stops at BIG_JUMP_DB. Exact token-bucket accounting, allocation-free; so gain + master + trim at
h·f0 rise by ≤ BIG_JUMP_DB + MAX_RISE_DB_PER_S·T (before: 12 dB at once and 12 dB/s with both rising). The slow part
of a trim rise (gap > BIG_JUMP_DB) goes at MAX_RISE_DB_PER_S and draws from the budget, which refills as fast: the
last ≈ 6 dB then come as two steps (≈ 5.88 + 0.12 dB).
Structure (the file header has the reasoning): the band-passes stay shared and the neutral output Σ g_i·BP_i is
unchanged; each band's output also runs through its companion allpass AP_i, Q_i = AP_i·BP_i; channel ch's bank is
Σ (g_i + u_i)·BP_i + v_i·Q_i = Σ BP_i·(g_i + u_i + v_i·AP_i), with (u, v) = inverse(response matrix) · [N_h·(T_h − 1)].
The inverse is computed when the design changes (configure, message handler); applying trims is a matrix-vector
product (≤ 256 MACs), so no solve ever runs in process(). Over fs 22.05–192 kHz, f0 20–2000 Hz, harmonics 1–8 and Q
from minTonalQ to 200 the inverse exists, its largest entry is < 5 and the response at the centres is N_h·T_h to
< 1e-11. Skirts: |Q_i| = |BP_i| at every frequency, so a trim changes each band by |g + u + v·AP|/|g| ≈ ≤ √2·|T|: below
f0 and above the top harmonic a single trim raises the chain by ≤ |T| + 5 dB (4.8 dB worst, 4 harmonics at Q 2; ≤ 4 dB
at Q ≥ 6). The former companion (the band's poles with an RBJ low-pass numerator) was −j·BP at the centre too but flat
at ~1/Q below it: +14 dB at 20 Hz for +6 dB / 90° on the fundamental (4 harmonics, Q 30), +15 dB for a 180° phase-only
trim at Q 2: wind and rumble at the reference passed into the tonal chain. Broadband noise through a bank with the
same T on every harmonic carries |T|² × the neutral bank's power within 1 dB at Q 30 (within 3 dB with 8 harmonics at
Q 6, where the bands overlap most and AP's phase, −90° only at each centre, changes the overlap between the centres).
Trim glides (0 → +6 dB/90° on 50/100/400 Hz tones) leave −120 / −119 / −104.5 dB above 2 kHz.

### meters.ts
```ts
export class LevelMeter { constructor(fs: number, rmsMs?: number /*300*/, peakHoldMs?: number /*1500*/); push(x: number): void; get rmsDb(): number; get peakDb(): number; reset(): void }  // dBFS, floor -120
```

### tap-buffer.ts
```ts
/** Collects `channels` values per frame into planar Float32Array blocks of `frames` frames.
 *  push() returns a full block (ownership passes to the caller, who transfers it) or null.
 *  Buffers come from a pool refilled by recycle(); if the pool is empty the block is dropped (counted). */
export class TapCollector { constructor(channels: number, frames: number, poolSize?: number); push(values: Float64Array): Float32Array | null; recycle(buf: Float32Array): void; get dropped(): number; reset(): void }
```

### response.ts
```ts
/** The time-domain chain the engine runs. configure() may allocate; process() does not (per-sample and
 *  allocation-free, hence block-size invariant). On a running chain, configure() keeps the filter states
 *  and glides the active sections and the resonator bank over COEFF_RAMP_MS. A section that becomes enabled
 *  starts from a clean state with its coefficients at once, and the bank is reset when tonal mode starts.
 *  reset() completes all glides, so configure() + reset() equals a new chain sample for sample. Repeated
 *  configure() calls with the same filter parameters do not disturb a running glide. */
export class ConditioningChain { constructor(fs: number, params?: EngineParams); configure(p: EngineParams): void; process(x: number): number; reset(): void
  /** Harmonic trims of channel ch (configure() ignores ChannelParams.harmonicTrims; kept across configure()
   *  and mode changes). Glide over COEFF_RAMP_MS by default, ≤ 1: at once. */
  setTrims(ch: 0 | 1, trims: readonly HarmonicTrim[], rampSamples?: number): void
  /** After process(): each channel's conditioning. process() returns the NEUTRAL output (the 'cond' tap);
   *  channelOut[ch] is the same value outside tonal mode or with neutral trims, else the trimmed bank's. */
  readonly channelOut: Float64Array /* 2 */ }
/** Conditioning chain (DC blocker → HP if enabled → barrier LP if enabled → Bessel LP if enabled →
 *  resonator bank if phaseMode === 'tonal'), exactly as the engine implements it (at rest, i.e. the
 *  response of the glide's target). NEUTRAL: harmonic trims are not included (the 'cond' tap). */
export function conditioningResponse(p: EngineParams, fs: number, f: number): Complex
/** The conditioning channel ch gets: in tonal mode the bank is ch's trimmed bank (exactly as the engine, also
 *  between the harmonics); at a used harmonic it is conditioningResponse × trimFactor(harmonicTrims[h − 1]).
 *  Equal to conditioningResponse outside tonal mode or with neutral trims. */
export function channelConditioningResponse(p: EngineParams, fs: number, f: number, ch: 0 | 1): Complex
/** Full channel response from reference input to final output (before limiter, assuming no limiting):
 *  channelConditioningResponse (trims included) · e^{-jω·computeChannelDelay(p, fs, ch).samples} · (hilbert
 *  rotation if mode hilbert, i.e. cosΔ·A − sinΔ·B) · (−1 if polarityInvert) · channel gain · master gain.
 *  0 if channel disabled. The trims' target values (the engine reaches a big rise at 6 dB/s). */
export function channelResponse(p: EngineParams, fs: number, f: number, ch: 0 | 1): Complex
/** channelResponse of one channel as a function of f: p clamped, the conditioning designed, the trimmed bank's
 *  weights solved and the delay computed once (identical values; the loop checks evaluate one set of params at
 *  hundreds of frequencies, and channelResponse solved the bank's weights at every one). */
export function channelResponseAt(p: EngineParams, fs: number, ch: 0 | 1): (f: number) => Complex
/** Group delay in samples of an arbitrary response function, by central difference of unwrapped phase. */
export function groupDelaySamples(h: (f: number) => Complex, f: number, fs: number): number
```

### fft.ts
```ts
export const nextPow2 = (n: number) => number
export class FFT { constructor(n: number /*power of 2*/); forward(re: Float64Array, im: Float64Array): void; inverse(re: Float64Array, im: Float64Array): void /* scaled 1/n */ }
```

### sweep.ts
```ts
export interface SweepSpec { fs: number; f1: number; f2: number; durationS: number; fadeMs: number; amplitude: number }
/** Exponential (Farina) sine sweep with raised-cosine fades, band f1..f2, peak `amplitude`. */
export function exponentialSweep(spec: SweepSpec): Float32Array
/** Impulse response of the system that turned `stimulus` into `recorded`, by regularised spectral
 *  division restricted to [f1, f2]; length = nextPow2(recorded.length + stimulus.length). */
export function deconvolve(recorded: Float32Array, stimulus: Float32Array, fs: number, f1: number, f2: number): Float64Array
/** Band-limited pink noise (for the amplifier-level step), seeded. */
export function pinkNoise(fs: number, durationS: number, f1: number, f2: number, amplitude: number, seed?: number): Float32Array
```

### latency-estimator.ts
```ts
export interface DelayEstimate {
  delaySamples: number   // first arrival, sub-sample
  sign: 1 | -1           // polarity of the first arrival
  peak: number           // signed IR value at the arrival
  qualityDb: number      // arrival peak vs pre-arrival noise floor (excluding main lobe)
  ok: boolean
  reason?: string        // Italian, shown to the user when !ok
}
/** First arrival = earliest local peak within -6 dB of the global |IR| maximum; parabolic (or better)
 *  sub-sample refinement; acceptance: acoustic qualityDb ≥ 10, loopback ≥ 18. */
export function analyzeImpulseResponse(ir: Float64Array, fs: number, mode: 'acoustic' | 'loopback'): DelayEstimate
export function estimateDelay(recorded: Float32Array, stimulus: Float32Array, fs: number, f1: number, f2: number, mode: 'acoustic' | 'loopback'): DelayEstimate & { ir: Float64Array }
/** Median of repeated estimates; ok=false if spread (std) > 0.5 samples or any sign disagrees. */
export function combineEstimates(list: DelayEstimate[]): DelayEstimate & { stdSamples: number; count: number }
/** Complex transfer function sampled at `freqs` from an impulse response. */
export function irToResponse(ir: Float64Array, fs: number, freqs: Float64Array): { re: Float64Array; im: Float64Array }
```

### howl-detector.ts
```ts
export interface HowlState { triggered: boolean; reason: HowlReason | null; freqHz: number | null }
export const HOWL_TARGET_RATE = 12000, HOWL_FFT_SIZE = 512, HOWL_PEAK_DB = 12, HOWL_GROWTH_DB_PER_S = 10,
  HOWL_GROWTH_MS = 300, HOWL_MIN_LEVEL_DBFS = -40, HOWL_LIMITER_GR_DB = 3, HOWL_LIMITER_MS = 300,
  HOWL_OVERLOAD_MARGIN_DB = 4, HOWL_OVERLOAD_MS = 200,
  HOWL_LOOP_MIN_MS = 5, HOWL_LOOP_MAX_MS = 600, HOWL_LONG_LOOP_MAX_MS = 1100, HOWL_PERIODIC_RHO = 0.8,
  HOWL_GROWTH_RHO = 0.3, HOWL_GROWTH_RHO_DROP = 0.01, HOWL_PERIODIC_EVERY_MS = 200, HOWL_PERIODIC_VALID_MS = 600,
  HOWL_SAT_GAP_MS = 200, HOWL_SAT_MS = 300, HOWL_SAT_LEVEL_DB = 12, HOWL_SAT_GR_DB = 0.5, HOWL_WIDE_BAND_HZ = 2000,
  HOWL_NARROW_BAND_HZ = 1600;
/** Upper edge of the band the conditioning passes (clamped params), for HowlDetector.setBandHz: the Bessel LP
 *  cutoff when enabled; in tonal mode at most harmonics·f0·(1 + 1/(2Q)) (upper −3 dB edge of the top band);
 *  Infinity when nothing bounds it (HP and barrier only shape the low end). */
export function howlBandHz(p: EngineParams): number
/** Size, lag range and pace of a PeriodicityAnalyzer (defaults: 8192 points, lags HOWL_LOOP_MIN_MS…
 *  HOWL_LOOP_MAX_MS, capped at half the FFT size, each analysis spread over 60 ms). */
export interface PeriodicityOptions { log2N?: number; lagMinMs?: number; lagMaxMs?: number; spreadMs?: number;
  /** |ρ(τ)| = min(|c(τ) + j·Hilbert(c)(τ)| / √(Σa²·Σb²(τ)), √share(τ)): the envelope of the correlation c (the
   *  analytic spectrum 2·C for 0<k<N/2, 0 above, through the same inverse FFT; same cost). share(τ) = the part of
   *  the window's energy whose lagged samples exist (1 once the history is full): the Cauchy–Schwarz bound.
   *  Without the cap the envelope reaches ~1 at the edge of a history that is still filling. A band-pass howl's
   *  correlation oscillates at its centre f: on the lag grid its peak can fall half a sample from the loop
   *  delay, cos(π·f/rate) below the envelope (0.63 at 3.4 kHz / 12 kHz). Default false. */
  envelope?: boolean }
/** Largest normalized autocorrelation |ρ(τ)| = |Σx[n]x[n−τ]| / √(Σx[n]²·Σx[n−τ]²) at a local maximum over
 *  lags 5–600 ms (default options). The sums run over the latest window: FFT size minus the longest lag,
 *  ≈ 0.77 s at 6 kHz with 8192 points. A new analysis starts every everyMs. Its work (copy, a DIF FFT of the
 *  two real sequences packed into one complex one, cross-spectrum, inverse DIT FFT, lag search) is cut into
 *  slices of `budget` units per push(), so each analysis completes in ≈ spreadMs (60 ms). push() is
 *  allocation-free. FFT passes are radix 2²: two radix-2 stages per pass over the arrays, the same butterflies,
 *  so results are bit-identical. A slice may run up to 3 work units over its budget; the next slice is that much
 *  shorter (analyses still complete on the same push as before).
 *  Typical |ρ|: howl with loop L ≈ 0.86–0.97 at L (the sign follows the loop polarity); broadband noise ≲ 0.2;
 *  tones ≈ 1. */
export class PeriodicityAnalyzer {
  constructor(rate: number, everyMs?: number /* 200 */, options?: PeriodicityOptions)
  readonly rate: number; readonly everySamples: number; readonly lagMinSamples: number;
  readonly lagMaxSamples: number; readonly windowSamples: number; readonly budget: number
  push(x: number): void
  get rho(): number          // latest completed analysis, 0 before the first
  get previousRho(): number  // the analysis before it
  get lagMs(): number
  get ageMs(): number        // since the end of the analysed window (Infinity before the first)
  readonly envelope: boolean
  get analyses(): number     // analyses completed since construction / reset()
  reset(): void
}
/** Pre-limiter output of one channel. Decimates to ~12 kHz (round(fs/12000)), 512-point Hann FFT every 256
 *  decimated samples. A PeriodicityAnalyzer runs at half that rate (~6 kHz, band ≤ ~2.4 kHz, after a
 *  4th-order Butterworth anti-alias filter), and a second one for long loops at half that again (~3 kHz,
 *  band ≤ ~1.2 kHz, same anti-alias): 8192 points, lags 500–1100 ms (the 500 ms channel delay plus up to
 *  600 ms of latency and acoustic delay), a new analysis every 400 ms spread over 120 ms.
 *  Main analysis rate (setBandHz):
 *  - at ~6 kHz (band ≤ ~2.4 kHz, 8192 points) as before, until setBandHz reports a band above
 *    HOWL_WIDE_BAND_HZ (NaN or Infinity count as unbounded);
 *  - then at the decimated rate (~12 kHz, band ≤ ~4.8 kHz behind the main anti-alias filter): 16384 points, the
 *    same 5–600 ms lags and ~0.77 s window, envelope ρ;
 *  - back at ~6 kHz only when the band is ≤ HOWL_NARROW_BAND_HZ; in between the rate is unchanged (hysteresis
 *    for a tracked f0 or a dragged LP);
 *  - at fs whose decimated rate is < 9 kHz there is nothing to switch;
 *  - the ~6 kHz stream and the long-loop analysis (500–1100 ms at ~3 kHz, ≤ 1.2 kHz) run in both modes.
 *  A rate change drops the main analysis' history and results at both rates (no main periodicity until the
 *  first analysis at the new rate, ~260 ms); the band-level 'peak-growth' rule needs the second analysis at the
 *  new rate (its 'not falling' test compares two analyses of the same path); level tracks and the long-loop
 *  analysis are kept; reset() keeps the rate (a property of the chain) and clears the wait. setBandHz is for
 *  the message handler only; push() never changes the rate and still allocates nothing.
 *  Typical |ρ| of broadband noise at ~12 kHz: ≲ 0.08 (0.06 measured with LP 4 kHz / LP off).
 *  Triggers (latched until reset()):
 *  - 'peak-growth': a level grows faster than the commanded gain explains: a run of 5-frame slopes
 *    > 10 dB/s lasting ≥ 300 ms; least-squares slope over the run > 10 dB/s and ≥ 6 standard errors; the
 *    level ends ≥ 3 dB above anything seen before the run (2.7 s history); above −40 dBFS. The level is
 *    either
 *    (i) a narrow peak (≥ 12 dB above the median spectrum and its ±16-bin neighbourhood; freqHz = the
 *        peak), or
 *    (ii) the total band power below ~4.8 kHz, averaged over the last 5 frames (~110 ms), divided by
 *        the commanded gain. Extra conditions:
 *        - the run lasts ≥ 300 ms + 4 hops (~385 ms): the average stretches any rise by 4 hops, so a
 *          source switching on in ≤ ~250 ms cannot pass for a 300 ms growth;
 *        - the latest short slope is ≥ 0.4 × the run's slope (a howl grows exponentially; a source
 *          switching on decelerates);
 *        - the latest periodicity analysis has a window ending ≥ 100 ms into the run, |ρ| ≥ 0.3, and ρ no
 *          more than 0.01 below the analysis before (what grows is the periodic part);
 *        - freqHz = the narrow peak if one is present, else the strongest bin.
 *  - 'limiter-saturated':
 *    (i) limiter GR > 3 dB for > 300 ms with a persistent narrow peak, or
 *    (ii) a periodic signal held at the ceiling: all of
 *        - the latest analysis, or the latest long-loop analysis (window ending ≤ 600 ms ago), found
 *          |ρ| ≥ 0.8;
 *        - pre-limiter samples have exceeded the ceiling at most 200 ms apart for > 300 ms;
 *        - the pre-limiter RMS (100 ms one-pole) is within 12 dB of the ceiling;
 *        - the limiter GR (50 ms one-pole) is ≥ 0.5 dB (a howl held at the ceiling needs 20·log10(G) dB:
 *          0.8 dB at loop gain 1.1; a hum whose peaks barely touch the ceiling plays);
 *        - freqHz = the narrow peak, else the strongest bin, else null.
 *        A long loop's howl reaches the ceiling unevenly (the loudest part of each trip first): it is held,
 *        and muted, within about two trips of the first gain reduction.
 *  - 'output-overload': RMS of the limited output (25 ms) > ceiling − 4 dB for > 200 ms. It waits while a
 *    narrow peak or a periodic signal is driving the limiter (≤ 300 ms), so a howl is reported as such.
 *  Must NOT trigger on a steady tone during soft-start/gain ramps (the commanded gain explains the growth),
 *  on broadband noise (stationary, stepping, fading in, rising smoothly), beating tones, a tone switching on,
 *  impulsive bursts or speech-like noise.
 *  Accepted false positives: (1) a tonal source that rises fast (e.g. a 1 s linear fade-in); (2) a periodic
 *  source (tone or machine) pushed into the limiter by ≥ 0.5 dB, as defined in 'limiter-saturated' (ii).
 *  Not seen by the periodicity: howls above ~2.4 kHz with no narrow peak when the chain's band is ≤ 2 kHz (above
 *  ~4.8 kHz otherwise); above ~1.2 kHz for loops of 600–1100 ms; loops longer than 1.1 s; the engine backstop
 *  covers gross cases of these. push()
 *  allocates nothing; its extra work in any one sample is bounded by the analyzers' slices.
 *  holdLevels(ms): the conditioning filters glide to a new response for the next `ms` (a live filter
 *  change, engine): those samples are recorded with an unknown commanded gain, so the frames that contain
 *  them are unreliable and end every level track (they restart on the new response, as after a
 *  commanded-gain jump); the full-rate rules and the periodicity are not affected. */
export class HowlDetector { constructor(fs: number); push(preLimiter: number, commandedGainDb: number, limiterGrDb: number, ceilingDb: number): void;
  holdLevels(ms: number): void; get state(): HowlState; reset(): void
  setBandHz(hz: number): void; get wideBand(): boolean }
```

### f0-tracker.ts
```ts
export interface F0Estimate { f0: number; confidence: number /*0..1*/; dominanceDb: number /* peak vs median spectrum */ }
/** Sub-bin f0 estimate (long windowed FFT + interpolation and/or phase-advance refinement, target
 *  ~0.01 % on a clean stable tone of ≥ 1 s), searching fmin..fmax, harmonic-aware (prefers the
 *  fundamental over its 2nd harmonic). Returns null if no tonal component. */
export function estimateF0(x: Float32Array | Float64Array, fs: number, fmin: number, fmax: number): F0Estimate | null
/** Keeps the last `windowS` seconds of pushed blocks; estimate() runs estimateF0 on them. */
export class F0Tracker { constructor(fs: number, windowS?: number, fmin?: number, fmax?: number); push(block: Float32Array): void; estimate(): F0Estimate | null; reset(): void }
/** Phase error in degrees at harmonic h caused by an f0 error df when the total delay is delaySeconds. */
export function driftPhaseErrorDeg(dfHz: number, harmonic: number, delaySeconds: number): number
```

### feedback-canceller.ts
```ts
/** Feedback neutralisation: x̂ = x − Σ_ch Σ_k h_ch[k]·y_ch[n − bulk_ch − k], y = final output samples.
 *  Allocation-free per sample. */
export class FeedbackCanceller {
  constructor(fs: number, maxBulkDelay: number, maxTaps: number)
  setFir(ch: 0 | 1, bulkDelay: number, taps: Float64Array | null): void
  /** Prediction of the feedback component in the current reference sample (call before computing outputs). */
  predict(): number
  /** Push this sample's final outputs (call after computing outputs). */
  pushOutputs(yL: number, yR: number): void
  reset(): void
}
/** Builds the FIR from a measured speaker→reference impulse response: bulkDelay = first arrival − margin,
 *  taps = IR window of `taps` length with a short fade-out. */
export function designFeedbackFir(ir: Float64Array, arrivalSamples: number, taps: number): { bulkDelay: number; taps: Float64Array }
```

### alignment.ts
```ts
export interface FreqResponse { freqs: Float64Array; re: Float64Array; im: Float64Array }
export type SpeakerConfig = 'single' | 'shared' | 'separate'
export interface AlignmentInput {
  fs: number
  band: [number, number]
  /** Primary path reference→zone (measured with the error mic, or geometric: pure delay·gain). */
  primary: FreqResponse
  /** Secondary path final output→zone per channel (null = channel unused). */
  secondary: [FreqResponse | null, FreqResponse | null]
  /** Conditioning response H_cond(f) (no delay/gain/polarity), from response.ts. */
  conditioning: (f: number) => Complex
  speakers: SpeakerConfig
  /** Optional weights per frequency (e.g. noise PSD); default flat in band. */
  weights?: Float64Array
  /** In tonal mode only the harmonics of f0 matter and delay is free modulo 1/f0. */
  tonalF0?: number
  /** Allowed delay range in samples for the tap (≥ 0 above D_MIN). */
  maxDelaySamples: number
}
export interface ChannelSolution { use: boolean; delaySamples: number /* total tap delay incl. D_MIN */; gainLin: number /* total linear gain incl. master */; polarityInvert: boolean }
export interface AlignmentResult {
  channels: [ChannelSolution, ChannelSolution]
  predictedAttenuationDb: number        // weighted, in band
  causal: boolean                       // false if the best broadband delay would be < D_MIN
  requiredExtraLeadSeconds: number      // how much earlier the noise must arrive to be causal (0 if causal)
}
/** Minimises the weighted residual |P − Σ g_ch·H_cond·S_ch·e^{-jωD_ch}·(±1)|² in band.
 *  'shared': joint solution (both speakers together cancel P once, default equal split);
 *  'separate': each channel solved against P independently (caller measures each zone separately);
 *  'single': only channel 0. */
export function solveAlignment(input: AlignmentInput): AlignmentResult
/** Open-loop gain |Σ_ch G_ch(f)·F_ch(f)| in dB across band (G = channelResponse), for the howl margin. */
export function loopGainDb(channelResp: [(f: number) => Complex, (f: number) => Complex], feedback: [FreqResponse | null, FreqResponse | null], band: [number, number]): { maxDb: number; atHz: number }
export function residualDb(input: AlignmentInput, channels: [ChannelSolution, ChannelSolution]): number
/** Bounds of |T| of harmonicFactors (the trims' gain range as linear amplitudes). */
export interface FactorBounds { min: number; max: number }
/** Per-harmonic factors T_ch that null p + Σ T_ch·b_ch: one speaker or separate zones T = −p/b; two on one zone
 *  ('shared') the minimum-norm change δ = −r·conj(b)/Σ|b|² (r = p + Σ b); the magnitude is clamped to `bounds`
 *  (relative tolerance 1e-9: a factor on a bound is no clamp) and, when only one channel is clamped, the other is
 *  re-solved against p plus the clamped contribution. capped[ch]: that channel's factor was clamped. */
export function harmonicFactors(p: Complex, b: [Complex | null, Complex | null], speakers: SpeakerConfig, bounds: FactorBounds): { t: [Complex | null, Complex | null]; capped: [boolean, boolean] }
```

### physics.ts
```ts
export const speedOfSound = (tempC: number) => number            // 331.3·√(1 + T/273.15)
export const attenuationDb = (gainRatio: number, phaseErrRad: number) => number   // 10·log10(1+g²−2g·cosΔφ)
export const phaseToleranceDeg = (targetDb: number) => number    // -20 → 5.73
export const timeToleranceSeconds = (targetDb: number, f: number) => number
/** Acoustic head start of the noise at the zone relative to the reference mic:
 *  (d_src→zone − d_src→mic)/c; for a far source at angle θ from the mic→zone line ≈ d_mic→zone·cosθ/c. */
export function headStartSeconds(g: { micToZoneM: number; sourceAngleDeg: number; sourceDistanceM?: number; tempC: number }): number
export interface GeometryDelay { delaySeconds: number; causal: boolean; requiredExtraLeadM: number; cosTheta: number; warnAngle: boolean /* cosθ < 0.7 */ }
/** Delay the tap must add: head start − τ_S(secondary: latency + speaker→zone) − conditioning group delay. */
export function geometryDelay(headStartS: number, secondaryDelayS: number, conditioningGroupDelayS: number, tempC: number): GeometryDelay
export const minMicLeadMeters = (latencySeconds: number, tempC?: number) => number
```

## Views (`src/ui/`) — pure DOM components, no audio access
Each exports a factory `createX(container: HTMLElement, opts): { update(...): void; destroy(): void }`.
They receive data and emit callbacks; wiring to the audio engine happens in `src/main.ts` (later).
See the agent prompt for the exact list.
- `device-panel.ts`: `permission` is 'granted' only once the mic is open (main.ts maps it); the "accesso negato"
  advice only for 'denied' (never 'prompt'/'unknown'); short select placeholders ('— consenti il microfono —',
  '— scegli —', '— nessun ingresso —' / '— nessuna uscita —'); both lists empty with the mic open → "Nessun
  dispositivo audio trovato…"; before the mic is open the input-2 line says it is available after the permission.
  Pure helpers `permissionText`, `devicePlaceholder`, `ch2Text(ch2, micOpen, trackChannels?, errorChannel?)`
  (names the input the engine checks: 1 when errorChannel = 0, else 2).
- `meters-view.ts`: `attenuationReport(errAvailable, dB, refAgeS?)`; |A| < `ATTENUATION_NO_CHANGE_DB` (0.05 dB) is "no
  change" (state 'flat', "0,0 dB", never a negative zero), 'worse' only from −0.05 dB down; `stale` (and a note
  "premi B per aggiornarlo") when the bypass reference is older than `ATTENUATION_STALE_S` (60 s).
- `status-bar.ts`: `StatusBarData.masterLocked` disables the master control while a calibration stimulus plays
  (the engine would abort it on a master change).
- `filters-panel.ts`: the resonator Q control's minimum is `minTonalQ(harmonics)`, with the hint
  "Con N armoniche il Q minimo è X: sotto, le bande si sovrappongono troppo." (`qHint`).
- `channel-strip.ts` «Armoniche» (tonal mode only, a collapsible section): `ChannelStripInfo.harmonics`; rows
  h ∈ units.ts `bankHarmonics(harmonics, f0, fs)` (rows above 0.45·fs hidden, with a note); trim controls
  'Guadagno armonica h' (−30…+6 dB) / 'Fase armonica h' (±180°) whose reset value is
  calibratedChannels[ch].harmonicTrims[h − 1], or 0 ('nessuna correzione'); «Fase totale» per row = units.ts
  `harmonicTotalPhaseDeg` (polarity + h·Δ + θ_h), and the strip's main «Fase totale … @ f0» readout includes θ_1 in
  tonal mode (one total phase per frequency); edits → onChange(next, 'harmonicTrims') with a deep copy of all 8
  trims; «Azzera armoniche» → `ChannelStripOptions.onClearTrims` (main.ts: controller.clearHarmonicTrims; without it
  the strip sends all-zero trims to onChange), its aria-label naming both channels while linked; then it is enabled
  while either channel has trims (`ChannelStripInfo.otherTrimmed`, main.ts: anyTrim of the other channel), unlinked
  while this one has. Only what changed is rewritten at meter rate.
- `scope-view.ts`: `ScopeViewData.trimmed` (main.ts `trimmedChannels(p)`: tonal mode and a trim on a harmonic in
  use): for the channel shown the legend reads `EXPECTED_LEGEND_TRIMMED` and `TRIMS_NOTE` says that the expected trace
  s·g·cond[n − D] leaves the trims out.
- `precision-control.ts`: `PrecisionControlPatch.resetLabel` (what double-click / ↺ returns to, in the title).

---

# Integration contracts (phase 2)

Decisions taken after phase 1:
- `GainRamp`: rises that start below `RISE_FLOOR_DB = -60` jump (one-pole) to -60 dB and are rate-limited only
  above it (otherwise un-muting the master from -∞ would take 14 s, 10 s of which inaudible). Silence still
  comes from the mute/soft-start envelopes, not from the master at -∞.
- Howl detection timing, checked in the closed-loop simulator (`tests/sim/howl.test.ts`): loop gain 1.5 with a
  3–100 ms loop → howl event and mute ≤ 1 s from closing the loop; any loop gain 1.1–2 with a loop up to 500 ms →
  ≤ 0.4 s after the limiter starts reducing gain in a sustained way (> 0.5 dB, or > 3 dB, in ≥ 80 % of a 200 ms
  window); the output never above the ceiling meanwhile (the limiter holds it); no trigger in 10 s of legitimate
  signals. (The plan's "≤ 300 ms" is not met: a growth run alone lasts HOWL_GROWTH_MS = 300 ms.)
  (d) Wide chains (howl-band): chain LP 4 kHz or LP off; speaker → reference path passing only 2.7–4.3 kHz (41-tap
  FIR), loop peaks at 3.3 / 3.5 kHz, anti-noise of a low-frequency source 25 dB below the ceiling. Loop gain 1.2 /
  1.5 with 12.5 / 50 / 100 / 200 ms loops: muted ≤ 344 ms after sustained limiting; for G 1.5 with loops ≤ 100 ms,
  ≤ 877 ms from closing the loop; output never above the ceiling. With the ~6 kHz analysis 11/16 stayed pinned and
  5/16 were muted 0.55–4.9 s after sustained limiting. Cost (Node 24, 48 kHz, 128-frame quanta, both channels): wide
  chain +0.31…0.33 % RT, default chain −0.12 % RT, +0.07 ms per quantum at p99.9, +~0.7 MB per detector
  (preallocated).
- `pinkNoise(amplitude)` is a PEAK value (RMS ≈ amplitude/4): "-20 dBFS RMS" means amplitude ≈ 0.4.
- Device keys do not include groupId (regenerated per document); groupId is only used within a session.
- Calibration data stored in presets must be plain JSON (number arrays, no typed arrays, no NaN).
- Structural chain changes (filter section enabled/disabled/order, phase mode, polarity flip) go through a
  15 ms per-channel "reconfigure" envelope: ramp to 0, swap in a chain configured in the message handler
  (configure() may allocate — never call it inside process()), ramp back. Coefficient-only changes within
  LIVE_FREQ_RATIO = 2 (HP, barrier and Bessel cutoffs), LIVE_Q_RATIO = 3 (tonal Q) and LIVE_F0_RATIO = 1.03
  (tonal f0) call configure() on the live chain, which glides over COEFF_RAMP_MS = 20 ms. Bigger jumps and
  structural changes use the 15 ms reconfigure envelope.
- Transient metric ('click'): input is a pure tone at 50, 100 or 400 Hz. The output is high-passed at 2 kHz
  (zero-phase, 8th order). Click = the peak over [−5 ms, +100 ms] around the change, in dB relative to the
  input peak. Live changes measure ≤ −100 dB, against −19 to −77 dB with the former abrupt switching.

## src/dsp/antiphase-engine.ts (pure, used by the worklet, tests and the simulator)
```ts
export interface EngineOptions { softStartMs?: number /*SOFT_START_MS*/; meterHz?: number /*30*/;
  /** Tests only: bypass DC blocker/HP/LP/resonators (cond = raw reference). */ testBypassConditioning?: boolean }
export class AntiPhaseEngine {
  constructor(fs: number, opts?: EngineOptions)
  handleMessage(msg: WorkletInMsg): void           // all control; may allocate
  /** in0/in1 = input channels (null when absent); outL/outR same length n (block length is n, never assume 128). */
  process(in0: Float32Array | null, in1: Float32Array | null, outL: Float32Array, outR: Float32Array): void
  /** Hands queued outgoing messages to `post` (with transferable buffers), then clears the queue. */
  drain(post: (msg: WorkletOutMsg, transfer?: ArrayBuffer[]) => void): void
  get params(): EngineParams
}
```
Per sample: x = in[refChannel] (0 if absent); x̂ = x − feedback.predict() (if an FIR is set);
cond = ConditioningChain(x̂) (bypassed in test mode; the neutral chain); lines[ch].write(chain.channelOut[ch])
for both channels (= cond outside tonal mode or with neutral trims, the channel's trimmed bank otherwise; one
DelayLine(delayCapacity(fs)) per channel, written in lockstep: 1 MiB each at 192 kHz);
per channel ch: v = SmoothDelayTap.next(lines[ch]) → [PhaseRotator if mode 'hilbert'] → polarity (±1) →
× (channel gain + master) ramp → × softStart × mute × bypass × reconfigure(ch) × channelSoftStart(ch) ×
measureGate → + stimulus × master (stimulus only on the measured channel; the master GainRamp scales only
the stimulus) → pre = v → howl[ch].push(pre, commandedGainDb, limiter[ch].gainReductionDb, ceilingDb)
→ y = limiter[ch].process(v) (limiter does the final clip) → out = armed ? y : 0.
Then feedback.pushOutputs(outL, outR); meters (raw x, err, outL, outR); tap values [x, cond, outL, outR, err]
('cond' is always the neutral chain output: harmonic trims do not show in it).
Rules:
- `arm` false → outputs exactly 0 (meters/taps/ch2 check keep running). `arm` true → soft-start from 0.
- `mute` true → 15 ms ramp down; `mute` false → soft-start from 0 (1.5 s). `bypass` → 15 ms ramp either way.
- `panic` → 10 ms ramp to 0, muted = true; leaving it requires `mute` false (→ soft-start).
- `measure` requires armed (else immediate measureResult aborted=true): measureGate ramps the anti-noise to 0
  in 15 ms, then the stimulus is played on `channel` (through master and limiter, not through mute/soft-start)
  and raw x and err are recorded from the first stimulus sample for stimulus.length + postRollFrames frames
  (sample-aligned by construction); then measureResult (buffers transferred) and the gate ramps back.
  measureProgress ~10/s. Howl detectors are paused and reset during measurement. `abortMeasure` stops at once.
- commandedGainDb = the (channel gain + master) ramp's current dB + 20·log10(envelope product) (floored at -120).
- On howl trigger (any channel): same as panic, then emit `howl` and `autoMute` (Italian reason). `resetHowl`
  resets detectors.
- Howl band hook: the constructor and setParams (right after `this.target = p`, before the armed check) run
  `for (const h of this.howl) h.setBandHz(howlBandHz(p))`: the detectors' analysis rate follows the target params
  at message time, even while a structural swap is pending (testBypassConditioning is ignored: they follow the
  params' band).
- Delay: computeChannelDelay(params, fs, ch); setTarget(..., jump) when |Δ| ≥ DELAY_JUMP_THRESHOLD_MS, in tonal
  mode also when |Δ| > T/2 (a D mod T wrap: slewed, it would sweep the phase the long way round, through
  in-phase), or on mode change; MeterSnapshot.delaySamples/delayClamped report it.
- `MeterSnapshot.masterDb` (src/audio/protocol.ts): the master applied right now, dB — the master GainRamp after
  smoothing, rise limiting (6 dB/s above BIG_JUMP_DB) and the stimulus override (it jumps to the target when a
  stimulus starts), i.e. what a stimulus sample is multiplied by now (the anti-noise gets the master inside the
  channel ramps, same rules); MASTER_MIN_DB when the master is silent; disarmed, the target at once. The
  wizard's step 2 can wait on it instead of a time model of the ramp.
- ch2 check every ~1 s: in1 null → 'absent' (mono track); rms(error input: in0 if errorChannel = 0, else in1)
  < -90 dBFS → 'silent'; corr(in0,in1) > 0.999 with RMS within 0.1 dB → 'duplicate'; else 'ok'. Post
  `channelCheck` on change. The UI names the input the check refers to (validation.ts errorMicInput: input 1
  when errorChannel = 0, else input 2; 'duplicate' names both).
- Meters at meterHz; taps (TapCollector, TAP_BLOCK_FRAMES) only while enabled; `recycle` returns buffers.
- Never throw from process(); NaN anywhere → that sample 0 and states reset.

## src/worklet/antiphase.worklet.ts
Thin adapter: `registerProcessor(PROCESSOR_NAME, class extends AudioWorkletProcessor { ... })`, engine =
new AntiPhaseEngine(sampleRate); port.onmessage → engine.handleMessage; process(inputs, outputs): passes
inputs[0]?.[0] ?? null, inputs[0]?.[1] ?? null, outputs[0][0], outputs[0][1] ?? scratch; engine.drain(port.postMessage);
try/catch around everything → zeros + `fault` + muted. Posts `ready` once. No `self`, `window`, `import.meta`.

## src/audio/audio-engine.ts (browser)
```ts
export type AudioEngineEvent = WorkletOutMsg | { type: 'lifecycle'; reason: string; autoMuted: boolean } | { type: 'stats'; stats: AudioStats }
export interface AudioStats { sampleRate: number; renderQuantum: number; baseLatencyMs: number | null; outputLatencyMs: number | null;
  trackLatencyMs: number | null; underruns: number | null; trackSettings: MediaTrackSettings | null; sinkId: string | null }
export class AudioEngine {
  constructor(opts?: { latencyMode?: 'normal' | 'ultra' })
  /** MUST be called synchronously inside a click handler: creates/resumes the AudioContext before any await. */
  openMic(inputDeviceId: string | null, opts: { wantErrorMic: boolean }): Promise<void>
  setOutput(sinkId: string): Promise<void>              // setSinkId, or recreate ctx with {sinkId} if needed
  ensureContextMatchesTrack(): Promise<boolean>          // recreate ctx if sampleRate/sinkId mismatch (true if recreated)
  send(msg: WorkletInMsg, transfer?: Transferable[]): void
  measure(stimulus: Float32Array, channel: 0 | 1, postRollFrames: number): Promise<{ ref: Float32Array; err: Float32Array | null; masterLin: number; aborted: boolean }>
  on(cb: (e: AudioEngineEvent) => void): () => void
  get stats(): AudioStats
  get state(): 'idle' | 'mic' | 'error'
  close(): Promise<void>
}
```
Lifecycle → automatic `mute` + `lifecycle` event: ctx state 'suspended'/'interrupted'/'closed' while running, ctx
'error' event, mediaDevices 'devicechange', ctx 'sinkchange', track 'ended'/'mute', node 'processorerror' (re-create
the node, keep params), playbackStats.underrunEvents > 3 within 10 s (single underruns only counted).
getUserMedia constraints exactly as in the plan; retry with channelCount {ideal: 2} when wantErrorMic and mono.
AudioContext({ latencyHint: 0, sinkId?, renderSizeHint: 128 | 64 }) with try/catch fallbacks.

## src/audio/calibration.ts + src/audio/analysis-client.ts + src/workers/analysis.worker.ts
```ts
export type MeasureFn = (stimulus: Float32Array, channel: 0 | 1, postRollFrames: number) => Promise<{ ref: Float32Array; err: Float32Array | null; masterLin: number; aborted: boolean }>
export type CaptureFn = (seconds: number) => Promise<{ ref: Float32Array; err: Float32Array | null }>   // ambient, anti-noise muted
export interface Analyzer {   // implemented in-process (tests/simulator) and by the worker client (browser)
  estimateDelay(recorded: Float32Array, stimulus: Float32Array, fs: number, f1: number, f2: number, mode: 'acoustic' | 'loopback'): Promise<DelayEstimate & { ir: Float64Array }>
  transfer(x: Float32Array, y: Float32Array, fs: number, freqs: Float64Array): Promise<{ re: Float64Array; im: Float64Array; coherence: Float64Array }>  // H1 = Gxy/Gxx, Welch
  estimateF0(x: Float32Array, fs: number, fmin: number, fmax: number): Promise<F0Estimate | null>
}
export const inProcessAnalyzer: Analyzer
/** analysis-client.ts: worker-backed Analyzer (same results; pending and later requests run in-process if the
 *  worker cannot start or dies; ok=false answers reject). */
export function createAnalyzer(): Analyzer & { readonly usingWorker: boolean; terminate(): void }
/** analysis-client.ts: the live f0 window (last windowS s of pushed float32 blocks, allocation-free push) whose
 *  estimate runs through an Analyzer — same window, samples and result as F0Tracker. */
export class F0Window { constructor(fs: number, windowS?: number, fmin?: number, fmax?: number); push(block: Float32Array): void;
  snapshot(): Float32Array /* copy, time order */; estimate(a: Pick<Analyzer, 'estimateF0'>): Promise<F0Estimate | null>; reset(): void }
export interface PathData { delaySamples: number; sign: 1 | -1; qualityDb: number; stdSamples: number;
  freqs: number[]; re: number[]; im: number[];  /** response of FINAL output sample → mic (divided by masterLin) */
  ir: number[]; irOffset: number /* IR window start (samples) */ }
export interface CalibrationData {   // plain JSON, stored in presets
  version: 1; fs: number; band: [number, number]; tempC: number; createdAt: number;
  feedback: [PathData | null, PathData | null];      // output ch → reference mic
  secondary: [PathData | null, PathData | null];     // output ch → error mic (or zone mic in two-position mode)
  primary: { freqs: number[]; re: number[]; im: number[]; coherence: number[]; tonalF0: number | null } | null;
  loopbackLatencyS: number | null;
  geometry: { micToZoneM: number; sourceAngleDeg: number; speakerToZoneM: [number, number]; tempC: number } | null;
  levels: { zoneDb: number | null; refDb: number | null };   // two-position calibration
  speakers: SpeakerConfig;
  /** Reference input the paths were measured with (EngineParams.refChannel); absent in calibrations saved before
   *  it was recorded (then the preset's refChannel). Kept by sanitizeCalibration when 0 or 1. */
  refChannel?: 0 | 1;
}
export interface MeasurePathOptions { fs: number; channel: 0 | 1; band: [number, number]; levelDbfs: number /* e.g. -20 */;
  repetitions: number; mic: 'ref' | 'err'; mode: 'acoustic' | 'loopback'; previewFirst: boolean; signal?: AbortSignal;
  /** 1-based input of the error mic, named when it is missing (default 2). */ errorInput?: 1 | 2 }
export function measurePath(measure: MeasureFn, analyzer: Analyzer, o: MeasurePathOptions): Promise<PathData & { ok: boolean; reason?: string }>
export function measurePrimary(capture: CaptureFn, analyzer: Analyzer, o: { fs: number; seconds: number; band: [number, number]; signal?: AbortSignal }): Promise<NonNullable<CalibrationData['primary']>>
export function looksLikeLoopback(path: PathData): boolean   // flat |H|, coherent, delay ≈ electronic only
export interface SolveOutcome { params: EngineParams; result: AlignmentResult; advice: string[] /* Italian */;
  /** Raw open-loop gain of the returned params through the feedback paths (no neutralisation). */
  loopGainDb: number | null;
  /** The same after neutralisation with the feedbackFirs(cal, feedbackTaps) the app loads. */
  loopGainAfterDb?: number | null;
  /** dB taken off every channel gain to keep the neutralised loop at LOOP_GAIN_DANGER_DB (0 = none). */
  loopLimitedDb?: number
  /** Tonal mode: the harmonics the trims act on (1 … tonal.harmonics, in the calibration band, below 0.45·fs);
   *  absent outside tonal mode and when the params were left unchanged. */
  harmonics?: HarmonicSolution[]
  /** Tonal mode: the delay/gain/polarity solution alone (no trims, no centring raise) under its own stability margin,
   *  and its prediction — what the trims are compared with: result.predictedAttenuationDb ≥ delayOnly.predictedDb.
   *  Absent outside tonal mode, when the params were left unchanged, and when the margin refused the delay alone. */
  delayOnly?: { params: EngineParams; predictedDb: number }
  /** result.predictedAttenuationDb is at the ceiling of the solve model (predictionCeiling); absent otherwise. */
  predictionCeiling?: PredictionCeiling }
// When the params are left unchanged (advice «… i parametri non sono stati cambiati») result.predictedAttenuationDb is
// the prediction refused (< 1 dB, the figure the advice quotes), so the outcome is never usefulOutcome; every applied
// outcome has a model and predictOutcome(o, o.params).predictedDb === result.predictedAttenuationDb.
export interface HarmonicSolution { h: number; freqHz: number /* h·f0 */;
  /** Predicted attenuation at the error mic at this harmonic with the params returned (trims, clamping, loop
   *  limit, coherence, closed loop), and the same for the delay/gain solution alone (no trims, no centring raise)
   *  under its own stability margin (SolveOutcome.delayOnly; its own cuts, uniform or of the trims at a harmonic):
   *  predictedDb is below untrimmedDb only where giving that harmonic the delay alone's response back takes the loop
   *  over the margin — and neither lowering its trims into the margin predicts more there and in total, nor lowering
   *  every gain predicts more in total by at least that cut (GIVE_BACK_CUT_GAIN_DB = 0.05 dB at most) — or needs a
   *  trim outside RANGES.trimGainDb. */
  predictedDb: number; untrimmedDb: number;
  /** The trim this harmonic needed was outside RANGES.trimGainDb and was clamped: partial cancellation. */
  capped: boolean;
  /** dB the stability margin left off this harmonic's trims (on the enabled channels with a feedback path, net of
   *  what it gave back after a uniform cut); absent when none. The uniform part is SolveOutcome.loopLimitedDb. */
  loopLimitedDb?: number;
  /** predictedDb is at the ceiling of the solve model (predictionCeiling); absent otherwise. */
  ceiling?: PredictionCeiling }
/** Accuracy floor of the solve model without an error mic: predictions ≤ about this many dB. */
export const GEOMETRY_ACCURACY_DB = 30
/** 'geometry': without an error mic, the model's accuracy limit (GEOMETRY_ACCURACY_DB); 'exact': with one, an exact
 *  null of the measured model, where the calculation stops (EXACT_NULL_DB). */
export type PredictionCeiling = 'geometry' | 'exact'
export const EXACT_NULL_DB /* −10·log10(1 − (1 − 1e-6)) ≈ 60 */, PREDICTION_CEILING_TOL_DB = 0.5
/** 'geometry' when !errorMic and db ≥ GEOMETRY_ACCURACY_DB − 0.5; 'exact' when errorMic and db ≥ EXACT_NULL_DB − 0.5;
 *  otherwise (and for NaN / ±∞) null. */
export function predictionCeiling(db: number, errorMic: boolean): PredictionCeiling | null
/** An applied outcome's predictions recomputed for other params p (same f0 and harmonics) with the outcome's own
 *  model (kept by the solvers in a WeakMap: outcomes stay plain data). Separate zones: on the merged params each
 *  zone's anti-noise goes through the loop of both speakers, 1/(1 − L₀ − L₁) (its model holds both neutralised
 *  feedback paths; the other speaker's path into the zone is not modelled: never measured). Per harmonic the
 *  outcome's capped / loopLimitedDb flags are kept, the ceilings recomputed; untrimmedDb on p: the outcome's delay
 *  alone with its channels moved by p's change of gain (on the outcome's own params, the outcome's own figures), also
 *  returned as `delayOnly` (params and total). `delayAlone`: p carries that delay alone on the outcome's channels (a
 *  zone that plays it, step 6): no capped flags, loopLimitedDb from p's negative trims (its own loop cuts),
 *  untrimmedDb = predictedDb. Null for an outcome without a model (not applied by a solver, or a copy). */
export interface OutcomePrediction { predictedDb: number; predictionCeiling: PredictionCeiling | null; harmonics?: HarmonicSolution[];
  delayOnly?: { params: EngineParams; predictedDb: number } }
export function predictOutcome(o: SolveOutcome, p: EngineParams, delayAlone = false): OutcomePrediction | null
/** Italian advice lines step 6 restates for separate zones played together: «Attenuazione prevista solo X dB: …»
 *  (null at ≥ 6 dB) and the out-of-range trims («Armonica N (F Hz): la correzione necessaria esce dal campo …»). */
export function weakAttenuationAdvice(predictedDb: number): string | null
export function cappedAdvice(list: readonly HarmonicSolution[]): string | null
export interface SolveOptions { feedbackTaps?: number /* neutraliser length loaded, default DEFAULT_FEEDBACK_TAPS */ }
export const LOOP_GAIN_LIMIT_DB = -20, LOOP_GAIN_DANGER_DB = -6, DEFAULT_FEEDBACK_TAPS = 256
export const FEEDBACK_TAPS_RANGE = { min: 128, max: 1024 }   // FEEDBACK_MAX_TAPS
export interface LoopGains { rawDb: number; atHz: number; afterDb: number }
/** Loop gain of `p` with both channels through the feedback paths, raw and after neutralisation with
 *  `taps`-tap FIRs; null without feedback data, at another rate or when not finite (the solvers' check). On the
 *  calibration grid plus the harmonic centres; afterDb is refined between them around the highest sampled peaks
 *  (within 3 dB of the highest, 8 at most; golden section): the 2–2.5 Hz grid missed up to 0.1 dB (0.3 dB with
 *  separate zones merged) of a peak between two harmonics' bands. Relative to the highest sample, so the result
 *  scales exactly with the gains (capLoopGain lands where it aims). */
export function checkLoop(cal: CalibrationData, p: EngineParams, fs: number, taps: number): LoopGains | null
/** The solvers' stability margin for params that did not come from a solve (autotune result, shorter
 *  neutraliser, loaded preset): when checkLoop(…).afterDb > LOOP_GAIN_DANGER_DB (+ 1e-7 dB of rounding allowance:
 *  a loop left at the margin is within it) every channel gain is lowered by the excess (clamped), landing at
 *  most at the margin (5e-8 dB below it), so capping twice changes nothing. Returns the params, the dB taken off
 *  (0 = unchanged) and their loop (null: nothing to check). */
export function capLoopGain(cal: CalibrationData, p: EngineParams, fs: number, taps: number): { params: EngineParams; limitedDb: number; loop: LoopGains | null }
/** Statistical bandwidth (Σ S·Δf)² / Σ S²·Δf of the noise at the error mic in the band (from the primary PSD),
 *  null without one; the autotune sizes its windows on it. */
export function noiseBandwidthHz(cal: CalibrationData): number | null
/** Stored paths with a loopback signature, as Italian names; [] for null. Checked before every un-mute. */
export function loopbackSuspectPaths(cal: CalibrationData | null): string[]
/** Error-mic (or two-position) solution: converts total delay/gain (incl. D_MIN and master) into ChannelParams
 *  (delayMs above D_MIN, Δ = 0, gainDb = total − master, polarity); clamps; advice (amplifierAdvice) when the
 *  needed channel gain is outside -30…-6 dB: above -6 dB raise the reference mic's input gain (preamp) on the
 *  interface — never the amplifier, whose wizard step-2 level (-20 dBFS ≈ noise in the zone) is the safety
 *  margin — below -30 dB lower the amplifier; computes the loop gain with the feedback paths. */
export function solveFromCalibration(cal: CalibrationData, params: EngineParams, fs: number, opts?: SolveOptions): SolveOutcome
/** Geometry-only solution (no error mic): primary = pure delay from headStartSeconds, secondary = measured or
 *  latency + speaker distance. */
export function solveFromGeometry(cal: CalibrationData, params: EngineParams, fs: number, latencyS: number, opts?: SolveOptions): SolveOutcome
export function feedbackFirs(cal: CalibrationData, taps: number): [{ bulkDelay: number; taps: Float64Array } | null, { bulkDelay: number; taps: Float64Array } | null]
export function sanitizeCalibration(raw: unknown): CalibrationData | null
```
`src/audio/autotune.ts`: pure state machine + async driver
```ts
export interface AutotuneDriver { apply(p: EngineParams, jump: boolean): void; settle(ms: number): Promise<void>;
  measureErrorDb(ms: number): Promise<number>; measureBypassDb(ms: number): Promise<number>; signal?: AbortSignal
  /** Optional, tonal mode: the error-mic level at each of `freqs` over `ms` (harmonicLevelsDb(x, fs, freqs, tolHz)),
   *  anti-noise active or bypassed (`bypass`: the driver toggles the bypass and settles as for measureBypassDb). */
  measureHarmonicsDb?(ms: number, freqs: readonly number[], bypass: boolean, tolHz?: readonly number[]): Promise<number[]> }
export interface HarmonicTuneResult { h: number; freqHz: number; startDb: number; endDb: number /* AutotuneResult.endDb's window */;
  bypassDb: number /* the bypass level at the endDb window's time, interpolated */;
  startBypassDb?: number /* the same at the startDb window */ }
export interface AutotuneResult {
  /** The params in place at the end: the best point, or the start when that did not hold (improved false). */
  params: EngineParams;
  /** The start window's level and that of the last window measured with `params` (the final window; the start
   *  window again when the start was kept or restored); per-harmonic search: power sums of the harmonic levels. */
  startDb: number; endDb: number;
  /** The bypass level at the endDb window's time, interpolated between the bypass windows around it (not the power
   *  mean of the run), and the same at the startDb window: attenuation = bypassDb − endDb, before = startBypassDb − startDb. */
  bypassDb: number; startBypassDb: number;
  /** params ≠ start and (bypassDb − endDb) > (startBypassDb − startDb): the final window confirmed the search. False when aborted. */
  improved: boolean;
  steps: number; aborted: boolean; reason?: string
  /** Per-harmonic search only (also on abort, endDb then the window that stopped it): attenuation = bypassDb − endDb. */
  harmonics?: HarmonicTuneResult[] }
export interface AutotuneOptions { fs: number; channels: (0 | 1)[]; rt60S: number; band: [number, number];
  /** noiseBandwidthHz of the zone noise: windows sized on it (never above the nominal band width). */
  effectiveBandHz?: number;
  /** Candidates for which this is false are skipped, never played (the start always is): the wizard passes
   *  logic.ts loopGuard, the loop-gain margin of the solve. */
  allowed?: (p: EngineParams) => boolean }
export const AUTOTUNE_BT = 150, MAX_WINDOW_MS = 4000, HARMONIC_DRIFT_MAX = 0.01
/** Hann lock-in level of `x` at each frequency, dBFS RMS (−200 for silence). With `tolHz` (per frequency or one for
 *  all): the highest level within ±tol (zero-padded FFT ≥ 2n, parabolic peak interpolation, < 0.15 dB), so a tone
 *  that drifted is read at its own frequency; on noise alone that reads a few dB high (pessimistic). */
export function harmonicLevelsDb(x: Float32Array | Float64Array, fs: number, freqs: readonly number[], tolHz?: number | readonly number[]): number[]
/** The drift a per-harmonic search tolerates: min(f/(2·tonal.q), HARMONIC_DRIFT_MAX·f) (the resonator half band). */
export function harmonicTolerancesHz(p: EngineParams, freqs: readonly number[]): number[]
/** Harmonics a per-harmonic search tunes: h ≤ tonal.harmonics (and MAX_HARMONICS) inside `band`, below 0.45·fs. */
export function tunedHarmonics(p: EngineParams, band: [number, number], fs: number): { h: number; freqHz: number }[]
/** Window (B·T ≥ AUTOTUNE_BT, capped at MAX_WINDOW_MS), settling, σ of one window's level
 *  (10/ln10/√(B·T) dB) and the abort margin (1 dB at AUTOTUNE_BT, widened as 1/√(B·T) for capped windows). */
export function autotuneTiming(o: AutotuneOptions): { windowMs: number; settleMs: number; sigmaDb: number; abortMarginDb: number }
export function runAutotune(driver: AutotuneDriver, start: EngineParams, o: AutotuneOptions): Promise<AutotuneResult>
```
Search: delay mode → delay + gain; tonal/hilbert → Δ + gain; one channel at a time (other active); settle =
max(smoothing, crossfade, rt60) ; measurement windows with B·T ≥ AUTOTUNE_BT (150) on the effective noise bandwidth.
Bypass windows: one before the start, one every 6 windows, one after the final window. Each window is referred to the
bypass level at its own time: linear in dB and time between the bypass windows around it, the nearest one outside
them (never extrapolated) — the ambient level may move several dB in a run of a minute or two. Abort (restore the
start) when a window exceeds that level + the abort margin; a window above the latest bypass + margin is decided at
once by a fresh bypass window, the others when the next bypass window arrives (a falling ambient would hide an
anti-noise that adds noise). The search compares points (best picks, parabola and paraboloid fits) by their level
relative to that reference, and keeps its best point only when it beats the start the same way; the final window must
confirm it ((bypassDb − endDb) > (startBypassDb − startDb)), else the start is restored (`improved` false) — with raw
levels a falling ambient passed a point 0.35 dB worse off as improved and a rising one threw a 1.7 dB gain away. The
closing bypass window carries the end reference alone: one more than 2 abort margins off the trend of the bypass
windows before it (linear through the last two) is measured again; the pair is averaged (at their mean time) when it
agrees within that, else the one nearer the trend is kept (a +6 dB transient there inflated the reported, stored
attenuation by 5.2 dB). The wizard passes effectiveBandHz = noiseBandwidthHz of the zone being tuned and, in tonal
mode, measures the level at the harmonics of f0 (synchronous detection) rather than the band RMS.
Tonal + measureHarmonicsDb → per-harmonic trim search instead of Δ + gain: every window probes all harmonics at
T·(1 ± δ) and T·(1 ± jδ) (δ ≤ 0.25, starting at 2× the harmonic's residual relative to bypass, ≥ 0.04, halved each
round); each harmonic moves to the vertex of a least-squares paraboloid (at most 2δ) and keeps its best measured
point; the abort test uses the power sum of the harmonic levels; Δ, delay and gain are untouched; two speakers are
tuned one channel at a time. A point assembled from the per-harmonic bests of several windows (never played as a
whole) is kept only when `allowed`: otherwise the measured point with the lowest total, plus the other harmonics'
bests that keep it allowed (largest gain first) — with overlapping bands (low Q) the loop between two harmonics adds
up from both, so an assembled point could exceed the margin of every window it came from. The levels are read with
the drift tolerance harmonicTolerancesHz(start, freqs): with the 4 s windows the wizard uses for tonal noise an
exact-frequency lock-in read a 0.2 % f0 drift over the run as up to 56 dB of extra attenuation.

## src/app/controller.ts (browser, owns app state; main.ts only mounts views and binds them)
Holds EngineParams (+ link L/R with offset), validation result and overrides, presets, calibration, f0 window on
'cond' taps (estimated every 2 s over the last 2 s off the main thread through `ControllerOptions.f0Analyzer`, default
the analysis worker from `createAnalyzer()` created on first use and terminated by dispose(); a tick is skipped while
an estimate is in flight, a failed estimate reads as "no tone", results arriving after dispose() or a sample-rate
change are dropped; suggests tonal mode when dominanceDb ≥ 15), spectra (FFT of tap blocks, Hann, 8192), attenuation
estimate with bypass windows (attenuation.ts; in 'interp' the raw reference only counts by its drop during the
ANC-on run, vs its own level over the run's first second, beyond 1 dB and 3 standard errors: the anti-noise it picks
up shifts it from the first window, by ± for a tone), the "ANC sta aumentando il rumore" guard (error > bypass
reference + 3 dB for > 2 s → mute), keyboard shortcuts (Esc capture → panic always; M/B ignored in form fields or with modifiers), status/stats.
Exposes a small typed API + subscribe() for the views and the wizard.

---

# Engine behaviour notes (phase 3)

Behaviour `src/dsp/antiphase-engine.ts` implements beyond the phase-2 contract (constants named as in the source).
- **Unmute always soft-starts from 0.** `mute` false (only when muted) sets the mute envelope back to 1 and, if the
  output is still audible (mute ramp not yet at 0), first fades the remaining level out at the mute slope
  (`MUTE_RAMP_MS` × level), then soft-starts from 0 over `SOFT_START_MS`; howl detectors are reset. `softStart`
  restarts the same way (a 15 ms fade first when audible). `arm` true always starts from 0 and resets limiters and
  detectors.
- **Silence until the delay line is clean.** The engine keeps a sample clock and the index of the first clean
  delay-line sample (`cleanFrom`): the sample after a chain swap, a reference-input or feedback-FIR change, or the
  end of a played stimulus (finished, or aborted while playing: the stimulus fades over `PANIC_RAMP_MS`). A channel
  ramps back up (15 ms reconfigure envelope), and the measurement gate reopens (`MUTE_RAMP_MS`), only when the
  delay tap's reach (`SmoothDelayTap.reach`: read position, crossfade 'from' position and target) satisfies
  n − ceil(reach) − `LAGRANGE_REACH` (3) ≥ that index. A non-structural delay change that would read earlier than
  `cleanFrom` takes the structural path (ramp down, snap, wait). A measurement aborted before its stimulus started
  leaves the delay line clean (no wait). Per-channel structural changes alone (polarity, enable, delay ↔ Hilbert
  mode) ramp the channel down, snap its delay tap to the target and ramp it up.
- **Limiter-saturation backstop.** For howls the detector cannot see (above its periodicity bands: ~2.4 kHz, or
  ~4.8 kHz for a chain wider than 2 kHz; ~1.2 kHz for loops of 600–1100 ms; loops longer than 1.1 s). While armed, outside measurements and with no howl latched, a channel whose limiter gain
  reduction, smoothed by a one-pole of `LIMITER_BACKSTOP_TAU_MS` (300 ms), stays above `LIMITER_BACKSTOP_DB`
  (4.5 dB) for more than `LIMITER_BACKSTOP_MS` (2 s) of consecutive samples triggers the howl path with reason
  'limiter-saturated' (freqHz null), like a detector trigger, whatever the content. GR alone cannot tell a howl
  from loud noise (broadband noise 6 dB below the ceiling averages ~3 dB of GR, as much as a howl with loop gain
  1.5), so the threshold sits above that: it catches loop gains ≳ 1.8 and gross overloads only. (It used to be
  unsmoothed GR > 0.5 dB for 2 s.) The detector's own rules: see the howl-detector
  section.
- **Howl texts.** `autoMute` reason: detector trigger → "Innesco rilevato[ a N Hz]: <reason text>. Uscita
  silenziata." with 'peak-growth' "il livello cresce più in fretta del guadagno impostato", 'limiter-saturated'
  "il limiter è saturo con un suono che si ripete (fischio o eco)", 'output-overload' "il livello d'uscita è
  vicino al ceiling"; backstop → "Innesco probabile: il limiter ha tolto più di 4,5 dB per più di 2 s. Uscita
  silenziata."
- **Live vs structural parameter changes.** Against the parameters the running (or standby) chain was configured
  with, HP / barrier / LP frequency changes within a factor `LIVE_FREQ_RATIO` (2), tonal Q within `LIVE_Q_RATIO` (3)
  and tonal f0 within `LIVE_F0_RATIO` (1.03) reconfigure the running chain in place (states kept, coefficients and
  resonator gains glide over `COEFF_RAMP_MS`, 20 ms: ≤ −100 dB of click on 50/100/400 Hz tones); larger changes
  take the structural path (standby chain configured, reset and warmed up in parallel, swapped under the
  reconfigure envelope). A live change that moves a cutoff, or tonal f0 or Q, calls `holdLevels(2·COEFF_RAMP_MS)`
  on both howl detectors: the response glides at constant commanded gain (a 50 Hz hum rises ~20 dB while the HP
  is dragged 200 → 20 Hz), which is no howl; the reconfigure envelope of larger changes is in the commanded gain
  the detectors compensate for. The number of harmonics is
  part of the chain structure (as are filter enable/order and entering/leaving the tonal mode), so changing it is
  always structural.
- **Rises are heard.** Channel gain and master share one GainRamp per channel (rise floor = channel gain +
  `RISE_FLOOR_DB`, so rises are free only while the master part is below −60 dB). While the bypass or the
  measurement gate holds the output at 0 (both reopen in 15 ms), gain increases wait (`GainRamp.next(holdRise)`)
  and the global and per-channel soft-starts pause. Enabling a channel while armed, removing a filter section
  (HP, barrier or LP switched off), leaving tonal mode or changing the reference input soft-starts the channels
  from 0 over `SOFT_START_MS`; the per-channel soft-start progresses only while the channel is audible. Narrower
  changes, polarity flips, Hilbert ↔ delay switches and the bypass keep the 15 ms ramp. Not soft-started (judged
  small): live cutoff increases up to ×2, LP/HP order changes and removing the feedback FIR.
- **Measurement master.** The master GainRamp only scales the calibration stimulus. When the stimulus starts (the
  anti-noise at 0) the master jumps to its target and both limiters reset, so every stimulus sample plays at the
  reported `masterLin`; a master change while it plays aborts the measurement (`measureResult` aborted, the
  stimulus fades over `PANIC_RAMP_MS`). The console disables the master control while a stimulus plays.
- **Harmonic trims.** Per channel a `TrimRamp` (harmonic-trims.ts) turns `harmonicTrims` targets into applied
  values, sharing the channel's GainRamp budget (channel gain + master): a harmonic's level rises by at most
  BIG_JUMP_DB + MAX_RISE_DB_PER_S·T however gain, master and trim are combined (one message raising gain and trim by
  6 dB each rose 12 dB at once; gain and trim both rising slowly gave 12 dB/s). Decreases, phase changes and rises
  within the budget apply at once, larger increases at MAX_RISE_DB_PER_S in steps of COEFF_RAMP_MS (≤ 0.12 dB each,
  glided back to back, C2: ≤ −100 dB above 2 kHz); a trim rise holds back a gain rise of that channel (and vice
  versa); while the bypass or the measurement gate is shut increases wait (like gains). The applied values go to a
  standby chain at once (a newly configured standby gets them even unchanged), so f0 live glides carry the trims
  and a harmonics-count swap starts with them, and to the live chain with a COEFF_RAMP_MS glide at most once per
  glide: a newer value that arrives while it glides (a dragged trim control sends one every 10–16 ms) waits for its
  end, so the weights move in whole back-to-back glides (restarted mid-glide they left −72…−85 dB above 2 kHz; now
  ≤ −100 dB). Disarmed: applied at once. A trim change is not structural; a hand-over that raises a harmonic faster
  than the 6 dB/s staircase (a fast rise from the budget, e.g. a dragged trim gain) calls holdLevels(LIVE_HOLD_MS) on
  that channel's detector, as a live filter change does (the level grows at constant commanded gain: a 6 dB drag
  over 0.3–0.5 s was a 'peak-growth' run and muted the output); the staircase itself does not (≤ 6 dB/s is no
  run), so a howl the trims close while rising is still seen. process() only applies precomputed inverses (no
  solve, no allocation). Cost (Node 24, 48 kHz, 128-frame blocks, tonal 8 harmonics, both channels, medians of 5
  runs of 5 s, after the review fixes): +0.14 % of real time over the engine of git 731a3f1 untrimmed (companions
  and the second line), +0.28 % with both channels trimmed, +0.83 % while all 16 trims rise. With every trim at 0
  the outputs equal the former engine's (tests/unit/antiphase-engine-golden.test.ts, stored reference values).
- **Ceiling.** While armed, ceiling decreases apply at once; increases rise at `MAX_RISE_DB_PER_S` (6 dB/s). Disarmed,
  every change (ceiling, gains, chain) applies at once.
- **Output clamp.** After the limiter each output sample is clamped to ± the largest float32 not above the linear
  ceiling (the output arrays are float32: rounding up would exceed it).
- **Message queue.** At `MAX_QUEUE` (256) undrained messages, new 'tap' blocks go straight back to the pool and
  'meters' / 'measureProgress' are dropped; every other message (measureResult, howl, autoMute, fault, channelCheck)
  is always queued. Taps come from a pool of `TAP_POOL_SIZE` (8) buffers transferred to the main thread, which must
  return each one with `recycle` (AudioEngine does it right after emitting the tap): without recycling the pool
  runs dry and blocks are dropped (counted).
- **Faults.** An exception inside `process()` zeroes that block, panics at once (`muted` true, mute envelope 0,
  measurement aborted), resets every recursive state and the howl detectors and posts `fault`. The worklet adapter
  additionally zeroes its outputs after any adapter-level fault (at most one `fault` per second) until it receives
  `mute` false; leaving the fault state therefore always needs `mute` false, which soft-starts from 0.

---

# App and UI behaviour notes (phase 3)

## src/audio/audio-engine.ts
- An auto-mute (lifecycle event) also posts `abortMeasure` and settles every pending `measure()` as aborted:
  the stimulus is played after the mute envelope and would otherwise keep sounding.
- Output latency: while the output is audible a jump of more than `LATENCY_JUMP_MS` (5 ms) between two readings
  auto-mutes. While it is silent (muted or disarmed) the reference stays at the latency last heard, and a jump
  that happened meanwhile is reported at the next un-mute or arm. A completed measurement resets the reference
  (the calibration made from it is valid for that latency).
- PANIC keeps any context created during a re-creation (latency mode, output or sample-rate change in progress)
  suspended until `resume()`.
- `setOutput`: a rejected `setSinkId` restores the previous sinkId (the context still plays there); Chrome's own
  'suspended' state during its `setSinkId` is not reported as a lifecycle event.
- A re-created processor gets the parameters replayed (`replayShadow`), but not feedback FIRs posted at another
  sample rate (they are sample-indexed).

## src/app/controller.ts
- `ControllerOptions.firsFromCalibration(calibration, taps)`: main.ts sanitises and calls
  `feedbackFirs(cal, taps)`. FIRs apply only at the calibration's sample rate (re-checked when the rate changes)
  and with the reference input the calibration was set (or its preset saved) with.
- `feedbackTaps` (state) / `setFeedbackTaps(n)`: neutraliser length, clamped to `FEEDBACK_TAPS_RANGE`, remembered
  with the device choices; changing it rebuilds the current calibration's FIRs. A shorter neutraliser leaves more
  of the loop: `capLoopGain` is checked for the new length and, above the margin, every channel gain (and the
  double-click values) comes down by the excess before the shorter FIRs are sent, with a warning ("Neutralizzatore
  a N tap: …"). `loadPreset` checks the preset's gains the same way for the taps in use (presets do not store
  them) and adds the warning to the preset message. The UI offers
  `src/ui/feedback-taps.ts FEEDBACK_TAP_CHOICES` (128/256/512/1024) under Fase e filtri › Avanzate; the wizard
  passes `state.feedbackTaps` to the solvers (`SolveOptions.feedbackTaps`) and to the merged-zones loop check.
- `setRefChannel` disarms an armed output (the alignment was for the other input) and re-applies the FIRs: they
  run only on the input the calibration records (`CalibrationData.refChannel`, else the current input at
  `setCalibration` or the preset's at `loadPreset`).
- Every un-mute (M, `arm()` un-muted, `loadPreset` while audible, `setCalibration` while audible) is refused while
  the calibration has a loopback signature (`ControllerOptions.loopbackSuspects`, default calibration.ts
  `loopbackSuspectPaths`; `ArmFailure` 'loopback'), while a calibration is loaded that records another reference
  input than the current one (`ArmFailure` 'ref-input'; arming muted for stimuli stays allowed), or while the
  output is headphones that were not confirmed.
  Headphones appearing while armed mute the anti-noise and clear the confirmation.
- `toggleMute()` (M, the status-bar Mute) while disarmed only shows "L'uscita non è attiva: usa «Attiva uscita»
  dopo la validazione." (and leaves a PANIC): a mute has nothing to silence there and `arm()` sets the mute
  state itself. `setMuted(true)` from code still mutes.
- A lifecycle auto-mute aborts a running measurement and fails a running capture.
- `setOutput` takes (shows, validates, remembers) the new output only after the engine has switched.
- `deviceKey` (state): device key of the open input and the chosen output, exactly as saved in presets
  ('' while the microphone is closed). The wizard compares it with a reused calibration's key.
- Spectra are split by ANC state: `err` only averages windows with the anti-noise on (after `PHASE_SETTLE_MS`),
  `errBypass` only windows with it off; `refOn` / `refOff` average the reference over the same windows (from x̂,
  the 'cond' tap, while a neutralisation FIR runs; from the raw reference otherwise; restarted when that source
  changes). spectrum-view `bandAttenuationDb` computes A = [L_err − L_rif]_off − [L_err − L_rif]_on per
  third-octave band from these four arrays (console hover and wizard step 7).
- `attenuationRefAgeS` (state): age of the newest bypass window the attenuation is measured against; the meters
  view flags the value as stale beyond `ATTENUATION_STALE_S` (60 s): "Riferimento di bypass di N s fa: premi B
  per aggiornarlo."
- f0 following (tonal mode, `f0Follow` on): an estimate with confidence ≥ 0.5 and dominance ≥ `TONAL_DOMINANCE_DB`
  (15 dB) within a ratio of 1.03 of `params.tonal.f0` replaces it (a 'params' message); bigger moves are the
  user's ("Aggancia al picco"). An async caller that copied the params before an await (the autotune, its final
  apply included) applies only what it changed over `currentParams` (ops.ts `liveWithChannels`): the whole copy would
  set f0 back by the drift since — beyond 3 %, out of the follower's reach for good — and the master to its old value.
- Harmonic trims travel inside ChannelParams (setChannel, applyParams, presets, calibratedChannels as deep copies).
  `setHarmonicTrim(idx, h, trim)`: one harmonic (h 1 … 8, clamped; a non-finite field unchanged) through setChannel.
  setChannel with 'harmonicTrims' applies «Collega L/R» per entry and field (link.ts): gainDb moves by the same
  delta on both channels, offsets kept, limited to −30…+6 dB; phaseDeg moves by the same delta wrapped into
  (−180°, 180°], never limited (an angle). `clearHarmonicTrims(idx)` («Azzera armoniche»): every trim of that
  channel to 0, of both while linked (link.ts `clearTrims`: an absolute reset is copied like a toggle, never applied
  as a delta, which left the channel trimmed and gave the other one new corrections).
- `CalibratedHarmonic { h; freqHz; attenuationDb; source: 'predicted' | 'measured'; capped? }` and
  `ControllerState.calibratedHarmonics`: set by `applyParams(p, { calibrated: true, harmonics })`, cleared by
  applyParams calibrated without `harmonics`, loadPreset, setCalibratedChannels and a `setFeedbackTaps` that lowers
  the gains (the attenuation came with the gains it lowers) (entries outside h 1 … 8 or not finite are dropped;
  empty → null).
- `LEGACY_FEEDBACK_TAPS_KEY` 'controfase.feedbackTaps.v1' is read once into the prefs when they have no feedbackTaps,
  clamped, then removed (also removed when the prefs already have a value); nothing writes it any more.
- `src/app/spectrum.ts harmonicAttenuationDb(spectra, f0, harmonics)`: the measured attenuation per harmonic (index
  h − 1) from the controller's ANC on/off spectra, A_h = [L_err − L_rif]_off − [L_err − L_rif]_on over the Hann main
  lobe (±2 bins) around h·f0; null above 0.45·fs or without a spectrum.

## src/audio/calibration.ts
- `finishSolve` enforces the stability margin (`bestLoopMargin`): the neutralised loop of the params returned stays at
  or below `LOOP_GAIN_DANGER_DB` (−6 dB) in band, checked on the loop frequencies with the peaks refined between them
  (checkLoop) and, in tonal mode, finely across each harmonic's band. `enforceLoopMargin`, in tonal mode, first lowers
  the trims of each harmonic whose neutralised loop peak in h·f0 ± min(h·f0/(2Q), f0/2) (17 samples, then golden
  section) exceeds the margin + 1e-7 dB, on the enabled channels with a feedback path, by the excess + 5e-8 dB (floor
  −30 dB; up to 4 passes, a peak off the centre follows the trim only nearly); a harmonic the cut would leave louder
  than with those trims at −30 dB (the closed loop's phase) gets −30 dB. What still exceeds the margin is taken off
  every channel gain (`loopLimitedDb`, capLoopGain's allowance, so capLoopGain afterwards changes nothing); that cut
  lowers the harmonics already cut too, so each then gets back as much of its own trim cut as the margin allows
  (largest gain at its harmonic first, ≤ 3 halvings, each checked on its band and the neighbours', the result on the
  whole loop, else again with whole-loop checks; not below a 0.1 dB uniform cut). `bestLoopMargin` keeps that or the
  uniform cut alone, whichever predicts more in total: a peak off a harmonic's centre that the neighbours' skirts feed
  too needed a deeper trim cut than the uniform excess (seed 435 of the review: 10.1 vs 13.1 dB). If the margin leaves
  less than 1 dB of predicted attenuation the params are not applied. Between `LOOP_GAIN_LIMIT_DB` (the plan's −20 dB)
  and −6 dB there is advice only (a FIR of a few ms cannot model the response near the low band edge; those loops are
  stable in the simulator). Advice: «Armonica N (F Hz): guadagno d'anello … ridotta di X dB (attenuazione prevista Y
  dB)…», with «, oltre alla riduzione di Z dB dei guadagni dei canali» when a uniform cut (≥ 0.05 dB) followed, and
  «; le altre armoniche non cambiano» only when none did and some trimmed harmonic is not listed.
- `predictedAttenuationDb` is recomputed for the params actually returned (clamping and loop limit included) and,
  with an error mic, includes the zone noise the reference does not hear (coherence). Predictions and tonal trims are
  the closed loop's: R = P + Σ G·S/(1 − L), L = Σ G·(F − F̂) with feedbackFirs(cal, taps) (|1 − L| < 1e-12 counts as
  uncancelled); the trims solve p + Σ T·(b − p·ℓ) = 0, ℓ = G·F̃ (without it a loop of −2.7 dB at one harmonic turned a
  predicted 60 dB into −1.1 dB in the simulator).
- Tonal mode: the primary path's bulk delay is removed before interpolating between harmonics; a measured f0
  outside the resonator bandwidth of `params.tonal.f0` makes the solve unusable (advice to re-estimate).
- Geometry-only solutions take the chain polarity from the measured feedback path F (advice when unknown).
- `feedbackFirs` covers every significant arrival of the feedback IR (within −20 dB of the strongest and above the
  diffuse floor), from the earliest to `taps` past the latest, up to `FEEDBACK_MAX_TAPS` (1024).
- `measurePath` two-level check: the tone-cleaned preview (`PREVIEW_DB` lower) and the real sweeps must give the
  same in-band gain, within 1 dB and 5 σ of the preview's own noise; skipped with fewer than 2 repetitions or a
  preview quality below 15 dB. Failure reason: "Il percorso dipende dal livello …".
- Loop advice: the neutralisation is automatic after the path measurement; "aumenta i tap del neutralizzatore
  (Avanzate)".
- Tonal solves set per-harmonic trims after solutionToParams (`solveHarmonicTrims`, from the untrimmed
  channelResponse with alignment.ts `harmonicFactors`: one speaker or separate zones an exact null per harmonic, two
  speakers on one zone the minimum change). A channel gain is raised first ("centring", as far as its range allows)
  so that no trim needs more than +6 dB — only for harmonics the raise brings into range: a harmonic out of reach
  even with the whole gain range (a speaker notch) is not worth the loop gain the raise adds at that harmonic.
  Candidates (`tonalSolution`), each judged after its own `bestLoopMargin`: the trims with the raise and, when a raise
  was made, without it, each reconciled with the delay alone (`reconcileTrims`); and the delay alone itself (no trims,
  no raise). Each then gets back the part of its uniform cut its loop no longer needs (`reclaimCut`: every gain raised
  by the slack, the harmonics given back keeping their response, checked on the whole loop; kept when it predicts more,
  or when it removes the whole cut and predicts no less: seed 211, 10.6 → 11.3 dB). The best total is kept (the raise,
  then the delay alone, only when strictly better), so a tonal solve never predicts less than its delay alone under
  the same rules. reconcileTrims, on the params within the margin: a harmonic left worse than the delay alone gets the
  delay alone's response back (trims reproducing it on the candidate's gains, raise and cuts included, inside the
  trims' range); when that takes the loop over the margin, its trims are lowered into it (secant steps, 8 at most, a
  last excess of ≤ 0.02 dB off every gain: equal steps stopped just above the margin, seed 580 zone R), or every gain
  is lowered by the excess instead when that predicts more in total by at least the cut (0.05 dB at most; up to 3
  rounds); then, one at a time, a harmonic whose trims cost the total (the bank's skirts reach the harmonics above
  tonal.harmonics: seed 126, h4 12 → −2.4 dB) goes back to the delay alone's response. Changes are checked on the
  harmonic's band, the neighbours' and the loop frequencies, then on the whole loop (else again with whole-loop
  checks). Harmonics given back take the delay alone's loop cuts and lose `capped`. A trim cut of ≤ 1e-7 dB
  (rounding of a give-back) is no cut. The loop a remaining uniform cut acts on (the advice's «… dB dopo la
  neutralizzazione del feedback: sopra −6 dB») is the loop returned scaled back up by the cut, bands included; a cut
  below 0.05 dB reads «meno di 0,1 dB». What is
  still out of range is clamped and reported as `capped` with the Italian advice "Armonica N (F Hz): …" ("Armoniche
  …"); a clamped trim that would leave the harmonic worse than no trim is reset to 0. A clamp on a harmonic whose
  primary coherence is below LOW_COHERENCE (0.5) is not reported (the noise there does not come from the
  reference: no statement about the speaker). The amplifier advice is recomputed for a raised gain.
- Usefulness, loop margin and prediction are judged on the candidate kept (after its margin); a refusal quotes, and
  carries in result.predictedAttenuationDb, the figure refused. Below 1 dB with a uniform cut under 0.05 dB
  (`cutRefuses`) the refusal is «Con queste misure non si ottiene attenuazione (prevista X dB) …», not «Rischio di
  innesco … ridotti di 0,0 dB» (seed 21). A non-tonal solve clears every trim (trims belong to the delay they were
  solved with).
- In tonal mode the loop gain (checkLoop, capLoopGain, loopGuard: through channelResponse, trims included) is also
  evaluated exactly at the harmonic centres h ≤ tonal.harmonics: at Q 30 the 2 Hz grid could miss the top of the
  fundamental's peak by several dB.
- Prediction limits: a measured coherence is taken as at most 1 − 1e-6, so an exact null of the measured model
  (what the trims make) predicts 60 dB, not an infinite (or, off a zero residual, 0 dB) attenuation. Without an
  error mic (coherence null: geometry and two-position solves) a residual floor of GEOMETRY_ACCURACY_DB below the
  noise bounds every prediction at ≈ 30 dB (it read ≈ 300 dB per harmonic, shown in step 6 and stored as
  calibratedHarmonics). A prediction within PREDICTION_CEILING_TOL_DB of its ceiling is flagged
  (`SolveOutcome.predictionCeiling`, `HarmonicSolution.ceiling`): it says how far the model reaches, not how much the
  room will give.
- Evaluating one set of params at many frequencies goes through response.ts `channelResponseAt` and the stored IRs
  are converted once per check (worst seeded tonal plant of the review, 8 harmonics at Q 6: ≈ 180 ms per solve).
  loopChecker keeps the feedback paths at the single frequencies its golden-section refinements visit (a few thousand
  per solve, mostly revisited) and evaluates each channel response once for the raw and the neutralised loop: with
  the extra delay-alone candidate the heaviest seeded solves take ≈ 110–190 ms (289: 230 → 110 ms; 446: 155 → 175 ms).

## Guided setup (src/ui/wizard/)
- Flows: 'full' (steps 1–7), 'recal' (start → devices → amp → placement → measure → align → verify) and 'reuse'
  (start → devices → amp → **check** «Controllo rapido» → verify). The check (step-check.ts) plays one short sweep per
  active speaker on the speaker → reference path and compares it with the saved one (logic.ts `compareToBaseline`,
  `quickVerdict`, `QUICK_CHECK`): within 0.5 samples and 0.5 dB 'ok'; up to 6 dB / 3 ms 'corrected' (channel gain
  and delay corrected, the saved feedback path replaced so the FIRs follow); bigger, a flipped polarity or a
  loopback signature 'mismatch' (re-measure); a failed measurement 'unverified' (must be accepted explicitly). A
  speaker the saved solution keeps off (separate zones) is neither swept nor judged: 'ok', «Cassa spenta nella
  calibrazione salvata: non usata, nessun controllo.». The start question («No, usa la calibrazione salvata») sets the
  speakers before loadPreset, so a saved speaker off stays off and loadPreset's loop check is of what plays (setSpeakers
  after it switched channel R on again with unsolved values).
  flow.ts: `checkGate` (step gate), `checkPassed` (every active channel ok/corrected/accepted, on the same
  `ampKey` output and rate), `reuseDeviceMatch` (saved `ReuseInfo.deviceKey` vs `state.deviceKey`: 'different'
  blocks steps 1 and 5, the un-mute and step 7), `relockReason` (a completed run locks the anti-noise again when
  the output, rate or interfaces change, or the reference input). Leaving step 2 gives the saved master back.
- `amplifierReleveled(data, ch)` (step 2 plays the level noise again, or a «Ho regolato…» box changes): the quick
  check of a reused calibration must be repeated; in the other flows that channel's measured paths and the
  solution go (their gains scale with the amplifier).
- `refInputChanged` (the calibration records another `refChannel` than the current one): blocks steps 5, check
  and 6, the un-mute (`REF_CHANGED_TEXT`) and, after completion, `relockReason` (`REF_RELOCK_TEXT`). Step 5 starts a
  new calibration record when the reference input changed.
- Staleness: `PathSlot.posKey` is the `positionsKey` of the channel's distances when the path was measured, and
  `SolutionInfo.geometryKey` the `geometryKey` a geometry-based solution used. Editing them makes the step 5/6
  gates and `unmuteBlockers` refuse the result; putting the old numbers back makes it valid again.
- Step 2 waits for the live master: the level noise starts only when logic.ts `liveMasterDb(s)`
  (MeterSnapshot.masterDb) is within `MASTER_READY_TOL_DB` (0.1 dB) of `AMP_STEP_MASTER_DB` (`masterReached`), or the
  wait ends when the master target is changed. If the meters do not confirm it within `masterWaitTimeoutMs(from, to)`
  = max(10 s, masterSettleMs + 2 s), the noise does not start and step-amp shows `masterTimeoutText` (the seconds
  rounded up: 10–13 s, as the README says). MasterRampTracker and StepEnv.masterReadyInMs are removed.
- Any running operation stops at once on 'panic', 'howl', 'fault', 'disarmed' and 'lifecycle' controller events
  (the autotune does not go through `measure()`).
- Step 6 reports the raw loop, the loop after neutralisation and the stability reduction (`loopOutcomeText`); the
  outcome is a warning when the gains were reduced or the neutralised loop is above `LOOP_GAIN_LIMIT_DB`.
  Separate zones are solved one zone at a time (the other speaker off), so logic.ts `mergeSeparateZones` merges the
  useful zones (`mergeZoneSolutions`: the speaker of a zone without a useful solution is switched off, never left
  playing its unsolved values: «Zona X: nessuna attenuazione utile, la cassa X resta spenta.», or, when that zone's own
  advice refused it, its «: i parametri non sono stati cambiati» reads «: la cassa X resta spenta» instead) and
  `mergedZonesLoop` checks the merged params with the speakers that play. Above `LOOP_GAIN_DANGER_DB` after
  neutralisation both channel gains are lowered by the excess. The per-zone loop advice, the per-harmonic trim-cut
  advice included («Armonica N (…): guadagno d'anello …», one speaker's figures), is replaced by one statement about
  the speakers that play (`mergedLoopAdvice(c, taps, alone)`: «Con entrambe le casse attive …» or «Con la sola cassa X
  attiva …»; `isLoopAdvice`; a zone the solver refused for stability keeps its reason). Each zone's broadband and
  per-harmonic predictions are then recomputed on the merged params (calibration.ts `predictOutcome`: through the loop
  of both speakers, the merged cap included). With both on, a zone whose delay alone (predictOutcome's `delayOnly`)
  predicts ≥ 0.05 dB more than its trims plays it instead when the merge gains in total with a headline no lower, with
  «Zona X: con entrambe le casse attive le correzioni per armonica attenuerebbero A dB, solo ritardo e guadagno B dB: la
  zona X usa solo ritardo e guadagno.» (seed 244, zone R: 2.4 → 7.8 dB). The playing zones' lines on what they
  attenuate («Attenuazione prevista solo …», harmonics out of the trims' range) are restated from the merged
  predictions (`restateZoneFigures`; seed 375 read «solo 5,7 dB» with zone R at 4.7 dB merged). A merge that leaves a
  zone below 1 dB keeps one speaker off and the other zone is merged and checked again on its own: alone a zone keeps
  its own solve's values and prediction, and the zone kept is the one that gives more alone (the zone left below 1 dB
  is switched off unless alone it gives ≥ 0.05 dB more than the other): «Zona X: con entrambe le casse attive, che
  tornano insieme al microfono di riferimento, l'attenuazione prevista scende a Y dB: la cassa X resta spenta.», or
  «Zona X: …, l'attenuazione prevista della zona W scende a Y dB: la cassa X resta spenta, perché da sola la zona W
  attenua di più (A dB contro B dB).». The off zone's lines on what it attenuates go with it, and the notes on a speaker
  kept off come before the loop statement. `refused` is a safety net: «Zona X: tenendo conto di ciò che la cassa
  rimanda al microfono di riferimento, l'attenuazione prevista scende a Y dB: i parametri non sono stati cambiati.»,
  with no «resta spenta» note. The headline is the lowest prediction among the zones that play. With one zone left,
  step 6's summary adds «Solo la zona X è attenuata: la cassa Y resta spenta (vedi i suggerimenti).» (a warning), and
  its loop figures, gain reduction (a channel's total: its zone's own cut plus the merged one) and causality are the
  playing zones' only. The per-harmonic list then has that zone's lines only. The autotune's zone radio disables a
  zone whose speaker is off (live params) and selects the zone that plays on every render (a rebuilt step starts on
  zone L), with a help line naming the zone it tunes; step-align `autotune()` refuses an off zone too
  (`tuneZoneOffText`). Before, a zone the other speaker's feedback undid at a harmonic still read «60 dB, limite di
  calcolo per un nullo esatto» (simulator: 41.6 dB predicted, 6.6 dB measured). A stability cut below 0.05 dB is not
  shown in the summary (`loopOutcomeText`).
- The step-6 autotune passes `loopGuard(cal, start, fs, taps)` as `AutotuneOptions.allowed` (no probe raises the
  neutralised loop above `LOOP_GAIN_DANGER_DB`, or above the start's own loop when that is higher) and applies its
  result with ops.ts `applyAutotuneResult`: only when `improved`; only its channels, over the live params
  (`liveWithChannels`: its own params carry the run's start f0 and master); through `capLoopGain`; with
  `calibratedMeasurements(r, capped.limitedDb)` for one zone (kept up to `LOOP_GUARD_TOL_DB` = 0.05 dB of cap, each at
  most `gainLimitedAttenuationDb` of it; null beyond). The outcome line is logic.ts `autotuneOutcomeText`: «Attenuazione
  nella zona da A dB a B dB (rispetto al bypass all’inizio e alla fine): valori migliorati applicati.» / «…: nessun
  miglioramento, valori invariati.».
- Error-mic texts name the input the engine checks (validation.ts `errorMicInput`): input 1 when the reference is
  on input 2 (the meters' note, `measurePath`'s missing-mic reason, step 5's loopback checklist and notes). The
  quick check's loopback signature names the reference input the sweep was recorded on.
- Third-octave bands are named by their nominal centres (spectrum-view `thirdOctaveLabel`: 20 Hz … 31,5 Hz …
  1,25 kHz) in the console readout (`spectrumReadout`: once) and in step 7's list.
- Tonal mode: step 6 lists `SolutionInfo.harmonics` (logic.ts `harmonicPredictions(outcomes, extraLimitedDb = 0)`:
  the outcomes' closed-loop predictions as they are — step 6 passes no extra cap; clipping them to the open-loop
  gain-error bound of their own uniform cut read up to 1 dB low, also in step 7's «prevista» —, separate zones the
  ones recomputed with `predictOutcome`; `harmonicPredictionText(x, errorMic)`: with the trims and, in brackets, with
  delay and gain alone under their own stability margin (HarmonicSolution.untrimmedDb); capped harmonics flagged, « · correzione ridotta di X dB per la stabilità» for
  `HarmonicPrediction.loopLimitedDb`, separate zones prefixed by zone) and, for one shared zone, passes
  `calibratedPredictions(...)` to applyParams `harmonics`. A prediction at its model's ceiling reads as that limit
  (`ceilingText`, `predictedDbText`, `predictedOutcomeText`; the model from `solvedWithErrorMic(SolutionInfo.method)`):
  «30 dB, limite di precisione del modello senza mic d’errore» or «60 dB, limite di calcolo per un nullo esatto del
  modello misurato», in step 6's summary and list and step 7's «prevista». The autotune driver (ops.ts) implements
  `measureHarmonicsDb` (autotune.ts harmonicLevelsDb with the search's drift tolerance, 400 ms bypass settle), so the
  tonal autotune refines the trims, and applies only the probes' channels over the live parameters (`liveWithChannels`:
  applied whole, every probe set tonal.f0 back to the run's start value and the follower moved it again within 2 s).
  The search asks for the harmonics of the start f0 while the bank follows the live one: the driver (made right before
  runAutotune, it notes that f0) reads them × live f0 / start f0 (the live f0 halfway through the window), and throws —
  the run stops and restores its start, over the live f0 — once the live f0 is further from the start than the search's
  tolerance (`harmonicTolerancesHz`, min(1/(2Q), 1 %)): «Autotune interrotto (la fondamentale si è spostata da X a Y Hz
  durante la misura, …)». A +2 % step of the source used to leave every later window reading the noise floor and the run
  completed with a false 37 dB improvement. The step shows `autotuneHarmonicTexts` (attenuation, and «prima» against
  `startBypassDb`; none for an aborted run: its last window is the probe that stopped it, the start values are back).
  Step 7 lists app/spectrum.ts `harmonicAttenuationDb` per harmonic (`harmonicMeasuredText`) next to
  `expectedHarmonic(state, solution, h)`: `calibratedHarmonics` only — never the step-6 prediction (one zone's is in
  calibratedHarmonics whenever it still applies; the fallback showed it after a capped autotune, a preset or a
  neutraliser change) — and none once h·f0 moved off the entry's frequency by more than `harmonicTolerancesHz`
  («Aggancia al picco») or once a channel differs from `calibratedChannels` at that harmonic (enabled, polarity,
  delay, Δ, gain, its trim: an edit, «Azzera armoniche»). `mergeZoneSolutions` deep-copies each zone's trims.

## Simulator (tests/sim, backward-compatible additions)
- `NoiseSpec` gains three optional fields: `broadbandGain(t)`, `toneGain(t, h)` and `custom(t)`.
- `Plant` gains `feedbackScale: [number, number]` (default [1, 1]), which multiplies each F_ch path; paths can have
  a diffuse `tail` (RT60) and the zone can have noise the reference does not hear (`zoneNoise`).
- `RigOptions` gains `onMessage(msg)`, which sees every message, meters included.
- The harness `autotuneDriver()` also implements measureHarmonicsDb (harmonicLevelsDb with the search's drift
  tolerance; the power sum is logged); `SimRig.timeS` and `Plant.elapsedSamples` give the simulated time.
- Tonal sims: 4b (two speakers, tonal, joint trims), 6c (per-harmonic autotune), 6d (the same with the wizard's 4 s
  windows while f0 drifts 0.2 %: the reported per-harmonic attenuation stays within 6 dB of the real one), realism 14
  (tonal stress case: HP 40 Hz, barrier, wall-like primary, speaker path with roll-off, reflections and an 80 ms
  tail; every harmonic ≥ 12 dB), closed-loop 9 (feedback at 300 Hz only: six late reflections × 0.35 one period
  apart, a 256-tap neutraliser; the per-harmonic margin cuts harmonic 3's trims by 6.3 dB where a uniform cut costs
  every harmonic as much — simulated per harmonic 58.7 / 54.9 / 6.6 / 50.8 dB against 5.8 / 5.8 / 6.6 / 5.8 dB — the
  loop on a 0.05 Hz grid ≤ −6 dB, the predictions within 0.5 dB of the simulator).
- `SimRig.listen(cb)`; harness.ts `SimEngine`: the rig as an EngineLike, so a real AppController (live f0 follower
  included) runs on it, the test advancing the rig and the fake timers together; `autotuneDriver().log.harmonics`
  ({ bypass, levels, t } per window).
- tests/sim/autotune-f0.test.ts (the wizard's driver on that controller, 4 s windows): A — f0 drifting 0.4 % over the
  run: every probe keeps the follower's f0 (never back to the start value); B — a +6 dB ramp: each harmonic's
  reference is the bypass level at the end window; C — the source's f0 steps mid-run: +0.8 % (inside the tolerance) is
  read where the bank now is, +2 % stops the run and restores the start's trims at the live f0.
- tests/unit/loop-margin.test.ts on seeded random tonal plants (loop-margin-plants.ts, the review's generator; the
  fine-grid loop and the closed-loop attenuation evaluated independently of the solver): the uniform cap kept when it
  predicts more (seed 435), cut trims given back after a uniform cut (446), the advice wording, the loop within the
  margin between bands (569), step 6's closed-loop predictions unclipped (257), separate zones predicted with both
  speakers (1, 39), a zone left below 1 dB by the other speaker kept off and the other zone played alone (450; 400:
  the zone that gives more alone kept), both zones kept on (1), a zone playing its delay alone with both speakers on
  (244), the zones' figures restated on the merged params (375), the refusal as a safety net (26 merged on feedback
  paths 18 dB louder), `mergedLoopAdvice` naming the speakers that play.
- tests/unit/solver-invariants.test.ts on the same plants (every 5th seed of 0–599 and the reviews' repros, each zone
  of separate zones alone): a refusal is never useful and quotes its figure; an applied outcome is reproduced exactly
  by predictOutcome; the total never below the delay alone (nor below a uniform-cut delay alone on the plant); a
  harmonic below untrimmedDb only under the rule of HarmonicSolution.untrimmedDb (give-back, lowered give-back and
  uniform cut checked on the fine grid); predictions equal to the plant's closed loop; the fine-grid loop within the
  margin; a uniform cut only where the loop needs it, with advice quoting a loop above the margin; no howl-risk
  refusal for a negligible cut (21).
- tests/unit/wizard-zones.test.ts, step 6 on the fake DOM for separate zones with the real solver on seeded plants
  (calibration.ts `measurePrimary` and `measurePath` mocked to the plant's paths). Seed 488 keeps zone R only: applied
  params, summary, advice order, channels line, per-harmonic list, the autotune zone choice (also once the step is
  built again) and its refusal without un-muting; seed 1 keeps both; 399: the summary's stability cut is a channel's
  total; 518 (delay mode): causality of the playing zone only, the off zone's refusal read as its speaker kept off.
  A saved solution with speaker R off (492): reused through the start question it stays off with the loop within the
  margin, and «Controllo rapido» neither sweeps nor judges that speaker.

## End-to-end tests (e2e/, Playwright, real Chrome with a fake microphone)
- `npm run test:e2e` builds the e2e bundle (`npm run build:e2e`: tsc -b, then vite build --mode e2e, which installs
  `window.__anc`) and serves it with the production CSP; Chrome plays a generated WAV as the microphone
  (global-setup.ts, 48 kHz mono 16-bit, deterministic) and runs muted (--mute-audio). One project per fixture, each
  running only its spec (helpers.ts `FIXTURE_SPECS`): tone-200hz and noise → anc.spec.ts (8 tests each),
  harmonic-100hz → trims.spec.ts (5 tests). e2e/app-page.ts: the guard fixture (fails a test on console errors, CSP
  violations, fault / processorerror / howl events) and the page helpers shared by the specs.
- Fixtures: tone-200hz.wav and noise.wav 10 s; harmonic-100hz.wav 100 Hz + harmonics 2–4 (amplitudes 1 / 0.6 / 0.4 /
  0.3, phases 0 / 45 / 110 / 250°), −20 dBFS peak, a whole number of periods, 30 s: Chrome's fake device zero-fills
  5–14 ms where it loops the file (from position 0 at every getUserMedia) and the bank rings for a second after it; at
  10 s that fell on the rise test's +6 dB arrival (9.5 s after the mic opened). No trims test reaches 30 s.
- trims.spec.ts measures each output's transfer Out/Cond and Cond/Ref at h·f0 by lock-in over whole periods of the
  tap blocks (sent to Node as base64 Float32), on windows whose two halves agree (≤ 0.002 dB / 0.01°: soft-start,
  glides and 6 dB/s rises waited out, no fixed sleeps), tolerance 0.01 dB / 0.05° (measured errors ≈ 1e-7 dB). f0
  following is off (`controller.setF0Follow(false)`: no UI). (a) typed and stepped trims change only their harmonic by
  10^(g/20)·e^{−jθ}, not the other channel nor the cond tap, and act only in tonal mode; (b) a fast keyboard drag of
  the fundamental's trim — at master 0 dB and channel gain −6 dB (output ≈ −26 dBFS, above the howl detector's
  −40 dBFS floor; at the defaults «no howl» could not fail), a +6 dB budget rise in 0.3 s (PageUp every 50 ms, the
  hand-over the engine holds the detector's levels for), then 225 more presses — never howls or auto-mutes and ends on
  +4 dB / −30°; (f) a −30 → +6 dB rise reaches the output at 6 dB/s (5.7–6.3 dB/s, never above −24 + 6·t dB) and
  +6 dB in a window starting between 5 s − 2 blocks and 5.5 s (the last 6 dB from the budget at once); (c) «Collega
  L/R» and «Azzera armoniche» (both channels while linked, its own unlinked); (d) the «Fase totale» readouts against
  the measured lag. Underrun auto-mutes on a loaded machine fail a test at once with their cause (no retry).
