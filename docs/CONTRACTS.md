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
  get currentDb(): number; get targetDb(): number; get currentLin(): number }
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
export class ResonatorBank { constructor(fs: number); configure(f0: number, harmonics: number, q: number, rampSamples?: number): void; process(x: number): number; reset(): void }
export function resonatorBankResponse(fs: number, f: number, f0: number, harmonics: number, q: number): Complex
```

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
export class ConditioningChain { constructor(fs: number, params?: EngineParams); configure(p: EngineParams): void; process(x: number): number; reset(): void }
/** Conditioning chain (DC blocker → HP if enabled → barrier LP if enabled → Bessel LP if enabled →
 *  resonator bank if phaseMode === 'tonal'), exactly as the engine implements it (at rest, i.e. the
 *  response of the glide's target). */
export function conditioningResponse(p: EngineParams, fs: number, f: number): Complex
/** Full channel response from reference input to final output (before limiter, assuming no limiting):
 *  conditioning · e^{-jω·computeChannelDelay(p, fs, ch).samples} · (hilbert rotation if mode hilbert,
 *  i.e. cosΔ·A − sinΔ·B) · (−1 if polarityInvert) · channel gain · master gain. 0 if channel disabled. */
export function channelResponse(p: EngineParams, fs: number, f: number, ch: 0 | 1): Complex
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
  HOWL_SAT_GAP_MS = 200, HOWL_SAT_MS = 300, HOWL_SAT_LEVEL_DB = 12, HOWL_SAT_GR_DB = 0.5;
/** Size, lag range and pace of a PeriodicityAnalyzer (defaults: 8192 points, lags HOWL_LOOP_MIN_MS…
 *  HOWL_LOOP_MAX_MS, capped at half the FFT size, each analysis spread over 60 ms). */
export interface PeriodicityOptions { log2N?: number; lagMinMs?: number; lagMaxMs?: number; spreadMs?: number }
/** Largest normalized autocorrelation |ρ(τ)| = |Σx[n]x[n−τ]| / √(Σx[n]²·Σx[n−τ]²) at a local maximum over
 *  lags 5–600 ms (default options). The sums run over the latest window: FFT size minus the longest lag,
 *  ≈ 0.77 s at 6 kHz with 8192 points. A new analysis starts every everyMs. Its work (copy, a DIF FFT of the
 *  two real sequences packed into one complex one, cross-spectrum, inverse DIT FFT, lag search) is cut into
 *  slices of `budget` units per push(), so each analysis completes in ≈ spreadMs (60 ms). push() is
 *  allocation-free.
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
  reset(): void
}
/** Pre-limiter output of one channel. Decimates to ~12 kHz (round(fs/12000)), 512-point Hann FFT every 256
 *  decimated samples. A PeriodicityAnalyzer runs at half that rate (~6 kHz, band ≤ ~2.4 kHz, after a
 *  4th-order Butterworth anti-alias filter), and a second one for long loops at half that again (~3 kHz,
 *  band ≤ ~1.2 kHz, same anti-alias): 8192 points, lags 500–1100 ms (the 500 ms channel delay plus up to
 *  600 ms of latency and acoustic delay), a new analysis every 400 ms spread over 120 ms. Triggers (latched
 *  until reset()):
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
 *  Not seen by the periodicity: howls above ~2.4 kHz with no narrow peak (above ~1.2 kHz for loops of
 *  600–1100 ms), and loops longer than 1.1 s; the engine backstop covers gross cases of these. push()
 *  allocates nothing; its extra work in any one sample is bounded by the analyzers' slices.
 *  holdLevels(ms): the conditioning filters glide to a new response for the next `ms` (a live filter
 *  change, engine): those samples are recorded with an unknown commanded gain, so the frames that contain
 *  them are unreliable and end every level track (they restart on the new response, as after a
 *  commanded-gain jump); the full-rate rules and the periodicity are not affected. */
export class HowlDetector { constructor(fs: number); push(preLimiter: number, commandedGainDb: number, limiterGrDb: number, ceilingDb: number): void;
  holdLevels(ms: number): void; get state(): HowlState; reset(): void }
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
cond = ConditioningChain(x̂) (bypassed in test mode); delayLine.write(cond);
per channel ch: v = SmoothDelayTap.next(line) → [PhaseRotator if mode 'hilbert'] → polarity (±1) →
× (channel gain + master) ramp → × softStart × mute × bypass × reconfigure(ch) × channelSoftStart(ch) ×
measureGate → + stimulus × master (stimulus only on the measured channel; the master GainRamp scales only
the stimulus) → pre = v → howl[ch].push(pre, commandedGainDb, limiter[ch].gainReductionDb, ceilingDb)
→ y = limiter[ch].process(v) (limiter does the final clip) → out = armed ? y : 0.
Then feedback.pushOutputs(outL, outR); meters (raw x, err, outL, outR); tap values [x, cond, outL, outR, err].
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
- Delay: computeChannelDelay(params, fs, ch); setTarget(..., jump) when |Δ| ≥ DELAY_JUMP_THRESHOLD_MS, in tonal
  mode also when |Δ| > T/2 (a D mod T wrap: slewed, it would sweep the phase the long way round, through
  in-phase), or on mode change; MeterSnapshot.delaySamples/delayClamped report it.
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
  loopLimitedDb?: number }
export interface SolveOptions { feedbackTaps?: number /* neutraliser length loaded, default DEFAULT_FEEDBACK_TAPS */ }
export const LOOP_GAIN_LIMIT_DB = -20, LOOP_GAIN_DANGER_DB = -6, DEFAULT_FEEDBACK_TAPS = 256
export const FEEDBACK_TAPS_RANGE = { min: 128, max: 1024 }   // FEEDBACK_MAX_TAPS
export interface LoopGains { rawDb: number; atHz: number; afterDb: number }
/** Loop gain of `p` with both channels through the feedback paths, raw and after neutralisation with
 *  `taps`-tap FIRs; null without feedback data, at another rate or when not finite (the solvers' check). */
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
  measureErrorDb(ms: number): Promise<number>; measureBypassDb(ms: number): Promise<number>; signal?: AbortSignal }
export interface AutotuneResult { params: EngineParams; startDb: number; endDb: number; bypassDb: number; steps: number; aborted: boolean; reason?: string }
export interface AutotuneOptions { fs: number; channels: (0 | 1)[]; rt60S: number; band: [number, number];
  /** noiseBandwidthHz of the zone noise: windows sized on it (never above the nominal band width). */
  effectiveBandHz?: number;
  /** Candidates for which this is false are skipped, never played (the start always is): the wizard passes
   *  logic.ts loopGuard, the loop-gain margin of the solve. */
  allowed?: (p: EngineParams) => boolean }
export const AUTOTUNE_BT = 150, MAX_WINDOW_MS = 4000
/** Window (B·T ≥ AUTOTUNE_BT, capped at MAX_WINDOW_MS), settling, σ of one window's level
 *  (10/ln10/√(B·T) dB) and the abort margin (1 dB at AUTOTUNE_BT, widened as 1/√(B·T) for capped windows). */
export function autotuneTiming(o: AutotuneOptions): { windowMs: number; settleMs: number; sigmaDb: number; abortMarginDb: number }
export function runAutotune(driver: AutotuneDriver, start: EngineParams, o: AutotuneOptions): Promise<AutotuneResult>
```
Search: delay mode → delay + gain; tonal/hilbert → Δ + gain; one channel at a time (other active); settle =
max(smoothing, crossfade, rt60) ; measurement windows with B·T ≥ AUTOTUNE_BT (150) on the effective noise bandwidth;
interleaved bypass windows; abort and restore the start params if error > bypass + the abort margin. The wizard
passes effectiveBandHz = noiseBandwidthHz of the zone being tuned and, in tonal mode, measures the level at the
harmonics of f0 (synchronous detection) rather than the band RMS.

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
- **Limiter-saturation backstop.** For howls the detector cannot see (above its periodicity bands, loops
  longer than 1.1 s). While armed, outside measurements and with no howl latched, a channel whose limiter gain
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
  user's ("Aggancia al picco").

## src/audio/calibration.ts
- `finishSolve` enforces the stability margin: when the neutralised loop exceeds `LOOP_GAIN_DANGER_DB` (−6 dB)
  both channel gains are lowered by `loopLimitedDb`; if that leaves less than 1 dB of predicted attenuation the
  params are not applied. Between `LOOP_GAIN_LIMIT_DB` (the plan's −20 dB) and −6 dB there is advice only (a FIR
  of a few ms cannot model the response near the low band edge; those loops are stable in the simulator).
- `predictedAttenuationDb` is recomputed for the params actually returned (clamping and loop limit included) and,
  with an error mic, includes the zone noise the reference does not hear (coherence).
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

## Guided setup (src/ui/wizard/)
- Flows: 'full' (steps 1–7), 'recal' (start → devices → amp → placement → measure → align → verify) and 'reuse'
  (start → devices → amp → **check** «Controllo rapido» → verify). The check (step-check.ts) plays one short sweep per
  active speaker on the speaker → reference path and compares it with the saved one (logic.ts `compareToBaseline`,
  `quickVerdict`, `QUICK_CHECK`): within 0.5 samples and 0.5 dB 'ok'; up to 6 dB / 3 ms 'corrected' (channel gain
  and delay corrected, the saved feedback path replaced so the FIRs follow); bigger, a flipped polarity or a
  loopback signature 'mismatch' (re-measure); a failed measurement 'unverified' (must be accepted explicitly).
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
- `StepEnv.masterReadyInMs()`: time until the engine's rate-limited master reaches the last master set (logic.ts
  `MasterRampTracker`, checked against the real GainRamp); step 2 starts the level noise only then. Since the
  engine fixes the master when a stimulus starts this is conservative, not required.
- Any running operation stops at once on 'panic', 'howl', 'fault', 'disarmed' and 'lifecycle' controller events
  (the autotune does not go through `measure()`).
- Step 6 reports the raw loop, the loop after neutralisation and the stability reduction (`loopOutcomeText`); the
  outcome is a warning when the gains were reduced or the neutralised loop is above `LOOP_GAIN_LIMIT_DB`.
  Separate zones are solved one zone at a time (the other speaker off), so logic.ts `mergedZonesLoop` checks the
  merged params with both speakers: above `LOOP_GAIN_DANGER_DB` after neutralisation both channel gains are
  lowered by the excess, the predicted attenuation is capped at what that gain error leaves
  (`gainLimitedAttenuationDb`), and the per-zone loop advice is replaced by one statement about both speakers
  (`isLoopAdvice`; a zone the solver refused for stability keeps its reason). `mergeZoneSolutions` switches off the
  speaker of a zone without a useful solution (never its unsolved values).
- The step-6 autotune passes `loopGuard(cal, start, fs, taps)` as `AutotuneOptions.allowed` (no probe raises the
  neutralised loop above `LOOP_GAIN_DANGER_DB`, or above the start's own loop when that is higher) and applies its
  result through `capLoopGain`.
- Error-mic texts name the input the engine checks (validation.ts `errorMicInput`): input 1 when the reference is
  on input 2 (the meters' note, `measurePath`'s missing-mic reason, step 5's loopback checklist and notes). The
  quick check's loopback signature names the reference input the sweep was recorded on.
- Third-octave bands are named by their nominal centres (spectrum-view `thirdOctaveLabel`: 20 Hz … 31,5 Hz …
  1,25 kHz) in the console readout (`spectrumReadout`: once) and in step 7's list.

## Simulator (tests/sim, backward-compatible additions)
- `NoiseSpec` gains three optional fields: `broadbandGain(t)`, `toneGain(t, h)` and `custom(t)`.
- `Plant` gains `feedbackScale: [number, number]` (default [1, 1]), which multiplies each F_ch path; paths can have
  a diffuse `tail` (RT60) and the zone can have noise the reference does not hear (`zoneNoise`).
- `RigOptions` gains `onMessage(msg)`, which sees every message, meters included.
