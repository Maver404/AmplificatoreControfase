/**
 * Parameter model shared by the AudioWorklet (which enforces it) and the UI (which edits it).
 * All clamping happens here so that the DSP never trusts the UI.
 *
 * Phase convention (single, used everywhere):
 *   total phase = polarity (0° or 180°) + Δ, with Δ = phaseDeltaDeg ∈ [-180, 180], positive = lag.
 *   Tonal mode, at harmonic h·f0: total phase = polarity + h·Δ (through the wrapped delay D mod T) + θ_h,
 *   with θ_h = harmonicTrims[h − 1].phaseDeg (positive = lag, on top of the delay); the trim's gainDb scales
 *   that harmonic alone. Trims act only in tonal mode (see harmonic-trims.ts).
 */

export type PhaseMode = 'delay' | 'tonal' | 'hilbert';

/**
 * Complex trim of one harmonic in tonal mode, relative to the untrimmed channel at h·f0: the channel's response
 * there is multiplied by 10^(gainDb/20)·e^{-j·phaseDeg·π/180} (positive phase = lag).
 */
export interface HarmonicTrim {
  gainDb: number;
  phaseDeg: number;
}

/** Number of harmonic trims per channel (= RANGES.harmonics.max): entry h − 1 trims harmonic h. */
export const MAX_HARMONICS = 8;

export interface ChannelParams {
  enabled: boolean;
  /** true = output multiplied by -1 (the 180° of anti-phase). */
  polarityInvert: boolean;
  gainDb: number;
  /** Base delay in ms, added on top of the intrinsic minimum D_MIN_SAMPLES. */
  delayMs: number;
  /** Δ in degrees at fRef (delay/hilbert mode) or f0 (tonal mode); positive = lag. */
  phaseDeltaDeg: number;
  /**
   * Exactly MAX_HARMONICS entries, entry h − 1 for harmonic h. Tonal mode only; entries above tonal.harmonics
   * or whose h·f0 ≥ 0.45·fs are ignored. All 0 = the untrimmed channel (sample-identical).
   */
  harmonicTrims: HarmonicTrim[];
}

export interface FilterParams {
  hpEnabled: boolean;
  hpFreq: number;
  hpOrder: 1 | 2;
  /** 1st-order low-pass emulating a wall/window between mic and zone (-6 dB/oct, ~-90°). */
  barrierEnabled: boolean;
  barrierFreq: number;
  lpEnabled: boolean;
  /** -3 dB cutoff of the Bessel low-pass. */
  lpFreq: number;
  lpOrder: 2 | 4;
}

export interface TonalParams {
  f0: number;
  harmonics: number;
  q: number;
}

export interface EngineParams {
  channels: [ChannelParams, ChannelParams];
  filters: FilterParams;
  phaseMode: PhaseMode;
  /** Reference frequency at which Δ is expressed in delay and hilbert mode. */
  fRef: number;
  tonal: TonalParams;
  /** Master volume; values ≤ MASTER_MIN_DB mean silence. */
  masterDb: number;
  ceilingDb: number;
  /** Allows channel gain up to +12 dB. Never persisted or imported. */
  gainUnlocked: boolean;
  /** Output is headphones: ceiling forced to ≤ HEADPHONES_CEILING_DB. */
  headphones: boolean;
  /** Input channel carrying the reference mic. */
  refChannel: 0 | 1;
  /** Input channel carrying the error mic, or -1 when absent. */
  errorChannel: 0 | 1 | -1;
}

export const D_MIN_SAMPLES = 1;
export const MASTER_MIN_DB = -120;
export const HEADPHONES_CEILING_DB = -24;
export const MAX_DELAY_MS = 500;

export const SOFT_START_MS = 1500;
export const MUTE_RAMP_MS = 15;
export const PANIC_RAMP_MS = 10;
export const GAIN_SMOOTH_MS = 20;
/** Gain increases larger than BIG_JUMP_DB rise at no more than MAX_RISE_DB_PER_S. */
export const BIG_JUMP_DB = 6;
export const MAX_RISE_DB_PER_S = 6;
/** Delay slew limit for small changes, in samples per sample (~5 cents of Doppler). */
export const DELAY_SLEW = 0.003;
/** Delay changes larger than this jump with a crossfade instead of slewing. */
export const DELAY_JUMP_THRESHOLD_MS = 1;
export const DELAY_CROSSFADE_MS = 30;
export const LIMITER_RELEASE_MS = 100;

export const RANGES = {
  gainDb: { min: -60, max: 0, maxUnlocked: 12 },
  masterDb: { min: MASTER_MIN_DB, max: 0 },
  delayMs: { min: 0, max: MAX_DELAY_MS },
  phaseDeltaDeg: { min: -180, max: 180 },
  fRef: { min: 20, max: 2000 },
  ceilingDb: { min: -24, max: -1 },
  hpFreq: { min: 10, max: 200 },
  barrierFreq: { min: 10, max: 200 },
  lpFreq: { min: 100, max: 4000 },
  f0: { min: 20, max: 2000 },
  harmonics: { min: 1, max: 8 },
  q: { min: 2, max: 200 },
  trimGainDb: { min: -30, max: 6 },
  trimPhaseDeg: { min: -180, max: 180 },
} as const;

/**
 * Lowest tonal Q per number of harmonics (index = harmonics). Below it the band skirts of the resonator
 * bank overlap so much that no design is 0 dB and 0° at every harmonic (the phase convention "Δ at f0"
 * would no longer hold). The exact limit is ≈ 2.4 / 3.2 / 4.0 / 4.9 for 5 / 6 / 7 / 8 harmonics over
 * f0 20–2000 Hz at 44.1–192 kHz; the table keeps ≥ 0.6 of margin.
 */
export const TONAL_MIN_Q: readonly number[] = [2, 2, 2, 2, 2, 3, 4, 5, 6];

/** Lowest allowed tonal Q for `harmonics` (rounded and clamped to the harmonics range; NaN → strictest). */
export const minTonalQ = (harmonics: number): number => {
  const { min, max } = RANGES.harmonics;
  const h = Number.isFinite(harmonics) ? Math.round(Math.min(max, Math.max(min, harmonics))) : max;
  return TONAL_MIN_Q[h];
};

/** MAX_HARMONICS untrimmed entries (a new array of new objects). */
export const neutralTrims = (): HarmonicTrim[] =>
  Array.from({ length: MAX_HARMONICS }, () => ({ gainDb: 0, phaseDeg: 0 }));

const defaultChannel = (): ChannelParams => ({
  enabled: true,
  polarityInvert: true,
  gainDb: -20,
  delayMs: 0,
  phaseDeltaDeg: 0,
  harmonicTrims: neutralTrims(),
});

const DEFAULT_CHANNEL: ChannelParams = defaultChannel();

/** Safe first-run defaults. */
export const DEFAULT_PARAMS: EngineParams = {
  channels: [defaultChannel(), defaultChannel()],
  filters: {
    hpEnabled: true,
    hpFreq: 20,
    hpOrder: 2,
    barrierEnabled: false,
    barrierFreq: 50,
    lpEnabled: true,
    lpFreq: 500,
    lpOrder: 2,
  },
  phaseMode: 'delay',
  fRef: 100,
  tonal: { f0: 100, harmonics: 4, q: 30 },
  masterDb: -30,
  ceilingDb: -12,
  gainUnlocked: false,
  headphones: false,
  refChannel: 0,
  errorChannel: -1,
};

export const dbToLin = (db: number): number => (db <= MASTER_MIN_DB ? 0 : Math.pow(10, db / 20));
export const linToDb = (lin: number): number =>
  lin <= 0 ? MASTER_MIN_DB : Math.max(MASTER_MIN_DB, 20 * Math.log10(lin));

const clamp = (v: unknown, min: number, max: number, fallback: number): number => {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : fallback;
  return Math.min(max, Math.max(min, n));
};
const bool = (v: unknown, fallback: boolean): boolean => (typeof v === 'boolean' ? v : fallback);
const oneOf = <T>(v: unknown, options: readonly T[], fallback: T): T =>
  options.includes(v as T) ? (v as T) : fallback;

/** Exactly MAX_HARMONICS new entries: clamped, missing or non-finite values 0, extra entries dropped. */
function clampTrims(t: unknown): HarmonicTrim[] {
  const src: unknown[] = Array.isArray(t) ? t : [];
  const { trimGainDb: g, trimPhaseDeg: ph } = RANGES;
  return Array.from({ length: MAX_HARMONICS }, (_, i) => {
    const e = src[i] as Partial<HarmonicTrim> | null | undefined;
    return { gainDb: clamp(e?.gainDb, g.min, g.max, 0), phaseDeg: clamp(e?.phaseDeg, ph.min, ph.max, 0) };
  });
}

function clampChannel(c: Partial<ChannelParams> | undefined, gainMax: number): ChannelParams {
  const d = DEFAULT_CHANNEL;
  return {
    enabled: bool(c?.enabled, d.enabled),
    polarityInvert: bool(c?.polarityInvert, d.polarityInvert),
    gainDb: clamp(c?.gainDb, RANGES.gainDb.min, gainMax, d.gainDb),
    delayMs: clamp(c?.delayMs, RANGES.delayMs.min, RANGES.delayMs.max, d.delayMs),
    phaseDeltaDeg: clamp(c?.phaseDeltaDeg, RANGES.phaseDeltaDeg.min, RANGES.phaseDeltaDeg.max, d.phaseDeltaDeg),
    harmonicTrims: clampTrims(c?.harmonicTrims),
  };
}

/**
 * Returns a fully valid deep copy of `p`: every number clamped to its range, NaN/missing fields
 * replaced by defaults, gain capped at 0 dB unless unlocked, ceiling capped for headphones, tonal Q
 * raised to minTonalQ(harmonics), harmonicTrims padded/truncated to MAX_HARMONICS (missing/NaN → 0).
 */
export function clampParams(p: Partial<EngineParams> | undefined): EngineParams {
  const d = DEFAULT_PARAMS;
  const gainUnlocked = bool(p?.gainUnlocked, d.gainUnlocked);
  const headphones = bool(p?.headphones, d.headphones);
  const gainMax = gainUnlocked ? RANGES.gainDb.maxUnlocked : RANGES.gainDb.max;
  const f = p?.filters;
  const df = d.filters;
  const t = p?.tonal;
  const harmonics = Math.round(clamp(t?.harmonics, RANGES.harmonics.min, RANGES.harmonics.max, d.tonal.harmonics));
  const ceilingMax = headphones ? HEADPHONES_CEILING_DB : RANGES.ceilingDb.max;
  return {
    channels: [clampChannel(p?.channels?.[0], gainMax), clampChannel(p?.channels?.[1], gainMax)],
    filters: {
      hpEnabled: bool(f?.hpEnabled, df.hpEnabled),
      hpFreq: clamp(f?.hpFreq, RANGES.hpFreq.min, RANGES.hpFreq.max, df.hpFreq),
      hpOrder: oneOf(f?.hpOrder, [1, 2] as const, df.hpOrder),
      barrierEnabled: bool(f?.barrierEnabled, df.barrierEnabled),
      barrierFreq: clamp(f?.barrierFreq, RANGES.barrierFreq.min, RANGES.barrierFreq.max, df.barrierFreq),
      lpEnabled: bool(f?.lpEnabled, df.lpEnabled),
      lpFreq: clamp(f?.lpFreq, RANGES.lpFreq.min, RANGES.lpFreq.max, df.lpFreq),
      lpOrder: oneOf(f?.lpOrder, [2, 4] as const, df.lpOrder),
    },
    phaseMode: oneOf(p?.phaseMode, ['delay', 'tonal', 'hilbert'] as const, d.phaseMode),
    fRef: clamp(p?.fRef, RANGES.fRef.min, RANGES.fRef.max, d.fRef),
    tonal: {
      f0: clamp(t?.f0, RANGES.f0.min, RANGES.f0.max, d.tonal.f0),
      harmonics,
      q: clamp(t?.q, minTonalQ(harmonics), RANGES.q.max, d.tonal.q),
    },
    masterDb: clamp(p?.masterDb, RANGES.masterDb.min, RANGES.masterDb.max, d.masterDb),
    ceilingDb: clamp(p?.ceilingDb, RANGES.ceilingDb.min, ceilingMax, d.ceilingDb),
    gainUnlocked,
    headphones,
    refChannel: oneOf(p?.refChannel, [0, 1] as const, d.refChannel),
    errorChannel: oneOf(p?.errorChannel, [0, 1, -1] as const, d.errorChannel),
  };
}

/** A valid copy of `p` with every harmonic trim neutral: the untrimmed channels (see trimForTarget). */
export function withoutTrims(p: EngineParams): EngineParams {
  const q = clampParams(p);
  for (const c of q.channels) c.harmonicTrims = neutralTrims();
  return q;
}

/** Frequency at which Δ is expressed for the current mode. */
export const phaseReferenceHz = (p: EngineParams): number => (p.phaseMode === 'tonal' ? p.tonal.f0 : p.fRef);

export interface ChannelDelay {
  /** Total read delay of the channel tap in samples (≥ D_MIN_SAMPLES). */
  samples: number;
  /** True when the requested delay was below the minimum and had to be limited. */
  clamped: boolean;
}

/**
 * The one place where (base delay, Δ, mode) become the actual tap delay. The engine, H_dsp and
 * the calibration code must all use this function so they never disagree.
 *  - delay:   D_MIN + base + Δ/(360·fRef), limited at D_MIN
 *  - tonal:   D_MIN + ((base + Δ/(360·f0)) mod T), T = 1/f0 — whole periods are free for periodic noise
 *  - hilbert: D_MIN + base (Δ is applied by the Hilbert rotation, not by delay)
 */
export function computeChannelDelay(p: EngineParams, fs: number, ch: 0 | 1): ChannelDelay {
  const c = p.channels[ch];
  const base = (c.delayMs / 1000) * fs;
  if (p.phaseMode === 'hilbert') return { samples: D_MIN_SAMPLES + base, clamped: false };
  const fPhase = phaseReferenceHz(p);
  const extra = (c.phaseDeltaDeg / 360) * (fs / fPhase);
  if (p.phaseMode === 'tonal') {
    const period = fs / p.tonal.f0;
    let d = (base + extra) % period;
    if (d < 0) d += period;
    return { samples: D_MIN_SAMPLES + d, clamped: false };
  }
  const want = base + extra;
  return want < 0 ? { samples: D_MIN_SAMPLES, clamped: true } : { samples: D_MIN_SAMPLES + want, clamped: false };
}

/** Ring-buffer capacity (power of two) that holds the longest possible delay plus interpolation taps. */
export const delayCapacity = (fs: number): number => {
  const need = Math.ceil(((MAX_DELAY_MS + 30) / 1000) * fs) + 8;
  let n = 1;
  while (n < need) n <<= 1;
  return n;
};
