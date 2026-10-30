/**
 * L/R link: when linked, editing one channel applies the same change to the other. Numeric fields
 * keep the current L/R offset (the change is limited so that neither channel leaves its range,
 * otherwise the offset would be silently lost); toggles are copied. Harmonic trims are linked like the
 * numeric fields, entry by entry and field by field (gainDb, phaseDeg): each channel keeps its own trims
 * (the solver sets them per channel) and a linked edit moves both by the same amount. A trim phase is an
 * angle: both move by the same delta, wrapped into (−180°, 180°], never limited (±180° is one point, so a
 * range limit would block an edit whenever the other channel sits near it). Clearing the trims
 * («Azzera armoniche», clearTrims) is an absolute reset, not an edit by a delta: like a toggle it is copied.
 */
import { MAX_HARMONICS, RANGES, neutralTrims } from '../dsp/params.ts';
import type { ChannelParams, HarmonicTrim } from '../dsp/params.ts';

export type ChannelPair = [ChannelParams, ChannelParams];

const NUMERIC = ['gainDb', 'delayMs', 'phaseDeltaDeg'] as const;
type NumericField = (typeof NUMERIC)[number];

const isNumeric = (k: keyof ChannelParams): k is NumericField => (NUMERIC as readonly string[]).includes(k);

export function fieldRange(field: NumericField, gainUnlocked: boolean): { min: number; max: number } {
  switch (field) {
    case 'gainDb':
      return { min: RANGES.gainDb.min, max: gainUnlocked ? RANGES.gainDb.maxUnlocked : RANGES.gainDb.max };
    case 'delayMs':
      return RANGES.delayMs;
    case 'phaseDeltaDeg':
      return RANGES.phaseDeltaDeg;
  }
}

const clamp = (v: number, min: number, max: number): number => Math.min(max, Math.max(min, v));

/**
 * The shared change of one linked numeric value: `want` − `prev` limited so that both `prev` and the other
 * channel's `o` stay in [min, max] (the offset is preserved exactly).
 */
function linkedDelta(prev: number, o: number, want: number, min: number, max: number): number {
  const lo = Math.max(min - prev, min - o);
  const hi = Math.min(max - prev, max - o);
  return lo > hi ? 0 : clamp(want - prev, lo, hi);
}

const TRIM_FIELDS = ['gainDb', 'phaseDeg'] as const;
const TRIM_RANGE = { gainDb: RANGES.trimGainDb, phaseDeg: RANGES.trimPhaseDeg } as const;

/** A trim value as stored: finite and in range (anything else 0, as clampParams). */
const trimValue = (v: number | undefined, field: (typeof TRIM_FIELDS)[number]): number =>
  v !== undefined && Number.isFinite(v) ? clamp(v, TRIM_RANGE[field].min, TRIM_RANGE[field].max) : 0;

/** An angle in degrees wrapped into (−180, 180]. */
export const wrapPhaseDeg = (d: number): number => d - 360 * Math.ceil((d - 180) / 360);

/** New harmonicTrims of both channels ([L, R]) after channel `idx` asked for `next` (see the file comment). */
function trimsChange(channels: ChannelPair, idx: 0 | 1, next: readonly HarmonicTrim[], linked: boolean): [HarmonicTrim[], HarmonicTrim[]] {
  const other: 0 | 1 = idx === 0 ? 1 : 0;
  const out: [HarmonicTrim[], HarmonicTrim[]] = [[], []];
  for (let k = 0; k < MAX_HARMONICS; k++) {
    const prev = channels[idx].harmonicTrims[k];
    const o = channels[other].harmonicTrims[k];
    const a: HarmonicTrim = { gainDb: 0, phaseDeg: 0 };
    const b: HarmonicTrim = { gainDb: 0, phaseDeg: 0 };
    for (const f of TRIM_FIELDS) {
      const p0 = trimValue(prev?.[f], f);
      const o0 = trimValue(o?.[f], f);
      const raw = Array.isArray(next) ? next[k]?.[f] : undefined;
      // A missing or non-finite value leaves the field as it is.
      const want = raw !== undefined && Number.isFinite(raw) ? trimValue(raw, f) : p0;
      if (!linked) {
        a[f] = want;
        b[f] = o0;
        continue;
      }
      if (f === 'phaseDeg') {
        a[f] = want;
        // An unchanged phase leaves the other one exactly as it is (−180° stays −180°).
        b[f] = want === p0 ? o0 : wrapPhaseDeg(o0 + want - p0);
        continue;
      }
      const d = linkedDelta(p0, o0, want, TRIM_RANGE[f].min, TRIM_RANGE[f].max);
      a[f] = p0 + d;
      b[f] = o0 + d;
    }
    out[idx].push(a);
    out[other].push(b);
  }
  return out;
}

/**
 * «Azzera armoniche» on channel `idx`: every trim of that channel to 0, and, when linked, of the other one too
 * (an absolute reset is copied like a toggle; moving the other channel by the cleared amounts would give it
 * corrections it never had). The other channel's trims are otherwise left exactly as they are.
 */
export function clearTrims(channels: ChannelPair, idx: 0 | 1, linked: boolean): ChannelPair {
  const zero = (c: ChannelParams): ChannelParams => ({ ...c, harmonicTrims: neutralTrims() });
  const keep = (c: ChannelParams): ChannelParams => ({ ...c, harmonicTrims: c.harmonicTrims.map((t) => ({ ...t })) });
  const out: ChannelPair = [keep(channels[0]), keep(channels[1])];
  out[idx] = zero(channels[idx]);
  if (linked) out[idx === 0 ? 1 : 0] = zero(channels[idx === 0 ? 1 : 0]);
  return out;
}

/**
 * New channel pair after the user changed `changed` on channel `idx` to `next[changed]`.
 * Unlinked: only that channel changes (its numeric field clamped to range). Linked: see above.
 */
export function applyChannelChange(
  channels: ChannelPair,
  idx: 0 | 1,
  next: ChannelParams,
  changed: keyof ChannelParams,
  linked: boolean,
  gainUnlocked = false,
): ChannelPair {
  const out: ChannelPair = [{ ...channels[0] }, { ...channels[1] }];
  const other: 0 | 1 = idx === 0 ? 1 : 0;
  if (changed === 'harmonicTrims') {
    const t = trimsChange(channels, idx, next.harmonicTrims, linked);
    out[0] = { ...out[0], harmonicTrims: t[0] };
    out[1] = { ...out[1], harmonicTrims: t[1] };
    return out;
  }
  if (!isNumeric(changed)) {
    const v = next[changed] === true;
    out[idx] = { ...out[idx], [changed]: v };
    if (linked) out[other] = { ...out[other], [changed]: v };
    return out;
  }
  const { min, max } = fieldRange(changed, gainUnlocked);
  const raw = next[changed];
  const prev = channels[idx][changed];
  const want = Number.isFinite(raw) ? clamp(raw, min, max) : prev;
  if (!linked) {
    out[idx] = { ...out[idx], [changed]: want };
    return out;
  }
  const o = channels[other][changed];
  // Limit the shared change so that both channels stay in range (the offset is preserved exactly).
  const delta = linkedDelta(prev, o, want, min, max);
  out[idx] = { ...out[idx], [changed]: prev + delta };
  out[other] = { ...out[other], [changed]: o + delta };
  return out;
}
