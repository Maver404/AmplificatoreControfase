/**
 * L/R link: when linked, editing one channel applies the same change to the other. Numeric fields
 * keep the current L/R offset (the change is limited so that neither channel leaves its range,
 * otherwise the offset would be silently lost); toggles are copied.
 */
import { RANGES } from '../dsp/params.ts';
import type { ChannelParams } from '../dsp/params.ts';

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
  let delta = want - prev;
  const lo = Math.max(min - prev, min - o);
  const hi = Math.min(max - prev, max - o);
  delta = lo > hi ? 0 : clamp(delta, lo, hi);
  out[idx] = { ...out[idx], [changed]: prev + delta };
  out[other] = { ...out[other], [changed]: o + delta };
  return out;
}
