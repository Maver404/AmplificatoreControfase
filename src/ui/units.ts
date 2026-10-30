/**
 * Pure helpers for the UI controls: stepping, unit conversion, slider mapping, parsing and
 * Italian number formatting. No DOM access, so everything here is unit-testable in Node.
 */

export interface UnitOption {
  id: string;
  label: string;
  /** Display units per base unit: display = base · factor. */
  factor: number;
  /** Fine step in display units (default: the control's base step · factor). */
  step?: number;
  /** Decimals shown in this unit (default: derived from the step). */
  decimals?: number;
  /** Other spellings accepted after a typed number (normalised as normalizeUnit does). */
  aliases?: readonly string[];
}

export type SliderScale = 'linear' | 'log' | 'cubic';

/** Number of discrete positions of the native range input. */
export const SLIDER_RESOLUTION = 10000;

export interface StepModifiers {
  shiftKey: boolean;
  altKey: boolean;
}

/** Removes binary floating-point noise (0.1 + 0.2 → 0.3) without touching meaningful digits. */
export const cleanFloat = (x: number): number => (Number.isFinite(x) ? Number(x.toPrecision(12)) : x);

/** Clamps to [min, max]; any non-finite input yields `fallback` (itself clamped). Never returns NaN. */
export function clampFinite(v: number, min: number, max: number, fallback: number = min): number {
  const n = Number.isFinite(v) ? v : Number.isFinite(fallback) ? fallback : min;
  return Math.min(max, Math.max(min, n));
}

/** Shift ×10, Alt ×0.1 (combined: ×1). */
export const modifierMultiplier = (m: StepModifiers): number => (m.shiftKey ? 10 : 1) * (m.altKey ? 0.1 : 1);

/**
 * Signed number of fine steps a key press means, or 0 if the key does not step.
 * `horizontal` = whether ArrowLeft/ArrowRight step (true on sliders, false in text fields).
 */
export function keySteps(key: string, m: StepModifiers, horizontal = true): number {
  const mult = modifierMultiplier(m);
  switch (key) {
    case 'ArrowUp':
      return mult;
    case 'ArrowDown':
      return -mult;
    case 'ArrowRight':
      return horizontal ? mult : 0;
    case 'ArrowLeft':
      return horizontal ? -mult : 0;
    case 'PageUp':
      return 100 * mult;
    case 'PageDown':
      return -100 * mult;
    default:
      return 0;
  }
}

/**
 * Moves `value` (base units) by `steps` fine steps expressed in display units, then clamps.
 * Stepping happens in the display unit so that e.g. 0.01 campioni is exact in samples.
 */
export function stepValue(
  value: number,
  steps: number,
  displayStep: number,
  factor: number,
  min: number,
  max: number,
): number {
  const f = Number.isFinite(factor) && factor > 0 ? factor : 1;
  const start = clampFinite(value, min, max, min);
  if (!Number.isFinite(steps) || !Number.isFinite(displayStep)) return start;
  // Round only in display space: rounding the converted base value would accumulate drift.
  const display = cleanFloat(start * f + steps * displayStep);
  return clampFinite(f === 1 ? display : display / f, min, max, start);
}

/** Integer variant: always moves by at least one unit in the step direction. */
export function stepInteger(value: number, steps: number, min: number, max: number): number {
  const start = Math.round(clampFinite(value, min, max, min));
  if (!Number.isFinite(steps) || steps === 0) return start;
  let next = Math.round(start + steps);
  if (next === start) next = start + Math.sign(steps);
  return clampFinite(next, min, max, start);
}

/** Rounds to the nearest multiple of `step` (no-op for step ≤ 0). */
export function snapToStep(v: number, step: number): number {
  if (!(step > 0) || !Number.isFinite(v)) return v;
  return cleanFloat(Math.round(v / step) * step);
}

/** Number of decimals needed to show multiples of `step` exactly (capped at 10). */
export function decimalsForStep(step: number): number {
  if (!(step > 0) || !Number.isFinite(step)) return 0;
  let d = 0;
  while (d < 10) {
    const scaled = step * 10 ** d;
    if (Math.abs(scaled - Math.round(scaled)) < 1e-9 * Math.max(1, scaled)) break;
    d++;
  }
  return d;
}

/** Value → slider position in [0, 1]. Log scale needs min > 0 (falls back to linear otherwise). */
export function valueToPosition(v: number, min: number, max: number, scale: SliderScale = 'linear'): number {
  if (!(max > min) || !Number.isFinite(v)) return 0;
  const x = Math.min(max, Math.max(min, v));
  let p: number;
  if (scale === 'log' && min > 0) p = Math.log(x / min) / Math.log(max / min);
  else if (scale === 'cubic') p = Math.cbrt((x - min) / (max - min));
  else p = (x - min) / (max - min);
  return Math.min(1, Math.max(0, p));
}

/** Slider position in [0, 1] → value. Inverse of valueToPosition. */
export function positionToValue(p: number, min: number, max: number, scale: SliderScale = 'linear'): number {
  if (!(max > min)) return min;
  const q = Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : 0;
  let v: number;
  if (scale === 'log' && min > 0) v = min * Math.pow(max / min, q);
  else if (scale === 'cubic') v = min + (max - min) * q * q * q;
  else v = min + (max - min) * q;
  return Math.min(max, Math.max(min, v));
}

/**
 * Parses a number typed by the user. Accepts Italian ("1,5", "1.234,5") and English ("1.5")
 * decimals, the typographic minus "−", trailing unit text ("12 ms", "−3dB") and "∞"/"inf".
 * Returns ±Infinity for infinities (the caller clamps), null when the text is not a number.
 * The unit text is ignored here: see parseNumberWithUnit.
 */
export function parseNumber(text: string): number | null {
  return parseNumberWithUnit(text)?.value ?? null;
}

/** Unit text in a comparable form: lower case, no spaces, "u"/Greek mu → micro sign. */
export function normalizeUnit(u: string): string {
  return u
    .trim()
    .replace(/\s+/g, '')
    .toLowerCase()
    .replace(/μ/g, 'µ')
    .replace(/^us$/, 'µs');
}

/** parseNumber plus the unit text typed after the number (normalizeUnit, '' when none). */
export function parseNumberWithUnit(text: string): { value: number; unit: string } | null {
  let s = text.trim().replace(/\s+/g, '').replace(/[−–—]/g, '-');
  if (s === '') return null;
  const inf = /^([-+]?)(∞|inf|infinity)([a-zà-ü°µμ%/]*)$/i.exec(s);
  if (inf) return { value: inf[1] === '-' ? -Infinity : Infinity, unit: normalizeUnit(inf[3] ?? '') };
  const hasDot = s.includes('.');
  const commas = (s.match(/,/g) ?? []).length;
  if (commas > 1) return null;
  if (hasDot && commas === 1) s = s.replace(/\./g, '').replace(',', '.');
  else if (commas === 1) s = s.replace(',', '.');
  const m = /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?/i.exec(s);
  if (!m) return null;
  const rest = s.slice(m[0].length);
  if (/[\d.,]/.test(rest)) return null;
  const n = Number(m[0]);
  return Number.isFinite(n) ? { value: n, unit: normalizeUnit(rest) } : null;
}

/** Spellings of the fixed units of the controls, with their display factor (value = base · factor). */
const FIXED_UNIT_ALIASES: Record<string, ReadonlyArray<readonly [string, number]>> = {
  db: [['db', 1]],
  dbfs: [['dbfs', 1], ['db', 1]],
  hz: [['hz', 1], ['khz', 0.001]],
  '°': [['°', 1], ['deg', 1], ['gradi', 1]],
};

/**
 * Display factor of the unit typed after a number: null = none typed (the selected unit applies);
 * the factor of one of `units` (by id, label or alias) or of the control's fixed unit; undefined
 * when the text names a unit the control does not have (the value must not be taken as is).
 */
export function typedUnitFactor(token: string, units: readonly UnitOption[], fixedUnit?: string): number | null | undefined {
  const t = normalizeUnit(token);
  if (t === '') return null;
  for (const u of units) {
    if ([u.id, u.label, ...(u.aliases ?? [])].some((a) => normalizeUnit(a) === t)) return u.factor;
  }
  if (units.length === 0 && fixedUnit !== undefined) {
    const f = normalizeUnit(fixedUnit);
    const known = FIXED_UNIT_ALIASES[f];
    if (known) return known.find(([a]) => a === t)?.[1];
    if (f === t) return 1;
  }
  return undefined;
}

const MINUS = '−';

/** Italian fixed-decimal formatting with a typographic minus; "−0,00" is shown as "0,00". */
export function formatNumber(v: number, decimals: number): string {
  if (Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : `${MINUS}∞`;
  const d = Math.min(10, Math.max(0, Math.round(decimals)));
  const body = Math.abs(v).toFixed(d);
  const isZero = Number(body) === 0;
  return (v < 0 && !isZero ? MINUS : '') + body.replace('.', ',');
}

/**
 * Like formatNumber with `decimals`, plus up to `extra` more decimals shown only when they are
 * non-zero, so that Alt (×0.1) steps stay visible without cluttering round values.
 */
export function formatAdaptive(v: number, decimals: number, extra = 1): string {
  const s = formatNumber(v, decimals + Math.max(0, extra));
  if (extra <= 0 || !Number.isFinite(v)) return s;
  let out = s;
  for (let i = 0; i < extra && out.endsWith('0') && out.includes(','); i++) {
    if (out.length - out.indexOf(',') - 1 <= decimals) break;
    out = out.slice(0, -1);
  }
  if (out.endsWith(',')) out = out.slice(0, -1);
  return out === `${MINUS}0` ? '0' : out;
}

/** Like formatNumber but always with a sign ("+0,83", "−0,83", "0,00"). */
export function formatSigned(v: number, decimals: number): string {
  const s = formatNumber(v, decimals);
  if (s.startsWith(MINUS) || s === '—' || Number(s.replace(',', '.')) === 0) return s;
  return `+${s}`;
}

/** dB with "−∞" at or below `floorDb`. */
export function formatDb(db: number, decimals = 1, floorDb = -120): string {
  if (!Number.isFinite(db) || db <= floorDb) return Number.isNaN(db) ? '—' : `${MINUS}∞`;
  return formatNumber(db, decimals);
}

/** Frequency in Hz with sensible decimals ("99,99 Hz", "1250 Hz"). */
export function formatHz(f: number): string {
  if (!Number.isFinite(f)) return '—';
  const decimals = f < 100 ? 2 : f < 1000 ? 1 : 0;
  return `${formatNumber(f, decimals)} Hz`;
}

/** Display units for a delay whose base unit is milliseconds. */
export function delayUnits(fs: number, speedOfSound: number): UnitOption[] {
  const sr = Number.isFinite(fs) && fs > 0 ? fs : 48000;
  const c = Number.isFinite(speedOfSound) && speedOfSound > 0 ? speedOfSound : 343;
  return [
    { id: 'ms', label: 'ms', factor: 1, step: 0.001, decimals: 4, aliases: ['msec'] },
    { id: 'us', label: 'µs', factor: 1000, step: 1, decimals: 1 },
    { id: 'samples', label: 'campioni', factor: sr / 1000, step: 0.01, decimals: 3, aliases: ['campione', 'camp', 'c', 'samples', 'sample', 'smp'] },
    // 1 ms of propagation = c · 1e-3 m = c / 10 cm.
    { id: 'cm', label: 'cm', factor: c / 10, step: 0.1, decimals: 2 },
  ];
}

/** Total phase shown to the user: polarity (180° or 0°) + Δ, wrapped to [0, 360). */
export function totalPhaseDeg(polarityInvert: boolean, deltaDeg: number): number {
  const d = Number.isFinite(deltaDeg) ? deltaDeg : 0;
  const t = (polarityInvert ? 180 : 0) + d;
  const w = ((t % 360) + 360) % 360;
  return cleanFloat(w) === 360 ? 0 : cleanFloat(w);
}

/** Time equivalent of a phase offset Δ at frequency f, in milliseconds (positive = lag). */
export const phaseDeltaToMs = (deltaDeg: number, fHz: number): number =>
  fHz > 0 && Number.isFinite(deltaDeg) ? (deltaDeg / (360 * fHz)) * 1000 : 0;

/** Phase error (degrees) at harmonic h per 1 Hz of f0 error, across a total delay τ (s). */
export const driftDegPerHz = (harmonic: number, delaySeconds: number): number =>
  360 * Math.max(0, harmonic) * Math.max(0, delaySeconds);

/**
 * Approximate DC group delay (ms) of the Bessel low-pass whose -3 dB point is `cutoffHz`.
 * Delay-normalised Bessel prototypes have τ(0) = 1 s with -3 dB at 1.3617 (2nd) / 2.1139 (4th) rad/s.
 */
export function besselGroupDelayMs(cutoffHz: number, order: 2 | 4): number {
  if (!(cutoffHz > 0)) return 0;
  const w3 = order === 4 ? 2.113917675 : 1.361654129;
  return (w3 / (2 * Math.PI * cutoffHz)) * 1000;
}

/** Maps a level in dB to the filled fraction of a meter spanning [floorDb, ceilDb]. */
export function dbToFraction(db: number, floorDb = -60, ceilDb = 0): number {
  if (!Number.isFinite(db) || !(ceilDb > floorDb)) return 0;
  return Math.min(1, Math.max(0, (db - floorDb) / (ceilDb - floorDb)));
}
