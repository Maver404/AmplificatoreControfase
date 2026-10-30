/**
 * Level meters: a reusable horizontal meter (RMS bar + peak marker + dBFS text on a −60…0 scale)
 * and a compact block with the reference / error mic levels and the measured attenuation.
 * Colour zones are positional (the gradient is fixed, a mask hides the unlit part), so a given
 * colour always means the same level.
 */
import type { LevelReading } from '../audio/protocol.ts';
import { dbToFraction, formatDb, formatNumber, formatSigned } from './units.ts';

export const METER_FLOOR_DB = -60;
/** Attenuations smaller than this (in magnitude) are reported as no change: they would read ±0,0 dB. */
export const ATTENUATION_NO_CHANGE_DB = 0.05;
/** A bypass reference older than this may no longer match the ambient noise: the readout says so. */
export const ATTENUATION_STALE_S = 60;

export interface LevelMeterOptions {
  label: string;
  /** Hide the visible label (still used for the accessible name). */
  hideLabel?: boolean;
}

export interface LevelMeter {
  readonly element: HTMLElement;
  update(reading: LevelReading | null): void;
  destroy(): void;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function createLevelMeter(container: HTMLElement, opts: LevelMeterOptions): LevelMeter {
  const root = el('div', 'meter');
  const label = el('span', 'meter-label', opts.label);
  if (opts.hideLabel) label.classList.add('visually-hidden');
  const track = el('div', 'meter-track');
  track.setAttribute('role', 'meter');
  track.setAttribute('aria-label', `Livello ${opts.label}`);
  track.setAttribute('aria-valuemin', String(METER_FLOOR_DB));
  track.setAttribute('aria-valuemax', '0');
  const mask = el('div', 'meter-mask');
  const peak = el('div', 'meter-peak');
  const ticks = el('div', 'meter-ticks');
  ticks.setAttribute('aria-hidden', 'true');
  for (const db of [-48, -36, -24, -12, -6]) {
    const t = el('span', 'meter-tick');
    t.style.left = `${dbToFraction(db, METER_FLOOR_DB) * 100}%`;
    ticks.append(t);
  }
  track.append(mask, ticks, peak);
  const text = el('span', 'meter-value num');
  text.setAttribute('aria-hidden', 'true');
  root.append(label, track, text);
  container.append(root);

  let lastText = '';
  function update(r: LevelReading | null): void {
    const rms = r ? r.rmsDb : -Infinity;
    const pk = r ? r.peakDb : -Infinity;
    const fr = dbToFraction(rms, METER_FLOOR_DB);
    const fp = dbToFraction(pk, METER_FLOOR_DB);
    mask.style.transform = `scaleX(${(1 - fr).toFixed(4)})`;
    peak.style.left = `${(fp * 100).toFixed(2)}%`;
    peak.hidden = !r || fp <= 0;
    const s = r ? `${formatDb(rms, 1)} / ${formatDb(pk, 1)}` : '—';
    if (s !== lastText) {
      lastText = s;
      text.textContent = s;
      track.setAttribute('aria-valuenow', String(Math.max(METER_FLOOR_DB, Math.min(0, Number.isFinite(rms) ? rms : METER_FLOOR_DB))));
      track.setAttribute(
        'aria-valuetext',
        r ? `RMS ${formatDb(rms, 1)} dBFS, picco ${formatDb(pk, 1)} dBFS` : 'nessun segnale',
      );
    }
    root.classList.toggle('is-hot', Number.isFinite(pk) && pk > -3);
    root.classList.toggle('is-idle', !r);
  }
  update(null);

  return {
    element: root,
    update,
    destroy(): void {
      root.remove();
    },
  };
}

export interface MetersViewData {
  ref: LevelReading | null;
  err: LevelReading | null;
  /** Error mic usable (channel 2 present and valid). */
  errAvailable: boolean;
  /** ANC output active (false = bypass / muted / stopped). */
  ancOn: boolean;
  /** Measured broadband attenuation in dB (positive = quieter with ANC), null if not measured yet. */
  attenuationDb: number | null;
  /** Age of the bypass windows the attenuation is measured against (s), null without any. */
  attenuationRefAgeS?: number | null;
  /** 1-based input the error mic uses (the one that is not the reference's; default 2). */
  errorInput?: 1 | 2;
}

export interface MetersView {
  update(data: MetersViewData): void;
  destroy(): void;
}

export type AttenuationState = 'none' | 'worse' | 'flat' | 'fair' | 'good';

export interface AttenuationReport {
  /** Readout ("—" when not measurable). */
  text: string;
  state: AttenuationState;
  note: string;
  /** The bypass reference is older than ATTENUATION_STALE_S (the note says how to refresh it). */
  stale: boolean;
}

/** "45 s", "3 min". */
const ageText = (s: number): string => (s < 120 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

/**
 * Readout, colour state and note of the measured attenuation (pure, for the view and the tests). A
 * value measured against a bypass reference older than ATTENUATION_STALE_S is flagged as stale.
 */
export function attenuationReport(
  errAvailable: boolean,
  attenuationDb: number | null,
  refAgeS: number | null = null,
  errorInput: 1 | 2 = 2,
): AttenuationReport {
  const r = attenuationValue(errAvailable, attenuationDb, errorInput);
  if (r.state === 'none' || refAgeS === null || !(refAgeS > ATTENUATION_STALE_S)) return r;
  return { ...r, stale: true, note: `${r.note} Riferimento di bypass di ${ageText(refAgeS)} fa: premi B per aggiornarlo.` };
}

function attenuationValue(errAvailable: boolean, attenuationDb: number | null, errorInput: 1 | 2): AttenuationReport {
  if (!errAvailable) {
    return { text: '—', state: 'none', note: `Serve il mic d’errore (ingresso ${errorInput}) per misurare l’attenuazione.`, stale: false };
  }
  if (attenuationDb === null || !Number.isFinite(attenuationDb)) {
    return { text: '—', state: 'none', note: 'In attesa di una finestra di confronto ANC attivo / bypass.', stale: false };
  }
  if (Math.abs(attenuationDb) < ATTENUATION_NO_CHANGE_DB) {
    return {
      text: `${formatNumber(0, 1)} dB`,
      state: 'flat',
      note: 'Nessuna variazione del rumore nella zona tra bypass e ANC attivo (banda larga).',
      stale: false,
    };
  }
  const text = `${formatSigned(attenuationDb, 1)} dB`;
  if (attenuationDb < 0) {
    return { text, state: 'worse', note: 'Il rumore nella zona è aumentato: controlla polarità, ritardo e guadagno.', stale: false };
  }
  return {
    text,
    state: attenuationDb >= 10 ? 'good' : 'fair',
    note: 'Differenza errore/riferimento tra bypass e ANC attivo (banda larga).',
    stale: false,
  };
}

export function createMetersView(container: HTMLElement): MetersView {
  const root = el('div', 'meters-view');
  const levels = el('div', 'meters-levels');
  const refMeter = createLevelMeter(levels, { label: 'Riferimento' });
  const errMeter = createLevelMeter(levels, { label: 'Mic d’errore' });

  const att = el('div', 'attenuation');
  const attHead = el('div', 'attenuation-head');
  const attLabel = el('span', 'attenuation-label', 'Attenuazione misurata');
  const chip = el('span', 'chip');
  attHead.append(attLabel, chip);
  const attValue = el('output', 'attenuation-value num');
  attValue.setAttribute('aria-live', 'off');
  const attNote = el('p', 'attenuation-note');
  att.append(attHead, attValue, attNote);

  root.append(levels, att);
  container.append(root);

  function update(d: MetersViewData): void {
    refMeter.update(d.ref);
    errMeter.update(d.errAvailable ? d.err : null);
    const chipText = d.ancOn ? 'ANC attivo' : 'ANC spento (bypass)';
    if (chip.textContent !== chipText) chip.textContent = chipText;
    chip.dataset['state'] = d.ancOn ? 'on' : 'off';
    const r = attenuationReport(d.errAvailable, d.attenuationDb, d.attenuationRefAgeS ?? null, d.errorInput ?? 2);
    if (attValue.textContent !== r.text) attValue.textContent = r.text;
    att.dataset['state'] = r.state;
    att.classList.toggle('is-stale', r.stale);
    if (attNote.textContent !== r.note) attNote.textContent = r.note;
  }
  update({ ref: null, err: null, errAvailable: false, ancOn: false, attenuationDb: null });

  return {
    update,
    destroy(): void {
      refMeter.destroy();
      errMeter.destroy();
      root.remove();
    },
  };
}
