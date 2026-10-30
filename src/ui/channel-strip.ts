/**
 * One output channel (L or R): enable, polarity, gain, delay (ms / µs / campioni / cm), fine Δ
 * phase with the total-phase readout, output meter, limiter badge and the clamped-delay warning.
 * The link toggle only reports intent (onLinkChange); applying the offset is the controller's job.
 */
import type { MeterSnapshot } from '../audio/protocol.ts';
import { DEFAULT_PARAMS, RANGES, dbToLin } from '../dsp/params.ts';
import type { ChannelParams, PhaseMode } from '../dsp/params.ts';
import { createLevelMeter } from './meters-view.ts';
import { createPrecisionControl } from './precision-control.ts';
import {
  delayUnits,
  formatAdaptive,
  formatHz,
  formatNumber,
  formatSigned,
  phaseDeltaToMs,
  totalPhaseDeg,
} from './units.ts';

export type ChannelId = 'L' | 'R';

export interface ChannelStripInfo {
  fs: number;
  /** Speed of sound in m/s (for the cm unit). */
  speedOfSound: number;
  /** Frequency at which Δ is expressed (fRef, or f0 in tonal mode). */
  phaseRefHz: number;
  /** Calibrated values: double-click / ↺ on a control returns to them. */
  calibrated?: ChannelParams | null;
  phaseMode?: PhaseMode;
  /** +12 dB gain unlock (EngineParams.gainUnlocked). */
  gainUnlocked?: boolean;
  /** Current L/R link state (owned by the controller). */
  linked?: boolean;
}

export interface ChannelStripOptions {
  channel: ChannelId;
  initial?: ChannelParams;
  /** Full new channel params plus the field the user changed (lets the controller apply links). */
  onChange: (next: ChannelParams, changed: keyof ChannelParams) => void;
  onLinkChange: (linked: boolean) => void;
}

export interface ChannelStrip {
  readonly element: HTMLElement;
  update(params: ChannelParams, meters: MeterSnapshot | null, info: ChannelStripInfo): void;
  destroy(): void;
}

const LIMITER_BADGE_DB = 0.1;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function switchButton(text: string): HTMLButtonElement {
  const b = el('button', 'switch');
  b.type = 'button';
  b.setAttribute('role', 'switch');
  const track = el('span', 'switch-track');
  track.setAttribute('aria-hidden', 'true');
  b.append(track, el('span', 'switch-text', text));
  return b;
}

export function createChannelStrip(container: HTMLElement, opts: ChannelStripOptions): ChannelStrip {
  const idx: 0 | 1 = opts.channel === 'L' ? 0 : 1;
  let params: ChannelParams = { ...(opts.initial ?? DEFAULT_PARAMS.channels[idx]) };
  let info: ChannelStripInfo = { fs: 48000, speedOfSound: 343, phaseRefHz: DEFAULT_PARAMS.fRef };
  let linked = false;
  const titleId = `strip-${opts.channel}-title`;

  const root = el('article', 'strip');
  root.setAttribute('aria-labelledby', titleId);
  root.dataset['channel'] = opts.channel;

  const head = el('header', 'strip-head');
  const title = el('h3', 'strip-title', `Canale ${opts.channel}`);
  title.id = titleId;
  const limiterBadge = el('span', 'badge badge--limiter');
  limiterBadge.hidden = true;
  const enabledBtn = switchButton('Attivo');
  head.append(title, limiterBadge, enabledBtn);

  const toggles = el('div', 'strip-toggles');
  const polarityBtn = switchButton('Inverti polarità (180°)');
  const linkBtn = el('button', 'toggle-btn link-btn');
  linkBtn.type = 'button';
  linkBtn.textContent = 'Collega L/R';
  linkBtn.title = 'Le modifiche a un canale si applicano anche all’altro mantenendo la differenza attuale';
  toggles.append(polarityBtn, linkBtn);

  const controls = el('div', 'strip-controls');
  const emit = (changed: keyof ChannelParams, patch: Partial<ChannelParams>): void => {
    params = { ...params, ...patch };
    renderToggles();
    renderReadouts();
    opts.onChange({ ...params }, changed);
  };

  const gain = createPrecisionControl(controls, {
    label: 'Guadagno',
    value: params.gainDb,
    min: RANGES.gainDb.min,
    max: RANGES.gainDb.max,
    step: 0.01,
    unit: 'dB',
    resetLabel: 'valore calibrato',
    describe: (v) => `×${formatNumber(dbToLin(v), 4)} lineare`,
    onChange: (v) => emit('gainDb', { gainDb: v }),
  });

  const delay = createPrecisionControl(controls, {
    label: 'Ritardo',
    value: params.delayMs,
    min: RANGES.delayMs.min,
    max: RANGES.delayMs.max,
    step: 0.001,
    units: delayUnits(info.fs, info.speedOfSound),
    unitId: 'ms',
    scale: 'cubic',
    resetLabel: 'valore calibrato',
    describe: (v, unit) =>
      delayUnits(info.fs, info.speedOfSound)
        .filter((u) => u.id !== unit?.id)
        .map((u) => `${formatNumber(v * u.factor, u.decimals ?? 2)} ${u.label}`)
        .join(' · '),
    onChange: (v) => emit('delayMs', { delayMs: v }),
  });

  const phase = createPrecisionControl(controls, {
    label: 'Δ fase fine',
    value: params.phaseDeltaDeg,
    min: RANGES.phaseDeltaDeg.min,
    max: RANGES.phaseDeltaDeg.max,
    step: 0.1,
    unit: '°',
    resetValue: 0,
    resetLabel: 'valore calibrato',
    describe: (v) => {
      if (info.phaseMode === 'hilbert') return 'Applicato come rotazione di fase (Hilbert), non come ritardo';
      const ms = phaseDeltaToMs(v, info.phaseRefHz);
      return `Equivale a ${formatSigned(ms, 3)} ms a ${formatHz(info.phaseRefHz)} (positivo = ritardo)`;
    },
    onChange: (v) => emit('phaseDeltaDeg', { phaseDeltaDeg: v }),
  });

  const phaseReadout = el('p', 'phase-readout');
  const clampWarn = el('p', 'inline-warn');
  clampWarn.setAttribute('role', 'status');
  clampWarn.hidden = true;

  const meterBox = el('div', 'strip-meter');
  const meter = createLevelMeter(meterBox, { label: `Uscita ${opts.channel}` });
  const actual = el('p', 'strip-actual num');

  root.append(head, toggles, controls, phaseReadout, clampWarn, meterBox, actual);
  container.append(root);

  function renderToggles(): void {
    enabledBtn.setAttribute('aria-checked', String(params.enabled));
    polarityBtn.setAttribute('aria-checked', String(params.polarityInvert));
    linkBtn.setAttribute('aria-pressed', String(linked));
    root.classList.toggle('is-off', !params.enabled);
  }

  let lastPhaseKey = '';
  function renderReadouts(): void {
    const total = totalPhaseDeg(params.polarityInvert, params.phaseDeltaDeg);
    // Called at meter rate: rebuild only when the text changes.
    const key = `${total}|${info.phaseRefHz}`;
    if (key === lastPhaseKey) return;
    lastPhaseKey = key;
    phaseReadout.replaceChildren(
      'Fase totale: ',
      el('strong', 'num', `${formatAdaptive(total, 1)}°`),
      ` @ ${formatHz(info.phaseRefHz)}`,
    );
  }

  enabledBtn.addEventListener('click', () => emit('enabled', { enabled: !params.enabled }));
  polarityBtn.addEventListener('click', () => emit('polarityInvert', { polarityInvert: !params.polarityInvert }));
  linkBtn.addEventListener('click', () => {
    linked = !linked;
    renderToggles();
    opts.onLinkChange(linked);
  });

  renderToggles();
  renderReadouts();
  meter.update(null);

  return {
    element: root,
    update(next: ChannelParams, meters: MeterSnapshot | null, nextInfo: ChannelStripInfo): void {
      params = { ...next };
      info = { ...nextInfo };
      if (nextInfo.linked !== undefined) linked = nextInfo.linked;
      const cal = nextInfo.calibrated ?? null;
      gain.update(params.gainDb, {
        max: nextInfo.gainUnlocked ? RANGES.gainDb.maxUnlocked : RANGES.gainDb.max,
        resetValue: cal ? cal.gainDb : null,
      });
      delay.update(params.delayMs, {
        units: delayUnits(nextInfo.fs, nextInfo.speedOfSound),
        resetValue: cal ? cal.delayMs : null,
      });
      phase.update(params.phaseDeltaDeg, { resetValue: cal ? cal.phaseDeltaDeg : 0 });
      renderToggles();
      renderReadouts();

      const out = meters ? (idx === 0 ? meters.outL : meters.outR) : null;
      meter.update(params.enabled ? out : null);
      const gr = meters ? meters.limiterGrDb[idx] : 0;
      const limiting = Number.isFinite(gr) && gr > LIMITER_BADGE_DB;
      limiterBadge.hidden = !limiting;
      const badge = limiting ? `Limiter −${formatNumber(gr, 1)} dB` : '';
      if (limiterBadge.textContent !== badge) limiterBadge.textContent = badge;

      const clamped = !!meters && meters.delayClamped[idx];
      clampWarn.hidden = !clamped;
      const clampText = clamped ? 'Ritardo limitato: ritardo base + Δ scende sotto il minimo del DSP. Aumenta il ritardo o riduci Δ.' : '';
      // A live region: written only when it changes, not at every meter update.
      if (clampWarn.textContent !== clampText) clampWarn.textContent = clampText;
      let actualText = '';
      if (meters && Number.isFinite(meters.delaySamples[idx]) && nextInfo.fs > 0) {
        const d = meters.delaySamples[idx];
        actualText = `Ritardo effettivo del tap: ${formatNumber(d, 3)} campioni = ${formatNumber((d / nextInfo.fs) * 1000, 4)} ms`;
      }
      if (actual.textContent !== actualText) actual.textContent = actualText;
    },
    destroy(): void {
      gain.destroy();
      delay.destroy();
      phase.destroy();
      meter.destroy();
      root.remove();
    },
  };
}
