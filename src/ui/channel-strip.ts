/**
 * One output channel (L or R): enable, polarity, gain, delay (ms / µs / campioni / cm), fine Δ
 * phase with the total-phase readout, output meter, limiter badge and the clamped-delay warning.
 * In tonal mode a collapsible «Armoniche» section trims each harmonic's gain and phase
 * (ChannelParams.harmonicTrims) and shows the total phase there (polarity + h·Δ + θ_h).
 * The link toggle only reports intent (onLinkChange); applying the offset is the controller's job.
 */
import type { MeterSnapshot } from '../audio/protocol.ts';
import { anyTrim } from '../dsp/harmonic-trims.ts';
import { DEFAULT_PARAMS, MAX_HARMONICS, RANGES, dbToLin } from '../dsp/params.ts';
import type { ChannelParams, HarmonicTrim, PhaseMode } from '../dsp/params.ts';
import { createLevelMeter } from './meters-view.ts';
import { createPrecisionControl } from './precision-control.ts';
import type { PrecisionControl } from './precision-control.ts';
import {
  BANK_MAX_FRACTION,
  bankHarmonics,
  delayUnits,
  formatAdaptive,
  formatHz,
  formatNumber,
  formatSigned,
  harmonicTotalPhaseDeg,
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
  /** Tonal mode: harmonics of the resonator bank (rows of «Armoniche»; f0 is phaseRefHz). */
  harmonics?: number;
  /** The other channel has trims: while linked, this channel's «Azzera armoniche» clears them too. */
  otherTrimmed?: boolean;
}

export interface ChannelStripOptions {
  channel: ChannelId;
  initial?: ChannelParams;
  /** Full new channel params plus the field the user changed (lets the controller apply links). */
  onChange: (next: ChannelParams, changed: keyof ChannelParams) => void;
  onLinkChange: (linked: boolean) => void;
  /**
   * «Azzera armoniche»: every trim of this channel to 0, an absolute reset that the controller must not apply
   * as a linked delta (controller.clearHarmonicTrims). Without it the strip sends all-zero trims to onChange.
   */
  onClearTrims?: () => void;
}

export interface ChannelStrip {
  readonly element: HTMLElement;
  update(params: ChannelParams, meters: MeterSnapshot | null, info: ChannelStripInfo): void;
  destroy(): void;
}

const LIMITER_BADGE_DB = 0.1;

/** MAX_HARMONICS trims as new objects (missing entries neutral): never share the controller's array. */
export function copyTrims(trims: readonly HarmonicTrim[] | undefined): HarmonicTrim[] {
  return Array.from({ length: MAX_HARMONICS }, (_, i) => {
    const t = trims?.[i];
    return { gainDb: t?.gainDb ?? 0, phaseDeg: t?.phaseDeg ?? 0 };
  });
}

interface HarmonicRow {
  root: HTMLElement;
  title: HTMLElement;
  total: HTMLElement;
  gain: PrecisionControl;
  phase: PrecisionControl;
  key: string;
}

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
  const init = opts.initial ?? DEFAULT_PARAMS.channels[idx];
  let params: ChannelParams = { ...init, harmonicTrims: copyTrims(init.harmonicTrims) };
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
    renderHarmonics();
    opts.onChange({ ...params, harmonicTrims: copyTrims(params.harmonicTrims) }, changed);
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

  // ---- «Armoniche» (tonal mode): one gain and one phase trim per harmonic, on top of the delay.
  const harm = el('details', 'advanced strip-harm');
  harm.hidden = true;
  const harmSummary = el('summary');
  const harmCount = el('span', 'strip-harm-count');
  harmSummary.append('Armoniche', harmCount);
  const harmBody = el('div', 'advanced-body strip-harm-body');
  const harmHelp = el(
    'p',
    'fp-help',
    'Correzione di guadagno e di fase di ogni armonica, in aggiunta al ritardo (fase positiva = ritardo). La calcolano la ' +
      'configurazione guidata e l’autotune; gli aumenti oltre 6 dB salgono a 6 dB al secondo.',
  );
  const harmList = el('div', 'strip-harm-list');
  const harmSkipped = el('p', 'fp-help');
  harmSkipped.hidden = true;
  const harmReset = el('button', 'btn btn--small strip-harm-reset', 'Azzera armoniche');
  harmReset.type = 'button';
  harmReset.setAttribute('aria-label', `Azzera le correzioni delle armoniche del canale ${opts.channel}`);
  const harmActions = el('div', 'strip-harm-actions');
  harmActions.append(harmReset);
  harmBody.append(harmHelp, harmList, harmSkipped, harmActions);
  harm.append(harmSummary, harmBody);

  const setTrim = (h: number, patch: Partial<HarmonicTrim>): void => {
    const trims = copyTrims(params.harmonicTrims);
    trims[h - 1] = { ...trims[h - 1]!, ...patch };
    emit('harmonicTrims', { harmonicTrims: trims });
  };

  const rows: HarmonicRow[] = [];
  for (let h = 1; h <= MAX_HARMONICS; h++) {
    const rowRoot = el('div', 'strip-harm-row');
    rowRoot.setAttribute('role', 'group');
    rowRoot.hidden = true;
    const titleEl = el('span', 'strip-harm-title');
    titleEl.id = `strip-${opts.channel}-h${h}-title`;
    rowRoot.setAttribute('aria-labelledby', titleEl.id);
    const total = el('span', 'strip-harm-total');
    const rowHead = el('p', 'strip-harm-head');
    rowHead.append(titleEl, total);
    const rowControls = el('div', 'strip-harm-controls');
    const gainCtl = createPrecisionControl(rowControls, {
      label: `Guadagno armonica ${h}`,
      value: 0,
      min: RANGES.trimGainDb.min,
      max: RANGES.trimGainDb.max,
      step: 0.01,
      unit: 'dB',
      resetValue: 0,
      resetLabel: 'nessuna correzione',
      compact: true,
      onChange: (v) => setTrim(h, { gainDb: v }),
    });
    const phaseCtl = createPrecisionControl(rowControls, {
      label: `Fase armonica ${h}`,
      value: 0,
      min: RANGES.trimPhaseDeg.min,
      max: RANGES.trimPhaseDeg.max,
      step: 0.1,
      unit: '°',
      resetValue: 0,
      resetLabel: 'nessuna correzione',
      compact: true,
      onChange: (v) => setTrim(h, { phaseDeg: v }),
    });
    rowRoot.append(rowHead, rowControls);
    harmList.append(rowRoot);
    rows.push({ root: rowRoot, title: titleEl, total, gain: gainCtl, phase: phaseCtl, key: '' });
  }

  harmReset.addEventListener('click', () => {
    if (opts.onClearTrims === undefined) {
      emit('harmonicTrims', { harmonicTrims: copyTrims(undefined) });
      return;
    }
    params = { ...params, harmonicTrims: copyTrims(undefined) };
    renderHarmonics();
    opts.onClearTrims();
  });

  const meterBox = el('div', 'strip-meter');
  const meter = createLevelMeter(meterBox, { label: `Uscita ${opts.channel}` });
  const actual = el('p', 'strip-actual num');

  root.append(head, toggles, controls, phaseReadout, clampWarn, harm, meterBox, actual);
  container.append(root);

  function renderToggles(): void {
    enabledBtn.setAttribute('aria-checked', String(params.enabled));
    polarityBtn.setAttribute('aria-checked', String(params.polarityInvert));
    linkBtn.setAttribute('aria-pressed', String(linked));
    root.classList.toggle('is-off', !params.enabled);
  }

  let lastPhaseKey = '';
  function renderReadouts(): void {
    // Tonal mode: the total phase at f0 includes the fundamental's trim θ_1 (as the «Fondamentale» row).
    const total =
      info.phaseMode === 'tonal'
        ? harmonicTotalPhaseDeg(params.polarityInvert, params.phaseDeltaDeg, 1, params.harmonicTrims[0]?.phaseDeg ?? 0)
        : totalPhaseDeg(params.polarityInvert, params.phaseDeltaDeg);
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

  let lastHarmKey = '';
  /** «Armoniche»: rows, readouts and reset values; called at meter rate, so only what changed is touched. */
  function renderHarmonics(): void {
    const tonal = info.phaseMode === 'tonal';
    harm.hidden = !tonal;
    if (!tonal) return;
    const f0 = info.phaseRefHz;
    const used = bankHarmonics(info.harmonics ?? 1, f0, info.fs);
    const trims = params.harmonicTrims;
    const cal = info.calibrated?.harmonicTrims ?? null;
    for (let h = 1; h <= MAX_HARMONICS; h++) {
      const row = rows[h - 1]!;
      const on = used.includes(h);
      row.root.hidden = !on;
      if (!on) continue;
      const t = trims[h - 1] ?? { gainDb: 0, phaseDeg: 0 };
      const c = cal?.[h - 1] ?? null;
      const total = harmonicTotalPhaseDeg(params.polarityInvert, params.phaseDeltaDeg, h, t.phaseDeg);
      const key = `${t.gainDb}|${t.phaseDeg}|${c ? `${c.gainDb}|${c.phaseDeg}` : '-'}|${f0}|${total}`;
      // The controls also follow a value the controller kept although the user typed another one.
      if (key === row.key && row.gain.value === t.gainDb && row.phase.value === t.phaseDeg) continue;
      row.key = key;
      const resetLabel = c ? 'valore calibrato' : 'nessuna correzione';
      row.gain.update(t.gainDb, { resetValue: c ? c.gainDb : 0, resetLabel });
      row.phase.update(t.phaseDeg, { resetValue: c ? c.phaseDeg : 0, resetLabel });
      row.title.textContent = `${h === 1 ? 'Fondamentale' : `Armonica ${h}`} · ${formatHz(h * f0)}`;
      row.total.replaceChildren('Fase totale ', el('strong', 'num', `${formatAdaptive(total, 1)}°`));
    }
    const n = f0 > 0 ? Math.min(MAX_HARMONICS, Math.max(1, Math.round(info.harmonics ?? 1))) : 0;
    const trimmed = used.filter((h) => anyTrim([trims[h - 1] ?? { gainDb: 0, phaseDeg: 0 }], 1)).length;
    // Trims of harmonics not in use count for the reset too (they come back with more harmonics).
    const any = anyTrim(trims);
    // Linked, the reset clears both channels (its label says so): it can be pressed while either has trims.
    const resettable = any || (linked && info.otherTrimmed === true);
    const key = `${n}|${used.length}|${trimmed}|${resettable}|${info.fs}|${linked}`;
    if (key === lastHarmKey) return;
    lastHarmKey = key;
    harmCount.textContent = trimmed > 0 ? ` · ${trimmed} ${trimmed === 1 ? 'corretta' : 'corrette'}` : '';
    harmReset.disabled = !resettable;
    const resetLabel = linked
      ? 'Azzera le correzioni delle armoniche dei canali L e R (collegati)'
      : `Azzera le correzioni delle armoniche del canale ${opts.channel}`;
    if (harmReset.getAttribute('aria-label') !== resetLabel) harmReset.setAttribute('aria-label', resetLabel);
    const skipped = n - used.length;
    harmSkipped.hidden = skipped === 0;
    harmSkipped.textContent =
      skipped > 0
        ? `${skipped === 1 ? 'L’ultima armonica supera' : `Le ultime ${skipped} armoniche superano`} il 45 % della frequenza di ` +
          `campionamento (${formatHz(BANK_MAX_FRACTION * info.fs)}): il banco non le genera.`
        : '';
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
  renderHarmonics();
  meter.update(null);

  return {
    element: root,
    update(next: ChannelParams, meters: MeterSnapshot | null, nextInfo: ChannelStripInfo): void {
      params = { ...next, harmonicTrims: copyTrims(next.harmonicTrims) };
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
      renderHarmonics();

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
      for (const r of rows) {
        r.gain.destroy();
        r.phase.destroy();
      }
      meter.destroy();
      root.remove();
    },
  };
}
