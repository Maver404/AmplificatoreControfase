/**
 * Conditioning filters and phase mode: Ritardo / Tonale (+ Rotazione Hilbert, fRef and the length
 * of the feedback-neutralisation FIRs under "Avanzate"), tonal f0 / harmonics / Q with the f0
 * estimate and drift sensitivity, HP protection, barrier compensation and the Bessel LP with its
 * group-delay readout.
 */
import { clampParams, DEFAULT_PARAMS, minTonalQ, RANGES } from '../dsp/params.ts';
import type { FilterParams, PhaseMode, TonalParams } from '../dsp/params.ts';
import { conditioningResponse, groupDelaySamples } from '../dsp/response.ts';
import { FEEDBACK_TAP_CHOICES } from './feedback-taps.ts';
import { createPrecisionControl } from './precision-control.ts';
import { besselGroupDelayMs, driftDegPerHz, formatHz, formatNumber } from './units.ts';

/** −20 dB of cancellation needs ≤ 5.73° of phase error. */
const PHASE_TOL_20DB_DEG = 5.73;

export interface DriftSensitivity {
  /** Degrees of phase error per Hz of f0 error, harmonic 1…N (those below 0.45·fs). */
  degPerHz: number[];
  /** f0 accuracy needed for −20 dB at the most sensitive harmonic, Hz. */
  tolHz: number;
}

/**
 * Phase error per Hz of f0 error at each harmonic h of the tonal chain: 360·h·(τ_cond(h·f0) + τ_align),
 * with τ_cond the group delay of the conditioning at h·f0 (the resonator bank's ≈ Q/(π·h·f0) included:
 * a tone off the bank's centre is shifted by ≈ atan(2Q·Δf/f0)) and τ_align the alignment mismatch in
 * whole periods (main.ts driftAlign).
 */
export function driftSensitivity(
  filters: FilterParams,
  tonal: TonalParams,
  fs: number,
  alignSeconds: number,
): DriftSensitivity {
  const p = clampParams({ ...DEFAULT_PARAMS, phaseMode: 'tonal', filters, tonal });
  const out: number[] = [];
  for (let h = 1; h <= Math.max(1, Math.round(p.tonal.harmonics)) && h * p.tonal.f0 < 0.45 * fs; h++) {
    const tau = groupDelaySamples((f) => conditioningResponse(p, fs, f), h * p.tonal.f0, fs) / fs;
    out.push(driftDegPerHz(h, (Number.isFinite(tau) ? Math.max(0, tau) : 0) + Math.max(0, alignSeconds)));
  }
  const worst = out.length > 0 ? Math.max(...out) : 0;
  return { degPerHz: out, tolHz: worst > 0 ? PHASE_TOL_20DB_DEG / worst : Infinity };
}

/** Help text of the resonator Q, with the minimum the number of harmonics allows (params.ts minTonalQ). */
export function qHint(harmonics: number): string {
  const base = 'Q alto = solo il tono, ma insegue peggio la deriva di f0.';
  const min = minTonalQ(harmonics);
  if (min <= RANGES.q.min) return base;
  return `${base} Con ${Math.round(harmonics)} armoniche il Q minimo è ${formatNumber(min, 0)}: sotto, le bande si sovrappongono troppo.`;
}

/** Writes text only when it changed (some of these are live regions, updated at meter rate). */
function setText(e: HTMLElement, text: string): void {
  if (e.textContent !== text) e.textContent = text;
}

export interface F0Readout {
  f0: number;
  /** 0..1 */
  confidence: number;
  /** Peak vs median spectrum, dB. */
  dominanceDb: number;
}

export interface FiltersPanelData {
  phaseMode: PhaseMode;
  fRef: number;
  tonal: TonalParams;
  filters: FilterParams;
  fs: number;
  f0Estimate?: F0Readout | null;
  /** The app detected a dominant tone and suggests the tonal mode. */
  tonalSuggested?: boolean;
  /**
   * Alignment mismatch in whole periods (s) for the drift sensitivity; `bound` = estimated from tap
   * delay and latencies (no calibration), an upper bound.
   */
  driftAlign?: { seconds: number; bound: boolean } | null;
  /** Length of the feedback-neutralisation FIRs (FEEDBACK_TAP_CHOICES). */
  feedbackTaps?: number;
  /** The neutralisation is running (a calibration with feedback paths is loaded). */
  feedbackActive?: boolean;
  /** Exact LP group delay in ms (from response.ts); an analog-prototype estimate is shown otherwise. */
  lpGroupDelayMs?: number | null;
  /** Group delay added by the Hilbert pair at fRef, ms. */
  hilbertGroupDelayMs?: number | null;
}

export interface FiltersPanelOptions {
  onPhaseModeChange: (mode: PhaseMode) => void;
  onFRefChange: (hz: number) => void;
  onTonalChange: (tonal: TonalParams) => void;
  onFiltersChange: (filters: FilterParams) => void;
  onSnapToPeak: () => void;
  onFeedbackTapsChange?: (taps: number) => void;
}

export interface FiltersPanel {
  update(data: FiltersPanelData): void;
  destroy(): void;
}

let seq = 0;

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

interface RadioGroup<T extends string | number> {
  readonly element: HTMLFieldSetElement;
  set(value: T): void;
  setDisabled(d: boolean): void;
}

function radioGroup<T extends string | number>(
  legend: string,
  name: string,
  options: ReadonlyArray<{ value: T; label: string }>,
  onPick: (v: T) => void,
): RadioGroup<T> {
  const fs = el('fieldset', 'segmented');
  fs.append(el('legend', 'segmented-legend', legend));
  const inputs: Array<{ input: HTMLInputElement; value: T }> = [];
  for (const o of options) {
    const lab = el('label', 'segmented-item');
    const input = el('input');
    input.type = 'radio';
    input.name = name;
    input.value = String(o.value);
    input.addEventListener('change', () => {
      if (input.checked) onPick(o.value);
    });
    lab.append(input, el('span', undefined, o.label));
    fs.append(lab);
    inputs.push({ input, value: o.value });
  }
  return {
    element: fs,
    set(v: T): void {
      for (const i of inputs) i.input.checked = i.value === v;
    },
    setDisabled(d: boolean): void {
      for (const i of inputs) i.input.disabled = d;
    },
  };
}

export function createFiltersPanel(container: HTMLElement, opts: FiltersPanelOptions): FiltersPanel {
  const id = `fp${++seq}`;
  let data: FiltersPanelData = {
    phaseMode: DEFAULT_PARAMS.phaseMode,
    fRef: DEFAULT_PARAMS.fRef,
    tonal: { ...DEFAULT_PARAMS.tonal },
    filters: { ...DEFAULT_PARAMS.filters },
    fs: 48000,
  };

  const root = el('div', 'filters-panel');

  // ---- Phase mode ----
  const modeBox = el('div', 'fp-section');
  modeBox.append(el('h3', 'fp-title', 'Modalità di fase'));
  const modeName = `${id}-mode`;
  const pickMode = (m: PhaseMode): void => {
    data = { ...data, phaseMode: m };
    render();
    opts.onPhaseModeChange(m);
  };
  const modeMain = radioGroup<PhaseMode>(
    'Modalità',
    modeName,
    [
      { value: 'delay', label: 'Ritardo' },
      { value: 'tonal', label: 'Tonale' },
    ],
    pickMode,
  );
  const modeHelp = el('p', 'fp-help');
  const suggestion = el('p', 'inline-info');
  suggestion.hidden = true;

  const advanced = el('details', 'advanced');
  advanced.append(el('summary', undefined, 'Avanzate'));
  const advBody = el('div', 'advanced-body');
  const modeAdv = radioGroup<PhaseMode>(
    'Modalità sperimentale',
    modeName,
    [{ value: 'hilbert', label: 'Rotazione Hilbert' }],
    pickMode,
  );
  const hilbertNote = el('p', 'fp-help');
  const hilbertWarn = el('p', 'inline-warn');
  hilbertWarn.hidden = true;
  const fRefBox = el('div');
  const tapsGroup = radioGroup<number>(
    'Tap del neutralizzatore del feedback',
    `${id}-taps`,
    FEEDBACK_TAP_CHOICES.map((n) => ({ value: n, label: String(n) })),
    (n) => {
      data = { ...data, feedbackTaps: n };
      opts.onFeedbackTapsChange?.(n);
    },
  );
  const tapsHelp = el('p', 'fp-help');
  advBody.append(modeAdv.element, hilbertNote, hilbertWarn, fRefBox, tapsGroup.element, tapsHelp);
  advanced.append(advBody);
  modeBox.append(modeMain.element, modeHelp, suggestion, advanced);

  const fRef = createPrecisionControl(fRefBox, {
    label: 'Frequenza di riferimento della fase (f_ref)',
    value: data.fRef,
    min: RANGES.fRef.min,
    max: RANGES.fRef.max,
    step: 0.01,
    unit: 'Hz',
    scale: 'log',
    describe: () => 'Δ di ogni canale è espresso a questa frequenza (modalità Ritardo e Hilbert).',
    onChange: (v) => {
      data = { ...data, fRef: v };
      opts.onFRefChange(v);
    },
  });

  // ---- Tonal ----
  const tonalBox = el('div', 'fp-section');
  tonalBox.append(el('h3', 'fp-title', 'Tono dominante'));
  const estimate = el('p', 'fp-readout num');
  // Changes every 2 s (confidence): readable on demand, not announced.
  estimate.setAttribute('aria-live', 'off');
  const snapBtn = el('button', 'btn', 'Aggancia al picco');
  snapBtn.type = 'button';
  snapBtn.addEventListener('click', () => opts.onSnapToPeak());
  const estimateRow = el('div', 'fp-row');
  estimateRow.append(estimate, snapBtn);
  const tonalControls = el('div', 'fp-tonal');
  const drift = el('p', 'fp-help');
  tonalBox.append(estimateRow, tonalControls, drift);

  const emitTonal = (patch: Partial<TonalParams>): void => {
    data = { ...data, tonal: { ...data.tonal, ...patch } };
    if (patch.harmonics !== undefined) {
      // More harmonics raise the Q minimum (clampParams raises Q the same way).
      const min = minTonalQ(data.tonal.harmonics);
      data = { ...data, tonal: { ...data.tonal, q: Math.max(data.tonal.q, min) } };
      q.update(data.tonal.q, { min });
    }
    renderTexts();
    opts.onTonalChange({ ...data.tonal });
  };
  const f0 = createPrecisionControl(tonalControls, {
    label: 'f0',
    value: data.tonal.f0,
    min: RANGES.f0.min,
    max: RANGES.f0.max,
    step: 0.01,
    unit: 'Hz',
    scale: 'log',
    onChange: (v) => emitTonal({ f0: v }),
  });
  const harmonics = createPrecisionControl(tonalControls, {
    label: 'Armoniche',
    value: data.tonal.harmonics,
    min: RANGES.harmonics.min,
    max: RANGES.harmonics.max,
    step: 1,
    integer: true,
    onChange: (v) => emitTonal({ harmonics: v }),
  });
  const q = createPrecisionControl(tonalControls, {
    label: 'Q dei risonatori',
    value: data.tonal.q,
    // More harmonics need a higher Q (clampParams raises it the same way).
    min: minTonalQ(data.tonal.harmonics),
    max: RANGES.q.max,
    step: 0.1,
    scale: 'log',
    describe: () => qHint(data.tonal.harmonics),
    onChange: (v) => emitTonal({ q: v }),
  });

  // ---- Filters ----
  const filtBox = el('div', 'fp-section');
  filtBox.append(el('h3', 'fp-title', 'Filtri di condizionamento'));
  const emitFilters = (patch: Partial<FilterParams>): void => {
    data = { ...data, filters: { ...data.filters, ...patch } };
    render();
    opts.onFiltersChange({ ...data.filters });
  };

  // HP
  const hpRow = el('div', 'fp-filter');
  const hpHead = el('div', 'fp-filter-head');
  const hpSwitch = switchButton('Passa-alto di protezione');
  hpSwitch.addEventListener('click', () => emitFilters({ hpEnabled: !data.filters.hpEnabled }));
  const hpOrder = radioGroup<1 | 2>(
    'Ordine',
    `${id}-hporder`,
    [
      { value: 1, label: '1°' },
      { value: 2, label: '2°' },
    ],
    (v) => emitFilters({ hpOrder: v }),
  );
  hpHead.append(hpSwitch, hpOrder.element);
  const hpCtl = el('div');
  const hpWarn = el('p', 'inline-warn');
  hpWarn.textContent =
    'Protezione cassa disattivata: vento, rombo e colpi sul microfono arrivano direttamente al woofer.';
  hpRow.append(hpHead, hpCtl, hpWarn);
  const hpFreq = createPrecisionControl(hpCtl, {
    label: 'Frequenza passa-alto',
    value: data.filters.hpFreq,
    min: RANGES.hpFreq.min,
    max: RANGES.hpFreq.max,
    step: 0.1,
    unit: 'Hz',
    scale: 'log',
    onChange: (v) => emitFilters({ hpFreq: v }),
  });

  // Barrier
  const barRow = el('div', 'fp-filter');
  const barHead = el('div', 'fp-filter-head');
  const barSwitch = switchButton('Compensazione barriera');
  barSwitch.addEventListener('click', () => emitFilters({ barrierEnabled: !data.filters.barrierEnabled }));
  barHead.append(barSwitch);
  const barCtl = el('div');
  const barHelp = el(
    'p',
    'fp-help',
    'Passa-basso del 1° ordine (−6 dB/ottava, circa −90°) che imita una parete o una finestra tra microfono e zona.',
  );
  barRow.append(barHead, barCtl, barHelp);
  const barFreq = createPrecisionControl(barCtl, {
    label: 'Frequenza barriera',
    value: data.filters.barrierFreq,
    min: RANGES.barrierFreq.min,
    max: RANGES.barrierFreq.max,
    step: 0.1,
    unit: 'Hz',
    scale: 'log',
    onChange: (v) => emitFilters({ barrierFreq: v }),
  });

  // LP
  const lpRow = el('div', 'fp-filter');
  const lpHead = el('div', 'fp-filter-head');
  const lpSwitch = switchButton('Passa-basso Bessel');
  lpSwitch.addEventListener('click', () => emitFilters({ lpEnabled: !data.filters.lpEnabled }));
  const lpOrder = radioGroup<2 | 4>(
    'Ordine',
    `${id}-lporder`,
    [
      { value: 2, label: '2°' },
      { value: 4, label: '4°' },
    ],
    (v) => emitFilters({ lpOrder: v }),
  );
  lpHead.append(lpSwitch, lpOrder.element);
  const lpCtl = el('div');
  const lpReadout = el('p', 'fp-readout num');
  const lpWarn = el('p', 'inline-warn');
  lpWarn.textContent =
    'Passa-basso disattivato: sopra qualche centinaio di Hz la cancellazione non è possibile e l’anti-rumore si somma al rumore.';
  lpRow.append(lpHead, lpCtl, lpReadout, lpWarn);
  const lpFreq = createPrecisionControl(lpCtl, {
    label: 'Taglio passa-basso (−3 dB)',
    value: data.filters.lpFreq,
    min: RANGES.lpFreq.min,
    max: RANGES.lpFreq.max,
    step: 1,
    unit: 'Hz',
    scale: 'log',
    onChange: (v) => emitFilters({ lpFreq: v }),
  });

  filtBox.append(hpRow, barRow, lpRow);
  root.append(modeBox, tonalBox, filtBox);
  container.append(root);

  function renderTexts(): void {
    const d = data;
    const est = d.f0Estimate ?? null;
    if (est && Number.isFinite(est.f0)) {
      setText(
        estimate,
        `Stima f0: ${formatNumber(est.f0, 3)} Hz · confidenza ${formatNumber(est.confidence * 100, 0)} % · ` +
          `${formatNumber(est.dominanceDb, 1)} dB sopra il fondo`,
      );
    } else {
      setText(estimate, 'Stima f0: nessun tono dominante rilevato');
    }
    snapBtn.disabled = !est || !Number.isFinite(est.f0);

    const al = d.driftAlign ?? null;
    if (d.phaseMode === 'tonal' && al !== null && Number.isFinite(al.seconds) && al.seconds >= 0) {
      const sens = driftSensitivity(d.filters, d.tonal, d.fs, al.seconds);
      const n = sens.degPerHz.length;
      // Without a calibration the alignment delay is an overestimate: the degrees are an upper bound and
      // the tolerance a lower bound (the true one is looser).
      const most = al.bound ? 'al più ' : '';
      const tol = `±${formatNumber(sens.tolHz, 3)} Hz`;
      setText(
        drift,
        `Sensibilità alla deriva (risonatori inclusi): 1 Hz di errore su f0 = ${most}${formatNumber(sens.degPerHz[0] ?? 0, 1)}° all’armonica 1` +
          (n > 1 ? ` · ${most}${formatNumber(sens.degPerHz[n - 1]!, 1)}° all’armonica ${n}` : '') +
          (al.bound
            ? `. Per −20 dB basta che f0 sia giusta entro ${tol} (stima prudente: senza calibrazione il ritardo d’allineamento è stimato per eccesso).`
            : `. Per −20 dB f0 deve essere giusta entro ${tol}.`),
      );
    } else if (d.phaseMode === 'tonal') {
      setText(drift, 'Sensibilità alla deriva: disponibile con il microfono aperto.');
    }

    const taps = d.feedbackTaps ?? FEEDBACK_TAP_CHOICES[1];
    tapsGroup.set(taps);
    setText(
      tapsHelp,
      `Lunghezza del filtro che toglie dal microfono di riferimento il suono della cassa (${taps} tap = ${formatNumber((taps / d.fs) * 1000, 1)} ms ` +
        'di risposta). Più tap neutralizzano riflessioni più lontane e abbassano il guadagno d’anello, con più carico di CPU. ' +
        'Con meno tap il guadagno d’anello sale: se supera il margine di stabilità, i guadagni dei canali vengono ridotti da soli. ' +
        (d.feedbackActive ? 'Attiva sulla calibrazione caricata.' : 'Si applica dopo la calibrazione.'),
    );

    const lpMs =
      d.lpGroupDelayMs !== undefined && d.lpGroupDelayMs !== null && Number.isFinite(d.lpGroupDelayMs)
        ? { v: d.lpGroupDelayMs, approx: false }
        : { v: besselGroupDelayMs(d.filters.lpFreq, d.filters.lpOrder), approx: true };
    lpReadout.textContent = d.filters.lpEnabled
      ? `Ritardo di gruppo del passa-basso: ${lpMs.approx ? '≈ ' : ''}${formatNumber(lpMs.v, 3)} ms (in banda passante)`
      : 'Ritardo di gruppo del passa-basso: 0 ms (disattivato)';

    const hg = d.hilbertGroupDelayMs;
    setText(
      hilbertNote,
      'Ruota la fase senza cambiare il ritardo, ma aggiunge ritardo di gruppo' +
        (hg !== undefined && hg !== null && Number.isFinite(hg) ? ` (${formatNumber(hg, 2)} ms a ${formatHz(d.fRef)})` : ' (circa 2,6 ms a 100 Hz)') +
        '. L’ottimo dipende dalla banda: usala con l’autotune.',
    );
    hilbertWarn.hidden = !(d.phaseMode === 'hilbert' && d.fs > 48000);
    hilbertWarn.textContent = `La rotazione Hilbert è tarata per 44,1/48 kHz: a ${formatNumber(d.fs / 1000, 1)} kHz la fase a bassa frequenza è meno precisa.`;
  }

  function render(): void {
    const d = data;
    modeMain.set(d.phaseMode);
    modeAdv.set(d.phaseMode);
    if (d.phaseMode === 'hilbert') advanced.open = true;
    modeHelp.textContent =
      d.phaseMode === 'delay'
        ? 'Ritardo: per rumore a banda larga o non noto. Δ diventa un ritardo extra a f_ref, senza dispersione.'
        : d.phaseMode === 'tonal'
          ? 'Tonale: riproduce solo il tono e le sue armoniche; il ritardo si allinea su periodi interi.'
          : 'Rotazione Hilbert: Δ ruota la fase di tutte le frequenze della stessa quantità.';
    suggestion.hidden = !(d.tonalSuggested && d.phaseMode !== 'tonal');
    suggestion.textContent = 'È stato rilevato un tono dominante: la modalità Tonale può cancellarlo meglio.';

    tonalControls.hidden = d.phaseMode !== 'tonal';
    drift.hidden = d.phaseMode !== 'tonal';

    const f = d.filters;
    hpSwitch.setAttribute('aria-checked', String(f.hpEnabled));
    hpOrder.set(f.hpOrder);
    hpOrder.setDisabled(!f.hpEnabled);
    hpFreq.update(f.hpFreq, { disabled: !f.hpEnabled });
    hpWarn.hidden = f.hpEnabled;
    hpRow.classList.toggle('is-off', !f.hpEnabled);

    barSwitch.setAttribute('aria-checked', String(f.barrierEnabled));
    barFreq.update(f.barrierFreq, { disabled: !f.barrierEnabled });
    barRow.classList.toggle('is-off', !f.barrierEnabled);

    lpSwitch.setAttribute('aria-checked', String(f.lpEnabled));
    lpOrder.set(f.lpOrder);
    lpOrder.setDisabled(!f.lpEnabled);
    lpFreq.update(f.lpFreq, { disabled: !f.lpEnabled });
    lpWarn.hidden = f.lpEnabled;
    lpRow.classList.toggle('is-off', !f.lpEnabled);

    fRef.update(d.fRef);
    f0.update(d.tonal.f0);
    harmonics.update(d.tonal.harmonics);
    q.update(d.tonal.q, { min: minTonalQ(d.tonal.harmonics) });
    renderTexts();
  }

  render();

  return {
    update(next: FiltersPanelData): void {
      data = {
        ...next,
        tonal: { ...next.tonal },
        filters: { ...next.filters },
      };
      render();
    },
    destroy(): void {
      for (const c of [fRef, f0, harmonics, q, hpFreq, barFreq, lpFreq]) c.destroy();
      root.remove();
    },
  };
}
