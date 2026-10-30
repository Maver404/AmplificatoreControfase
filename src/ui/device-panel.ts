/**
 * Audio devices: microphone permission, input / output selection, reference input channel,
 * error mic source, latency mode and the input-channel-2 status line.
 */
import { errorMicInput } from '../app/validation.ts';
import type { Ch2Status } from '../audio/protocol.ts';

export type LatencyMode = 'normal' | 'ultra';
export type MicPermission = 'unknown' | 'prompt' | 'granted' | 'denied';

export interface DeviceOption {
  deviceId: string;
  label: string;
  /** Short qualifier appended to the label (e.g. "Bluetooth: non utilizzabile"). */
  note?: string;
  /** The option cannot be chosen (shown, but disabled). */
  blocked?: boolean;
}

export interface DevicePanelData {
  permission: MicPermission;
  inputs: readonly DeviceOption[];
  outputs: readonly DeviceOption[];
  inputId: string | null;
  outputId: string | null;
  /** HTMLMediaElement/AudioContext setSinkId available. */
  sinkIdSupported: boolean;
  refChannel: 0 | 1;
  errorChannel: 0 | 1 | -1;
  latencyMode: LatencyMode;
  ch2: Ch2Status;
  /** Channels actually delivered by the input track, if known. */
  trackChannels?: number | null;
  /** A device change is in progress (context being recreated): controls disabled. */
  busy?: boolean;
}

export interface DevicePanelOptions {
  onAllowMic: () => void;
  onInputChange: (deviceId: string) => void;
  onOutputChange: (deviceId: string) => void;
  onRefChannelChange: (channel: 0 | 1) => void;
  onErrorChannelChange: (channel: 0 | 1 | -1) => void;
  onLatencyModeChange: (mode: LatencyMode) => void;
}

export interface DevicePanel {
  update(data: DevicePanelData): void;
  destroy(): void;
}

/** Status of the error-mic input `n` (input 2, or 1 when the reference is on input 2). */
function ch2StatusText(ch2: Ch2Status, n: 1 | 2): string {
  switch (ch2) {
    case 'pending':
      return `Verifica dell’ingresso ${n} (mic d’errore) in corso…`;
    case 'ok':
      return `Ingresso ${n} attivo: il mic d’errore è utilizzabile.`;
    case 'absent':
      return 'Secondo ingresso assente: il browser consegna una traccia mono. Il mic d’errore resta disattivato.';
    case 'silent':
      return `Ingresso ${n} muto: collega il mic d’errore, alza il gain o attiva l’alimentazione phantom se serve.`;
    case 'duplicate':
      return `L’ingresso ${n} è identico all’ingresso ${3 - n} (l’interfaccia duplica il canale): mic d’errore disattivato.`;
  }
}

const CH2_BEFORE_MIC = 'Secondo ingresso (mic d’errore): disponibile dopo aver consentito il microfono.';

/**
 * Permission line. `permission` is 'granted' only once the microphone is open (main.ts maps it), and
 * the "negato" advice appears only when the browser really reports 'denied' ('prompt'/'unknown' get
 * the neutral request). `noDevices`: the mic is open but both device lists are empty.
 */
export function permissionText(permission: MicPermission, noDevices: boolean): string {
  if (permission === 'granted') {
    return noDevices
      ? 'Nessun dispositivo audio trovato: collega l’interfaccia USB e controlla che il sistema la riconosca ' +
          '(l’elenco si aggiorna da solo). L’uscita resta a zero.'
      : 'Microfono consentito. L’uscita resta a zero finché la configurazione non è validata.';
  }
  if (permission === 'denied') {
    return 'Accesso al microfono negato: consentilo dalle impostazioni del sito (icona accanto all’indirizzo) e dalle impostazioni privacy del sistema, poi riprova.';
  }
  return 'Serve il permesso del microfono per elencare i dispositivi. L’uscita resta a zero.';
}

/**
 * First (disabled) option of a device select. Kept short: the select is narrow in the two-column
 * layout and a long placeholder gets truncated (the explanation is on the permission line).
 */
export function devicePlaceholder(kind: 'input' | 'output', micOpen: boolean, count: number): string {
  if (!micOpen) return '— consenti il microfono —';
  if (count > 0) return '— scegli —';
  return kind === 'input' ? '— nessun ingresso —' : '— nessuna uscita —';
}

/**
 * Error-mic input status line; before the microphone is open there is nothing to check yet. The
 * engine checks input 2, or input 1 when the error mic is there (`errorChannel` 0).
 */
export function ch2Text(ch2: Ch2Status, micOpen: boolean, trackChannels?: number | null, errorChannel = -1): string {
  if (!micOpen) return CH2_BEFORE_MIC;
  const text = ch2StatusText(ch2, errorMicInput(errorChannel));
  return trackChannels !== undefined && trackChannels !== null ? `${text} (canali ricevuti: ${trackChannels})` : text;
}

let seq = 0;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function field(labelText: string, control: HTMLElement, id: string): HTMLDivElement {
  const f = el('div', 'field');
  const l = el('label', 'field-label', labelText);
  l.htmlFor = id;
  control.id = id;
  f.append(l, control);
  return f;
}

function fillSelect(sel: HTMLSelectElement, list: readonly DeviceOption[], current: string | null, placeholderText: string): void {
  // The placeholder is part of the key: an empty list stays empty when the microphone opens.
  const key =
    placeholderText +
    '\u0002' +
    list.map((d) => `${d.deviceId}\u0000${d.label}\u0000${d.note ?? ''}\u0000${d.blocked ? 1 : 0}`).join('\u0001');
  if (sel.dataset['key'] !== key) {
    const opts: HTMLOptionElement[] = [];
    const placeholder = el('option', undefined, placeholderText);
    placeholder.value = '';
    placeholder.disabled = true;
    opts.push(placeholder);
    list.forEach((d, i) => {
      const name = d.label.trim() || `Dispositivo ${i + 1}`;
      const o = el('option', undefined, d.note ? `${name} (${d.note})` : name);
      o.value = d.deviceId;
      o.disabled = !!d.blocked;
      opts.push(o);
    });
    sel.replaceChildren(...opts);
    sel.dataset['key'] = key;
  }
  sel.value = current !== null && list.some((d) => d.deviceId === current) ? current : '';
}

export function createDevicePanel(container: HTMLElement, opts: DevicePanelOptions): DevicePanel {
  const id = `dp${++seq}`;
  let data: DevicePanelData | null = null;

  const root = el('div', 'device-panel');

  const permRow = el('div', 'dp-perm');
  const allowBtn = el('button', 'btn btn--primary', 'Consenti microfono');
  allowBtn.type = 'button';
  allowBtn.addEventListener('click', () => opts.onAllowMic());
  const permText = el('p', 'dp-perm-text');
  permText.setAttribute('role', 'status');
  permRow.append(allowBtn, permText);

  const grid = el('div', 'dp-grid');
  const inputSel = el('select');
  inputSel.addEventListener('change', () => {
    if (inputSel.value) opts.onInputChange(inputSel.value);
  });
  const outputSel = el('select');
  outputSel.addEventListener('change', () => {
    if (outputSel.value) opts.onOutputChange(outputSel.value);
  });
  const inputField = field('Ingresso (interfaccia audio)', inputSel, `${id}-in`);
  const outputField = field('Uscita (interfaccia audio)', outputSel, `${id}-out`);
  const sinkNote = el(
    'p',
    'inline-info',
    'Questo browser non permette di scegliere l’uscita: viene usata l’uscita predefinita del sistema. ' +
      'Impostala sulla stessa interfaccia USB nelle impostazioni audio di macOS o Windows, poi ricarica la pagina.',
  );
  sinkNote.hidden = true;

  const refSel = el('select');
  for (const [v, t] of [
    ['0', 'Ingresso 1'],
    ['1', 'Ingresso 2'],
  ] as const) {
    const o = el('option', undefined, t);
    o.value = v;
    refSel.append(o);
  }
  refSel.addEventListener('change', () => opts.onRefChannelChange(refSel.value === '1' ? 1 : 0));
  const refField = field('Canale del mic di riferimento', refSel, `${id}-ref`);

  const errSel = el('select');
  errSel.addEventListener('change', () => {
    const v = errSel.value;
    opts.onErrorChannelChange(v === '0' ? 0 : v === '1' ? 1 : -1);
  });
  const errField = field('Mic d’errore', errSel, `${id}-err`);

  grid.append(inputField, outputField, refField, errField);

  const ch2Line = el('p', 'dp-ch2');
  ch2Line.setAttribute('role', 'status');

  const advanced = el('details', 'advanced');
  advanced.append(el('summary', undefined, 'Avanzate'));
  const advBody = el('div', 'advanced-body');
  const latFs = el('fieldset', 'segmented');
  latFs.append(el('legend', 'segmented-legend', 'Latenza del motore audio'));
  const latInputs: Array<{ input: HTMLInputElement; mode: LatencyMode }> = [];
  for (const [mode, text] of [
    ['normal', 'Normale (blocco 128)'],
    ['ultra', 'Ultra-bassa (blocco 64, sperimentale)'],
  ] as const) {
    const lab = el('label', 'segmented-item');
    const input = el('input');
    input.type = 'radio';
    input.name = `${id}-lat`;
    input.value = mode;
    input.addEventListener('change', () => {
      if (input.checked) opts.onLatencyModeChange(mode);
    });
    lab.append(input, el('span', undefined, text));
    latFs.append(lab);
    latInputs.push({ input, mode });
  }
  const latHelp = el(
    'p',
    'fp-help',
    'Ultra-bassa riduce la latenza di qualche millisecondo ma può causare interruzioni (underrun) su computer meno potenti.',
  );
  advBody.append(latFs, latHelp);
  advanced.append(advBody);

  root.append(permRow, grid, sinkNote, ch2Line, advanced);
  container.append(root);

  function render(d: DevicePanelData): void {
    const granted = d.permission === 'granted';
    const noDevices = granted && d.inputs.length === 0 && d.outputs.length === 0;
    allowBtn.hidden = granted;
    allowBtn.disabled = !!d.busy;
    // Live texts: rewritten only when they change (update() runs at meter rate inside the wizard).
    const perm = permissionText(d.permission, noDevices);
    if (permText.textContent !== perm) permText.textContent = perm;
    permText.classList.toggle('is-error', d.permission === 'denied' || noDevices);

    fillSelect(inputSel, d.inputs, d.inputId, devicePlaceholder('input', granted, d.inputs.length));
    fillSelect(outputSel, d.outputs, d.outputId, devicePlaceholder('output', granted, d.outputs.length));
    inputSel.disabled = !granted || !!d.busy;
    outputSel.disabled = !granted || !!d.busy;
    outputField.hidden = !d.sinkIdSupported;
    sinkNote.hidden = d.sinkIdSupported;

    refSel.value = String(d.refChannel);
    refSel.disabled = !!d.busy;
    const other: 0 | 1 = d.refChannel === 0 ? 1 : 0;
    const errKey = `${other}`;
    if (errSel.dataset['other'] !== errKey) {
      const none = el('option', undefined, 'Nessuno');
      none.value = '-1';
      const second = el('option', undefined, `Ingresso ${other + 1}`);
      second.value = String(other);
      errSel.replaceChildren(none, second);
      errSel.dataset['other'] = errKey;
    }
    errSel.value = d.errorChannel === other ? String(other) : '-1';
    errSel.disabled = !!d.busy;

    const ch2 = ch2Text(d.ch2, granted, d.trackChannels, d.errorChannel);
    if (ch2Line.textContent !== ch2) ch2Line.textContent = ch2;
    ch2Line.dataset['state'] = granted ? d.ch2 : 'closed';

    for (const l of latInputs) {
      l.input.checked = l.mode === d.latencyMode;
      l.input.disabled = !!d.busy;
    }
    if (d.latencyMode === 'ultra') advanced.open = true;
    root.classList.toggle('is-busy', !!d.busy);
    root.setAttribute('aria-busy', String(!!d.busy));
  }

  render({
    permission: 'unknown',
    inputs: [],
    outputs: [],
    inputId: null,
    outputId: null,
    sinkIdSupported: true,
    refChannel: 0,
    errorChannel: -1,
    latencyMode: 'normal',
    ch2: 'pending',
  });

  return {
    update(next: DevicePanelData): void {
      data = next;
      render(data);
    },
    destroy(): void {
      root.remove();
    },
  };
}
