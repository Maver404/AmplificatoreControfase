/**
 * Device/settings validation for step 2 of the start-up sequence. Pure: the caller gathers the
 * facts (devices, track settings, context state) and decides what to do with the result; the
 * worklet stays disarmed (exact zeros) until `ok` is true.
 */
import { classifyLabel, stripPseudoPrefix, type DeviceOption, type Platform } from '../audio/devices.ts';
import type { Ch2Status } from '../audio/protocol.ts';

export interface TrackSettingsInfo {
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  sampleRate?: number;
  channelCount?: number;
}

export interface SetupInfo {
  input: DeviceOption | null;
  /** null = system default output (no explicit selection possible or made). */
  output: DeviceOption | null;
  trackSettings: TrackSettingsInfo;
  ctxSampleRate: number | null;
  outputLatencyS: number | null;
  sinkSupported: boolean;
  chromium: boolean;
  platform: Platform;
  wantErrorMic: boolean;
  /** Input the error mic is on (0/1; −1 or omitted = input 2): the one the engine checks. */
  errorChannel?: number;
  ch2Status: Ch2Status;
}

export type IssueLevel = 'block' | 'error' | 'warn' | 'info';

export interface Issue {
  id: string;
  level: IssueLevel;
  /** Italian, shown to the user. */
  text: string;
  /** Only 'error' issues can be overridable ("continua comunque"). */
  overridable: boolean;
  /** Set on overridable issues the user chose to ignore (still listed so the UI can show them). */
  overridden?: boolean;
}

export interface ValidationResult {
  /** No block and no error that is not overridden. */
  ok: boolean;
  /** Sorted by severity: block, error, warn, info. */
  issues: Issue[];
  /** Output is headphones: the caller forces the ceiling to HEADPHONES_CEILING_DB and asks confirmation. */
  headphones: boolean;
  /** Input and output share a (non-empty) groupId, i.e. one physical interface with one clock. */
  sameDevice: boolean;
}

/** Above this the broadband acoustic head start needed becomes impractical (~13.7 m at 40 ms). */
export const MAX_OUTPUT_LATENCY_S = 0.04;

export const ISSUE = {
  noInput: 'no-input',
  bluetoothOutput: 'bluetooth-output',
  airplayOutput: 'airplay-output',
  bluetoothInput: 'bluetooth-input',
  builtinPair: 'builtin-pair',
  trackProcessing: 'track-processing',
  outputLatency: 'output-latency',
  outputLatencyUnknown: 'output-latency-unknown',
  sampleRateMismatch: 'sample-rate-mismatch',
  errorMicUnavailable: 'error-mic-unavailable',
  errorMicPending: 'error-mic-pending',
  headphones: 'headphones',
  clockDrift: 'clock-drift',
  sameDeviceUnknown: 'same-device-unknown',
  outputUnverified: 'output-unverified',
  nonChromium: 'non-chromium',
  sinkUnsupported: 'sink-unsupported',
  windowsSetup: 'windows-setup',
} as const;

const LEVEL_RANK: Record<IssueLevel, number> = { block: 0, error: 1, warn: 2, info: 3 };

/** Speed of sound at ~20 °C, only used to explain latency in metres. */
const C_ROOM = 343;

const decimal = (x: number, digits: number): string => x.toFixed(digits).replace('.', ',');
const formatKHz = (sr: number): string => `${String(Number((sr / 1000).toFixed(3))).replace('.', ',')} kHz`;

function deviceName(d: DeviceOption): string {
  const label = d.pseudo ? stripPseudoPrefix(d.label) : d.label;
  return `«${label.trim() || 'senza nome'}»`;
}

/**
 * 1-based number of the input the engine checks for the error mic: input 2, or input 1 when the
 * reference is on input 2 (errorChannel 0).
 */
export const errorMicInput = (errorChannel: number | undefined): 1 | 2 => (errorChannel === 0 ? 1 : 2);

function errorMicReason(status: Ch2Status, n: 1 | 2): string | null {
  switch (status) {
    case 'absent':
      return "il microfono arriva su un solo canale (traccia mono, l'ingresso 2 non è disponibile)";
    case 'silent':
      return `l'ingresso ${n} è muto`;
    case 'duplicate':
      return `l'ingresso ${n} riceve lo stesso segnale dell'ingresso ${3 - n} (canale duplicato)`;
    default:
      return null;
  }
}

export function validateSetup(info: SetupInfo, overrides: ReadonlySet<string> = new Set()): ValidationResult {
  const issues: Issue[] = [];
  const add = (id: string, level: IssueLevel, text: string, overridable = false): void => {
    issues.push({ id, level, text, overridable });
  };

  const { input, output } = info;
  const inClass = input ? classifyLabel(input.label) : null;
  const outClass = output ? classifyLabel(output.label) : null;

  // --- Blocking: never overridable -------------------------------------------------------------
  if (!input) {
    add(
      ISSUE.noInput,
      'block',
      "Nessun microfono di riferimento selezionato: scegli l'ingresso dell'interfaccia audio a cui è collegato il microfono esterno.",
    );
  }
  if (output && outClass?.bluetooth) {
    add(
      ISSUE.bluetoothOutput,
      'block',
      `L'uscita ${deviceName(output)} sembra un dispositivo Bluetooth o senza fili: aggiunge 150–275 ms di latenza e rende impossibile la cancellazione. Collega l'amplificatore a un'uscita cablata dell'interfaccia audio.`,
    );
  }
  if (output && outClass?.airplay) {
    add(
      ISSUE.airplayOutput,
      'block',
      `L'uscita ${deviceName(output)} è un dispositivo AirPlay: la latenza (centinaia di millisecondi o più) rende impossibile la cancellazione. Collega l'amplificatore a un'uscita cablata dell'interfaccia audio.`,
    );
  }
  if (input && inClass?.bluetooth) {
    add(
      ISSUE.bluetoothInput,
      'block',
      `L'ingresso ${deviceName(input)} sembra un microfono Bluetooth o senza fili: latenza alta e banda ridotta rendono impossibile la cancellazione. Usa un microfono collegato all'interfaccia audio.`,
    );
  }
  if (inClass?.builtin && outClass?.builtin) {
    add(
      ISSUE.builtinPair,
      'block',
      "Microfono e altoparlanti integrati nel computer: il suono degli altoparlanti rientra subito nel microfono (rischio di innesco) e non c'è vantaggio acustico sul rumore. Usa un'interfaccia audio esterna con microfono e amplificatore separati.",
    );
  }
  const ts = info.trackSettings;
  const processing: string[] = [];
  if (ts.echoCancellation === true) processing.push("cancellazione dell'eco");
  if (ts.noiseSuppression === true) processing.push('soppressione del rumore');
  if (ts.autoGainControl === true) processing.push('controllo automatico del guadagno');
  if (processing.length > 0) {
    add(
      ISSUE.trackProcessing,
      'block',
      `Il browser sta ancora elaborando il microfono (${processing.join(', ')}): queste elaborazioni cambiano fase e livello del segnale e devono essere disattivate. Usa Google Chrome e, se l'avviso resta, disattivale nelle impostazioni audio del sistema.`,
    );
  }

  // --- Errors -----------------------------------------------------------------------------------
  const lat = info.outputLatencyS;
  if (lat !== null && Number.isFinite(lat) && lat > 0) {
    if (lat > MAX_OUTPUT_LATENCY_S) {
      add(
        ISSUE.outputLatency,
        'error',
        `Latenza d'uscita alta: ${decimal(lat * 1000, 1)} ms (consigliato al massimo ${MAX_OUTPUT_LATENCY_S * 1000} ms). Per il rumore a banda larga il microfono dovrebbe essere almeno ${decimal(lat * C_ROOM, 1)} m più vicino alla sorgente rispetto alla zona. Usa l'interfaccia audio come uscita, la latenza Normale o Ultra-bassa e chiudi le altre app audio.`,
        true,
      );
    }
  } else {
    add(
      ISSUE.outputLatencyUnknown,
      'info',
      "Latenza d'uscita non disponibile in questo momento: verrà misurata durante la calibrazione.",
    );
  }
  const trackSr = ts.sampleRate;
  const ctxSr = info.ctxSampleRate;
  if (
    typeof trackSr === 'number' &&
    Number.isFinite(trackSr) &&
    trackSr > 0 &&
    ctxSr !== null &&
    Number.isFinite(ctxSr) &&
    ctxSr > 0 &&
    trackSr !== ctxSr
  ) {
    add(
      ISSUE.sampleRateMismatch,
      'error',
      `Frequenze di campionamento diverse: microfono a ${formatKHz(trackSr)}, audio dell'app a ${formatKHz(ctxSr)}. Il ricampionamento aggiunge latenza e ritardi variabili: imposta la stessa frequenza (es. 48 kHz) su ingresso e uscita dell'interfaccia.`,
      true,
    );
  }
  if (info.wantErrorMic) {
    const n = errorMicInput(info.errorChannel);
    const reason = errorMicReason(info.ch2Status, n);
    if (reason) {
      // Not overridable on purpose: the way forward is turning the error-mic option off (or fixing the wiring).
      add(
        ISSUE.errorMicUnavailable,
        'error',
        `Mic d'errore disattivato: ${reason}. Collega il microfono d'errore all'ingresso ${n} della stessa interfaccia (phantom e gain se servono) oppure disattiva l'opzione «mic d'errore» per continuare senza.`,
      );
    } else if (info.ch2Status === 'pending') {
      add(ISSUE.errorMicPending, 'info', `Verifica dell'ingresso ${n} (mic d'errore) in corso…`);
    }
  }

  // --- Warnings ---------------------------------------------------------------------------------
  const headphones = outClass?.headphones === true;
  if (output && headphones) {
    add(
      ISSUE.headphones,
      'warn',
      `L'uscita ${deviceName(output)} sembra un paio di cuffie: il limite d'uscita viene forzato a −24 dBFS e ti verrà chiesta una conferma. Non indossarle durante le prime prove.`,
    );
  }
  let sameDevice = false;
  if (input && output) {
    if (input.groupId !== '' && output.groupId !== '') {
      sameDevice = input.groupId === output.groupId;
      if (!sameDevice) {
        add(
          ISSUE.clockDrift,
          'warn',
          'Ingresso e uscita appartengono a dispositivi diversi: i loro clock non sono sincronizzati e il ritardo deriva nel tempo, rovinando la cancellazione. Usa ingresso e uscita della stessa interfaccia audio.',
        );
      }
    } else {
      add(
        ISSUE.sameDeviceUnknown,
        'info',
        'Non è possibile verificare se ingresso e uscita appartengono alla stessa interfaccia audio: usa ingresso e uscita dello stesso dispositivo.',
      );
    }
  }
  if (!output) {
    add(
      ISSUE.outputUnverified,
      'warn',
      "Uscita non identificata (predefinita di sistema): l'app non può verificare che non sia Bluetooth, AirPlay o gli altoparlanti integrati. Imposta l'interfaccia audio come uscita predefinita nelle impostazioni del sistema.",
    );
  }
  if (!info.chromium) {
    add(
      ISSUE.nonChromium,
      'warn',
      "Browser non basato su Chromium: la scelta dell'uscita audio non è disponibile e la latenza è più alta. Usa Google Chrome.",
    );
  }

  // --- Information ------------------------------------------------------------------------------
  if (!info.sinkSupported) {
    add(
      ISSUE.sinkUnsupported,
      'info',
      "Questo browser non permette di scegliere l'uscita audio: viene usata quella predefinita del sistema. Impostala sull'interfaccia audio prima di continuare.",
    );
  }
  if (info.platform === 'windows') {
    add(
      ISSUE.windowsSetup,
      'info',
      "Windows: imposta 48 kHz sia sull'ingresso sia sull'uscita dell'interfaccia (Impostazioni audio → Proprietà del dispositivo → Avanzate) e disattiva i miglioramenti audio su entrambi.",
    );
  }

  for (const i of issues) {
    if (i.overridable && overrides.has(i.id)) i.overridden = true;
  }
  const ok = !issues.some((i) => i.level === 'block' || (i.level === 'error' && i.overridden !== true));
  // Array.prototype.sort is stable, so issues of the same level keep their rule order.
  issues.sort((a, b) => LEVEL_RANK[a.level] - LEVEL_RANK[b.level]);
  return { ok, issues, headphones, sameDevice };
}
