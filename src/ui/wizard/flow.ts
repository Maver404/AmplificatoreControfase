/**
 * Guided-setup flow: which steps run, what each one needs before "Avanti", and what must hold
 * before the anti-noise may be un-muted. Pure (no DOM): tested in tests/unit/wizard-flow.test.ts.
 *
 * Safety rules encoded here: step 1 (devices + validation + output armed muted) is never skipped;
 * step 2 (amplifier level) must be done in this session, for every active channel, on the current
 * output device and sample rate, before any un-mute; the anti-noise only starts at step 7.
 * Measurements and solutions only count for the distances they were made with, and a saved
 * calibration is reused only on its interface and after a quick re-measurement (step "check").
 */
import type { CalibrationData, PathData } from '../../audio/calibration.ts';
import { compareDeviceKeys } from '../../audio/devices.ts';
import type { DeviceKeyMatch } from '../../audio/devices.ts';
import type { ControllerState } from '../../app/controller.ts';
import { firstProblem } from '../../app/controller.ts';
import { errorMicInput } from '../../app/validation.ts';
import { activeChannels, CH_NAME, DEFAULT_GEOMETRY, geometryKey, loopbackSuspects, parseGeometry, positionsKey } from './logic.ts';
import type { GeometryDraft, LayoutChoice } from './logic.ts';

export type StepId = 'start' | 'devices' | 'amp' | 'layout' | 'placement' | 'measure' | 'check' | 'align' | 'verify';

/**
 * full = first configuration; recal = saved calibration but something moved (steps 4–6 again);
 * reuse = saved calibration, verified by a quick re-measurement of the speaker → reference paths.
 */
export type FlowMode = 'full' | 'recal' | 'reuse';

export const STEP_INFO: Record<StepId, { n: number | null; title: string }> = {
  start: { n: null, title: 'Calibrazione salvata' },
  devices: { n: 1, title: 'Dispositivi e verifica' },
  amp: { n: 2, title: 'Livello dell’amplificatore' },
  layout: { n: 3, title: 'Disposizione' },
  placement: { n: 4, title: 'Posizionamento' },
  measure: { n: 5, title: 'Misura dei percorsi' },
  check: { n: 5, title: 'Controllo rapido' },
  align: { n: 6, title: 'Allineamento' },
  verify: { n: 7, title: 'Verifica A/B' },
};

export const LOCK_TEXT =
  'Anti-rumore bloccato dalla configurazione guidata: si attiva solo all’ultimo passo («Verifica A/B»). ' +
  'L’uscita può riprodurre solo i suoni di prova e di misura.';

export function flowSteps(mode: FlowMode, hasSaved: boolean): StepId[] {
  const head: StepId[] = hasSaved ? ['start'] : [];
  switch (mode) {
    case 'reuse':
      return [...head, 'devices', 'amp', 'check', 'verify'];
    case 'recal':
      return [...head, 'devices', 'amp', 'placement', 'measure', 'align', 'verify'];
    default:
      return [...head, 'devices', 'amp', 'layout', 'placement', 'measure', 'align', 'verify'];
  }
}

export function nextStep(steps: readonly StepId[], cur: StepId): StepId | null {
  const i = steps.indexOf(cur);
  return i >= 0 && i + 1 < steps.length ? steps[i + 1]! : null;
}

export function prevStep(steps: readonly StepId[], cur: StepId): StepId | null {
  const i = steps.indexOf(cur);
  return i > 0 ? steps[i - 1]! : null;
}

// ------------------------------------------------------------------------------------- data

export type PathState = 'idle' | 'running' | 'ok' | 'failed' | 'aborted';

export interface PathSlot {
  state: PathState;
  reason: string | null;
  path: PathData | null;
  /** positionsKey() of the channel when it was measured (the result only counts for those distances). */
  posKey: string | null;
}

export const emptySlot = (): PathSlot => ({ state: 'idle', reason: null, path: null, posKey: null });

/** Guided loopback on input 2: none → cable connected (checklist) → measured → cable removed and checked. */
export type LoopbackStage = 'none' | 'connect' | 'measured' | 'verified';

export interface SolutionInfo {
  method: 'error-mic' | 'two-position' | 'geometry' | 'preset';
  applied: boolean;
  predictedDb: number | null;
  loopGainDb: number | null;
  causal: boolean | null;
  advice: string[];
  /** geometryKey() the solution was computed with; null when it does not depend on the geometry. */
  geometryKey: string | null;
}

/** The saved calibration chosen at the start question (reuse / recal). */
export interface ReuseInfo {
  presetId: string;
  name: string;
  /** Interfaces it was measured with (presets.ts deviceKey; '' = unknown). */
  deviceKey: string;
  /** Saved master volume: the amplifier step raises it to 0 dB and gives it back afterwards. */
  masterDb: number;
}

/** Outcome of the quick re-measurement of one channel against the saved calibration. */
export interface CheckSlot {
  state: 'running' | 'ok' | 'corrected' | 'unverified' | 'mismatch' | 'aborted';
  text: string;
}

export interface WizardData {
  mode: FlowMode;
  hasSaved: boolean;
  /** The start question was answered (only meaningful with hasSaved). */
  modeChosen: boolean;
  presetId: string | null;
  layout: LayoutChoice;
  geometry: GeometryDraft;
  amp: {
    /** amp at minimum · hand on the volume · Esc needs the window focused */
    checklist: [boolean, boolean, boolean];
    /** Seconds of noise played per channel. */
    playedS: [number, number];
    leveled: [boolean, boolean];
    /** Output device and sample rate the levels were set on ("outputId|fs"). */
    key: [string | null, string | null];
  };
  measure: {
    feedback: [PathSlot, PathSlot];
    secondary: [PathSlot, PathSlot];
    /** The user accepted continuing although the speaker was not heard at the reference mic. */
    acceptWeakFeedback: [boolean, boolean];
    loopback: LoopbackStage;
    /** Error-mic input to restore after the loopback (input 2 is borrowed for the cable). */
    loopbackPrevErr: 0 | 1 | -1;
  };
  calibration: CalibrationData | null;
  solution: SolutionInfo | null;
  reuse: ReuseInfo | null;
  check: {
    /** ampKey() the check was made on (output and rate). */
    key: string | null;
    channels: [CheckSlot | null, CheckSlot | null];
    /** The user accepted the saved values for channels that could not be verified. */
    acceptUnverified: boolean;
  };
  verify: { unmuted: boolean };
  /** Device key when the run was completed (a later change of interface locks the anti-noise again). */
  doneDeviceKey: string | null;
}

export function createWizardData(opts: { hasSaved: boolean; layout?: LayoutChoice; geometry?: GeometryDraft }): WizardData {
  const g = opts.geometry ?? DEFAULT_GEOMETRY;
  return {
    mode: 'full',
    hasSaved: opts.hasSaved,
    modeChosen: !opts.hasSaved,
    presetId: null,
    layout: opts.layout ?? 'A',
    geometry: { ...g, speakerToZoneM: [g.speakerToZoneM[0], g.speakerToZoneM[1]] },
    amp: { checklist: [false, false, false], playedS: [0, 0], leveled: [false, false], key: [null, null] },
    measure: {
      feedback: [emptySlot(), emptySlot()],
      secondary: [emptySlot(), emptySlot()],
      acceptWeakFeedback: [false, false],
      loopback: 'none',
      loopbackPrevErr: -1,
    },
    calibration: null,
    solution: null,
    reuse: null,
    check: emptyCheck(),
    verify: { unmuted: false },
    doneDeviceKey: null,
  };
}

export const emptyCheck = (): WizardData['check'] => ({ key: null, channels: [null, null], acceptUnverified: false });

/**
 * The amplifier of channel `ch` is being levelled again (step 2: level noise played, or its "Ho
 * regolato…" box confirmed): everything measured through it may be off by any amount. The quick check
 * of a reused calibration must be repeated; in the other flows the channel's measured paths and the
 * solution computed from them go (their gains scale with the amplifier).
 */
export function amplifierReleveled(d: WizardData, ch: 0 | 1): void {
  d.check = emptyCheck();
  if (d.mode === 'reuse') return;
  const m = d.measure;
  const measured = m.feedback[ch].state !== 'idle' || m.secondary[ch].state !== 'idle';
  if (!measured && d.solution === null) return;
  m.feedback[ch] = emptySlot();
  m.secondary[ch] = emptySlot();
  m.acceptWeakFeedback[ch] = false;
  if (d.calibration !== null) {
    d.calibration.feedback[ch] = null;
    d.calibration.secondary[ch] = null;
  }
  d.solution = null;
}

/** Clears measurements and the solution (devices, sample rate or positions changed). */
export function resetMeasurements(d: WizardData): void {
  d.measure.feedback = [emptySlot(), emptySlot()];
  d.measure.secondary = [emptySlot(), emptySlot()];
  d.measure.acceptWeakFeedback = [false, false];
  d.calibration = null;
  d.solution = null;
  d.check = emptyCheck();
}

// ------------------------------------------------------------------------------------- gates

/** The controller state the gates look at (a subset, so tests can build it). */
export type GateState = Pick<
  ControllerState,
  | 'micOpen'
  | 'opening'
  | 'armed'
  | 'panic'
  | 'validation'
  | 'headphonesConfirmed'
  | 'outputId'
  | 'fs'
  | 'ch2'
  | 'measuring'
  | 'capturing'
  | 'params'
  | 'speakers'
  | 'deviceKey'
>;

export interface Gate {
  ok: boolean;
  /** Italian: what is missing before "Avanti". */
  reason: string | null;
}

const pass: Gate = { ok: true, reason: null };
const block = (reason: string): Gate => ({ ok: false, reason });

export const ampKey = (s: Pick<GateState, 'outputId' | 'fs'>): string => `${s.outputId ?? 'default'}|${s.fs ?? 0}`;

/** Amplifier level of channel `ch` set in this session on the current output and sample rate. */
export const ampLeveledOk = (d: WizardData, s: Pick<GateState, 'outputId' | 'fs'>, ch: 0 | 1): boolean =>
  d.amp.leveled[ch] && d.amp.key[ch] === ampKey(s);

export const errorMicOn = (s: Pick<GateState, 'params'>): boolean => s.params.errorChannel >= 0;

/**
 * Interfaces now vs the ones the reused calibration was measured with ('same' outside the reuse
 * flow; 'unknown' when either key is not known, e.g. before the microphone is open).
 */
export function reuseDeviceMatch(d: WizardData, s: GateState): DeviceKeyMatch | 'unknown' {
  if (d.mode !== 'reuse' || d.reuse === null) return 'same';
  const cur = s.deviceKey;
  if (d.reuse.deviceKey === '' || cur === '') return 'unknown';
  return compareDeviceKeys(d.reuse.deviceKey, cur);
}

export const REUSE_OTHER_DEVICE_TEXT =
  'La calibrazione salvata è stata misurata con un’altra interfaccia audio: scegli quella interfaccia oppure rifai le misure.';

/** A measured path (or an accepted failure) whose channel distances were edited afterwards. */
const slotStale = (d: WizardData, slot: PathSlot, ch: 0 | 1): boolean =>
  (slot.state === 'ok' || slot.state === 'failed') && slot.posKey !== positionsKey(d.geometry, ch);

/** Paths measured before the distances of their channel were changed (Italian names). */
export function stalePaths(s: GateState, d: WizardData): string[] {
  const out: string[] = [];
  for (const ch of activeChannels(s.speakers)) {
    if (slotStale(d, d.measure.feedback[ch], ch)) out.push(`cassa ${CH_NAME[ch]} → microfono di riferimento`);
    if (slotStale(d, d.measure.secondary[ch], ch)) out.push(`cassa ${CH_NAME[ch]} → mic d’errore`);
  }
  return out;
}

/** The applied solution used distances, angle or temperature that have been edited since. */
export function solutionStale(s: GateState, d: WizardData): boolean {
  const k = d.solution?.geometryKey ?? null;
  return k !== null && k !== geometryKey(d.geometry, activeChannels(s.speakers));
}

const STALE_SOLUTION_TEXT = 'Hai cambiato distanze, angolo o temperatura dopo l’allineamento: premi di nuovo «Calcola e applica» (passo 6).';

/**
 * The calibration (measured paths, the solution, a reused one and its quick check) was made with the
 * reference microphone on the other input: «Canale del riferimento» changed since. The controller then
 * drops the feedback neutralisation too.
 */
export function refInputChanged(s: Pick<GateState, 'params'>, d: WizardData): boolean {
  const r = d.calibration?.refChannel;
  return r !== undefined && r !== s.params.refChannel;
}

export const REF_CHANGED_TEXT =
  'Le misure sono state fatte con il microfono di riferimento sull’altro ingresso: rimetti com’era «Canale del riferimento» (passo 1) ' +
  'oppure rifai le misure.';

/** The quick check of a reused calibration passed for every active channel, on this output and rate. */
export function checkPassed(s: GateState, d: WizardData): boolean {
  if (d.check.key !== ampKey(s)) return false;
  return activeChannels(s.speakers).every((ch) => {
    const r = d.check.channels[ch];
    return r !== null && (r.state === 'ok' || r.state === 'corrected' || (r.state === 'unverified' && d.check.acceptUnverified));
  });
}

function devicesGate(s: GateState, d: WizardData): Gate {
  if (s.opening) return block('Apertura del microfono in corso…');
  if (!s.micOpen) return block('Premi «Consenti microfono» e scegli l’interfaccia audio.');
  if (!s.validation.ok) {
    const p = firstProblem(s.validation);
    return block(p ? `Risolvi prima: ${p.text}` : 'Validazione dei dispositivi non superata.');
  }
  if (s.validation.headphones && !s.headphonesConfirmed) return block('Conferma l’avviso sulle cuffie.');
  if (s.panic) return block('PANIC attivo: riprendi prima l’uscita.');
  if (!s.armed) return block('Premi «Abilita l’uscita per le misure» (l’anti-rumore resta muto).');
  if (reuseDeviceMatch(d, s) === 'different') return block(REUSE_OTHER_DEVICE_TEXT);
  return pass;
}

function ampGate(s: GateState, d: WizardData): Gate {
  if (!d.amp.checklist.every(Boolean)) return block('Spunta le tre voci della lista di sicurezza.');
  if (!ampLeveledOk(d, s, 0)) {
    return block('Fai suonare il rumore di prova sul canale L e conferma di aver regolato l’amplificatore.');
  }
  // With a saved configuration the speakers are already known (the layout step is skipped).
  if (d.mode !== 'full' && activeChannels(s.speakers).includes(1) && !ampLeveledOk(d, s, 1)) {
    return block('La configurazione usa due casse: fai suonare il rumore di prova anche sul canale R e conferma.');
  }
  return pass;
}

function layoutGate(s: GateState, d: WizardData): Gate {
  if (activeChannels(s.speakers).includes(1) && !ampLeveledOk(d, s, 1)) {
    return block('Con due casse regola anche l’amplificatore del canale R: torna al passo 2.');
  }
  if (errorMicOn(s) && s.ch2 !== 'ok') {
    const n = errorMicInput(s.params.errorChannel);
    return block(
      s.ch2 === 'pending'
        ? `Verifica dell’ingresso ${n} (mic d’errore) in corso…`
        : `Il mic d’errore sull’ingresso ${n} non è utilizzabile: collegalo o scegli «senza mic d’errore».`,
    );
  }
  return pass;
}

function placementGate(s: GateState, d: WizardData): Gate {
  const g = parseGeometry(d.geometry, activeChannels(s.speakers));
  return g.ok ? pass : block(g.errors[0] ?? 'Completa le distanze.');
}

function checkGate(s: GateState, d: WizardData): Gate {
  if (reuseDeviceMatch(d, s) === 'different') return block(REUSE_OTHER_DEVICE_TEXT);
  if (d.calibration === null || d.calibration.fs !== s.fs) {
    return block('La calibrazione salvata è a un’altra frequenza di campionamento: rifai le misure.');
  }
  if (refInputChanged(s, d)) return block(REF_CHANGED_TEXT);
  if (d.check.key !== ampKey(s)) return block('Esegui il controllo rapido delle casse.');
  for (const ch of activeChannels(s.speakers)) {
    const r = d.check.channels[ch];
    if (r === null || r.state === 'running' || r.state === 'aborted') return block(`Esegui il controllo rapido della cassa ${CH_NAME[ch]}.`);
    if (r.state === 'mismatch') return block(`Cassa ${CH_NAME[ch]}: ${r.text}`);
    if (r.state === 'unverified' && !d.check.acceptUnverified) {
      return block(`Cassa ${CH_NAME[ch]} non verificata: conferma di voler usare i valori salvati o rifai le misure.`);
    }
  }
  return pass;
}

function measureGate(s: GateState, d: WizardData): Gate {
  if (d.measure.loopback === 'connect' || d.measure.loopback === 'measured') {
    return block('Scollega il cavo di loopback e completa la verifica prima di continuare.');
  }
  const cal = d.calibration;
  if (refInputChanged(s, d)) return block(REF_CHANGED_TEXT);
  const stale = stalePaths(s, d);
  if (stale.length > 0) {
    return block(`Hai cambiato le distanze dopo la misura (${stale.join(', ')}): ripetila, microfono o casse sono stati spostati.`);
  }
  for (const ch of activeChannels(s.speakers)) {
    const fb = d.measure.feedback[ch];
    if (fb.state !== 'ok' && !(fb.state === 'failed' && d.measure.acceptWeakFeedback[ch])) {
      return block(`Misura il percorso cassa ${CH_NAME[ch]} → microfono di riferimento.`);
    }
    if (errorMicOn(s) && d.measure.secondary[ch].state !== 'ok') {
      return block(`Misura il percorso cassa ${CH_NAME[ch]} → mic d’errore.`);
    }
  }
  if (cal === null || cal.fs !== s.fs) return block('Ripeti le misure: la frequenza di campionamento è cambiata.');
  return pass;
}

function alignGate(s: GateState, d: WizardData): Gate {
  if (refInputChanged(s, d)) return block(REF_CHANGED_TEXT);
  if (d.solution === null || !d.solution.applied) return block('Calcola e applica l’allineamento.');
  if (solutionStale(s, d)) return block(STALE_SOLUTION_TEXT);
  if (d.calibration !== null && d.calibration.fs !== s.fs) {
    return block('La frequenza di campionamento è cambiata dopo le misure: ripetile.');
  }
  return pass;
}

function startGate(d: WizardData): Gate {
  return d.modeChosen ? pass : block('Rispondi alla domanda per continuare.');
}

/** What step `step` still needs before "Avanti". Measuring/capturing always blocks navigation. */
export function stepGate(step: StepId, s: GateState, d: WizardData): Gate {
  if (s.measuring || s.capturing) return block('Attendi la fine della misura in corso.');
  switch (step) {
    case 'start':
      return startGate(d);
    case 'devices':
      return devicesGate(s, d);
    case 'amp': {
      const dev = devicesGate(s, d);
      return dev.ok ? ampGate(s, d) : dev;
    }
    case 'layout':
      return layoutGate(s, d);
    case 'placement':
      return placementGate(s, d);
    case 'measure':
      return measureGate(s, d);
    case 'check':
      return checkGate(s, d);
    case 'align':
      return alignGate(s, d);
    case 'verify':
      return pass;
  }
}

/**
 * Everything that must hold before the anti-noise may be un-muted (Italian, empty = go). Checked
 * by step 7 and by the autotune, which also needs the anti-noise playing.
 */
export function unmuteBlockers(s: GateState, d: WizardData): string[] {
  const out: string[] = [];
  if (s.panic) out.push('PANIC attivo: riprendi prima l’uscita.');
  if (!s.micOpen) out.push('Microfono non aperto (passo 1).');
  else if (!s.validation.ok) out.push('Validazione dei dispositivi non superata (passo 1).');
  else if (!s.armed) out.push('Uscita non abilitata (passo 1).');
  for (const ch of activeChannels(s.speakers)) {
    if (!ampLeveledOk(d, s, ch)) out.push(`Livello dell’amplificatore del canale ${CH_NAME[ch]} non regolato in questa sessione (passo 2).`);
  }
  if (d.solution === null || !d.solution.applied) out.push('Nessun allineamento applicato (passo 6).');
  else if (solutionStale(s, d)) out.push(STALE_SOLUTION_TEXT);
  if (refInputChanged(s, d)) out.push(REF_CHANGED_TEXT);
  const stale = stalePaths(s, d);
  if (stale.length > 0) out.push(`Hai cambiato le distanze dopo le misure (${stale.join(', ')}): ripetile (passo 5).`);
  if (d.mode === 'reuse') {
    if (reuseDeviceMatch(d, s) === 'different') out.push(REUSE_OTHER_DEVICE_TEXT);
    else if (!checkPassed(s, d)) out.push('Controllo rapido della calibrazione salvata non completato (passo 5).');
  }
  const cal = d.calibration;
  if (cal !== null && s.fs !== null && cal.fs !== s.fs) {
    out.push(
      `La calibrazione è a ${(cal.fs / 1000).toString().replace('.', ',')} kHz, l’audio ora a ${(s.fs / 1000)
        .toString()
        .replace('.', ',')} kHz: rifai le misure.`,
    );
  }
  if (d.measure.loopback === 'connect' || d.measure.loopback === 'measured') {
    out.push('Cavo di loopback: scollegalo e completa la verifica (passo 5).');
  }
  const suspects = loopbackSuspects(cal);
  if (suspects.length > 0) {
    out.push(
      `Le misure ${suspects.join(', ')} hanno la firma di un cavo di loopback, non di un percorso acustico: ` +
        'scollega il cavo e ripeti le misure.',
    );
  }
  if (s.measuring || s.capturing) out.push('Misura in corso.');
  return out;
}

export const REF_RELOCK_TEXT =
  '«Canale del riferimento» cambiato dopo la configurazione guidata: l’allineamento e la neutralizzazione del feedback valgono per ' +
  'l’altro ingresso. L’anti-rumore resta muto finché non riprendi la configurazione guidata: rimetti il canale com’era (passo 1) ' +
  'oppure rifai le misure.';

export const RELOCK_TEXT =
  'Dispositivi o frequenza di campionamento cambiati dopo la configurazione guidata: l’anti-rumore resta muto finché non la ' +
  'riprendi (almeno dispositivi e livello dell’amplificatore; con un’altra frequenza anche le misure).';

/**
 * Why a completed setup must lock the anti-noise again (the console could otherwise arm it un-muted):
 * the amplifier level of an active channel no longer matches the output or rate, the calibration
 * is at another rate or was measured with the reference microphone on the other input, or the
 * interfaces changed since the run was completed. null = still valid.
 */
export function relockReason(s: GateState, d: WizardData): string | null {
  if (refInputChanged(s, d)) return REF_RELOCK_TEXT;
  if (!s.micOpen || s.opening || s.fs === null) return null;
  const amp = activeChannels(s.speakers).some((ch) => !ampLeveledOk(d, s, ch));
  const rate = d.calibration !== null && d.calibration.fs !== s.fs;
  const now = s.deviceKey;
  const device = d.doneDeviceKey !== null && d.doneDeviceKey !== '' && now !== '' && now !== d.doneDeviceKey;
  return amp || rate || device ? RELOCK_TEXT : null;
}

// ------------------------------------------------------------------------------------- banner

export type BannerAction = 'resume' | 'arm' | 'devices';

export interface Banner {
  level: 'info' | 'warn' | 'error';
  text: string;
  action: BannerAction | null;
}

/** Problem shown above every step (PANIC, output disarmed, microphone closed). */
export function bannerFor(step: StepId, s: GateState & { canArm: boolean }): Banner | null {
  if (s.panic) {
    return {
      level: 'error',
      text: 'PANIC: uscita azzerata e audio sospeso. Controlla l’amplificatore (abbassalo se serve), poi riprendi: l’anti-rumore resta muto.',
      action: 'resume',
    };
  }
  if (step === 'start' || step === 'devices') return null;
  if (!s.micOpen && !s.opening) {
    return { level: 'warn', text: 'Il microfono non è aperto: torna al passo 1.', action: 'devices' };
  }
  if (s.micOpen && !s.armed && !s.opening) {
    return s.canArm
      ? {
          level: 'warn',
          text: 'L’uscita è stata disabilitata (dispositivi cambiati o verifica ripetuta): riabilitala per le misure; l’anti-rumore resta muto.',
          action: 'arm',
        }
      : {
          level: 'warn',
          text: 'L’uscita è disabilitata e la verifica dei dispositivi non passa: torna al passo 1.',
          action: 'devices',
        };
  }
  return null;
}
