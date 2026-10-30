/**
 * Top bar: Avvio/Stop, PANIC (always visible), Mute, Bypass A/B, master volume, engine state,
 * sample rate, render quantum, latencies, underrun counter, a warnings badge (link to the list)
 * and the howl / auto-mute message. Callbacks only: the controller binds Escape / M / B and
 * decides what each action does. Also hosts the light/dark theme toggle (pure presentation,
 * persisted per browser). Updated at meter rate: texts are written only when they change (the
 * state chip is a live region).
 */
import { MASTER_MIN_DB, RANGES } from '../dsp/params.ts';
import { createPrecisionControl } from './precision-control.ts';
import { formatDb, formatNumber } from './units.ts';
import type { WarningLevel } from './warnings.ts';

export type EngineStatus = 'stopped' | 'starting' | 'listening' | 'running' | 'muted' | 'panic' | 'error';

export interface StatusMessage {
  level: 'info' | 'warn' | 'error';
  text: string;
}

export interface StatusBarData {
  status: EngineStatus;
  /** Overrides the default chip text for the status. */
  statusText?: string;
  /** Overrides the chip's data-state (e.g. 'measuring', 'bypass'). */
  statusState?: string;
  /** Running = audio context active (Stop is offered). */
  running: boolean;
  /** Start is allowed (validation passed). */
  canStart: boolean;
  /** Why Start is not allowed (tooltip of the disabled button). */
  startHint?: string;
  /** Count and worst level of the warnings list (#warnings); omitted = no badge. */
  warnings?: { count: number; level: WarningLevel | null };
  muted: boolean;
  bypass: boolean;
  masterDb: number;
  /**
   * A calibration stimulus is playing: the engine would abort it on a master change (its level is
   * fixed while it plays), so the control is disabled until it ends.
   */
  masterLocked?: boolean;
  sampleRate: number | null;
  renderQuantum: number | null;
  baseLatencyMs: number | null;
  outputLatencyMs: number | null;
  /** Input track latency (MediaStreamTrack settings), ms. */
  trackLatencyMs: number | null;
  underruns: number;
  /** Howl / auto-mute / fault message, shown prominently. */
  message: StatusMessage | null;
}

export interface StatusBarOptions {
  onStartStop: () => void;
  onPanic: () => void;
  onMuteToggle: () => void;
  onBypassToggle: () => void;
  onMasterChange: (db: number) => void;
  onMessageDismiss?: () => void;
  /** Show the theme toggle (default true). */
  themeToggle?: boolean;
}

export interface StatusBar {
  update(data: StatusBarData): void;
  destroy(): void;
}

/** Output latency above this is flagged (see the plan's validation rules). */
export const OUTPUT_LATENCY_WARN_MS = 40;

const STATUS_TEXT: Record<EngineStatus, string> = {
  stopped: 'Fermo',
  starting: 'Avvio…',
  listening: 'In ascolto · uscita bloccata',
  running: 'Attivo',
  muted: 'Muto',
  panic: 'PANIC · silenziato',
  error: 'Errore',
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function button(text: string, className: string): HTMLButtonElement {
  const b = el('button', className, text);
  b.type = 'button';
  return b;
}

const ms = (v: number | null): string => (v === null || !Number.isFinite(v) ? '—' : `${formatNumber(v, 1)} ms`);

/** Writes text only when it changed (live regions must not be re-announced at meter rate). */
function setText(e: HTMLElement, text: string): void {
  if (e.textContent !== text) e.textContent = text;
}

/** "1 avviso" / "3 avvisi". */
export const warningsBadgeText = (n: number): string => (n === 1 ? '1 avviso' : `${n} avvisi`);

/**
 * Chip text when the engine state alone would mislead: a test stimulus or sweep playing through
 * the amplifier while the anti-noise is muted («Muto»), or the A/B bypass while the output runs.
 */
export function statusOverride(s: {
  panic: boolean;
  armed: boolean;
  muted: boolean;
  bypass: boolean;
  measuring: boolean;
}): Pick<StatusBarData, 'statusText' | 'statusState'> {
  if (s.panic || !s.armed) return {};
  if (s.measuring) return { statusText: 'Suono di prova in corso', statusState: 'measuring' };
  if (s.bypass && !s.muted) return { statusText: 'Bypass A/B · anti-rumore spento', statusState: 'bypass' };
  return {};
}

/** What «Avvio» can do now, and the real reason when it cannot (its tooltip). */
export function startControl(s: {
  armed: boolean;
  panic: boolean;
  canArm: boolean;
  micOpen: boolean;
  opening: boolean;
  measuring: boolean;
  /** First validation problem (Italian), null when the validation passes. */
  problem: string | null;
  /** Headphones detected and not confirmed yet. */
  headphonesPending: boolean;
}): { canStart: boolean; hint: string } {
  // Armed after a PANIC: Avvio resumes; disarmed: Avvio is «Attiva uscita».
  if (s.armed && s.panic) return { canStart: true, hint: '' };
  if (!s.micOpen) return { canStart: false, hint: s.opening ? 'Apertura del microfono in corso…' : 'Prima consenti il microfono (pannello Dispositivi).' };
  if (s.measuring) return { canStart: false, hint: 'Misura in corso: attendi la fine.' };
  if (s.problem !== null) return { canStart: false, hint: `Uscita bloccata: ${s.problem}` };
  if (s.headphonesPending) return { canStart: false, hint: 'Conferma l’avviso sulle cuffie nel riquadro Uscita.' };
  return s.canArm ? { canStart: true, hint: '' } : { canStart: false, hint: 'Uscita non attivabile ora: controlla i dispositivi.' };
}

export function createStatusBar(container: HTMLElement, opts: StatusBarOptions): StatusBar {
  const root = el('div', 'status-bar');

  // Primary actions
  const actions = el('div', 'sb-actions');
  const startBtn = button('Avvio', 'btn btn--primary sb-start');
  startBtn.addEventListener('click', () => opts.onStartStop());
  const panicBtn = button('', 'panic-btn');
  panicBtn.append(el('span', 'panic-main', 'PANIC'), el('span', 'panic-key', 'Esc'));
  panicBtn.setAttribute('aria-keyshortcuts', 'Escape');
  panicBtn.setAttribute('aria-label', 'PANIC: silenzia subito tutte le uscite (tasto Esc)');
  panicBtn.addEventListener('click', () => opts.onPanic());
  const muteBtn = button('', 'toggle-btn sb-mute');
  muteBtn.append(el('span', undefined, 'Mute'), el('kbd', undefined, 'M'));
  muteBtn.setAttribute('aria-keyshortcuts', 'M');
  muteBtn.addEventListener('click', () => opts.onMuteToggle());
  const bypassBtn = button('', 'toggle-btn sb-bypass');
  bypassBtn.append(el('span', undefined, 'Bypass A/B'), el('kbd', undefined, 'B'));
  bypassBtn.setAttribute('aria-keyshortcuts', 'B');
  bypassBtn.title = 'Confronto: anti-rumore spento / acceso (rampa di mute)';
  bypassBtn.addEventListener('click', () => opts.onBypassToggle());
  actions.append(panicBtn, startBtn, muteBtn, bypassBtn);

  // Master
  const masterBox = el('div', 'sb-master');
  const master = createPrecisionControl(masterBox, {
    label: 'Volume master',
    value: -30,
    min: RANGES.masterDb.min,
    max: RANGES.masterDb.max,
    step: 0.1,
    unit: 'dB',
    compact: true,
    format: (v) => formatDb(v, 1, MASTER_MIN_DB),
    onChange: (v) => opts.onMasterChange(v),
  });

  // State + info
  const info = el('div', 'sb-info');
  const chip = el('span', 'chip sb-state');
  chip.setAttribute('role', 'status');
  const dl = el('dl', 'sb-stats');
  const stat = (label: string, title?: string): HTMLElement => {
    const wrap = el('div', 'sb-stat');
    const dt = el('dt', undefined, label);
    if (title) wrap.title = title;
    const dd = el('dd', 'num', '—');
    wrap.append(dt, dd);
    dl.append(wrap);
    return dd;
  };
  const srDd = stat('Freq. camp.');
  const rqDd = stat('Blocco', 'Dimensione del blocco di elaborazione (render quantum)');
  const latDd = stat('Latenza', 'Base + uscita + ingresso, come riportate dal browser');
  const urDd = stat('Underrun', 'Interruzioni del flusso audio');
  const warnBadge = el('a', 'sb-warn-badge');
  warnBadge.href = '#warnings';
  warnBadge.hidden = true;
  info.append(chip, warnBadge, dl);

  let themeBtn: HTMLButtonElement | null = null;
  if (opts.themeToggle !== false) {
    themeBtn = createThemeToggle(info);
  }

  const msg = el('div', 'sb-message');
  msg.setAttribute('role', 'alert');
  const msgText = el('span', 'sb-message-text');
  const msgClose = button('Chiudi', 'btn btn--small');
  msgClose.addEventListener('click', () => opts.onMessageDismiss?.());
  msg.append(msgText);
  if (opts.onMessageDismiss) msg.append(msgClose);

  root.append(actions, masterBox, info, msg);
  container.append(root);

  function update(d: StatusBarData): void {
    setText(startBtn, d.running ? 'Stop' : 'Avvio');
    startBtn.disabled = !d.running && !d.canStart;
    startBtn.classList.toggle('btn--primary', !d.running);
    startBtn.title = !d.running && !d.canStart ? (d.startHint ?? 'Completa la configurazione e la validazione dei dispositivi') : '';
    muteBtn.setAttribute('aria-pressed', String(d.muted));
    bypassBtn.setAttribute('aria-pressed', String(d.bypass));
    master.update(d.masterDb, { disabled: d.masterLocked === true });
    masterBox.title = d.masterLocked === true ? 'Volume master bloccato durante il suono di prova (cambiarlo lo interromperebbe).' : '';

    setText(chip, d.statusText ?? STATUS_TEXT[d.status]);
    chip.dataset['state'] = d.statusState ?? d.status;
    root.dataset['state'] = d.status;

    const w = d.warnings;
    warnBadge.hidden = !w || w.count <= 0;
    if (w && w.count > 0) {
      setText(warnBadge, warningsBadgeText(w.count));
      warnBadge.dataset['level'] = w.level ?? 'info';
      warnBadge.setAttribute('aria-label', `${warningsBadgeText(w.count)}: vai all’elenco degli avvisi`);
    }

    setText(srDd, d.sampleRate ? `${formatNumber(d.sampleRate / 1000, 1)} kHz` : '—');
    setText(rqDd, d.renderQuantum ? `${d.renderQuantum} campioni` : '—');
    const parts = [d.baseLatencyMs, d.outputLatencyMs, d.trackLatencyMs];
    const known = parts.filter((v): v is number => v !== null && Number.isFinite(v));
    const total = known.reduce((a, b) => a + b, 0);
    setText(latDd, known.length ? `≈ ${formatNumber(total, 1)} ms` : '—');
    latDd.parentElement!.title =
      `Base ${ms(d.baseLatencyMs)} + uscita ${ms(d.outputLatencyMs)} + ingresso ${ms(d.trackLatencyMs)}`;
    const latHigh = d.outputLatencyMs !== null && d.outputLatencyMs > OUTPUT_LATENCY_WARN_MS;
    latDd.classList.toggle('is-warn', latHigh);
    setText(urDd, String(Math.max(0, Math.floor(d.underruns))));
    urDd.classList.toggle('is-warn', d.underruns > 0);

    const m = d.message;
    msg.dataset['level'] = m ? m.level : '';
    msg.classList.toggle('is-empty', !m);
    setText(msgText, m?.text ?? '');
    msgClose.hidden = !m;
  }

  update({
    status: 'stopped',
    running: false,
    canStart: false,
    muted: false,
    bypass: false,
    masterDb: -30,
    sampleRate: null,
    renderQuantum: null,
    baseLatencyMs: null,
    outputLatencyMs: null,
    trackLatencyMs: null,
    underruns: 0,
    message: null,
  });

  return {
    update,
    destroy(): void {
      master.destroy();
      themeBtn?.remove();
      root.remove();
    },
  };
}

// ---- Theme ----

export type ThemeChoice = 'auto' | 'light' | 'dark';
/** Must match the pre-paint script public/theme-init.js (loaded by index.html before the first paint). */
export const THEME_STORAGE_KEY = 'controfase.theme';
const THEME_LABEL: Record<ThemeChoice, string> = { auto: 'Automatico', light: 'Chiaro', dark: 'Scuro' };
const THEME_NEXT: Record<ThemeChoice, ThemeChoice> = { auto: 'light', light: 'dark', dark: 'auto' };

export function readThemeChoice(): ThemeChoice {
  try {
    const t = window.localStorage.getItem(THEME_STORAGE_KEY);
    return t === 'light' || t === 'dark' ? t : 'auto';
  } catch {
    return 'auto';
  }
}

export function applyThemeChoice(choice: ThemeChoice): void {
  const rootEl = document.documentElement;
  if (choice === 'auto') delete rootEl.dataset['theme'];
  else rootEl.dataset['theme'] = choice;
  try {
    if (choice === 'auto') window.localStorage.removeItem(THEME_STORAGE_KEY);
    else window.localStorage.setItem(THEME_STORAGE_KEY, choice);
  } catch {
    // Storage blocked (private mode): the choice still applies for this page view.
  }
}

/** Button cycling Automatico → Chiaro → Scuro; sets data-theme on <html>. */
export function createThemeToggle(container: HTMLElement): HTMLButtonElement {
  let choice = readThemeChoice();
  const btn = button('', 'btn btn--small theme-toggle');
  const render = (): void => {
    btn.textContent = `Tema: ${THEME_LABEL[choice]}`;
    btn.setAttribute('aria-label', `Tema: ${THEME_LABEL[choice]}. Cambia tema`);
  };
  btn.addEventListener('click', () => {
    choice = THEME_NEXT[choice];
    applyThemeChoice(choice);
    render();
  });
  applyThemeChoice(choice);
  render();
  container.append(btn);
  return btn;
}
