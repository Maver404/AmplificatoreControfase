/**
 * "Configurazione guidata": the guided setup shell. Shown by default at start (first run, or the
 * "Hai spostato…?" question when a calibration is saved), re-openable from its card in the console.
 * While a run is in progress the controller's anti-noise lock is engaged: the output can only play
 * stimuli, and the anti-noise is un-muted only at step 7 (or by the autotune of step 6, after the
 * same checks). Every step is a view (step-*.ts); gating and the step sequence live in flow.ts.
 * Once a run is completed and the wizard closed, a change of output, rate or interface locks the
 * anti-noise again (flow.ts relockReason): the console cannot arm it un-muted on an amplifier
 * that was never levelled.
 */
import './wizard.css';
import type { AppController, ControllerChange, ControllerState } from '../../app/controller.ts';
import { createAnalyzer } from '../../audio/analysis-client.ts';
import type { WorkerAnalyzer } from '../../audio/analysis-client.ts';
import { button, el, setText } from './dom.ts';
import type { Op, StepEnv, StepFactory, StepView } from './env.ts';
import {
  bannerFor,
  createWizardData,
  flowSteps,
  LOCK_TEXT,
  nextStep,
  prevStep,
  relockReason,
  STEP_INFO,
  stepGate,
  unmuteBlockers,
} from './flow.ts';
import type { FlowMode, StepId, WizardData } from './flow.ts';
import { parseWizardPrefs, presetsWithCalibration, serializeWizardPrefs, WIZARD_PREFS_KEY } from './logic.ts';
import { createAlignStep } from './step-align.ts';
import { createAmpStep } from './step-amp.ts';
import { createCheckStep } from './step-check.ts';
import { createDevicesStep } from './step-devices.ts';
import { createLayoutStep } from './step-layout.ts';
import { createMeasureStep } from './step-measure.ts';
import { createPlacementStep } from './step-placement.ts';
import { createStartStep } from './step-start.ts';
import { createVerifyStep } from './step-verify.ts';

export interface WizardOptions {
  /** Open at start (default true). The e2e build passes false: the console stays as it was. */
  autoOpen?: boolean;
  /** Where layout and distances are remembered: undefined = localStorage if usable, null = nowhere. */
  storage?: Storage | null;
}

export interface Wizard {
  open(): void;
  close(): void;
  readonly isOpen: boolean;
  destroy(): void;
}

const FACTORIES: Record<StepId, StepFactory> = {
  start: createStartStep,
  devices: createDevicesStep,
  amp: createAmpStep,
  layout: createLayoutStep,
  placement: createPlacementStep,
  measure: createMeasureStep,
  check: createCheckStep,
  align: createAlignStep,
  verify: createVerifyStep,
};

const INTRO: Record<StepId, string> = {
  start: 'Trovata una calibrazione salvata.',
  devices: 'Microfono, interfaccia audio e controlli di sicurezza.',
  amp: 'Il volume dell’amplificatore, con un rumore di prova.',
  layout: 'Dove sono microfono, casse e zona da silenziare.',
  placement: 'Distanze e direzione della sorgente: il rumore arriva in tempo?',
  measure: 'Ritardi e livelli reali, con brevi sweep.',
  check: 'Le casse suonano ancora come nella calibrazione salvata?',
  align: 'Ritardo, guadagno e polarità dell’anti-rumore.',
  verify: 'Anti-rumore acceso, confronto con il bypass, salvataggio.',
};

/**
 * Events after which a running operation must stop at once. 'lifecycle' is an auto-mute (device,
 * context, track, underruns, latency): measurements already come back aborted, but the autotune
 * does not go through measure() and would keep stepping with the anti-noise muted.
 */
const STOP_EVENTS = new Set(['panic', 'howl', 'fault', 'disarmed', 'lifecycle']);

function safeLocalStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}

export function createWizard(container: HTMLElement, controller: AppController, opts: WizardOptions = {}): Wizard {
  const storage = opts.storage === undefined ? safeLocalStorage() : opts.storage;
  const readPrefs = (): ReturnType<typeof parseWizardPrefs> => {
    try {
      return parseWizardPrefs(storage?.getItem(WIZARD_PREFS_KEY) ?? null);
    } catch {
      return parseWizardPrefs(null);
    }
  };

  let isOpen = false;
  /** The current run reached the end. */
  let everCompleted = false;
  /** Some run reached the end in this page session. */
  let completedOnce = false;
  let data: WizardData | null = null;
  let steps: StepId[] = [];
  let current: StepId = 'devices';
  let view: StepView | null = null;
  let op: { ctrl: AbortController; label: string } | null = null;
  let analyzer: WorkerAnalyzer | null = null;
  let raf = 0;
  let destroyed = false;

  // ------------------------------------------------------------------ open shell

  const title = el('h2', 'wz-title', 'Configurazione guidata');
  const closeBtn = button('Chiudi', () => close(), 'btn btn--small');
  const stepper = el('ol', 'wz-steps');
  stepper.setAttribute('aria-label', 'Passi della configurazione');
  const bannerText = el('p', 'wz-banner-text');
  const bannerBtn = el('button', 'btn btn--small');
  bannerBtn.type = 'button';
  const banner = el('div', 'wz-banner', bannerText, bannerBtn);
  banner.setAttribute('role', 'alert');
  banner.hidden = true;
  const stepTitle = el('h3', 'wz-step-title');
  stepTitle.tabIndex = -1;
  const stepIntro = el('p', 'fp-help wz-step-intro');
  const body = el('div', 'wz-body');
  const backBtn = button('Indietro', () => goBack(), 'btn');
  const nextBtn = button('Avanti', () => goNext(), 'btn btn--primary');
  const navReason = el('p', 'wz-nav-reason');
  navReason.setAttribute('aria-live', 'polite');
  const busyLine = el('p', 'wz-busy');
  busyLine.setAttribute('role', 'status');
  const nav = el('div', 'wz-nav', backBtn, el('div', 'wz-nav-mid', navReason, busyLine), nextBtn);
  const openRoot = el(
    'div',
    'wizard',
    el('div', 'wz-head', title, closeBtn),
    stepper,
    banner,
    el('section', 'wz-step', stepTitle, stepIntro, body),
    nav,
  );

  // ------------------------------------------------------------------ closed card

  const cardText = el('p', 'wz-card-text');
  const cardBtn = button('Apri la configurazione guidata', () => open(true), 'btn btn--primary');
  const closedRoot = el('div', 'wz-card', el('h2', 'wz-title', 'Configurazione guidata'), cardText, cardBtn);

  // ------------------------------------------------------------------ operations

  function startOp(label: string): Op {
    abortOp();
    const me = { ctrl: new AbortController(), label };
    op = me;
    schedule();
    return {
      signal: me.ctrl.signal,
      end() {
        if (op === me) {
          op = null;
          schedule();
        }
      },
    };
  }

  function abortOp(): void {
    const cur = op;
    if (cur === null) return;
    op = null;
    cur.ctrl.abort();
    controller.abortMeasure();
    schedule();
  }

  const env: StepEnv = {
    controller,
    get data(): WizardData {
      return data!;
    },
    analyzer() {
      if (analyzer === null) analyzer = createAnalyzer();
      return analyzer;
    },
    startOp,
    abortOp,
    get busy(): string | null {
      return op?.label ?? null;
    },
    refresh: () => schedule(),
    savePrefs() {
      if (!data) return;
      try {
        storage?.setItem(WIZARD_PREFS_KEY, serializeWizardPrefs({ layout: data.layout, geometry: data.geometry }));
      } catch {
        // Not remembered: harmless.
      }
    },
    goTo: (step) => enter(step, true),
    setMode(mode: FlowMode) {
      if (!data) return;
      data.mode = mode;
      steps = flowSteps(mode, data.hasSaved);
      const n = nextStep(steps, 'start');
      if (n) enter(n, true);
    },
    complete() {
      everCompleted = true;
      completedOnce = true;
      if (data) data.doneDeviceKey = controller.state.deviceKey;
      controller.setAncLock(null);
      close();
    },
  };

  // ------------------------------------------------------------------ navigation

  function enter(step: StepId, focus: boolean): void {
    if (!data || !steps.includes(step)) return;
    abortOp();
    view?.leave();
    current = step;
    // Entering any step locks (and mutes) the anti-noise again; only step 7's button releases it.
    controller.setAncLock(LOCK_TEXT);
    view = FACTORIES[step](env);
    body.replaceChildren(view.element);
    const info = STEP_INFO[step];
    setText(stepTitle, info.n !== null ? `Passo ${info.n} di 7 · ${info.title}` : info.title);
    setText(stepIntro, INTRO[step]);
    render();
    if (focus) {
      scrollToTop();
      stepTitle.focus({ preventScroll: true });
    }
  }

  /** Brings the wizard's top into view below the sticky status bar (when it is sticky). */
  function scrollToTop(): void {
    const bar = document.querySelector<HTMLElement>('.status-bar-host');
    const sticky = bar !== null && getComputedStyle(bar).position === 'sticky';
    const offset = sticky ? bar.getBoundingClientRect().height : 0;
    const top = openRoot.getBoundingClientRect().top + window.scrollY - offset - 12;
    window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  }

  function goNext(): void {
    if (!data || op !== null) return;
    if (!stepGate(current, controller.state, data).ok) return;
    const n = nextStep(steps, current);
    if (n) enter(n, true);
  }

  function goBack(): void {
    if (!data || op !== null) return;
    const p = prevStep(steps, current);
    if (p) enter(p, true);
  }

  function renderStepper(): void {
    const idx = steps.indexOf(current);
    const key = `${steps.join(',')}|${current}|${op !== null}`;
    if (stepper.dataset['key'] === key) return;
    stepper.dataset['key'] = key;
    stepper.replaceChildren(
      ...steps.map((id, i) => {
        const info = STEP_INFO[id];
        const label = info.n !== null ? `${info.n}. ${info.title}` : info.title;
        // Number and title apart: narrow screens show only the numbers of the other steps.
        const parts = (): HTMLElement[] =>
          info.n !== null
            ? [el('span', 'wz-steps-n', `${info.n}.`), el('span', 'wz-steps-t', info.title)]
            : [el('span', 'wz-steps-n', 'Inizio'), el('span', 'wz-steps-t', `· ${info.title}`)];
        const state = i < idx ? 'done' : i === idx ? 'current' : 'todo';
        const li = el('li', `wz-steps-item wz-steps-item--${state}`);
        if (state === 'done' && op === null) {
          const b = button('', () => enter(id, true), 'wz-steps-link');
          b.replaceChildren(...parts());
          b.setAttribute('aria-label', `Torna a: ${label}`);
          li.append(b);
        } else {
          const span = el('span', 'wz-steps-chip', ...parts());
          span.setAttribute('aria-label', label);
          li.append(span);
        }
        if (state === 'current') li.setAttribute('aria-current', 'step');
        return li;
      }),
    );
  }

  function renderBanner(s: ControllerState): void {
    const b = bannerFor(current, s);
    banner.hidden = b === null;
    if (b === null) return;
    banner.dataset['level'] = b.level;
    setText(bannerText, b.text);
    bannerBtn.hidden = b.action === null;
    setText(
      bannerBtn,
      b.action === 'resume' ? 'Riprendi (anti-rumore muto)' : b.action === 'arm' ? 'Riabilita l’uscita per le misure' : 'Vai al passo 1',
    );
    bannerBtn.onclick = () => {
      const now = controller.state;
      if (b.action === 'devices') enter('devices', true);
      else if (now.canArm) controller.arm({ muted: true });
      // Disarmed after a PANIC: un-muting only resumes the suspended audio (the output stays at zero).
      else if (b.action === 'resume') controller.setMuted(false);
    };
  }

  function render(): void {
    raf = 0;
    if (destroyed) return;
    const s = controller.state;
    if (!isOpen) {
      setText(
        cardText,
        s.ancLock !== null
          ? 'Anti-rumore bloccato finché non completi la configurazione guidata: riprendila per arrivare all’ultimo passo.'
          : everCompleted
            ? 'Configurazione completata. Riaprila per rifare misure e allineamento (per esempio se hai spostato qualcosa).'
            : 'Ti accompagna passo per passo: dispositivi, livello dell’amplificatore, posizioni, misure, allineamento e verifica.',
      );
      setText(cardBtn, data !== null && !everCompleted ? 'Riprendi la configurazione guidata' : 'Apri la configurazione guidata');
      return;
    }
    if (!data || !view) return;
    view.update(s);
    renderStepper();
    renderBanner(s);
    const busy = op !== null;
    const gate = stepGate(current, s, data);
    const n = nextStep(steps, current);
    backBtn.disabled = busy || prevStep(steps, current) === null;
    nextBtn.hidden = n === null || current === 'start';
    nextBtn.disabled = busy || !gate.ok;
    setText(navReason, busy || n === null || current === 'start' ? '' : (gate.reason ?? ''));
    setText(busyLine, busy ? `${op!.label} in corso…` : '');
  }

  function schedule(): void {
    if (raf !== 0 || destroyed) return;
    raf = typeof requestAnimationFrame === 'function' ? requestAnimationFrame(render) : (setTimeout(render, 16) as unknown as number);
  }

  // ------------------------------------------------------------------ open / close

  function open(focus = false): void {
    if (isOpen || destroyed) return;
    const hasSaved = presetsWithCalibration(controller.state.presets.list).length > 0;
    if (data === null || everCompleted) {
      const prefs = readPrefs();
      const fresh = createWizardData({ hasSaved, layout: prefs.layout, geometry: prefs.geometry });
      // Results of this page session still count (each is checked again against devices and rate).
      if (data !== null) {
        fresh.amp = data.amp;
        fresh.measure = data.measure;
        fresh.calibration = data.calibration;
        fresh.solution = data.solution;
      }
      data = fresh;
      steps = flowSteps(data.mode, hasSaved);
      current = steps[0]!;
      everCompleted = false;
    }
    isOpen = true;
    container.replaceChildren(openRoot);
    enter(current, focus);
  }

  function close(): void {
    if (!isOpen) return;
    abortOp();
    view?.leave();
    view = null;
    isOpen = false;
    // A completed setup that still satisfies every un-mute condition gives the console back.
    if (everCompleted || (completedOnce && data !== null && unmuteBlockers(controller.state, data).length === 0)) {
      controller.setAncLock(null);
    }
    // The pressed «Chiudi»/«Fine» is about to disappear: keep the keyboard focus in the card.
    const hadFocus = typeof document !== 'undefined' && openRoot.contains(document.activeElement);
    container.replaceChildren(closedRoot);
    render();
    if (hadFocus) cardBtn.focus();
  }

  /** After completion (wizard closed), a change of output, rate or interface locks the anti-noise again. */
  function guardCompleted(s: ControllerState): void {
    if (isOpen || !completedOnce || data === null || s.ancLock !== null) return;
    const why = relockReason(s, data);
    if (why !== null) controller.setAncLock(why);
  }

  const unsub = controller.subscribe((s, change: ControllerChange) => {
    if (change === 'spectrum') return;
    if (change === 'state') guardCompleted(s);
    if (!isOpen && change !== 'state') return;
    schedule();
  });
  const unsubEvents = controller.onEvent((e) => {
    if (STOP_EVENTS.has(e.type) && op !== null) abortOp();
  });

  container.replaceChildren(closedRoot);
  render();
  if (opts.autoOpen !== false) open(false);

  return {
    open: () => open(true),
    close,
    get isOpen() {
      return isOpen;
    },
    destroy() {
      if (destroyed) return;
      abortOp();
      view?.leave();
      unsub();
      unsubEvents();
      analyzer?.terminate();
      destroyed = true;
      container.replaceChildren();
    },
  };
}
