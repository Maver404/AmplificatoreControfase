/**
 * Step 5 — path measurements with exponential sweeps, per active channel: speaker → reference mic
 * (always: feedback, chain polarity, loop gain) and speaker → error mic (with the error mic).
 * Optional guided loopback on input 2 (phantom off, gain at minimum) for the electronic latency,
 * followed by "scollega il cavo" and a check that the loopback signature is gone. The loopback is
 * shown in samples with its spread (README hardware check: repeatable within 0.5 samples) and can
 * be repeated 10 dB louder to compare the two levels. Each path remembers the distances it was
 * measured with (flow.ts stalePaths).
 */
import type { ControllerState } from '../../app/controller.ts';
import { errorMicInput } from '../../app/validation.ts';
import {
  calibrationFreqs,
  createCalibration,
  defaultSweepBand,
  looksLikeLoopback,
  measurePath,
  pathResponse,
} from '../../audio/calibration.ts';
import type { CalibrationData, PathResult } from '../../audio/calibration.ts';
import { loopGainDb } from '../../dsp/alignment.ts';
import type { FreqResponse } from '../../dsp/alignment.ts';
import { channelResponse } from '../../dsp/response.ts';
import { formatHz, formatNumber } from '../units.ts';
import { button, checkbox, el, list, liveNote, note, progressBar, radioCards, setText } from './dom.ts';
import type { Op, StepEnv, StepView } from './env.ts';
import { emptySlot, errorMicOn } from './flow.ts';
import type { PathSlot } from './flow.ts';
import { activeChannels, CH_NAME, fmtMs, parseGeometry, pathSummary, positionsKey, stimulusLevels, toPathData } from './logic.ts';

type Kind = 'feedback' | 'secondary';
const REPETITIONS = 3;
/** Preview + repetitions, for the progress text. */
const SWEEPS_PER_PATH = REPETITIONS + 1;
/** The loopback sweep is this much quieter than the acoustic ones (line level straight into input 2). */
const LOOPBACK_EXTRA_DB = -10;
/** Loopback repetitions (README hardware check: 5 measurements per level). */
const LOOPBACK_REPETITIONS = 5;
/** The optional second loopback level, relative to the first (−30 → −20 dBFS). */
const LOOPBACK_SECOND_DB = 10;
/** Largest difference between the two loopback levels that still means "no OS processing". */
const LOOPBACK_LEVEL_TOL_SAMPLES = 0.5;

/** One loopback measurement: level at the output and the result, in samples. */
interface LoopRun {
  levelDbfs: number;
  delaySamples: number;
  stdSamples: number;
  count: number;
}

/** "−30 dBFS: 1234,567 campioni (25,720 ms), σ 0,012 campioni su 5 misure". */
export function loopRunText(r: LoopRun, fs: number): string {
  return (
    `${formatNumber(r.levelDbfs, 0)} dBFS: ${formatNumber(r.delaySamples, 3)} campioni (${fmtMs(r.delaySamples / fs, 3)}), ` +
    `σ ${formatNumber(r.stdSamples, 3)} campioni su ${r.count} misure`
  );
}

/** Comparison of the two loopback levels (Italian); null with fewer than two levels. */
export function loopLevelsText(runs: readonly LoopRun[]): { ok: boolean; text: string } | null {
  if (runs.length < 2) return null;
  const a = runs[0]!;
  const b = runs[runs.length - 1]!;
  const diff = Math.abs(b.delaySamples - a.delaySamples);
  const ok = diff <= LOOPBACK_LEVEL_TOL_SAMPLES;
  return {
    ok,
    text:
      `Differenza tra ${formatNumber(a.levelDbfs, 0)} e ${formatNumber(b.levelDbfs, 0)} dBFS: ${formatNumber(diff, 3)} campioni ` +
      (ok
        ? `(entro ${formatNumber(LOOPBACK_LEVEL_TOL_SAMPLES, 1)}: nessuna elaborazione del sistema).`
        : `(oltre ${formatNumber(LOOPBACK_LEVEL_TOL_SAMPLES, 1)}: il sistema operativo o il driver elaborano il segnale; disattiva effetti e miglioramenti audio).`),
  };
}

const PATH_NAME: Record<Kind, (ch: 0 | 1) => string> = {
  feedback: (ch) => `Cassa ${CH_NAME[ch]} → microfono di riferimento`,
  secondary: (ch) => `Cassa ${CH_NAME[ch]} → mic d’errore`,
};

interface Prompt {
  text: string;
  resolve: (ok: boolean) => void;
}

export function createMeasureStep(env: StepEnv): StepView {
  const { controller, data } = env;
  let running: { kind: Kind | 'loopback' | 'verify'; ch: 0 | 1 } | null = null;
  let done = 0;
  let total = 0;
  let prompt: Prompt | null = null;
  let loopNote: { level: 'ok' | 'warn' | 'error' | 'info'; text: string } | null = null;
  let loopCh: 0 | 1 = 0;
  let loopInfo: { key: string; text: string } | null = null;
  let loopRuns: LoopRun[] = [];

  // ------------------------------------------------------------------ measurement core

  const sweepLevel = (s: ControllerState): number =>
    Math.min(0, stimulusLevels(s.params.ceilingDb).sweepPeakDbfs - s.params.masterDb);

  const slotOf = (kind: Kind, ch: 0 | 1): PathSlot => (kind === 'feedback' ? data.measure.feedback[ch] : data.measure.secondary[ch]);
  const setSlot = (kind: Kind, ch: 0 | 1, slot: PathSlot): void => {
    if (kind === 'feedback') data.measure.feedback[ch] = slot;
    else data.measure.secondary[ch] = slot;
  };

  /**
   * The calibration record the measurements go into (new when the rate, the band or the reference
   * input changed: paths measured with the other microphone do not mix with these).
   */
  function ensureCalibration(s: ControllerState): CalibrationData | null {
    const fs = s.fs;
    if (fs === null) return null;
    const band = defaultSweepBand(s.params, fs);
    const g = parseGeometry(data.geometry, activeChannels(s.speakers));
    const tempC = g.ok ? g.value.tempC : 20;
    let cal = data.calibration;
    if (cal === null || cal.fs !== fs || cal.band[0] !== band[0] || cal.band[1] !== band[1] || cal.refChannel !== s.params.refChannel) {
      const loop = cal !== null && cal.fs === fs ? cal.loopbackLatencyS : null;
      cal = createCalibration(fs, band, s.speakers, tempC);
      cal.refChannel = s.params.refChannel;
      cal.loopbackLatencyS = loop;
      data.measure.feedback = [emptySlot(), emptySlot()];
      data.measure.secondary = [emptySlot(), emptySlot()];
      data.solution = null;
    }
    cal.speakers = s.speakers;
    cal.tempC = tempC;
    data.calibration = cal;
    return cal;
  }

  async function sweep(
    op: Op,
    ch: 0 | 1,
    mic: 'ref' | 'err',
    mode: 'acoustic' | 'loopback',
    extraDb = 0,
    repetitions = REPETITIONS,
  ): Promise<PathResult | null> {
    const s = controller.state;
    const cal = ensureCalibration(s);
    if (cal === null) return null;
    return measurePath(controller.measure, env.analyzer(), {
      fs: cal.fs,
      channel: ch,
      band: cal.band,
      levelDbfs: sweepLevel(s) + extraDb,
      repetitions,
      mic,
      errorInput: errorMicInput(s.params.errorChannel),
      mode,
      previewFirst: true,
      signal: op.signal,
    });
  }

  async function measureOne(op: Op, kind: Kind, ch: 0 | 1): Promise<boolean> {
    running = { kind, ch };
    setSlot(kind, ch, { state: 'running', reason: null, path: null, posKey: null });
    data.solution = null;
    env.refresh();
    // The distances the result will count for.
    const posKey = positionsKey(data.geometry, ch);
    const r = await sweep(op, ch, kind === 'feedback' ? 'ref' : 'err', 'acoustic');
    done += SWEEPS_PER_PATH;
    const cal = data.calibration;
    if (r === null || op.signal.aborted || r.aborted) {
      setSlot(kind, ch, { state: 'aborted', reason: 'Misura annullata.', path: null, posKey: null });
      return false;
    }
    if (r.ok) {
      const path = toPathData(r);
      setSlot(kind, ch, { state: 'ok', reason: null, path, posKey });
      if (cal) {
        if (kind === 'feedback') cal.feedback[ch] = path;
        else cal.secondary[ch] = path;
      }
      if (kind === 'feedback') data.measure.acceptWeakFeedback[ch] = false;
    } else {
      setSlot(kind, ch, { state: 'failed', reason: r.reason ?? 'Misura non riuscita.', path: null, posKey });
      if (cal) {
        if (kind === 'feedback') cal.feedback[ch] = null;
        else cal.secondary[ch] = null;
      }
    }
    env.refresh();
    return true;
  }

  function ask(text: string, signal: AbortSignal): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      const finish = (ok: boolean): void => {
        signal.removeEventListener('abort', onAbort);
        prompt = null;
        env.refresh();
        resolve(ok);
      };
      const onAbort = (): void => finish(false);
      signal.addEventListener('abort', onAbort, { once: true });
      prompt = { text, resolve: finish };
      env.refresh();
    });
  }

  async function runPlan(plan: { kind: Kind; ch: 0 | 1 }[], label: string): Promise<void> {
    const s = controller.state;
    if (!s.armed || s.panic || s.fs === null) return;
    const op = env.startOp(label);
    done = 0;
    total = plan.length * SWEEPS_PER_PATH;
    const separateZones = s.speakers === 'separate' && errorMicOn(s);
    let zone: 0 | 1 | null = null;
    try {
      for (const item of plan) {
        if (op.signal.aborted) break;
        if (item.kind === 'secondary' && separateZones && zone !== item.ch) {
          const ok = await ask(
            `Metti il mic d’errore nella zona della cassa ${CH_NAME[item.ch]} (dove vuoi il silenzio per quella cassa), poi continua.`,
            op.signal,
          );
          if (!ok) break;
          zone = item.ch;
        }
        if (!(await measureOne(op, item.kind, item.ch))) break;
      }
    } finally {
      running = null;
      op.end();
      loopInfo = null;
      env.refresh();
    }
  }

  function fullPlan(s: ControllerState): { kind: Kind; ch: 0 | 1 }[] {
    const plan: { kind: Kind; ch: 0 | 1 }[] = [];
    for (const ch of activeChannels(s.speakers)) {
      plan.push({ kind: 'feedback', ch });
      if (errorMicOn(s)) plan.push({ kind: 'secondary', ch });
    }
    return plan;
  }

  // ------------------------------------------------------------------ loopback

  /** 1-based number of the input the loopback borrows: the one that is not the reference microphone's. */
  const loopInput = (s: ControllerState): 1 | 2 => (s.params.refChannel === 0 ? 2 : 1);

  /** First loopback measurement, or the same 10 dB louder (`second`) with the cable still connected. */
  async function runLoopback(second = false): Promise<void> {
    const s = controller.state;
    const other: 0 | 1 = s.params.refChannel === 0 ? 1 : 0;
    const n = loopInput(s);
    if (data.measure.loopback === 'none' || data.measure.loopback === 'verified') {
      data.measure.loopbackPrevErr = s.params.errorChannel;
      data.measure.loopback = 'connect';
    }
    controller.setInput2Loopback(true);
    if (s.params.errorChannel !== other) controller.setErrorChannel(other);
    const now = controller.state;
    if (!now.armed || now.panic || now.opening) {
      loopNote = { level: 'warn', text: `Ingresso ${n} riconfigurato: riabilita l’uscita (in alto) e premi di nuovo «Misura il loopback».` };
      env.refresh();
      return;
    }
    const op = env.startOp('Misura del loopback');
    running = { kind: 'loopback', ch: loopCh };
    done = 0;
    total = LOOPBACK_REPETITIONS + 1;
    loopNote = null;
    if (!second) loopRuns = [];
    env.refresh();
    try {
      const extra = LOOPBACK_EXTRA_DB + (second ? LOOPBACK_SECOND_DB : 0);
      const levelDbfs = sweepLevel(controller.state) + controller.state.params.masterDb + extra;
      const r = await sweep(op, loopCh, 'err', 'loopback', extra, LOOPBACK_REPETITIONS);
      if (r === null || op.signal.aborted || r.aborted) return;
      const cal = data.calibration;
      if (!r.ok) {
        loopNote = { level: 'error', text: r.reason ?? 'Misura del loopback non riuscita.' };
      } else if (!looksLikeLoopback(r) || cal === null) {
        loopNote = {
          level: 'error',
          text:
            'La misura non ha la firma di un collegamento diretto (risposta non piatta o ritardo non puro): controlla che il cavo vada dall’uscita ' +
            `${CH_NAME[loopCh]} all’ingresso ${n} e che la phantom sia spenta.`,
        };
      } else {
        const run: LoopRun = { levelDbfs, delaySamples: r.delaySamples, stdSamples: r.stdSamples, count: LOOPBACK_REPETITIONS };
        loopRuns = second ? [...loopRuns.slice(0, 1), run] : [run];
        // The latency used by the alignment stays the first (quieter, safer) level's.
        if (!second) cal.loopbackLatencyS = r.delaySamples / cal.fs;
        data.measure.loopback = 'measured';
        const cmp = loopLevelsText(loopRuns);
        loopNote = {
          level: cmp && !cmp.ok ? 'warn' : 'ok',
          text:
            `Latenza elettronica misurata: ${loopRuns.map((x) => loopRunText(x, cal.fs)).join(' · ')}.` +
            (cmp ? ` ${cmp.text}` : '') +
            ' Ora scollega il cavo (o ripeti la misura 10 dB più forte per controllare che il sistema non elabori il segnale).',
        };
      }
    } finally {
      running = null;
      op.end();
      env.refresh();
    }
  }

  async function verifyUnplugged(): Promise<void> {
    const s = controller.state;
    if (!s.armed || s.panic) return;
    const n = loopInput(s);
    const op = env.startOp('Verifica del cavo scollegato');
    running = { kind: 'verify', ch: loopCh };
    done = 0;
    total = 2;
    env.refresh();
    try {
      const r = await sweep(op, loopCh, 'err', 'loopback', LOOPBACK_EXTRA_DB, 1);
      if (r === null || op.signal.aborted || r.aborted) return;
      const cal = data.calibration;
      const lat = cal !== null && cal.loopbackLatencyS !== null ? { seconds: cal.loopbackLatencyS, fs: cal.fs } : undefined;
      if (r.ok && looksLikeLoopback(r, lat)) {
        loopNote = { level: 'error', text: `Il cavo di loopback sembra ancora collegato all’ingresso ${n}: scollegalo e ripeti la verifica.` };
        return;
      }
      // The borrowed input back to its previous use first, then normal validation of a silent input.
      const prev = data.measure.loopbackPrevErr;
      if (controller.state.params.errorChannel !== prev) controller.setErrorChannel(prev);
      controller.setInput2Loopback(false);
      data.measure.loopback = 'verified';
      loopNote = {
        level: 'ok',
        text:
          prev >= 0
            ? 'Cavo scollegato. Se il mic d’errore usa la phantom, riaccendila ora.'
            : `Cavo scollegato: l’ingresso ${n} è libero.`,
      };
    } finally {
      running = null;
      op.end();
      env.refresh();
    }
  }

  // ------------------------------------------------------------------ view

  const measureAllBtn = button('Misura tutti i percorsi', () => void runPlan(fullPlan(controller.state), 'Misura dei percorsi'), 'btn btn--primary');
  const stopBtn = button('Interrompi', () => env.abortOp(), 'btn btn--danger');
  const progress = progressBar('Avanzamento delle misure');
  const promptText = el('p', 'wz-prompt-text');
  const promptBox = el(
    'div',
    'wz-prompt',
    promptText,
    el('div', 'wz-actions', button('Fatto, continua', () => prompt?.resolve(true), 'btn btn--primary'), button('Annulla', () => prompt?.resolve(false))),
  );
  promptBox.hidden = true;
  const table = el('div', 'wz-paths');
  const loopGain = liveNote();
  const masterNote = liveNote();

  // Loopback block: its texts name the borrowed input (input 2, or 1 when the reference is on 2).
  const loopTexts = (n: 1 | 2): string[] => [
    `L’alimentazione phantom (48 V) dell’ingresso ${n} è SPENTA.`,
    `Il guadagno dell’ingresso ${n} è al minimo.`,
    `Il mic d’errore (se c’è) è scollegato dall’ingresso ${n}.`,
    `Il cavo va dall’uscita scelta qui sotto all’ingresso ${n} (mai all’ingresso del microfono di riferimento).`,
  ];
  const loopCheckText = loopTexts(2).map((t) => el('span', null, t));
  const loopChecks = loopCheckText.map((t) => checkbox(t, () => env.refresh()));
  const loopChChoice = radioCards<'0' | '1'>(
    'Uscita collegata al cavo',
    [
      { value: '0', title: 'Uscita L' },
      { value: '1', title: 'Uscita R' },
    ],
    (v) => {
      loopCh = v === '1' ? 1 : 0;
    },
  );
  loopChChoice.set('0');
  const loopBtn = button('Ho collegato il cavo: misura il loopback', () => void runLoopback(), 'btn');
  const loopAgainBtn = button('Ripeti 10 dB più forte (controllo di linearità)', () => void runLoopback(true), 'btn btn--small');
  const unpluggedText = el('span', null, 'Ho scollegato il cavo di loopback dall’ingresso 2.');
  const unplugged = checkbox(unpluggedText, () => env.refresh());
  const verifyBtn = button('Verifica che il cavo sia scollegato', () => void verifyUnplugged(), 'btn btn--primary');
  const loopStatus = liveNote();
  const loopIntroText = (n: 1 | 2): string =>
    `Un cavo dall’uscita all’ingresso ${n} misura la latenza elettronica esatta (utile senza mic d’errore). La cassa di quel canale ` +
    'suonerà anche lei lo sweep. Dopo la misura il cavo va scollegato: l’app verifica che non sia rimasto collegato prima di continuare.';
  const loopIntro = el('p', 'fp-help', loopIntroText(2));
  const loopSetup = el('div', 'wz-checklist', ...loopChecks.map((c) => c.root), loopChChoice.root, el('div', 'wz-actions', loopBtn));
  const loopUnplug = el('div', 'wz-checklist', el('div', 'wz-actions', loopAgainBtn), unplugged.root, el('div', 'wz-actions', verifyBtn));
  const loopDetails = el(
    'details',
    'advanced wz-loopback',
    el('summary', null, 'Latenza esatta con un cavo di loopback (facoltativo)'),
    el(
      'div',
      'advanced-body',
      loopIntro,
      loopSetup,
      loopUnplug,
      loopStatus.root,
    ),
  );

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'L’app fa suonare brevi sweep (glissandi dal grave all’acuto) su ogni cassa e li registra: misura così i ritardi reali, la polarità ' +
        'della catena e quanto la cassa rientra nel microfono di riferimento (rischio di innesco). L’anti-rumore resta muto.',
    ),
    list(
      [
        ['Tieni la mano sul volume dell’amplificatore: ogni sweep parte con un’anteprima 20 dB più bassa.'],
        ['Durante le misure resta in silenzio e ferma; il rumore di fondo va bene, i colpi improvvisi no.'],
        ['Ogni percorso richiede circa 10 secondi. ', el('kbd', null, 'Esc'), ' o «Interrompi» fermano tutto.'],
      ],
      false,
    ),
    masterNote.root,
    el('div', 'wz-actions', measureAllBtn, stopBtn),
    progress.root,
    promptBox,
    table,
    loopGain.root,
    loopDetails,
    note('info', 'Una misura fallita mostra il motivo: correggi (collegamenti, guadagno, volume) e premi «Ripeti» su quella riga.'),
  );

  function renderTable(s: ControllerState): void {
    const chans = activeChannels(s.speakers);
    const kinds: Kind[] = errorMicOn(s) ? ['feedback', 'secondary'] : ['feedback'];
    const fs = s.fs ?? data.calibration?.fs ?? 48000;
    const busy = env.busy !== null;
    const rows: HTMLElement[] = [];
    const keyParts: string[] = [String(busy)];
    for (const ch of chans) {
      for (const kind of kinds) {
        const slot = slotOf(kind, ch);
        keyParts.push(`${kind}${ch}:${slot.state}:${slot.reason ?? ''}:${data.measure.acceptWeakFeedback[ch]}:${slot.path?.delaySamples ?? ''}`);
      }
    }
    const key = keyParts.join('|');
    if (table.dataset['key'] === key) return;
    table.dataset['key'] = key;
    for (const ch of chans) {
      for (const kind of kinds) {
        const slot = slotOf(kind, ch);
        const status = el('span', `wz-badge wz-badge--${slot.state}`, {
          idle: 'da misurare',
          running: 'in corso…',
          ok: 'misurato',
          failed: 'non riuscito',
          aborted: 'annullato',
        }[slot.state]);
        const detail = el('p', 'fp-help');
        if (slot.state === 'ok' && slot.path) {
          const sm = pathSummary(slot.path, fs);
          detail.textContent =
            `Ritardo ${fmtMs(sm.delayMs / 1000, 2)} · qualità ${formatNumber(sm.qualityDb, 0)} dB` +
            (sm.peakDb !== null ? ` · livello ${formatNumber(sm.peakDb, 1)} dB` : '') +
            (kind === 'feedback' ? ` · polarità della catena ${sm.inverted ? 'invertita' : 'normale'}` : '');
        } else if (slot.reason) detail.textContent = slot.reason;
        const retry = button(slot.state === 'idle' ? 'Misura' : 'Ripeti', () => void runPlan([{ kind, ch }], PATH_NAME[kind](ch)), 'btn btn--small');
        retry.disabled = busy || !s.armed || s.panic;
        const row = el('div', 'wz-path-row', el('div', 'wz-path-head', el('strong', null, PATH_NAME[kind](ch)), status, retry), detail);
        if (kind === 'feedback' && slot.state === 'failed') {
          const acc = checkbox(
            'Continua comunque: la cassa non si sente al microfono di riferimento (retroazione trascurabile, ma il guadagno d’anello non è verificato).',
            (on) => {
              data.measure.acceptWeakFeedback[ch] = on;
              env.refresh();
            },
          );
          acc.input.checked = data.measure.acceptWeakFeedback[ch];
          acc.input.disabled = busy;
          row.append(acc.root);
        }
        rows.push(row);
      }
    }
    table.replaceChildren(...rows);
  }

  function renderLoopGain(s: ControllerState): void {
    const cal = data.calibration;
    const fs = s.fs;
    if (cal === null || fs === null || cal.fs !== fs || (cal.feedback[0] === null && cal.feedback[1] === null)) {
      loopGain.clear();
      return;
    }
    const p = s.params;
    const key = `${cal.feedback[0]?.delaySamples}|${cal.feedback[1]?.delaySamples}|${JSON.stringify(p.channels)}|${p.masterDb}|${JSON.stringify(p.filters)}|${p.phaseMode}`;
    if (loopInfo === null || loopInfo.key !== key) {
      const freqs = calibrationFreqs(fs, cal.band);
      const fb: [FreqResponse | null, FreqResponse | null] = [
        cal.feedback[0] ? pathResponse(cal.feedback[0], fs, freqs) : null,
        cal.feedback[1] ? pathResponse(cal.feedback[1], fs, freqs) : null,
      ];
      const lg = loopGainDb([(f) => channelResponse(p, fs, f, 0), (f) => channelResponse(p, fs, f, 1)], fb, cal.band);
      loopInfo = {
        key,
        text: Number.isFinite(lg.maxDb)
          ? `Guadagno d’anello cassa → microfono di riferimento con le impostazioni attuali: ${formatNumber(lg.maxDb, 1)} dB a ${formatHz(lg.atHz)} ` +
            '(deve restare sotto −20 dB dopo la neutralizzazione del feedback; verrà ricontrollato con i valori calibrati al passo 6).'
          : '',
      };
    }
    if (loopInfo.text) loopGain.set('info', loopInfo.text);
    else loopGain.clear();
  }

  return {
    element,
    update(s: ControllerState) {
      const busy = env.busy !== null;
      const canMeasure = s.armed && !s.panic && s.fs !== null && !busy;
      measureAllBtn.disabled = !canMeasure;
      stopBtn.disabled = running === null && prompt === null;
      const frac = total > 0 ? Math.min(1, (done + (s.measureProgress ?? 0)) / total) : null;
      const what =
        running === null
          ? ''
          : running.kind === 'loopback'
            ? 'Loopback'
            : running.kind === 'verify'
              ? 'Verifica del cavo'
              : PATH_NAME[running.kind](running.ch);
      progress.set(running !== null, frac, what ? `${what}…` : '');
      promptBox.hidden = prompt === null;
      if (prompt) setText(promptText, prompt.text);
      if (s.params.masterDb < -10) {
        masterNote.set(
          'warn',
          `Il volume master è a ${formatNumber(s.params.masterDb, 1)} dB: le misure saranno deboli. Riportalo a 0 dB (passo 2) se l’amplificatore è stato regolato lì.`,
        );
      } else masterNote.clear();
      renderTable(s);
      renderLoopGain(s);

      const stage = data.measure.loopback;
      const n = loopInput(s);
      setText(loopIntro, loopIntroText(n));
      loopTexts(n).forEach((t, i) => setText(loopCheckText[i]!, t));
      setText(unpluggedText, `Ho scollegato il cavo di loopback dall’ingresso ${n}.`);
      loopSetup.hidden = stage === 'measured';
      loopUnplug.hidden = stage === 'none' || stage === 'verified';
      if (stage === 'connect' || stage === 'measured') loopDetails.open = true;
      loopBtn.disabled = !canMeasure || !loopChecks.every((c) => c.input.checked);
      for (const c of loopChecks) c.input.disabled = busy;
      loopChChoice.disable(busy || stage === 'measured');
      verifyBtn.disabled = !canMeasure || !unplugged.input.checked;
      loopAgainBtn.hidden = stage !== 'measured';
      loopAgainBtn.disabled = !canMeasure || unplugged.input.checked;
      if (loopNote) loopStatus.set(loopNote.level, loopNote.text);
      else if (stage === 'verified' && data.calibration?.loopbackLatencyS != null) {
        const cal = data.calibration;
        const lat = cal.loopbackLatencyS!;
        loopStatus.set(
          'ok',
          `Latenza elettronica: ${formatNumber(lat * cal.fs, 3)} campioni (${fmtMs(lat, 3)}), cavo scollegato e verificato.`,
        );
      } else loopStatus.clear();
    },
    leave() {
      if (running !== null || prompt !== null) env.abortOp();
    },
  };
}
