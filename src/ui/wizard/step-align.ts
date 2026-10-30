/**
 * Step 6 — alignment. With the error mic: ambient noise reference → error (anti-noise muted),
 * solveFromCalibration, feedback FIRs to the engine, optional autotune. Without: two-position
 * calibration with the single mic (in the zone: speaker → zone path and noise level; back upstream:
 * noise level) or geometry only, then solveFromGeometry. Shows the amplifier advice, the predicted
 * attenuation and the loop gain. Two separate zones are solved one zone at a time; their loop gain
 * is then checked with both speakers on (they feed back into the reference mic together).
 */
import type { ControllerState } from '../../app/controller.ts';
import { runAutotune } from '../../audio/autotune.ts';
import type { AutotuneResult } from '../../audio/autotune.ts';
import {
  bandRmsDbfs,
  CalibrationError,
  capLoopGain,
  LOOP_GAIN_LIMIT_DB,
  measurePath,
  measurePrimary,
  noiseBandwidthHz,
  solveFromCalibration,
  solveFromGeometry,
} from '../../audio/calibration.ts';
import type { CalibrationData, PrimaryData, SolveOutcome } from '../../audio/calibration.ts';
import type { EngineParams } from '../../dsp/params.ts';
import { formatNumber } from '../units.ts';
import { button, checkbox, el, liveNote, note, progressBar, radioCards, setListItems, setText } from './dom.ts';
import type { Op, StepEnv, StepView } from './env.ts';
import { errorMicOn, LOCK_TEXT, unmuteBlockers } from './flow.ts';
import {
  activeChannels,
  CH_NAME,
  estimatedLatencyS,
  fmtDb,
  gainLimitedAttenuationDb,
  geometryKey,
  loopGuard,
  mergeZoneSolutions,
  mergedZonesLoop,
  parseGeometry,
  powerMeanDb,
  rt60ForAutotune,
  stimulusLevels,
  toPathData,
  usefulOutcome,
} from './logic.ts';
import { autotuneDriver, waitSoftStart } from './ops.ts';

type Method = 'error-mic' | 'two-position' | 'geometry';
/** Zone index: the channel of a separate zone, or 0 for the one shared zone. */
type Zone = 0 | 1;

const CAPTURE_CHOICES = [5, 10, 20] as const;

/**
 * Loop-gain part of the solve outcome: raw loop of the applied params, the loop left after the
 * (automatic) feedback neutralisation and, when the solver lowered the gains for stability, by how much.
 */
export function loopOutcomeText(raw: number | null, after: number | null, limitedDb: number): string {
  let t = raw !== null ? ` Guadagno d’anello (senza neutralizzazione) ${fmtDb(raw)}.` : '';
  if (after !== null) {
    t +=
      ` Guadagno d’anello dopo la neutralizzazione ${fmtDb(after)}` +
      (limitedDb > 0 ? `: guadagni ridotti di ${fmtDb(limitedDb)} per la stabilità` : '') +
      '.';
  } else if (limitedDb > 0) t += ` Guadagni ridotti di ${fmtDb(limitedDb)} per la stabilità.`;
  return t;
}

const zonesOf = (s: ControllerState): Zone[] => (s.speakers === 'separate' ? [0, 1] : [0]);
const zoneName = (s: ControllerState, z: Zone): string => (s.speakers === 'separate' ? `zona della cassa ${CH_NAME[z]}` : 'zona');

export function createAlignStep(env: StepEnv): StepView {
  const { controller, data } = env;
  let method: Method = 'two-position';
  let seconds: number = 10;
  const primaries: [PrimaryData | null, PrimaryData | null] = [null, null];
  const zoneDb: [number | null, number | null] = [null, null];
  let refDb: number | null = null;
  let running: string | null = null;
  let opError: string | null = null;
  let outcomeText: { level: 'ok' | 'warn' | 'error'; text: string } | null = null;
  let advice: string[] = [];
  let tune: { measured: number; result: AutotuneResult | null; error: string | null } = { measured: 0, result: null, error: null };
  let tuneZone: Zone = 0;
  /** "The mic is in the zone" confirmations of the two-position method. */
  const zoneConfirmed: [boolean, boolean] = [false, false];

  const fsNow = (): number | null => controller.state.fs;

  /** Same record as step 5 (null if the rate changed since). */
  function calibration(): CalibrationData | null {
    const cal = data.calibration;
    const fs = fsNow();
    return cal !== null && fs !== null && cal.fs === fs ? cal : null;
  }

  function begin(label: string): Op {
    opError = null;
    running = label;
    const op = env.startOp(label);
    env.refresh();
    return op;
  }

  function finish(op: Op): void {
    running = null;
    op.end();
    env.refresh();
  }

  // ------------------------------------------------------------------ error mic

  async function capturePrimary(z: Zone): Promise<void> {
    const cal = calibration();
    if (cal === null) return;
    const op = begin('Misura del rumore ambiente');
    try {
      primaries[z] = await measurePrimary(controller.capture, env.analyzer(), { fs: cal.fs, seconds, band: cal.band, signal: op.signal });
      data.solution = null;
    } catch (e) {
      if (!op.signal.aborted && !(e instanceof CalibrationError && e.aborted)) {
        opError = e instanceof Error ? e.message : 'Misura del rumore non riuscita.';
      }
    } finally {
      finish(op);
    }
  }

  // ------------------------------------------------------------------ two-position

  async function measureInZone(z: Zone): Promise<void> {
    const s = controller.state;
    const cal = calibration();
    if (cal === null) return;
    const chans = s.speakers === 'separate' ? [z] : activeChannels(s.speakers);
    const op = begin(`Misura nella ${zoneName(s, z)}`);
    try {
      for (const ch of chans) {
        const r = await measurePath(controller.measure, env.analyzer(), {
          fs: cal.fs,
          channel: ch,
          band: cal.band,
          levelDbfs: Math.min(0, stimulusLevels(s.params.ceilingDb).sweepPeakDbfs - s.params.masterDb),
          repetitions: 3,
          mic: 'ref',
          mode: 'acoustic',
          previewFirst: true,
          signal: op.signal,
        });
        if (op.signal.aborted || r.aborted) return;
        if (!r.ok) {
          opError = `Cassa ${CH_NAME[ch]} → zona: ${r.reason ?? 'misura non riuscita.'}`;
          cal.secondary[ch] = null;
          return;
        }
        cal.secondary[ch] = toPathData(r);
      }
      const rec = await controller.capture(seconds);
      if (op.signal.aborted) return;
      zoneDb[z] = bandRmsDbfs(rec.ref, cal.fs, cal.band);
      // The mic is in the zone now: the "back upstream" confirmation must be given again.
      refBox.input.checked = false;
      data.solution = null;
    } catch (e) {
      if (!op.signal.aborted) opError = e instanceof Error ? e.message : 'Misura non riuscita.';
    } finally {
      finish(op);
    }
  }

  async function measureAtReference(): Promise<void> {
    const cal = calibration();
    if (cal === null) return;
    const op = begin('Livello del rumore al riferimento');
    try {
      const rec = await controller.capture(seconds);
      if (op.signal.aborted) return;
      refDb = bandRmsDbfs(rec.ref, cal.fs, cal.band);
      // The mic is upstream now: "the mic is in the zone" must be confirmed again before a new zone measurement.
      zoneConfirmed[0] = zoneConfirmed[1] = false;
      twoZones.dataset['key'] = '';
      data.solution = null;
    } catch (e) {
      if (!op.signal.aborted) opError = e instanceof Error ? e.message : 'Misura non riuscita.';
    } finally {
      finish(op);
    }
  }

  // ------------------------------------------------------------------ solve

  function solveZones(cal: CalibrationData, p: EngineParams, fs: number, m: Method): { outcomes: (SolveOutcome | null)[]; params: EngineParams } {
    const s = controller.state;
    const g = parseGeometry(data.geometry, activeChannels(s.speakers));
    cal.geometry = g.ok ? g.value : null;
    if (g.ok) cal.tempC = g.value.tempC;
    const latencyS = cal.loopbackLatencyS ?? estimatedLatencyS(s.stats) ?? 0;
    const zones = zonesOf(s);
    // The loop-gain check uses the neutraliser length the controller loads.
    const opts = { feedbackTaps: s.feedbackTaps };
    const solveOne = (c: CalibrationData): SolveOutcome =>
      m === 'error-mic' ? solveFromCalibration(c, p, fs, opts) : solveFromGeometry(c, p, fs, latencyS, opts);
    if (zones.length === 1) {
      const c = { ...cal };
      if (m === 'error-mic') cal.primary = c.primary = primaries[0];
      if (m === 'two-position') cal.levels = c.levels = { zoneDb: zoneDb[0], refDb };
      if (m === 'geometry') c.levels = { zoneDb: null, refDb: null };
      const o = solveOne(c);
      return { outcomes: [o], params: o.params };
    }
    const outs: SolveOutcome[] = zones.map((z) => {
      const c: CalibrationData = {
        ...cal,
        secondary: z === 0 ? [cal.secondary[0], null] : [null, cal.secondary[1]],
        primary: m === 'error-mic' ? primaries[z] : null,
        levels: m === 'two-position' ? { zoneDb: zoneDb[z], refDb } : { zoneDb: null, refDb: null },
      };
      return solveOne(c);
    });
    // The record keeps the last zone's primary and the mean zone level (one field each).
    if (m === 'error-mic') cal.primary = primaries[1] ?? primaries[0];
    if (m === 'two-position') cal.levels = { zoneDb: powerMeanDb(zoneDb), refDb };
    // A zone without a useful solution keeps its speaker off.
    const solved = (z: Zone): EngineParams | null => (usefulOutcome(outs[z]) ? outs[z].params : null);
    return { outcomes: outs, params: mergeZoneSolutions(p, [solved(0), solved(1)]) };
  }

  function solveAndApply(): void {
    const s = controller.state;
    const cal = calibration();
    const fs = s.fs;
    if (cal === null || fs === null) return;
    const m: Method = errorMicOn(s) ? 'error-mic' : method;
    const solved = solveZones(cal, s.params, fs, m);
    const outcomes = solved.outcomes;
    let params = solved.params;
    advice = outcomes.flatMap((o, i) => (o ? (outcomes.length > 1 ? o.advice.map((a) => `Zona ${CH_NAME[i]}: ${a}`) : o.advice) : []));
    let useful = outcomes.filter(usefulOutcome);
    let predicted = useful.length > 0 ? Math.min(...useful.map((o) => o.result.predictedAttenuationDb)) : null;
    const maxOf = (v: (number | null | undefined)[]): number | null => {
      const xs = v.filter((x): x is number => x !== null && x !== undefined && Number.isFinite(x));
      return xs.length > 0 ? Math.max(...xs) : null;
    };
    let loop = maxOf(outcomes.map((o) => o?.loopGainDb));
    let loopAfter = maxOf(outcomes.map((o) => o?.loopGainAfterDb));
    let limited = maxOf(outcomes.map((o) => o?.loopLimitedDb)) ?? 0;
    if (outcomes.length > 1) {
      if (useful.length === 1) {
        const off = outcomes.findIndex((o) => !usefulOutcome(o));
        if (off >= 0) advice.push(`Zona ${CH_NAME[off]}: nessuna attenuazione utile, la cassa ${CH_NAME[off]} resta spenta.`);
      }
      // Each zone was checked with one speaker: the merged parameters play both, so the loop and the
      // stability limit are checked on them.
      const merged = mergedZonesLoop(cal, params, fs, s.feedbackTaps, advice);
      advice = merged.advice;
      params = merged.params;
      loop = merged.loopGainDb ?? loop;
      loopAfter = merged.loopGainAfterDb ?? loopAfter;
      if (merged.limitedDb > 0) {
        limited = Math.max(limited, merged.limitedDb);
        if (predicted !== null) predicted = Math.min(predicted, gainLimitedAttenuationDb(merged.limitedDb));
        if (predicted !== null && predicted < 1) useful = [];
      }
    }
    const causal = outcomes.every((o) => o === null || o.result.causal);
    // Geometry-based solutions only hold for the distances, angle and temperature they used.
    const gKey = m === 'error-mic' ? null : geometryKey(data.geometry, activeChannels(s.speakers));
    if (useful.length === 0) {
      data.solution = { method: m, applied: false, predictedDb: predicted, loopGainDb: loop, causal, advice, geometryKey: gKey };
      outcomeText = { level: 'error', text: 'Nessuna attenuazione ottenibile con queste misure: i parametri non sono stati cambiati. Leggi i suggerimenti qui sotto.' };
      env.refresh();
      return;
    }
    // Plain-JSON copy into the controller (feedback FIRs follow), then the calibrated values.
    controller.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    controller.applyParams(params, { calibrated: true });
    data.solution = { method: m, applied: true, predictedDb: predicted, loopGainDb: loop, causal, advice, geometryKey: gKey };
    const loopWarn = limited > 0 || (loopAfter !== null && loopAfter > LOOP_GAIN_LIMIT_DB);
    outcomeText = {
      level: predicted !== null && predicted >= 6 && causal && !loopWarn ? 'ok' : 'warn',
      text:
        `Valori applicati (l’anti-rumore resta muto fino al passo 7). Attenuazione prevista ${predicted !== null ? fmtDb(predicted) : '—'}` +
        `${m === 'error-mic' ? '' : ' (stima: senza mic d’errore non è misurabile)'}.${loopOutcomeText(loop, loopAfter, limited)}` +
        (causal ? '' : ' Allineamento non causale per il rumore a banda larga: vedi i suggerimenti.'),
    };
    tune = { measured: 0, result: null, error: null };
    env.refresh();
  }

  // ------------------------------------------------------------------ autotune

  async function autotune(): Promise<void> {
    const s0 = controller.state;
    const cal = calibration();
    const fs = s0.fs;
    if (cal === null || fs === null) return;
    const blockers = unmuteBlockers(s0, data);
    if (blockers.length > 0) {
      tune = { measured: 0, result: null, error: blockers[0] ?? null };
      env.refresh();
      return;
    }
    const op = begin('Autotune');
    tune = { measured: 0, result: null, error: null };
    const stop = controller.onEvent((e) => {
      if (['anc-worse', 'howl', 'panic', 'fault', 'lifecycle', 'disarmed'].includes(e.type)) env.abortOp();
    });
    try {
      controller.setAncLock(null);
      if (!controller.setMuted(false)) throw new Error('Impossibile attivare l’anti-rumore: controlla la verifica dei dispositivi.');
      await waitSoftStart(controller, op.signal);
      const channels = s0.speakers === 'separate' ? [tuneZone] : activeChannels(s0.speakers);
      const driver = autotuneDriver(controller, {
        fs,
        band: cal.band,
        signal: op.signal,
        onMeasure: () => {
          tune.measured++;
          env.refresh();
        },
      });
      // Windows sized on the noise actually heard in the zone (its statistical bandwidth), not the sweep band.
      const zone = s0.speakers === 'separate' ? tuneZone : 0;
      const effectiveBandHz = noiseBandwidthHz({ ...cal, primary: primaries[zone] ?? cal.primary }) ?? undefined;
      // The solve's stability cap lowered the gains below the cancelling ones: the error mic would lead
      // the search back above it. No probe may raise the neutralised loop past the margin.
      const taps = s0.feedbackTaps;
      const start = controller.currentParams;
      const r = await runAutotune(driver, start, {
        fs,
        channels,
        rt60S: rt60ForAutotune(cal),
        band: cal.band,
        ...(effectiveBandHz !== undefined ? { effectiveBandHz } : {}),
        allowed: loopGuard(cal, start, fs, taps),
      });
      tune.result = r;
      if (!r.aborted && r.endDb < r.startDb) controller.applyParams(capLoopGain(cal, r.params, fs, taps).params, { calibrated: true });
    } catch (e) {
      if (!op.signal.aborted) tune.error = e instanceof Error ? e.message : 'Autotune non riuscito.';
    } finally {
      stop();
      controller.setMuted(true);
      controller.setAncLock(LOCK_TEXT);
      finish(op);
    }
  }

  // ------------------------------------------------------------------ view

  const methodChoice = radioCards<'two-position' | 'geometry'>(
    'Metodo (senza mic d’errore)',
    [
      { value: 'two-position', title: 'Due posizioni (consigliato)', text: 'Sposti il microfono di riferimento nella zona per una misura, poi lo rimetti a monte.' },
      { value: 'geometry', title: 'Solo geometria', text: 'Usa distanze e latenza: più rapido, meno preciso.' },
    ],
    (v) => {
      method = v;
      data.solution = null;
      env.refresh();
    },
  );
  methodChoice.set('two-position');

  const secChoice = radioCards<'5' | '10' | '20'>(
    'Durata di ogni registrazione del rumore',
    CAPTURE_CHOICES.map((v) => ({ value: String(v) as '5' | '10' | '20', title: `${v} s` })),
    (v) => {
      seconds = Number(v);
    },
  );
  secChoice.set('10');

  // Error-mic zones
  const errZones = el('div', 'wz-zones');
  // Two-position zones + reference
  const twoZones = el('div', 'wz-zones');
  const refBox = checkbox('Ho rimesso il microfono di riferimento nella sua posizione a monte, verso la sorgente.', () => env.refresh());
  const refBtn = button('Misura il livello al riferimento', () => void measureAtReference(), 'btn');
  const refStatus = el('p', 'fp-help');
  const twoRef = el('div', 'wz-zone', el('h5', null, 'Microfono di nuovo a monte'), refBox.root, el('div', 'wz-actions', refBtn), refStatus);

  const errorBlock = el(
    'div',
    'wz-block',
    note(
      'info',
      'Con il mic d’errore nella zona l’app registra il rumore vero (anti-rumore muto) e calcola ritardo, guadagno e polarità che lo cancellano meglio. ' +
        'Il rumore da cancellare deve essere presente durante la registrazione.',
    ),
    errZones,
  );
  const twoBlock = el(
    'div',
    'wz-block',
    note(
      'warn',
      'Il rumore da cancellare deve essere presente durante entrambe le registrazioni. Senza mic d’errore il risultato non è misurabile: verificherai a orecchio.',
    ),
    twoZones,
    twoRef,
  );

  const solveBtn = button('Calcola e applica', () => solveAndApply(), 'btn btn--primary');
  const stopBtn = button('Interrompi', () => env.abortOp(), 'btn btn--danger');
  const progress = progressBar('Operazione in corso');
  const errNote = liveNote();
  const outcome = liveNote();
  const adviceList = el('ul', 'wz-list wz-advice');
  const paramsLine = el('p', 'fp-help');

  const tuneZoneChoice = radioCards<'0' | '1'>(
    'Zona in cui si trova ora il mic d’errore',
    [
      { value: '0', title: 'Zona L' },
      { value: '1', title: 'Zona R' },
    ],
    (v) => {
      tuneZone = v === '1' ? 1 : 0;
    },
  );
  tuneZoneChoice.set('0');
  const tuneBtn = button('Avvia l’autotune', () => void autotune(), 'btn');
  const tuneNote = liveNote();
  const tuneBlock = el(
    'details',
    'advanced wz-autotune',
    el('summary', null, 'Regolazione automatica fine (autotune, facoltativa)'),
    el(
      'div',
      'advanced-body',
      note(
        'warn',
        'L’autotune ATTIVA l’anti-rumore (con il soft-start di 1,5 s) e prova piccole variazioni di ritardo e guadagno misurando il mic d’errore. ' +
          'Resta con la mano sul volume: si interrompe e ripristina i valori da solo se il rumore aumenta. Alla fine l’anti-rumore torna muto.',
      ),
      tuneZoneChoice.root,
      el('div', 'wz-actions', tuneBtn),
      tuneNote.root,
    ),
  );

  const element = el(
    'div',
    'wz-step-body',
    el('p', 'wz-lead', 'Ora l’app calcola ritardo, guadagno e polarità dell’anti-rumore per ogni cassa.'),
    methodChoice.root,
    secChoice.root,
    errorBlock,
    twoBlock,
    el('div', 'wz-actions', solveBtn, stopBtn),
    progress.root,
    errNote.root,
    outcome.root,
    adviceList,
    paramsLine,
    tuneBlock,
  );

  function zoneCards(host: HTMLElement, s: ControllerState, kind: 'err' | 'two'): void {
    const zones = zonesOf(s);
    const busy = env.busy !== null;
    const ready = s.armed && !s.panic && calibration() !== null && !busy;
    const key = `${kind}|${zones.join(',')}|${s.speakers}|${busy}|${ready}|${primaries.map((p) => (p ? 1 : 0)).join('')}|${zoneDb.join(',')}`;
    if (host.dataset['key'] === key) return;
    host.dataset['key'] = key;
    host.replaceChildren(
      ...zones.map((z) => {
        const name = zoneName(s, z);
        if (kind === 'err') {
          const p = primaries[z];
          let status = 'Rumore non ancora registrato.';
          if (p) {
            let c = 0;
            for (const v of p.coherence) c += v;
            status = `Registrato: coerenza media ${formatNumber(p.coherence.length ? c / p.coherence.length : 0, 2)}` + (p.tonalF0 !== null ? ` · tono a ${formatNumber(p.tonalF0, 2)} Hz` : '') + '.';
          }
          const btn = button(p ? 'Registra di nuovo' : 'Registra il rumore ambiente', () => void capturePrimary(z), p ? 'btn' : 'btn btn--primary');
          btn.disabled = !ready;
          return el(
            'div',
            'wz-zone',
            el('h5', null, s.speakers === 'separate' ? `Mic d’errore nella ${name}` : 'Rumore nella zona'),
            s.speakers === 'separate' ? el('p', 'fp-help', `Metti il mic d’errore nella ${name} prima di registrare.`) : null,
            el('div', 'wz-actions', btn),
            el('p', 'fp-help', status),
          );
        }
        const btn = button('Misura nella zona', () => {
          if (zoneConfirmed[z]) void measureInZone(z);
        }, 'btn btn--primary');
        const box = checkbox(`Il microfono di riferimento è nella ${name}, dove vuoi il silenzio (all’altezza della testa).`, (on) => {
          zoneConfirmed[z] = on;
          btn.disabled = !ready || !on;
        });
        box.input.checked = zoneConfirmed[z];
        box.input.disabled = busy;
        btn.disabled = !ready || !zoneConfirmed[z];
        const zdb = zoneDb[z];
        return el(
          'div',
          'wz-zone',
          el('h5', null, `Microfono nella ${name}`),
          box.root,
          el('div', 'wz-actions', btn),
          el('p', 'fp-help', zdb !== null ? `Fatto: percorso cassa → zona misurato, rumore nella zona ${formatNumber(zdb, 1)} dBFS.` : 'Misura cassa → zona e livello del rumore nella zona.'),
        );
      }),
    );
  }

  return {
    element,
    update(s: ControllerState) {
      const busy = env.busy !== null;
      const errMic = errorMicOn(s);
      const cal = calibration();
      const m: Method = errMic ? 'error-mic' : method;
      methodChoice.root.hidden = errMic;
      methodChoice.disable(busy);
      secChoice.root.hidden = m === 'geometry';
      secChoice.disable(busy);
      errorBlock.hidden = m !== 'error-mic';
      twoBlock.hidden = m !== 'two-position';
      if (m === 'error-mic') zoneCards(errZones, s, 'err');
      if (m === 'two-position') {
        zoneCards(twoZones, s, 'two');
        refBtn.disabled = busy || !s.armed || calibration() === null || !refBox.input.checked;
        setText(refStatus, refDb !== null ? `Fatto: rumore al riferimento ${formatNumber(refDb, 1)} dBFS.` : 'Misura il livello del rumore con il microfono nella posizione normale.');
      }
      const zones = zonesOf(s);
      const inputsReady =
        m === 'error-mic'
          ? zones.every((z) => primaries[z] !== null)
          : m === 'two-position'
            ? zones.every((z) => zoneDb[z] !== null) && refDb !== null
            : true;
      solveBtn.disabled = busy || cal === null || !inputsReady;
      stopBtn.disabled = running === null;
      progress.set(running !== null, s.measureProgress, running ? `${running}…` : '');
      if (cal === null) errNote.set('error', 'Le misure del passo 5 mancano o non valgono più (frequenza di campionamento cambiata): torna al passo 5.');
      else if (opError) errNote.set('error', opError);
      else errNote.clear();
      if (outcomeText) outcome.set(outcomeText.level, outcomeText.text);
      else outcome.clear();
      setListItems(adviceList, data.solution ? data.solution.advice : advice);
      if (data.solution?.applied) {
        const c = s.params.channels;
        setText(
          paramsLine,
          activeChannels(s.speakers)
            .map((ch) =>
              c[ch].enabled
                ? `${CH_NAME[ch]}: ritardo ${formatNumber(c[ch].delayMs, 3)} ms, guadagno ${formatNumber(c[ch].gainDb, 1)} dB, polarità ${c[ch].polarityInvert ? 'invertita (180°)' : 'diretta (0°)'}`
                : `${CH_NAME[ch]}: spento`,
            )
            .join(' · ') +
            (data.solution.loopGainDb !== null && data.solution.loopGainDb > LOOP_GAIN_LIMIT_DB
              ? ' · neutralizzazione del feedback attiva'
              : ''),
        );
      } else setText(paramsLine, '');

      tuneBlock.hidden = !errMic || !data.solution?.applied;
      tuneZoneChoice.root.hidden = s.speakers !== 'separate';
      tuneZoneChoice.disable(busy);
      tuneBtn.disabled = busy || unmuteBlockers(s, data).length > 0;
      if (running === 'Autotune') {
        tuneNote.set('warn', `Autotune in corso: misura ${tune.measured}… (anti-rumore attivo, Esc per fermare).`);
      } else if (tune.error) tuneNote.set('error', tune.error);
      else if (tune.result) {
        const r = tune.result;
        tuneNote.set(
          r.aborted ? 'warn' : 'ok',
          r.aborted
            ? r.reason ?? 'Autotune interrotto: valori iniziali ripristinati.'
            : `Errore nella zona da ${fmtDb(r.startDb)} a ${fmtDb(r.endDb)} (bypass ${fmtDb(r.bypassDb)}): ` +
                (r.endDb < r.startDb ? 'valori migliorati applicati.' : 'nessun miglioramento, valori invariati.'),
        );
      } else tuneNote.clear();
    },
    leave() {
      if (running !== null) env.abortOp();
    },
  };
}
