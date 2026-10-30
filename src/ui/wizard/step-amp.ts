/**
 * Step 2 — amplifier level: with the amplifier at minimum, band-limited pink noise (≈ −20 dBFS RMS
 * at the output, through master and limiter) plays on one channel; the user raises the amplifier
 * until, in the zone, it is about as loud as the noise to cancel (never above 85 dBA). Stop button
 * and Esc (PANIC) stop it. Required before any un-mute, for every channel that will be used.
 * The master is brought to 0 dB on entering, and the noise starts only once the engine reports
 * (MeterSnapshot.masterDb) that its rate-limited master (6 dB/s) is within 0.1 dB of it: the user sets
 * the amplifier against the final level from the first second. If the meters never confirm it
 * (masterWaitTimeoutMs) the noise does not start and the step says why. Reusing a saved calibration,
 * the saved master comes back when leaving the step.
 */
import type { ControllerState } from '../../app/controller.ts';
import { defaultSweepBand } from '../../audio/calibration.ts';
import { formatNumber } from '../units.ts';
import { button, checkbox, el, liveNote, note, progressBar, radioCards, setText } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { ampKey, ampLeveledOk, amplifierReleveled } from './flow.ts';
import {
  AMP_STEP_MASTER_DB,
  CH_NAME,
  liveMasterDb,
  masterReached,
  masterWaitTimeoutMs,
  refLevelAdvice,
  stimulusLevels,
} from './logic.ts';
import { NOISE_MAX_S, playLevelNoise, waitForState } from './ops.ts';

/** Noise heard for at least this long before the level can be confirmed. */
const MIN_LISTEN_S = 3;

/** The live master never reached the step's level (meters missing, or the engine not following). */
export function masterTimeoutText(timeoutMs: number): string {
  return (
    `Il volume master non è arrivato a 0 dB entro ${formatNumber(Math.ceil(timeoutMs / 1000), 0)} s: il rumore di prova non è partito. ` +
    'Controlla che l’audio sia attivo (gli indicatori di livello si muovono) e premi di nuovo «Fai suonare il rumore di prova».'
  );
}

/** The live master has reached the step's level. */
const masterAtStep = (s: ControllerState): boolean => masterReached(liveMasterDb(s), AMP_STEP_MASTER_DB);

export function createAmpStep(env: StepEnv): StepView {
  const { controller, data } = env;
  let channel: 0 | 1 = 0;
  let playing: 0 | 1 | null = null;
  let playErr: string | null = null;
  /** Waiting for the live master before the noise starts. */
  let preparing = false;

  /** Brings the master to the step's level (silent: the anti-noise is muted, no stimulus plays yet). */
  function raiseMaster(s: ControllerState): void {
    if (!s.armed || s.panic || s.params.masterDb === AMP_STEP_MASTER_DB) return;
    controller.setMaster(AMP_STEP_MASTER_DB);
  }

  const checks = [
    'L’amplificatore è acceso con il volume al minimo.',
    'Durante la prova tengo una mano sulla manopola del volume (o sull’interruttore) dell’amplificatore.',
    'So che Esc ferma tutto solo se la finestra del browser è in primo piano: se ho un dubbio abbasso l’amplificatore a mano.',
  ].map((text, i) =>
    checkbox(text, (on) => {
      data.amp.checklist[i] = on;
      env.refresh();
    }),
  );

  const refLevel = liveNote();

  const chChoice = radioCards<'0' | '1'>(
    'Canale su cui suona il rumore di prova',
    [
      { value: '0', title: 'Canale L', text: 'Usalo se hai una sola cassa.' },
      { value: '1', title: 'Canale R', text: 'Solo con due casse.' },
    ],
    (v) => {
      channel = v === '1' ? 1 : 0;
      env.refresh();
    },
  );
  chChoice.set('0');

  const playBtn = button('Fai suonare il rumore di prova', () => void start(), 'btn btn--primary');
  const stopBtn = button('Stop', () => env.abortOp(), 'btn btn--danger');
  const progress = progressBar('Rumore di prova');
  const playNote = liveNote();

  const leveled = ([0, 1] as const).map((ch) =>
    checkbox(
      `Ho regolato l’amplificatore del canale ${CH_NAME[ch]}: nella zona il rumore di prova si sente circa forte quanto il rumore da cancellare (mai oltre 85 dBA).`,
      (on) => {
        const s = controller.state;
        data.amp.leveled[ch] = on;
        data.amp.key[ch] = on ? ampKey(s) : null;
        amplifierReleveled(data, ch);
        env.refresh();
      },
    ),
  );
  const chStatus = el('ul', 'wz-list wz-status-list');
  const reuseNote = note(
    'info',
    `Stai riusando la calibrazione «${data.reuse?.name ?? ''}»: regola l’amplificatore come la volta scorsa. Al passo successivo l’app ` +
      'misura di nuovo le casse con un breve sweep e corregge il guadagno se l’amplificatore è un po’ diverso; uscendo da questo passo ' +
      'il volume master torna al valore salvato.',
  );
  reuseNote.hidden = data.mode !== 'reuse' || data.reuse === null;

  async function start(): Promise<void> {
    const s = controller.state;
    const fs = s.fs;
    if (fs === null || !s.armed || s.panic) return;
    const ch = channel;
    playErr = null;
    raiseMaster(s);
    const op = env.startOp(`Rumore di prova sul canale ${CH_NAME[ch]}`);
    playing = ch;
    data.amp.playedS[ch] = 0;
    // The amplifier may be turned now: the quick check, or this channel's measurements, must be redone.
    amplifierReleveled(data, ch);
    env.refresh();
    try {
      // The level the user sets the amplifier against must be the final one from the first second:
      // wait until the engine's meters report the master there (or the target is changed meanwhile).
      if (!masterAtStep(controller.state)) {
        preparing = true;
        env.refresh();
        const timeoutMs = masterWaitTimeoutMs(liveMasterDb(controller.state), AMP_STEP_MASTER_DB);
        try {
          await waitForState(controller, (st) => masterAtStep(st) || st.params.masterDb !== AMP_STEP_MASTER_DB, timeoutMs, op.signal);
        } catch (e) {
          if (op.signal.aborted) throw e;
          playErr = masterTimeoutText(timeoutMs);
          return;
        }
        preparing = false;
      }
      if (controller.state.params.masterDb !== AMP_STEP_MASTER_DB) {
        playErr = 'Il volume master è stato cambiato: premi di nuovo «Fai suonare il rumore di prova».';
        return;
      }
      const reason = await playLevelNoise(controller, ch, {
        fs,
        band: defaultSweepBand(controller.state.params, fs),
        rmsDbfs: stimulusLevels(controller.state.params.ceilingDb).noiseRmsDbfs,
        signal: op.signal,
        onPlayed: (t) => {
          data.amp.playedS[ch] = t;
          env.refresh();
        },
      });
      if (reason !== null && !op.signal.aborted) playErr = reason;
    } catch {
      // Aborted while waiting for the master (Stop, Esc, step left).
    } finally {
      preparing = false;
      playing = null;
      op.end();
      env.refresh();
    }
  }

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'Questo passo regola l’amplificatore così che il livello digitale di lavoro (circa −20 dBFS) corrisponda al rumore da cancellare: ' +
        'il limite d’uscita resta così pochi dB sopra quel livello e un errore non può mai diventare forte.',
    ),
    el('div', 'wz-checklist', ...checks.map((c) => c.root)),
    el('h4', 'wz-sub', 'Livello del microfono di riferimento'),
    refLevel.root,
    el('h4', 'wz-sub', 'Rumore di prova'),
    chChoice.root,
    el('div', 'wz-actions', playBtn, stopBtn),
    progress.root,
    playNote.root,
    note(
      'info',
      'Mentre suona: vai nella zona da silenziare e alza lentamente l’amplificatore finché il rumore di prova si sente circa forte quanto il rumore da cancellare. ' +
        'Se hai un fonometro non superare 85 dBA. Poi premi Stop. Il volume master dell’app viene portato a 0 dB (sale di 6 dB al secondo: il rumore parte ' +
        'quando il motore audio conferma che è arrivato): da qui in poi il livello lo decide l’amplificatore.',
    ),
    reuseNote,
    el('div', 'wz-checklist', ...leveled.map((c) => c.root)),
    chStatus,
  );

  return {
    element,
    update(s: ControllerState) {
      // Levels set on another output device or sample rate no longer count.
      if (s.micOpen && s.fs !== null) {
        for (const ch of [0, 1] as const) {
          if (data.amp.key[ch] !== null && data.amp.key[ch] !== ampKey(s)) {
            data.amp.leveled[ch] = false;
            data.amp.key[ch] = null;
            data.amp.playedS[ch] = 0;
          }
        }
      }
      if (env.busy === null) raiseMaster(s);
      checks.forEach((c, i) => (c.input.checked = data.amp.checklist[i] === true));
      const m = s.meters;
      const adv = refLevelAdvice(s.micOpen && m ? m.ref.rmsDb : null, s.micOpen && m ? m.ref.peakDb : null);
      const lv = s.micOpen && m ? ` (RMS ${formatNumber(m.ref.rmsDb, 1)} dBFS, picco ${formatNumber(m.ref.peakDb, 1)} dBFS)` : '';
      refLevel.set(adv.level === 'ok' ? 'ok' : adv.level === 'high' ? 'error' : 'warn', `${adv.text}${lv}`);

      const ready = data.amp.checklist.every(Boolean) && s.armed && !s.panic && s.fs !== null;
      const busy = env.busy !== null;
      // Still rising according to the engine (without meters nothing is known: «Fai suonare» waits and says so).
      const live = liveMasterDb(s);
      const rising = s.armed && !s.panic && live !== null && !masterReached(live, AMP_STEP_MASTER_DB);
      chChoice.set(channel === 1 ? '1' : '0');
      chChoice.disable(busy);
      playBtn.disabled = !ready || busy;
      stopBtn.disabled = playing === null;
      const t = playing !== null ? data.amp.playedS[playing] : 0;
      progress.set(
        playing !== null && !preparing,
        t / NOISE_MAX_S,
        playing !== null ? `Canale ${CH_NAME[playing]}: ${formatNumber(t, 0)} s (si ferma da solo dopo ${NOISE_MAX_S / 60} min)` : '',
      );
      // A live region: no dB figure that changes at meter rate.
      const waitText = preparing
        ? 'Il volume master sta salendo a 0 dB (6 dB al secondo): il rumore di prova parte appena il motore audio conferma che è arrivato.'
        : 'Il volume master sta salendo a 0 dB (6 dB al secondo): se premi «Fai suonare il rumore di prova» ora, il rumore parte appena è arrivato.';
      if (playErr) playNote.set('error', playErr);
      else if (!data.amp.checklist.every(Boolean)) playNote.set('info', 'Spunta prima la lista di sicurezza.');
      else if (!s.armed || s.panic) playNote.set('warn', 'L’uscita non è abilitata: torna al passo 1 o riprendi dopo il PANIC.');
      else if (preparing || (playing === null && rising)) playNote.set('info', waitText);
      else if (playing !== null) playNote.set('warn', 'Rumore di prova in corso: Stop o Esc per fermarlo.');
      else playNote.clear();

      for (const ch of [0, 1] as const) {
        const c = leveled[ch]!;
        c.input.checked = ampLeveledOk(data, s, ch);
        c.input.disabled = busy || data.amp.playedS[ch] < MIN_LISTEN_S;
        c.root.hidden = data.amp.playedS[ch] < MIN_LISTEN_S && !ampLeveledOk(data, s, ch);
      }
      const texts = ([0, 1] as const).map(
        (ch) => `Canale ${CH_NAME[ch]}: ${ampLeveledOk(data, s, ch) ? 'regolato' : ch === 1 ? 'da regolare solo se usi due casse' : 'da regolare'}`,
      );
      if (chStatus.dataset['key'] !== texts.join('|')) {
        chStatus.dataset['key'] = texts.join('|');
        chStatus.replaceChildren(...texts.map((t2) => el('li', null, t2)));
      }
      setText(stopBtn, 'Stop');
    },
    leave() {
      if (playing !== null) env.abortOp();
      // Reusing a saved calibration: its master volume is part of the saved anti-noise level.
      const r = data.mode === 'reuse' ? data.reuse : null;
      if (r !== null && controller.state.params.masterDb !== r.masterDb) controller.setMaster(r.masterDb);
    },
  };
}
