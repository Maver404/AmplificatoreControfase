/**
 * Step "Controllo rapido" (reuse flow, plan: "ricalibrazione rapida … rispetto a una baseline
 * salvata"): one short sweep per active speaker on the speaker → reference path, compared with
 * the saved calibration. Within 0.5 samples and 0.5 dB the saved values stay; a small change
 * (amplifier re-levelled by ear at step 2, the latency of a new audio context) corrects channel
 * gain and delay and replaces the saved feedback path (the neutralisation FIRs follow); a big
 * change, a flipped polarity or another interface asks for new measurements. A failed measurement
 * keeps the saved values and says so: the user must accept that explicitly.
 */
import type { ControllerState } from '../../app/controller.ts';
import { looksLikeLoopback, measurePath } from '../../audio/calibration.ts';
import type { CalibrationData } from '../../audio/calibration.ts';
import { parseDeviceKey } from '../../audio/devices.ts';
import type { ChannelParams } from '../../dsp/params.ts';
import { formatSigned } from '../units.ts';
import { button, checkbox, el, liveNote, progressBar, setListItems, setText } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { ampKey, emptyCheck, resetMeasurements, reuseDeviceMatch, REUSE_OTHER_DEVICE_TEXT } from './flow.ts';
import type { CheckSlot } from './flow.ts';
import { activeChannels, CH_NAME, compareToBaseline, quickVerdict, stimulusLevels, toPathData } from './logic.ts';

/** Sweeps per channel after the preview (median of these). */
const QUICK_REPETITIONS = 2;

const SLOT_LABEL: Record<CheckSlot['state'], string> = {
  running: 'in corso…',
  ok: 'confermata',
  corrected: 'corretta',
  unverified: 'non verificata',
  mismatch: 'da rifare',
  aborted: 'annullata',
};

/** Italian description of the interfaces a device key names. */
function deviceText(key: string): string {
  const k = parseDeviceKey(key);
  if (!k) return '';
  const part = (p: { label: string } | null): string => (p === null ? 'nessuno' : p.label ? `«${p.label}»` : 'sconosciuto');
  return `ingresso ${part(k.input)}, uscita ${part(k.output)}`;
}

export function createCheckStep(env: StepEnv): StepView {
  const { controller, data } = env;
  let running = false;
  let done = 0;
  let total = 0;

  /** Measurements are no longer trusted: back to the devices step with the "rifai le misure" flow. */
  const remeasure = (): void => {
    resetMeasurements(data);
    data.check = emptyCheck();
    env.setMode('recal');
  };

  async function runCheck(): Promise<void> {
    const s = controller.state;
    const cal = data.calibration;
    if (cal === null || s.fs === null || cal.fs !== s.fs || !s.armed || s.panic) return;
    const op = env.startOp('Controllo rapido delle casse');
    running = true;
    const chans = activeChannels(s.speakers);
    data.check = { key: ampKey(s), channels: [null, null], acceptUnverified: false };
    done = 0;
    total = chans.length * (QUICK_REPETITIONS + 1);
    const next: CalibrationData = JSON.parse(JSON.stringify(cal)) as CalibrationData;
    const start = controller.currentParams;
    const channels: [ChannelParams, ChannelParams] = [{ ...start.channels[0] }, { ...start.channels[1] }];
    let corrected = false;
    try {
      for (const ch of chans) {
        data.check.channels[ch] = { state: 'running', text: 'Sweep in corso…' };
        env.refresh();
        const now = controller.state;
        const r = await measurePath(controller.measure, env.analyzer(), {
          fs: cal.fs,
          channel: ch,
          band: cal.band,
          levelDbfs: Math.min(0, stimulusLevels(now.params.ceilingDb).sweepPeakDbfs - now.params.masterDb),
          repetitions: QUICK_REPETITIONS,
          mic: 'ref',
          mode: 'acoustic',
          previewFirst: true,
          signal: op.signal,
        });
        done += QUICK_REPETITIONS + 1;
        if (op.signal.aborted || r.aborted) {
          data.check.channels[ch] = { state: 'aborted', text: 'Controllo annullato.' };
          return;
        }
        const saved = cal.feedback[ch];
        if (!r.ok || saved === null) {
          data.check.channels[ch] = {
            state: 'unverified',
            text: !r.ok
              ? `Misura non riuscita (${r.reason ?? 'motivo sconosciuto'}): restano i valori salvati, non verificati.`
              : 'Nella calibrazione salvata questa cassa non si sentiva al microfono di riferimento: restano i valori salvati, non verificati.',
          };
          continue;
        }
        const lat = cal.loopbackLatencyS !== null ? { seconds: cal.loopbackLatencyS, fs: cal.fs } : undefined;
        if (looksLikeLoopback(r, lat)) {
          // The sweep is recorded on the reference input: that is where the cable must be.
          const input = controller.state.params.refChannel + 1;
          data.check.channels[ch] = {
            state: 'mismatch',
            text: `La misura ha la firma di un cavo di loopback: scollega il cavo dall’ingresso ${input} (microfono di riferimento) e ripeti.`,
          };
          continue;
        }
        const path = toPathData(r);
        const v = quickVerdict(compareToBaseline(saved, path, cal.fs, cal.band), channels[ch], cal.fs);
        if (v.kind === 'ok') {
          data.check.channels[ch] = {
            state: 'ok',
            text:
              `Come nella calibrazione salvata: ritardo ${formatSigned(v.diff.delaySamples, 2)} campioni, ` +
              `livello ${formatSigned(v.diff.gainDb, 1)} dB.`,
          };
        } else if (v.kind === 'corrected') {
          channels[ch] = v.channel;
          next.feedback[ch] = path;
          corrected = true;
          data.check.channels[ch] = { state: 'corrected', text: v.text };
        } else {
          data.check.channels[ch] = { state: 'mismatch', text: v.text };
        }
      }
    } finally {
      if (corrected) {
        // Anti-noise still muted (lock): the corrected values and the new feedback paths only
        // reach the engine; they become the double-click values and the record saved with a preset.
        data.calibration = next;
        controller.setCalibration(JSON.parse(JSON.stringify(next)) as unknown);
        controller.applyParams({ ...controller.currentParams, channels }, { calibrated: true });
      }
      running = false;
      op.end();
      env.refresh();
    }
  }

  const deviceNote = liveNote();
  const otherBtn = button('Rifai le misure (passi 4–6)', remeasure, 'btn');
  const checkBtn = button('Controlla le casse', () => void runCheck(), 'btn btn--primary');
  const stopBtn = button('Interrompi', () => env.abortOp(), 'btn btn--danger');
  const progress = progressBar('Controllo rapido');
  const results = el('ul', 'wz-list wz-status-list');
  const accept = checkbox(
    'Uso comunque i valori salvati per le casse non verificate (l’anti-rumore potrebbe essere sfasato: verifica all’ascolto con il bypass).',
    (on) => {
      data.check.acceptUnverified = on;
      env.refresh();
    },
  );
  const outcome = liveNote();

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'L’app fa suonare un breve sweep su ogni cassa e lo confronta con la calibrazione salvata. Se l’amplificatore (regolato a orecchio ' +
        'al passo 2) o la latenza sono cambiati di poco, corregge guadagno e ritardo; se qualcosa è stato spostato ti chiede di rifare le misure.',
    ),
    deviceNote.root,
    el('div', 'wz-actions', checkBtn, stopBtn),
    progress.root,
    results,
    accept.root,
    outcome.root,
    el('div', 'wz-actions', otherBtn),
  );

  return {
    element,
    update(s: ControllerState) {
      const busy = env.busy !== null;
      const match = reuseDeviceMatch(data, s);
      const saved = data.reuse?.deviceKey ?? '';
      if (match === 'different') deviceNote.set('error', `${REUSE_OTHER_DEVICE_TEXT} Salvata con: ${deviceText(saved)}.`);
      else if (match === 'same-model') {
        deviceNote.set(
          'warn',
          'La calibrazione è stata salvata con lo stesso modello di interfaccia ma su un altro esemplare, un’altra porta USB o un altro indirizzo ' +
            'dell’app: il controllo rapido verifica ritardi e livelli.',
        );
      } else if (saved === '') deviceNote.set('info', 'Il preset non indica con quale interfaccia è stato creato: il controllo rapido verifica ritardi e livelli.');
      else deviceNote.clear();

      const cal = data.calibration;
      const rateOk = cal !== null && s.fs !== null && cal.fs === s.fs;
      checkBtn.disabled = busy || !s.armed || s.panic || !rateOk || match === 'different';
      stopBtn.disabled = !running;
      otherBtn.disabled = busy;
      progress.set(running, total > 0 ? Math.min(1, (done + (s.measureProgress ?? 0)) / total) : null, running ? 'Sweep di controllo…' : '');

      const current = data.check.key === ampKey(s);
      const chans = activeChannels(s.speakers);
      setListItems(
        results,
        current
          ? chans.flatMap((ch) => {
              const r = data.check.channels[ch];
              return r ? [`Cassa ${CH_NAME[ch]} — ${SLOT_LABEL[r.state]}: ${r.text}`] : [];
            })
          : [],
      );
      const unverified = current && chans.some((ch) => data.check.channels[ch]?.state === 'unverified');
      accept.root.hidden = !unverified;
      accept.input.checked = data.check.acceptUnverified;
      accept.input.disabled = busy;

      if (!rateOk) outcome.set('error', 'La calibrazione salvata è a un’altra frequenza di campionamento: rifai le misure.');
      else if (!current) outcome.set('info', data.check.key === null ? 'Premi «Controlla le casse».' : 'Uscita o frequenza cambiate: ripeti il controllo.');
      else if (chans.some((ch) => data.check.channels[ch]?.state === 'mismatch')) {
        outcome.set('error', 'Qualcosa è cambiato troppo per correggerlo: rifai le misure (pulsante qui sotto).');
      } else if (chans.every((ch) => ['ok', 'corrected'].includes(data.check.channels[ch]?.state ?? ''))) {
        outcome.set('ok', 'Calibrazione verificata: puoi andare avanti.');
      } else outcome.clear();
      setText(checkBtn, data.check.key === null ? 'Controlla le casse' : 'Ripeti il controllo');
    },
    leave() {
      if (running) env.abortOp();
    },
  };
}
