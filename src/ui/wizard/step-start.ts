/**
 * Start question when a calibration is already saved: "Hai spostato mic, casse o amplificatore?"
 * No → reuse it (devices, amplifier level, quick re-check of the paths, verification); yes →
 * measure again (steps 4–6); or the whole configuration from scratch.
 */
import type { ControllerState } from '../../app/controller.ts';
import { formatNumber } from '../units.ts';
import { el, note, setText, uid } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { resetMeasurements } from './flow.ts';
import type { FlowMode } from './flow.ts';
import { geometryToDraft, presetsWithCalibration } from './logic.ts';

const dateText = (t: number): string => {
  if (!(t > 0)) return '';
  const d = new Date(t);
  return d.toLocaleString('it-IT', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
};

export function createStartStep(env: StepEnv): StepView {
  const { controller, data } = env;
  const selId = uid('wz-preset');
  const select = el('select');
  select.id = selId;
  const selLabel = el('label', 'field-label', 'Calibrazione salvata');
  selLabel.htmlFor = selId;
  const selField = el('div', 'field wz-field', selLabel, select);
  const summary = el('p', 'fp-help');

  const choose = (mode: FlowMode): void => {
    const saved = presetsWithCalibration(controller.state.presets.list);
    const pick = saved.find((x) => x.preset.id === select.value) ?? saved[0];
    // A previous answer may have loaded the saved solution: each answer starts from clean measurements.
    resetMeasurements(data);
    if (mode !== 'full' && pick) {
      // Loaded while the output is silent: values only, then the steps take it from there.
      controller.loadPreset(pick.preset.id);
      controller.setSpeakers(pick.calibration.speakers);
      data.presetId = pick.preset.id;
      data.reuse = {
        presetId: pick.preset.id,
        name: pick.preset.name,
        deviceKey: pick.preset.deviceKey,
        masterDb: controller.state.params.masterDb,
      };
      const g = pick.calibration.geometry;
      if (g !== null) data.geometry = geometryToDraft(g);
      if (mode === 'reuse') {
        // Saved before the reference input was recorded: it is the preset's (now loaded).
        data.calibration = { ...pick.calibration, refChannel: pick.calibration.refChannel ?? controller.state.params.refChannel };
        data.solution = { method: 'preset', applied: true, predictedDb: null, loopGainDb: null, causal: null, advice: [], geometryKey: null };
      }
    } else data.reuse = null;
    data.modeChosen = true;
    env.setMode(mode);
  };

  const noBtn = el('button', 'btn btn--primary wz-big-btn', 'No, usa la calibrazione salvata');
  noBtn.type = 'button';
  noBtn.addEventListener('click', () => choose('reuse'));
  const yesBtn = el('button', 'btn wz-big-btn', 'Sì, ho spostato qualcosa: rifai le misure');
  yesBtn.type = 'button';
  yesBtn.addEventListener('click', () => choose('recal'));
  const fullBtn = el('button', 'btn btn--small', 'Rifai tutta la configurazione da capo');
  fullBtn.type = 'button';
  fullBtn.addEventListener('click', () => choose('full'));

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'Hai spostato il microfono, le casse o l’amplificatore (o cambiato il volume dell’amplificatore) dall’ultima calibrazione?',
    ),
    selField,
    summary,
    el('div', 'wz-actions', noBtn, yesBtn),
    note(
      'info',
      'In entrambi i casi ripeterai la scelta dei dispositivi e il controllo del livello dell’amplificatore: ' +
        'sono passi di sicurezza. Con «No» l’app ricontrolla poi le casse con un breve sweep e corregge guadagno e ritardo se ' +
        'l’amplificatore o la latenza sono cambiati di poco; con «Sì» rifarai posizionamento, misure e allineamento.',
    ),
    el('div', 'wz-actions', fullBtn),
  );

  return {
    element,
    update(s: ControllerState) {
      const saved = presetsWithCalibration(s.presets.list);
      const key = saved.map((x) => `${x.preset.id}:${x.preset.name}`).join('|');
      if (select.dataset['key'] !== key) {
        const prev = select.value || data.presetId || s.presets.activeId;
        select.replaceChildren(
          ...saved.map((x) => {
            const o = el('option', null, `${x.preset.name} — ${dateText(x.preset.createdAt)}`);
            o.value = x.preset.id;
            return o;
          }),
        );
        select.dataset['key'] = key;
        if (prev && saved.some((x) => x.preset.id === prev)) select.value = prev;
      }
      selField.hidden = saved.length < 2;
      const cur = saved.find((x) => x.preset.id === select.value) ?? saved[0];
      setText(
        summary,
        cur
          ? `«${cur.preset.name}», salvata il ${dateText(cur.preset.createdAt)} · ${formatNumber(cur.calibration.fs / 1000, 1)} kHz · ${
              cur.calibration.speakers === 'single' ? 'una cassa' : 'due casse'
            }`
          : 'Nessuna calibrazione salvata trovata.',
      );
      noBtn.disabled = yesBtn.disabled = !cur;
    },
    leave() {},
  };
}
