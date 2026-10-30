/**
 * Step 1 — devices and validation: microphone permission (openMic inside the click), input,
 * output, reference channel, error mic; validation issues (block / error with "continua comunque"
 * / warn / info); headphones confirmation; then the output is armed with the anti-noise muted.
 * Reusing a saved calibration, the interfaces must be the ones it was measured with (else: choose
 * them, or measure again).
 */
import type { ControllerState } from '../../app/controller.ts';
import { describeDevice } from '../../app/controller.ts';
import type { DeviceOption } from '../../audio/devices.ts';
import { createDevicePanel } from '../device-panel.ts';
import type { DeviceOption as PanelDeviceOption } from '../device-panel.ts';
import { createWarnings } from '../warnings.ts';
import type { WarningItem } from '../warnings.ts';
import { button, checkbox, el, liveNote, note, setText } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { emptyCheck, resetMeasurements, reuseDeviceMatch, REUSE_OTHER_DEVICE_TEXT } from './flow.ts';

const panelOption = (d: DeviceOption): PanelDeviceOption => {
  const { note: n, blocked } = describeDevice(d);
  return n !== undefined ? { deviceId: d.deviceId, label: d.label, note: n, blocked } : { deviceId: d.deviceId, label: d.label, blocked };
};

function issueItems(s: ControllerState, env: StepEnv): WarningItem[] {
  if (!s.micOpen) return [];
  return s.validation.issues.map((i) => {
    const item: WarningItem = {
      id: `wz-${i.id}`,
      level: i.overridden ? 'warn' : i.level,
      text: i.overridden ? `Ignorato su tua richiesta: ${i.text}` : i.text,
    };
    if (i.overridable) {
      item.action = i.overridden
        ? { label: 'Annulla', onClick: () => env.controller.setOverride(i.id, false) }
        : { label: 'Continua comunque', onClick: () => env.controller.setOverride(i.id, true) };
    }
    return item;
  });
}

export function createDevicesStep(env: StepEnv): StepView {
  const { controller, data } = env;

  const panelHost = el('div', 'wz-devices');
  const panel = createDevicePanel(panelHost, {
    // Synchronous call inside the click: the AudioContext is created before any await.
    onAllowMic: () => void controller.openMic(),
    onInputChange: (id) => void controller.setInput(id),
    onOutputChange: (id) => void controller.setOutput(id),
    onRefChannelChange: (ch) => controller.setRefChannel(ch),
    onErrorChannelChange: (ch) => controller.setErrorChannel(ch),
    onLatencyModeChange: (mode) => void controller.setLatencyMode(mode),
  });

  const issuesHost = el('div', 'wz-issues');
  const issues = createWarnings(issuesHost, { emptyText: 'Nessun problema rilevato.' });
  const issuesTitle = el('h4', 'wz-sub', 'Verifica dei dispositivi');

  const hp = checkbox(
    'Confermo: le cuffie non sono indossate e il loro volume è al minimo (limite d’uscita −24 dBFS).',
    (on) => controller.confirmHeadphones(on),
  );

  const armBtn = el('button', 'btn btn--primary', 'Abilita l’uscita per le misure');
  armBtn.type = 'button';
  armBtn.addEventListener('click', () => {
    const s = controller.state;
    controller.arm({ muted: true, headphonesConfirmed: s.headphonesConfirmed || hp.input.checked });
  });
  const status = liveNote();
  const deviceNote = liveNote();
  const remeasureBtn = button(
    'Rifai le misure con questa interfaccia',
    () => {
      resetMeasurements(data);
      data.check = emptyCheck();
      env.setMode('recal');
    },
    'btn',
  );
  const remeasureRow = el('div', 'wz-actions', remeasureBtn);

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'Collega il microfono di riferimento (all’esterno, verso la sorgente del rumore) e l’amplificatore alla stessa interfaccia audio USB. ' +
        'Poi consenti il microfono e scegli ingresso e uscita dell’interfaccia.',
    ),
    note(
      'warn',
      'Prima di tutto abbassa al minimo il volume dell’amplificatore. In qualunque momento ',
      el('kbd', null, 'Esc'),
      ' (con la finestra del browser in primo piano) o il pulsante PANIC azzerano l’uscita.',
    ),
    panelHost,
    issuesTitle,
    issuesHost,
    hp.root,
    el('div', 'wz-actions', armBtn),
    status.root,
    deviceNote.root,
    remeasureRow,
    el(
      'p',
      'fp-help',
      'L’uscita viene abilitata con l’anti-rumore muto: nei prossimi passi potranno suonare solo il rumore di prova e gli sweep di misura.',
    ),
  );

  return {
    element,
    update(s: ControllerState) {
      panel.update({
        permission: s.micOpen ? 'granted' : s.permission === 'denied' ? 'denied' : 'prompt',
        inputs: s.inputs.map(panelOption),
        outputs: s.outputs.map(panelOption),
        inputId: s.inputId,
        outputId: s.outputId,
        sinkIdSupported: s.sinkSupported,
        refChannel: s.params.refChannel,
        errorChannel: s.params.errorChannel,
        latencyMode: s.latencyMode,
        ch2: s.ch2,
        trackChannels: s.trackChannels,
        busy: s.opening,
      });
      issues.update(issueItems(s, env));
      issuesTitle.hidden = issuesHost.hidden = !s.micOpen;
      hp.root.hidden = !(s.micOpen && s.validation.headphones);
      hp.input.checked = s.headphonesConfirmed;
      armBtn.hidden = s.armed && !s.panic;
      armBtn.disabled = !s.canArm || (s.validation.headphones && !s.headphonesConfirmed);
      if (s.micError && !s.micOpen) status.set('error', s.micError);
      else if (!s.micOpen) status.set('info', s.opening ? 'Apertura del microfono in corso…' : 'Microfono non ancora consentito.');
      else if (!s.validation.ok) status.set('warn', 'Risolvi i problemi bloccanti (o usa «Continua comunque» dove è consentito).');
      else if (s.armed && !s.panic) status.set('ok', 'Uscita abilitata per le misure, anti-rumore muto. Puoi andare avanti.');
      else status.set('info', 'Verifica superata: abilita l’uscita per le misure.');
      setText(armBtn, s.panic ? 'Riprendi (anti-rumore muto)' : 'Abilita l’uscita per le misure');
      if (s.panic) armBtn.hidden = false;
      const other = reuseDeviceMatch(data, s) === 'different';
      if (other) deviceNote.set('error', REUSE_OTHER_DEVICE_TEXT);
      else deviceNote.clear();
      remeasureRow.hidden = !other;
      remeasureBtn.disabled = env.busy !== null;
    },
    leave() {},
  };
}
