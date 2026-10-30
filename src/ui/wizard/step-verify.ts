/**
 * Step 7 — A/B verification: the only place where the guided setup un-mutes the anti-noise (with
 * the 1.5 s soft-start), after every safety condition holds. Live attenuation with the error mic,
 * broadband and per third-octave band (normalised by the reference of each state) and, in tonal
 * mode, at each harmonic h·f0 (the tone's own bins, next to the calibrated values' prediction or
 * autotune measurement; a prediction at its model's ceiling worded as that limit, logic.ts ceilingText),
 * bypass toggling (B), the controller's "ANC sta aumentando il rumore" guard, save as preset.
 */
import type { ControllerEvent, ControllerState } from '../../app/controller.ts';
import { defaultSweepBand } from '../../audio/calibration.ts';
import { harmonicAttenuationDb } from '../../app/spectrum.ts';
import { bandAttenuationDb, thirdOctaveCenters, thirdOctaveLabel } from '../spectrum-view.ts';
import { bankHarmonics, formatSigned } from '../units.ts';
import { button, el, liveNote, note, setListItems, setText, uid } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { errorMicOn, LOCK_TEXT, unmuteBlockers } from './flow.ts';
import { defaultPresetName, expectedHarmonic, fmtDb, harmonicMeasuredText, howlAlarmText } from './logic.ts';

/** The band list is rewritten at most this often (it is a polite live region). */
const BANDS_EVERY_MS = 1500;

export function createVerifyStep(env: StepEnv): StepView {
  const { controller, data } = env;
  let alarm: string | null = null;

  const off = controller.onEvent((e: ControllerEvent) => {
    if (e.type === 'anc-worse') alarm = 'L’ANC stava aumentando il rumore nella zona: uscita silenziata. Controlla posizioni e misure (passi 4–6).';
    else if (e.type === 'howl') alarm = howlAlarmText(e.detail);
    else return;
    env.refresh();
  });

  // Per-band attenuation, normalised by the controller's reference spectrum of each ANC state.
  let bandTexts: string[] = [];
  let harmTexts: string[] = [];
  let bandsAt = 0;
  const offSpectra = controller.subscribe((s, change) => {
    if (change !== 'spectrum') return;
    const now = performance.now();
    if (now - bandsAt < BANDS_EVERY_MS) return;
    bandsAt = now;
    const sp = controller.spectra;
    const band = data.calibration?.band ?? (s.fs !== null ? defaultSweepBand(s.params, s.fs) : [20, 500]);
    bandTexts = thirdOctaveCenters(Math.max(20, band[0]), Math.min(4000, band[1])).flatMap((fc) => {
      const a = bandAttenuationDb({ errOff: sp.errBypass, refOff: sp.refOff, errOn: sp.err, refOn: sp.refOn }, sp.fs, sp.fftSize, fc);
      return a === null ? [] : [`${thirdOctaveLabel(fc)}: ${formatSigned(a, 1)} dB`];
    });
    // Tonal mode: the anti-noise only plays the harmonics, so each is measured on its own bins.
    const p = s.params;
    const measured = p.phaseMode === 'tonal' ? harmonicAttenuationDb(sp, p.tonal.f0, p.tonal.harmonics) : [];
    harmTexts =
      p.phaseMode === 'tonal' && sp.fs > 0
        ? bankHarmonics(p.tonal.harmonics, p.tonal.f0, sp.fs).map((h) =>
            harmonicMeasuredText(h, h * p.tonal.f0, measured[h - 1] ?? null, expectedHarmonic(s, data.solution, h)),
          )
        : [];
    env.refresh();
  });

  const blockers = el('ul', 'wz-list wz-errors');
  const startBtn = button('Attiva l’anti-rumore (soft-start 1,5 s)', () => {
    if (unmuteBlockers(controller.state, data).length > 0) return;
    alarm = null;
    controller.setAncLock(null);
    if (controller.setMuted(false)) data.verify.unmuted = true;
    else controller.setAncLock(LOCK_TEXT);
    env.refresh();
  }, 'btn btn--primary wz-big-btn');
  const muteBtn = button('Silenzia', () => controller.setMuted(true), 'btn');
  const abBtn = button('Confronto A/B: bypass', () => controller.toggleBypass(), 'toggle-btn');
  const state = liveNote();
  const big = el('p', 'wz-readout', '—');
  const bigLabel = el('p', 'fp-help');
  const bandsTitle = el('h4', 'wz-sub', 'Attenuazione per terzi d’ottava');
  const bands = el('ul', 'wz-list wz-bands');
  const bandsHelp = el(
    'p',
    'fp-help',
    'Per ogni banda: (errore − riferimento) in bypass meno (errore − riferimento) ad anti-rumore acceso; positivo = più silenzio. ' +
      'Servono alcuni secondi in ciascuno dei due stati (alterna con B).',
  );
  const harmTitle = el('h4', 'wz-sub', 'Attenuazione per armonica');
  const harmList = el('ul', 'wz-list wz-harmonics');
  const harmHelp = el(
    'p',
    'fp-help',
    'Modalità tonale: la stessa misura solo sulle righe dello spettro alle armoniche di f0, dove agisce l’anti-rumore; ' +
      'tra parentesi la previsione del passo 6 (o la misura dell’autotune).',
  );
  const alarmNote = liveNote();

  const nameId = uid('wz-preset-name');
  const nameInput = el('input');
  nameInput.type = 'text';
  nameInput.id = nameId;
  nameInput.maxLength = 80;
  nameInput.value = defaultPresetName(new Date());
  const nameLabel = el('label', 'field-label', 'Nome del preset');
  nameLabel.htmlFor = nameId;
  const saveBtn = button('Salva come preset', () => {
    controller.savePreset(nameInput.value.trim() || defaultPresetName(new Date()));
    env.refresh();
  });
  const saveNote = liveNote();
  const finishBtn = button('Fine: chiudi la configurazione guidata', () => env.complete(), 'btn btn--primary');

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'Ultimo passo: attiva l’anti-rumore e confronta con il bypass. Resta nella zona con la mano sul volume dell’amplificatore; ',
      el('kbd', null, 'Esc'),
      ' = PANIC, ',
      el('kbd', null, 'B'),
      ' = bypass A/B, ',
      el('kbd', null, 'M'),
      ' = muto.',
    ),
    blockers,
    el('div', 'wz-actions', startBtn, abBtn, muteBtn),
    state.root,
    el('div', 'wz-readout-box', big, bigLabel),
    bandsTitle,
    bands,
    bandsHelp,
    harmTitle,
    harmList,
    harmHelp,
    alarmNote.root,
    note(
      'info',
      'Se con l’anti-rumore attivo il rumore aumenta o compare un fischio: Esc subito, poi controlla polarità e posizioni. ' +
        'Con il mic d’errore l’app silenzia da sola l’uscita se l’errore supera di 3 dB il livello in bypass per più di 2 secondi.',
    ),
    el('h4', 'wz-sub', 'Salva la configurazione'),
    el('div', 'field wz-field', nameLabel, nameInput),
    el('div', 'wz-actions', saveBtn),
    saveNote.root,
    el('div', 'wz-actions wz-finish', finishBtn),
  );

  return {
    element,
    update(s: ControllerState) {
      const list = unmuteBlockers(s, data);
      setListItems(blockers, list);
      const on = s.armed && !s.muted && !s.panic;
      startBtn.hidden = on;
      startBtn.disabled = list.length > 0 || env.busy !== null;
      muteBtn.hidden = !on;
      abBtn.disabled = !on;
      abBtn.setAttribute('aria-pressed', String(s.bypass));
      setText(abBtn, s.bypass ? 'Confronto A/B: bypass attivo (B)' : 'Confronto A/B: bypass (B)');

      if (!on) state.set(list.length > 0 ? 'warn' : 'info', list.length > 0 ? 'Prima risolvi i punti elencati sopra.' : 'Anti-rumore muto: premi «Attiva l’anti-rumore».');
      else if (s.status === 'starting') state.set('warn', 'Soft-start in corso…');
      else if (s.bypass) state.set('info', 'Bypass: anti-rumore spento per il confronto (B per riattivarlo).');
      else state.set('ok', 'Anti-rumore attivo.');

      if (errorMicOn(s) && s.ch2 === 'ok') {
        const a = s.attenuationDb;
        setText(big, a !== null ? fmtDb(a) : '—');
        big.dataset['state'] = a === null ? 'none' : a >= 3 ? 'good' : a < -1 ? 'worse' : 'none';
        setText(
          bigLabel,
          a !== null
            ? `Attenuazione misurata dal mic d’errore (ANC acceso rispetto al bypass). ${a < 0 ? 'Negativa = il rumore aumenta.' : ''}`
            : 'Attenuazione: servono alcuni secondi con l’anti-rumore acceso e qualche finestra di bypass (B).',
        );
      } else {
        setText(big, '—');
        big.dataset['state'] = 'none';
        setText(bigLabel, 'Senza mic d’errore l’attenuazione non è misurabile: ascolta nella zona alternando con il bypass (B).');
      }
      const showBands = errorMicOn(s) && s.ch2 === 'ok';
      bandsTitle.hidden = bandsHelp.hidden = !showBands;
      setListItems(bands, showBands ? bandTexts : []);
      const showHarm = showBands && s.params.phaseMode === 'tonal' && harmTexts.length > 0;
      harmTitle.hidden = harmHelp.hidden = !showHarm;
      setListItems(harmList, showHarm ? harmTexts : []);
      if (alarm) alarmNote.set('error', alarm);
      else alarmNote.clear();

      const pm = s.presets.message;
      if (pm) saveNote.set(pm.level === 'info' ? 'ok' : pm.level, pm.text);
      else saveNote.clear();
      saveBtn.disabled = s.measuring || !s.hasCalibration;
      finishBtn.disabled = list.length > 0;
      finishBtn.title = list.length > 0 ? 'Risolvi prima i punti elencati.' : '';
    },
    leave() {
      off();
      offSpectra();
    },
  };
}
