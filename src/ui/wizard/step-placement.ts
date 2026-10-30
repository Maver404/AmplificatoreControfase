/**
 * Step 4 — placement: distances mic → zone and speaker → zone per channel, source angle θ and
 * temperature, with a live causality check (acoustic head start vs measured/estimated latency),
 * the cos θ < 0.7 warning and the tonal-mode suggestion. Measurements and alignments keep the
 * distances they were made with: after an edit the gates of steps 5–7 ask to repeat them (going
 * back to the same numbers makes them valid again).
 */
import type { ControllerState } from '../../app/controller.ts';
import { defaultSweepBand } from '../../audio/calibration.ts';
import { formatHz, formatNumber } from '../units.ts';
import { button, el, liveNote, note, numberField, setText } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { solutionStale, stalePaths } from './flow.ts';
import { activeChannels, causality, CH_NAME, conditioningDelayS, estimatedLatencyS, fmtM, fmtMs, parseGeometry } from './logic.ts';

export function createPlacementStep(env: StepEnv): StepView {
  const { controller, data } = env;

  const changed = (): void => {
    env.savePrefs();
    env.refresh();
  };
  const mic = numberField('Distanza microfono di riferimento → zona', 'm', (t) => {
    data.geometry.micToZoneM = t;
    changed();
  });
  const spk = ([0, 1] as const).map((ch) =>
    numberField(`Distanza cassa ${CH_NAME[ch]} → zona`, 'm', (t) => {
      data.geometry.speakerToZoneM[ch] = t;
      changed();
    }),
  );
  const angle = numberField('Angolo θ della sorgente', '°', (t) => {
    data.geometry.sourceAngleDeg = t;
    changed();
  });
  const temp = numberField('Temperatura dell’aria', '°C', (t) => {
    data.geometry.tempC = t;
    changed();
  });
  mic.input.value = data.geometry.micToZoneM;
  spk.forEach((f, ch) => (f.input.value = data.geometry.speakerToZoneM[ch as 0 | 1]));
  angle.input.value = data.geometry.sourceAngleDeg;
  temp.input.value = data.geometry.tempC;

  const errors = el('ul', 'wz-list wz-errors');
  const staleNote = liveNote();
  const latencyLine = el('p', 'fp-help');
  const report = liveNote();
  const angleWarn = liveNote();
  const tonal = liveNote();
  const tonalBtn = button('Passa alla modalità Tonale', () => {
    const s = controller.state;
    const f0 = s.f0;
    if (!f0) return;
    controller.applyParams({ ...s.params, phaseMode: 'tonal', tonal: { ...s.params.tonal, f0: f0.f0 } });
    data.solution = null;
    env.refresh();
  });
  const tonalRow = el('div', 'wz-actions', tonalBtn);

  const element = el(
    'div',
    'wz-step-body',
    el(
      'p',
      'wz-lead',
      'Misura le distanze con un metro (bastano pochi centimetri di precisione). Servono a capire se il rumore arriva al microfono ' +
        'abbastanza prima che alla zona: l’anti-rumore deve attraversare computer, amplificatore e aria prima del rumore stesso.',
    ),
    el('div', 'wz-grid', mic.root, ...spk.map((f) => f.root), angle.root, temp.root),
    el(
      'p',
      'fp-help',
      'θ è l’angolo tra la direzione da cui arriva il rumore al microfono e la linea microfono → zona: 0° = sorgente, microfono e zona in fila; ' +
        '90° = la sorgente è di lato. La zona è il punto in cui vuoi il silenzio (per esempio all’altezza della testa).',
    ),
    errors,
    staleNote.root,
    latencyLine,
    report.root,
    angleWarn.root,
    tonal.root,
    tonalRow,
    note(
      'info',
      'Per un rumore periodico e stabile (motore, compressore, trasformatore) la modalità Tonale funziona anche senza vantaggio acustico: ' +
        'allinea l’anti-rumore su periodi interi del tono.',
    ),
  );

  return {
    element,
    update(s: ControllerState) {
      const chans = activeChannels(s.speakers);
      spk[1]!.root.hidden = !chans.includes(1);
      const g = parseGeometry(data.geometry, chans);
      const errs = g.ok ? [] : g.errors;
      const key = errs.join('|');
      if (errors.dataset['key'] !== key) {
        errors.dataset['key'] = key;
        errors.replaceChildren(...errs.map((t) => el('li', null, t)));
      }
      mic.setInvalid(data.geometry.micToZoneM !== '' && errs.some((t) => t.startsWith('Distanza microfono')));
      for (const ch of [0, 1] as const) {
        spk[ch]!.setInvalid(data.geometry.speakerToZoneM[ch] !== '' && errs.some((t) => t.startsWith(`Distanza cassa ${CH_NAME[ch]}`)));
      }
      angle.setInvalid(errs.some((t) => t.startsWith('Angolo')));
      temp.setInvalid(errs.some((t) => t.startsWith('Temperatura')));
      const stale = stalePaths(s, data);
      const staleSolution = data.solution?.applied === true && solutionStale(s, data);
      if (stale.length > 0 || staleSolution) {
        staleNote.set(
          'warn',
          (stale.length > 0 ? `Le misure fatte con le distanze precedenti (${stale.join(', ')}) vanno ripetute al passo 5. ` : '') +
            (staleSolution ? 'L’allineamento va ricalcolato al passo 6. ' : '') +
            'Se hai solo corretto un errore di battitura, rimetti i valori di prima.',
        );
      } else staleNote.clear();

      const fs = s.fs ?? 48000;
      const loop = data.calibration?.loopbackLatencyS ?? null;
      const est = estimatedLatencyS(s.stats);
      const lat = loop ?? est;
      setText(
        latencyLine,
        loop !== null
          ? `Latenza elettronica misurata con il loopback: ${fmtMs(loop)}.`
          : est !== null
            ? `Latenza elettronica stimata dal browser: ${fmtMs(est)} (verrà misurata al passo 5).`
            : 'Latenza non ancora disponibile: apri il microfono e abilita l’uscita (passo 1).',
      );
      if (g.ok && lat !== null) {
        const v = g.value;
        const r = causality({
          micToZoneM: v.micToZoneM,
          sourceAngleDeg: v.sourceAngleDeg,
          speakerToZoneM: Math.max(...chans.map((ch) => v.speakerToZoneM[ch])),
          tempC: v.tempC,
          latencyS: lat,
          conditioningS: conditioningDelayS(s.params, fs, defaultSweepBand(s.params, fs)),
        });
        const head = `Il rumore impiega ${fmtMs(r.headStartS)} dal microfono alla zona; l’anti-rumore ne richiede ${fmtMs(r.neededS)} (latenza, cassa → zona e filtri).`;
        if (r.causal) {
          report.set('ok', `${head} Margine ${fmtMs(r.marginS)}: la cancellazione a banda larga è possibile.`);
        } else {
          report.set(
            'error',
            `${head} Mancano ${fmtMs(-r.marginS)}: l’anti-rumore arriverebbe in ritardo sul rumore a banda larga. ` +
              `Sposta il microfono di riferimento di almeno ${fmtM(r.extraLeadM)} più verso la sorgente, avvicina la cassa alla zona, ` +
              'usa la latenza Ultra-bassa (passo 1, Avanzate) oppure, se il rumore è un tono stabile, la modalità Tonale.',
          );
        }
        if (r.warnAngle) {
          angleWarn.set(
            'warn',
            `cos θ = ${formatNumber(r.cosTheta, 2)} (sotto 0,7): la sorgente è troppo di lato rispetto alla linea microfono → zona, ` +
              'il vantaggio acustico dipende molto dalla direzione e la causalità non è garantita. Metti il microfono sulla linea sorgente → zona.',
          );
        } else angleWarn.clear();
      } else {
        report.clear();
        angleWarn.clear();
      }

      const f0 = s.f0;
      if (s.params.phaseMode === 'tonal') {
        tonal.set('info', `Modalità Tonale attiva (f0 = ${formatHz(s.params.tonal.f0)}).`);
        tonalRow.hidden = true;
      } else if (s.tonalSuggested && f0) {
        tonal.set(
          'info',
          `Il microfono sente un tono dominante a ${formatHz(f0.f0)} (${formatNumber(f0.dominanceDb, 0)} dB sopra il fondo): la modalità Tonale può cancellarlo meglio.`,
        );
        tonalRow.hidden = false;
      } else {
        tonal.clear();
        tonalRow.hidden = true;
      }
      tonalBtn.disabled = env.busy !== null;
    },
    leave() {},
  };
}
