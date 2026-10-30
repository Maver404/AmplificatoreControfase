/**
 * Step 3 — layout: A (recommended) or B with the guide's diagrams, speaker configuration (one
 * speaker / two speakers on the same zone / two separate zones), wall or window between mic and
 * zone (barrier filter), error mic yes/no.
 */
import type { ControllerState } from '../../app/controller.ts';
import { errorMicInput } from '../../app/validation.ts';
import type { SpeakerConfig } from '../../dsp/alignment.ts';
import { createGuidePanel } from '../guide-panel.ts';
import { el, liveNote, note, radioCards } from './dom.ts';
import type { StepEnv, StepView } from './env.ts';
import { resetMeasurements } from './flow.ts';
import type { LayoutChoice } from './logic.ts';

/** The guide's layout A/B figures, taken from a detached guide instance (unique SVG ids per instance). */
function layoutFigures(): HTMLElement[] {
  const host = document.createElement('div');
  const guide = createGuidePanel(host);
  const figs = [...host.querySelectorAll<HTMLElement>('[data-section="layout"] figure')];
  for (const f of figs) f.remove();
  guide.destroy();
  return figs;
}

/** Status of the error-mic input (input 2, or input 1 when the reference is on input 2). */
export function errorMicNote(ch2: ControllerState['ch2'], errorChannel: number): string {
  const n = errorMicInput(errorChannel);
  switch (ch2) {
    case 'pending':
      return `Verifica dell’ingresso ${n} in corso…`;
    case 'ok':
      return `Ingresso ${n} attivo: il mic d’errore è utilizzabile.`;
    case 'absent':
      return 'Il browser consegna un solo canale: il mic d’errore non è disponibile su questa interfaccia.';
    case 'silent':
      return `L’ingresso ${n} è muto: collega il mic d’errore (e accendi la phantom se serve) oppure scegli «No».`;
    case 'duplicate':
      return `L’ingresso ${n} duplica l’ingresso ${3 - n}: il mic d’errore non è utilizzabile.`;
  }
}

export function createLayoutStep(env: StepEnv): StepView {
  const { controller, data } = env;
  const figs = layoutFigures();

  const layout = radioCards<LayoutChoice>(
    'Disposizione',
    [
      { value: 'A', title: 'Layout A (consigliato)', text: 'Mic all’esterno verso la sorgente, cassa dall’altra parte vicino all’apertura, rivolta verso la zona.' },
      { value: 'B', title: 'Layout B (sconsigliato)', text: 'Cassa oltre la zona, rivolta verso il mic: retroazione forte e quiete solo a strati.' },
    ],
    (v) => {
      data.layout = v;
      env.savePrefs();
      env.refresh();
    },
  );
  const layoutWarn = liveNote();

  const speakers = radioCards<SpeakerConfig>(
    'Casse',
    [
      { value: 'single', title: 'Una cassa', text: 'Solo l’uscita L; l’uscita R resta muta.' },
      { value: 'shared', title: 'Due casse, stessa zona', text: 'Soluzione congiunta: le due casse insieme cancellano lo stesso rumore.' },
      { value: 'separate', title: 'Due zone separate', text: 'Ogni cassa ha la sua zona: ogni canale si misura con il microfono nella sua zona.' },
    ],
    (v) => {
      if (v !== controller.state.speakers) {
        controller.setSpeakers(v);
        resetMeasurements(data);
        env.refresh();
      }
    },
  );

  const barrier = radioCards<'yes' | 'no'>(
    'C’è una parete o una finestra tra il microfono di riferimento e la zona?',
    [
      { value: 'yes', title: 'Sì', text: 'Attiva la compensazione barriera (il rumore che attraversa la parete perde gli acuti).' },
      { value: 'no', title: 'No', text: 'Microfono e zona nello stesso ambiente o all’aperto.' },
    ],
    (v) => {
      const f = controller.state.params.filters;
      if ((v === 'yes') !== f.barrierEnabled) {
        controller.setFilters({ ...f, barrierEnabled: v === 'yes' });
        data.solution = null;
        env.refresh();
      }
    },
  );

  const errorMic = radioCards<'yes' | 'no'>(
    'Usi un microfono d’errore nella zona (ingresso 2, o 1 se il riferimento è sul 2)?',
    [
      { value: 'yes', title: 'Sì', text: 'Misura l’attenuazione reale, allineamento sul rumore vero e regolazione automatica.' },
      { value: 'no', title: 'No', text: 'Allineamento da geometria e calibrazione a due posizioni: risultato non misurabile.' },
    ],
    (v) => {
      const s = controller.state;
      const want = v === 'yes' ? (s.params.refChannel === 0 ? 1 : 0) : -1;
      if (want !== s.params.errorChannel) {
        controller.setErrorChannel(want);
        resetMeasurements(data);
        env.refresh();
      }
    },
  );
  const errNote = liveNote();

  const element = el(
    'div',
    'wz-step-body',
    el('p', 'wz-lead', 'Scegli come sono disposti microfono, casse e zona da silenziare.'),
    el('div', 'wz-figures', ...figs),
    layout.root,
    layoutWarn.root,
    speakers.root,
    barrier.root,
    errorMic.root,
    errNote.root,
    note('info', 'Puoi cambiare queste scelte anche più tardi: le misure dei passi 5 e 6 andranno ripetute.'),
  );

  return {
    element,
    update(s: ControllerState) {
      const busy = env.busy !== null;
      layout.set(data.layout);
      if (data.layout === 'B') {
        layoutWarn.set(
          'warn',
          'Layout B: la cassa punta verso il microfono di riferimento (rischio di innesco) e rumore e anti-rumore viaggiano in versi opposti: ' +
            'la quiete si forma solo su strati sottili distanti mezza lunghezza d’onda, con rinforzi in mezzo. Usalo solo se il layout A è impossibile.',
        );
      } else layoutWarn.clear();
      speakers.set(s.speakers);
      barrier.set(s.params.filters.barrierEnabled ? 'yes' : 'no');
      errorMic.set(s.params.errorChannel >= 0 ? 'yes' : 'no');
      for (const r of [layout, speakers, barrier, errorMic]) r.disable(busy || s.opening);
      if (s.params.errorChannel >= 0) {
        errNote.set(s.ch2 === 'ok' ? 'ok' : s.ch2 === 'pending' ? 'info' : 'warn', errorMicNote(s.ch2, s.params.errorChannel));
      }
      else errNote.set('info', 'Senza mic d’errore l’app non può misurare l’attenuazione: verificherai a orecchio.');
    },
    leave() {},
  };
}
