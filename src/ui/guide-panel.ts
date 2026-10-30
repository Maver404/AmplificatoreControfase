/**
 * In-app guide (Italian): how the app works, physical limits with numbers, layout A vs B
 * diagrams, placement, safety, macOS/Windows settings and troubleshooting.
 *
 * Static content built with DOM APIs only: no innerHTML and no inline style attributes (the
 * few presentational values go through CSSOM, which a strict Content-Security-Policy allows).
 * Diagram colours use currentColor plus --guide-* CSS variables with fallbacks, so they follow
 * the page theme and can be restyled from the stylesheet.
 */

type Child = Node | string | null | undefined | false;

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Diagram palette: override any --guide-* variable in CSS; fallbacks work on light and dark. */
const COLOR = {
  noise: 'var(--guide-noise, #e8590c)',
  anti: 'var(--guide-anti, #1c7ed6)',
  mic: 'var(--guide-mic, #2f9e44)',
  errMic: 'var(--guide-err-mic, #ae3ec9)',
  zoneFill: 'var(--guide-zone, rgba(64, 192, 87, 0.16))',
  zoneStroke: 'var(--guide-zone-stroke, #2f9e44)',
  glass: 'var(--guide-glass, rgba(116, 192, 252, 0.35))',
  warn: 'var(--guide-warn, #e03131)',
} as const;

let instanceCounter = 0;

// ---------------------------------------------------------------------------------------------
// DOM helpers

function appendChildren(parent: Node, children: readonly Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    parent.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
}

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> | null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  appendChildren(el, children);
  return el;
}

const p = (...c: Child[]) => h('p', null, ...c);
const strong = (t: string) => h('strong', null, t);
const em = (t: string) => h('em', null, t);
const kbd = (t: string) => h('kbd', null, t);
const ul = (...items: Child[][]) => h('ul', null, ...items.map((i) => h('li', null, ...i)));
const ol = (...items: Child[][]) => h('ol', null, ...items.map((i) => h('li', null, ...i)));

function note(kind: 'info' | 'warning', ...c: Child[]): HTMLParagraphElement {
  const el = h('p', { class: `guide-note guide-note-${kind}` }, ...c);
  if (kind === 'warning') el.setAttribute('role', 'note');
  return el;
}

function table(caption: string, head: string[], rows: Child[][]): HTMLTableElement {
  return h(
    'table',
    { class: 'guide-table' },
    h('caption', null, caption),
    h('thead', null, h('tr', null, ...head.map((t) => h('th', { scope: 'col' }, t)))),
    h('tbody', null, ...rows.map((r) => h('tr', null, ...r.map((cell) => h('td', null, cell))))),
  );
}

function section(id: string, title: string, open: boolean, ...body: Child[]): HTMLDetailsElement {
  const d = h(
    'details',
    { class: 'guide-section', 'data-section': id },
    h('summary', null, title),
    h('div', { class: 'guide-section-body' }, ...body),
  );
  d.open = open;
  return d;
}

// ---------------------------------------------------------------------------------------------
// SVG helpers

type SvgAttrs = Record<string, string | number>;
type SvgStyle = Record<string, string>;

function s<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: SvgAttrs,
  style?: SvgStyle | null,
  ...children: (SVGElement | string)[]
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  if (style) for (const [k, v] of Object.entries(style)) el.style.setProperty(k, v);
  for (const c of children) el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  return el;
}

function label(x: number, y: number, text: string, size = 13, anchor = 'middle', weight = 'normal'): SVGTextElement {
  return s('text', {
    x,
    y,
    'text-anchor': anchor,
    'font-size': size,
    'font-weight': weight,
    fill: 'currentColor',
  }, null, text);
}

interface DiagramIds {
  noise: string;
  anti: string;
  warn: string;
  hatch: string;
  clip: string;
}

function defs(uid: string): { el: SVGDefsElement; ids: DiagramIds } {
  const ids: DiagramIds = {
    noise: `${uid}-arrow-noise`,
    anti: `${uid}-arrow-anti`,
    warn: `${uid}-arrow-warn`,
    hatch: `${uid}-hatch`,
    clip: `${uid}-zone-clip`,
  };
  const marker = (id: string, color: string) =>
    s('marker', {
      id,
      viewBox: '0 0 10 10',
      refX: 8,
      refY: 5,
      markerWidth: 11,
      markerHeight: 11,
      markerUnits: 'userSpaceOnUse',
      orient: 'auto-start-reverse',
    }, null, s('path', { d: 'M0,0 L10,5 L0,10 z' }, { fill: color }));
  const hatch = s('pattern', {
    id: ids.hatch,
    width: 7,
    height: 7,
    patternUnits: 'userSpaceOnUse',
    patternTransform: 'rotate(45)',
  }, null, s('line', { x1: 0, y1: 0, x2: 0, y2: 7, stroke: 'currentColor', 'stroke-width': 2.5, opacity: 0.55 }));
  const el = s('defs', {}, null, marker(ids.noise, COLOR.noise), marker(ids.anti, COLOR.anti),
    marker(ids.warn, COLOR.warn), hatch);
  return { el, ids };
}

function arrow(x1: number, y1: number, x2: number, y2: number, color: string, markerId: string,
  opts: { dashed?: boolean; width?: number; opacity?: number } = {}): SVGLineElement {
  const attrs: SvgAttrs = {
    x1, y1, x2, y2,
    'stroke-width': opts.width ?? 3,
    'stroke-linecap': 'round',
    'marker-end': `url(#${markerId})`,
  };
  if (opts.dashed) attrs['stroke-dasharray'] = '8 6';
  if (opts.opacity !== undefined) attrs.opacity = opts.opacity;
  return s('line', attrs, { stroke: color });
}

function sourceGlyph(x: number, y: number): SVGGElement {
  const wave = (r: number) =>
    s('path', {
      d: `M ${x + r * Math.cos(-0.7)} ${y + r * Math.sin(-0.7)} A ${r} ${r} 0 0 1 ${x + r * Math.cos(0.7)} ${y + r * Math.sin(0.7)}`,
      fill: 'none',
      'stroke-width': 2.5,
      'stroke-linecap': 'round',
    }, { stroke: COLOR.noise });
  return s('g', {}, null,
    s('circle', { cx: x, cy: y, r: 16 }, { fill: COLOR.noise }),
    wave(24), wave(32),
    label(x, y + 40, 'Sorgente'),
    label(x, y + 56, 'del rumore'));
}

/** Microphone glyph; `dir` = -1 aims the capsule to the left, +1 to the right. */
function micGlyph(x: number, y: number, color: string, dir: -1 | 1): SVGGElement {
  const bodyX = dir < 0 ? x : x - 26;
  return s('g', {}, null,
    s('rect', { x: bodyX, y: y - 4, width: 26, height: 8, rx: 3, opacity: 0.75 }, { fill: color }),
    s('circle', { cx: x, cy: y, r: 9, stroke: 'currentColor', 'stroke-width': 1 }, { fill: color }));
}

/** Loudspeaker glyph; `dir` = +1 radiates to the right, -1 to the left. */
function speakerGlyph(x: number, y: number, dir: -1 | 1): SVGGElement {
  const front = x + dir * 9;
  const mouth = x + dir * 26;
  return s('g', {}, null,
    s('rect', { x: x - 9, y: y - 20, width: 18, height: 40, rx: 2, stroke: 'currentColor', 'stroke-width': 1.5 },
      { fill: COLOR.anti }),
    s('polygon', {
      points: `${front},${y - 10} ${mouth},${y - 24} ${mouth},${y + 24} ${front},${y + 10}`,
      stroke: 'currentColor',
      'stroke-width': 1.5,
      'stroke-linejoin': 'round',
      opacity: 0.85,
    }, { fill: COLOR.anti }));
}

function wallGlyph(x: number, top: number, bottom: number, hatchId: string): SVGGElement {
  const winTop = top + 56;
  const winBottom = top + 116;
  return s('g', {}, null,
    s('rect', { x, y: top, width: 14, height: bottom - top, fill: `url(#${hatchId})`, stroke: 'currentColor', 'stroke-width': 1.5 }),
    s('rect', { x, y: winTop, width: 14, height: winBottom - winTop, stroke: 'currentColor', 'stroke-width': 1.5 },
      { fill: COLOR.glass }),
    label(x + 7, bottom + 18, 'Parete / finestra', 12));
}

function svgRoot(uid: string, title: string, desc: string, ...children: SVGElement[]): SVGSVGElement {
  const titleId = `${uid}-title`;
  const descId = `${uid}-desc`;
  const svg = s('svg', {
    viewBox: '0 0 720 240',
    width: '100%',
    role: 'img',
    'aria-labelledby': `${titleId} ${descId}`,
    class: 'guide-diagram',
  }, { display: 'block', 'max-width': '720px', height: 'auto' },
  s('title', { id: titleId }, null, title),
  s('desc', { id: descId }, null, desc),
  ...children);
  return svg;
}

function layoutADiagram(uid: string): SVGSVGElement {
  const { el, ids } = defs(uid);
  return svgRoot(uid,
    'Layout A (consigliato)',
    'Da sinistra: sorgente del rumore, microfono di riferimento rivolto verso la sorgente, parete o ' +
      'finestra, cassa subito dentro rivolta verso destra, zona silenziata con il microfono d\'errore. ' +
      'Rumore e anti-rumore viaggiano entrambi verso destra.',
    el,
    // Propagation of the noise: outside, then (attenuated) through the wall into the room.
    arrow(80, 80, 250, 80, COLOR.noise, ids.noise),
    arrow(288, 80, 468, 80, COLOR.noise, ids.noise, { width: 2, opacity: 0.6 }),
    label(165, 70, 'rumore', 12),
    sourceGlyph(52, 120),
    micGlyph(175, 120, COLOR.mic, -1),
    label(187, 150, 'Mic di riferimento', 12),
    wallGlyph(262, 34, 204, ids.hatch),
    s('ellipse', { cx: 552, cy: 125, rx: 115, ry: 70, 'stroke-width': 2, 'stroke-dasharray': '6 5' },
      { fill: COLOR.zoneFill, stroke: COLOR.zoneStroke }),
    label(552, 46, 'Zona silenziata', 13, 'middle', 'bold'),
    speakerGlyph(306, 128, 1),
    label(310, 178, 'Cassa', 12),
    arrow(340, 128, 470, 128, COLOR.anti, ids.anti, { dashed: true }),
    label(395, 117, 'anti-rumore', 12),
    micGlyph(560, 132, COLOR.errMic, -1),
    label(572, 164, 'Mic d\'errore (opz.)', 12),
    label(552, 228, 'stessa direzione di propagazione →', 12),
  );
}

function layoutBDiagram(uid: string): SVGSVGElement {
  const { el, ids } = defs(uid);
  const cx = 440;
  const cy = 128;
  const rx = 105;
  const ry = 66;
  // Quiet layers: thin bands λ/2 apart, clipped to the zone.
  const layers = s('g', { 'clip-path': `url(#${ids.clip})` }, null,
    ...[360, 400, 440, 480, 520].map((x) =>
      s('rect', { x: x - 3, y: cy - ry, width: 6, height: 2 * ry, opacity: 0.45 }, { fill: COLOR.zoneStroke })));
  el.appendChild(s('clipPath', { id: ids.clip }, null, s('ellipse', { cx, cy, rx, ry })));
  return svgRoot(uid,
    'Layout B (sconsigliato)',
    'Da sinistra: sorgente, microfono di riferimento, parete o finestra, zona silenziata con strati ' +
      'di quiete sottili e microfono d\'errore, cassa oltre la zona rivolta verso sinistra, cioè verso ' +
      'il microfono. Rumore verso destra, anti-rumore verso sinistra; una freccia indica la retroazione ' +
      'dalla cassa al microfono di riferimento.',
    el,
    arrow(80, 80, 250, 80, COLOR.noise, ids.noise),
    arrow(288, 80, 575, 80, COLOR.noise, ids.noise, { width: 2, opacity: 0.6 }),
    label(165, 70, 'rumore', 12),
    sourceGlyph(52, 120),
    micGlyph(175, 120, COLOR.mic, -1),
    label(187, 150, 'Mic di riferimento', 12),
    wallGlyph(262, 34, 204, ids.hatch),
    s('ellipse', { cx, cy, rx, ry, 'stroke-width': 2, 'stroke-dasharray': '6 5' },
      { fill: COLOR.zoneFill, stroke: COLOR.zoneStroke }),
    layers,
    label(cx, 54, 'Zona silenziata', 13, 'middle', 'bold'),
    label(cx, 214, 'solo strati di quiete sottili, distanti λ/2', 12),
    micGlyph(cx, 158, COLOR.errMic, -1),
    label(cx + 10, 184, 'Mic d\'errore', 12),
    speakerGlyph(645, 128, -1),
    label(645, 178, 'Cassa', 12),
    arrow(612, 128, 352, 128, COLOR.anti, ids.anti, { dashed: true }),
    label(560, 117, 'anti-rumore', 12),
    // Feedback path: the speaker points straight at the reference mic.
    s('path', {
      d: 'M 632 104 C 560 10, 270 10, 194 106',
      fill: 'none',
      'stroke-width': 2,
      'stroke-dasharray': '4 5',
      'marker-end': `url(#${ids.warn})`,
    }, { stroke: COLOR.warn }),
    label(412, 22, 'retroazione cassa → mic: rischio di innesco', 12),
  );
}

function figure(svg: SVGSVGElement, caption: string): HTMLElement {
  return h('figure', { class: 'guide-figure' }, svg, h('figcaption', null, caption));
}

// ---------------------------------------------------------------------------------------------
// Content

function howItWorks(): HTMLDetailsElement {
  return section('come-funziona', 'Come funziona', true,
    p('Il ', strong('microfono di riferimento'), ', all\'esterno e rivolto verso la sorgente, capta il ' +
      'rumore prima che arrivi alla zona da silenziare. L\'app lo filtra, lo ritarda della quantità ' +
      'giusta, ne inverte la polarità e lo invia all\'amplificatore: nella zona l\'', strong('anti-rumore'),
    ' emesso dalla cassa si somma al rumore e, se arriva con la stessa ampiezza e in opposizione di ' +
      'fase, lo attenua.'),
    ol(
      [strong('Condizionamento: '), 'blocco della continua, passa-alto di protezione (vento e rombo, ' +
        'default 20 Hz), compensazione barriera opzionale (parete/finestra), passa-basso Bessel (default 500 Hz).'],
      [strong('Solo in modalità tonale: '), 'un banco di risonatori sulle armoniche di f0 riproduce solo il ' +
        'tono, non il rumore a banda larga; ogni uscita può correggere guadagno e fase di ciascuna armonica.'],
      [strong('Per ogni uscita L/R: '), 'ritardo frazionario (precisione sotto il campione), polarità, ' +
        'guadagno, Δ di fase, muto e avvio graduale.'],
      [strong('Uscita: '), 'volume master e limitatore con tetto (ceiling) in dBFS; il rilevatore di ' +
        'innesco osserva il segnale prima del limitatore.'],
    ),
    p(strong('Fase totale = polarità + Δ'), ': la polarità vale 0° o 180° (default 180°, cioè ' +
      'controfase); Δ va da −180° a +180°, positivo = ritardo, ed è riferita alla frequenza f_ref ' +
      '(a f0 in modalità tonale).'),
    table('Modalità di fase', ['Modalità', 'Quando usarla', 'Come agisce Δ'], [
      [strong('Ritardo (default)'), 'Qualunque rumore, banda larga compresa.',
        'Ritardo aggiuntivo Δ/(360°·f_ref); nessuna dispersione.'],
      [strong('Tonale'), 'Rumori periodici stabili: motori, compressori, trasformatori (l\'app la ' +
        'suggerisce quando rileva un tono dominante).',
      'Il ritardo si avvolge su periodi interi 1/f0, quindi funziona anche se il rumore arriva ' +
        '"in ritardo"; f0 viene stimata e inseguita. All\'armonica h Δ vale h·Δ, più la correzione ' +
        'di quell\'armonica.'],
      [strong('Hilbert (avanzata)'), 'Ottimizzazione con il mic d\'errore e la regolazione automatica.',
        'Rotazione di fase a larga banda; aggiunge ritardo di gruppo (≈ 2,6 ms a 100 Hz).'],
    ]),
    p(strong('Correzioni per armonica (solo modalità tonale). '), 'Un unico ritardo sposta la fase di ogni ' +
      'armonica h di h·Δ: basta solo se filtri, casse e stanza non aggiungono altro, e di solito non è così ' +
      '(per esempio la fondamentale si cancella e la 4ª armonica molto meno). Per questo ogni uscita ha, per ' +
      'ogni armonica, una correzione di guadagno (−30…+6 dB) e di fase (−180…+180°, positivo = ritardo) che ' +
      'si aggiunge al ritardo: fase totale all\'armonica h = polarità + h·Δ + θ_h. Le calcola il passo 6 ' +
      'della configurazione guidata insieme a ritardo, guadagno e polarità, e vengono applicate con la soluzione; ' +
      'con il mic d\'errore l\'autotune del passo 6 le affina misurando ogni armonica. ' +
      'Le vedi e le ritocchi nella sezione «Armoniche» di ogni canale (doppio clic sul nome = valore calibrato, ' +
      '«Azzera armoniche» = nessuna correzione; con «Collega L/R» azzera entrambi i canali). Come per i guadagni, ' +
      'un aumento di oltre 6 dB sale a 6 dB al secondo (da −30 a +6 dB servono 5–6 s), e il limite vale per ' +
      'guadagno, master e correzione insieme. Nell\'oscilloscopio la traccia attesa non le include.'),
    p('Il ', strong('mic d\'errore'), ' (facoltativo, ingresso 2, o 1 se il riferimento è sul 2) sta nella zona: misura l\'attenuazione ' +
      'reale (confronto A/B con il bypass) e permette la regolazione automatica. Senza, l\'app usa la ' +
      'geometria e una calibrazione a due posizioni con l\'unico microfono, ma il risultato non è misurabile.'),
    p('Ritardi e latenze vengono sempre ', strong('misurati'), ' (sweep), mai presi da valori fissi.'),
  );
}

function physicalLimits(): HTMLDetailsElement {
  return section('limiti-fisici', 'Limiti fisici (con numeri)', false,
    table('Latenza e vantaggio acustico', ['Grandezza', 'Valore tipico', 'Conseguenza'], [
      ['Latenza mic → cassa, macOS + interfaccia USB + Chrome', '≈ 15–30 ms', 'Deve essere coperta dal vantaggio acustico.'],
      ['Latenza su Windows (WASAPI condiviso)', 'più alta (≈ 10 ms in più per direzione)', 'Serve più distanza; per i toni usa la modalità tonale.'],
      ['Impostazioni di default del browser', '55–120 ms', 'Usa la latenza Normale o Ultra-bassa dell\'app.'],
      ['Bluetooth / AirPlay', '+150–275 ms', 'ANC impossibile: questi dispositivi sono bloccati.'],
      ['Vantaggio richiesto (rumore a banda larga)', '≈ 3,4 m ogni 10 ms',
        'Es. 25 ms + cassa a 1 m dalla zona (≈ 3 ms): il mic deve stare ≈ 9,6 m più vicino alla sorgente rispetto alla zona.'],
    ]),
    p(strong('Causalità (banda larga). '), 'Il rumore deve impiegare, per andare dal mic alla zona, ' +
      'almeno il tempo della catena mic → computer → cassa → zona: (d sorgente→zona − d sorgente→mic)/c ≥ ' +
      'latenza + percorso cassa→zona. Se non basta, l\'app lo segnala e indica di quanto allontanare il mic.'),
    p(strong('Rumore periodico e stabile. '), 'Si allinea su periodi interi, quindi funziona a qualunque ' +
      'distanza, purché f0 sia stabile e stimata con precisione (≈ 0,01 %). Esempio: con 0,1 Hz di deriva ' +
      'e 100 ms di ritardo totale l\'errore è 3,6° alla fondamentale e 14,4° alla 4ª armonica; i risonatori ' +
      'della modalità tonale aggiungono ≈ atan(2Q·Δf/f0) a ogni armonica (con Q 30 a 100 Hz altri 3,4° per 0,1 Hz).'),
    table('Tolleranze di allineamento', ['Attenuazione voluta', 'Errore di fase max', 'Errore di ampiezza max', 'Equivale a 100 Hz'], [
      ['10 dB', '18°', '−3,3 / +2,4 dB', '± 0,5 ms'],
      ['20 dB', '5,7°', '≈ ± 0,8 dB', '± 159 µs'],
    ]),
    p('Residuo: A = 10·log10(1 + g² − 2g·cos Δφ), con g rapporto di ampiezza e Δφ errore di fase. ' +
      'Per questo il ritardo è regolabile in frazioni di campione.'),
    p(strong('Zona di quiete locale, ≈ λ/10 '), 'attorno al punto ottimizzato: 50 Hz ≈ 69 cm · ' +
      '100 Hz ≈ 34 cm · 200 Hz ≈ 17 cm · 500 Hz ≈ 7 cm.'),
    p(strong('Risultato realistico: 10–20 dB '), 'su circa un\'ottava sotto ≈ 500 Hz, oppure sui toni. ' +
      'Sopra ≈ 500 Hz la zona si riduce a pochi centimetri: per questo il passa-basso di default è a 500 Hz.'),
    p(strong('Parete o finestra tra mic e zona: '), 'il rumore che la attraversa viene filtrato (perde ' +
      'gli acuti e ruota di fase), non solo ritardato, e l\'attenuazione ottenibile cala. La compensazione ' +
      'barriera (passa-basso del 1° ordine) ne imita solo l\'effetto principale.'),
    p(strong('Temperatura: '), 'c = 331,3·√(1 + T/273,15) m/s, cioè 343 m/s a 20 °C. 10 °C in meno ' +
      'allungano di ≈ 0,5 ms un percorso di 10 m (≈ 18° a 100 Hz): se la temperatura cambia molto, ricalibra.'),
  );
}

function layouts(uid: string): HTMLDetailsElement {
  return section('layout', 'Layout A e B', false,
    figure(layoutADiagram(`${uid}-a`),
      'Layout A (consigliato): rumore e anti-rumore viaggiano nella stessa direzione; la parete separa la cassa dal mic di riferimento.'),
    ul(
      ['Mic ', strong('all\'esterno'), ', rivolto verso la sorgente (oltre la finestra o la parete).'],
      ['Cassa ', strong('all\'interno, vicino all\'apertura'), ', rivolta verso la zona da silenziare.'],
      ['Le due onde viaggiano nello stesso verso: la cancellazione resta coerente lungo il percorso.'],
      ['La parete attenua il ritorno dalla cassa al mic: meno retroazione e più guadagno utilizzabile.'],
    ),
    figure(layoutBDiagram(`${uid}-b`),
      'Layout B (sconsigliato): la cassa oltre la zona è rivolta verso il mic; l\'anti-rumore va incontro al rumore.'),
    ul(
      ['Cassa ', strong('oltre la zona'), ', rivolta verso il mic e la sorgente.'],
      ['Le onde viaggiano in versi opposti: si cancellano solo su strati sottili distanti λ/2 (a 100 Hz ' +
        'ogni ≈ 1,7 m), con rinforzi fino a +6 dB in mezzo.'],
      ['La cassa punta verso il mic: forte retroazione e rischio di innesco (larsen).'],
      ['Supportato con avvisi: usalo solo se il layout A è impossibile.'],
    ),
  );
}

function placement(): HTMLDetailsElement {
  return section('posizionamento', 'Posizionamento', false,
    ul(
      [strong('Mic di riferimento: '), 'il più a monte possibile, verso la sorgente. Direzionale ' +
        '(cardioide o supercardioide) puntato alla sorgente, con la cassa dietro, nella direzione di minima ' +
        'sensibilità. All\'aperto: antivento (spugna + pelliccia) e riparo dalla pioggia. Fissato saldamente.'],
      [strong('Allineamento: '), 'sorgente, mic e zona circa sulla stessa linea. Se l\'angolo θ tra la ' +
        'direzione della sorgente e la linea mic → zona supera 45° (cos θ < 0,7) il vantaggio acustico si ' +
        'riduce e l\'app avvisa.'],
      [strong('Cassa: '), 'vicino all\'apertura, rivolta verso la zona, capace di riprodurre le basse ' +
        'frequenze (sotto ≈ 80 Hz serve un subwoofer). Con due casse: stessa zona (soluzione congiunta, ' +
        '−6 dB ciascuna) oppure due zone separate, ognuna calibrata con il mic d\'errore nella sua zona.'],
      [strong('Mic d\'errore: '), 'nel punto in cui vuoi il silenzio (per esempio all\'altezza della testa), ' +
        'non a ridosso della cassa, e fermo durante misure e regolazioni.'],
      [strong('Misure: '), 'distanze (m) e temperatura servono all\'assistente geometria (passo 4), che ' +
        'verifica la causalità.'],
      [strong('Dopo ogni spostamento '), 'di mic, casse o amplificatore, o se cambi il volume dell\'ampli, ' +
        'ripeti la misura dei percorsi (passo 5).'],
    ),
  );
}

function safety(): HTMLDetailsElement {
  return section('sicurezza', 'Sicurezza', false,
    note('warning', strong('Prima di ogni avvio porta l\'amplificatore al minimo '), 'e tieni la mano sulla ' +
      'manopola del volume o sull\'interruttore finché non sei sicuro del comportamento.'),
    ul(
      [strong('PANIC'), ' (sempre visibile) o ', kbd('Esc'), ': rampa a zero in 10 ms e sospensione ' +
        'dell\'audio. ', kbd('Esc'), ' funziona solo se la finestra dell\'app ha il fuoco: se stai usando ' +
        'un\'altra finestra, agisci sull\'amplificatore.'],
      [kbd('M'), ' = muto, ', kbd('B'), ' = bypass A/B (confronto con l\'anti-rumore spento).'],
      ['L\'uscita resta ', strong('a zero'), ' finché i dispositivi non sono validati e la configurazione ' +
        'guidata non è completata; all\'attivazione il volume sale gradualmente (1,5 s).'],
      [strong('Cuffie: '), 'tetto d\'uscita forzato a −24 dBFS. ', strong('Mai Bluetooth o AirPlay'), '.'],
      [strong('Livelli: '), 'nella zona non superare ≈ 85 dBA. La calibrazione parte da un\'anteprima ' +
        '20 dB più bassa e si annulla con ', kbd('Esc'), '.'],
      [strong('Innesco (larsen): '), 'il rilevatore silenzia automaticamente e mostra la frequenza. Abbassa ' +
        'il guadagno, orienta il mic lontano dalla cassa, preferisci il layout A. Il rilevatore può scattare ' +
        'anche senza innesco con un tono che sale molto in fretta, oppure con un suono periodico (tono, macchina) ' +
        'che il limitatore deve ridurre di almeno 0,5 dB senza sosta.'],
      [strong('Cavo di loopback'), ' (misura opzionale della latenza): solo sull\'ingresso libero (il 2, o l\'1 se il riferimento è sul 2), alimentazione ' +
        'phantom (48 V) spenta, gain al minimo; scollegalo subito dopo. L\'app rifiuta di attivare l\'uscita ' +
        'se rileva ancora un loopback.'],
      ['Silenziamento automatico su cambio o scollegamento dei dispositivi, sospensione dell\'audio, ' +
        'errori del processore e interruzioni ripetute.'],
      ['Lo sblocco del guadagno a +12 dB non viene mai salvato né importato nei preset.'],
    ),
  );
}

function osSettings(): HTMLDetailsElement {
  return section('impostazioni-os', 'Impostazioni macOS e Windows', false,
    h('h4', null, 'macOS'),
    ol(
      ['Configurazione MIDI Audio (Audio MIDI Setup): seleziona l\'interfaccia e imposta il formato a ',
        strong('48.000 Hz'), ' sia in ingresso sia in uscita.'],
      ['Impostazioni di Sistema → Privacy e sicurezza → Microfono: abilita ', strong('Google Chrome'), '.'],
      ['In Chrome consenti il microfono al sito (icona a sinistra dell\'indirizzo).'],
      ['Nell\'app scegli la stessa interfaccia USB per ingresso e uscita: riferimento sull\'ingresso 1, ' +
        'mic d\'errore sull\'ingresso 2.'],
      ['Se compare "Modalità microfono", lascia ', em('Standard'), ' (niente isolamento della voce).'],
      ['Collega l\'alimentatore: il risparmio energetico può causare interruzioni dell\'audio.'],
    ),
    h('h4', null, 'Windows'),
    ol(
      ['Pannello di controllo Audio (', kbd('Win+R'), ' → mmsys.cpl): nelle schede ', strong('Riproduzione'),
        ' e ', strong('Registrazione'), ' apri l\'interfaccia → Proprietà → Avanzate → Formato predefinito: ',
        strong('lo stesso 48000 Hz'), ' su entrambe.'],
      ['Disattiva Miglioramenti audio (Audio enhancements), Audio spaziale (Spatial sound) ed ' +
        'Equalizzazione volume (Loudness Equalization).'],
      ['Scheda ', strong('Comunicazioni'), ': "Non eseguire alcuna operazione" (Do nothing).'],
      ['Impostazioni → Privacy e sicurezza → Microfono: consenti l\'accesso alle app desktop e a Chrome.'],
      ['Chrome usa WASAPI in modalità condivisa: ', strong('ASIO non viene usato'), ' e la latenza è più ' +
        'alta che su macOS. Alcuni driver hanno un buffer WDM separato da quello ASIO: impostalo al minimo stabile.'],
      ['Piano energetico "Prestazioni elevate"; evita hub USB alimentati male.'],
    ),
  );
}

function troubleshooting(): HTMLDetailsElement {
  const item = (q: string, ...a: Child[]): Child[] => [h('dt', null, q), h('dd', null, ...a)];
  return section('problemi', 'Risoluzione problemi', false,
    h('dl', { class: 'guide-faq' },
      ...item('Il microfono non viene concesso',
        'Controlla il permesso di sistema (macOS: Privacy e sicurezza → Microfono; Windows: Privacy → ' +
        'Microfono) e quello del sito in Chrome. Il microfono funziona solo in un contesto sicuro: ' +
        'https:// oppure http://localhost (un indirizzo IP in http non basta).'),
      ...item('Nessun livello in ingresso',
        'Ingresso sbagliato, gain dell\'interfaccia a zero o mic a condensatore senza phantom (48 V). ' +
        'Il riferimento deve stare sull\'ingresso 1.'),
      ...item('Mic d\'errore disattivato (canale 2 assente, muto o duplicato)',
        'Scegli il dispositivo che espone gli ingressi 1–2 come coppia stereo; su Windows alcune ' +
        'interfacce mostrano ogni coppia di ingressi come un dispositivo separato. Se il canale 2 è ' +
        'identico al canale 1 il driver sta duplicando un ingresso mono.'),
      ...item('Frequenze di campionamento diverse',
        'Imposta 48 kHz ovunque (Configurazione MIDI Audio o Formato predefinito di Windows) e riavvia Chrome.'),
      ...item('Latenza alta (oltre 40 ms)',
        'Usa la latenza Normale o Ultra-bassa, chiudi le altre app audio, niente Bluetooth; su Windows ' +
        'disattiva i miglioramenti audio e prova un\'altra porta USB.'),
      ...item('Crepitii o interruzioni',
        'Torna alla latenza Normale (128 campioni), chiudi le schede pesanti, collega l\'alimentatore. ' +
        'Più di 3 interruzioni in 10 s silenziano l\'uscita per sicurezza.'),
      ...item('Innesco o fischio',
        'Abbassa il guadagno, allontana o orienta il mic, usa un mic direzionale e il layout A. La ' +
        'neutralizzazione del feedback si applica automaticamente dopo la misura dei percorsi (lunghezza: ' +
        'Fase e filtri › Avanzate → Tap del neutralizzatore, 128–1024).'),
      ...item('"L\'ANC sta aumentando il rumore"',
        'Mic, casse o ampli sono stati spostati, oppure la temperatura è cambiata: ripeti la misura dei ' +
        'percorsi e l\'allineamento. Verifica che la polarità sia quella calibrata.'),
      ...item('Attenuazione scarsa',
        'Se l\'app segnala un problema di causalità allontana il mic verso la sorgente; per rumori ' +
        'periodici passa alla modalità tonale; restringi la banda (passa-basso più basso); con una ' +
        'parete in mezzo attiva la compensazione barriera.'),
      ...item('Silenziamento automatico improvviso',
        'Un dispositivo è cambiato o si è scollegato (cavo, hub USB, sospensione del computer): ' +
        'ricontrolla i dispositivi e riattiva l\'uscita.'),
    ),
  );
}

// ---------------------------------------------------------------------------------------------

export function createGuidePanel(container: HTMLElement): { destroy(): void } {
  const uid = `guide-${++instanceCounter}`;

  const expandAll = h('button', { type: 'button', class: 'btn btn--small guide-expand' }, 'Espandi tutto');
  const collapseAll = h('button', { type: 'button', class: 'btn btn--small guide-collapse' }, 'Comprimi tutto');
  const sections = [
    howItWorks(),
    physicalLimits(),
    layouts(uid),
    placement(),
    safety(),
    osSettings(),
    troubleshooting(),
  ];
  const root = h('section', { class: 'guide-panel', 'aria-labelledby': `${uid}-heading` },
    h('div', { class: 'guide-header' },
      h('h2', { id: `${uid}-heading` }, 'Guida'),
      h('div', { class: 'guide-actions' }, expandAll, ' ', collapseAll)),
    ...sections);

  const setAll = (open: boolean) => () => {
    for (const d of sections) d.open = open;
  };
  const onExpand = setAll(true);
  const onCollapse = setAll(false);
  expandAll.addEventListener('click', onExpand);
  collapseAll.addEventListener('click', onCollapse);
  container.appendChild(root);

  let destroyed = false;
  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      expandAll.removeEventListener('click', onExpand);
      collapseAll.removeEventListener('click', onCollapse);
      root.remove();
    },
  };
}
