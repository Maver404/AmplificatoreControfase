/**
 * Spectrum plot on a log-frequency axis (20 Hz – 4 kHz) of dB arrays (bin k = k·fs/fftSize):
 * reference, error with ANC, and an optional error spectrum averaged in bypass. Optional
 * 1/3-octave bars (power-summed bins). Drawing happens in requestAnimationFrame.
 * The attenuation it reports is the plan's A = [L_err − L_rif]_off − [L_err − L_rif]_on, with the
 * controller's reference spectra of each ANC state (refOn / refOff: x̂ while the feedback neutraliser
 * runs), so a source that got quieter since the last bypass does not read as attenuation.
 */
import { formatNumber, formatSigned } from './units.ts';

export const SPECTRUM_F_MIN = 20;
export const SPECTRUM_F_MAX = 4000;
const RANGE_DB = 90;

export type DbArray = Float32Array | Float64Array | readonly number[];

export interface SpectrumViewData {
  fs: number;
  fftSize: number;
  /** Arrays of fftSize/2 + 1 levels in dB (dBFS). Shorter arrays are read as far as they go. */
  ref: DbArray | null;
  err: DbArray | null;
  /** Error spectrum captured with ANC bypassed, for comparison. */
  errBypass: DbArray | null;
  /** Reference spectra averaged over the same windows as `err` / `errBypass` (L_rif of the attenuation). */
  refOn?: DbArray | null;
  refOff?: DbArray | null;
}

export interface SpectrumView {
  update(data: SpectrumViewData): void;
  destroy(): void;
}

/** Nominal 1/3-octave centre frequencies (base-10 series) within [fmin, fmax]. */
export function thirdOctaveCenters(fmin = SPECTRUM_F_MIN, fmax = SPECTRUM_F_MAX): number[] {
  const out: number[] = [];
  for (let n = -30; n <= 20; n++) {
    const fc = 1000 * Math.pow(10, n / 10);
    // Half a third-octave of tolerance: the base-10 series puts the nominal "20 Hz" band at 19.95 Hz.
    if (fc >= fmin * Math.pow(2, -1 / 12) && fc <= fmax * Math.pow(2, 1 / 12)) out.push(fc);
  }
  return out;
}

/** ISO 266 preferred numbers (R10): the nominal names of the base-10 third-octave centres. */
const R10 = [1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8];

/** Nominal name of the third-octave band centred at `fc` (base-10 series): "20 Hz", "31,5 Hz", "1,25 kHz". */
export function thirdOctaveLabel(fc: number): string {
  if (!(fc > 0)) return '—';
  const n = Math.round(10 * Math.log10(fc / 1000));
  const nominal = (R10[((n % 10) + 10) % 10] as number) * Math.pow(10, Math.floor(n / 10) + 3);
  const v = nominal >= 1000 ? nominal / 1000 : nominal;
  const rounded = Math.round(v * 100) / 100;
  const decimals = Number.isInteger(rounded) ? 0 : Number.isInteger(rounded * 10) ? 1 : 2;
  return `${formatNumber(rounded, decimals)} ${nominal >= 1000 ? 'kHz' : 'Hz'}`;
}

/**
 * Hover readout: the frequency, the levels, and the attenuation of the third-octave band centred at
 * `fc`. The third-octave view leads with the band; otherwise the attenuation says which band it is.
 */
export function spectrumReadout(f: number, levels: readonly string[], thirds: boolean, fc: number, attenuationDb: number | null): string {
  const band = thirdOctaveLabel(fc);
  const parts = [`${formatNumber(f, f < 100 ? 1 : 0)} Hz`, ...levels];
  if (attenuationDb !== null) parts.push(`attenuazione ${formatSigned(attenuationDb, 1)} dB${thirds ? '' : ` (banda ${band})`}`);
  return (thirds ? `Banda ${band} · ` : '') + parts.join(' · ');
}

/**
 * Power sum (in dB) of the bins whose centre falls in [fc·2^(−1/6), fc·2^(1/6)). A band narrower than
 * one bin takes the nearest bin. Returns −Infinity when the band is outside the array.
 */
export function bandLevelDb(db: DbArray, fs: number, fftSize: number, fc: number): number {
  const binHz = fs / fftSize;
  const lo = fc * Math.pow(2, -1 / 6);
  const hi = fc * Math.pow(2, 1 / 6);
  const n = db.length;
  let k0 = Math.ceil(lo / binHz);
  let k1 = Math.ceil(hi / binHz) - 1;
  if (k1 < k0) {
    k0 = Math.round(fc / binHz);
    k1 = k0;
  }
  k0 = Math.max(0, k0);
  k1 = Math.min(n - 1, k1);
  if (k1 < k0) return -Infinity;
  let p = 0;
  for (let k = k0; k <= k1; k++) {
    const v = db[k] as number;
    if (Number.isFinite(v)) p += Math.pow(10, v / 10);
  }
  return p > 0 ? 10 * Math.log10(p) : -Infinity;
}

/**
 * Attenuation of one third-octave band, A = [L_err − L_rif]_off − [L_err − L_rif]_on (dB, positive =
 * quieter with the anti-noise); null when a term is missing.
 */
export function bandAttenuationDb(
  s: { errOff: DbArray | null; refOff: DbArray | null; errOn: DbArray | null; refOn: DbArray | null },
  fs: number,
  fftSize: number,
  fc: number,
): number | null {
  if (!s.errOff || !s.refOff || !s.errOn || !s.refOn) return null;
  const v = [s.errOff, s.refOff, s.errOn, s.refOn].map((a) => bandLevelDb(a, fs, fftSize, fc));
  if (!v.every(Number.isFinite)) return null;
  return v[0]! - v[1]! - (v[2]! - v[3]!);
}

/** x position (0..1) of frequency f on the log axis. */
export const freqToX = (f: number, fmin = SPECTRUM_F_MIN, fmax = SPECTRUM_F_MAX): number =>
  Math.log(f / fmin) / Math.log(fmax / fmin);
export const xToFreq = (x: number, fmin = SPECTRUM_F_MIN, fmax = SPECTRUM_F_MAX): number =>
  fmin * Math.pow(fmax / fmin, x);

interface Palette {
  bg: string;
  grid: string;
  gridStrong: string;
  text: string;
  ref: string;
  err: string;
  bypass: string;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

const GRID_FREQS = [20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000];
const LABEL_FREQS = new Set([20, 50, 100, 200, 500, 1000, 2000, 4000]);
const fmtFreq = (f: number): string => (f >= 1000 ? `${formatNumber(f / 1000, f % 1000 ? 1 : 0)}k` : String(Math.round(f)));

export function createSpectrumView(container: HTMLElement): SpectrumView {
  let data: SpectrumViewData | null = null;
  let thirds = false;
  let topDb = 0;
  let raf = 0;
  let dirty = true;
  let cssW = 0;
  let cssH = 0;
  let palette: Palette | null = null;
  let hoverX: number | null = null;
  const centers = thirdOctaveCenters();
  const bandRef = new Float64Array(centers.length);
  const bandErr = new Float64Array(centers.length);
  const bandByp = new Float64Array(centers.length);

  const root = el('div', 'spectrum-view');
  const toolbar = el('div', 'view-toolbar');
  const thirdsLabel = el('label', 'tb-field tb-check');
  const thirdsBox = el('input');
  thirdsBox.type = 'checkbox';
  thirdsLabel.append(thirdsBox, ' Terzi d’ottava');
  toolbar.append(thirdsLabel);
  const legend = el('ul', 'legend');
  for (const [cls, text] of [
    ['legend-ref', 'Riferimento'],
    ['legend-err', 'Errore, ANC attivo'],
    ['legend-bypass', 'Errore, bypass'],
  ] as const) {
    const li = el('li', cls);
    li.append(el('span', 'legend-swatch'), text);
    legend.append(li);
  }
  const wrap = el('div', 'canvas-wrap canvas-wrap--spectrum');
  const canvas = el('canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Spettro da 20 Hz a 4 kHz: riferimento, errore con ANC attivo ed errore in bypass');
  wrap.append(canvas);
  const readout = el('p', 'view-readout num');
  readout.textContent = 'Passa il puntatore sul grafico per leggere i livelli.';
  root.append(toolbar, legend, wrap, readout);
  container.append(root);
  const ctx = canvas.getContext('2d');

  thirdsBox.addEventListener('change', () => {
    thirds = thirdsBox.checked;
    requestDraw();
  });
  canvas.addEventListener('pointermove', (e) => {
    const rect = canvas.getBoundingClientRect();
    hoverX = e.clientX - rect.left;
    updateReadout();
    requestDraw();
  });
  canvas.addEventListener('pointerleave', () => {
    hoverX = null;
    requestDraw();
  });

  const ro = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const box = entry.devicePixelContentBoxSize?.[0];
      cssW = entry.contentRect.width;
      cssH = entry.contentRect.height;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, box ? box.inlineSize : Math.round(cssW * dpr));
      canvas.height = Math.max(1, box ? box.blockSize : Math.round(cssH * dpr));
    }
    requestDraw();
  });
  try {
    ro.observe(canvas, { box: 'device-pixel-content-box' });
  } catch {
    ro.observe(canvas);
  }
  const invalidatePalette = (): void => {
    palette = null;
    requestDraw();
  };
  const mo = new MutationObserver(invalidatePalette);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener('change', invalidatePalette);

  function readPalette(): Palette {
    const cs = getComputedStyle(canvas);
    const v = (name: string, fallback: string): string => cs.getPropertyValue(name).trim() || fallback;
    return {
      bg: v('--canvas-bg', '#101418'),
      grid: v('--grid', 'rgba(127,127,127,0.18)'),
      gridStrong: v('--grid-strong', 'rgba(127,127,127,0.4)'),
      text: v('--axis-text', '#8a96a3'),
      ref: v('--trace-ref', '#8b9bb0'),
      err: v('--trace-err', '#e5484d'),
      bypass: v('--trace-bypass', '#f5a524'),
    };
  }

  function requestDraw(): void {
    dirty = true;
    if (!raf) raf = window.requestAnimationFrame(frame);
  }
  function frame(): void {
    raf = 0;
    if (!dirty || !ctx || cssW < 2 || cssH < 2 || document.hidden) return;
    dirty = false;
    draw(ctx);
  }

  /** Level at frequency f: interpolated between bins, or the max over the bins of one pixel column. */
  function levelAt(db: DbArray, f0: number, f1: number, binHz: number): number {
    const n = db.length;
    const b0 = f0 / binHz;
    const b1 = f1 / binHz;
    if (b1 - b0 < 1) {
      const b = (b0 + b1) / 2;
      const i = Math.floor(b);
      if (i < 0 || i + 1 >= n) return -Infinity;
      const t = b - i;
      return (db[i] as number) * (1 - t) + (db[i + 1] as number) * t;
    }
    let mx = -Infinity;
    for (let k = Math.ceil(b0); k <= Math.floor(b1) && k < n; k++) {
      const v = db[k] as number;
      if (v > mx) mx = v;
    }
    return mx;
  }

  function computeTop(d: SpectrumViewData): void {
    const binHz = d.fs / d.fftSize;
    let mx = -Infinity;
    for (const arr of [d.ref, d.err, d.errBypass]) {
      if (!arr) continue;
      const k1 = Math.min(arr.length - 1, Math.ceil(SPECTRUM_F_MAX / binHz));
      for (let k = Math.floor(SPECTRUM_F_MIN / binHz); k <= k1; k++) {
        const v = arr[k] as number;
        if (Number.isFinite(v) && v > mx) mx = v;
      }
    }
    if (!Number.isFinite(mx)) return;
    // Hysteresis in 10 dB steps so the axis does not jump on every frame.
    const want = Math.min(20, Math.max(-60, Math.ceil((mx + 3) / 10) * 10));
    if (want > topDb || want < topDb - 20) topDb = want;
  }

  function draw(c: CanvasRenderingContext2D): void {
    if (!palette) palette = readPalette();
    const p = palette;
    c.setTransform(canvas.width / cssW, 0, 0, canvas.height / cssH, 0, 0);
    c.fillStyle = p.bg;
    c.fillRect(0, 0, cssW, cssH);
    c.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';

    const left = 38;
    const right = 8;
    const top = 8;
    const bottom = 20;
    const w = cssW - left - right;
    const h = cssH - top - bottom;
    if (w < 10 || h < 10) return;
    const yOf = (db: number): number => top + ((topDb - db) / RANGE_DB) * h;
    const xOf = (f: number): number => left + freqToX(f) * w;

    // Grid
    c.strokeStyle = p.grid;
    c.lineWidth = 1;
    c.beginPath();
    for (const f of GRID_FREQS) {
      const x = Math.round(xOf(f)) + 0.5;
      c.moveTo(x, top);
      c.lineTo(x, top + h);
    }
    for (let db = topDb; db >= topDb - RANGE_DB; db -= 10) {
      const y = Math.round(yOf(db)) + 0.5;
      c.moveTo(left, y);
      c.lineTo(left + w, y);
    }
    c.stroke();
    c.strokeStyle = p.gridStrong;
    c.strokeRect(left + 0.5, top + 0.5, w - 1, h - 1);
    c.fillStyle = p.text;
    c.textAlign = 'center';
    c.textBaseline = 'top';
    for (const f of GRID_FREQS) if (LABEL_FREQS.has(f)) c.fillText(fmtFreq(f), xOf(f), top + h + 4);
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    for (let db = topDb; db >= topDb - RANGE_DB; db -= 20) c.fillText(String(db), left - 4, yOf(db));
    c.textAlign = 'start';

    const d = data;
    if (!d || !(d.fs > 0) || !(d.fftSize > 0)) {
      c.textAlign = 'center';
      c.fillText('In attesa di dati…', left + w / 2, top + h / 2);
      c.textAlign = 'start';
      return;
    }
    const binHz = d.fs / d.fftSize;

    c.save();
    c.beginPath();
    c.rect(left, top, w, h);
    c.clip();
    if (thirds) {
      const series: Array<[Float64Array, string, boolean]> = [
        [bandRef, p.ref, !!d.ref],
        [bandByp, p.bypass, !!d.errBypass],
        [bandErr, p.err, !!d.err],
      ];
      const shown = series.filter((s) => s[2]);
      for (let i = 0; i < centers.length; i++) {
        const fc = centers[i] as number;
        const xa = xOf(fc * Math.pow(2, -1 / 6));
        const xb = xOf(fc * Math.pow(2, 1 / 6));
        const bw = Math.max(1, ((xb - xa) * 0.85) / Math.max(1, shown.length));
        const x = xa + (xb - xa) * 0.075;
        shown.forEach(([arr, color], j) => {
          const v = arr[i] as number;
          if (!Number.isFinite(v)) return;
          const y = Math.max(top, yOf(v));
          c.fillStyle = color;
          c.fillRect(x + j * bw, y, bw - 1, top + h - y);
        });
      }
    } else {
      const traces: Array<[DbArray | null, string, number[]]> = [
        [d.ref, p.ref, []],
        [d.errBypass, p.bypass, [5, 4]],
        [d.err, p.err, []],
      ];
      const cols = Math.max(2, Math.floor(w));
      for (const [arr, color, dash] of traces) {
        if (!arr) continue;
        c.strokeStyle = color;
        c.lineWidth = 1.5;
        c.setLineDash(dash);
        c.beginPath();
        let pen = false;
        for (let col = 0; col < cols; col++) {
          const fA = xToFreq(col / cols);
          const fB = xToFreq((col + 1) / cols);
          const v = levelAt(arr, fA, fB, binHz);
          if (!Number.isFinite(v)) {
            pen = false;
            continue;
          }
          const x = left + col + 0.5;
          const y = yOf(v);
          if (pen) c.lineTo(x, y);
          else c.moveTo(x, y);
          pen = true;
        }
        c.stroke();
      }
      c.setLineDash([]);
    }
    c.restore();

    if (hoverX !== null && hoverX >= left && hoverX <= left + w) {
      c.strokeStyle = p.gridStrong;
      c.beginPath();
      c.moveTo(Math.round(hoverX) + 0.5, top);
      c.lineTo(Math.round(hoverX) + 0.5, top + h);
      c.stroke();
    }
  }

  function updateReadout(): void {
    const d = data;
    if (!d || hoverX === null || !(cssW > 0)) return;
    const left = 38;
    const w = cssW - left - 8;
    const x = (hoverX - left) / w;
    if (x < 0 || x > 1) return;
    const f = xToFreq(x);
    const fc = nearestCenter(f);
    const binHz = d.fs / d.fftSize;
    const levels: string[] = [];
    const show = (name: string, arr: DbArray | null): void => {
      if (!arr) return;
      const v = thirds ? bandLevelDb(arr, d.fs, d.fftSize, fc) : levelAt(arr, f, f, binHz);
      levels.push(`${name} ${Number.isFinite(v) ? formatNumber(v, 1) : '—'} dB`);
    };
    show('rif.', d.ref);
    show('errore', d.err);
    show('bypass', d.errBypass);
    // Normalised by the reference of each state, per third-octave band (plan).
    const a = bandAttenuationDb({ errOff: d.errBypass, refOff: d.refOff ?? null, errOn: d.err, refOn: d.refOn ?? null }, d.fs, d.fftSize, fc);
    const text = spectrumReadout(f, levels, thirds, fc, a);
    if (readout.textContent !== text) readout.textContent = text;
  }

  function nearestCenter(f: number): number {
    let best = centers[0] ?? f;
    for (const c of centers) if (Math.abs(Math.log(c / f)) < Math.abs(Math.log(best / f))) best = c;
    return best;
  }

  requestDraw();

  return {
    update(next: SpectrumViewData): void {
      data = next;
      if (next.fs > 0 && next.fftSize > 0) {
        computeTop(next);
        for (let i = 0; i < centers.length; i++) {
          const fc = centers[i] as number;
          bandRef[i] = next.ref ? bandLevelDb(next.ref, next.fs, next.fftSize, fc) : -Infinity;
          bandErr[i] = next.err ? bandLevelDb(next.err, next.fs, next.fftSize, fc) : -Infinity;
          bandByp[i] = next.errBypass ? bandLevelDb(next.errBypass, next.fs, next.fftSize, fc) : -Infinity;
        }
      }
      updateReadout();
      requestDraw();
    },
    destroy(): void {
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0;
      ro.disconnect();
      mo.disconnect();
      mq.removeEventListener('change', invalidatePalette);
      root.remove();
    },
  };
}
