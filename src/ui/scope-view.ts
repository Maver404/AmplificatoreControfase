/**
 * Oscilloscope that verifies the DSP: for the selected channel it draws the expected anti-noise
 * e[n] = s·g·cond[n − D] (s = −1 when the polarity is inverted) over the actual output out[n] —
 * the two traces coincide when the chain is right — plus the error mic in a second lane.
 * Zero-crossing trigger on e, correlation e/out as a number.
 *
 * Tap blocks are planar per TAP_CHANNELS; pushTap() only copies into fixed ring buffers (the caller
 * may recycle the block right after), drawing happens in requestAnimationFrame with fixed scratch
 * arrays. The fractional delay uses the same 3rd-order Lagrange taps as the DSP's DelayLine.
 */
import { TAP_CHANNELS } from '../audio/protocol.ts';
import { D_MIN_SAMPLES } from '../dsp/params.ts';
import { formatDb, formatNumber, formatSigned } from './units.ts';

const CH_COUNT = TAP_CHANNELS.length;
const CH_COND = TAP_CHANNELS.indexOf('cond');
const CH_OUT_L = TAP_CHANNELS.indexOf('outL');
const CH_OUT_R = TAP_CHANNELS.indexOf('outR');
const CH_ERR = TAP_CHANNELS.indexOf('err');

/** Seconds of history kept (must cover MAX_DELAY_MS + the analysed region). */
export const SCOPE_HISTORY_S = 1;
/** Longest timebase window and minimum trigger search span (one period of 20 Hz). */
export const SCOPE_MAX_WINDOW_S = 0.1;
export const SCOPE_MIN_SEARCH_S = 0.05;
export const SCOPE_TIMEBASES_MS = [2, 5, 10, 20, 50, 100] as const;

export interface ScopeAlignment {
  channel: 0 | 1;
  /** Total tap delay in samples, including D_MIN (as in MeterSnapshot.delaySamples). */
  delaySamples: number;
  /** Total linear gain from cond to the output (channel · master · soft-start, as known). */
  gainLin: number;
  polarityInvert: boolean;
}

const nextPow2 = (n: number): number => {
  let p = 1;
  while (p < n) p *= 2;
  return p;
};

/**
 * x[n − d] from a ring buffer, with the DSP's taps: k = ⌊d⌋ − 1, u = d − k ∈ [1, 2),
 * Lagrange through x[n−k], x[n−k−1], x[n−k−2], x[n−k−3]. `n` is an absolute frame index.
 */
export function lagrangeDelayed(buf: Float32Array, mask: number, n: number, d: number): number {
  const dd = d < D_MIN_SAMPLES ? D_MIN_SAMPLES : d;
  const k = Math.floor(dd) - 1;
  const u = dd - k;
  const i = n - k;
  const x0 = buf[i & mask] as number;
  const x1 = buf[(i - 1) & mask] as number;
  const x2 = buf[(i - 2) & mask] as number;
  const x3 = buf[(i - 3) & mask] as number;
  const um1 = u - 1;
  const um2 = u - 2;
  const um3 = u - 3;
  return (
    (-um1 * um2 * um3 * x0) / 6 + (u * um2 * um3 * x1) / 2 - (u * um1 * um3 * x2) / 2 + (u * um1 * um2 * x3) / 6
  );
}

/** Ring buffers of the taps the scope needs, with a shared absolute write counter. */
export class ScopeHistory {
  readonly capacity: number;
  readonly mask: number;
  readonly cond: Float32Array;
  readonly outL: Float32Array;
  readonly outR: Float32Array;
  readonly err: Float32Array;
  private written = 0;

  constructor(fs: number, seconds = SCOPE_HISTORY_S) {
    this.capacity = nextPow2(Math.max(1024, Math.ceil(fs * seconds)));
    this.mask = this.capacity - 1;
    this.cond = new Float32Array(this.capacity);
    this.outL = new Float32Array(this.capacity);
    this.outR = new Float32Array(this.capacity);
    this.err = new Float32Array(this.capacity);
  }

  /** Total frames written so far (absolute index of the next frame). */
  get frames(): number {
    return this.written;
  }

  /** Copies one planar tap block; returns false (and ignores it) if the layout is inconsistent. */
  push(block: Float32Array, frames: number): boolean {
    if (!Number.isInteger(frames) || frames <= 0 || block.length < CH_COUNT * frames) return false;
    const m = this.mask;
    const base = this.written;
    for (let i = 0; i < frames; i++) {
      const w = (base + i) & m;
      this.cond[w] = finite(block[CH_COND * frames + i] as number);
      this.outL[w] = finite(block[CH_OUT_L * frames + i] as number);
      this.outR[w] = finite(block[CH_OUT_R * frames + i] as number);
      this.err[w] = finite(block[CH_ERR * frames + i] as number);
    }
    this.written = base + frames;
    return true;
  }

  reset(): void {
    this.written = 0;
    this.cond.fill(0);
    this.outL.fill(0);
    this.outR.fill(0);
    this.err.fill(0);
  }
}

const finite = (v: number): number => (Number.isFinite(v) ? v : 0);

export interface ScopeResult {
  ok: boolean;
  reason: 'no-data' | 'delay-too-long' | 'bad-alignment' | null;
  /** Window length actually filled in e/out/err. */
  length: number;
  triggered: boolean;
  /** Normalised zero-lag correlation of e and out over the analysed region (null if silent). */
  correlation: number | null;
  /** Output level relative to the expected signal, dB (null if silent). */
  levelRatioDb: number | null;
  peak: number;
  errPeak: number;
}

const SILENT_ENERGY = 1e-14;

/** Fills fixed windows of e / out / err for drawing; allocation-free after construction. */
export class ScopeAnalyzer {
  readonly e: Float32Array;
  readonly out: Float32Array;
  readonly err: Float32Array;
  private readonly region: Float32Array;
  readonly maxWindow: number;
  readonly maxSearch: number;
  readonly result: ScopeResult = {
    ok: false,
    reason: 'no-data',
    length: 0,
    triggered: false,
    correlation: null,
    levelRatioDb: null,
    peak: 0,
    errPeak: 0,
  };

  constructor(maxWindow: number, maxSearch: number) {
    this.maxWindow = Math.max(2, Math.ceil(maxWindow));
    this.maxSearch = Math.max(1, Math.ceil(maxSearch));
    this.e = new Float32Array(this.maxWindow);
    this.out = new Float32Array(this.maxWindow);
    this.err = new Float32Array(this.maxWindow);
    this.region = new Float32Array(this.maxWindow + this.maxSearch);
  }

  analyze(h: ScopeHistory, a: ScopeAlignment, windowLen: number, searchLen: number): ScopeResult {
    const r = this.result;
    const W = Math.min(this.maxWindow, Math.max(2, Math.floor(windowLen)));
    const S = Math.min(this.maxSearch, Math.max(1, Math.floor(searchLen)));
    const R = W + S;
    r.length = 0;
    r.triggered = false;
    r.correlation = null;
    r.levelRatioDb = null;
    r.peak = 0;
    r.errPeak = 0;
    if (!Number.isFinite(a.delaySamples) || !Number.isFinite(a.gainLin)) {
      r.ok = false;
      r.reason = 'bad-alignment';
      return r;
    }
    const D = Math.max(D_MIN_SAMPLES, a.delaySamples);
    const n = h.frames;
    const first = n - R;
    // Oldest sample touched by the interpolator for the first frame of the region.
    const oldest = first - Math.floor(D) - 2;
    if (first < 0 || oldest < 0) {
      r.ok = false;
      r.reason = 'no-data';
      return r;
    }
    if (oldest < n - h.capacity) {
      r.ok = false;
      r.reason = 'delay-too-long';
      return r;
    }
    const s = a.polarityInvert ? -a.gainLin : a.gainLin;
    const outBuf = a.channel === 0 ? h.outL : h.outR;
    const m = h.mask;
    const reg = this.region;
    let peakE = 0;
    let see = 0;
    let soo = 0;
    let seo = 0;
    for (let j = 0; j < R; j++) {
      const t = first + j;
      const e = s * lagrangeDelayed(h.cond, m, t, D);
      reg[j] = e;
      const o = outBuf[t & m] as number;
      see += e * e;
      soo += o * o;
      seo += e * o;
      const ae = Math.abs(e);
      if (ae > peakE) peakE = ae;
    }

    // Rising zero crossing of e with hysteresis; the latest one whose window still fits.
    const hyst = peakE * 0.05;
    let armed = false;
    let trig = -1;
    for (let j = 1; j <= S; j++) {
      const v = reg[j] as number;
      if (v < -hyst) armed = true;
      else if (armed && (reg[j - 1] as number) < 0 && v >= 0) {
        trig = j;
        armed = false;
      }
    }
    const start = trig >= 0 ? trig : S;
    let peak = 0;
    let errPeak = 0;
    for (let k = 0; k < W; k++) {
      const t = first + start + k;
      const e = reg[start + k] as number;
      const o = outBuf[t & m] as number;
      const er = h.err[t & m] as number;
      this.e[k] = e;
      this.out[k] = o;
      this.err[k] = er;
      const ae = Math.abs(e);
      const ao = Math.abs(o);
      if (ae > peak) peak = ae;
      if (ao > peak) peak = ao;
      const ar = Math.abs(er);
      if (ar > errPeak) errPeak = ar;
    }
    r.ok = true;
    r.reason = null;
    r.length = W;
    r.triggered = trig >= 0;
    r.peak = peak;
    r.errPeak = errPeak;
    if (see > SILENT_ENERGY && soo > SILENT_ENERGY) {
      r.correlation = Math.max(-1, Math.min(1, seo / Math.sqrt(see * soo)));
      r.levelRatioDb = 10 * Math.log10(soo / see);
    }
    return r;
  }
}

// ---------------------------------------------------------------- view

export interface ScopeViewOptions {
  fs: number;
  timebaseMs?: number;
  /** The user picked another channel: the controller should call setAlignment for it. */
  onChannelChange?: (channel: 0 | 1) => void;
}

export interface ScopeViewData {
  fs?: number;
  /** Selects the displayed channel programmatically. */
  channel?: 0 | 1;
  /** Optional note under the plot (e.g. "In modalità Hilbert la traccia attesa non include la rotazione"). */
  note?: string;
}

export interface ScopeView {
  /** Copies the block: the caller may recycle it as soon as this returns. */
  pushTap(block: Float32Array, frames: number): void;
  /** Stores the alignment of `a.channel`; the view shows the channel selected by the user (or update({ channel })). */
  setAlignment(a: ScopeAlignment): void;
  update(data: ScopeViewData): void;
  destroy(): void;
}

interface Palette {
  bg: string;
  grid: string;
  gridStrong: string;
  text: string;
  e: string;
  out: string;
  err: string;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

let seq = 0;

export function createScopeView(container: HTMLElement, opts: ScopeViewOptions): ScopeView {
  const id = `scope${++seq}`;
  let fs = opts.fs > 0 ? opts.fs : 48000;
  let timebaseMs = (SCOPE_TIMEBASES_MS as readonly number[]).includes(opts.timebaseMs ?? 0) ? (opts.timebaseMs as number) : 20;
  // One alignment per channel: the channel picker never depends on which one the controller sent last.
  const alignments: [ScopeAlignment | null, ScopeAlignment | null] = [null, null];
  let selected: 0 | 1 = 0;
  let frozen = false;
  let history = new ScopeHistory(fs);
  let analyzer = makeAnalyzer(fs);
  let scaleTop = 1e-3;
  let scaleErr = 1e-3;
  let raf = 0;
  let dirty = true;
  let cssW = 0;
  let cssH = 0;
  let palette: Palette | null = null;
  let lastReadout = '';

  function makeAnalyzer(rate: number): ScopeAnalyzer {
    const w = Math.ceil(rate * SCOPE_MAX_WINDOW_S) + 1;
    return new ScopeAnalyzer(w, Math.max(w, Math.ceil(rate * SCOPE_MIN_SEARCH_S)));
  }

  // ---- DOM ----
  const root = el('div', 'scope-view');
  const toolbar = el('div', 'view-toolbar');
  const chLabel = el('label', 'tb-field');
  const chSel = el('select');
  for (const [v, t] of [
    ['0', 'L'],
    ['1', 'R'],
  ] as const) {
    const o = el('option', undefined, t);
    o.value = v;
    chSel.append(o);
  }
  chLabel.append('Canale ', chSel);
  const tbLabel = el('label', 'tb-field');
  const tbSel = el('select');
  for (const t of SCOPE_TIMEBASES_MS) {
    const o = el('option', undefined, `${t} ms`);
    o.value = String(t);
    tbSel.append(o);
  }
  tbSel.value = String(timebaseMs);
  tbLabel.append('Finestra ', tbSel);
  const freezeBtn = el('button', 'toggle-btn', 'Blocca');
  freezeBtn.type = 'button';
  freezeBtn.setAttribute('aria-pressed', 'false');
  toolbar.append(chLabel, tbLabel, freezeBtn);

  const legend = el('ul', 'legend');
  for (const [cls, text] of [
    ['legend-e', 'Atteso e[n] = s·g·cond[n − D]'],
    ['legend-out', 'Uscita reale'],
    ['legend-err', 'Mic d’errore'],
  ] as const) {
    const li = el('li', cls);
    li.append(el('span', 'legend-swatch'), text);
    legend.append(li);
  }

  const wrap = el('div', 'canvas-wrap canvas-wrap--scope');
  const canvas = el('canvas');
  canvas.setAttribute('role', 'img');
  canvas.id = `${id}-canvas`;
  wrap.append(canvas);
  const readout = el('p', 'view-readout num');
  readout.id = `${id}-readout`;
  canvas.setAttribute('aria-describedby', readout.id);
  canvas.setAttribute(
    'aria-label',
    'Oscilloscopio: segnale anti-rumore atteso sovrapposto all’uscita reale, e segnale del mic d’errore',
  );
  const note = el('p', 'fp-help');
  note.hidden = true;
  root.append(toolbar, legend, wrap, readout, note);
  container.append(root);

  const ctx = canvas.getContext('2d');

  // ---- events ----
  chSel.addEventListener('change', () => {
    const ch: 0 | 1 = chSel.value === '1' ? 1 : 0;
    selected = ch;
    requestDraw();
    opts.onChannelChange?.(ch);
  });
  tbSel.addEventListener('change', () => {
    const v = Number(tbSel.value);
    if ((SCOPE_TIMEBASES_MS as readonly number[]).includes(v)) timebaseMs = v;
    requestDraw();
  });
  freezeBtn.addEventListener('click', () => {
    frozen = !frozen;
    freezeBtn.setAttribute('aria-pressed', String(frozen));
    freezeBtn.textContent = frozen ? 'Bloccato' : 'Blocca';
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
      e: v('--trace-e', '#f5a524'),
      out: v('--trace-out', '#3fb8ff'),
      err: v('--trace-err', '#e5484d'),
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

  function drawTrace(
    c: CanvasRenderingContext2D,
    data: Float32Array,
    len: number,
    x0: number,
    w: number,
    yMid: number,
    half: number,
    scale: number,
  ): void {
    const k = half / scale;
    c.beginPath();
    if (len <= w * 2) {
      const dx = w / Math.max(1, len - 1);
      for (let i = 0; i < len; i++) {
        const x = x0 + i * dx;
        const y = yMid - (data[i] as number) * k;
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
    } else {
      const cols = Math.max(1, Math.floor(w));
      for (let col = 0; col < cols; col++) {
        const a = Math.floor((col / cols) * len);
        const b = Math.max(a + 1, Math.floor(((col + 1) / cols) * len));
        let mn = Infinity;
        let mx = -Infinity;
        for (let i = a; i < b; i++) {
          const v = data[i] as number;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        const x = x0 + col + 0.5;
        if (col === 0) c.moveTo(x, yMid - mx * k);
        else c.lineTo(x, yMid - mx * k);
        c.lineTo(x, yMid - mn * k);
      }
    }
    c.stroke();
  }

  function drawGrid(c: CanvasRenderingContext2D, p: Palette, x: number, y: number, w: number, h: number): void {
    c.strokeStyle = p.grid;
    c.lineWidth = 1;
    c.beginPath();
    for (let i = 1; i < 10; i++) {
      const gx = Math.round(x + (i * w) / 10) + 0.5;
      c.moveTo(gx, y);
      c.lineTo(gx, y + h);
    }
    for (const f of [0.25, 0.75]) {
      const gy = Math.round(y + f * h) + 0.5;
      c.moveTo(x, gy);
      c.lineTo(x + w, gy);
    }
    c.stroke();
    c.strokeStyle = p.gridStrong;
    c.beginPath();
    const mid = Math.round(y + h / 2) + 0.5;
    c.moveTo(x, mid);
    c.lineTo(x + w, mid);
    c.rect(x + 0.5, y + 0.5, w - 1, h - 1);
    c.stroke();
  }

  function draw(c: CanvasRenderingContext2D): void {
    if (!palette) palette = readPalette();
    const p = palette;
    const sx = canvas.width / cssW;
    const sy = canvas.height / cssH;
    c.setTransform(sx, 0, 0, sy, 0, 0);
    c.fillStyle = p.bg;
    c.fillRect(0, 0, cssW, cssH);
    c.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    c.textBaseline = 'top';

    const pad = 6;
    const gap = 10;
    const x0 = pad;
    const w = cssW - 2 * pad;
    const topH = Math.round((cssH - 2 * pad - gap) * 0.64);
    const botH = cssH - 2 * pad - gap - topH;
    const topY = pad;
    const botY = pad + topH + gap;
    drawGrid(c, p, x0, topY, w, topH);
    drawGrid(c, p, x0, botY, w, botH);

    const windowLen = Math.round((timebaseMs / 1000) * fs);
    const searchLen = Math.max(windowLen, Math.round(SCOPE_MIN_SEARCH_S * fs));
    const alignment = alignments[selected];
    const r = alignment ? analyzer.analyze(history, alignment, windowLen, searchLen) : null;

    c.fillStyle = p.text;
    c.fillText(`${formatNumber(timebaseMs / 10, timebaseMs < 10 ? 1 : 0)} ms/div`, x0 + w - 70, topY + 4);

    let text: string;
    if (!r || !r.ok) {
      const msg = !r
        ? `In attesa dei parametri del canale ${selected === 0 ? 'L' : 'R'}…`
        : r.reason === 'delay-too-long'
          ? 'Ritardo oltre la cronologia dell’oscilloscopio'
          : r.reason === 'bad-alignment'
            ? 'Allineamento non valido'
            : 'In attesa di dati…';
      c.textBaseline = 'middle';
      c.textAlign = 'center';
      c.fillText(msg, x0 + w / 2, topY + topH / 2);
      c.textAlign = 'start';
      text = msg;
    } else {
      // Autoscale: grow at once, shrink slowly so the picture does not breathe.
      const tTop = Math.max(1e-5, r.peak * 1.15);
      scaleTop = tTop > scaleTop ? tTop : Math.max(tTop, scaleTop * 0.95);
      const tErr = Math.max(1e-5, r.errPeak * 1.15);
      scaleErr = tErr > scaleErr ? tErr : Math.max(tErr, scaleErr * 0.95);

      c.save();
      c.beginPath();
      c.rect(x0, topY, w, topH);
      c.clip();
      c.lineJoin = 'round';
      // e is a wide translucent band, out a thin line on top: when the DSP is right the line
      // runs inside the band, so both stay visible.
      c.globalAlpha = 0.45;
      c.strokeStyle = p.e;
      c.lineWidth = 6;
      drawTrace(c, analyzer.e, r.length, x0, w, topY + topH / 2, topH / 2, scaleTop);
      c.globalAlpha = 1;
      c.strokeStyle = p.out;
      c.lineWidth = 1.5;
      drawTrace(c, analyzer.out, r.length, x0, w, topY + topH / 2, topH / 2, scaleTop);
      c.restore();

      c.save();
      c.beginPath();
      c.rect(x0, botY, w, botH);
      c.clip();
      c.strokeStyle = p.err;
      c.lineWidth = 1.25;
      drawTrace(c, analyzer.err, r.length, x0, w, botY + botH / 2, botH / 2, scaleErr);
      c.restore();

      c.fillStyle = p.text;
      c.fillText(`atteso / uscita · ±${formatDb(20 * Math.log10(scaleTop), 0)} dBFS`, x0 + 4, topY + 4);
      c.fillText(`mic d’errore · ±${formatDb(20 * Math.log10(scaleErr), 0)} dBFS`, x0 + 4, botY + 4);

      const corr = r.correlation === null ? '—' : formatNumber(r.correlation, 3);
      const ratio = r.levelRatioDb === null ? '—' : `${formatSigned(r.levelRatioDb, 1)} dB`;
      text =
        `Canale ${selected === 0 ? 'L' : 'R'} · correlazione atteso/uscita: ${corr} · ` +
        `livello uscita/atteso: ${ratio} · trigger: ${r.triggered ? 'agganciato' : 'libero'}` +
        (frozen ? ' · bloccato' : '');
    }
    if (text !== lastReadout) {
      lastReadout = text;
      readout.textContent = text;
      const corr = r ? r.correlation : null;
      readout.dataset['quality'] = corr === null ? '' : corr > 0.99 ? 'good' : corr > 0.9 ? 'fair' : 'bad';
    }
  }

  function setFs(next: number): void {
    if (!(next > 0) || next === fs) return;
    fs = next;
    history = new ScopeHistory(fs);
    analyzer = makeAnalyzer(fs);
    requestDraw();
  }

  requestDraw();

  return {
    pushTap(block: Float32Array, frames: number): void {
      if (frozen) return;
      if (history.push(block, frames)) requestDraw();
    },
    setAlignment(a: ScopeAlignment): void {
      alignments[a.channel === 1 ? 1 : 0] = { ...a };
      requestDraw();
    },
    update(data: ScopeViewData): void {
      if (data.fs !== undefined) setFs(data.fs);
      if (data.channel !== undefined && data.channel !== selected) {
        selected = data.channel;
        chSel.value = String(selected);
        requestDraw();
      }
      if (data.note !== undefined) {
        note.textContent = data.note;
        note.hidden = data.note === '';
      }
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
