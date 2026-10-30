import { describe, expect, it } from 'vitest';
import { TAP_CHANNELS } from '../../src/audio/protocol.ts';
import { ScopeAnalyzer, ScopeHistory, lagrangeDelayed } from '../../src/ui/scope-view.ts';
import type { ScopeAlignment } from '../../src/ui/scope-view.ts';

/** Seeded PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Reference Lagrange interpolation written independently (product form over the 4 nodes). */
function lagrangeRef(x: (i: number) => number, n: number, d: number): number {
  const k = Math.floor(d) - 1;
  const u = d - k;
  let y = 0;
  for (let j = 0; j < 4; j++) {
    let w = 1;
    for (let m = 0; m < 4; m++) if (m !== j) w *= (u - m) / (j - m);
    y += w * x(n - k - j);
  }
  return y;
}

const FS = 48000;
const FRAMES = 2048;
const CH = Object.fromEntries(TAP_CHANNELS.map((c, i) => [c, i])) as Record<(typeof TAP_CHANNELS)[number], number>;

/** Builds planar tap blocks where out = s·g·cond[n − D] exactly as the DSP contract defines it. */
function makeSignal(totalFrames: number, a: ScopeAlignment, otherGain = 0.1): Float32Array[] {
  const r = rng(7);
  const cap = 1 << 17;
  const cond = new Float32Array(cap);
  const blocks: Float32Array[] = [];
  const s = a.polarityInvert ? -a.gainLin : a.gainLin;
  let n = 0;
  let lp = 0;
  while (n < totalFrames) {
    const b = new Float32Array(TAP_CHANNELS.length * FRAMES);
    for (let i = 0; i < FRAMES; i++, n++) {
      lp += 0.05 * (r() * 2 - 1 - lp);
      const c = 0.3 * Math.sin((2 * Math.PI * 97 * n) / FS) + lp;
      cond[n & (cap - 1)] = c;
      const e = s * lagrangeDelayed(cond, cap - 1, n, a.delaySamples);
      b[CH.ref * FRAMES + i] = c;
      b[CH.cond * FRAMES + i] = c;
      b[(a.channel === 0 ? CH.outL : CH.outR) * FRAMES + i] = e;
      b[(a.channel === 0 ? CH.outR : CH.outL) * FRAMES + i] = otherGain * c;
      b[CH.err * FRAMES + i] = 0.01 * Math.sin((2 * Math.PI * 50 * n) / FS);
    }
    blocks.push(b);
  }
  return blocks;
}

describe('lagrangeDelayed', () => {
  const cap = 1024;
  const buf = new Float32Array(cap);
  const r = rng(1);
  for (let i = 0; i < cap; i++) buf[i] = r() * 2 - 1;
  const x = (i: number): number => buf[i & (cap - 1)] as number;

  it('is exact for integer delays', () => {
    for (const d of [1, 2, 5, 17, 300]) expect(lagrangeDelayed(buf, cap - 1, 700, d)).toBe(x(700 - d));
  });

  it('matches an independent Lagrange implementation for fractional delays', () => {
    for (const d of [1.25, 1.999, 12.37, 123.5, 400.01]) {
      expect(lagrangeDelayed(buf, cap - 1, 900, d)).toBeCloseTo(lagrangeRef(x, 900, d), 6);
    }
  });

  it('clamps delays below D_MIN to D_MIN', () => {
    expect(lagrangeDelayed(buf, cap - 1, 500, 0.2)).toBe(x(499));
  });

  it('delays a low-frequency sine accurately', () => {
    const b = new Float32Array(4096);
    const f = 200;
    for (let i = 0; i < b.length; i++) b[i] = Math.sin((2 * Math.PI * f * i) / FS);
    const d = 12.37;
    const got = lagrangeDelayed(b, 4095, 3000, d);
    expect(got).toBeCloseTo(Math.sin((2 * Math.PI * f * (3000 - d)) / FS), 4);
  });
});

describe('ScopeHistory', () => {
  it('rejects blocks whose length does not match the planar layout', () => {
    const h = new ScopeHistory(FS);
    expect(h.push(new Float32Array(10), 4)).toBe(false);
    expect(h.push(new Float32Array(TAP_CHANNELS.length * 4), 0)).toBe(false);
    expect(h.push(new Float32Array(TAP_CHANNELS.length * 4), 4)).toBe(true);
    expect(h.frames).toBe(4);
  });

  it('keeps at least one second and replaces non-finite samples with 0', () => {
    const h = new ScopeHistory(44100);
    expect(h.capacity).toBeGreaterThanOrEqual(44100);
    const b = new Float32Array(TAP_CHANNELS.length * 2);
    b[CH.cond * 2] = Number.NaN;
    b[CH.cond * 2 + 1] = 0.5;
    h.push(b, 2);
    expect(h.cond[0]).toBe(0);
    expect(h.cond[1]).toBe(0.5);
  });
});

describe('ScopeAnalyzer', () => {
  const window = Math.round(0.02 * FS);
  const search = Math.round(0.05 * FS);

  function run(a: ScopeAlignment, analyzeWith: ScopeAlignment = a) {
    const h = new ScopeHistory(FS);
    for (const b of makeSignal(FS, a)) h.push(b, FRAMES);
    const an = new ScopeAnalyzer(Math.ceil(0.1 * FS) + 1, Math.ceil(0.1 * FS) + 1);
    return { an, r: an.analyze(h, analyzeWith, window, search) };
  }

  it('reports correlation 1 and 0 dB when out = s·g·cond[n − D] (fractional D, channel L)', () => {
    const a: ScopeAlignment = { channel: 0, delaySamples: 123.37, gainLin: 0.5, polarityInvert: true };
    const { an, r } = run(a);
    expect(r.ok).toBe(true);
    expect(r.correlation).not.toBeNull();
    expect(r.correlation as number).toBeGreaterThan(0.99999);
    expect(Math.abs(r.levelRatioDb as number)).toBeLessThan(0.01);
    for (let k = 0; k < r.length; k++) expect(an.out[k]).toBeCloseTo(an.e[k] as number, 6);
  });

  it('uses the selected channel (R)', () => {
    const a: ScopeAlignment = { channel: 1, delaySamples: 480.5, gainLin: 0.25, polarityInvert: false };
    const { r } = run(a);
    expect(r.correlation as number).toBeGreaterThan(0.99999);
  });

  it('detects a polarity error as correlation ≈ −1', () => {
    const a: ScopeAlignment = { channel: 0, delaySamples: 50.25, gainLin: 0.5, polarityInvert: true };
    const { r } = run(a, { ...a, polarityInvert: false });
    expect(r.correlation as number).toBeLessThan(-0.99999);
  });

  it('detects a gain error in the level ratio', () => {
    const a: ScopeAlignment = { channel: 0, delaySamples: 50.25, gainLin: 0.5, polarityInvert: true };
    const { r } = run(a, { ...a, gainLin: 0.25 });
    expect(r.levelRatioDb as number).toBeCloseTo(20 * Math.log10(2), 3);
  });

  it('a delay error lowers the correlation', () => {
    const a: ScopeAlignment = { channel: 0, delaySamples: 200, gainLin: 0.5, polarityInvert: true };
    const { r } = run(a, { ...a, delaySamples: 260 });
    expect(r.correlation as number).toBeLessThan(0.95);
  });

  it('triggers on a rising zero crossing of e', () => {
    const a: ScopeAlignment = { channel: 0, delaySamples: 10, gainLin: 1, polarityInvert: false };
    const { an, r } = run(a);
    expect(r.triggered).toBe(true);
    expect(an.e[0]).toBeGreaterThanOrEqual(0);
    expect(an.e[0] as number).toBeLessThan(0.2 * r.peak);
  });

  it('waits for enough history and refuses delays longer than the history', () => {
    const h = new ScopeHistory(FS);
    const an = new ScopeAnalyzer(window, search);
    const a: ScopeAlignment = { channel: 0, delaySamples: 100, gainLin: 1, polarityInvert: true };
    expect(an.analyze(h, a, window, search).reason).toBe('no-data');
    for (const b of makeSignal(2 * h.capacity, a)) h.push(b, FRAMES);
    expect(an.analyze(h, { ...a, delaySamples: h.capacity }, window, search).reason).toBe('delay-too-long');
    expect(an.analyze(h, { ...a, delaySamples: Number.NaN }, window, search).reason).toBe('bad-alignment');
    expect(an.analyze(h, a, window, search).ok).toBe(true);
  });

  it('returns null correlation on silence instead of NaN', () => {
    const h = new ScopeHistory(FS);
    for (let i = 0; i < 30; i++) h.push(new Float32Array(TAP_CHANNELS.length * FRAMES), FRAMES);
    const an = new ScopeAnalyzer(window, search);
    const r = an.analyze(h, { channel: 0, delaySamples: 10, gainLin: 1, polarityInvert: true }, window, search);
    expect(r.ok).toBe(true);
    expect(r.correlation).toBeNull();
    expect(r.levelRatioDb).toBeNull();
  });
});
