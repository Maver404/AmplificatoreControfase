import { describe, expect, it } from 'vitest';
import { cAbs, cArg, cDiv } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { HilbertPair, PhaseRotator, hilbertResponse } from '../../src/dsp/hilbert-pair.ts';

const deg = (rad: number) => (rad * 180) / Math.PI;

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Steady-state response(s) for input sin(ωn) of the given per-sample processors. */
function measure(fs: number, f: number, step: (x: number) => number[], seconds = 1): Complex[] {
  const w = (2 * Math.PI * f) / fs;
  const n = Math.round(seconds * fs);
  const start = Math.round(n / 2);
  let ss = 0, sc = 0, cc = 0;
  let ys: number[] = [];
  let yc: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = Math.sin(w * i);
    const c = Math.cos(w * i);
    const y = step(s);
    if (i === 0) {
      ys = y.map(() => 0);
      yc = y.map(() => 0);
    }
    if (i >= start) {
      ss += s * s; sc += s * c; cc += c * c;
      for (let k = 0; k < y.length; k++) {
        ys[k] += y[k] * s;
        yc[k] += y[k] * c;
      }
    }
  }
  const det = ss * cc - sc * sc;
  return ys.map((_, k) => ({ re: (ys[k] * cc - yc[k] * sc) / det, im: (yc[k] * ss - ys[k] * sc) / det }));
}

describe('hilbertResponse', () => {
  it('B leads A by 90° ± 0.75° from 30 Hz to 20 kHz at 44.1 and 48 kHz', () => {
    for (const fs of [44100, 48000]) {
      let worst = 0;
      for (let f = 30; f <= 20000; f *= 1.005) {
        const { a, b } = hilbertResponse(f, fs);
        const diff = deg(cArg(cDiv(b, a)));
        worst = Math.max(worst, Math.abs(diff - 90));
        // both paths are allpass
        expect(Math.abs(cAbs(a) - 1)).toBeLessThan(1e-9);
        expect(Math.abs(cAbs(b) - 1)).toBeLessThan(1e-9);
      }
      const end = hilbertResponse(20000, fs);
      worst = Math.max(worst, Math.abs(deg(cArg(cDiv(end.b, end.a))) - 90));
      expect(worst).toBeLessThan(0.75);
    }
  });
});

describe('HilbertPair', () => {
  it('time-domain outputs match hilbertResponse', () => {
    const fs = 48000;
    for (const f of [40, 100, 1000, 10000]) {
      const pair = new HilbertPair();
      const [ma, mb] = measure(fs, f, (x) => {
        pair.process(x);
        return [pair.a, pair.b];
      });
      const { a, b } = hilbertResponse(f, fs);
      for (const [m, h] of [[ma, a], [mb, b]] as const) {
        expect(Math.abs(cAbs(m) - cAbs(h))).toBeLessThan(1e-6);
        expect(Math.abs(deg(cArg(cDiv(m, h))))).toBeLessThan(1e-4);
      }
    }
  });

  it('survives NaN input and resets', () => {
    const pair = new HilbertPair();
    pair.process(1);
    pair.process(Number.NaN);
    pair.process(Number.POSITIVE_INFINITY);
    for (let i = 0; i < 100; i++) {
      pair.process(Math.sin(i));
      expect(Number.isFinite(pair.a) && Number.isFinite(pair.b)).toBe(true);
    }
    pair.reset();
    expect(pair.a).toBe(0);
    expect(pair.b).toBe(0);
  });
});

describe('PhaseRotator', () => {
  it('Δ = 0 equals path A exactly', () => {
    const rot = new PhaseRotator();
    const pair = new HilbertPair();
    const rnd = mulberry32(5);
    for (let i = 0; i < 10000; i++) {
      const x = rnd() * 2 - 1;
      pair.process(x);
      expect(rot.process(x)).toBe(pair.a);
    }
  });

  it('Δ = +30° lags A by 30° with unit gain', () => {
    for (const fs of [44100, 48000]) {
      for (const f of [50, 100, 300, 1000, 5000]) {
        const rot = new PhaseRotator(fs);
        rot.setDeltaDeg(30, true);
        const pair = new HilbertPair();
        const [y, a] = measure(fs, f, (x) => {
          pair.process(x);
          return [rot.process(x), pair.a];
        });
        const rel = cDiv(y, a);
        expect(Math.abs(deg(cArg(rel)) + 30)).toBeLessThan(0.75);
        expect(Math.abs(20 * Math.log10(cAbs(rel)))).toBeLessThan(0.1);
      }
    }
  });

  it('negative Δ leads A', () => {
    const fs = 48000;
    const rot = new PhaseRotator(fs);
    rot.setDeltaDeg(-120, true);
    const pair = new HilbertPair();
    const [y, a] = measure(fs, 200, (x) => {
      pair.process(x);
      return [rot.process(x), pair.a];
    });
    expect(Math.abs(deg(cArg(cDiv(y, a))) - 120)).toBeLessThan(0.75);
  });

  it('changes Δ smoothly (no click) and settles on the new value', () => {
    const fs = 48000;
    const f = 100;
    const w = (2 * Math.PI * f) / fs;
    const rot = new PhaseRotator(fs);
    let prev = 0;
    let maxStep = 0;
    for (let n = 0; n < fs; n++) {
      if (n === 20000) rot.setDeltaDeg(180);
      const y = rot.process(Math.sin(w * n));
      if (n > 5000) maxStep = Math.max(maxStep, Math.abs(y - prev));
      prev = y;
    }
    // natural max step of a unit 100 Hz sine is 0.0131; a hard 180° flip would step by up to 2
    expect(maxStep).toBeLessThan(0.03);
    // settled: now behaves like Δ = 180°
    const pair = new HilbertPair();
    const ref = new PhaseRotator(fs);
    ref.setDeltaDeg(180, true);
    const [y, a] = measure(fs, f, (x) => {
      pair.process(x);
      return [ref.process(x), pair.a];
    });
    expect(Math.abs(Math.abs(deg(cArg(cDiv(y, a)))) - 180)).toBeLessThan(0.75);
  });

  it('takes the shortest arc between angles', () => {
    const fs = 48000;
    const rot = new PhaseRotator(fs);
    rot.setDeltaDeg(170, true);
    expect(rot.currentDeg).toBeCloseTo(170, 9);
    rot.setDeltaDeg(-170);
    // after one 20 ms time constant the angle has moved 63% of the 20° arc through ±180°
    for (let i = 0; i < 960; i++) rot.process(0);
    const expected = 170 + 20 * (1 - Math.exp(-1)) - 360;
    expect(rot.currentDeg).toBeCloseTo(expected, 3);
    for (let i = 0; i < 48000; i++) rot.process(0);
    expect(rot.currentDeg).toBeCloseTo(-170, 9);
    rot.setDeltaDeg(Number.NaN);
    rot.process(0);
    expect(rot.currentDeg).toBeCloseTo(-170, 9);
  });
});
