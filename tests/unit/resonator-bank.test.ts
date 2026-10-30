import { describe, expect, it } from 'vitest';
import { Biquad, bandpass, biquadResponse, highpass } from '../../src/dsp/biquad.ts';
import { cAbs, cArg } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { RANGES, TONAL_MIN_Q, clampParams, minTonalQ } from '../../src/dsp/params.ts';
import { ResonatorBank, designResonatorBank, resonatorBankResponse } from '../../src/dsp/resonator-bank.ts';

const db = (h: Complex) => 20 * Math.log10(cAbs(h));
const deg = (h: Complex) => (cArg(h) * 180) / Math.PI;

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

function measure(bank: ResonatorBank, fs: number, f: number, seconds: number): Complex {
  bank.reset();
  const w = (2 * Math.PI * f) / fs;
  const n = Math.round(seconds * fs);
  const start = n - Math.round(0.5 * fs);
  let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
  for (let i = 0; i < n; i++) {
    const s = Math.sin(w * i);
    const y = bank.process(s);
    if (i >= start) {
      const c = Math.cos(w * i);
      ss += s * s; sc += s * c; cc += c * c; ys += y * s; yc += y * c;
    }
  }
  const det = ss * cc - sc * sc;
  return { re: (ys * cc - yc * sc) / det, im: (yc * ss - ys * sc) / det };
}

describe('resonatorBankResponse', () => {
  it('0 dB and 0° at every harmonic centre (overlap corrected)', () => {
    for (const fs of [44100, 48000]) {
      for (const f0 of [50, 100, 237.3, 1000]) {
        for (const harmonics of [1, 2, 4, 8]) {
          for (const q of [2, 3, 5, 10, 30, 100, 200]) {
            const d = designResonatorBank(fs, f0, harmonics, q);
            expect(d.converged).toBe(true);
            for (let h = 1; h <= d.harmonicsHz.length; h++) {
              const r = resonatorBankResponse(fs, h * f0, f0, harmonics, q);
              expect(Math.abs(db(r))).toBeLessThan(0.001);
              expect(Math.abs(deg(r))).toBeLessThan(0.01);
            }
          }
        }
      }
    }
  });

  it('the correction matters: a plain sum of band-passes is off at the centres', () => {
    const fs = 48000;
    const f0 = 100;
    const plain = (f: number): Complex => {
      let re = 0;
      let im = 0;
      for (let h = 1; h <= 4; h++) {
        const r = biquadResponse(bandpass(fs, h * f0, 10), f, fs);
        re += r.re;
        im += r.im;
      }
      return { re, im };
    };
    let worstPlain = 0;
    for (let h = 1; h <= 4; h++) worstPlain = Math.max(worstPlain, Math.abs(deg(plain(h * f0))));
    expect(worstPlain).toBeGreaterThan(1);
  });

  it('strongly rejects frequencies between and away from the harmonics', () => {
    const fs = 48000;
    const f0 = 100;
    for (const q of [30, 100]) {
      for (let h = 1; h < 4; h++) expect(db(resonatorBankResponse(fs, (h + 0.5) * f0, f0, 4, q))).toBeLessThan(-20);
      expect(db(resonatorBankResponse(fs, f0 / 2, f0, 4, q))).toBeLessThan(-25);
      expect(db(resonatorBankResponse(fs, 2000, f0, 4, q))).toBeLessThan(-30);
      expect(db(resonatorBankResponse(fs, 15000, f0, 4, q))).toBeLessThan(-50);
      expect(cAbs(resonatorBankResponse(fs, 0, f0, 4, q))).toBeLessThan(1e-12);
    }
    // never more than ~1 dB above the centres anywhere (edge effect just above the top harmonic)
    for (const harmonics of [1, 4, 8]) {
      let peak = -Infinity;
      for (let f = 10; f < fs / 2; f *= 1.001) peak = Math.max(peak, db(resonatorBankResponse(fs, f, f0, harmonics, 30)));
      expect(peak).toBeLessThan(1.1);
    }
  });

  it('skips harmonics at or above 0.45·fs', () => {
    const d = designResonatorBank(32000, 2000, 8, 30);
    expect(Array.from(d.harmonicsHz)).toEqual([2000, 4000, 6000, 8000, 10000, 12000, 14000]);
  });

  it('a Q below minTonalQ(harmonics) is raised to it: still 0 dB and 0° at every harmonic', () => {
    // Below the limit the overlap cannot be corrected (formerly a plain fallback bank, up to −7 dB and ±40°).
    expect(Array.from(TONAL_MIN_Q)).toEqual([2, 2, 2, 2, 2, 3, 4, 5, 6]);
    for (const fs of [44100, 48000, 192000]) {
      for (const f0 of [20, 31.7, 100, 447.7, 2000]) {
        for (let harmonics = 1; harmonics <= RANGES.harmonics.max; harmonics++) {
          const qMin = minTonalQ(harmonics);
          for (const q of [RANGES.q.min, qMin, qMin * 1.3]) {
            const d = designResonatorBank(fs, f0, harmonics, q);
            expect(d.converged).toBe(true);
            expect(d.q).toBe(Math.max(q, qMin));
            for (let h = 1; h <= d.harmonicsHz.length; h++) {
              const r = resonatorBankResponse(fs, h * f0, f0, harmonics, q);
              expect(Math.abs(db(r))).toBeLessThan(0.001);
              expect(Math.abs(deg(r))).toBeLessThan(0.01);
            }
          }
        }
      }
    }
    // clampParams applies the same limit, so the UI shows the Q actually used.
    const p = clampParams({ tonal: { f0: 100, harmonics: 8, q: 2 } });
    expect(p.tonal.q).toBe(6);
    expect(clampParams({ tonal: { f0: 100, harmonics: 4, q: 2 } }).tonal.q).toBe(2);
    expect(minTonalQ(Number.NaN)).toBe(6);
    expect(minTonalQ(99)).toBe(6);
  });

  it('clamps invalid parameters', () => {
    const d = designResonatorBank(48000, Number.NaN, 99, -1);
    expect(d.harmonicsHz.length).toBe(8);
    expect(d.gains.every(Number.isFinite)).toBe(true);
  });
});

describe('ResonatorBank', () => {
  it('time-domain output matches resonatorBankResponse (0 dB, 0° ± 0.5° at the centres)', () => {
    const fs = 48000;
    const f0 = 123.4;
    const bank = new ResonatorBank(fs);
    bank.configure(f0, 4, 30);
    for (const f of [f0, 2 * f0, 3 * f0, 4 * f0, 1.5 * f0, 333]) {
      const m = measure(bank, fs, f, 2.5);
      const h = resonatorBankResponse(fs, f, f0, 4, 30);
      expect(Math.abs(db(m) - db(h))).toBeLessThan(0.01);
      expect(Math.abs(deg(m) - deg(h))).toBeLessThan(0.05);
      if (Math.abs(f / f0 - Math.round(f / f0)) < 1e-9) {
        expect(Math.abs(db(m))).toBeLessThan(0.05);
        expect(Math.abs(deg(m))).toBeLessThan(0.5);
      }
    }
  });

  it('passes little broadband noise', () => {
    const fs = 48000;
    const bank = new ResonatorBank(fs);
    bank.configure(100, 4, 30);
    const rnd = mulberry32(11);
    let pin = 0;
    let pout = 0;
    for (let i = 0; i < 4 * fs; i++) {
      const x = rnd() * 2 - 1;
      const y = bank.process(x);
      if (i > fs) {
        pin += x * x;
        pout += y * y;
      }
    }
    expect(10 * Math.log10(pout / pin)).toBeLessThan(-20);
  });

  it('reconfiguration keeps running bands and is skipped when unchanged', () => {
    const bank = new ResonatorBank(48000);
    bank.configure(100, 4, 30);
    const d1 = bank.design;
    bank.configure(100, 4, 30);
    expect(bank.design).toBe(d1);
    for (let i = 0; i < 1000; i++) bank.process(Math.sin(i * 0.013));
    bank.configure(100.5, 4, 30);
    expect(bank.design).not.toBe(d1);
    const y = bank.process(0);
    expect(Number.isFinite(y)).toBe(true);
    expect(Math.abs(y)).toBeGreaterThan(0); // states were kept
  });

  it('warm-started reconfiguration (f0 tracking) gives the same bank as a cold design', () => {
    const fs = 48000;
    const bank = new ResonatorBank(fs);
    let f0 = 100;
    for (let i = 0; i < 50; i++) {
      f0 += 0.37;
      bank.configure(f0, 8, 30);
    }
    const warm = bank.design!;
    const cold = designResonatorBank(fs, f0, 8, 30);
    expect(warm.converged).toBe(true);
    for (let i = 0; i < 8; i++) {
      expect(Math.abs(warm.gains[i] - cold.gains[i])).toBeLessThan(1e-8);
      expect(Math.abs(warm.ratios[i] - cold.ratios[i])).toBeLessThan(1e-10);
    }
    for (let h = 1; h <= 8; h++) {
      const r = resonatorBankResponse(fs, h * f0, f0, 8, 30);
      expect(Math.abs(db(r))).toBeLessThan(0.001);
    }
  });

  it('NaN input is treated as silence', () => {
    const bank = new ResonatorBank(48000);
    expect(bank.process(Number.NaN)).toBe(0);
    for (let i = 0; i < 100; i++) expect(Number.isFinite(bank.process(Math.sin(i)))).toBe(true);
  });
});

describe('ResonatorBank live changes', () => {
  const fs = 48000;

  /** Peak of y above 2 kHz (zero-phase 8th-order high-pass) in [from, to), dB re the input peak. */
  function clickDb(y: Float64Array, from: number, to: number): number {
    const e = Float64Array.from(y);
    for (const q of [0.5411961001461969, 1.3065629648763766]) {
      const c = highpass(fs, 2000, q);
      let s = new Biquad(c);
      for (let i = 0; i < e.length; i++) e[i] = s.process(e[i]);
      s = new Biquad(c);
      for (let i = e.length - 1; i >= 0; i--) e[i] = s.process(e[i]);
    }
    let pk = 0;
    for (let i = from; i < to; i++) pk = Math.max(pk, Math.abs(e[i]));
    return 20 * Math.log10(pk);
  }

  it('Q ×3 / ÷3 and f0 ±3 % glide click-free (nothing above 2 kHz) and settle on the new design', () => {
    const changes: [number, number, number][][] = [
      [[100, 4, 30], [100, 4, 10]],
      [[100, 4, 10], [100, 4, 30]],
      [[100, 8, 200], [100, 8, 200 / 3]],
      [[100, 4, 30], [103, 4, 30]],
      [[103, 4, 10], [100, 4, 10]],
    ];
    for (const [from, to] of changes) {
      for (const f of [50, 100, 400]) {
        const bank = new ResonatorBank(fs);
        bank.configure(...from);
        const n0 = 2 * fs;
        const y = new Float64Array(n0 + fs / 2);
        const w = (2 * Math.PI * f) / fs;
        for (let i = 0; i < y.length; i++) {
          if (i === n0) bank.configure(...to);
          y[i] = bank.process(Math.sin(w * i));
        }
        expect(clickDb(y, n0 - 240, n0 + 4800), `${from} → ${to} at ${f} Hz`).toBeLessThan(-100);
      }
      const bank = new ResonatorBank(fs);
      bank.configure(...from);
      for (let i = 0; i < fs; i++) bank.process(Math.sin(i * 0.01));
      bank.configure(...to);
      for (let i = 0; i < 960; i++) bank.process(Math.sin(i * 0.01));
      // the glide is over: the time-domain response is the new design's (measure() resets, which changes nothing now)
      const f0 = to[0];
      const m = measure(bank, fs, 2 * f0, 3);
      const h = resonatorBankResponse(fs, 2 * f0, ...to);
      expect(Math.abs(db(m) - db(h))).toBeLessThan(0.01);
      expect(Math.abs(deg(m) - deg(h))).toBeLessThan(0.05);
    }
  });

  it('a bank at rest (new or reset) applies a new design at once', () => {
    const a = new ResonatorBank(fs);
    const b = new ResonatorBank(fs);
    a.configure(90, 3, 20);
    b.configure(90, 3, 20, 0);
    for (let i = 0; i < 4000; i++) expect(a.process(Math.sin(i * 0.02))).toBe(b.process(Math.sin(i * 0.02)));
    a.reset();
    b.reset();
    a.configure(95, 3, 25);
    b.configure(95, 3, 25, 0);
    for (let i = 0; i < 4000; i++) expect(a.process(Math.sin(i * 0.02))).toBe(b.process(Math.sin(i * 0.02)));
  });

  it('reset() completes a glide: the bank is then sample-identical to a new one', () => {
    const a = new ResonatorBank(fs);
    a.configure(100, 4, 30);
    for (let i = 0; i < 3000; i++) a.process(Math.sin(i * 0.013));
    a.configure(120, 4, 10);
    a.process(0.2);
    a.reset();
    const b = new ResonatorBank(fs);
    b.configure(120, 4, 10);
    for (let i = 0; i < 4000; i++) expect(a.process(Math.sin(i * 0.013))).toBe(b.process(Math.sin(i * 0.013)));
  });

  it('a harmonic crossing 0.45·fs fades out over the glide instead of dropping out', () => {
    const fsLow = 32000; // 0.45·fs = 14400 Hz: the 8th harmonic of 1790 Hz is in, that of 1810 Hz is out
    const top = 8 * 1790;
    const run = (ramp: number | undefined) => {
      const bank = new ResonatorBank(fsLow);
      bank.configure(1790, 8, 30);
      const w = (2 * Math.PI * top) / fsLow;
      let i = 0;
      for (; i < fsLow; i++) bank.process(Math.sin(w * i));
      if (ramp === undefined) bank.configure(1810, 8, 30);
      else bank.configure(1810, 8, 30, ramp);
      expect(bank.design!.harmonicsHz.length).toBe(7);
      // peak over the first 1 ms and over 30–40 ms after the change
      let early = 0;
      let late = 0;
      for (let k = 0; k < 0.04 * fsLow; k++, i++) {
        const y = Math.abs(bank.process(Math.sin(w * i)));
        if (k < 0.001 * fsLow) early = Math.max(early, y);
        if (k >= 0.03 * fsLow) late = Math.max(late, y);
      }
      return { early, late };
    };
    const glide = run(undefined);
    expect(glide.early).toBeGreaterThan(0.9); // still the full tone right after the change
    expect(glide.late).toBeLessThan(0.15); // gone after the glide (only the 7th band's skirt is left)
    const cut = run(0);
    expect(cut.early).toBeLessThan(0.15); // the band is dropped at once: a step
  });
});
