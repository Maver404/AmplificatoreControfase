import { describe, expect, it } from 'vitest';
import {
  Biquad,
  COEFF_RAMP_MS,
  bandpass,
  coeffRampSamples,
  highpass,
  lowpass,
  onePoleHighpass,
  onePoleLowpass,
  rampShape,
} from '../../src/dsp/biquad.ts';
import type { BiquadCoeffs } from '../../src/dsp/biquad.ts';
import { besselLowpassSections } from '../../src/dsp/bessel-lp.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

/** The previous implementation (transposed direct form II, coefficients switched at once): the reference. */
class Tdf2 {
  private z1 = 0;
  private z2 = 0;
  constructor(public c: BiquadCoeffs) {}
  process(x: number): number {
    const c = this.c;
    const y = c.b0 * x + this.z1;
    this.z1 = c.b1 * x - c.a1 * y + this.z2;
    this.z2 = c.b2 * x - c.a2 * y;
    return y;
  }
}

// Double-double arithmetic (≈ 32 significant digits) for an exact-enough reference.
type DD = [number, number];
const twoSum = (a: number, b: number): DD => {
  const s = a + b;
  const bb = s - a;
  return [s, a - (s - bb) + (b - bb)];
};
const split = (a: number): DD => {
  const c = 134217729 * a;
  const h = c - (c - a);
  return [h, a - h];
};
const twoProd = (a: number, b: number): DD => {
  const p = a * b;
  const [ah, al] = split(a);
  const [bh, bl] = split(b);
  return [p, ah * bh - p + ah * bl + al * bh + al * bl];
};
const ddAdd = (x: DD, y: DD): DD => {
  const [s, e] = twoSum(x[0], y[0]);
  return twoSum(s, e + x[1] + y[1]);
};
const ddMul = (x: DD, b: number): DD => {
  const [p, e] = twoProd(x[0], b);
  return twoSum(p, e + x[1] * b);
};

/** Direct form I in double-double with the same (double) coefficients. */
class ExactDf1 {
  private x1 = 0;
  private x2 = 0;
  private y1: DD = [0, 0];
  private y2: DD = [0, 0];
  constructor(private readonly c: BiquadCoeffs) {}
  process(x: number): number {
    const c = this.c;
    let acc = ddAdd(ddAdd(twoProd(c.b0, x), twoProd(c.b1, this.x1)), twoProd(c.b2, this.x2));
    acc = ddAdd(ddAdd(acc, ddMul(this.y1, -c.a1)), ddMul(this.y2, -c.a2));
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = acc;
    return acc[0] + acc[1];
  }
}

const noise = (seed: number) => {
  const r = mulberry32(seed);
  return () => r() * 2 - 1;
};

/** Low-cutoff and high-Q sections at high sample rates: the hardest cases for rounding. */
const HARD: { name: string; fs: number; c: BiquadCoeffs }[] = [
  { name: 'HP2 10 Hz @192k', fs: 192000, c: highpass(192000, 10, Math.SQRT1_2) },
  { name: 'HP1 10 Hz @192k', fs: 192000, c: onePoleHighpass(192000, 10) },
  { name: 'barrier 10 Hz @192k', fs: 192000, c: onePoleLowpass(192000, 10) },
  { name: 'Bessel4 100 Hz s1 @192k', fs: 192000, c: besselLowpassSections(192000, 100, 4)[0] },
  { name: 'Bessel4 100 Hz s2 @192k', fs: 192000, c: besselLowpassSections(192000, 100, 4)[1] },
  { name: 'BP 20 Hz Q200 @192k', fs: 192000, c: bandpass(192000, 20, 200) },
  { name: 'LP 4 kHz @44.1k', fs: 44100, c: lowpass(44100, 4000, 0.8) },
];

describe('Biquad (direct form I)', () => {
  it('matches the previous transposed form II within 1e-9 of the output RMS', () => {
    for (const { name, fs, c } of HARD) {
      for (const sig of ['noise', 'tone'] as const) {
        const nx = noise(3);
        const a = new Biquad(c);
        const b = new Tdf2(c);
        const n = 2 * fs;
        let p = 0;
        let worst = 0;
        for (let i = 0; i < n; i++) {
          const x = sig === 'noise' ? nx() : Math.sin((2 * Math.PI * 37 * i) / fs);
          const ya = a.process(x);
          const yb = b.process(x);
          if (i >= fs) {
            p += yb * yb;
            worst = Math.max(worst, Math.abs(ya - yb));
          }
        }
        const rms = Math.sqrt(p / fs);
        expect(worst / rms, `${name} ${sig}`).toBeLessThan(1e-9);
      }
    }
  });

  it('rounding error at a 10 Hz cutoff and 192 kHz is ≈ 1e-11 of the output RMS (double-double reference)', () => {
    for (const { name, fs, c } of HARD.slice(0, 3)) {
      const nx = noise(9);
      const a = new Biquad(c);
      const r = new ExactDf1(c);
      let e2 = 0;
      let p = 0;
      for (let i = 0; i < 3 * fs; i++) {
        // noise on a DC offset: low-frequency content is what stresses poles near z = 1
        const x = 0.5 + 0.5 * nx();
        const y = a.process(x);
        const ref = r.process(x);
        if (i >= fs) {
          e2 += (y - ref) ** 2;
          p += ref * ref;
        }
      }
      expect(Math.sqrt(e2 / p), name).toBeLessThan(1e-10);
    }
  });

  it('a 10 Hz high-pass at 192 kHz is stable: the impulse response decays and sums to its DC gain (0)', () => {
    const b = new Biquad(highpass(192000, 10, Math.SQRT1_2));
    let sum = b.process(1);
    let last = 0;
    for (let i = 1; i < 5 * 192000; i++) {
      last = b.process(0);
      sum += last;
    }
    expect(Math.abs(last)).toBeLessThan(1e-12);
    expect(Math.abs(sum)).toBeLessThan(1e-9);
  });

  it('NaN resets to silence and denormal outputs are flushed (at rest after silence)', () => {
    const b = new Biquad(lowpass(48000, 100, 0.7));
    for (let i = 0; i < 100; i++) b.process(Math.sin(i));
    expect(b.process(Number.NaN)).toBe(0);
    expect(b.atRest).toBe(true);
    expect(Number.isFinite(b.process(1))).toBe(true);
    for (let i = 0; i < 48000 * 5; i++) b.process(0);
    expect(b.atRest).toBe(true);
  });
});

describe('Biquad coefficient glide', () => {
  const fs = 48000;
  const a = besselLowpassSections(fs, 500, 2)[0];
  const z = besselLowpassSections(fs, 1000, 2)[0];
  const keys = ['b0', 'b1', 'b2', 'a1', 'a2'] as const;
  const running = (c: BiquadCoeffs): Biquad => {
    const b = new Biquad(c);
    for (let i = 0; i < 1000; i++) b.process(Math.sin(i * 0.01));
    return b;
  };

  it('COEFF_RAMP_MS is 20 ms', () => {
    expect(COEFF_RAMP_MS).toBe(20);
    expect(coeffRampSamples(48000)).toBe(960);
    expect(coeffRampSamples(44100)).toBe(882);
  });

  it('setCoeffs without a ramp switches at the next sample', () => {
    const b = running(a);
    b.setCoeffs(z);
    expect(b.ramping).toBe(false);
    expect(b.coeffs).toEqual(z);
  });

  it('glides along the smoothstep and lands exactly on the target after n samples', () => {
    const b = running(a);
    const n = 960;
    b.setCoeffs(z, n);
    expect(b.ramping).toBe(true);
    expect(b.coeffs).toEqual(a); // the change starts with the next sample
    for (let k = 1; k < n; k++) {
      b.process(0.1);
      const w = rampShape(k / n);
      const c = b.coeffs;
      for (const key of keys) expect(Math.abs(c[key] - (a[key] + w * (z[key] - a[key])))).toBeLessThan(1e-15);
    }
    b.process(0.1);
    expect(b.ramping).toBe(false);
    expect(b.coeffs).toEqual(z);
    expect(rampShape(0)).toBe(0);
    expect(rampShape(1)).toBe(1);
    expect(rampShape(0.5)).toBe(0.5);
  });

  it('a new target mid-glide restarts from where it is; the same target again changes nothing', () => {
    const third = besselLowpassSections(fs, 2000, 2)[0];
    const b = running(a);
    const c = running(a);
    b.setCoeffs(z, 100);
    c.setCoeffs(z, 100);
    for (let k = 0; k < 40; k++) {
      expect(b.process(0.3)).toBe(c.process(0.3));
      c.setCoeffs(z, 100); // repeated: no restart
    }
    const mid = b.coeffs;
    b.setCoeffs(third, 100);
    b.process(0.3);
    const w = rampShape(1 / 100);
    for (const key of keys) expect(b.coeffs[key]).toBeCloseTo(mid[key] + w * (third[key] - mid[key]), 15);
    for (let k = 0; k < 99; k++) b.process(0.3);
    expect(b.coeffs).toEqual(third);
  });

  it('a filter at rest switches at once, and reset() completes a glide', () => {
    const fresh = new Biquad(a);
    fresh.setCoeffs(z, 960);
    expect(fresh.ramping).toBe(false);
    expect(fresh.coeffs).toEqual(z);
    const b = running(a);
    b.setCoeffs(z, 960);
    b.process(0.5);
    b.reset();
    expect(b.ramping).toBe(false);
    expect(b.coeffs).toEqual(z);
    expect(b.atRest).toBe(true);
  });

  it('every intermediate filter of a glide is stable (inside the stability triangle)', () => {
    const pairs: [BiquadCoeffs, BiquadCoeffs][] = [
      [highpass(192000, 10, Math.SQRT1_2), highpass(192000, 200, Math.SQRT1_2)],
      [onePoleHighpass(192000, 10), onePoleLowpass(192000, 200)],
      [besselLowpassSections(48000, 100, 4)[1], besselLowpassSections(48000, 4000, 4)[1]],
      [bandpass(48000, 20, 200), bandpass(48000, 2000, 2)],
    ];
    for (const [p, q] of pairs) {
      for (const [from, to] of [[p, q], [q, p]]) {
        const b = new Biquad(from);
        b.process(1);
        b.setCoeffs(to, 500);
        for (let k = 0; k < 500; k++) {
          b.process(0);
          const { a1, a2 } = b.coeffs;
          expect(Math.abs(a2)).toBeLessThan(1);
          expect(Math.abs(a1)).toBeLessThan(1 + a2);
        }
      }
    }
  });

  /** Peak of the output above 2 kHz (zero-phase 8th-order high-pass) around a change, dB re the input peak. */
  function clickDb(y: Float64Array, n0: number): number {
    const e = Float64Array.from(y);
    for (const q of [0.5411961001461969, 1.3065629648763766]) {
      const c = highpass(fs, 2000, q);
      let s = new Biquad(c);
      for (let i = 0; i < e.length; i++) e[i] = s.process(e[i]);
      s = new Biquad(c);
      for (let i = e.length - 1; i >= 0; i--) e[i] = s.process(e[i]);
    }
    let pk = 0;
    for (let i = n0 - 240; i < n0 + 4800; i++) pk = Math.max(pk, Math.abs(e[i]));
    return 20 * Math.log10(pk);
  }

  it('live changes are click-free: ×2 cutoff on a 100 Hz tone leaves nothing above 2 kHz (≤ −100 dB)', () => {
    const cases: [string, (f: number) => BiquadCoeffs[], number, number][] = [
      ['HP2', (f) => [highpass(fs, f, Math.SQRT1_2)], 10, 20],
      ['barrier', (f) => [onePoleLowpass(fs, f)], 200, 100],
      ['Bessel4', (f) => besselLowpassSections(fs, f, 4), 500, 1000],
      ['Bessel2', (f) => besselLowpassSections(fs, f, 2), 2000, 1000],
    ];
    const n0 = fs;
    const n = n0 + fs / 2;
    for (const [name, mk, f1, f2] of cases) {
      const glide = mk(f1).map((c) => new Biquad(c));
      const old = mk(f1).map((c) => new Tdf2(c));
      const yg = new Float64Array(n);
      const yo = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (i === n0) {
          mk(f2).forEach((c, k) => {
            glide[k].setCoeffs(c, coeffRampSamples(fs));
            old[k].c = c;
          });
        }
        const x = Math.sin((2 * Math.PI * 100 * i) / fs);
        yg[i] = glide.reduce((v, b) => b.process(v), x);
        yo[i] = old.reduce((v, b) => b.process(v), x);
      }
      expect(clickDb(yg, n0), name).toBeLessThan(-100);
      // the previous behaviour (transposed form II, coefficients switched at once) clicked
      expect(clickDb(yo, n0), name).toBeGreaterThan(-80);
    }
  });
});
