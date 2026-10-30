import { describe, expect, it } from 'vitest';
import { Biquad, biquadResponse, highpass } from '../../src/dsp/biquad.ts';
import { GainRamp } from '../../src/dsp/gain-ramp.ts';
import { cAbs, cArg, cMul } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import {
  TRIM_WEIGHTS,
  TrimRamp,
  anyTrim,
  quadratureCompanion,
  trimBasis,
  trimFactor,
  trimForTarget,
  trimWeights,
  trimmedBankResponseOf,
} from '../../src/dsp/harmonic-trims.ts';
import {
  BIG_JUMP_DB,
  DEFAULT_PARAMS,
  MAX_HARMONICS,
  MAX_RISE_DB_PER_S,
  RANGES,
  clampParams,
  minTonalQ,
  neutralTrims,
  withoutTrims,
} from '../../src/dsp/params.ts';
import type { HarmonicTrim } from '../../src/dsp/params.ts';
import { ResonatorBank, designResonatorBank, resonatorBankResponse } from '../../src/dsp/resonator-bank.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const db = (h: Complex): number => 20 * Math.log10(cAbs(h));
const deg = (h: Complex): number => (cArg(h) * 180) / Math.PI;
const wrapDeg = (d: number): number => ((((d + 180) % 360) + 360) % 360) - 180;

/** Trims with harmonic h (1-based) set, the others neutral. */
const only = (h: number, gainDb: number, phaseDeg: number): HarmonicTrim[] => {
  const t = neutralTrims();
  t[h - 1] = { gainDb, phaseDeg };
  return t;
};

const randomTrims = (rnd: () => number): HarmonicTrim[] =>
  Array.from({ length: MAX_HARMONICS }, () => ({ gainDb: -30 + 36 * rnd(), phaseDeg: -180 + 360 * rnd() }));

describe('ChannelParams.harmonicTrims (params.ts)', () => {
  it('defaults: MAX_HARMONICS (= harmonics max) neutral entries, one array per channel', () => {
    expect(MAX_HARMONICS).toBe(RANGES.harmonics.max);
    expect(RANGES.trimGainDb).toEqual({ min: -30, max: 6 });
    expect(RANGES.trimPhaseDeg).toEqual({ min: -180, max: 180 });
    const p = clampParams(undefined);
    for (const c of [...p.channels, ...DEFAULT_PARAMS.channels]) {
      expect(c.harmonicTrims).toEqual(Array.from({ length: 8 }, () => ({ gainDb: 0, phaseDeg: 0 })));
    }
    expect(DEFAULT_PARAMS.channels[0].harmonicTrims).not.toBe(DEFAULT_PARAMS.channels[1].harmonicTrims);
    expect(neutralTrims()).not.toBe(neutralTrims());
  });

  it('clampParams deep-copies, clamps, fills missing/NaN with 0, pads and truncates to 8', () => {
    const raw = clampParams(undefined);
    const t0: unknown[] = [
      { gainDb: 20, phaseDeg: 270 },
      { gainDb: -50, phaseDeg: -300 },
      { gainDb: Number.NaN, phaseDeg: Number.POSITIVE_INFINITY },
      { gainDb: 3.5 },
      null,
      { phaseDeg: -12.5 },
      'x',
      { gainDb: -6, phaseDeg: 45 },
      { gainDb: 1, phaseDeg: 1 },
    ];
    raw.channels[0].harmonicTrims = t0 as HarmonicTrim[];
    raw.channels[1].harmonicTrims = [{ gainDb: -3, phaseDeg: 10 }];
    const p = clampParams(raw);
    expect(p.channels[0].harmonicTrims).toEqual([
      { gainDb: 6, phaseDeg: 180 },
      { gainDb: -30, phaseDeg: -180 },
      { gainDb: 0, phaseDeg: 0 },
      { gainDb: 3.5, phaseDeg: 0 },
      { gainDb: 0, phaseDeg: 0 },
      { gainDb: 0, phaseDeg: -12.5 },
      { gainDb: 0, phaseDeg: 0 },
      { gainDb: -6, phaseDeg: 45 },
    ]);
    expect(p.channels[1].harmonicTrims.length).toBe(8);
    expect(p.channels[1].harmonicTrims[0]).toEqual({ gainDb: -3, phaseDeg: 10 });
    expect(p.channels[1].harmonicTrims.slice(1).every((t) => t.gainDb === 0 && t.phaseDeg === 0)).toBe(true);
    // a deep copy: nothing is shared with the input or between calls
    const q = clampParams(p);
    expect(q.channels[1].harmonicTrims).not.toBe(p.channels[1].harmonicTrims);
    expect(q.channels[1].harmonicTrims[0]).not.toBe(p.channels[1].harmonicTrims[0]);
    q.channels[1].harmonicTrims[0].gainDb = -20;
    expect(p.channels[1].harmonicTrims[0].gainDb).toBe(-3);
    // not an array → neutral
    const bad = clampParams({ channels: [{ harmonicTrims: 5 }, { harmonicTrims: { 0: { gainDb: 3 } } }] } as never);
    expect(bad.channels[0].harmonicTrims).toEqual(neutralTrims());
    expect(bad.channels[1].harmonicTrims).toEqual(neutralTrims());
    // withoutTrims: a clamped copy with neutral trims, the input untouched
    const w = withoutTrims(p);
    expect(w.channels[0].harmonicTrims).toEqual(neutralTrims());
    expect({ ...w, channels: null }).toEqual({ ...p, channels: null });
    expect(p.channels[0].harmonicTrims[0]).toEqual({ gainDb: 6, phaseDeg: 180 });
  });
});

describe('harmonic-trims.ts building blocks', () => {
  it('the quadrature companion (an allpass after the band) is −j·band-pass at the centre and |band-pass| everywhere', () => {
    for (const fs of [44100, 48000, 192000]) {
      for (const [f, q] of [[20, 2], [100, 30], [2000, 6], [9000, 200]]) {
        const d = designResonatorBank(fs, f, 1, q);
        const nu = d.ratios[0] * d.harmonicsHz[0];
        const c = quadratureCompanion(fs, nu);
        expect(d.quadCoeffs[0]).toEqual(c);
        // a first-order allpass: unit magnitude at every frequency, 90° lag at the band centre
        for (const g of [1, nu / 10, nu / 2, nu, 2 * nu, 0.45 * fs]) expect(cAbs(biquadResponse(c, g, fs))).toBeCloseTo(1, 12);
        const r = biquadResponse(c, nu, fs);
        expect(Math.abs(r.re)).toBeLessThan(1e-9);
        expect(Math.abs(r.im + 1)).toBeLessThan(1e-9);
      }
    }
  });

  it('a trim keeps the skirts of the tonal chain: below f0 and above the harmonics ≤ |T| + 5 dB (was up to +26 dB)', () => {
    // The companion has the band-pass's own skirts (|AP| = 1), so a trim does not open the tonal chain to
    // wind and rumble: one band alone changes by |Re T + Im T·AP| ≤ √2·|T|, the overlap correction adds ≤ 2 dB
    // (4 harmonics at Q 2). The former companion (band-pass poles, low-pass numerator, flat at ~1/Q below the
    // band) gave +14 dB at 20 Hz for +6 dB / 90° (4 harmonics, Q 30) and +26 dB at 5 Hz (Q 2).
    for (const [h, q] of [[1, 30], [4, 2], [4, 30], [8, 6]] as const) {
      const fs = 48000;
      const f0 = 100;
      for (const t of [{ gainDb: 6, phaseDeg: 90 }, { gainDb: 0, phaseDeg: 180 }, { gainDb: 0, phaseDeg: 45 }, { gainDb: 6, phaseDeg: -150 }]) {
        const trims = only(1, t.gainDb, t.phaseDeg);
        const lim = Math.max(0, t.gainDb) + 5;
        for (const f of [2, 5, 10, 20, 35, 50, 2 * (h + 1) * f0, 5000, 20000]) {
          const rise = db(resonatorBankResponse(fs, f, f0, h, q, trims)) - db(resonatorBankResponse(fs, f, f0, h, q));
          expect(rise, `H${h} Q${q} ${t.gainDb} dB / ${t.phaseDeg}° at ${f} Hz`).toBeLessThanOrEqual(lim);
        }
      }
    }
  });

  it('weights realise N_h·T_h at every used harmonic over the whole parameter range (well conditioned)', () => {
    const rnd = mulberry32(5);
    let worst = 0;
    let worstInv = 0;
    for (const fs of [22050, 44100, 48000, 96000, 192000]) {
      for (const f0 of [20, 50, 100, 300, 1000, 2000]) {
        for (let h = 1; h <= 8; h++) {
          for (const q of [minTonalQ(h), 10, 30, 200]) {
            const d = designResonatorBank(fs, f0, h, q);
            const b = trimBasis(d, fs);
            expect(b.m).toBe(d.coeffs.length);
            expect(b.inv).not.toBeNull();
            for (const v of b.inv!) worstInv = Math.max(worstInv, Math.abs(v));
            const trims = randomTrims(rnd);
            const w = new Float64Array(TRIM_WEIGHTS);
            expect(trimWeights(b, trims, w, new Float64Array(TRIM_WEIGHTS))).toBe(true);
            for (let k = 0; k < b.m; k++) {
              const got = trimmedBankResponseOf(d, w, d.harmonicsHz[k], fs);
              const want = cMul({ re: b.nRe[k], im: b.nIm[k] }, trimFactor(trims[k]));
              worst = Math.max(worst, cAbs({ re: got.re - want.re, im: got.im - want.im }) / cAbs(want));
              // the neutral bank is 0 dB / 0° there, so the trim is the response
              expect(Math.abs(db(got) - trims[k].gainDb)).toBeLessThan(1e-4);
              expect(Math.abs(wrapDeg(-deg(got) - trims[k].phaseDeg))).toBeLessThan(1e-3);
            }
            for (let i = b.m; i < MAX_HARMONICS; i++) {
              expect(w[i]).toBe(0);
              expect(w[MAX_HARMONICS + i]).toBe(0);
            }
          }
        }
      }
    }
    expect(worst).toBeLessThan(1e-11);
    expect(worstInv).toBeLessThan(5);
  });

  it('neutral trims → weights exactly 0; trims beyond the design are ignored; NaN counts as 0', () => {
    const fs = 22050;
    const d = designResonatorBank(fs, 1500, 8, 20); // 0.45·fs = 9922 Hz: 6 harmonics used
    const b = trimBasis(d, fs);
    expect(b.m).toBe(6);
    const w = new Float64Array(TRIM_WEIGHTS).fill(7);
    const r = new Float64Array(TRIM_WEIGHTS);
    expect(trimWeights(b, neutralTrims(), w, r)).toBe(false);
    expect(w.every((v) => v === 0)).toBe(true);
    const beyond = neutralTrims();
    beyond[6] = { gainDb: 6, phaseDeg: 90 };
    beyond[7] = { gainDb: -30, phaseDeg: -90 };
    expect(anyTrim(beyond, b.m)).toBe(false);
    expect(trimWeights(b, beyond, w, r)).toBe(false);
    expect(w.every((v) => v === 0)).toBe(true);
    expect(trimWeights(b, only(2, Number.NaN, Number.NaN), w, r)).toBe(false);
    expect(trimWeights(b, only(2, 0, 40), w, r)).toBe(true);
  });

  it('trimForTarget inverts trimFactor and clamps to the trim ranges', () => {
    const rnd = mulberry32(8);
    for (let i = 0; i < 50; i++) {
      const cur = { re: rnd() - 0.5, im: rnd() - 0.5 };
      const t = { gainDb: -30 + 36 * rnd(), phaseDeg: -179 + 358 * rnd() };
      const back = trimForTarget(cMul(cur, trimFactor(t)), cur);
      expect(back.gainDb).toBeCloseTo(t.gainDb, 9);
      expect(back.phaseDeg).toBeCloseTo(t.phaseDeg, 9);
    }
    // a lag of 30° (desired = current·e^{−j30°}) is +30
    const lag = trimForTarget({ re: Math.cos(-Math.PI / 6), im: Math.sin(-Math.PI / 6) }, { re: 1, im: 0 });
    expect(lag.gainDb).toBeCloseTo(0, 12);
    expect(lag.phaseDeg).toBeCloseTo(30, 12);
    expect(trimForTarget({ re: 10, im: 0 }, { re: 1, im: 0 }).gainDb).toBe(6);
    expect(trimForTarget({ re: 0.001, im: 0 }, { re: 1, im: 0 }).gainDb).toBe(-30);
    expect(trimForTarget({ re: 1, im: 0 }, { re: 0, im: 0 })).toEqual({ gainDb: 0, phaseDeg: 0 });
    expect(trimForTarget({ re: 0, im: 0 }, { re: 1, im: 0 })).toEqual({ gainDb: 0, phaseDeg: 0 });
  });
});

describe('TrimRamp (the gain rise rule per harmonic)', () => {
  const fs = 48000;

  /** Runs scheduled steps from `from` to `to` samples, returning [clock, applied gain of harmonic h] after each. */
  const steps = (r: TrimRamp, from: number, to: number, h: number, hold = false): [number, number][] => {
    const out: [number, number][] = [];
    for (let c = from; c <= to; c++) {
      if (r.due(c)) {
        r.step(c, hold);
        out.push([c, r.applied[h - 1].gainDb]);
      }
    }
    return out;
  };

  it('−30 → +6 dB: 30 dB at 6 dB/s, then the last 6 dB (≥ 5 s in all); decreases apply at once', () => {
    const r = new TrimRamp(fs);
    r.jump(only(3, -30, 20), 0);
    expect(r.applied[2]).toEqual({ gainDb: -30, phaseDeg: 20 });
    expect(r.setTarget(only(3, 6, 20), fs, false)).toBe(false); // 36 dB gap: nothing at once
    expect(r.pending).toBe(true);
    const path = steps(r, fs, 8 * fs, 3);
    const done = path.find(([, g]) => g === 6)!;
    expect((done[0] - fs) / fs).toBeGreaterThanOrEqual(5);
    expect((done[0] - fs) / fs).toBeLessThan(5.05);
    // over any interval T: ≤ BIG_JUMP_DB + 6·T, and each step ≤ 0.12 dB until the last BIG_JUMP_DB
    for (let i = 0; i < path.length; i++) {
      for (let j = i + 1; j < path.length; j++) {
        const t = (path[j][0] - path[i][0]) / fs;
        expect(path[j][1] - path[i][1]).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S * t + 1e-9);
      }
    }
    // 0.12 dB steps while more than BIG_JUMP_DB remains (the last ≈ 6 dB come from the budget)
    for (let i = 1; i < path.length; i++) if (path[i][1] <= 6 - BIG_JUMP_DB) expect(path[i][1] - path[i - 1][1]).toBeLessThan(0.121);
    expect(r.pending).toBe(false);
    // decrease: at once
    expect(r.setTarget(only(3, -30, 20), 9 * fs, false)).toBe(true);
    expect(r.applied[2].gainDb).toBe(-30);
    expect(r.pending).toBe(false);
  });

  it('increases ≤ BIG_JUMP_DB and phase changes apply at once; the budget stops a series of small steps', () => {
    const r = new TrimRamp(fs);
    r.jump(neutralTrims(), 0);
    expect(r.setTarget(only(1, 5, -170), 0, false)).toBe(true);
    expect(r.applied[0]).toEqual({ gainDb: 5, phaseDeg: -170 });
    // +3 dB more right away: 1 dB of budget left → 6 dB now, the rest at 6 dB/s
    r.setTarget(only(1, -1, 0), 0, false);
    expect(r.applied[0]).toEqual({ gainDb: -1, phaseDeg: 0 });
    let clock = 0;
    let g = -30;
    r.jump(only(1, -30, 0), 0);
    // +5 dB every 100 ms: the average rise is limited to 6 dB/s after the first BIG_JUMP_DB
    for (let k = 1; k <= 7; k++) {
      clock = k * 0.1 * fs;
      g += 5;
      r.setTarget(only(1, Math.min(6, g), 0), clock, false);
      steps(r, clock + 1, clock + 0.1 * fs - 1, 1);
      const t = (clock + 0.1 * fs) / fs;
      expect(r.applied[0].gainDb + 30).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S * t + 1e-9);
    }
  });

  it('while held, increases wait (the budget refills) and decreases and phase go on', () => {
    const r = new TrimRamp(fs);
    r.jump(only(2, -20, 0), 0);
    expect(r.setTarget(only(2, -16, 90), 0, true)).toBe(true); // only the phase
    expect(r.applied[1]).toEqual({ gainDb: -20, phaseDeg: 90 });
    steps(r, 1, 2 * fs, 2, true);
    expect(r.applied[1].gainDb).toBe(-20);
    // released: the 4 dB come from the full budget at the next step
    const p = steps(r, 2 * fs + 1, 3 * fs, 2);
    expect(p[0][1]).toBe(-16);
    r.setTarget(only(2, -25, 90), 3 * fs, true);
    expect(r.applied[1].gainDb).toBe(-25);
    // a big rise held for 3 s then released continues at 6 dB/s (no catch-up beyond the budget)
    r.setTarget(only(2, 6, 90), 3 * fs, true);
    steps(r, 3 * fs + 1, 6 * fs, 2, true);
    expect(r.applied[1].gainDb).toBe(-25);
    const q = steps(r, 6 * fs + 1, 7 * fs, 2);
    expect(q[q.length - 1][1]).toBeLessThan(-25 + 6 + 0.13);
  });

  it('shares the channel GainRamp budget: gain + trim at a harmonic rise by ≤ 6 dB + 6 dB/s, other harmonics keep theirs', () => {
    const g = new GainRamp(fs, -20);
    const r = new TrimRamp(fs, g);
    r.jump(neutralTrims(), 0);
    // harmonic 1 takes the whole budget at once: the gain (which also raises harmonic 1) is left with none
    expect(r.setTarget(only(1, 6, 0), 0, false)).toBe(true);
    expect(r.applied[0].gainDb).toBe(6);
    expect(g.budgetDb).toBe(0);
    g.setTargetDb(-14);
    for (let i = 0; i < fs / 10; i++) g.next();
    expect(g.currentDb).toBeCloseTo(-20 + MAX_RISE_DB_PER_S * 0.1, 3);
    // harmonic 2 still had its own 6 dB: 0.6 dB of gain + 6 dB of trim = 6 dB + 6 dB/s · 0.1 s
    const t2 = only(1, 6, 0);
    t2[1] = { gainDb: 6, phaseDeg: 0 };
    r.setTarget(t2, fs / 10, false);
    expect(r.applied[1].gainDb).toBeCloseTo(6, 3);
    // the other way round: a gain rise first leaves a trim only what is left of the budget
    const g2 = new GainRamp(fs, -20);
    const r2 = new TrimRamp(fs, g2);
    r2.jump(neutralTrims(), 0);
    g2.setTargetDb(-16);
    for (let i = 0; i < fs / 2; i++) g2.next();
    expect(g2.currentDb).toBeCloseTo(-16, 3);
    expect(g2.budgetDb).toBeCloseTo(5, 2);
    r2.setTarget(only(3, 6, 0), fs / 2, false);
    // 4 dB of gain in 0.5 s: 6 + 3 − 4 = 5 dB left for harmonic 3 (the budget refilled 3 dB meanwhile), the rest later
    expect(r2.applied[2].gainDb).toBeCloseTo(5, 2);
    expect(r2.applied[2].gainDb + 4).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S * 0.5 + 1e-6);
    expect(r2.pending).toBe(true);
    expect(g2.budgetDb).toBeLessThan(1e-3);
  });

  it('unchanged targets are a no-op', () => {
    const r = new TrimRamp(fs);
    r.jump(only(4, -10, 5), 0);
    expect(r.setTarget(only(4, -10, 5), 100, false)).toBe(false);
    r.setTarget(only(4, 6, 5), 200, false);
    const g = r.applied[3].gainDb;
    expect(r.setTarget(only(4, 6, 5), 300, false)).toBe(false);
    expect(r.applied[3].gainDb).toBe(g);
  });
});

describe('ResonatorBank with harmonic trims', () => {
  const fs = 48000;

  /** Steady-state response of out[ch] to sin(ωn) over the last 0.5 s of `seconds` (bank reset first). */
  function measureOut(bank: ResonatorBank, ch: number, f: number, seconds = 2.5): Complex {
    bank.reset();
    const w = (2 * Math.PI * f) / fs;
    const n = Math.round(seconds * fs);
    const start = n - Math.round(0.5 * fs);
    let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
    for (let i = 0; i < n; i++) {
      const s = Math.sin(w * i);
      bank.process(s);
      const y = bank.out[ch];
      if (i >= start) {
        const c = Math.cos(w * i);
        ss += s * s; sc += s * c; cc += c * c; ys += y * s; yc += y * c;
      }
    }
    const det = ss * cc - sc * sc;
    return { re: (ys * cc - yc * sc) / det, im: (yc * ss - ys * sc) / det };
  }

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

  it('neutral trims: out[0] and out[1] are exactly the neutral output', () => {
    const bank = new ResonatorBank(fs);
    bank.configure(97, 6, 12);
    const rnd = mulberry32(3);
    bank.setTrims(1, neutralTrims());
    for (let i = 0; i < 20000; i++) {
      const y = bank.process(rnd() - 0.5);
      expect(bank.out[0]).toBe(y);
      expect(bank.out[1]).toBe(y);
    }
    // trims on, then back to neutral: exact again once the glide is over
    bank.setTrims(0, only(2, 4, 30));
    for (let i = 0; i < 3000; i++) bank.process(rnd() - 0.5);
    bank.setTrims(0, neutralTrims());
    for (let i = 0; i < 960; i++) bank.process(rnd() - 0.5);
    expect(bank.trimGliding(0)).toBe(false);
    for (let i = 0; i < 5000; i++) {
      const y = bank.process(rnd() - 0.5);
      expect(bank.out[0]).toBe(y);
    }
  });

  it('out[ch] matches resonatorBankResponse(…, trims) at and between the harmonics, channels independent', () => {
    const rnd = mulberry32(21);
    const f0 = 123.4;
    const bank = new ResonatorBank(fs);
    bank.configure(f0, 4, 20);
    const tL = randomTrims(rnd);
    const tR = randomTrims(rnd);
    bank.setTrims(0, tL);
    bank.setTrims(1, tR);
    for (const f of [f0, 2 * f0, 3 * f0, 4 * f0, 1.5 * f0, 60, 900]) {
      for (const [ch, t] of [[0, tL], [1, tR]] as const) {
        const m = measureOut(bank, ch, f);
        const h = resonatorBankResponse(fs, f, f0, 4, 20, t);
        expect(Math.abs(db(m) - db(h)), `${f} Hz ch ${ch}`).toBeLessThan(0.01);
        expect(Math.abs(wrapDeg(deg(m) - deg(h))), `${f} Hz ch ${ch}`).toBeLessThan(0.05);
        const k = Math.round(f / f0);
        if (Math.abs(f / f0 - k) < 1e-9) {
          expect(Math.abs(db(m) - t[k - 1].gainDb)).toBeLessThan(0.01);
          expect(Math.abs(wrapDeg(-deg(m) - t[k - 1].phaseDeg))).toBeLessThan(0.05);
        }
      }
    }
    // without trims the response function is the neutral bank
    expect(resonatorBankResponse(fs, 2 * f0, f0, 4, 20, neutralTrims())).toEqual(resonatorBankResponse(fs, 2 * f0, f0, 4, 20));
  });

  it('trim changes glide over COEFF_RAMP_MS click-free (nothing above 2 kHz) and settle', () => {
    const f0 = 100;
    const changes: [HarmonicTrim[], HarmonicTrim[]][] = [
      [neutralTrims(), only(1, 6, 90)],
      [only(1, 6, 90), only(1, -30, -180)],
      [neutralTrims(), Array.from({ length: 8 }, () => ({ gainDb: 6, phaseDeg: 180 }))],
      [only(4, -12, 60), only(4, 3, -150)],
    ];
    for (const [from, to] of changes) {
      for (const f of [50, 100, 400]) {
        const bank = new ResonatorBank(fs);
        bank.configure(f0, 4, 30);
        bank.setTrims(0, from, 0);
        const n0 = 2 * fs;
        const y = new Float64Array(n0 + fs / 2);
        const w = (2 * Math.PI * f) / fs;
        for (let i = 0; i < y.length; i++) {
          if (i === n0) bank.setTrims(0, to);
          bank.process(Math.sin(w * i));
          y[i] = bank.out[0];
        }
        expect(clickDb(y, n0 - 240, n0 + 4800), `${JSON.stringify(to[0])} at ${f} Hz`).toBeLessThan(-100);
      }
      const bank = new ResonatorBank(fs);
      bank.configure(f0, 4, 30);
      bank.setTrims(0, from, 0);
      for (let i = 0; i < fs; i++) bank.process(Math.sin(i * 0.01));
      bank.setTrims(0, to);
      for (let i = 0; i < 960; i++) bank.process(Math.sin(i * 0.01));
      expect(bank.trimGliding(0)).toBe(false);
      const m = measureOut(bank, 0, 4 * f0);
      const h = resonatorBankResponse(fs, 4 * f0, f0, 4, 30, to);
      expect(Math.abs(db(m) - db(h))).toBeLessThan(0.01);
      expect(Math.abs(wrapDeg(deg(m) - deg(h)))).toBeLessThan(0.05);
    }
  });

  it('an f0 / Q glide carries the trims along click-free and settles on the trimmed design', () => {
    const t = [{ gainDb: 4, phaseDeg: 50 }, { gainDb: -6, phaseDeg: -120 }, { gainDb: 2, phaseDeg: 170 }, ...neutralTrims().slice(3)];
    for (const [from, to] of [[[100, 4, 30], [103, 4, 30]], [[100, 4, 30], [100, 4, 10]]] as [number, number, number][][]) {
      for (const f of [100, 200, 300]) {
        const bank = new ResonatorBank(fs);
        bank.configure(...from);
        bank.setTrims(1, t, 0);
        const n0 = 2 * fs;
        const y = new Float64Array(n0 + fs / 2);
        const w = (2 * Math.PI * f) / fs;
        for (let i = 0; i < y.length; i++) {
          if (i === n0) bank.configure(...to);
          bank.process(Math.sin(w * i));
          y[i] = bank.out[1];
        }
        expect(clickDb(y, n0 - 240, n0 + 4800), `${from} → ${to} at ${f} Hz`).toBeLessThan(-100);
      }
      const bank = new ResonatorBank(fs);
      bank.configure(...from);
      bank.setTrims(1, t, 0);
      for (let i = 0; i < fs; i++) bank.process(Math.sin(i * 0.01));
      bank.configure(...to);
      for (let i = 0; i < 960; i++) bank.process(Math.sin(i * 0.01));
      for (const h of [1, 2, 3]) {
        const m = measureOut(bank, 1, h * to[0], 3);
        expect(Math.abs(db(m) - t[h - 1].gainDb)).toBeLessThan(0.01);
        expect(Math.abs(wrapDeg(-deg(m) - t[h - 1].phaseDeg))).toBeLessThan(0.05);
      }
    }
  });

  it('a trim change during a band fade-out (harmonic crossing 0.45·fs) leaves no step', () => {
    const fsLow = 32000;
    const bank = new ResonatorBank(fsLow);
    bank.configure(1790, 8, 30);
    bank.setTrims(0, Array.from({ length: 8 }, () => ({ gainDb: 6, phaseDeg: 90 })), 0);
    const w = (2 * Math.PI * 8 * 1790) / fsLow;
    let i = 0;
    for (; i < fsLow; i++) bank.process(Math.sin(w * i));
    bank.configure(1810, 8, 30); // 8th band fades out over 640 samples
    let prev = bank.out[0];
    let worst = 0;
    for (let k = 0; k < 2000; k++, i++) {
      if (k === 320) bank.setTrims(0, Array.from({ length: 8 }, () => ({ gainDb: 3, phaseDeg: -60 })));
      bank.process(Math.sin(w * i));
      worst = Math.max(worst, Math.abs(bank.out[0] - prev));
      prev = bank.out[0];
    }
    // a 14.3 kHz tone at ≤ 2 (+6 dB) changes by ≤ 2·2·sin(π·14320/32000) ≈ 3.96 per sample
    expect(worst).toBeLessThan(4);
    expect(bank.trimGliding(0)).toBe(false);
  });

  it('setTrims with the current target does not disturb a glide; reset() completes it', () => {
    const a = new ResonatorBank(fs);
    const b = new ResonatorBank(fs);
    for (const k of [a, b]) k.configure(100, 3, 20);
    const t = only(2, 5, 70);
    for (let i = 0; i < 6000; i++) {
      if (i === 1000) {
        a.setTrims(0, t);
        b.setTrims(0, t);
      }
      if (i === 1300 || i === 1800) b.setTrims(0, t);
      expect(b.process(Math.sin(i * 0.02))).toBe(a.process(Math.sin(i * 0.02)));
      expect(b.out[0]).toBe(a.out[0]);
    }
    const c = new ResonatorBank(fs);
    c.configure(100, 3, 20);
    for (let i = 0; i < 3000; i++) c.process(Math.sin(i * 0.02));
    c.setTrims(0, t);
    c.process(0.1);
    c.reset();
    const d = new ResonatorBank(fs);
    d.configure(100, 3, 20);
    d.setTrims(0, t);
    for (let i = 0; i < 3000; i++) {
      expect(c.process(Math.sin(i * 0.02))).toBe(d.process(Math.sin(i * 0.02)));
      expect(c.out[0]).toBe(d.out[0]);
    }
  });

  it('broadband noise: the trimmed bank passes about |T|² times the neutral power', () => {
    // The same T on every harmonic: within 1 dB at Q 30; where the bands overlap most (8 harmonics at Q 6) the
    // companions' allpass phase, −90° only at each centre, changes the overlap between centres: within 3 dB.
    const rnd = mulberry32(31);
    for (const [h, q, tol] of [[4, 30, 1], [8, 6, 3]]) {
      for (const t of [{ gainDb: 0, phaseDeg: 30 }, { gainDb: 6, phaseDeg: 90 }, { gainDb: -6, phaseDeg: -150 }]) {
        const bank = new ResonatorBank(fs);
        bank.configure(100, h, q);
        bank.setTrims(0, Array.from({ length: 8 }, () => t), 0);
        let pn = 0;
        let pt = 0;
        for (let i = 0; i < 4 * fs; i++) {
          const y = bank.process(rnd() * 2 - 1);
          if (i > fs) {
            pn += y * y;
            pt += bank.out[0] * bank.out[0];
          }
        }
        expect(Math.abs(10 * Math.log10(pt / pn) - t.gainDb), `${h} harmonics Q ${q} ${JSON.stringify(t)}`).toBeLessThan(tol);
      }
    }
  });

  it('NaN trims count as neutral; NaN input is silence', () => {
    const bank = new ResonatorBank(fs);
    bank.setTrims(0, only(1, Number.NaN, Number.NaN));
    bank.setTrims(1, only(1, 3, 20));
    expect(bank.process(Number.NaN)).toBe(0);
    for (let i = 0; i < 2000; i++) {
      const y = bank.process(Math.sin(i * 0.013));
      expect(bank.out[0]).toBe(y);
      expect(Number.isFinite(bank.out[1])).toBe(true);
    }
  });
});
