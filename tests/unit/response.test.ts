import { describe, expect, it } from 'vitest';
import { Biquad, coeffRampSamples, highpass, onePoleHighpass, onePoleLowpass } from '../../src/dsp/biquad.ts';
import { besselLowpassSections } from '../../src/dsp/bessel-lp.ts';
import { cAbs, cArg, cDelay, cDiv, cMul } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { DcBlocker } from '../../src/dsp/dc-blocker.ts';
import { DelayLine } from '../../src/dsp/fractional-delay.ts';
import { PhaseRotator } from '../../src/dsp/hilbert-pair.ts';
import { trimFactor, trimForTarget } from '../../src/dsp/harmonic-trims.ts';
import { clampParams, computeChannelDelay, dbToLin, delayCapacity, D_MIN_SAMPLES, neutralTrims, withoutTrims } from '../../src/dsp/params.ts';
import type { EngineParams, HarmonicTrim } from '../../src/dsp/params.ts';
import { ResonatorBank } from '../../src/dsp/resonator-bank.ts';
import {
  ConditioningChain,
  channelConditioningResponse,
  channelResponse,
  conditioningResponse,
  groupDelaySamples,
} from '../../src/dsp/response.ts';

type DeepPartial<T> = { [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K] };

function makeParams(over: DeepPartial<EngineParams> & { ch?: [Partial<EngineParams['channels'][0]>, Partial<EngineParams['channels'][1]>] }): EngineParams {
  const base = clampParams(undefined);
  return clampParams({
    ...base,
    ...(over as Partial<EngineParams>),
    filters: { ...base.filters, ...(over.filters ?? {}) },
    tonal: { ...base.tonal, ...(over.tonal ?? {}) },
    channels: [
      { ...base.channels[0], ...(over.ch?.[0] ?? {}) },
      { ...base.channels[1], ...(over.ch?.[1] ?? {}) },
    ],
  });
}

/** The chain built independently from the primitives, as the engine runs it (per sample). */
function primitiveChain(p: EngineParams, fs: number, ch: 0 | 1): (x: number) => number {
  const f = p.filters;
  const dc = new DcBlocker(fs);
  const bq: Biquad[] = [];
  if (f.hpEnabled) bq.push(new Biquad(f.hpOrder === 1 ? onePoleHighpass(fs, f.hpFreq) : highpass(fs, f.hpFreq, Math.SQRT1_2)));
  if (f.barrierEnabled) bq.push(new Biquad(onePoleLowpass(fs, f.barrierFreq)));
  if (f.lpEnabled) for (const c of besselLowpassSections(fs, f.lpFreq, f.lpOrder)) bq.push(new Biquad(c));
  let bank: ResonatorBank | null = null;
  if (p.phaseMode === 'tonal') {
    bank = new ResonatorBank(fs);
    bank.configure(p.tonal.f0, p.tonal.harmonics, p.tonal.q);
    bank.setTrims(ch, p.channels[ch].harmonicTrims, 0);
  }
  const line = new DelayLine(delayCapacity(fs));
  const c = p.channels[ch];
  const d = computeChannelDelay(p, fs, ch).samples;
  let rot: PhaseRotator | null = null;
  if (p.phaseMode === 'hilbert') {
    rot = new PhaseRotator(fs);
    rot.setDeltaDeg(c.phaseDeltaDeg, true);
  }
  const g = (c.enabled ? 1 : 0) * (c.polarityInvert ? -1 : 1) * dbToLin(c.gainDb) * dbToLin(p.masterDb);
  return (x: number) => {
    let y = dc.process(x);
    for (const b of bq) y = b.process(y);
    if (bank) {
      bank.process(y);
      y = bank.out[ch];
    }
    line.write(y);
    let z = line.read(d);
    if (rot) z = rot.process(z);
    return g * z;
  };
}

/** Steady-state response to sin(ωn) measured over the last 0.5 s of `seconds`. */
function measure(step: (x: number) => number, fs: number, f: number, seconds = 2.5): Complex {
  const w = (2 * Math.PI * f) / fs;
  const n = Math.round(seconds * fs);
  const start = n - Math.round(0.5 * fs);
  let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
  for (let i = 0; i < n; i++) {
    const s = Math.sin(w * i);
    const y = step(s);
    if (i >= start) {
      const c = Math.cos(w * i);
      ss += s * s; sc += s * c; cc += c * c; ys += y * s; yc += y * c;
    }
  }
  const det = ss * cc - sc * sc;
  return { re: (ys * cc - yc * sc) / det, im: (yc * ss - ys * sc) / det };
}

function expectClose(measured: Complex, predicted: Complex) {
  const ratio = cDiv(measured, predicted);
  expect(Math.abs(20 * Math.log10(cAbs(ratio)))).toBeLessThan(0.05);
  expect(Math.abs((cArg(ratio) * 180) / Math.PI)).toBeLessThan(0.5);
}

/** Harmonic trims from [gainDb, phaseDeg] pairs (the rest neutral). */
function trimsOf(pairs: [number, number][]): HarmonicTrim[] {
  const t = neutralTrims();
  pairs.forEach(([gainDb, phaseDeg], i) => (t[i] = { gainDb, phaseDeg }));
  return t;
}

const SCENARIOS: { name: string; fs: number; p: EngineParams; freqs: number[] }[] = [
  {
    name: 'delay mode, defaults, Δ = +30°, 3.2 ms base',
    fs: 48000,
    p: makeParams({ ch: [{ delayMs: 3.2, phaseDeltaDeg: 30 }, { delayMs: 0.75, phaseDeltaDeg: -20, gainDb: -6 }] }),
    freqs: [40, 100, 250, 700],
  },
  {
    name: 'delay mode, Δ below the minimum (clamped), 4th-order LP, 1st-order HP, barrier',
    fs: 44100,
    p: makeParams({
      filters: { hpOrder: 1, hpFreq: 35, barrierEnabled: true, barrierFreq: 60, lpOrder: 4, lpFreq: 900 },
      fRef: 100,
      masterDb: -12,
      ch: [{ delayMs: 0, phaseDeltaDeg: -45, polarityInvert: false }, { delayMs: 10, phaseDeltaDeg: 90 }],
    }),
    freqs: [50, 120, 400, 1000],
  },
  {
    name: 'tonal mode with delay wrap and Δ',
    fs: 48000,
    p: makeParams({
      phaseMode: 'tonal',
      tonal: { f0: 123.4, harmonics: 4, q: 30 },
      ch: [{ delayMs: 20, phaseDeltaDeg: 30 }, { delayMs: 47.3, phaseDeltaDeg: -100, gainDb: -3 }],
    }),
    freqs: [123.4, 246.8, 370.2, 493.6, 185.1],
  },
  {
    name: 'tonal mode with harmonic trims (L ≠ R), at and between the harmonics',
    fs: 44100,
    p: makeParams({
      phaseMode: 'tonal',
      tonal: { f0: 98, harmonics: 5, q: 15 },
      ch: [
        { delayMs: 3, phaseDeltaDeg: 40, harmonicTrims: trimsOf([[4, 50], [-6, -120], [0, 0], [2.5, 170], [-20, 10]]) },
        { delayMs: 11, phaseDeltaDeg: -15, gainDb: -6, harmonicTrims: trimsOf([[-3, -60], [6, 90], [-30, 0]]) },
      ],
    }),
    freqs: [98, 196, 294, 392, 490, 147, 60, 700],
  },
  {
    name: 'hilbert mode, Δ = 60°, no filters but the DC blocker',
    fs: 48000,
    p: makeParams({
      phaseMode: 'hilbert',
      filters: { hpEnabled: false, lpEnabled: false },
      masterDb: -10,
      ch: [{ delayMs: 1, phaseDeltaDeg: 60, polarityInvert: false, gainDb: -6 }, { delayMs: 2.5, phaseDeltaDeg: -150 }],
    }),
    freqs: [40, 100, 1000, 5000],
  },
  {
    name: 'gain unlocked +6 dB, master 0 dB',
    fs: 48000,
    p: makeParams({ gainUnlocked: true, masterDb: 0, ch: [{ gainDb: 6, delayMs: 5 }, { enabled: false }] }),
    freqs: [80, 300],
  },
];

describe('channelResponse matches the chain built from the primitives', () => {
  for (const sc of SCENARIOS) {
    it(sc.name, () => {
      for (const ch of [0, 1] as const) {
        for (const f of sc.freqs) {
          const predicted = channelResponse(sc.p, sc.fs, f, ch);
          const measured = measure(primitiveChain(sc.p, sc.fs, ch), sc.fs, f);
          if (!sc.p.channels[ch].enabled) {
            expect(predicted).toEqual({ re: 0, im: 0 });
            expect(cAbs(measured)).toBe(0);
            continue;
          }
          expectClose(measured, predicted);
        }
      }
    });
  }
});

describe('conditioningResponse', () => {
  it('matches ConditioningChain and the primitive conditioning in steady state', () => {
    for (const sc of SCENARIOS) {
      const f = sc.freqs[1];
      const chain = new ConditioningChain(sc.fs, sc.p);
      expectClose(measure((x) => chain.process(x), sc.fs, f), conditioningResponse(sc.p, sc.fs, f));
    }
  });

  it('ConditioningChain is sample-identical to the primitives', () => {
    const p = makeParams({
      phaseMode: 'tonal',
      filters: { barrierEnabled: true, lpOrder: 4 },
      tonal: { f0: 90, harmonics: 3, q: 20 },
      masterDb: 0,
      ch: [{ delayMs: 0, phaseDeltaDeg: 0, polarityInvert: false, gainDb: 0 }, {}],
    });
    const fs = 48000;
    const chain = new ConditioningChain(fs, p);
    // primitive chain with the tap at exactly D_MIN (= 1 sample) and unity gain
    const prim = primitiveChain(p, fs, 0);
    let prevChain = 0;
    for (let n = 0; n < 5000; n++) {
      const x = Math.sin(n * 0.037) + 0.3 * Math.sin(n * 0.11);
      expect(prim(x)).toBe(prevChain);
      prevChain = chain.process(x);
    }
  });

  it('is the identity apart from the DC blocker when all filters are off', () => {
    const p = makeParams({ filters: { hpEnabled: false, lpEnabled: false, barrierEnabled: false } });
    const h = conditioningResponse(p, 48000, 1000);
    // (1 − z⁻¹)/(1 − R z⁻¹) rises to 2/(1 + R) ≈ +0.0017 dB in the audio band
    expect(Math.abs(20 * Math.log10(cAbs(h)))).toBeLessThan(0.002);
    expect(Math.abs(cArg(h))).toBeLessThan(0.01);
  });

  it('reconfiguring enables a section from a clean state and keeps the others', () => {
    const fs = 48000;
    const a = makeParams({ filters: { lpEnabled: false } });
    const b = makeParams({ filters: { lpEnabled: true } });
    const chain = new ConditioningChain(fs, a);
    for (let n = 0; n < 1000; n++) chain.process(Math.sin(n * 0.05));
    chain.configure(b);
    for (let n = 0; n < 10; n++) expect(Number.isFinite(chain.process(Math.sin(n * 0.05)))).toBe(true);
  });
});

describe('ConditioningChain live coefficient changes', () => {
  const fs = 48000;
  const tone = (i: number) => Math.sin((2 * Math.PI * 100 * i) / fs);
  const a = makeParams({ filters: { hpFreq: 40, barrierEnabled: true, barrierFreq: 100, lpOrder: 4, lpFreq: 1000 } });
  const b = makeParams({ filters: { hpFreq: 20, barrierEnabled: true, barrierFreq: 50, lpOrder: 4, lpFreq: 500 } });

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

  it('halving every cutoff at once glides click-free (nothing above 2 kHz) and settles on the new response', () => {
    const chain = new ConditioningChain(fs, a);
    const n0 = fs;
    const y = new Float64Array(2 * fs);
    for (let i = 0; i < y.length; i++) {
      if (i === n0) chain.configure(b);
      y[i] = chain.process(tone(i));
    }
    expect(clickDb(y, n0 - 240, n0 + 4800)).toBeLessThan(-100);
    expectClose(measure((x) => chain.process(x), fs, 100, 1), conditioningResponse(b, fs, 100));
  });

  it('repeating configure() (same filters, other parameters changed) does not disturb a glide', () => {
    const one = new ConditioningChain(fs, a);
    const many = new ConditioningChain(fs, a);
    const again = { ...b, masterDb: -12 };
    for (let i = 0; i < 6000; i++) {
      if (i === 1000) {
        one.configure(b);
        many.configure(b);
      }
      if (i === 1100 || i === 1500 || i === 1960) many.configure(again);
      expect(many.process(tone(i))).toBe(one.process(tone(i)));
    }
  });

  it('COEFF_RAMP_MS after configure() the chain runs exactly the new coefficients', () => {
    const glide = new ConditioningChain(fs, a);
    const n = coeffRampSamples(fs);
    for (let i = 0; i < 3000; i++) glide.process(tone(i));
    glide.configure(b);
    for (let i = 3000; i < 3000 + n; i++) glide.process(tone(i));
    // only the states differ from a new chain now, and that difference decays (HP 20 Hz: τ ≈ 11 ms)
    const fresh = new ConditioningChain(fs, b);
    let worst = 0;
    for (let i = 3000 + n; i < 3000 + n + fs; i++) {
      const y = glide.process(tone(i));
      const z = fresh.process(tone(i));
      if (i > 3000 + n + 0.75 * fs) worst = Math.max(worst, Math.abs(y - z));
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it('reset() completes a glide: the chain is then sample-identical to a new one', () => {
    const chain = new ConditioningChain(fs, a);
    for (let i = 0; i < 2000; i++) chain.process(tone(i));
    chain.configure(b);
    chain.process(0.5);
    chain.reset();
    const fresh = new ConditioningChain(fs, b);
    for (let i = 0; i < 3000; i++) expect(chain.process(tone(i))).toBe(fresh.process(tone(i)));
  });

  it('tonal mode: Q ×3 and f0 +3 % glide click-free and settle on the new response', () => {
    const t1 = makeParams({ phaseMode: 'tonal', tonal: { f0: 100, harmonics: 4, q: 30 } });
    const t2 = makeParams({ phaseMode: 'tonal', tonal: { f0: 103, harmonics: 4, q: 10 } });
    const chain = new ConditioningChain(fs, t1);
    const n0 = 2 * fs;
    const y = new Float64Array(3 * fs);
    for (let i = 0; i < y.length; i++) {
      if (i === n0) chain.configure(t2);
      y[i] = chain.process(tone(i));
    }
    expect(clickDb(y, n0 - 240, n0 + 4800)).toBeLessThan(-100);
    expectClose(measure((x) => chain.process(x), fs, 103, 2), conditioningResponse(t2, fs, 103));
  });
});

describe('channelResponse conventions', () => {
  it('uses computeChannelDelay: phase at fRef is polarity + Δ on top of the conditioning', () => {
    const fs = 48000;
    for (const delta of [0, 30, -30, 90]) {
      const p = makeParams({ ch: [{ delayMs: 0, phaseDeltaDeg: delta }, {}] });
      // isolate the tap: divide out conditioning and the intrinsic D_MIN delay
      const rel = cDiv(cDiv(channelResponse(p, fs, p.fRef, 0), conditioningResponse(p, fs, p.fRef)), cDelay(p.fRef, fs, D_MIN_SAMPLES));
      let phase = (-cArg(rel) * 180) / Math.PI; // lag in degrees
      phase = ((phase % 360) + 360) % 360;
      const expected = delta < 0 ? 180 : 180 + delta; // negative Δ with 0 base delay is clamped
      expect(Math.abs(phase - expected)).toBeLessThan(1e-6);
      expect(computeChannelDelay(p, fs, 0).clamped).toBe(delta < 0);
    }
  });

  it('gains multiply in linear terms; master ≤ -120 dB silences', () => {
    const fs = 48000;
    const p = makeParams({ masterDb: -30, ch: [{ gainDb: -20 }, {}] });
    const rel = cAbs(cDiv(channelResponse(p, fs, 200, 0), conditioningResponse(p, fs, 200)));
    expect(20 * Math.log10(rel)).toBeCloseTo(-50, 9);
    const q = makeParams({ masterDb: -200 });
    expect(cAbs(channelResponse(q, fs, 200, 0))).toBe(0);
  });

  it('clamps parameters like the engine (gain above 0 dB needs the unlock)', () => {
    const fs = 48000;
    const raw = { ...clampParams(undefined), masterDb: 0 };
    raw.channels = [{ ...raw.channels[0], gainDb: 12 }, raw.channels[1]];
    const rel = cAbs(cDiv(channelResponse(raw, fs, 200, 0), conditioningResponse(raw, fs, 200)));
    expect(20 * Math.log10(rel)).toBeCloseTo(0, 9);
  });
});

describe('harmonic trims in the response functions', () => {
  const fs = 48000;
  const trims = trimsOf([[3, 30], [-6, -45], [0, 90], [6, 180]]);
  const p = makeParams({ phaseMode: 'tonal', tonal: { f0: 110, harmonics: 4, q: 25 }, ch: [{ harmonicTrims: trims }, {}] });

  it('conditioningResponse stays neutral; channelConditioningResponse = it × the trim at each harmonic', () => {
    const neutral = makeParams({ phaseMode: 'tonal', tonal: { f0: 110, harmonics: 4, q: 25 } });
    for (const f of [55, 110, 165, 220, 330, 440, 1000]) {
      expect(conditioningResponse(p, fs, f)).toEqual(conditioningResponse(neutral, fs, f));
      expect(channelConditioningResponse(p, fs, f, 1)).toEqual(conditioningResponse(neutral, fs, f));
    }
    for (let h = 1; h <= 4; h++) {
      const want = cMul(conditioningResponse(p, fs, h * 110), trimFactor(trims[h - 1]));
      const got = channelConditioningResponse(p, fs, h * 110, 0);
      expect(cAbs({ re: got.re - want.re, im: got.im - want.im }) / cAbs(want)).toBeLessThan(1e-5);
      // channelResponse: the same factor on top of the delay, polarity and gains
      const q = makeParams({ phaseMode: 'tonal', tonal: { f0: 110, harmonics: 4, q: 25 } });
      const r = cDiv(channelResponse(p, fs, h * 110, 0), channelResponse(q, fs, h * 110, 0));
      const t = trimFactor(trims[h - 1]);
      expect(cAbs({ re: r.re - t.re, im: r.im - t.im })).toBeLessThan(1e-9);
    }
    // outside tonal mode trims do nothing
    const d = makeParams({ ch: [{ harmonicTrims: trims }, {}] });
    expect(channelResponse(d, fs, 220, 0)).toEqual(channelResponse(makeParams({}), fs, 220, 0));
  });

  it('trimForTarget + withoutTrims give the trims that put the channel response on any target at each harmonic', () => {
    const q = makeParams({ phaseMode: 'tonal', tonal: { f0: 90, harmonics: 6, q: 20 }, ch: [{ delayMs: 7, phaseDeltaDeg: 33, harmonicTrims: trims }, {}] });
    const untrimmed = withoutTrims(q);
    const targets = [1, 2, 3, 4, 5, 6].map((h) => {
      const cur = channelResponse(untrimmed, fs, h * 90, 0);
      const g = 0.5 + 0.2 * h; // within −30…+6 dB of the untrimmed response
      const a = 0.4 * h - 1.3;
      return cMul(cur, { re: g * Math.cos(a), im: g * Math.sin(a) });
    });
    const set = clampParams(q);
    for (let h = 1; h <= 6; h++) {
      set.channels[0].harmonicTrims[h - 1] = trimForTarget(targets[h - 1], channelResponse(untrimmed, fs, h * 90, 0));
    }
    for (let h = 1; h <= 6; h++) {
      const got = channelResponse(set, fs, h * 90, 0);
      const want = targets[h - 1];
      expect(cAbs({ re: got.re - want.re, im: got.im - want.im }) / cAbs(want)).toBeLessThan(1e-9);
    }
  });

  it('ConditioningChain.channelOut: exactly the neutral output when untrimmed or not tonal, the trimmed bank otherwise', () => {
    for (const mode of ['delay', 'tonal', 'hilbert'] as const) {
      const q = makeParams({ phaseMode: mode, tonal: { f0: 110, harmonics: 4, q: 25 } });
      const chain = new ConditioningChain(fs, q);
      chain.setTrims(0, trims, 0);
      for (let n = 0; n < 4000; n++) {
        const y = chain.process(Math.sin(n * 0.037) + 0.3 * Math.sin(n * 0.11));
        expect(chain.channelOut[1]).toBe(y);
        if (mode !== 'tonal') expect(chain.channelOut[0]).toBe(y);
      }
    }
    const chain = new ConditioningChain(fs, p);
    chain.setTrims(0, trims, 0);
    for (const f of [110, 220, 165, 700]) {
      const m = measure((x) => (chain.process(x), chain.channelOut[0]), fs, f);
      expectClose(m, channelConditioningResponse(p, fs, f, 0));
    }
  });
});

describe('groupDelaySamples', () => {
  it('measures pure delays exactly', () => {
    const fs = 48000;
    for (const d of [0, 12.37, 1000.5, 25000]) {
      for (const f of [20, 100, 1000, 10000]) {
        expect(groupDelaySamples((x) => cDelay(x, fs, d), f, fs)).toBeCloseTo(d, 6);
      }
    }
  });

  it('channel minus conditioning group delay equals the tap delay in delay mode', () => {
    const fs = 48000;
    const p = makeParams({ ch: [{ delayMs: 7.5, phaseDeltaDeg: 45 }, {}] });
    const d = computeChannelDelay(p, fs, 0).samples;
    for (const f of [60, 200, 600]) {
      const gdCh = groupDelaySamples((x) => channelResponse(p, fs, x, 0), f, fs);
      const gdCond = groupDelaySamples((x) => conditioningResponse(p, fs, x), f, fs);
      expect(gdCh - gdCond).toBeCloseTo(d, 6);
    }
  });

  it('Bessel LP group delay at low frequency matches its time-domain delay', () => {
    const fs = 48000;
    const p = makeParams({ filters: { hpEnabled: false, lpEnabled: true, lpFreq: 500, lpOrder: 2 } });
    const f = 100;
    const gd = groupDelaySamples((x) => conditioningResponse(p, fs, x), f, fs);
    // analog 2nd-order Bessel normalised to -3 dB at fc: τ(0) = 1.36165/(2π·fc) (0.433 ms at 500 Hz),
    // plus the 3 Hz DC blocker (1st-order high-pass): τ = ωc/(ωc² + ω²)
    const bessel = (1.36165412871613 / (2 * Math.PI * 500)) * fs;
    const wc = 2 * Math.PI * 3;
    const w = 2 * Math.PI * f;
    const dcBlock = (wc / (wc * wc + w * w)) * fs;
    const expected = bessel + dcBlock;
    expect(Math.abs(gd - expected) / expected).toBeLessThan(0.01);
  });
});
