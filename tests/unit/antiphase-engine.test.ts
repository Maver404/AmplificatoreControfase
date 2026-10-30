import { afterEach, describe, expect, it, vi } from 'vitest';
import { TAP_BLOCK_FRAMES, TAP_CHANNELS } from '../../src/audio/protocol.ts';
import type { WorkletInMsg, WorkletOutMsg } from '../../src/audio/protocol.ts';
import { AntiPhaseEngine, LIMITER_BACKSTOP_MS } from '../../src/dsp/antiphase-engine.ts';
import type { EngineOptions } from '../../src/dsp/antiphase-engine.ts';
import { Biquad, highpass } from '../../src/dsp/biquad.ts';
import { cAbs, cArg, cDelay, cMul } from '../../src/dsp/complex.ts';
import { DelayLine } from '../../src/dsp/fractional-delay.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { trimForTarget } from '../../src/dsp/harmonic-trims.ts';
import { BIG_JUMP_DB, MASTER_MIN_DB, MAX_RISE_DB_PER_S, clampParams, computeChannelDelay, neutralTrims } from '../../src/dsp/params.ts';
import type { EngineParams, HarmonicTrim, PhaseMode } from '../../src/dsp/params.ts';
import { hilbertResponse } from '../../src/dsp/hilbert-pair.ts';
import { HowlDetector } from '../../src/dsp/howl-detector.ts';
import { channelResponse, conditioningResponse } from '../../src/dsp/response.ts';
import { exponentialSweep, mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;
const ms = (t: number): number => Math.round((t / 1000) * fs);

type Msg = WorkletOutMsg & { at: number };

/** Engine + message log; `at` is the absolute sample index of the block after which a message was drained. */
class Rig {
  readonly e: AntiPhaseEngine;
  readonly msgs: Msg[] = [];
  /** Absolute index of the next sample to process. */
  pos = 0;
  /** When set, tap blocks are copied here and their buffers recycled at once. */
  tapCopies: Float32Array[] | null = null;

  constructor(opts?: EngineOptions) {
    this.e = new AntiPhaseEngine(fs, opts);
  }

  send(m: WorkletInMsg): void {
    this.e.handleMessage(m);
    this.drain();
  }

  params(mod: (p: EngineParams) => void): void {
    const p = this.e.params;
    mod(p);
    this.send({ type: 'params', params: p });
  }

  drain(): void {
    const recycle: Float32Array[] = [];
    this.e.drain((m) => {
      this.msgs.push({ ...m, at: this.pos });
      if (m.type === 'tap' && this.tapCopies !== null) {
        this.tapCopies.push(m.block.slice());
        recycle.push(m.block);
      }
    });
    for (const b of recycle) this.e.handleMessage({ type: 'recycle', buffer: b });
  }

  /** Processes `n` samples in blocks; inputs are read from absolute index `pos` (null = absent). */
  run(
    n: number,
    in0: Float32Array | null,
    in1: Float32Array | null = null,
    block = 128,
  ): { L: Float32Array; R: Float32Array } {
    const L = new Float32Array(n);
    const R = new Float32Array(n);
    for (let off = 0; off < n; off += block) {
      const m = Math.min(block, n - off);
      const a = this.pos;
      this.e.process(in0 ? in0.subarray(a, a + m) : null, in1 ? in1.subarray(a, a + m) : null, L.subarray(off, off + m), R.subarray(off, off + m));
      this.pos += m;
      this.drain();
    }
    return { L, R };
  }

  /**
   * Closed loop: the input of every sample is computed from earlier outputs (`feed` sees the full output
   * history of both channels up to the previous block). Block size must not exceed the loop delay.
   */
  runLoop(
    n: number,
    feed: (i: number, outL: Float32Array, outR: Float32Array) => number,
    outL: Float32Array,
    outR: Float32Array,
    inp: Float32Array,
    block = 128,
  ): void {
    const end = this.pos + n;
    while (this.pos < end) {
      const a = this.pos;
      const m = Math.min(block, end - a);
      for (let i = a; i < a + m; i++) inp[i] = feed(i, outL, outR);
      this.e.process(inp.subarray(a, a + m), null, outL.subarray(a, a + m), outR.subarray(a, a + m));
      this.pos += m;
      this.drain();
    }
  }

  of<T extends Msg['type']>(type: T): Extract<Msg, { type: T }>[] {
    return this.msgs.filter((m): m is Extract<Msg, { type: T }> => m.type === type);
  }

  lastMeters(): Extract<Msg, { type: 'meters' }> {
    const m = this.of('meters');
    expect(m.length).toBeGreaterThan(0);
    return m[m.length - 1];
  }
}

/** Clean identity-like setup: 0 dB everywhere, ceiling −1 dBFS, inverted polarity, no delay. */
const unity = (p: EngineParams): void => {
  p.masterDb = 0;
  p.ceilingDb = -1;
  for (const c of p.channels) {
    c.gainDb = 0;
    c.polarityInvert = true;
    c.delayMs = 0;
    c.phaseDeltaDeg = 0;
  }
};

const noise = (n: number, rms: number, seed: number): Float32Array => {
  const rnd = mulberry32(seed);
  const x = new Float32Array(n);
  const a = rms * Math.sqrt(3);
  for (let i = 0; i < n; i++) x[i] = a * (2 * rnd() - 1);
  return x;
};

const sine = (n: number, f: number, amp: number, phase = 0): Float32Array => {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * f * i) / fs + phase);
  return x;
};

const constant = (n: number, v: number): Float32Array => new Float32Array(n).fill(v);

/** Single-bin DFT of x[from, from + len) at f. */
function dft(x: Float32Array, f: number, from: number, len: number): Complex {
  let re = 0;
  let im = 0;
  for (let i = from; i < from + len; i++) {
    const w = (2 * Math.PI * f * i) / fs;
    re += x[i] * Math.cos(w);
    im -= x[i] * Math.sin(w);
  }
  return { re, im };
}

const ratio = (a: Complex, b: Complex): Complex => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};

const deg = (rad: number): number => (rad * 180) / Math.PI;
const wrapDeg = (d: number): number => ((((d + 180) % 360) + 360) % 360) - 180;
const db = (c: Complex): number => 20 * Math.log10(Math.hypot(c.re, c.im));

/** Generic Lagrange polynomial through the 4 samples around n − d (independent of DelayLine). */
function lagrangeAt(x: Float32Array, n: number, d: number): number {
  const k = Math.floor(d) - 1;
  const t = n - d;
  const pts = [n - k, n - k - 1, n - k - 2, n - k - 3];
  let acc = 0;
  for (let j = 0; j < 4; j++) {
    let w = 1;
    for (let m = 0; m < 4; m++) if (m !== j) w *= (t - pts[m]) / (pts[j] - pts[m]);
    acc += w * (pts[j] >= 0 ? x[pts[j]] : 0);
  }
  return acc;
}

const maxAbsDiff = (y: Float32Array, from: number, to: number): number => {
  let m = 0;
  for (let i = Math.max(1, from); i < to; i++) m = Math.max(m, Math.abs(y[i] - y[i - 1]));
  return m;
};

describe('AntiPhaseEngine: arming gate', () => {
  it('not armed → exact zeros on both outputs whatever the input; meters still report the reference', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params(unity);
    // 2 s: the 300 ms RMS meter has settled to 0.01 dB
    const n = 2 * fs;
    const x = sine(n + 2 * fs, 100, 0.1);
    const big = constant(n, 7);
    const { L, R } = r.run(n, x, big);
    expect(L.every((v) => v === 0)).toBe(true);
    expect(R.every((v) => v === 0)).toBe(true);
    const m = r.lastMeters();
    expect(m.armed).toBe(false);
    expect(m.ref.rmsDb).toBeCloseTo(20 * Math.log10(0.1 / Math.SQRT2), 1);
    expect(m.outL.rmsDb).toBe(-120);
    // disarming while running is immediate too
    r.send({ type: 'arm', armed: true });
    r.run(ms(1600), x);
    r.send({ type: 'arm', armed: false });
    const after = r.run(1000, x);
    expect(after.L.every((v) => v === 0) && after.R.every((v) => v === 0)).toBe(true);
  });
});

describe('AntiPhaseEngine: transfer with conditioning bypassed', () => {
  it('y[n] = −x[n − 1] exactly after the soft-start, both channels', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params(unity);
    r.send({ type: 'arm', armed: true });
    const n = 2 * fs;
    const x = noise(n, 0.1, 1);
    const { L, R } = r.run(n, x);
    let worst = 0;
    for (let i = ms(1500) + 2; i < n; i++) {
      worst = Math.max(worst, Math.abs(L[i] + x[i - 1]), Math.abs(R[i] + x[i - 1]));
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it('5.37 ms of base delay → −x[n − 1 − 257.76] (Lagrange reference)', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.channels[0].delayMs = 5.37;
    });
    r.send({ type: 'arm', armed: true });
    const n = 2 * fs;
    const x = noise(n, 0.1, 2);
    const { L, R } = r.run(n, x);
    const d = 1 + 257.76;
    expect(computeChannelDelay(r.e.params, fs, 0).samples).toBeCloseTo(d, 9);
    let worst = 0;
    let worstR = 0;
    for (let i = ms(1500) + 300; i < n; i++) {
      worst = Math.max(worst, Math.abs(L[i] + lagrangeAt(x, i, d)));
      worstR = Math.max(worstR, Math.abs(R[i] + x[i - 1]));
    }
    expect(worst).toBeLessThan(1e-6);
    expect(worstR).toBeLessThan(1e-9);
  });

  it('polarity off and channel gain: y = +g·x[n − 1]; a disabled channel is silent', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.channels[0].polarityInvert = false;
      p.channels[0].gainDb = -6;
      p.channels[1].enabled = false;
    });
    r.send({ type: 'arm', armed: true });
    const n = 2 * fs;
    const x = noise(n, 0.1, 3);
    const { L, R } = r.run(n, x);
    const g = Math.pow(10, -6 / 20);
    let worst = 0;
    for (let i = ms(1500) + 2; i < n; i++) worst = Math.max(worst, Math.abs(L[i] - g * x[i - 1]));
    expect(worst).toBeLessThan(1e-7);
    expect(R.every((v) => v === 0)).toBe(true);
  });
});

describe('AntiPhaseEngine: block-size invariance', () => {
  for (const mode of ['delay', 'tonal', 'hilbert'] as const) {
    it(`identical outputs for blocks of 1, 64, 128, 256 and 4096 (${mode} mode, through the soft-start)`, () => {
      const n = fs;
      const x = noise(n, 0.05, 4);
      const tone = sine(n, 137, 0.2);
      for (let i = 0; i < n; i++) x[i] += tone[i];
      const err = noise(n, 0.02, 5);
      const runWith = (block: number) => {
        const r = new Rig({ softStartMs: 400 });
        r.params((p) => {
          p.masterDb = -6;
          p.ceilingDb = -6;
          p.phaseMode = mode;
          p.tonal.f0 = 137;
          p.channels[0].delayMs = 3.21;
          p.channels[0].phaseDeltaDeg = 17;
          p.channels[1].gainDb = -3;
          p.channels[1].polarityInvert = false;
          p.channels[1].phaseDeltaDeg = -40;
          p.errorChannel = 1;
        });
        r.send({ type: 'arm', armed: true });
        r.send({ type: 'tap', enabled: true });
        return r.run(n, x, err, block);
      };
      const ref = runWith(128);
      let energy = 0;
      for (let i = 0; i < n; i++) energy += ref.L[i] * ref.L[i];
      expect(energy).toBeGreaterThan(1);
      for (const block of [1, 64, 256, 4096]) {
        const o = runWith(block);
        let worst = 0;
        for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(o.L[i] - ref.L[i]), Math.abs(o.R[i] - ref.R[i]));
        expect(worst).toBeLessThan(1e-12);
      }
    });
  }
});

describe('AntiPhaseEngine: phase convention vs response.ts', () => {
  const f = 100;
  /** Steady-state response of outL relative to the input at f. */
  const measure = (mod: (p: EngineParams) => void): { h: Complex; p: EngineParams } => {
    const r = new Rig();
    r.params(mod);
    r.send({ type: 'arm', armed: true });
    const n = 4 * fs;
    const x = sine(n, f, 0.1, 0.4);
    const { L } = r.run(n, x);
    const from = 3 * fs;
    const h = ratio(dft(L, f, from, fs), dft(x, f, from, fs));
    return { h, p: r.e.params };
  };

  for (const mode of ['delay', 'tonal', 'hilbert'] as PhaseMode[]) {
    it(`${mode} mode matches channelResponse at f_ref/f0 for Δ = 0 and Δ = +30° (lag)`, () => {
      const phases: number[] = [];
      for (const delta of [0, 30]) {
        const { h, p } = measure((q) => {
          q.phaseMode = mode;
          q.fRef = f;
          q.tonal.f0 = f;
          q.channels[0].phaseDeltaDeg = delta;
          q.channels[0].delayMs = 0.73;
        });
        const want = channelResponse(p, fs, f, 0);
        expect(Math.abs(wrapDeg(deg(cArg(h) - cArg(want))))).toBeLessThan(0.5);
        expect(Math.abs(db(h) - db(want))).toBeLessThan(0.05);
        phases.push(deg(cArg(h)));
      }
      // Δ = +30° adds 30° of lag
      const lag = wrapDeg(phases[0] - phases[1]);
      expect(Math.abs(lag - 30)).toBeLessThan(mode === 'hilbert' ? 1.5 : 0.5);
    });
  }

  // The plan's convention: total phase = polarity (180° by default) + Δ, at f_ref, ±0.1° in every mode.
  for (const mode of ['delay', 'tonal', 'hilbert'] as PhaseMode[]) {
    it(`${mode} mode, default params: 180° ± 0.1° at f_ref, and Δ = +30° gives 210° ± 0.1°`, () => {
      // Reference path without the anti-phase: conditioning · base tap delay (Δ = 0) · path A in Hilbert mode.
      const base = (p: EngineParams): Complex => {
        const p0: EngineParams = { ...p, channels: [{ ...p.channels[0], phaseDeltaDeg: 0 }, p.channels[1]] };
        const path = cMul(conditioningResponse(p, fs, f), cDelay(f, fs, computeChannelDelay(p0, fs, 0).samples));
        return mode === 'hilbert' ? cMul(path, hilbertResponse(f, fs).a) : path;
      };
      for (const delta of [0, 30]) {
        const { h, p } = measure((q) => {
          q.phaseMode = mode;
          q.channels[0].phaseDeltaDeg = delta;
        });
        expect(p.fRef).toBe(f);
        expect(p.tonal.f0).toBe(f);
        // Lag of the output behind the reference path, in degrees (positive = later).
        const lag = -deg(cArg(h) - cArg(base(p)));
        expect(Math.abs(wrapDeg(lag - (180 + delta))), `${mode}, Δ = ${delta}°: ${wrapDeg(lag).toFixed(4)}°`).toBeLessThan(0.1);
      }
    });
  }

  it('default params at f_ref give total phase 180° + delay phase (and the conditioning phase)', () => {
    const { h, p } = measure(() => {});
    const d = computeChannelDelay(p, fs, 0).samples;
    const path = cMul(conditioningResponse(p, fs, f), cDelay(f, fs, d));
    expect(Math.abs(wrapDeg(deg(cArg(h) - cArg(path)) - 180))).toBeLessThan(0.5);
    // default gains: −20 dB channel, −30 dB master
    expect(Math.abs(db(h) - (db(path) - 50))).toBeLessThan(0.05);
  });
});

describe('AntiPhaseEngine: safety envelopes', () => {
  /** DC through the bypassed chain shows the envelope directly: out = −0.1·env. */
  const envRig = () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params(unity);
    r.send({ type: 'arm', armed: true });
    const x = constant(30 * fs, 0.1);
    return { r, x, env: (n: number) => Array.from(r.run(n, x).L, (v) => -v / 0.1) };
  };

  it('arm → linear 1.5 s soft-start', () => {
    const { env } = envRig();
    const e = env(ms(1600));
    expect(e[ms(375)]).toBeCloseTo(0.25, 2);
    expect(e[ms(750)]).toBeCloseTo(0.5, 2);
    expect(e[ms(1500) + 1]).toBeCloseTo(1, 6);
  });

  it('panic → |out| < 1e-6 within 10 ms + one block; leaving it needs mute=false → soft-start', () => {
    const { r, env } = envRig();
    env(ms(1600));
    r.send({ type: 'panic' });
    const e = env(ms(10) + 128 + 1000);
    for (let i = ms(10) + 128; i < e.length; i++) expect(Math.abs(e[i]) * 0.1).toBeLessThan(1e-6);
    expect(r.lastMeters().muted).toBe(true);
    // bypass toggles and re-arming do not leave the panic
    r.send({ type: 'bypass', bypass: true });
    r.send({ type: 'bypass', bypass: false });
    r.send({ type: 'arm', armed: true });
    expect(env(fs).every((v) => v === 0)).toBe(true);
    r.send({ type: 'mute', muted: false });
    const s = env(ms(1600));
    expect(s[0]).toBeLessThan(0.001);
    expect(s[ms(750)]).toBeCloseTo(0.5, 2);
    expect(s[ms(1500) + 1]).toBeCloseTo(1, 6);
  });

  it('mute = 15 ms linear ramp down; unmute = soft-start from 0 (≈ 1.5 s linear)', () => {
    const { r, env } = envRig();
    env(ms(1600));
    r.send({ type: 'mute', muted: true });
    const m = env(ms(100));
    expect(m[ms(7.5)]).toBeCloseTo(0.5, 2);
    expect(Math.abs(m[ms(15) + 1])).toBe(0);
    expect(m.slice(ms(15) + 1).every((v) => v === 0)).toBe(true);
    expect(maxAbsDiff(Float32Array.from(m), 0, m.length)).toBeLessThan(1.01 / ms(15));
    r.send({ type: 'mute', muted: false });
    const u = env(ms(1600));
    expect(u[ms(375)]).toBeCloseTo(0.25, 2);
    expect(u[ms(750)]).toBeCloseTo(0.5, 2);
    expect(u[ms(1125)]).toBeCloseTo(0.75, 2);
    expect(u[ms(1500) + 1]).toBeCloseTo(1, 6);
  });

  it('un-muting halfway through the mute ramp fades to 0 at the mute slope, then soft-starts (no click)', () => {
    const { r, env } = envRig();
    env(ms(1600));
    r.send({ type: 'mute', muted: true });
    const a = env(ms(7.5));
    r.send({ type: 'mute', muted: false });
    const b = env(ms(1600));
    expect(Math.abs(b[0] - a[a.length - 1])).toBeLessThan(0.002);
    // the remaining 0.5 fades in 7.5 ms (sample ms(7.5) − 1 is the zero), then rises at 1/72000 per sample
    expect(Math.abs(b[ms(7.5) - 1])).toBe(0);
    expect(maxAbsDiff(Float32Array.from(b), 0, b.length)).toBeLessThan(1.01 / ms(15));
    expect(b[ms(7.5) + ms(750)]).toBeCloseTo(0.5, 2);
    expect(b[ms(7.5) + ms(1500) + 2]).toBeCloseTo(1, 6);
  });

  it('mute and unmute in the same tick still soft-start from 0', () => {
    const { r, env } = envRig();
    env(ms(1600));
    r.send({ type: 'mute', muted: true });
    r.send({ type: 'mute', muted: false });
    const b = env(ms(1600));
    expect(maxAbsDiff(Float32Array.from(b), 0, b.length)).toBeLessThan(1.01 / ms(15));
    expect(Math.abs(Math.min(...b.slice(0, ms(30))))).toBe(0);
    expect(b[ms(15) + ms(750)]).toBeCloseTo(0.5, 2);
  });

  it('bypass ramps 15 ms both ways (no soft-start)', () => {
    const { r, env } = envRig();
    env(ms(1600));
    r.send({ type: 'bypass', bypass: true });
    const d = env(ms(50));
    expect(d[ms(7.5)]).toBeCloseTo(0.5, 2);
    expect(Math.abs(d[ms(15) + 1])).toBe(0);
    expect(r.lastMeters().bypass).toBe(true);
    r.send({ type: 'bypass', bypass: false });
    const u = env(ms(50));
    expect(u[ms(7.5)]).toBeCloseTo(0.5, 2);
    expect(u[ms(15) + 1]).toBeCloseTo(1, 6);
  });

  it('a big gain increase while running rises at ≤ 6 dB/s', () => {
    const { r, env } = envRig();
    r.params((p) => (p.masterDb = -30));
    env(ms(1600));
    r.params((p) => (p.masterDb = 0));
    const e = env(4 * fs);
    // 30 dB gap > 6 dB: linear 6 dB/s from the start
    expect(20 * Math.log10(e[fs])).toBeCloseTo(-24, 1);
    expect(20 * Math.log10(e[2 * fs])).toBeCloseTo(-18, 1);
    expect(20 * Math.log10(e[4 * fs - 1])).toBeCloseTo(-6, 1);
  });
});

describe('AntiPhaseEngine: rises are always heard (soft-start or ≤ 6 dB + 6 dB/s)', () => {
  /** DC through the bypassed chain: out = −0.1·(gain · envelopes) per channel. */
  const dcRig = (mod: (p: EngineParams) => void = () => {}) => {
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      mod(p);
    });
    r.send({ type: 'arm', armed: true });
    const x = constant(40 * fs, 0.1);
    const run = (n: number) => {
      const { L, R } = r.run(n, x);
      return { L: Array.from(L, (v) => -v / 0.1), R: Array.from(R, (v) => -v / 0.1) };
    };
    run(ms(1600));
    return { r, run };
  };
  const toDb = (v: number): number => 20 * Math.log10(Math.abs(v));

  it('enabling a channel soft-starts it from 0, even at a gain raised while it was off', () => {
    const { r, run } = dcRig((p) => {
      p.channels[1].enabled = false;
      p.channels[1].gainDb = -40;
    });
    // the gain ramp climbs to 0 dB unheard (≈ 7 s at 6 dB/s)
    r.params((p) => (p.channels[1].gainDb = 0));
    expect(run(8 * fs).R.every((v) => v === 0)).toBe(true);
    r.params((p) => (p.channels[1].enabled = true));
    const { L, R } = run(ms(1700));
    // 15 ms reconfigure ramp × a 1.5 s soft-start, instead of full level after 15 ms
    expect(R[ms(200)]).toBeCloseTo(ms(200) / ms(1500), 2);
    expect(R[ms(750)]).toBeCloseTo(0.5, 2);
    expect(R[ms(1600)]).toBeCloseTo(1, 6);
    expect(maxAbsDiff(Float32Array.from(R), 0, R.length)).toBeLessThan(1.01 / ms(15));
    // the other channel is not touched
    expect(L.every((v) => Math.abs(v - 1) < 1e-6)).toBe(true);
  });

  it('un-muting while bypassed: the soft-start waits for the bypass to end', () => {
    const { r, run } = dcRig();
    r.send({ type: 'bypass', bypass: true });
    run(ms(50));
    r.send({ type: 'mute', muted: true });
    run(ms(50));
    r.send({ type: 'mute', muted: false });
    expect(run(2 * fs).L.every((v) => v === 0)).toBe(true);
    expect(r.lastMeters().softStart).toBe(0);
    r.send({ type: 'bypass', bypass: false });
    const u = run(ms(1600));
    expect(u.L[ms(50)]).toBeCloseTo(ms(50) / ms(1500), 3);
    expect(u.L[ms(750)]).toBeCloseTo(0.5, 2);
    expect(u.L[ms(1600) - 1]).toBeCloseTo(1, 6);
  });

  it('a gain raised while bypassed, or during a measurement, rises at 6 dB/s once the anti-noise is heard again', () => {
    const { r, run } = dcRig((p) => {
      for (const c of p.channels) c.gainDb = -40;
    });
    r.send({ type: 'bypass', bypass: true });
    run(ms(50));
    r.params((p) => (p.channels[0].gainDb = 0));
    run(8 * fs);
    r.send({ type: 'bypass', bypass: false });
    let y = run(fs).L;
    // last heard −40 dB: one second later −34 dB, not 0 dB 15 ms after the bypass
    expect(toDb(y[ms(20)])).toBeLessThan(-39.8);
    expect(toDb(y[fs - 1])).toBeCloseTo(-34, 1);
    // the same behind the measurement gate (here: the gain of R during a 2 s stimulus on L)
    r.send({ type: 'measure', id: 1, stimulus: new Float32Array(2 * fs), channel: 0, postRollFrames: 0 });
    run(ms(20));
    r.params((p) => (p.channels[1].gainDb = 0));
    run(2 * fs);
    expect(r.of('measureResult')[0].aborted).toBe(false);
    y = run(fs).R;
    expect(toDb(y[ms(40)])).toBeLessThan(-39.6);
    expect(toDb(y[fs - 1])).toBeLessThan(-33.9);
  });

  it('master and channel gain raised together share one budget: ≤ 6 dB + 6 dB/s in all', () => {
    const { r, run } = dcRig((p) => {
      p.masterDb = -30;
      for (const c of p.channels) c.gainDb = -40;
    });
    r.params((p) => {
      p.masterDb = 0;
      for (const c of p.channels) c.gainDb = -10;
    });
    const L = run(12 * fs).L.map(toDb);
    // −70 → −10 dB: 6 dB/s (12 dB/s when each ramp had its own budget)
    expect(L[fs]).toBeCloseTo(-64, 1);
    expect(L[5 * fs]).toBeCloseTo(-40, 1);
    expect(L[12 * fs - 1]).toBeCloseTo(-10, 3);
    let worst = 0;
    for (let i = fs; i < L.length; i += 48) worst = Math.max(worst, L[i] - L[i - fs]);
    expect(worst).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S + 0.01);
    // un-muting a master at −∞ still skips the inaudible part (master below −60 dB) at once
    r.params((p) => (p.masterDb = MASTER_MIN_DB));
    expect(Math.abs(run(fs).L[fs - 1])).toBe(0);
    r.params((p) => (p.masterDb = 0));
    const u = run(fs).L.map(toDb);
    expect(u[ms(50)]).toBeGreaterThan(-10 - 60 - 0.01);
    expect(u[fs - 1]).toBeCloseTo(-64, 0);
  });

  describe('a broader chain soft-starts both channels, a narrower one does not', () => {
    const f = 100;
    const amp = 0.5;
    /** Smallest per-period peak of L between 30 and 130 ms after the change (after the reconfigure dip). */
    const levelAfter = (mod: (p: EngineParams) => void, start: (p: EngineParams) => void = () => {}) => {
      const r = new Rig();
      r.params((p) => {
        unity(p);
        p.tonal.f0 = f;
        start(p);
      });
      r.send({ type: 'arm', armed: true });
      const x = sine(10 * fs, f, amp);
      r.run(3 * fs, x, x);
      r.params(mod);
      const y = r.run(ms(130), x, x).L;
      let pk = 0;
      for (let i = ms(120); i < ms(130); i++) pk = Math.max(pk, Math.abs(y[i]));
      const z = r.run(2 * fs, x, x).L;
      let full = 0;
      for (let i = z.length - 480; i < z.length; i++) full = Math.max(full, Math.abs(z[i]));
      return pk / full;
    };
    for (const [name, mod, start] of [
      ['LP off', (p: EngineParams) => (p.filters.lpEnabled = false)],
      ['HP off', (p: EngineParams) => (p.filters.hpEnabled = false)],
      ['barrier off', (p: EngineParams) => (p.filters.barrierEnabled = false), (p: EngineParams) => (p.filters.barrierEnabled = true)],
      ['tonal → delay mode', (p: EngineParams) => (p.phaseMode = 'delay'), (p: EngineParams) => (p.phaseMode = 'tonal')],
      ['reference input 1 → 2', (p: EngineParams) => (p.refChannel = 1)],
    ] as const) {
      it(`${name}: soft-start`, () => expect(levelAfter(mod, start)).toBeLessThan(0.1));
    }
    for (const [name, mod, start] of [
      ['LP 4th order on', (p: EngineParams) => {
        p.filters.lpEnabled = true;
        p.filters.lpOrder = 4;
      }, (p: EngineParams) => (p.filters.lpEnabled = false)],
      ['barrier on', (p: EngineParams) => (p.filters.barrierEnabled = true)],
      ['HP order 1 → 2', (p: EngineParams) => (p.filters.hpOrder = 2), (p: EngineParams) => (p.filters.hpOrder = 1)],
      ['polarity flip', (p: EngineParams) => (p.channels[0].polarityInvert = false)],
    ] as const) {
      it(`${name}: back at full level after the 15 ms reconfigure ramp`, () => expect(levelAfter(mod, start)).toBeGreaterThan(0.95));
    }
  });
});

describe('AntiPhaseEngine: structural changes without clicks', () => {
  const f = 100;
  const amp = 0.5;
  const setup = () => {
    const r = new Rig();
    r.params((p) => {
      unity(p);
      p.fRef = f;
      p.tonal.f0 = f;
      p.channels[0].delayMs = 2;
    });
    r.send({ type: 'arm', armed: true });
    const x = sine(20 * fs, f, amp);
    r.run(2 * fs, x);
    const steady = r.run(fs / 2, x).L;
    return { r, x, limit: maxAbsDiff(steady, 0, steady.length), steady };
  };

  it('polarity flip: bounded |Δy|, then the output is inverted', () => {
    const { r, x, limit, steady } = setup();
    r.params((p) => (p.channels[0].polarityInvert = false));
    const y = r.run(fs, x).L;
    expect(maxAbsDiff(y, 0, y.length)).toBeLessThan(1.25 * limit);
    // fs/2 and fs are whole periods of 100 Hz: compare with the steady state one second earlier
    let worst = 0;
    for (let i = fs / 2; i < fs; i++) worst = Math.max(worst, Math.abs(y[i] + steady[i - fs / 2]));
    expect(worst).toBeLessThan(1e-4);
  });

  for (const mode of ['hilbert', 'tonal'] as const) {
    it(`delay → ${mode} mode switch: bounded |Δy|, settles to the new mode`, () => {
      const { r, x, limit } = setup();
      r.params((p) => (p.phaseMode = mode));
      const a = r.pos;
      const y = r.run(2 * fs, x).L;
      expect(maxAbsDiff(y, 0, y.length)).toBeLessThan(1.25 * limit);
      const h = ratio(dft(y, f, fs, fs), dft(x.subarray(a), f, fs, fs));
      const want = channelResponse(r.e.params, fs, f, 0);
      expect(Math.abs(wrapDeg(deg(cArg(h) - cArg(want))))).toBeLessThan(0.5);
    });
  }

  it('filter structure change (LP off, 4th order back on) is click-free', () => {
    const { r, x, limit } = setup();
    r.params((p) => (p.filters.lpEnabled = false));
    let y = r.run(fs, x).L;
    expect(maxAbsDiff(y, 0, y.length)).toBeLessThan(1.25 * limit);
    r.params((p) => {
      p.filters.lpEnabled = true;
      p.filters.lpOrder = 4;
    });
    y = r.run(fs, x).L;
    expect(maxAbsDiff(y, 0, y.length)).toBeLessThan(1.25 * limit);
  });

  it('small cutoff and Δ changes are applied live, without a gap', () => {
    const { r, x, limit } = setup();
    r.params((p) => {
      p.filters.lpFreq = 505;
      p.filters.hpFreq = 20.3;
      p.channels[0].phaseDeltaDeg = 20;
    });
    const y = r.run(fs / 2, x).L;
    expect(maxAbsDiff(y, 0, y.length)).toBeLessThan(1.05 * limit);
    let minPeak = Infinity;
    for (let p0 = 0; p0 + 480 <= y.length; p0 += 480) {
      let pk = 0;
      for (let i = p0; i < p0 + 480; i++) pk = Math.max(pk, Math.abs(y[i]));
      minPeak = Math.min(minPeak, pk);
    }
    expect(minPeak).toBeGreaterThan(0.45);
  });

  for (const tonal of [false, true]) {
    it(`cutoffs ×2 (${tonal ? 'tonal Q ÷3 and f0 +3 %' : 'LP 500 → 1000 Hz, HP 20 → 40 Hz'}) glide live: no gap, no click, settles`, () => {
      const { r, x } = setup();
      if (tonal) {
        r.params((p) => (p.phaseMode = 'tonal'));
        r.run(2 * fs, x);
      }
      const before = r.run(fs / 2, x).L;
      const limit = maxAbsDiff(before, 0, before.length);
      r.params((p) => {
        if (tonal) {
          p.tonal.q = 10;
          p.tonal.f0 = 103;
        } else {
          p.filters.lpFreq = 1000;
          p.filters.hpFreq = 40;
        }
      });
      const a = r.pos;
      const y = r.run(2 * fs, x).L;
      const head = y.subarray(0, fs / 2);
      expect(maxAbsDiff(head, 0, head.length)).toBeLessThan(1.05 * limit);
      // the reconfigure envelope would dip to 0 for 30 ms: every 10 ms period keeps its level
      let minPeak = Infinity;
      for (let p0 = 0; p0 + 480 <= head.length; p0 += 480) {
        let pk = 0;
        for (let i = p0; i < p0 + 480; i++) pk = Math.max(pk, Math.abs(head[i]));
        minPeak = Math.min(minPeak, pk);
      }
      expect(minPeak).toBeGreaterThan(0.45);
      const h = ratio(dft(y, f, fs, fs), dft(x.subarray(a), f, fs, fs));
      const want = channelResponse(r.e.params, fs, f, 0);
      expect(Math.abs(wrapDeg(deg(cArg(h) - cArg(want))))).toBeLessThan(0.5);
      expect(Math.abs(db(h) - db(want))).toBeLessThan(0.05);
    });
  }

  for (const [name, mod] of [
    ['Bessel cutoff 500 → 1500 Hz', (p: EngineParams) => (p.filters.lpFreq = 1500)],
    ['tonal Q 30 → 6', (p: EngineParams) => (p.tonal.q = 6)],
    ['tonal harmonics 4 → 2', (p: EngineParams) => (p.tonal.harmonics = 2)],
  ] as const) {
    it(`a big coefficient jump (${name}) goes through the reconfigure ramp: no click, then takes effect`, () => {
      const { r, x } = setup();
      if (name.startsWith('tonal')) {
        r.params((p) => (p.phaseMode = 'tonal'));
        r.run(2 * fs, x);
      }
      const before = r.run(fs / 2, x).L;
      const limit = maxAbsDiff(before, 0, before.length);
      r.params(mod);
      const a = r.pos;
      const y = r.run(2 * fs, x).L;
      expect(maxAbsDiff(y, 0, y.length)).toBeLessThan(1.25 * limit);
      const h = ratio(dft(y, f, fs, fs), dft(x.subarray(a), f, fs, fs));
      const want = channelResponse(r.e.params, fs, f, 0);
      expect(Math.abs(wrapDeg(deg(cArg(h) - cArg(want))))).toBeLessThan(0.5);
      expect(Math.abs(db(h) - db(want))).toBeLessThan(0.05);
    });
  }
});

describe('AntiPhaseEngine: the delay line across discontinuities', () => {
  /** 100 Hz, amplitude 0.1, 200 ms of delay; barrier LP switched on at 0.8 s (chain swap, both channels hold). */
  function swapScenario(after: (r: Rig) => void): { step: number; natural: number } {
    const r = new Rig({ softStartMs: 10 });
    r.params((p) => {
      unity(p);
      for (const c of p.channels) c.delayMs = 200;
    });
    const x = sine(3 * fs, 100, 0.1);
    r.send({ type: 'arm', armed: true });
    r.run(ms(800), x);
    r.params((p) => (p.filters.barrierEnabled = true));
    r.run(ms(60), x);
    after(r);
    const { L } = r.run(fs, x);
    return { step: maxAbsDiff(L, 1, L.length), natural: (0.1 * 2 * Math.PI * 100) / fs };
  }

  it('control: a chain swap holds each channel until its tap has passed the swap point (no click)', () => {
    const { step, natural } = swapScenario(() => {});
    expect(step).toBeLessThan(2 * natural);
  });

  it('a polarity flip during that hold does not cut it short', () => {
    // formerly: 0.037 step 200 ms after the swap (28× the natural slope)
    const { step, natural } = swapScenario((r) => r.params((p) => (p.channels[0].polarityInvert = false)));
    expect(step).toBeLessThan(2 * natural);
  });

  it('a delay increase that would read from before the swap waits until the line is clean', () => {
    const { step, natural } = swapScenario((r) => r.params((p) => (p.channels[0].delayMs = 250)));
    expect(step).toBeLessThan(2 * natural);
  });

  it('after a measurement, a longer delay never replays what was recorded meanwhile', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.channels[0].delayMs = 10;
      p.channels[1].enabled = false;
    });
    const x = sine(6 * fs, 100, 0.01);
    r.send({ type: 'arm', armed: true });
    r.run(ms(1600), x);
    // the reference picks up the stimulus while it plays (0.3 s from the end of the 15 ms gate)
    const t0 = r.pos + ms(15);
    for (let i = t0; i < t0 + ms(300); i++) x[i] += 0.5 * Math.sin((2 * Math.PI * 1000 * i) / fs);
    r.send({ type: 'measure', id: 1, stimulus: new Float32Array(ms(300)), channel: 0, postRollFrames: 0 });
    r.run(ms(500), x);
    expect(r.of('measureResult')[0].aborted).toBe(false);
    // 400 ms of delay reaches back into the measurement: structural path, held until clean
    r.params((p) => (p.channels[0].delayMs = 400));
    const { L } = r.run(2 * fs, x);
    let peak = 0;
    for (const v of L) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeLessThan(0.0102);
    // and it does come back
    let tail = 0;
    for (let i = L.length - 480; i < L.length; i++) tail = Math.max(tail, Math.abs(L[i]));
    expect(tail).toBeGreaterThan(0.0099);
  });

  it('tonal mode above 1 kHz: a Δ change across the D mod T wrap is crossfaded, not slewed the long way round', () => {
    const f0 = 1500; // period 32 samples < 1 ms
    const r = new Rig({ softStartMs: 10, testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.phaseMode = 'tonal';
      p.tonal.f0 = f0;
      for (const c of p.channels) c.phaseDeltaDeg = 10;
    });
    const x = sine(2 * fs, f0, 0.1);
    r.send({ type: 'arm', armed: true });
    r.run(ms(500), x);
    const d0 = computeChannelDelay(r.e.params, fs, 0).samples;
    r.params((p) => {
      for (const c of p.channels) c.phaseDeltaDeg = -10;
    });
    // tap delay 1.89 → 32.11 samples: equivalent to 20° less lag
    expect(computeChannelDelay(r.e.params, fs, 0).samples - d0).toBeGreaterThan(30);
    const a = r.pos;
    const { L } = r.run(ms(600), x);
    const T = fs / f0;
    const phase = (k: number): number => cArg(ratio(dft(L, f0, k, T), dft(x.subarray(a), f0, k, T)));
    const final = deg(phase(L.length - 32 * T));
    expect(Math.abs(wrapDeg(deg(phase(0)) - (final - 20)))).toBeLessThan(1);
    let worst = 0;
    for (let k = 0; k + T <= L.length; k += T) {
      const ph = deg(phase(k));
      // between the old value (20° more lag) and the new one: formerly it swept ~340° (through in-phase)
      worst = Math.max(worst, Math.min(Math.abs(wrapDeg(ph - final)), Math.abs(wrapDeg(ph - final + 20))));
    }
    expect(worst).toBeLessThan(11);
  });
});

describe('AntiPhaseEngine: measurement', () => {
  const L = 300;
  const k = 0.5;
  const stimulus = exponentialSweep({ fs, f1: 50, f2: 2000, durationS: 0.5, fadeMs: 20, amplitude: 0.3 });
  const post = 1000;

  const loopRig = (mod: (p: EngineParams) => void = () => {}) => {
    const r = new Rig();
    r.params((p) => {
      unity(p);
      p.masterDb = -6;
      mod(p);
    });
    const n = 10 * fs;
    const outL = new Float32Array(n);
    const outR = new Float32Array(n);
    const inp = new Float32Array(n);
    const feed = (i: number, oL: Float32Array) => (i >= L ? k * oL[i - L] : 0);
    const step = (samples: number) => r.runLoop(samples, feed, outL, outR, inp);
    return { r, outL, outR, step };
  };

  it('records ref = stimulus delayed by the loop × 0.5 × masterLin, aligned to stimulus sample 0; anti-noise muted', () => {
    const { r, outL, outR, step } = loopRig();
    r.send({ type: 'arm', armed: true });
    r.send({ type: 'mute', muted: true }); // the stimulus ignores mute and soft-start
    step(ms(100));
    const at = r.pos;
    const stim = Float32Array.from(stimulus);
    r.send({ type: 'measure', id: 7, stimulus: stim, channel: 0, postRollFrames: post });
    step(2 * fs);
    const res = r.of('measureResult');
    expect(res.length).toBe(1);
    const m = res[0];
    expect(m.aborted).toBe(false);
    expect(m.id).toBe(7);
    expect(m.err).toBeNull();
    expect(m.ref.length).toBe(stimulus.length + post);
    const masterLin = Math.pow(10, -6 / 20);
    expect(m.masterLin).toBeCloseTo(masterLin, 12);
    let worst = 0;
    for (let i = 0; i < m.ref.length; i++) {
      const want = i >= L && i - L < stimulus.length ? k * masterLin * stimulus[i - L] : 0;
      worst = Math.max(worst, Math.abs(m.ref[i] - want));
    }
    expect(worst).toBeLessThan(1e-6);
    // the stimulus starts after the 15 ms gate, on L only, and nothing else is played meanwhile
    const t0 = at + ms(15);
    let worstOut = 0;
    for (let i = at; i < t0 + stimulus.length + post; i++) {
      const want = i >= t0 && i - t0 < stimulus.length ? masterLin * stimulus[i - t0] : 0;
      worstOut = Math.max(worstOut, Math.abs(outL[i] - want), Math.abs(outR[i]));
    }
    expect(worstOut).toBeLessThan(1e-7);
    const prog = r.of('measureProgress');
    expect(prog.length).toBeGreaterThanOrEqual(4);
    for (let i = 1; i < prog.length; i++) expect(prog[i].fraction).toBeGreaterThan(prog[i - 1].fraction);
    expect(prog.every((p) => p.id === 7 && p.fraction > 0 && p.fraction < 1)).toBe(true);
    const meters = r.of('meters').filter((mm) => mm.at > at && mm.at < t0 + stimulus.length);
    expect(meters.length).toBeGreaterThan(0);
    expect(meters.every((mm) => mm.measuring)).toBe(true);
  });

  it('the stimulus goes through the limiter: never above the ceiling', () => {
    const { r, outL, step } = loopRig((p) => {
      p.masterDb = 0;
      p.ceilingDb = -12;
    });
    r.send({ type: 'arm', armed: true });
    step(ms(20));
    const loud = exponentialSweep({ fs, f1: 50, f2: 2000, durationS: 0.3, fadeMs: 10, amplitude: 0.9 });
    r.send({ type: 'measure', id: 1, stimulus: loud, channel: 0, postRollFrames: 500 });
    step(fs);
    const ceil = Math.pow(10, -12 / 20);
    let peak = 0;
    for (let i = 0; i < r.pos; i++) peak = Math.max(peak, Math.abs(outL[i]));
    expect(peak).toBeLessThanOrEqual(ceil + 1e-9);
    expect(peak).toBeGreaterThan(0.99 * ceil);
    const m = r.of('measureResult')[0];
    expect(m.aborted).toBe(false);
    let refPeak = 0;
    for (const v of m.ref) refPeak = Math.max(refPeak, Math.abs(v));
    expect(refPeak).toBeLessThanOrEqual(k * ceil + 1e-7);
  });

  it('stimulus on channel 1 plays on R; the anti-noise resumes after the measurement', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params(unity);
    r.send({ type: 'arm', armed: true });
    const x = noise(4 * fs, 0.05, 9);
    r.run(ms(1600), x);
    const stim = constant(1000, 0.25);
    r.send({ type: 'measure', id: 2, stimulus: stim, channel: 1, postRollFrames: 0 });
    const a = r.pos;
    const { L, R } = r.run(fs, x);
    const t0 = ms(15);
    for (let i = t0; i < t0 + 1000; i++) {
      expect(Math.abs(L[i])).toBe(0);
      expect(R[i]).toBeCloseTo(0.25, 6);
    }
    // gate held while the delay line still holds the measured period, then a 15 ms ramp
    const back = t0 + 1000 + 4 + ms(15) + 2;
    for (let i = back; i < back + 1000; i++) expect(Math.abs(L[i] + x[a + i - 1])).toBeLessThan(1e-9);
  });

  it('a master changed just before the measurement: the whole stimulus plays at the reported masterLin', () => {
    const r = new Rig({ softStartMs: 10 });
    r.params((p) => {
      unity(p);
      p.masterDb = -30;
    });
    r.send({ type: 'arm', armed: true });
    const silence = new Float32Array(5 * fs);
    r.run(ms(500), silence);
    // e.g. the wizard's «riportalo a 0 dB», then the measurement at once (the master would take ~4.5 s)
    r.params((p) => (p.masterDb = 0));
    const stim = sine(ms(3000), 1000, 0.01);
    r.send({ type: 'measure', id: 1, stimulus: stim, channel: 0, postRollFrames: ms(100) });
    const { L } = r.run(ms(3500), silence);
    const res = r.of('measureResult')[0];
    expect(res.aborted).toBe(false);
    expect(res.masterLin).toBe(1);
    const t0 = ms(15);
    let worst = 0;
    for (let i = 0; i < stim.length; i++) worst = Math.max(worst, Math.abs(L[t0 + i] - stim[i]));
    expect(worst).toBeLessThan(1e-9);
  });

  it('changing the master while the stimulus plays aborts the measurement (one masterLin per result)', () => {
    const r = new Rig();
    r.params((p) => {
      unity(p);
      p.masterDb = -6;
    });
    r.send({ type: 'arm', armed: true });
    const silence = new Float32Array(2 * fs);
    r.run(ms(20), silence);
    r.send({ type: 'measure', id: 2, stimulus: constant(fs, 0.1), channel: 0, postRollFrames: 0 });
    r.run(ms(100), silence);
    // other parameters (and the same master) do not disturb it
    r.params((p) => (p.channels[1].gainDb = -12));
    r.run(ms(100), silence);
    expect(r.of('measureResult').length).toBe(0);
    r.params((p) => (p.masterDb = -12));
    const res = r.of('measureResult');
    expect(res.length).toBe(1);
    expect(res[0].aborted).toBe(true);
    const at = r.pos;
    const { L } = r.run(ms(100), silence);
    for (let i = ms(10) + 1; i < L.length; i++) expect(Math.abs(L[i])).toBeLessThan(1e-6);
    void at;
  });

  it('gain reduction left by the anti-noise does not attenuate the start of the stimulus', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.ceilingDb = -12;
    });
    r.send({ type: 'arm', armed: true });
    // a 100 ms burst of anti-noise drives the limiter ~12 dB into gain reduction (no howl yet)
    const x = constant(3 * fs, 0);
    x.fill(1, ms(1600), ms(1700));
    r.run(ms(1700), x);
    expect(r.lastMeters().limiterGrDb[0]).toBeGreaterThan(10);
    expect(r.of('howl').length).toBe(0);
    r.send({ type: 'measure', id: 3, stimulus: constant(1000, 0.1), channel: 0, postRollFrames: 0 });
    const { L } = r.run(ms(100), x);
    // formerly ≈ −9 dB at the first sample, recovering over the 100 ms release
    for (let i = ms(15); i < ms(15) + 1000; i++) expect(L[i]).toBeCloseTo(0.1, 6);
  });

  it('measure while not armed → aborted at once', () => {
    const r = new Rig();
    r.send({ type: 'measure', id: 3, stimulus: Float32Array.from(stimulus), channel: 0, postRollFrames: 10 });
    const res = r.of('measureResult');
    expect(res.length).toBe(1);
    expect(res[0].aborted).toBe(true);
    expect(res[0].id).toBe(3);
  });

  it('abortMeasure stops at once: aborted result, the stimulus fades within 10 ms', () => {
    const { r, outL, step } = loopRig();
    r.send({ type: 'arm', armed: true });
    r.send({ type: 'mute', muted: true });
    step(ms(20));
    r.send({ type: 'measure', id: 4, stimulus: Float32Array.from(stimulus), channel: 0, postRollFrames: post });
    step(ms(200));
    expect(Math.max(...Array.from(outL.subarray(r.pos - 500, r.pos), Math.abs))).toBeGreaterThan(0.01);
    r.send({ type: 'abortMeasure' });
    const res = r.of('measureResult');
    expect(res.length).toBe(1);
    expect(res[0].aborted).toBe(true);
    expect(res[0].id).toBe(4);
    const at = r.pos;
    step(ms(100));
    for (let i = at + ms(10) + 1; i < r.pos; i++) expect(Math.abs(outL[i])).toBeLessThan(1e-6);
    // a second measurement works after the abort
    r.send({ type: 'measure', id: 5, stimulus: Float32Array.from(stimulus), channel: 0, postRollFrames: 10 });
    step(fs);
    const r2 = r.of('measureResult');
    expect(r2.length).toBe(2);
    expect(r2[1].aborted).toBe(false);
  });

  it('panic aborts a measurement and silences the stimulus', () => {
    const { r, outL, step } = loopRig();
    r.send({ type: 'arm', armed: true });
    r.send({ type: 'measure', id: 6, stimulus: Float32Array.from(stimulus), channel: 0, postRollFrames: post });
    step(ms(200));
    r.send({ type: 'panic' });
    const at = r.pos;
    step(ms(100));
    expect(r.of('measureResult')[0].aborted).toBe(true);
    for (let i = at + ms(10) + 128; i < r.pos; i++) expect(Math.abs(outL[i])).toBeLessThan(1e-6);
  });

  it('records the error channel when present', () => {
    const r = new Rig();
    r.params((p) => (p.errorChannel = 1));
    r.send({ type: 'arm', armed: true });
    const x0 = noise(fs, 0.01, 10);
    const x1 = noise(fs, 0.02, 11);
    r.run(256, x0, x1);
    const start = r.pos;
    r.send({ type: 'measure', id: 8, stimulus: new Float32Array(2000), channel: 0, postRollFrames: 0 });
    r.run(ms(100), x0, x1);
    const m = r.of('measureResult')[0];
    expect(m.err).not.toBeNull();
    const t0 = start + ms(15);
    for (let i = 0; i < 2000; i++) {
      expect(m.ref[i]).toBe(x0[t0 + i]);
      expect(m.err![i]).toBe(x1[t0 + i]);
    }
  });
});

describe('AntiPhaseEngine: howl protection', () => {
  const ceilingDb = -12;
  const ceil = Math.pow(10, ceilingDb / 20);

  /** Speaker → reference loop closed at `onset` with gain `fb` and delay `delay` samples (default chain). */
  function howlRun(delay: number, fb: number, seconds: number) {
    const r = new Rig();
    r.params((p) => {
      unity(p);
      p.ceilingDb = ceilingDb;
      p.channels[1].enabled = false;
    });
    r.send({ type: 'arm', armed: true });
    const n = Math.round(seconds * fs);
    const onset = 2 * fs;
    const amb = noise(n, 0.01, 12);
    const outL = new Float32Array(n);
    const outR = new Float32Array(n);
    const inp = new Float32Array(n);
    r.runLoop(n, (i, oL) => amb[i] + (i >= onset ? fb * oL[i - delay] : 0), outL, outR, inp);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(outL[i]), Math.abs(outR[i]));
    return { r, onset, peak, outL };
  }

  it('loop gain ≈ 1.5 → autoMute + howl within 700 ms of the onset, never above the ceiling', () => {
    // conditioning ≈ 0 dB at the low comb teeth, unity gains: loop gain = fb
    const { r, onset, peak, outL } = howlRun(300, 1.5, 3.5);
    const howl = r.of('howl');
    const auto = r.of('autoMute');
    expect(howl.length).toBe(1);
    expect(auto.length).toBe(1);
    expect(howl[0].at - onset).toBeLessThan(ms(700));
    expect(auto[0].at).toBe(howl[0].at);
    expect(auto[0].reason).toMatch(/Innesco/);
    expect(peak).toBeLessThanOrEqual(ceil + 1e-9);
    // muted with the panic ramp: silent 10 ms + one block after the event
    for (let i = howl[0].at + ms(10) + 128; i < outL.length; i++) expect(Math.abs(outL[i])).toBe(0);
    expect(r.lastMeters().muted).toBe(true);
  });

  it('no trigger on the soft-start and a stable loop (loop gain 0.5)', () => {
    const { r } = howlRun(300, 0.5, 4);
    expect(r.of('howl').length).toBe(0);
    expect(r.of('autoMute').length).toBe(0);
  });

  it('a dense-comb howl (50 ms loop, no narrow peak) is muted within 1 s, never above the ceiling', () => {
    const { r, onset, peak } = howlRun(2400, 1.5, 4);
    const howl = r.of('howl');
    expect(howl.length).toBe(1);
    expect(howl[0].at - onset).toBeLessThan(ms(1000));
    expect(r.of('autoMute')[0].reason).toMatch(/^Innesco rilevato/);
    expect(peak).toBeLessThanOrEqual(ceil + 1e-9);
  });

  /** Unity chain without conditioning, ceiling −12 dBFS, right channel off. */
  function rawRig(): Rig {
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.ceilingDb = ceilingDb;
      p.channels[1].enabled = false;
    });
    r.send({ type: 'arm', armed: true });
    return r;
  }

  it('loud Gaussian noise 6 dB below the ceiling (the limiter always busy) is not taken for a howl', () => {
    // The former backstop (GR > 0.5 dB for 2 s) muted this after 2 s.
    const r = rawRig();
    const n = 10 * fs;
    const rnd = mulberry32(21);
    const rms = ceil * Math.pow(10, -6 / 20);
    const x = Float32Array.from({ length: n }, () => rms * Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()));
    const { L } = r.run(n, x);
    expect(r.of('howl').length).toBe(0);
    expect(r.of('autoMute').length).toBe(0);
    const gr = r.of('meters').map((m) => m.limiterGrDb[0]);
    expect(gr.reduce((a, b) => a + b, 0) / gr.length).toBeGreaterThan(1.5);
    expect(Math.min(...gr.slice(gr.length >> 1))).toBeGreaterThan(0);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(L[i]));
    expect(peak).toBeLessThanOrEqual(ceil + 1e-9);
  });

  it('backstop: GR held above LIMITER_BACKSTOP_DB for LIMITER_BACKSTOP_MS mutes whatever the content', () => {
    // Sparse random clicks 20 dB over the ceiling: aperiodic, white, stationary, low RMS, but the limiter
    // never recovers (~10 dB of GR on average). No detector rule fires; the backstop does.
    const r = rawRig();
    const n = 5 * fs;
    const rnd = mulberry32(22);
    const x = new Float32Array(n);
    const start = 2 * fs;
    for (let i = start; i < n; i += 1 + Math.floor(rnd() * 2 * ms(30))) x[i] = (rnd() < 0.5 ? -10 : 10) * ceil;
    const { L } = r.run(n, x);
    const howl = r.of('howl');
    expect(howl.length).toBe(1);
    expect(howl[0].reason).toBe('limiter-saturated');
    expect(howl[0].freqHz).toBeNull();
    expect(howl[0].at - start).toBeGreaterThan(ms(LIMITER_BACKSTOP_MS));
    expect(howl[0].at - start).toBeLessThan(ms(LIMITER_BACKSTOP_MS + 400));
    expect(r.of('autoMute')[0].reason).toMatch(/^Innesco probabile/);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(L[i]));
    expect(peak).toBeLessThanOrEqual(ceil + 1e-9);
  });

  /**
   * A cutoff dragged with the slider (v-dsp H2): one params message every `updateMs` for `dragS` s from
   * t = 2 s, steps within LIVE_FREQ_RATIO (applied live, no reconfigure envelope), `tone` at the reference.
   * Returns the howl events and the output level before and after the drag (dB).
   */
  function dragRun(set: (p: EngineParams, u: number) => void, tone: (t: number) => number, dragS: number, updateMs: number) {
    const r = new Rig();
    r.params((p) => {
      p.masterDb = 0;
      p.channels[0].gainDb = 0;
      p.channels[1].enabled = false;
      set(p, 0);
    });
    r.send({ type: 'arm', armed: true });
    const n = Math.round((2 + dragS + 2) * fs);
    const x = Float32Array.from({ length: n }, (_, i) => tone(i / fs));
    const step = ms(updateMs);
    let next = 2 * fs;
    const out = new Float32Array(n);
    while (r.pos < n) {
      if (r.pos >= next && r.pos <= (2 + dragS) * fs + step) {
        const u = Math.min(1, (r.pos - 2 * fs) / (dragS * fs));
        r.params((p) => set(p, u));
        next += step;
      }
      const at = r.pos;
      out.set(r.run(Math.min(128, n - at), x).L, at);
    }
    const level = (from: number): number => {
      let e = 0;
      for (let i = from; i < from + ms(200); i++) e += out[i] * out[i];
      return 10 * Math.log10(e / ms(200));
    };
    return { howls: [...r.of('howl'), ...r.of('autoMute')], before: level(ms(1700)), after: level(n - ms(300)) };
  }

  for (const [dragS, updateMs] of [
    [0.75, 16],
    [1.5, 16],
    [1.5, 33],
    [3, 16],
  ] as const) {
    it(`dragging the HP cutoff 200 → 20 Hz in ${dragS} s (${updateMs} ms updates) with a 50 Hz hum is no howl (v-dsp H2)`, () => {
      const o = dragRun(
        (p, u) => {
          p.filters.hpFreq = 200 * Math.pow(0.1, u);
        },
        (t) => 0.16 * Math.sin(2 * Math.PI * 50 * t),
        dragS,
        updateMs,
      );
      // The hum's anti-noise rises by ~20 dB with no commanded-gain change.
      expect(o.after - o.before).toBeGreaterThan(15);
      expect(o.howls).toEqual([]);
    });
  }

  it('dragging the LP cutoff 300 → 3000 Hz in 1.5 s with a 1.5 kHz tone is no howl (v-dsp H2)', () => {
    const o = dragRun(
      (p, u) => {
        p.filters.lpFreq = 300 * Math.pow(10, u);
      },
      (t) => 0.1 * Math.sin(2 * Math.PI * 1500 * t),
      1.5,
      16,
    );
    expect(o.after - o.before).toBeGreaterThan(15);
    expect(o.howls).toEqual([]);
  });

  it('after a howl, unmuting re-arms the detectors (they are not left latched)', () => {
    const r = new Rig();
    r.params((p) => {
      unity(p);
      p.ceilingDb = ceilingDb;
      p.channels[1].enabled = false;
    });
    r.send({ type: 'arm', armed: true });
    const n = 8 * fs;
    const amb = noise(n, 0.01, 13);
    const outL = new Float32Array(n);
    const outR = new Float32Array(n);
    const inp = new Float32Array(n);
    const feed = (i: number, oL: Float32Array) => amb[i] + (i >= 2 * fs ? 1.5 * oL[i - 300] : 0);
    r.runLoop(3 * fs, feed, outL, outR, inp);
    expect(r.of('howl').length).toBe(1);
    r.send({ type: 'mute', muted: false });
    r.runLoop(5 * fs, feed, outL, outR, inp);
    expect(r.of('howl').length).toBe(2);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(outL[i]));
    expect(peak).toBeLessThanOrEqual(ceil + 1e-9);
  });
});

describe('AntiPhaseEngine: feedback neutralisation', () => {
  it('with the exact FIR the fed-back component in cond is reduced by > 30 dB', () => {
    const bulk = 1300;
    const rnd = mulberry32(14);
    const h = new Float64Array(48);
    for (let i = 0; i < h.length; i++) h[i] = 0.4 * (2 * rnd() - 1) * Math.exp(-i / 10);
    const run = (withFir: boolean) => {
      const r = new Rig({ testBypassConditioning: true });
      r.params((p) => {
        unity(p);
        p.channels[0].gainDb = -6;
        p.channels[1].enabled = false;
      });
      if (withFir) r.send({ type: 'feedbackFir', channel: 0, bulkDelay: bulk, taps: h });
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.tapCopies = [];
      const n = 3 * fs;
      const s = noise(n, 0.05, 15);
      const fed = new Float64Array(n);
      const outL = new Float32Array(n);
      const outR = new Float32Array(n);
      const inp = new Float32Array(n);
      r.runLoop(
        n,
        (i, oL) => {
          let acc = 0;
          for (let j = 0; j < h.length; j++) if (i - bulk - j >= 0) acc += h[j] * oL[i - bulk - j];
          fed[i] = acc;
          return s[i] + acc;
        },
        outL,
        outR,
        inp,
      );
      // cond − (input − fed) = fed − prediction: the part of the feedback left in cond
      const F = TAP_BLOCK_FRAMES;
      expect(r.tapCopies.length).toBe(Math.floor(n / F));
      let pFed = 0;
      let pRes = 0;
      let idx = 0;
      for (const b of r.tapCopies) {
        for (let j = 0; j < F; j++, idx++) {
          if (idx < 2 * fs) continue;
          const cond = b[TAP_CHANNELS.indexOf('cond') * F + j];
          const res = cond - (inp[idx] - fed[idx]);
          pFed += fed[idx] * fed[idx];
          pRes += res * res;
        }
      }
      expect(pFed).toBeGreaterThan(0);
      return 10 * Math.log10(pFed / pRes);
    };
    expect(run(true)).toBeGreaterThan(30);
    expect(run(false)).toBeLessThan(1);
  });

  it('setting the FIR while running goes through the reconfigure ramp (no step in the output)', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params(unity);
    r.send({ type: 'arm', armed: true });
    const x = constant(4 * fs, 0.1);
    r.run(ms(1600), x);
    const before = r.run(1000, x).L;
    const taps = new Float64Array(16).fill(0.01);
    r.send({ type: 'feedbackFir', channel: 0, bulkDelay: 500, taps });
    const y = r.run(fs, x).L;
    const steps = maxAbsDiff(y, 0, y.length);
    expect(maxAbsDiff(before, 0, before.length)).toBe(0);
    // ramps of ≈ 0.1/720 per sample instead of a 0.016 step in x̂
    expect(steps).toBeLessThan(0.2 / ms(15));
  });
});

describe('AntiPhaseEngine: channel 2 check', () => {
  const statusAfter = (in1: Float32Array | null, in0: Float32Array) => {
    const r = new Rig();
    r.run(ms(2100), in0, in1);
    const checks = r.of('channelCheck');
    expect(checks.length).toBeGreaterThan(0);
    expect(r.lastMeters().ch2).toBe(checks[checks.length - 1].status);
    return checks[checks.length - 1].status;
  };
  const n = 3 * fs;
  const a = noise(n, 0.05, 16);

  it('absent', () => expect(statusAfter(null, a)).toBe('absent'));
  it('silent', () => expect(statusAfter(new Float32Array(n), a)).toBe('silent'));
  it('duplicate', () => expect(statusAfter(a, a)).toBe('duplicate'));
  it('ok', () => expect(statusAfter(noise(n, 0.05, 17), a)).toBe('ok'));
  describe('error mic on input 1 (reference on input 2): the check is about input 1', () => {
    const statusOf = (in0: Float32Array, in1: Float32Array) => {
      const r = new Rig();
      r.params((p) => {
        p.refChannel = 1;
        p.errorChannel = 0;
      });
      r.run(ms(2100), in0, in1);
      return r.of('channelCheck').at(-1)?.status;
    };
    it('silent error mic → silent (formerly ok: input 2 carries the reference)', () => expect(statusOf(new Float32Array(n), a)).toBe('silent'));
    it('ok', () => expect(statusOf(noise(n, 0.05, 17), a)).toBe('ok'));
    it('duplicate', () => expect(statusOf(a, a)).toBe('duplicate'));
    it('mono track → absent', () => {
      const r = new Rig();
      r.params((p) => (p.errorChannel = 0));
      r.run(ms(2100), a, null);
      expect(r.of('channelCheck').at(-1)?.status).toBe('absent');
    });
  });
  it('starts pending and reports only changes', () => {
    const r = new Rig();
    r.run(ms(500), a, null);
    expect(r.lastMeters().ch2).toBe('pending');
    r.run(ms(3100), a, null);
    expect(r.of('channelCheck').length).toBe(1);
  });
});

describe('AntiPhaseEngine: taps', () => {
  it('planar blocks with the 5 TAP_CHANNELS in order; recycle reuses buffers', () => {
    expect(TAP_CHANNELS).toEqual(['ref', 'cond', 'outL', 'outR', 'err']);
    const r = new Rig({ testBypassConditioning: true });
    r.params((p) => {
      unity(p);
      p.errorChannel = 1;
    });
    r.send({ type: 'arm', armed: true });
    r.send({ type: 'tap', enabled: true });
    const n = 12 * TAP_BLOCK_FRAMES;
    const x0 = noise(n, 0.1, 18);
    const x1 = noise(n, 0.1, 19);
    const out = r.run(n, x0, x1);
    const taps = r.of('tap');
    // pool of 8 and nothing recycled: the rest is dropped
    expect(taps.length).toBe(8);
    const F = TAP_BLOCK_FRAMES;
    for (const [b, t] of taps.entries()) {
      expect(t.frames).toBe(F);
      expect(t.block.length).toBe(TAP_CHANNELS.length * F);
      const o = b * F;
      for (let j = 0; j < F; j += 97) {
        expect(t.block[j]).toBe(x0[o + j]);
        expect(t.block[F + j]).toBe(x0[o + j]);
        expect(t.block[2 * F + j]).toBe(out.L[o + j]);
        expect(t.block[3 * F + j]).toBe(out.R[o + j]);
        expect(t.block[4 * F + j]).toBe(x1[o + j]);
      }
    }
    const first = taps[0].block;
    r.send({ type: 'recycle', buffer: first });
    r.run(2 * F, x0, x1);
    const more = r.of('tap');
    expect(more.length).toBe(9);
    expect(more[8].block).toBe(first);
    // disabled taps produce nothing
    r.send({ type: 'tap', enabled: false });
    for (const t of more) r.send({ type: 'recycle', buffer: t.block });
    r.run(3 * F, x0, x1);
    expect(r.of('tap').length).toBe(9);
  });
});

describe('AntiPhaseEngine: robustness', () => {
  it('NaN/Infinity inputs, parameters (harmonic trims too) and stimuli never yield NaN outputs', () => {
    const r = new Rig();
    r.params((p) => {
      p.masterDb = Number.NaN;
      p.channels[0].delayMs = Number.POSITIVE_INFINITY;
      p.channels[0].gainDb = 0;
      p.fRef = Number.NaN;
      p.phaseMode = 'tonal';
      p.channels[0].harmonicTrims[0] = { gainDb: Number.NaN, phaseDeg: Number.POSITIVE_INFINITY };
      p.channels[0].harmonicTrims[1] = { gainDb: Number.POSITIVE_INFINITY, phaseDeg: Number.NaN };
      p.channels[1].harmonicTrims = [null, { gainDb: 1e300, phaseDeg: -1e300 }] as unknown as HarmonicTrim[];
    });
    r.send({ type: 'arm', armed: true });
    const n = 3 * fs;
    const x = noise(n, 0.1, 20);
    for (let i = 0; i < n; i += 997) x[i] = Number.NaN;
    for (let i = 500; i < n; i += 1999) x[i] = Number.POSITIVE_INFINITY;
    const x1 = noise(n, 0.1, 21);
    x1.fill(Number.NaN, 1000, 3000);
    const a = r.run(fs, x, x1);
    const stim = noise(4000, 0.1, 22);
    stim[100] = Number.NaN;
    r.send({ type: 'measure', id: 1, stimulus: stim, channel: 0, postRollFrames: 100 });
    const b = r.run(2 * fs, x, x1, 64);
    for (const y of [a.L, a.R, b.L, b.R]) expect(y.every(Number.isFinite)).toBe(true);
    let energy = 0;
    for (const v of b.L) energy += v * v;
    expect(energy).toBeGreaterThan(0);
    const m = r.of('measureResult')[0];
    expect(m.ref.every(Number.isFinite)).toBe(true);
    for (const mm of r.of('meters')) {
      expect(Number.isFinite(mm.ref.rmsDb) && Number.isFinite(mm.outL.peakDb)).toBe(true);
    }
  });

  it('short or mismatched arrays and malformed messages do not throw', () => {
    const r = new Rig();
    r.send({ type: 'arm', armed: true });
    const L = new Float32Array(128);
    const R = new Float32Array(64);
    expect(() => r.e.process(new Float32Array(10), null, L, R)).not.toThrow();
    expect(L.every(Number.isFinite)).toBe(true);
    const bad = [null, 42, { type: 'nope' }, { type: 'params' }, { type: 'recycle', buffer: [1, 2] }, { type: 'measure' }];
    for (const m of bad) expect(() => r.e.handleMessage(m as unknown as WorkletInMsg)).not.toThrow();
    expect(() => r.e.process(null, null, L, L)).not.toThrow();
    expect(r.e.params.masterDb).toBe(clampParams(undefined).masterDb);
  });

  it('meters report delay, clamping, block size and limiter GR', () => {
    const r = new Rig();
    r.params((p) => {
      p.channels[0].delayMs = 1;
      p.channels[1].phaseDeltaDeg = -90;
    });
    r.run(ms(100), noise(fs, 0.1, 23), null, 64);
    const m = r.lastMeters();
    expect(m.delaySamples[0]).toBeCloseTo(1 + 48, 9);
    expect(m.delayClamped).toEqual([false, true]);
    expect(m.delaySamples[1]).toBe(1);
    expect(m.blockSize).toBe(64);
    expect(m.limiterGrDb).toEqual([0, 0]);
    expect(m.err).toBeNull();
    expect(r.of('meters').length).toBeGreaterThanOrEqual(2);
  });
});

describe('AntiPhaseEngine: fuzzed control sequences', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  for (const seed of [101, 202, 303]) {
    it(`random messages and block sizes: no throw, no NaN, never above −1 dBFS, zeros while disarmed (seed ${seed})`, () => {
      const rnd = mulberry32(seed);
      const pick = <T,>(a: readonly T[]): T => a[Math.floor(rnd() * a.length)];
      const r = new Rig({ softStartMs: 200 });
      const maxCeil = Math.pow(10, -1 / 20);
      const x0 = noise(13 * fs, 0.5, seed);
      const x1 = noise(13 * fs, 0.5, seed + 1);
      for (let i = 0; i < 13 * fs; i += 4801) x0[i] = Number.NaN;
      let armed = false;
      let blocks = 0;
      let nonFinite = 0;
      let overCeiling = 0;
      let leaks = 0;
      let energy = 0;
      while (r.pos < 12 * fs && blocks < 400) {
        blocks++;
        const kind = Math.floor(rnd() * 14);
        switch (kind) {
          case 0:
          case 1:
            r.params((p) => {
              p.masterDb = -60 + 60 * rnd();
              p.ceilingDb = -24 + 23 * rnd();
              p.phaseMode = pick(['delay', 'tonal', 'hilbert'] as const);
              p.fRef = 30 + 500 * rnd();
              p.tonal.f0 = 40 + 400 * rnd();
              p.tonal.harmonics = 1 + Math.floor(8 * rnd());
              p.tonal.q = 2 + 100 * rnd();
              p.filters.lpEnabled = rnd() < 0.8;
              p.filters.lpFreq = 100 + 3000 * rnd();
              p.filters.lpOrder = pick([2, 4] as const);
              p.filters.hpEnabled = rnd() < 0.8;
              p.filters.hpFreq = 10 + 100 * rnd();
              p.filters.barrierEnabled = rnd() < 0.3;
              p.refChannel = pick([0, 1] as const);
              p.errorChannel = pick([0, 1, -1] as const);
              p.gainUnlocked = rnd() < 0.2;
              p.headphones = rnd() < 0.1;
              for (const c of p.channels) {
                c.enabled = rnd() < 0.9;
                c.polarityInvert = rnd() < 0.5;
                c.gainDb = -60 + 72 * rnd();
                c.delayMs = rnd() < 0.3 ? 0 : 500 * rnd() * rnd();
                c.phaseDeltaDeg = -180 + 360 * rnd();
                c.harmonicTrims =
                  rnd() < 0.3
                    ? neutralTrims()
                    : Array.from({ length: 8 }, () => ({
                        gainDb: rnd() < 0.05 ? Number.NaN : -40 + 50 * rnd(),
                        phaseDeg: -200 + 400 * rnd(),
                      }));
              }
            });
            break;
          case 2:
            armed = rnd() < 0.7;
            r.send({ type: 'arm', armed });
            break;
          case 3:
            r.send({ type: 'mute', muted: rnd() < 0.5 });
            break;
          case 4:
            r.send({ type: 'bypass', bypass: rnd() < 0.5 });
            break;
          case 5:
            r.send({ type: 'panic' });
            break;
          case 6:
            r.send({ type: 'softStart' });
            break;
          case 7: {
            const stim = noise(Math.floor(rnd() * 20000), 0.5 + rnd(), seed + blocks);
            r.send({ type: 'measure', id: blocks, stimulus: stim, channel: pick([0, 1] as const), postRollFrames: Math.floor(rnd() * 5000) });
            break;
          }
          case 8:
            r.send({ type: 'abortMeasure' });
            break;
          case 9: {
            const taps = rnd() < 0.2 ? null : noise(1 + Math.floor(rnd() * 1500), 0.05, seed + blocks);
            r.send({ type: 'feedbackFir', channel: pick([0, 1] as const), bulkDelay: 1 + rnd() * 30000, taps: taps ? Float64Array.from(taps) : null });
            break;
          }
          case 10:
            r.send({ type: 'tap', enabled: rnd() < 0.7 });
            break;
          case 11:
            for (const t of r.of('tap').slice(-4)) r.send({ type: 'recycle', buffer: t.block });
            break;
          case 12:
            r.send({ type: 'resetHowl' });
            break;
          default:
            break;
        }
        const n = pick([1, 7, 64, 128, 128, 128, 256, 1000, 4096]);
        const { L, R } = r.run(n, rnd() < 0.1 ? null : x0, rnd() < 0.3 ? null : x1, n);
        for (let i = 0; i < n; i++) {
          const l = L[i];
          const rr = R[i];
          if (!Number.isFinite(l) || !Number.isFinite(rr)) nonFinite++;
          if (Math.abs(l) > maxCeil || Math.abs(rr) > maxCeil) overCeiling++;
          if (!armed && (l !== 0 || rr !== 0)) leaks++;
          if (armed) energy += l * l + rr * rr;
        }
      }
      expect({ nonFinite, overCeiling, leaks }).toEqual({ nonFinite: 0, overCeiling: 0, leaks: 0 });
      expect(energy).toBeGreaterThan(0);
      expect(r.of('fault')).toEqual([]);
    });
  }

  it('an internal exception becomes zeros + fault + muted, and recovers on unmute', () => {
    const r = new Rig({ testBypassConditioning: true });
    r.params(unity);
    r.send({ type: 'arm', armed: true });
    const x = noise(6 * fs, 0.1, 30);
    r.run(ms(1600), x);
    const spy = vi.spyOn(DelayLine.prototype, 'write').mockImplementationOnce(() => {
      throw new Error('kaputt');
    });
    const out = r.run(128, x);
    spy.mockRestore();
    expect(out.L.every((v) => v === 0) && out.R.every((v) => v === 0)).toBe(true);
    const faults = r.of('fault');
    expect(faults.length).toBe(1);
    expect(faults[0].message).toMatch(/kaputt/);
    const after = r.run(fs, x);
    expect(after.L.every((v) => v === 0)).toBe(true);
    expect(r.lastMeters().muted).toBe(true);
    r.send({ type: 'mute', muted: false });
    const back = r.run(2 * fs, x);
    const a = r.pos - 2 * fs;
    for (let i = ms(1500) + 2; i < 2 * fs; i++) expect(Math.abs(back.L[i] + x[a + i - 1])).toBeLessThan(1e-9);
  });
});

describe('AntiPhaseEngine: harmonic trims (tonal mode)', () => {
  const randomTrims = (rnd: () => number): HarmonicTrim[] =>
    Array.from({ length: 8 }, () => ({ gainDb: -12 + 18 * rnd(), phaseDeg: -180 + 360 * rnd() }));
  const only = (h: number, gainDb: number, phaseDeg: number): HarmonicTrim[] => {
    const t = neutralTrims();
    t[h - 1] = { gainDb, phaseDeg };
    return t;
  };

  /** Engine at `rate` with the params `mod` sets, armed, run over x (128-sample blocks). */
  function runAt(rate: number, mod: (p: EngineParams) => void, x: Float32Array): { L: Float32Array; R: Float32Array; p: EngineParams } {
    const e = new AntiPhaseEngine(rate, { softStartMs: 50 });
    const p = e.params;
    mod(p);
    e.handleMessage({ type: 'params', params: p });
    e.handleMessage({ type: 'arm', armed: true });
    const L = new Float32Array(x.length);
    const R = new Float32Array(x.length);
    for (let off = 0; off < x.length; off += 128) {
      const m = Math.min(128, x.length - off);
      e.process(x.subarray(off, off + m), null, L.subarray(off, off + m), R.subarray(off, off + m));
      e.drain(() => {});
    }
    return { L, R, p: e.params };
  }

  function dftAt(x: Float32Array, f: number, from: number, len: number, rate: number): Complex {
    let re = 0;
    let im = 0;
    for (let i = from; i < from + len; i++) {
      const w = (2 * Math.PI * f * i) / rate;
      re += x[i] * Math.cos(w);
      im -= x[i] * Math.sin(w);
    }
    return { re, im };
  }

  /** Sum of the first `h` harmonics of f0, 0.04 each, random phases. */
  function harmonicsSignal(n: number, f0: number, h: number, rate: number, rnd: () => number): Float32Array {
    const ph = Array.from({ length: h }, () => 2 * Math.PI * rnd());
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let k = 1; k <= h; k++) v += 0.04 * Math.sin((2 * Math.PI * k * f0 * i) / rate + ph[k - 1]);
      x[i] = v;
    }
    return x;
  }

  const tonal = (f0: number, harmonics: number) => (p: EngineParams): void => {
    p.phaseMode = 'tonal';
    p.tonal = { f0, harmonics, q: 30 };
    p.filters.lpFreq = 2000;
    p.masterDb = -6;
    p.ceilingDb = -1;
    p.channels[0].gainDb = 0;
    p.channels[0].delayMs = 1.3;
    p.channels[0].phaseDeltaDeg = 25;
    p.channels[1].gainDb = -3;
    p.channels[1].delayMs = 4.1;
    p.channels[1].phaseDeltaDeg = -70;
    p.channels[1].polarityInvert = false;
  };

  /** Peak of y above 2 kHz (zero-phase 8th-order high-pass) in [from, to), dB re `ref`. */
  function clickDb(y: Float32Array, from: number, to: number, ref: number): number {
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
    return 20 * Math.log10(pk / ref);
  }

  for (const [rate, f0] of [[44100, 90], [48000, 120], [96000, 75]] as const) {
    it(`${rate} Hz, harmonics 1…8, random trims (L ≠ R): output = channelResponse at every harmonic (±0.5°, ±0.05 dB)`, () => {
      const rnd = mulberry32(rate);
      for (let h = 1; h <= 8; h++) {
        const n = Math.round(2.5 * rate);
        const x = harmonicsSignal(n, f0, h, rate, rnd);
        const tL = randomTrims(rnd);
        const tR = randomTrims(rnd);
        const { L, R, p } = runAt(rate, (q) => {
          tonal(f0, h)(q);
          q.channels[0].harmonicTrims = tL;
          q.channels[1].harmonicTrims = tR;
        }, x);
        const from = n - rate;
        for (let k = 1; k <= h; k++) {
          const xin = dftAt(x, k * f0, from, rate, rate);
          for (const [ch, y] of [[0, L], [1, R]] as const) {
            const got = ratio(dftAt(y, k * f0, from, rate, rate), xin);
            const want = channelResponse(p, rate, k * f0, ch);
            expect(Math.abs(wrapDeg(deg(cArg(got) - cArg(want)))), `h=${h} k=${k} ch=${ch}`).toBeLessThan(0.5);
            expect(Math.abs(db(got) - db(want)), `h=${h} k=${k} ch=${ch}`).toBeLessThan(0.05);
          }
        }
      }
    });
  }

  it('total phase at h·f0 = polarity + h·Δ + θ_h: a +30° trim at h = 3 lags 3·f0 by 30° more, nothing else moves', () => {
    const f0 = 100;
    const x = harmonicsSignal(3 * fs, f0, 4, fs, mulberry32(77));
    const base = runAt(fs, tonal(f0, 4), x);
    const lag = runAt(fs, (q) => {
      tonal(f0, 4)(q);
      q.channels[0].harmonicTrims = only(3, 0, 30);
      q.channels[1].harmonicTrims = only(2, -6, 0);
    }, x);
    const from = 2 * fs;
    for (let k = 1; k <= 4; k++) {
      for (const ch of ['L', 'R'] as const) {
        const a = dftAt(base[ch], k * f0, from, fs, fs);
        const b = dftAt(lag[ch], k * f0, from, fs, fs);
        const extraLag = -wrapDeg(deg(cArg(b) - cArg(a)));
        const dDb = db(b) - db(a);
        const wantLag = ch === 'L' && k === 3 ? 30 : 0;
        const wantDb = ch === 'R' && k === 2 ? -6 : 0;
        expect(Math.abs(extraLag - wantLag), `${ch} h${k}`).toBeLessThan(0.1);
        expect(Math.abs(dDb - wantDb), `${ch} h${k}`).toBeLessThan(0.02);
      }
    }
  });

  it('trims act only in tonal mode, on the harmonics in use; the cond tap stays the neutral bank', () => {
    const x = harmonicsSignal(fs, 100, 8, fs, mulberry32(5));
    const trims = randomTrims(mulberry32(6));
    for (const [name, mod] of [
      ['delay mode', (p: EngineParams) => (p.phaseMode = 'delay')],
      ['hilbert mode', (p: EngineParams) => (p.phaseMode = 'hilbert')],
    ] as const) {
      const a = runAt(fs, (p) => { tonal(100, 4)(p); mod(p); }, x);
      const b = runAt(fs, (p) => {
        tonal(100, 4)(p);
        mod(p);
        p.channels[0].harmonicTrims = trims;
        p.channels[1].harmonicTrims = trims;
      }, x);
      expect(b.L.every((v, i) => v === a.L[i]), name).toBe(true);
      expect(b.R.every((v, i) => v === a.R[i]), name).toBe(true);
    }
    // harmonics 3: trims of h4…h8 are ignored
    const above = neutralTrims().map((t, i) => (i >= 3 ? { gainDb: 6, phaseDeg: 120 } : t));
    const a = runAt(fs, tonal(100, 3), x);
    const b = runAt(fs, (p) => { tonal(100, 3)(p); p.channels[0].harmonicTrims = above; }, x);
    expect(b.L.every((v, i) => v === a.L[i])).toBe(true);
    // cond tap: identical with and without trims
    const conds: Float32Array[] = [];
    for (const t of [neutralTrims(), trims]) {
      const r = new Rig({ softStartMs: 50 });
      r.tapCopies = [];
      r.params((p) => {
        tonal(100, 4)(p);
        p.channels[0].harmonicTrims = t;
      });
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.run(4 * TAP_BLOCK_FRAMES, x);
      expect(r.tapCopies.length).toBe(4);
      const F = TAP_BLOCK_FRAMES;
      conds.push(Float32Array.from(r.tapCopies.flatMap((blk) => Array.from(blk.subarray(F, 2 * F)))));
    }
    expect(conds[1].every((v, i) => v === conds[0][i])).toBe(true);
  });

  it('block sizes 1, 64, 128, 256, 4096 give identical outputs with trims, a trim change and a rise under way', () => {
    const n = fs;
    const x = harmonicsSignal(n, 137, 4, fs, mulberry32(9));
    const nz = noise(n, 0.02, 10);
    for (let i = 0; i < n; i++) x[i] += nz[i];
    const runWith = (block: number) => {
      const r = new Rig({ softStartMs: 300 });
      r.params((p) => {
        tonal(137, 4)(p);
        p.channels[0].harmonicTrims = only(2, -30, 40);
        p.channels[1].harmonicTrims = only(1, 3, -60);
      });
      r.send({ type: 'arm', armed: true });
      const a = r.run(ms(400), x, null, block);
      r.params((p) => {
        p.channels[0].harmonicTrims = only(2, 6, -20); // a rise of 36 dB: 6 dB/s
        p.channels[1].harmonicTrims = only(3, -4, 90);
        p.tonal.f0 = 138;
      });
      const b = r.run(n - ms(400), x, null, block);
      return { L: [...a.L, ...b.L], R: [...a.R, ...b.R] };
    };
    const ref = runWith(128);
    for (const block of [1, 64, 256, 4096]) {
      const o = runWith(block);
      let worst = 0;
      for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(o.L[i] - ref.L[i]), Math.abs(o.R[i] - ref.R[i]));
      expect(worst, `block ${block}`).toBeLessThan(1e-12);
    }
  });

  it('a trim gain rise of more than BIG_JUMP_DB goes at ≤ 6 dB/s (−30 → +6 dB: ≥ 5 s); the decrease is immediate', () => {
    const f0 = 100;
    const r = new Rig({ softStartMs: 50 });
    r.params((p) => {
      tonal(f0, 4)(p);
      p.channels[0].harmonicTrims = only(2, -30, 0);
    });
    r.send({ type: 'arm', armed: true });
    const x = sine(12 * fs, 2 * f0, 0.2);
    r.run(fs, x);
    const win = ms(20); // 4 periods of 200 Hz
    const level = (y: Float32Array, at: number) => db(dftAt(y, 2 * f0, at, win, fs));
    const start = level(r.run(win, x).L, 0);
    r.params((p) => (p.channels[0].harmonicTrims = only(2, 6, 0)));
    const rise = r.run(7 * fs, x).L;
    let reached = Infinity;
    for (let at = 0; at + win <= rise.length; at += win) {
      const rel = level(rise, at) - start;
      const t = (at + win) / fs;
      expect(rel, `${t.toFixed(2)} s`).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S * t + 0.05);
      if (rel > 35.9 && reached === Infinity) reached = at / fs;
    }
    expect(reached).toBeGreaterThanOrEqual(4.95);
    expect(reached).toBeLessThan(5.3);
    expect(Math.abs(level(rise, rise.length - win) - start - 36)).toBeLessThan(0.05);
    r.params((p) => (p.channels[0].harmonicTrims = only(2, -30, 0)));
    const fall = r.run(ms(100), x).L;
    expect(Math.abs(level(fall, ms(40)) - start)).toBeLessThan(0.05);
  });

  it('a trim raised while bypassed waits, then rises at 6 dB/s once the anti-noise is heard again', () => {
    const r = new Rig({ softStartMs: 50 });
    r.params((p) => {
      tonal(100, 4)(p);
      p.channels[0].harmonicTrims = only(1, -30, 0);
    });
    r.send({ type: 'arm', armed: true });
    const x = sine(8 * fs, 100, 0.2);
    r.run(fs, x);
    const level = (y: Float32Array, at: number) => db(dftAt(y, 100, at, ms(20), fs));
    const start = level(r.run(ms(20), x).L, 0);
    r.send({ type: 'bypass', bypass: true });
    r.run(ms(50), x);
    r.params((p) => (p.channels[0].harmonicTrims = only(1, 6, 0)));
    r.run(3 * fs, x);
    r.send({ type: 'bypass', bypass: false });
    const y = r.run(fs, x).L;
    // after the 15 ms bypass ramp: ≤ 6 dB + 6 dB/s, not the +36 dB the target asks for
    expect(level(y, ms(30)) - start).toBeLessThan(BIG_JUMP_DB + MAX_RISE_DB_PER_S * 0.05 + 0.1);
    expect(level(y, fs - ms(20)) - start).toBeLessThan(BIG_JUMP_DB + MAX_RISE_DB_PER_S * 1 + 0.1);
    expect(level(y, fs - ms(20)) - start).toBeGreaterThan(MAX_RISE_DB_PER_S * 0.9);
  });

  it('trim changes and the 6 dB/s staircase are click-free (nothing above 2 kHz, ≤ −100 dB)', () => {
    for (const f of [50, 100, 400]) {
      const r = new Rig({ softStartMs: 50 });
      r.params((p) => {
        tonal(f, 4)(p);
        p.masterDb = 0;
      });
      r.send({ type: 'arm', armed: true });
      const amp = 0.2;
      const x = sine(9 * fs, f, amp);
      r.run(2 * fs, x);
      const pre = r.run(ms(100), x).L;
      r.params((p) => (p.channels[0].harmonicTrims = only(1, 6, 90)));
      const post = r.run(ms(400), x).L;
      const y = Float32Array.from([...pre, ...post]);
      expect(clickDb(y, ms(100) - 240, ms(100) + 4800, amp), `0 → +6 dB / 90° at ${f} Hz`).toBeLessThan(-100);
      r.params((p) => (p.channels[0].harmonicTrims = only(1, -30, -170)));
      r.run(ms(100), x);
      r.params((p) => (p.channels[0].harmonicTrims = only(1, 6, -170)));
      const rise = r.run(6 * fs, x).L;
      expect(clickDb(rise, ms(50), rise.length - ms(50), amp), `6 dB/s rise at ${f} Hz`).toBeLessThan(-100);
    }
  });

  it('an f0 glide (live) and a harmonics change (structural) keep the trims: settled output = channelResponse', () => {
    const t = randomTrims(mulberry32(44));
    const f1 = 102;
    const r = new Rig({ softStartMs: 50 });
    r.params((p) => {
      tonal(100, 4)(p);
      p.channels[0].harmonicTrims = t;
      p.channels[1].harmonicTrims = only(4, 4, -45);
    });
    r.send({ type: 'arm', armed: true });
    const x = harmonicsSignal(10 * fs, f1, 6, fs, mulberry32(45));
    r.run(fs, x);
    const before = r.run(ms(200), x).L;
    const limit = maxAbsDiff(before, 0, before.length);
    r.params((p) => (p.tonal.f0 = f1));
    let a = r.pos;
    let y = r.run(3 * fs, x);
    expect(maxAbsDiff(y.L, 0, ms(200))).toBeLessThan(2 * limit);
    const check = (k: number, from: number) => {
      for (const [ch, out] of [[0, y.L], [1, y.R]] as const) {
        const got = ratio(dftAt(out, k * f1, from, fs, fs), dftAt(x.subarray(a), k * f1, from, fs, fs));
        const want = channelResponse(r.e.params, fs, k * f1, ch);
        expect(Math.abs(wrapDeg(deg(cArg(got) - cArg(want)))), `h${k} ch${ch}`).toBeLessThan(0.5);
        expect(Math.abs(db(got) - db(want)), `h${k} ch${ch}`).toBeLessThan(0.05);
      }
    };
    for (let k = 1; k <= 4; k++) check(k, 2 * fs);
    // 4 → 6 harmonics: a structural swap; the new chain takes the trims of h5 and h6 too
    r.params((p) => {
      p.tonal.harmonics = 6;
      p.channels[0].harmonicTrims[4] = { gainDb: 3, phaseDeg: 100 };
    });
    a = r.pos;
    y = r.run(3 * fs, x);
    for (let k = 1; k <= 6; k++) check(k, 2 * fs);
  });

  it('measurements are unaffected: the same stimulus output and recording with or without trims', () => {
    const results = [neutralTrims(), Array.from({ length: 8 }, () => ({ gainDb: 6, phaseDeg: 90 }))].map((t) => {
      const r = new Rig({ softStartMs: 50 });
      r.params((p) => {
        tonal(100, 8)(p);
        p.channels[0].harmonicTrims = t;
        p.channels[1].harmonicTrims = t;
      });
      r.send({ type: 'arm', armed: true });
      const x = harmonicsSignal(3 * fs, 100, 8, fs, mulberry32(3));
      r.run(fs, x);
      const stim = exponentialSweep({ fs, f1: 50, f2: 2000, durationS: 0.5, fadeMs: 20, amplitude: 0.3 });
      r.send({ type: 'measure', id: 1, stimulus: stim, channel: 0, postRollFrames: 1000 });
      const { L, R } = r.run(fs, x);
      const res = r.of('measureResult')[0];
      expect(res.aborted).toBe(false);
      // while the stimulus plays (after the 15 ms gate) the anti-noise is at 0
      return { L: L.subarray(ms(15), ms(15) + stim.length), R: R.subarray(ms(15), ms(15) + stim.length), ref: res.ref };
    });
    const [a, b] = results;
    expect(b.L.every((v, i) => v === a.L[i])).toBe(true);
    expect(b.R.every((v, i) => v === 0 && a.R[i] === 0)).toBe(true);
    expect(b.ref.every((v, i) => v === a.ref[i])).toBe(true);
  });

  it('+6 dB trims on every harmonic of a loud tone: never above the ceiling', () => {
    const r = new Rig({ softStartMs: 50 });
    r.params((p) => {
      tonal(100, 8)(p);
      p.masterDb = 0;
      p.ceilingDb = -12;
      p.gainUnlocked = true;
      p.channels[0].gainDb = 6;
      for (const c of p.channels) c.harmonicTrims = Array.from({ length: 8 }, (_, k) => ({ gainDb: 6, phaseDeg: 45 * k }));
    });
    r.send({ type: 'arm', armed: true });
    const x = harmonicsSignal(4 * fs, 100, 8, fs, mulberry32(12));
    for (let i = 0; i < x.length; i++) x[i] *= 10;
    const { L, R } = r.run(4 * fs, x);
    const ceil = Math.pow(10, -12 / 20);
    let peak = 0;
    for (let i = 0; i < L.length; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
    expect(peak).toBeLessThanOrEqual(ceil);
    expect(peak).toBeGreaterThan(0.99 * ceil);
  });

  describe('a harmonic rises with the channel gain (and master) and its trim: one budget for both', () => {
    /** Level at f0 of channel L, 0.1 s windows (10 periods), relative to the 0.1 s before `change` at t = 1 s. */
    function rise(start: (p: EngineParams) => void, change: (p: EngineParams) => void, at: number[]): number[] {
      const r = new Rig({ softStartMs: 10 });
      r.params((p) => {
        tonal(100, 4)(p);
        p.masterDb = 0;
        p.ceilingDb = -1;
        p.channels[0].gainDb = -20;
        start(p);
      });
      r.send({ type: 'arm', armed: true });
      const x = sine(5 * fs, 100, 0.01);
      const before = r.run(fs, x).L;
      const level = (y: Float32Array, from: number) => db(dft(y, 100, from, ms(100)));
      const ref = level(before, fs - ms(100));
      r.params(change);
      const y = r.run(3 * fs, x).L;
      return at.map((t) => level(y, ms(t * 1000)) - ref);
    }
    const allowed = (t: number): number => BIG_JUMP_DB + MAX_RISE_DB_PER_S * t;

    it('gain +6 dB and trim +6 dB in one message: ≤ 6 dB + 6 dB/s (was +12 dB at once)', () => {
      const [a, b] = rise(() => {}, (p) => {
        p.channels[0].gainDb = -14;
        p.channels[0].harmonicTrims = only(1, 6, 0);
      }, [0.04, 1.5]);
      // window [40, 140] ms ends 140 ms after the change
      expect(a).toBeLessThanOrEqual(allowed(0.14) + 0.05);
      expect(a).toBeGreaterThan(BIG_JUMP_DB - 0.2);
      expect(b).toBeCloseTo(12, 1);
    });

    it('master +6 dB and trim +6 dB: the same', () => {
      const [a] = rise((p) => (p.masterDb = -6), (p) => {
        p.masterDb = 0;
        p.channels[0].harmonicTrims = only(1, 6, 0);
      }, [0.04]);
      expect(a).toBeLessThanOrEqual(allowed(0.14) + 0.05);
    });

    it('gain −40 → −20 dB with trim −30 → +6 dB: the harmonic rises at 6 dB/s, not 12', () => {
      const t = [0.4, 0.9, 1.4, 1.9];
      const got = rise((p) => {
        p.channels[0].gainDb = -40;
        p.channels[0].harmonicTrims = only(1, -30, 0);
      }, (p) => {
        p.channels[0].gainDb = -20;
        p.channels[0].harmonicTrims = only(1, 6, 0);
      }, t);
      got.forEach((g, i) => expect(g, `${t[i]} s`).toBeLessThanOrEqual(allowed(t[i] + 0.1) + 0.05));
      expect(got[3]).toBeGreaterThan(allowed(1.9) - 1);
    });

    it('a trim rise holds the gain back on the same channel only, and other harmonics keep their own budget', () => {
      // h1 takes +6 dB at once; the gain must then wait (it also raises h1), harmonic 2 may still jump.
      const r = new Rig({ softStartMs: 10 });
      r.params((p) => {
        tonal(100, 4)(p);
        p.masterDb = 0;
        p.ceilingDb = -1;
        p.channels[0].gainDb = -20;
        p.channels[1].gainDb = -20;
      });
      r.send({ type: 'arm', armed: true });
      const x = new Float32Array(3 * fs);
      for (let i = 0; i < x.length; i++) x[i] = 0.01 * (Math.sin((2 * Math.PI * 100 * i) / fs) + Math.sin((2 * Math.PI * 200 * i) / fs));
      const before = r.run(fs, x);
      const lv = (y: Float32Array, f: number, from: number) => db(dft(y, f, from, ms(100)));
      r.params((p) => (p.channels[0].harmonicTrims = only(1, 6, 0)));
      r.params((p) => (p.channels[0].harmonicTrims[1] = { gainDb: 6, phaseDeg: 0 }));
      r.params((p) => {
        p.channels[0].gainDb = -14;
        p.channels[1].gainDb = -14;
      });
      const y = r.run(fs, x);
      // channel L: h1 and h2 +6 dB at once, the gain at 6 dB/s; channel R (no trims): the gain at once
      expect(lv(y.L, 100, ms(40)) - lv(before.L, 100, fs - ms(100))).toBeLessThan(allowed(0.14) + 0.05);
      expect(lv(y.L, 200, ms(40)) - lv(before.L, 200, fs - ms(100))).toBeLessThan(allowed(0.14) + 0.05);
      expect(lv(y.L, 200, ms(40)) - lv(before.L, 200, fs - ms(100))).toBeGreaterThan(BIG_JUMP_DB - 0.2);
      expect(lv(y.R, 100, ms(150)) - lv(before.R, 100, fs - ms(100))).toBeGreaterThan(BIG_JUMP_DB - 0.2);
    });
  });

  it('a dragged trim control (a message every 10–16 ms) moves in whole glides: nothing above 2 kHz (≤ −100 dB)', () => {
    // The live bank gets one new trim per COEFF_RAMP_MS at most; a newer value waits for the glide's end.
    for (const field of ['phaseDeg', 'gainDb'] as const) {
      for (const every of [10, 16]) {
        const r = new Rig({ softStartMs: 50 });
        r.params((p) => {
          tonal(100, 4)(p);
          p.masterDb = 0;
          p.filters.lpEnabled = false;
          p.channels[0].harmonicTrims = only(1, field === 'gainDb' ? -6 : 0, 0);
        });
        r.send({ type: 'arm', armed: true });
        const amp = 0.2;
        const x = sine(4 * fs, 100, amp);
        const pre = r.run(2 * fs, x).L;
        const parts: Float32Array[] = [pre.subarray(pre.length - ms(100))];
        for (let k = 1; k <= 30; k++) {
          r.params((p) => {
            p.channels[0].harmonicTrims = field === 'gainDb' ? only(1, -6 + (12 * k) / 30, 0) : only(1, 0, (180 * k) / 30);
          });
          parts.push(r.run(ms(every), x).L);
        }
        parts.push(r.run(ms(200), x).L);
        const y = new Float32Array(parts.reduce((a, q) => a + q.length, 0));
        let o = 0;
        for (const q of parts) {
          y.set(q, o);
          o += q.length;
        }
        expect(clickDb(y, ms(95), y.length - ms(50), amp), `${field} every ${every} ms`).toBeLessThan(-100);
        // a fast trim rise at constant commanded gain is no howl (the detector restarts its level tracks)
        expect(r.of('howl'), `${field} every ${every} ms`).toHaveLength(0);
        // and it ends on the last value asked for
        expect(r.e.params.channels[0].harmonicTrims[0][field]).toBeCloseTo(field === 'gainDb' ? 6 : 180, 9);
      }
    }
  });

  it('only a trim rise faster than the 6 dB/s staircase restarts the howl level tracks (holdLevels)', () => {
    const hold = vi.spyOn(HowlDetector.prototype, 'holdLevels');
    try {
      const r = new Rig({ softStartMs: 10 });
      r.params((p) => {
        tonal(100, 4)(p);
        p.channels[0].harmonicTrims = only(1, -30, 0);
      });
      r.send({ type: 'arm', armed: true });
      const x = sine(8 * fs, 100, 0.1);
      r.run(fs, x);
      hold.mockClear();
      // −30 → 0 dB: 24 dB at 6 dB/s (no hold), then the last 6 dB from the budget (at most two fast hand-overs)
      r.params((p) => (p.channels[0].harmonicTrims = only(1, 0, 0)));
      r.run(ms(3900), x);
      expect(hold).not.toHaveBeenCalled();
      r.run(ms(1100), x);
      expect(hold.mock.calls.length).toBeLessThanOrEqual(2);
      // a drag: +0.5 dB every 20 ms from the budget, each hand-over held on channel L's detector only
      hold.mockClear();
      for (let k = 1; k <= 8; k++) {
        r.params((p) => (p.channels[0].harmonicTrims = only(1, 0.5 * k, 0)));
        r.run(ms(20), x);
      }
      expect(hold.mock.calls.length).toBeGreaterThanOrEqual(7);
      expect(r.of('howl')).toHaveLength(0);
    } finally {
      hold.mockRestore();
    }
  });

  it('howl detection sees a loop that the trims close (loop gain 1.5 in phase at f0) and mutes', () => {
    const delay = 300;
    const f0 = 100;
    const r = new Rig();
    // the loop phase at f0 is set to 0 by the trim of h1: x = amb + fb·outL[n − delay]
    const p0 = r.e.params;
    tonal(f0, 4)(p0);
    p0.masterDb = 0;
    p0.ceilingDb = -12;
    p0.channels[1].enabled = false;
    const open = cMul(channelResponse(p0, fs, f0, 0), cDelay(f0, fs, delay));
    const fb = 1.5 / cAbs(open);
    p0.channels[0].harmonicTrims = only(1, 0, 0);
    p0.channels[0].harmonicTrims[0] = trimForTarget({ re: cAbs(open), im: 0 }, open);
    r.send({ type: 'params', params: p0 });
    const loop = cMul(channelResponse(r.e.params, fs, f0, 0), cDelay(f0, fs, delay));
    expect(Math.abs(cArg(loop))).toBeLessThan(1e-6);
    r.send({ type: 'arm', armed: true });
    const n = Math.round(4 * fs);
    const onset = 2 * fs;
    const amb = noise(n, 0.01, 12);
    const outL = new Float32Array(n);
    const outR = new Float32Array(n);
    const inp = new Float32Array(n);
    r.runLoop(n, (i, oL) => amb[i] + (i >= onset ? fb * oL[i - delay] : 0), outL, outR, inp);
    const howl = r.of('howl');
    expect(howl.length).toBe(1);
    expect(howl[0].at - onset).toBeLessThan(ms(1500));
    expect(r.of('autoMute').length).toBe(1);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(outL[i]));
    expect(peak).toBeLessThanOrEqual(Math.pow(10, -12 / 20));
  });
});

describe('AntiPhaseEngine: MeterSnapshot.masterDb (the master applied now)', () => {
  it('follows the rate-limited master ramp, jumps with a stimulus, MASTER_MIN_DB when silent', () => {
    const r = new Rig({ softStartMs: 10, meterHz: 100 });
    r.params((p) => {
      unity(p);
      p.masterDb = -30;
    });
    // disarmed: applied at once
    const silence = new Float32Array(12 * fs);
    r.run(ms(50), silence);
    expect(r.lastMeters().masterDb).toBe(-30);
    r.send({ type: 'arm', armed: true });
    r.run(ms(50), silence);
    r.params((p) => (p.masterDb = 0));
    r.run(fs, silence);
    // 30 dB gap > 6 dB: 6 dB/s (meters every 10 ms: within one meter period)
    expect(r.lastMeters().masterDb).toBeGreaterThan(-24 - 0.07);
    expect(r.lastMeters().masterDb).toBeLessThan(-24 + 0.07);
    r.run(4 * fs, silence);
    expect(r.lastMeters().masterDb).toBeCloseTo(0, 6);
    r.params((p) => (p.masterDb = -30));
    r.run(ms(500), silence);
    expect(r.lastMeters().masterDb).toBeCloseTo(-30, 6);
    // a stimulus fixes the master at its target when it starts
    r.params((p) => (p.masterDb = -6));
    r.send({ type: 'measure', id: 1, stimulus: constant(ms(200), 0.01), channel: 0, postRollFrames: 0 });
    r.run(ms(30), silence);
    expect(r.lastMeters().measuring).toBe(true);
    expect(r.lastMeters().masterDb).toBe(-6);
    r.run(ms(300), silence);
    r.params((p) => (p.masterDb = -500));
    r.run(fs, silence);
    expect(r.lastMeters().masterDb).toBe(MASTER_MIN_DB);
  });
});
