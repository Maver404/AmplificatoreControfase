/**
 * Reference scenarios for the "trims at 0 change nothing" guarantee: the engine with every harmonic trim at 0
 * must reproduce, within 1e-12, the outputs of the engine before harmonic trims existed (git 731a3f1). The
 * stored values (antiphase-engine.golden-data.ts) were produced by running these scenarios on that engine;
 * antiphase-engine-golden.test.ts runs them on the current one. The runner only uses the message protocol, so
 * the same code drives both versions.
 */
import { TAP_BLOCK_FRAMES } from '../../src/audio/protocol.ts';
import type { WorkletInMsg, WorkletOutMsg } from '../../src/audio/protocol.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

export interface GoldenEngine {
  handleMessage(msg: WorkletInMsg): void;
  process(in0: Float32Array | null, in1: Float32Array | null, outL: Float32Array, outR: Float32Array): void;
  drain(post: (msg: WorkletOutMsg, transfer?: ArrayBuffer[]) => void): void;
  readonly params: EngineParams;
}

export type EngineFactory = (fs: number, opts: { softStartMs: number }) => GoldenEngine;

/** Every DECIMATE-th output sample is stored as is; the checksums cover all of them. */
export const DECIMATE = 397;

export interface GoldenSummary {
  /** outL / outR samples 0, DECIMATE, 2·DECIMATE, … */
  L: number[];
  R: number[];
  /** [Σy, Σy², Σ w_i·y] over the whole run for outL, outR, the 'cond' tap and the last measured reference. */
  sums: { L: number[]; R: number[]; cond: number[]; measRef: number[] };
}

/** Deterministic ±1 weights for the third checksum. */
function checksums(y: ArrayLike<number>): number[] {
  const rnd = mulberry32(9001);
  let s = 0;
  let s2 = 0;
  let sw = 0;
  for (let i = 0; i < y.length; i++) {
    const v = y[i];
    s += v;
    s2 += v * v;
    sw += (rnd() < 0.5 ? -1 : 1) * v;
  }
  return [s, s2, sw];
}

class GoldenRig {
  readonly e: GoldenEngine;
  readonly fs: number;
  private readonly x0: Float32Array;
  private readonly x1: Float32Array;
  readonly L: number[] = [];
  readonly R: number[] = [];
  readonly cond: number[] = [];
  measRef: Float32Array = new Float32Array(0);
  pos = 0;

  constructor(make: EngineFactory, fs: number, seconds: number, seed: number, tones: number[]) {
    this.fs = fs;
    this.e = make(fs, { softStartMs: 300 });
    const n = Math.ceil(seconds * fs);
    const rnd = mulberry32(seed);
    this.x0 = new Float32Array(n);
    this.x1 = new Float32Array(n);
    const ph = tones.map(() => 2 * Math.PI * rnd());
    for (let i = 0; i < n; i++) {
      let v = 0.04 * (2 * rnd() - 1);
      for (let k = 0; k < tones.length; k++) v += (0.12 / (k + 1)) * Math.sin((2 * Math.PI * tones[k] * i) / fs + ph[k]);
      this.x0[i] = v;
      this.x1[i] = 0.02 * (2 * rnd() - 1);
    }
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

  private drain(): void {
    const back: Float32Array[] = [];
    this.e.drain((m) => {
      if (m.type === 'tap') {
        const F = TAP_BLOCK_FRAMES;
        for (let j = 0; j < F; j++) this.cond.push(m.block[F + j]);
        back.push(m.block);
      } else if (m.type === 'measureResult' && !m.aborted) this.measRef = m.ref;
    });
    for (const b of back) this.e.handleMessage({ type: 'recycle', buffer: b });
  }

  run(seconds: number, block = 128): void {
    const n = Math.round(seconds * this.fs);
    const L = new Float32Array(block);
    const R = new Float32Array(block);
    for (let off = 0; off < n; off += block) {
      const m = Math.min(block, n - off);
      const a = this.pos;
      this.e.process(this.x0.subarray(a, a + m), this.x1.subarray(a, a + m), L.subarray(0, m), R.subarray(0, m));
      for (let i = 0; i < m; i++) {
        this.L.push(L[i]);
        this.R.push(R[i]);
      }
      this.pos += m;
      this.drain();
    }
  }

  summary(): GoldenSummary {
    const dec = (y: number[]) => y.filter((_, i) => i % DECIMATE === 0);
    return {
      L: dec(this.L),
      R: dec(this.R),
      sums: { L: checksums(this.L), R: checksums(this.R), cond: checksums(this.cond), measRef: checksums(this.measRef) },
    };
  }
}

const channels = (p: EngineParams): void => {
  p.masterDb = -6;
  p.ceilingDb = -3;
  p.channels[0].gainDb = 0;
  p.channels[0].delayMs = 3.21;
  p.channels[0].phaseDeltaDeg = 17;
  p.channels[1].gainDb = -3;
  p.channels[1].delayMs = 0.4;
  p.channels[1].polarityInvert = false;
  p.channels[1].phaseDeltaDeg = -40;
  p.errorChannel = 1;
};

export const GOLDEN_SCENARIOS: { name: string; run: (make: EngineFactory) => GoldenSummary }[] = [
  {
    name: 'delay mode, Δ live change, 48 kHz',
    run: (make) => {
      const r = new GoldenRig(make, 48000, 1.2, 11, [137, 274]);
      r.params(channels);
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.run(0.6);
      r.params((p) => {
        p.channels[0].phaseDeltaDeg = 40;
        p.filters.lpFreq = 600;
      });
      r.run(0.6, 64);
      return r.summary();
    },
  },
  {
    name: 'tonal mode, 4 harmonics at 137 Hz, f0 and Q glide live, 48 kHz',
    run: (make) => {
      const r = new GoldenRig(make, 48000, 1.5, 12, [137, 274, 411, 548]);
      r.params((p) => {
        channels(p);
        p.phaseMode = 'tonal';
        p.tonal = { f0: 137, harmonics: 4, q: 30 };
      });
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.run(0.7);
      r.params((p) => {
        p.tonal.f0 = 138.5;
        p.tonal.q = 20;
        p.channels[1].phaseDeltaDeg = 10;
      });
      r.run(0.8, 256);
      return r.summary();
    },
  },
  {
    name: 'tonal mode, 8 harmonics at 60 Hz, then 5 harmonics (structural), 96 kHz',
    run: (make) => {
      const r = new GoldenRig(make, 96000, 1.4, 13, [60, 120, 180, 240, 300]);
      r.params((p) => {
        channels(p);
        p.phaseMode = 'tonal';
        p.tonal = { f0: 60, harmonics: 8, q: 12 };
        p.filters.lpFreq = 900;
      });
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.run(0.6);
      r.params((p) => (p.tonal.harmonics = 5));
      r.run(0.8);
      return r.summary();
    },
  },
  {
    name: 'tonal mode, 8 harmonics at 1500 Hz above 0.45·fs (6 used), 22.05 kHz, with a measurement',
    run: (make) => {
      const r = new GoldenRig(make, 22050, 1.6, 14, [1500, 3000, 4500]);
      r.params((p) => {
        channels(p);
        p.phaseMode = 'tonal';
        p.tonal = { f0: 1500, harmonics: 8, q: 20 };
        p.filters.lpFreq = 4000;
      });
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.run(0.5);
      const stim = new Float32Array(Math.round(0.2 * 22050));
      for (let i = 0; i < stim.length; i++) stim[i] = 0.1 * Math.sin(i * 0.31) * Math.sin((Math.PI * i) / stim.length);
      r.send({ type: 'measure', id: 1, stimulus: stim, channel: 1, postRollFrames: 2000 });
      r.run(1.1, 4096);
      return r.summary();
    },
  },
  {
    name: 'hilbert mode, master change, 44.1 kHz',
    run: (make) => {
      const r = new GoldenRig(make, 44100, 1.2, 15, [90, 333]);
      r.params((p) => {
        channels(p);
        p.phaseMode = 'hilbert';
      });
      r.send({ type: 'arm', armed: true });
      r.send({ type: 'tap', enabled: true });
      r.run(0.6, 1);
      r.params((p) => (p.masterDb = -9));
      r.run(0.6);
      return r.summary();
    },
  },
];
