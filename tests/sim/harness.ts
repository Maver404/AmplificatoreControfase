/**
 * Closed-loop rig: the real AntiPhaseEngine around the simulated plant, driven block by block, with the
 * MeasureFn / CaptureFn / AutotuneDriver the browser gets from the worklet — so the calibration and
 * autotune code runs unmodified against the simulator. Everything is synchronous underneath; the async
 * functions resolve once the simulated audio has run. SimEngine puts a whole AppController on the rig.
 */
import type { EngineLike } from '../../src/app/controller.ts';
import type { AudioEngineEvent, AudioStats, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import { harmonicLevelsDb } from '../../src/audio/autotune.ts';
import type { AutotuneDriver } from '../../src/audio/autotune.ts';
import type { CaptureFn, MeasureFn } from '../../src/audio/calibration.ts';
import type { WorkletInMsg, WorkletOutMsg } from '../../src/audio/protocol.ts';
import { AntiPhaseEngine } from '../../src/dsp/antiphase-engine.ts';
import type { EngineOptions } from '../../src/dsp/antiphase-engine.ts';
import { nextPow2, sharedFFT } from '../../src/dsp/fft.ts';
import { MUTE_RAMP_MS, SOFT_START_MS, clampParams } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { Plant } from './plant.ts';
import type { PlantConfig } from './plant.ts';

export type Signal = 'ref' | 'err' | 'outL' | 'outR';

export interface RigOptions {
  block?: number;
  engine?: EngineOptions;
  /** Whether the engine receives the error mic on input 2 (default true; the plant always computes it). */
  errorMic?: boolean;
  /** Sees every message the engine posts, meters and taps included, before the rig handles it. */
  onMessage?: (msg: WorkletOutMsg) => void;
}

/** Level (dB, arbitrary but fixed reference) of x in [lo, hi] Hz: Welch, Hann, 4096-point segments. */
export function bandLevelDb(x: Float32Array, fs: number, lo: number, hi: number): number {
  const n = Math.min(4096, nextPow2(x.length) / 2);
  const fft = sharedFFT(n);
  const w = Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
  const sw2 = w.reduce((a, b) => a + b * b, 0);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const k0 = Math.ceil((lo * n) / fs);
  const k1 = Math.floor((hi * n) / fs);
  let acc = 0;
  let count = 0;
  for (let s = 0; s + n <= x.length; s += n / 2) {
    for (let i = 0; i < n; i++) {
      re[i] = x[s + i] * w[i];
      im[i] = 0;
    }
    fft.forward(re, im);
    for (let k = k0; k <= k1; k++) acc += re[k] * re[k] + im[k] * im[k];
    count++;
  }
  return 10 * Math.log10(Math.max(1e-30, acc / (count * sw2)));
}

/** Amplitude (dB) of the component of x at exactly `f` Hz (Hann-windowed synchronous detection). */
export function toneLevelDb(x: Float32Array, fs: number, f: number): number {
  let re = 0;
  let im = 0;
  let sw = 0;
  const n = x.length;
  for (let i = 0; i < n; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    const a = (2 * Math.PI * f * i) / fs;
    re += w * x[i] * Math.cos(a);
    im -= w * x[i] * Math.sin(a);
    sw += w;
  }
  return 20 * Math.log10(Math.max(1e-30, (2 * Math.hypot(re, im)) / sw));
}

export class SimRig {
  readonly fs: number;
  readonly block: number;
  readonly plant: Plant;
  readonly engine: AntiPhaseEngine;
  /** Every message the engine posted except meters, taps and progress. */
  readonly events: WorkletOutMsg[] = [];
  /** Largest |output| sample seen so far (both channels). */
  maxAbsOut = 0;
  private readonly errorMic: boolean;
  private readonly onMessage: ((msg: WorkletOutMsg) => void) | null;
  private readonly listeners = new Set<(msg: WorkletOutMsg) => void>();
  private readonly outL: Float32Array;
  private readonly outR: Float32Array;
  private measureId = 0;
  private params: EngineParams;
  private readonly softStartS: number;

  constructor(cfg: PlantConfig, opts: RigOptions = {}) {
    this.fs = cfg.fs;
    this.block = opts.block ?? 128;
    this.plant = new Plant(cfg, this.block);
    this.engine = new AntiPhaseEngine(cfg.fs, opts.engine);
    this.errorMic = opts.errorMic ?? true;
    this.onMessage = opts.onMessage ?? null;
    this.outL = new Float32Array(this.block);
    this.outR = new Float32Array(this.block);
    this.params = clampParams(undefined);
    this.softStartS = (opts.engine?.softStartMs ?? SOFT_START_MS) / 1000;
  }

  send(msg: WorkletInMsg): void {
    if (msg.type === 'params') this.params = clampParams(msg.params);
    this.engine.handleMessage(msg);
  }

  get currentParams(): EngineParams {
    return clampParams(this.params);
  }

  /** Simulated time so far, s. */
  get timeS(): number {
    return this.plant.elapsedSamples / this.fs;
  }

  /** Like RigOptions.onMessage, added later (SimEngine); returns the unsubscribe function. */
  listen(cb: (msg: WorkletOutMsg) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** One block of closed-loop audio. */
  private step(rec?: Map<Signal, Float32Array>, offset = 0): void {
    const { ref, err } = this.plant.next();
    if (rec !== undefined) {
      const r = rec.get('ref');
      if (r !== undefined) r.set(ref.subarray(0, Math.min(this.block, r.length - offset)), offset);
      const e = rec.get('err');
      if (e !== undefined) e.set(err.subarray(0, Math.min(this.block, e.length - offset)), offset);
    }
    this.engine.process(ref, this.errorMic ? err : null, this.outL, this.outR);
    this.plant.push(this.outL, this.outR);
    for (let i = 0; i < this.block; i++) {
      const a = Math.max(Math.abs(this.outL[i]), Math.abs(this.outR[i]));
      if (a > this.maxAbsOut) this.maxAbsOut = a;
    }
    if (rec !== undefined) {
      const l = rec.get('outL');
      if (l !== undefined) l.set(this.outL.subarray(0, Math.min(this.block, l.length - offset)), offset);
      const r = rec.get('outR');
      if (r !== undefined) r.set(this.outR.subarray(0, Math.min(this.block, r.length - offset)), offset);
    }
    this.engine.drain((msg) => {
      if (this.onMessage !== null) this.onMessage(msg);
      for (const cb of [...this.listeners]) cb(msg);
      if (msg.type === 'meters' || msg.type === 'measureProgress') return;
      if (msg.type === 'tap') {
        this.engine.handleMessage({ type: 'recycle', buffer: msg.block });
        return;
      }
      this.events.push(msg);
    });
  }

  /** Runs `seconds` of audio (whole blocks), recording the requested signals. */
  run(seconds: number, record: Signal[] = []): Partial<Record<Signal, Float32Array>> {
    const frames = Math.max(0, Math.round(seconds * this.fs));
    const blocks = Math.ceil(frames / this.block);
    const rec = new Map<Signal, Float32Array>();
    for (const s of record) rec.set(s, new Float32Array(frames));
    for (let b = 0; b < blocks; b++) this.step(rec.size > 0 ? rec : undefined, b * this.block);
    return Object.fromEntries(rec) as Partial<Record<Signal, Float32Array>>;
  }

  /** Time for a change at the outputs to reach the mics, plus the mute ramp and a margin. */
  get settleS(): number {
    return this.plant.maxOutputLagS + MUTE_RAMP_MS / 1000 + 0.02;
  }

  /** Disarms, applies `p` at once (silent), re-arms and waits for the soft-start and the paths to settle. */
  start(p: EngineParams, extraS = 0.3): void {
    this.send({ type: 'arm', armed: false });
    this.send({ type: 'params', params: p });
    this.send({ type: 'arm', armed: true });
    this.run(this.softStartS + this.settleS + extraS);
  }

  /** Error-mic level in [lo, hi] with the current settings. */
  errorLevelDb(band: [number, number], seconds: number): number {
    const { err } = this.run(seconds, ['err']);
    return bandLevelDb(err!, this.fs, band[0], band[1]);
  }

  /** Error-mic level in [lo, hi] with the anti-noise bypassed. */
  bypassLevelDb(band: [number, number], seconds: number): number {
    const err = this.bypassed(() => this.run(seconds, ['err']).err!);
    return bandLevelDb(err, this.fs, band[0], band[1]);
  }

  /** Runs `fn` with the anti-noise bypassed (and settled), then un-bypasses and settles again. */
  bypassed<T>(fn: () => T): T {
    this.send({ type: 'bypass', bypass: true });
    this.run(this.settleS);
    const out = fn();
    this.send({ type: 'bypass', bypass: false });
    this.run(this.settleS);
    return out;
  }

  get autoMuted(): boolean {
    return this.events.some((e) => e.type === 'autoMute');
  }

  /** MeasureFn backed by the engine's 'measure' message. */
  readonly measure: MeasureFn = async (stimulus, channel, postRollFrames) => {
    const id = ++this.measureId;
    this.send({ type: 'measure', id, stimulus, channel, postRollFrames });
    const limit = Math.ceil((stimulus.length + postRollFrames + this.fs) / this.block);
    for (let b = 0; b < limit; b++) {
      this.step();
      const i = this.events.findIndex((e) => e.type === 'measureResult' && e.id === id);
      if (i >= 0) {
        const r = this.events[i];
        this.events.splice(i, 1);
        if (r.type !== 'measureResult') break;
        return { ref: r.ref, err: r.err, masterLin: r.masterLin, aborted: r.aborted };
      }
    }
    throw new Error('simulated measurement did not finish');
  };

  /** CaptureFn: ambient recording with the anti-noise bypassed. */
  readonly capture: CaptureFn = async (seconds) => {
    const r = this.bypassed(() => this.run(seconds, ['ref', 'err']));
    return { ref: r.ref!, err: this.errorMic ? r.err! : null };
  };

  /** AutotuneDriver on the simulator (band levels, and per-harmonic levels for the tonal search), logging every measurement and apply. */
  autotuneDriver(
    band: [number, number],
    signal?: AbortSignal,
  ): AutotuneDriver & {
    log: {
      errors: number[];
      bypass: number[];
      applied: { p: EngineParams; jump: boolean }[];
      harmonics: { bypass: boolean; levels: number[]; t: number }[];
    };
  } {
    const log = {
      errors: [] as number[],
      bypass: [] as number[],
      applied: [] as { p: EngineParams; jump: boolean }[],
      /** Every per-harmonic window, in order, with its centre (simulated time, s). */
      harmonics: [] as { bypass: boolean; levels: number[]; t: number }[],
    };
    return {
      log,
      signal,
      apply: (p, jump) => {
        log.applied.push({ p: clampParams(p), jump });
        this.send({ type: 'params', params: p });
      },
      settle: async (ms) => {
        this.run(ms / 1000);
      },
      measureErrorDb: async (ms) => {
        const db = this.errorLevelDb(band, ms / 1000);
        log.errors.push(db);
        return db;
      },
      measureBypassDb: async (ms) => {
        const db = this.bypassLevelDb(band, ms / 1000);
        log.bypass.push(db);
        return db;
      },
      // Tonal search: synchronous detection of each harmonic; the logs get the power sum (the tonal level).
      measureHarmonicsDb: async (ms, freqs, bypass, tolHz) => {
        let t = 0;
        const rec = (): Float32Array => {
          t = this.timeS + ms / 2000;
          return this.run(ms / 1000, ['err']).err!;
        };
        const levels = harmonicLevelsDb(bypass ? this.bypassed(rec) : rec(), this.fs, freqs, tolHz);
        const total = 10 * Math.log10(levels.reduce((a, db) => a + Math.pow(10, db / 10), 0));
        (bypass ? log.bypass : log.errors).push(total);
        log.harmonics.push({ bypass, levels: [...levels], t });
        return levels;
      },
    };
  }
}

/**
 * The rig as an AppController's audio engine (EngineLike), so the controller runs unmodified on the simulator:
 * messages go to the engine, everything the engine posts (taps and meters included) comes back as engine
 * events. Nothing runs by itself: the test advances the rig and the (fake) timers together.
 */
export class SimEngine implements EngineLike {
  state: 'idle' | 'mic' | 'error' = 'idle';
  readonly error: string | null = null;
  readonly sinkSupported = true;
  stats: AudioStats;
  trackInfo: TrackInfo | null = null;
  /** Every message the controller sent, with the simulated time (s). */
  readonly sent: { t: number; msg: WorkletInMsg }[] = [];
  private readonly listeners = new Set<(e: AudioEngineEvent) => void>();
  private readonly settings: MediaTrackSettings;

  constructor(readonly rig: SimRig) {
    this.settings = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      sampleRate: rig.fs,
      channelCount: 2,
      deviceId: 'sim-in',
    };
    this.stats = { ...this.statsOpen(), sampleRate: 0, trackSettings: null };
    rig.listen((msg) => {
      for (const cb of [...this.listeners]) cb(msg);
    });
  }

  private statsOpen(): AudioStats {
    return {
      sampleRate: this.rig.fs,
      renderQuantum: this.rig.block,
      baseLatencyMs: 2.7,
      outputLatencyMs: 12,
      trackLatencyMs: 5,
      underruns: 0,
      trackSettings: this.settings,
      sinkId: '',
    };
  }

  async openMic(): Promise<void> {
    this.state = 'mic';
    this.stats = this.statsOpen();
    this.trackInfo = { deviceId: 'sim-in', groupId: 'sim', label: 'Interfaccia simulata', settings: this.settings, channelCount: 2 };
  }

  async setOutput(): Promise<void> {}

  async ensureContextMatchesTrack(): Promise<boolean> {
    return false;
  }

  async setLatencyMode(): Promise<void> {}

  send(msg: WorkletInMsg): void {
    this.sent.push({ t: this.rig.timeS, msg });
    this.rig.send(msg);
  }

  /** Runs the simulated audio itself (synchronously): not for use while the test drives the timers. */
  measure(stimulus: Float32Array, channel: 0 | 1, postRollFrames: number): Promise<MeasureResult> {
    return this.rig.measure(stimulus, channel, postRollFrames);
  }

  panic(): void {
    this.send({ type: 'panic' });
  }

  async resume(): Promise<void> {}

  on(cb: (e: AudioEngineEvent) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }
}
