import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { inProcessAnalyzer } from '../../src/audio/analysis-client.ts';
import type { F0Analyzer } from '../../src/audio/analysis-client.ts';
import type { AudioEngineEvent, AudioStats, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import { TAP_BLOCK_FRAMES, TAP_CHANNELS } from '../../src/audio/protocol.ts';
import type { WorkletInMsg } from '../../src/audio/protocol.ts';
import { AppController } from '../../src/app/controller.ts';
import type { DeviceProvider, EngineLike } from '../../src/app/controller.ts';
import { F0Tracker } from '../../src/dsp/f0-tracker.ts';
import type { F0Estimate } from '../../src/dsp/f0-tracker.ts';

/** The periodic f0 estimate of the 'cond' tap runs through an analyzer (the analysis worker in the browser). */

const FS = 48000;
const COND = TAP_CHANNELS.indexOf('cond');
const BLOCK_MS = (TAP_BLOCK_FRAMES / FS) * 1000;

const STATS: AudioStats = {
  sampleRate: FS,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: FS, channelCount: 2 },
  sinkId: '',
};

class Engine implements EngineLike {
  state: 'idle' | 'mic' | 'error' = 'idle';
  error: string | null = null;
  stats: AudioStats = { ...STATS, sampleRate: 0 };
  trackInfo: TrackInfo | null = null;
  sinkSupported = true;
  private listeners = new Set<(e: AudioEngineEvent) => void>();
  async openMic(): Promise<void> {
    this.state = 'mic';
    this.stats = STATS;
    this.trackInfo = { deviceId: 'usb-in', groupId: 'g', label: 'USB', settings: STATS.trackSettings!, channelCount: 2 };
  }
  async setOutput(): Promise<void> {}
  async ensureContextMatchesTrack(): Promise<boolean> {
    return false;
  }
  async setLatencyMode(): Promise<void> {}
  sent: WorkletInMsg[] = [];
  send(msg: WorkletInMsg): void {
    this.sent.push(msg);
  }
  async measure(): Promise<MeasureResult> {
    return { ref: new Float32Array(0), err: null, masterLin: 0, aborted: true };
  }
  panic(): void {}
  async resume(): Promise<void> {}
  on(cb: (e: AudioEngineEvent) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
  emit(e: AudioEngineEvent): void {
    for (const cb of [...this.listeners]) cb(e);
  }
}

const devices: DeviceProvider = {
  list: async () => ({
    inputs: [{ deviceId: 'usb-in', kind: 'audioinput', label: 'USB', groupId: 'g', pseudo: false }],
    outputs: [{ deviceId: 'usb-out', kind: 'audiooutput', label: 'USB', groupId: 'g', pseudo: false }],
  }),
  platform: () => 'mac',
  chromium: () => true,
};

/** A deferred fake analyzer: records every request, the test decides when (and what) it answers. */
class ManualAnalyzer implements F0Analyzer {
  calls: Array<{ x: Float32Array; fs: number; fmin: number; fmax: number; resolve: (r: F0Estimate | null) => void }> = [];
  estimateF0(x: Float32Array, fs: number, fmin: number, fmax: number): Promise<F0Estimate | null> {
    return new Promise((resolve) => this.calls.push({ x, fs, fmin, fmax, resolve }));
  }
}

let clockMs = 0;
let sampleIndex = 0;
const controllers: AppController[] = [];

beforeEach(() => {
  vi.useFakeTimers();
  clockMs = 1000;
  sampleIndex = 0;
});

afterEach(() => {
  for (const c of controllers.splice(0)) c.dispose();
  vi.useRealTimers();
  delete (globalThis as { Worker?: unknown }).Worker;
});

async function open(f0Analyzer?: F0Analyzer): Promise<{ c: AppController; e: Engine }> {
  const e = new Engine();
  const c = new AppController({ engine: e, devices, presetStorage: null, prefsStorage: null, now: () => clockMs, f0Analyzer });
  controllers.push(c);
  await c.openMic();
  return { c, e };
}

/** Feeds `seconds` of a 200 Hz tone (plus a little 3 kHz) on 'cond', returning every cond sample fed. */
function feed(e: Engine, seconds: number): Float32Array[] {
  const blocks: Float32Array[] = [];
  for (let k = 0; k < Math.ceil((seconds * 1000) / BLOCK_MS); k++) {
    const b = new Float32Array(TAP_CHANNELS.length * TAP_BLOCK_FRAMES);
    for (let i = 0; i < TAP_BLOCK_FRAMES; i++, sampleIndex++) {
      const t = sampleIndex / FS;
      b[COND * TAP_BLOCK_FRAMES + i] = 0.3 * Math.sin(2 * Math.PI * 200 * t) + 0.01 * Math.sin(2 * Math.PI * 3000 * t);
    }
    blocks.push(b.slice(COND * TAP_BLOCK_FRAMES, (COND + 1) * TAP_BLOCK_FRAMES));
    clockMs += BLOCK_MS;
    e.emit({ type: 'tap', block: b, frames: TAP_BLOCK_FRAMES });
  }
  return blocks;
}

const flush = async (): Promise<void> => {
  for (let i = 0; i < 5; i++) await Promise.resolve();
};

describe('AppController: periodic f0 estimate', () => {
  it('sends the last 2 s of cond to the analyzer every 2 s and applies the answer', async () => {
    const a = new ManualAnalyzer();
    const { c, e } = await open(a);
    const blocks = feed(e, 2.5);
    expect(a.calls).toHaveLength(0);
    vi.advanceTimersByTime(2000);
    expect(a.calls).toHaveLength(1);
    const req = a.calls[0]!;
    expect([req.fs, req.fmin, req.fmax]).toEqual([FS, 20, 2000]);
    // Exactly the window F0Tracker would hold.
    const tracker = new F0Tracker(FS, 2, 20, 2000);
    for (const b of blocks) tracker.push(b);
    expect(req.x.length).toBe(2 * FS);
    const all = new Float32Array(blocks.length * TAP_BLOCK_FRAMES);
    blocks.forEach((b, i) => all.set(b, i * TAP_BLOCK_FRAMES));
    expect(req.x).toEqual(all.subarray(all.length - 2 * FS));
    const expected = tracker.estimate();
    expect(await inProcessAnalyzer.estimateF0(req.x, FS, 20, 2000)).toEqual(expected);

    req.resolve(expected);
    await flush();
    expect(c.state.f0).toEqual(expected);
    expect(c.state.f0!.f0).toBeCloseTo(200, 2);
    expect(c.state.tonalSuggested).toBe(true);
  });

  it('skips a tick while an estimate is in flight and drops results after dispose', async () => {
    const a = new ManualAnalyzer();
    const { c, e } = await open(a);
    feed(e, 2.1);
    vi.advanceTimersByTime(2000);
    expect(a.calls).toHaveLength(1);
    feed(e, 2.1);
    vi.advanceTimersByTime(2000);
    expect(a.calls).toHaveLength(1);
    a.calls[0]!.resolve(null);
    await flush();
    feed(e, 0.5);
    vi.advanceTimersByTime(2000);
    expect(a.calls).toHaveLength(2);
    c.dispose();
    a.calls[1]!.resolve({ f0: 200, confidence: 1, dominanceDb: 40 });
    await flush();
    expect(c.state.f0).toBeNull();
  });

  it('a failed estimate reads as "no tone" and the next tick runs again', async () => {
    const failing: F0Analyzer = { estimateF0: vi.fn(async () => Promise.reject(new Error('Analisi non riuscita'))) };
    const { c, e } = await open(failing);
    feed(e, 2.1);
    vi.advanceTimersByTime(2000);
    await flush();
    expect(c.state.f0).toBeNull();
    feed(e, 0.5);
    vi.advanceTimersByTime(2000);
    expect(failing.estimateF0).toHaveBeenCalledTimes(2);
  });

  it('without a usable worker the estimate runs on the main thread with the same result', async () => {
    // A worker that dies on its first message: the client falls back in-process.
    let created = 0;
    (globalThis as { Worker?: unknown }).Worker = class {
      onmessage: unknown = null;
      onmessageerror: unknown = null;
      onerror: ((e: { preventDefault(): void }) => void) | null = null;
      constructor() {
        created++;
      }
      postMessage(): void {
        queueMicrotask(() => this.onerror?.({ preventDefault() {} }));
      }
      terminate(): void {}
    };
    const { c, e } = await open();
    const blocks = feed(e, 2.1);
    vi.advanceTimersByTime(2000);
    await flush();
    await vi.waitFor(() => expect(c.state.f0).not.toBeNull());
    expect(created).toBe(1);
    const tracker = new F0Tracker(FS, 2, 20, 2000);
    for (const b of blocks) tracker.push(b);
    expect(c.state.f0).toEqual(tracker.estimate());
  });
});

// ------------------------------------------------------------------------------- f0 following

/** Feeds `seconds` of a tone on 'cond' whose frequency is `freq(t)` (t = seconds since the first sample fed). */
function feedTone(e: Engine, seconds: number, freq: (t: number) => number, state: { phase: number }): void {
  for (let k = 0; k < Math.ceil((seconds * 1000) / BLOCK_MS); k++) {
    const b = new Float32Array(TAP_CHANNELS.length * TAP_BLOCK_FRAMES);
    for (let i = 0; i < TAP_BLOCK_FRAMES; i++, sampleIndex++) {
      state.phase += (2 * Math.PI * freq(sampleIndex / FS)) / FS;
      b[COND * TAP_BLOCK_FRAMES + i] = 0.3 * Math.sin(state.phase);
    }
    clockMs += BLOCK_MS;
    e.emit({ type: 'tap', block: b, frames: TAP_BLOCK_FRAMES });
  }
}

/** Tonal mode at f0 = 200 Hz; one 2 s tick answered with `est`. Returns the f0 in use and the params messages sent. */
async function followOnce(
  est: F0Estimate,
  setup: (c: AppController) => void = () => {},
): Promise<{ f0: number; params: number[] }> {
  const a = new ManualAnalyzer();
  const { c, e } = await open(a);
  c.setPhaseMode('tonal');
  c.setTonal({ f0: 200, harmonics: 4, q: 30 });
  setup(c);
  e.sent = [];
  feed(e, 2.1);
  vi.advanceTimersByTime(2000);
  expect(a.calls).toHaveLength(1);
  a.calls[0]!.resolve(est);
  await flush();
  expect(c.state.f0).toEqual(est);
  const params = e.sent.flatMap((m) => (m.type === 'params' ? [m.params.tonal.f0] : []));
  return { f0: c.state.params.tonal.f0, params };
}

describe('AppController: f0 following in tonal mode', () => {
  const good: F0Estimate = { f0: 202, confidence: 0.9, dominanceDb: 30 };

  it('follows an estimate within 3 % (confidence ≥ 0.5, dominance ≥ 15 dB) and sends the new params', async () => {
    const r = await followOnce(good);
    expect(r.f0).toBe(202);
    expect(r.params).toEqual([202]);
    // At the limits.
    expect((await followOnce({ f0: 200 * 1.03, confidence: 0.5, dominanceDb: 15 })).f0).toBeCloseTo(206, 9);
    expect((await followOnce({ f0: 200 / 1.03, confidence: 0.5, dominanceDb: 15 })).f0).toBeCloseTo(200 / 1.03, 9);
  });

  it('does not follow a bigger jump, a weak or buried tone, with following off, or in delay mode', async () => {
    const cases: Array<[string, F0Estimate, (c: AppController) => void]> = [
      ['ratio > 1.03', { ...good, f0: 207 }, () => {}],
      ['confidence < 0.5', { ...good, confidence: 0.49 }, () => {}],
      ['dominance < 15 dB', { ...good, dominanceDb: 14.9 }, () => {}],
      ['following off', good, (c) => c.setF0Follow(false)],
      ['delay mode', good, (c) => c.setPhaseMode('delay')],
    ];
    for (const [what, est, setup] of cases) {
      const r = await followOnce(est, setup);
      expect(r.f0, what).toBe(200);
      expect(r.params, what).toEqual([]);
    }
  });

  it('tracks a tone drifting 100 → 101 Hz over 10 s within 0.05 Hz after every 2 s tick', async () => {
    const { c, e } = await open(inProcessAnalyzer);
    c.setPhaseMode('tonal');
    c.setTonal({ f0: 100, harmonics: 4, q: 30 });
    const freq = (t: number): number => 100 + 0.1 * Math.min(10, t);
    const st = { phase: 0 };
    for (let tick = 1; tick <= 5; tick++) {
      const before = c.state.f0;
      feedTone(e, 2, freq, st);
      vi.advanceTimersByTime(2000);
      await vi.waitFor(() => expect(c.state.f0).not.toBe(before));
      await flush();
      // The estimate is over the last 2 s: the tone's mean frequency there is the one 1 s ago.
      const t = sampleIndex / FS;
      expect(Math.abs(c.state.params.tonal.f0 - freq(t - 1)), `tick ${tick}`).toBeLessThan(0.05);
      expect(Math.abs(c.state.params.tonal.f0 - freq(t)), `tick ${tick}`).toBeLessThan(0.15);
    }
    expect(c.state.params.tonal.f0).toBeGreaterThan(100.85);
  });
});
