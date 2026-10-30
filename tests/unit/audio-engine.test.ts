/**
 * AudioEngine (browser side) on a minimal fake Web Audio / getUserMedia stack: the lifecycle safety
 * net around the worklet (auto-mute, measurement abort, latency jumps, PANIC, sink switching).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AudioEngine } from '../../src/audio/audio-engine.ts';
import type { AudioEngineEvent } from '../../src/audio/audio-engine.ts';
import type { WorkletInMsg } from '../../src/audio/protocol.ts';

// ------------------------------------------------------------------------------- fakes

class FakeTrack extends EventTarget {
  readyState: 'live' | 'ended' = 'live';
  label = 'Scarlett 2i2 USB';
  constructor(public settings: MediaTrackSettings) {
    super();
  }
  getSettings(): MediaTrackSettings {
    return { ...this.settings };
  }
  stop(): void {
    this.readyState = 'ended';
  }
}

class FakeStream {
  constructor(readonly track: FakeTrack) {}
  getAudioTracks(): FakeTrack[] {
    return [this.track];
  }
  getTracks(): FakeTrack[] {
    return [this.track];
  }
}

class FakeNode {
  static all: FakeNode[] = [];
  readonly posted: WorkletInMsg[] = [];
  onprocessorerror: (() => void) | null = null;
  readonly port = {
    onmessage: null as ((e: { data: unknown }) => void) | null,
    postMessage: (m: WorkletInMsg): void => {
      this.posted.push(m);
    },
  };
  constructor(readonly ctx: FakeCtx) {
    FakeNode.all.push(this);
  }
  connect(): void {}
  disconnect(): void {}
  types(): string[] {
    return this.posted.map((m) => m.type);
  }
}

class FakeCtx extends EventTarget {
  static all: FakeCtx[] = [];
  /** setSinkId rejects these ids (device busy). */
  static failingSinks = new Set<string>();
  /** setSinkId as Chrome sequences it: 'suspended' while switching, 'running' after 'sinkchange'. */
  static chromeSinkSwitch = false;
  state: AudioContextState = 'running';
  sampleRate = 48000;
  baseLatency = 0.003;
  outputLatency = 0.012;
  sinkId = '';
  readonly playbackStats = { underrunEvents: 0 };
  readonly destination = { channelCount: 2, maxChannelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete' };
  readonly audioWorklet = { addModule: async (): Promise<void> => undefined };
  constructor(opts?: { sampleRate?: number; sinkId?: string }) {
    super();
    if (opts?.sampleRate) this.sampleRate = opts.sampleRate;
    if (typeof opts?.sinkId === 'string') this.sinkId = opts.sinkId;
    FakeCtx.all.push(this);
  }
  setState(s: AudioContextState): void {
    if (this.state === s) return;
    this.state = s;
    this.dispatchEvent(new Event('statechange'));
  }
  async resume(): Promise<void> {
    if (this.state !== 'closed') this.setState('running');
  }
  async suspend(): Promise<void> {
    if (this.state !== 'closed') this.setState('suspended');
  }
  async close(): Promise<void> {
    this.setState('closed');
  }
  async setSinkId(id: string): Promise<void> {
    if (FakeCtx.chromeSinkSwitch) this.setState('suspended');
    await Promise.resolve();
    if (FakeCtx.failingSinks.has(id)) throw new DOMException('Device busy', 'AbortError');
    this.sinkId = id;
    queueMicrotask(() => {
      this.dispatchEvent(new Event('sinkchange'));
      if (FakeCtx.chromeSinkSwitch) this.setState('running');
    });
  }
  createMediaStreamSource(): { connect(): void; disconnect(): void } {
    return { connect() {}, disconnect() {} };
  }
}

let track: FakeTrack;
let md: EventTarget & { getUserMedia: () => Promise<FakeStream> };

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance', 'Date'] });
  FakeCtx.all = [];
  FakeNode.all = [];
  FakeCtx.failingSinks = new Set();
  FakeCtx.chromeSinkSwitch = false;
  track = new FakeTrack({ deviceId: 'usb-in', groupId: 'g-usb', sampleRate: 48000, channelCount: 2 });
  md = Object.assign(new EventTarget(), { getUserMedia: async () => new FakeStream(track) });
  vi.stubGlobal('AudioContext', FakeCtx);
  vi.stubGlobal('AudioWorkletNode', FakeNode);
  vi.stubGlobal('navigator', { mediaDevices: md, userAgent: 'Mozilla/5.0 (Macintosh) Chrome/156' });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const flush = async (): Promise<void> => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

async function opened(): Promise<{ eng: AudioEngine; events: AudioEngineEvent[] }> {
  const eng = new AudioEngine();
  const events: AudioEngineEvent[] = [];
  eng.on((e) => events.push(e));
  const p = eng.openMic('usb-in', { wantErrorMic: false });
  await flush();
  await p;
  return { eng, events };
}

const lifecycles = (events: AudioEngineEvent[], cause?: string): AudioEngineEvent[] =>
  events.filter((e) => e.type === 'lifecycle' && (cause === undefined || e.cause === cause));

const lastNode = (): FakeNode => FakeNode.all[FakeNode.all.length - 1]!;
const lastCtx = (): FakeCtx => FakeCtx.all[FakeCtx.all.length - 1]!;

// ------------------------------------------------------------------------------- tests

describe('AudioEngine: auto-mute stops a running measurement', () => {
  it('devicechange, track mute and a context error abort the stimulus and settle measure() as aborted', async () => {
    for (const fire of [
      () => md.dispatchEvent(new Event('devicechange')),
      () => track.dispatchEvent(new Event('mute')),
      () => lastCtx().dispatchEvent(new Event('error')),
    ]) {
      const { eng } = await opened();
      eng.send({ type: 'arm', armed: true });
      eng.send({ type: 'mute', muted: true });
      const result = eng.measure(new Float32Array(48000), 0, 4800);
      const node = lastNode();
      node.posted.length = 0;
      fire();
      expect(node.types()).toEqual(['mute', 'abortMeasure']);
      expect((await result).aborted).toBe(true);
      await eng.close();
    }
  });

  it('without a measurement running, an auto-mute only mutes', async () => {
    const { eng } = await opened();
    const node = lastNode();
    node.posted.length = 0;
    md.dispatchEvent(new Event('devicechange'));
    expect(node.types()).toEqual(['mute']);
    await eng.close();
  });
});

describe('AudioEngine: output-latency jumps', () => {
  async function audibleForAWhile(): Promise<{ eng: AudioEngine; events: AudioEngineEvent[]; ctx: FakeCtx }> {
    const r = await opened();
    r.eng.send({ type: 'arm', armed: true });
    r.eng.send({ type: 'mute', muted: false });
    // The latency baseline settles 2 s after the context starts.
    await vi.advanceTimersByTimeAsync(3000);
    return { ...r, ctx: lastCtx() };
  }

  it('a jump while audible auto-mutes at once', async () => {
    const { eng, events, ctx } = await audibleForAWhile();
    ctx.outputLatency = 0.06;
    await vi.advanceTimersByTimeAsync(1000);
    expect(lifecycles(events, 'latency-jump')).toHaveLength(1);
    await eng.close();
  });

  it('a jump while muted is reported at the next un-mute (which is muted again)', async () => {
    const { eng, events, ctx } = await audibleForAWhile();
    eng.send({ type: 'mute', muted: true });
    ctx.outputLatency = 0.06;
    await vi.advanceTimersByTimeAsync(5000);
    expect(lifecycles(events, 'latency-jump')).toHaveLength(0);
    const node = lastNode();
    node.posted.length = 0;
    eng.send({ type: 'mute', muted: false });
    await flush();
    const jumps = lifecycles(events, 'latency-jump');
    expect(jumps).toHaveLength(1);
    expect(jumps[0]).toMatchObject({ reason: expect.stringMatching(/48,0 ms.*anti-rumore era spento/) });
    expect(node.posted).toEqual([
      { type: 'mute', muted: false },
      { type: 'mute', muted: true },
    ]);
    // Reported once: the next un-mute plays.
    eng.send({ type: 'mute', muted: false });
    await vi.advanceTimersByTimeAsync(2000);
    expect(lifecycles(events, 'latency-jump')).toHaveLength(1);
    await eng.close();
  });

  it('also while disarmed (reported when armed audible); a measurement after the jump re-baselines it', async () => {
    const { eng, events, ctx } = await audibleForAWhile();
    eng.send({ type: 'arm', armed: false });
    ctx.outputLatency = 0.03;
    await vi.advanceTimersByTimeAsync(2000);
    eng.send({ type: 'arm', armed: true });
    await flush();
    expect(lifecycles(events, 'latency-jump')).toHaveLength(1);

    // Muted again, another jump, then a completed measurement (a recalibration at the new latency).
    eng.send({ type: 'mute', muted: true });
    ctx.outputLatency = 0.05;
    await vi.advanceTimersByTimeAsync(2000);
    const result = eng.measure(new Float32Array(480), 0, 0);
    const node = lastNode();
    const m = node.posted.find((x) => x.type === 'measure');
    if (!m || m.type !== 'measure') throw new Error('no measure posted');
    node.port.onmessage?.({ data: { type: 'measureResult', id: m.id, ref: new Float32Array(480), err: null, masterLin: 1, aborted: false } });
    expect((await result).aborted).toBe(false);
    eng.send({ type: 'mute', muted: false });
    await vi.advanceTimersByTimeAsync(2000);
    expect(lifecycles(events, 'latency-jump')).toHaveLength(1);
    await eng.close();
  });
});

describe('AudioEngine: PANIC', () => {
  it('a PANIC while the context is being re-created keeps the new context suspended until resume()', async () => {
    const { eng, events } = await opened();
    const pending = eng.setLatencyMode('ultra');
    eng.panic();
    await pending;
    await vi.advanceTimersByTimeAsync(200);
    expect(FakeCtx.all).toHaveLength(2);
    const ctx = lastCtx();
    expect(ctx.state).toBe('suspended');
    // Our own suspension is not a system event.
    expect(lifecycles(events, 'context-state')).toHaveLength(0);
    await eng.resume();
    expect(ctx.state).toBe('running');
    await eng.close();
  });

  it('PANIC suspends the running context after the ramp', async () => {
    const { eng } = await opened();
    eng.panic();
    await vi.advanceTimersByTimeAsync(100);
    expect(lastCtx().state).toBe('suspended');
    await eng.resume();
    expect(lastCtx().state).toBe('running');
    await eng.close();
  });
});

describe('AudioEngine: output device', () => {
  it('a rejected setSinkId keeps the working sink requested (no needless re-creation)', async () => {
    const { eng } = await opened();
    await eng.setOutput('usb-out');
    expect(lastCtx().sinkId).toBe('usb-out');
    FakeCtx.failingSinks.add('busy-out');
    await expect(eng.setOutput('busy-out')).rejects.toThrow();
    expect(await eng.ensureContextMatchesTrack()).toBe(false);
    expect(FakeCtx.all).toHaveLength(1);
    await eng.close();
  });

  it("Chrome's suspension during our own setSinkId is not reported as a system suspension", async () => {
    FakeCtx.chromeSinkSwitch = true;
    const { eng, events } = await opened();
    await eng.setOutput('usb-out');
    await flush();
    expect(lastCtx().state).toBe('running');
    expect(lifecycles(events)).toEqual([]);
    // A real system suspension afterwards still is.
    lastCtx().setState('suspended');
    expect(lifecycles(events, 'context-state')).toHaveLength(1);
    await eng.close();
  });
});

describe('AudioEngine: feedback FIRs and the sample rate', () => {
  const fir: WorkletInMsg = { type: 'feedbackFir', channel: 0, bulkDelay: 140, taps: new Float64Array(256).fill(0.01) };
  const firsPosted = (n: FakeNode): WorkletInMsg[] => n.posted.filter((m) => m.type === 'feedbackFir' && m.taps !== null);

  it('a context re-created at another rate does not get the old FIRs replayed', async () => {
    const { eng } = await opened();
    eng.send(fir);
    track.settings = { ...track.settings, sampleRate: 44100 };
    expect(await eng.ensureContextMatchesTrack()).toBe(true);
    const node = lastNode();
    expect(node.ctx.sampleRate).toBe(44100);
    expect(firsPosted(node)).toEqual([]);
    await eng.close();
  });

  it('a processor re-created at the same rate gets them back', async () => {
    const { eng } = await opened();
    eng.send(fir);
    const first = lastNode();
    first.onprocessorerror?.();
    const node = lastNode();
    expect(node).not.toBe(first);
    expect(firsPosted(node)).toHaveLength(1);
    await eng.close();
  });
});
