import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AudioEngineEvent, AudioStats, LatencyMode, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import type { DeviceOption } from '../../src/audio/devices.ts';
import { TAP_BLOCK_FRAMES, TAP_CHANNELS } from '../../src/audio/protocol.ts';
import type { MeterSnapshot, WorkletInMsg } from '../../src/audio/protocol.ts';
import {
  AppController,
  PHASE_SETTLE_MS,
  describeDevice,
  firstProblem,
  safeApplySequence,
} from '../../src/app/controller.ts';
import type { DeviceProvider, EngineLike, MicPermission } from '../../src/app/controller.ts';
import { ISSUE } from '../../src/app/validation.ts';
import { DEFAULT_PARAMS, HEADPHONES_CEILING_DB, clampParams } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';

// ------------------------------------------------------------------------------- fakes

class MemoryStorage implements Storage {
  private m = new Map<string, string>();
  get length(): number {
    return this.m.size;
  }
  clear(): void {
    this.m.clear();
  }
  getItem(k: string): string | null {
    return this.m.get(k) ?? null;
  }
  key(i: number): string | null {
    return [...this.m.keys()][i] ?? null;
  }
  removeItem(k: string): void {
    this.m.delete(k);
  }
  setItem(k: string, v: string): void {
    this.m.set(k, String(v));
  }
}

const dev = (deviceId: string, kind: DeviceOption['kind'], label: string, groupId: string): DeviceOption => ({
  deviceId,
  kind,
  label,
  groupId,
  pseudo: deviceId === 'default' || deviceId === 'communications',
});

const INPUTS = [dev('usb-in', 'audioinput', 'Scarlett 2i2 USB', 'g-usb'), dev('mac-in', 'audioinput', 'Microfono MacBook Pro', 'g-mac')];
const OUTPUTS = [
  dev('default', 'audiooutput', 'Predefinito - Altoparlanti MacBook Pro', 'g-mac'),
  dev('usb-out', 'audiooutput', 'Scarlett 2i2 USB', 'g-usb'),
  dev('mac-out', 'audiooutput', 'Altoparlanti MacBook Pro', 'g-mac'),
  dev('hp-out', 'audiooutput', 'External Headphones', 'g-jack'),
  dev('bt-out', 'audiooutput', 'Cuffie AirPods di Marco', 'g-bt'),
];

const goodStats = (over: Partial<AudioStats> = {}): AudioStats => ({
  sampleRate: 48000,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    sampleRate: 48000,
    channelCount: 2,
    deviceId: 'usb-in',
  },
  sinkId: '',
  ...over,
});

class FakeEngine implements EngineLike {
  sent: WorkletInMsg[] = [];
  state: 'idle' | 'mic' | 'error' = 'idle';
  error: string | null = null;
  stats: AudioStats = { ...goodStats(), sampleRate: 0, outputLatencyMs: null, trackSettings: null };
  trackInfo: TrackInfo | null = null;
  sinkSupported = true;
  channels = 2;
  sink = '';
  failOpen: Error | null = null;
  /** setOutput rejects these ids (the sink stays where it was). */
  failingSinks = new Set<string>();
  /** measure() stays pending until abortMeasure (or the test) settles it. */
  holdMeasure = false;
  pendingMeasure: ((r: MeasureResult) => void) | null = null;
  private listeners = new Set<(e: AudioEngineEvent) => void>();
  readonly openMic = vi.fn(async (id: string | null, _opts: { wantErrorMic: boolean }) => {
    if (this.failOpen) {
      this.state = 'error';
      this.error = this.failOpen.message;
      throw this.failOpen;
    }
    const d = INPUTS.find((x) => x.deviceId === id) ?? INPUTS[0]!;
    this.state = 'mic';
    this.stats = goodStats({ trackSettings: { ...goodStats().trackSettings, channelCount: this.channels, deviceId: d.deviceId } });
    this.trackInfo = {
      deviceId: d.deviceId,
      groupId: d.groupId,
      label: d.label,
      settings: this.stats.trackSettings!,
      channelCount: this.channels,
    };
  });
  readonly setOutput = vi.fn(async (sinkId: string) => {
    if (this.failingSinks.has(sinkId)) throw new Error('Dispositivo occupato');
    this.sink = sinkId;
  });
  readonly ensureContextMatchesTrack = vi.fn(async () => false);
  readonly setLatencyMode = vi.fn(async (_m: LatencyMode) => undefined);
  readonly measure = vi.fn(
    async (s: Float32Array, _c: 0 | 1, p: number): Promise<MeasureResult> =>
      this.holdMeasure
        ? new Promise<MeasureResult>((resolve) => {
            this.pendingMeasure = resolve;
          })
        : { ref: new Float32Array(s.length + p), err: null, masterLin: 0.03, aborted: false },
  );
  readonly panic = vi.fn(() => this.send({ type: 'panic' }));
  readonly resume = vi.fn(async () => undefined);
  send(msg: WorkletInMsg): void {
    this.sent.push(msg);
    if (msg.type === 'abortMeasure' && this.pendingMeasure) {
      const r = this.pendingMeasure;
      this.pendingMeasure = null;
      r({ ref: new Float32Array(0), err: null, masterLin: 0, aborted: true });
    }
  }
  on(cb: (e: AudioEngineEvent) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
  emit(e: AudioEngineEvent): void {
    for (const cb of [...this.listeners]) cb(e);
  }
  types(): string[] {
    return this.sent.map((m) => m.type);
  }
  lastParams(): EngineParams {
    const p = [...this.sent].reverse().find((m) => m.type === 'params');
    if (!p || p.type !== 'params') throw new Error('no params sent');
    return p.params;
  }
}

const devices: DeviceProvider = {
  list: async () => ({ inputs: [...INPUTS], outputs: [...OUTPUTS] }),
  platform: () => 'mac',
  chromium: () => true,
};

let clockMs = 1000;
const controllers: AppController[] = [];

function setup(opts: { channels?: number } = {}): { c: AppController; e: FakeEngine; storage: MemoryStorage } {
  const e = new FakeEngine();
  if (opts.channels !== undefined) e.channels = opts.channels;
  const storage = new MemoryStorage();
  const c = new AppController({ engine: e, devices, presetStorage: storage, prefsStorage: null, now: () => clockMs });
  controllers.push(c);
  return { c, e, storage };
}

/** Opens the mic on the USB interface and routes the output to it. */
async function openUsb(c: AppController, e: FakeEngine): Promise<void> {
  await c.openMic();
  await c.setOutput('usb-out');
  e.sent = [];
}

const meters = (over: Partial<MeterSnapshot> = {}): AudioEngineEvent => ({
  type: 'meters',
  ref: { rmsDb: -30, peakDb: -20 },
  err: { rmsDb: -40, peakDb: -30 },
  outL: { rmsDb: -60, peakDb: -50 },
  outR: { rmsDb: -60, peakDb: -50 },
  limiterGrDb: [0, 0],
  armed: true,
  muted: false,
  bypass: false,
  softStart: 1,
  delaySamples: [1, 1],
  delayClamped: [false, false],
  blockSize: 128,
  ch2: 'ok',
  measuring: false,
  ...over,
});

/** A tap block with sines of the given RMS levels (dBFS) on ref/cond and err. */
function tapBlock(refDb: number, errDb: number): Float32Array {
  const n = TAP_BLOCK_FRAMES;
  const b = new Float32Array(TAP_CHANNELS.length * n);
  const a = (db: number): number => Math.SQRT2 * Math.pow(10, db / 20);
  const ref = TAP_CHANNELS.indexOf('ref');
  const cond = TAP_CHANNELS.indexOf('cond');
  const err = TAP_CHANNELS.indexOf('err');
  for (let i = 0; i < n; i++) {
    const s = Math.sin((2 * Math.PI * 375 * i) / 48000);
    b[ref * n + i] = a(refDb) * s;
    b[cond * n + i] = a(refDb) * s;
    b[err * n + i] = a(errDb) * s;
  }
  return b;
}

/** Feeds `seconds` of tap blocks, advancing the fake clock by the block duration. */
function feedTaps(e: FakeEngine, seconds: number, refDb: number, errDb: number): void {
  const blockMs = (TAP_BLOCK_FRAMES / 48000) * 1000;
  const count = Math.ceil((seconds * 1000) / blockMs);
  for (let i = 0; i < count; i++) {
    clockMs += blockMs;
    e.emit({ type: 'tap', block: tapBlock(refDb, errDb), frames: TAP_BLOCK_FRAMES });
  }
}

afterEach(() => {
  for (const c of controllers.splice(0)) c.dispose();
  vi.useRealTimers();
});

// ------------------------------------------------------------------------------- pure helpers

describe('pure helpers', () => {
  it('safeApplySequence: audible → mute, params, un-mute (soft-start); silent → params only', () => {
    const p = clampParams(DEFAULT_PARAMS);
    expect(safeApplySequence(p, true).map((m) => (m.type === 'mute' ? `mute:${m.muted}` : m.type))).toEqual([
      'mute:true',
      'params',
      'mute:false',
    ]);
    expect(safeApplySequence(p, false).map((m) => m.type)).toEqual(['params']);
  });

  it('describeDevice blocks wireless outputs and flags built-in / headphones', () => {
    expect(describeDevice(OUTPUTS[4]!).blocked).toBe(true);
    expect(describeDevice(dev('x', 'audiooutput', 'AirPlay Salotto', '')).blocked).toBe(true);
    expect(describeDevice(OUTPUTS[1]!)).toEqual({ blocked: false });
    expect(describeDevice(OUTPUTS[2]!).note).toMatch(/integrat/);
    expect(describeDevice(OUTPUTS[3]!).note).toMatch(/cuffie/);
  });
});

// ------------------------------------------------------------------------------- arming

describe('AppController: validation gate', () => {
  it('starts disarmed with the safe first-run defaults', () => {
    const { c, e } = setup();
    const s = c.state;
    expect(s.armed).toBe(false);
    expect(s.params.masterDb).toBe(-30);
    expect(s.params.channels[0].gainDb).toBe(-20);
    expect(s.params.channels[1].gainDb).toBe(-20);
    expect(s.params.ceilingDb).toBe(-12);
    expect(s.params.gainUnlocked).toBe(false);
    // No device key while the microphone is closed (the wizard reads it as "unknown").
    expect(s.deviceKey).toBe('');
    // The engine got the params (replayed on its processor), never an arm.
    expect(e.types()).toContain('params');
    expect(e.sent.some((m) => m.type === 'arm' && m.armed)).toBe(false);
  });

  it('refuses to arm before the microphone is open', () => {
    const { c, e } = setup();
    const r = c.arm();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('no-mic');
    expect(e.sent.some((m) => m.type === 'arm')).toBe(false);
  });

  it('opening the mic alone never arms; "Attiva uscita" sends params, un-mute, arm in this order', async () => {
    const { c, e } = setup();
    await c.openMic();
    expect(e.openMic).toHaveBeenCalledTimes(1);
    expect(c.state.micOpen).toBe(true);
    expect(e.sent.some((m) => m.type === 'arm' && m.armed)).toBe(false);
    expect(e.sent).toContainEqual({ type: 'tap', enabled: true });
    // The output of the same interface is chosen automatically.
    expect(c.state.outputId).toBe('usb-out');
    expect(e.setOutput).toHaveBeenCalledWith('usb-out');
    expect(c.state.validation.ok).toBe(true);
    expect(c.state.canArm).toBe(true);

    e.sent = [];
    const r = c.arm();
    expect(r).toEqual({ ok: true });
    expect(e.sent.map((m) => m.type)).toEqual(['params', 'mute', 'arm']);
    expect(e.sent[1]).toEqual({ type: 'mute', muted: false });
    expect(e.sent[2]).toEqual({ type: 'arm', armed: true });
    expect(c.state.armed).toBe(true);
  });

  it('un-muting (M) never arms the output', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.setMuted(true);
    expect(c.setMuted(false)).toBe(false);
    c.toggleMute();
    c.toggleMute();
    expect(e.sent.some((m) => m.type === 'arm')).toBe(false);
    expect(c.state.armed).toBe(false);
  });

  it('the first M while disarmed explains «Attiva uscita» (no silent mute) and never arms', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    expect(c.state.muted).toBe(false);
    c.toggleMute();
    expect(c.state.message).toEqual({ level: 'info', text: "L'uscita non è attiva: usa «Attiva uscita» dopo la validazione." });
    expect(c.state.armed).toBe(false);
    expect(c.state.status).toBe('listening');
    c.dismissMessage();
    c.toggleMute();
    expect(c.state.message?.text).toMatch(/uscita non è attiva/);
    expect(e.sent.some((m) => m.type === 'arm')).toBe(false);
    // «Attiva uscita» still sets the mute state itself.
    expect(c.arm()).toEqual({ ok: true });
    expect(c.state.muted).toBe(false);
    c.toggleMute();
    expect(c.state.muted).toBe(true);
  });

  it('a Bluetooth output blocks arming and cannot be overridden', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    await c.setOutput('bt-out');
    const v = c.state.validation;
    expect(v.ok).toBe(false);
    expect(firstProblem(v)?.id).toBe(ISSUE.bluetoothOutput);
    c.setOverride(ISSUE.bluetoothOutput, true);
    const r = c.arm();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('validation');
    expect(e.sent.some((m) => m.type === 'arm' && m.armed)).toBe(false);
  });

  it('an armed output is disarmed as soon as validation fails; the latency error can be overridden', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    expect(c.arm().ok).toBe(true);
    e.sent = [];
    e.stats = goodStats({ outputLatencyMs: 80 });
    e.emit({ type: 'stats', stats: e.stats });
    expect(c.state.armed).toBe(false);
    expect(e.sent).toContainEqual({ type: 'arm', armed: false });
    expect(firstProblem(c.state.validation)?.id).toBe(ISSUE.outputLatency);
    expect(c.state.message?.level).toBe('error');

    c.setOverride(ISSUE.outputLatency, true);
    expect(c.state.validation.ok).toBe(true);
    expect(c.arm().ok).toBe(true);
  });

  it('forcedOverrides (e2e build) keep the latency rule overridden across device changes', async () => {
    const e = new FakeEngine();
    const c = new AppController({
      engine: e,
      devices,
      presetStorage: null,
      prefsStorage: null,
      forcedOverrides: [ISSUE.outputLatency],
    });
    controllers.push(c);
    await openUsb(c, e);
    e.stats = goodStats({ outputLatencyMs: 90 });
    e.emit({ type: 'stats', stats: e.stats });
    await c.setOutput('usb-out');
    expect(c.state.validation.ok).toBe(true);
    expect(c.state.validation.issues.find((i) => i.id === ISSUE.outputLatency)?.overridden).toBe(true);
  });

  it('changing a device disarms and clears the overrides', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.setOverride(ISSUE.outputLatency, true);
    expect(c.arm().ok).toBe(true);
    await c.setOutput('mac-out');
    expect(c.state.armed).toBe(false);
    expect(c.state.overrides).toEqual([]);
    expect(e.sent).toContainEqual({ type: 'arm', armed: false });
  });

  it('headphones force the ceiling to −24 dBFS and need an explicit confirmation', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.setCeiling(-3);
    await c.setOutput('hp-out');
    expect(c.state.validation.headphones).toBe(true);
    expect(c.state.params.headphones).toBe(true);
    expect(c.state.params.ceilingDb).toBeLessThanOrEqual(HEADPHONES_CEILING_DB);
    expect(e.lastParams().ceilingDb).toBeLessThanOrEqual(HEADPHONES_CEILING_DB);
    c.setCeiling(-1);
    expect(e.lastParams().ceilingDb).toBeLessThanOrEqual(HEADPHONES_CEILING_DB);

    const r = c.arm();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('headphones-confirm');
    expect(c.arm({ headphonesConfirmed: true }).ok).toBe(true);
  });

  it('a mono track cannot carry the reference on input 2 nor an error mic', async () => {
    const { c, e } = setup({ channels: 1 });
    await openUsb(c, e);
    expect(c.state.ch2).toBe('absent');
    c.setRefChannel(1);
    expect(c.state.validation.ok).toBe(false);
    c.setRefChannel(0);
    expect(c.state.validation.ok).toBe(true);
    // Asking for the error mic on a mono track re-opens the input asking for two channels.
    c.setErrorChannel(1);
    await vi.waitFor(() => expect(e.openMic).toHaveBeenCalledTimes(2));
    expect(e.openMic.mock.calls[1]?.[1]).toEqual({ wantErrorMic: true });
    await vi.waitFor(() => expect(c.state.opening).toBe(false));
    expect(c.state.validation.issues.some((i) => i.id === ISSUE.errorMicUnavailable)).toBe(true);
    expect(c.state.validation.ok).toBe(false);
  });

  it('a microphone failure is reported in Italian and leaves the output disarmed', async () => {
    const { c, e } = setup();
    const err = Object.assign(new Error('Accesso al microfono negato: consentilo dalle impostazioni del sito.'), {
      code: 'denied',
    });
    e.failOpen = err;
    await c.openMic();
    const s = c.state;
    expect(s.micOpen).toBe(false);
    expect(s.permission).toBe('denied');
    expect(s.micError).toMatch(/negato/);
    expect(s.message?.level).toBe('error');
    expect(c.arm().ok).toBe(false);
  });
});

// ------------------------------------------------------------------------------- safety actions

describe('AppController: PANIC, mute, bypass, keyboard, lifecycle', () => {
  it('PANIC never throws, even with no microphone, and shows the PANIC state', () => {
    const { c, e } = setup();
    e.panic.mockImplementation(() => {
      throw new Error('boom');
    });
    expect(() => c.panic()).not.toThrow();
    expect(c.state.status).toBe('panic');
    expect(c.state.muted).toBe(true);
    // The fallback still reached the processor.
    expect(e.sent).toContainEqual({ type: 'panic' });
  });

  it('after PANIC, un-mute resumes the context and soft-starts; arm() does the same', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    c.panic();
    expect(e.panic).toHaveBeenCalledTimes(1);
    expect(c.state.panic).toBe(true);
    e.sent = [];
    c.toggleMute();
    expect(e.resume).toHaveBeenCalled();
    expect(e.sent).toEqual([{ type: 'mute', muted: false }]);
    expect(c.state.panic).toBe(false);
    expect(c.state.muted).toBe(false);

    c.panic();
    e.sent = [];
    expect(c.arm().ok).toBe(true);
    expect(e.sent).toContainEqual({ type: 'mute', muted: false });
    expect(c.state.status).not.toBe('panic');
  });

  it('keyboard: Escape panics from a form field; M/B are ignored there and with modifiers', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    expect(c.handleKey({ key: 'm', target: { tagName: 'INPUT' } })).toBe(false);
    expect(c.handleKey({ key: 'b', ctrlKey: true })).toBe(false);
    expect(c.state.muted).toBe(false);
    expect(c.handleKey({ key: 'm', target: { tagName: 'BUTTON' } })).toBe(true);
    expect(c.state.muted).toBe(true);
    expect(c.handleKey({ key: 'B' })).toBe(true);
    expect(c.state.bypass).toBe(true);
    expect(e.sent).toContainEqual({ type: 'bypass', bypass: true });
    expect(c.handleKey({ key: 'Escape', target: { tagName: 'INPUT' } })).toBe(true);
    expect(c.state.panic).toBe(true);
  });

  it('lifecycle events mute; howl and fault messages are shown', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    e.emit({ type: 'lifecycle', reason: 'I dispositivi audio sono cambiati.', autoMuted: true, cause: 'devicechange' });
    expect(c.state.muted).toBe(true);
    expect(c.state.message?.text).toMatch(/dispositivi/);

    c.setMuted(false);
    e.emit({ type: 'howl', reason: 'peak-growth', freqHz: 440 });
    e.emit({ type: 'autoMute', reason: 'Innesco rilevato a 440 Hz. Uscita silenziata.' });
    expect(c.state.muted).toBe(true);
    expect(c.state.message).toEqual({ level: 'error', text: 'Innesco rilevato a 440 Hz. Uscita silenziata.' });
  });

  it('a track that ends disarms the output (no input any more)', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    e.state = 'error';
    e.error = 'Il microfono si è scollegato.';
    e.emit({ type: 'lifecycle', reason: 'Il microfono si è scollegato.', autoMuted: true, cause: 'track-ended' });
    expect(c.state.armed).toBe(false);
    expect(c.state.micError).toMatch(/scollegato/);
  });
});

// ------------------------------------------------------------------------------- parameters and presets

describe('AppController: parameters and presets', () => {
  it('link L/R keeps the offset between channels', () => {
    const { c, e } = setup();
    const ch = c.state.params.channels;
    c.setChannel(1, { ...ch[1], gainDb: -26 }, 'gainDb');
    c.setLinked(true);
    c.setChannel(0, { ...c.state.params.channels[0], gainDb: -17 }, 'gainDb');
    expect(c.state.params.channels[0].gainDb).toBeCloseTo(-17, 9);
    expect(c.state.params.channels[1].gainDb).toBeCloseTo(-23, 9);
    expect(e.lastParams().channels[1].gainDb).toBeCloseTo(-23, 9);
  });

  it('preset load while audible: mute → apply → soft-start, never restoring the +12 dB unlock', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    c.setGainUnlocked(true);
    const hot = {
      id: 'hot',
      name: 'Troppo forte',
      createdAt: 1,
      deviceKey: '',
      sampleRate: 48000,
      params: {
        ...DEFAULT_PARAMS,
        gainUnlocked: true,
        masterDb: 0,
        ceilingDb: -1,
        channels: [
          { ...DEFAULT_PARAMS.channels[0], gainDb: 12 },
          { ...DEFAULT_PARAMS.channels[1], gainDb: 9 },
        ],
      },
    };
    c.importPresets(JSON.stringify(hot));
    expect(c.state.presets.list.map((p) => p.id)).toEqual(['hot']);
    e.sent = [];
    c.loadPreset('hot');
    const seq = e.sent.map((m) => (m.type === 'mute' ? `mute:${m.muted}` : m.type));
    expect(seq).toEqual(['mute:true', 'params', 'mute:false']);
    const p = e.lastParams();
    expect(p.gainUnlocked).toBe(false);
    expect(p.channels[0].gainDb).toBe(0);
    expect(p.channels[1].gainDb).toBe(0);
    expect(p.ceilingDb).toBe(-1);
    expect(c.state.params.gainUnlocked).toBe(false);
    expect(c.state.calibratedChannels?.[0].gainDb).toBe(0);
    // No device key in the preset: the user is warned.
    expect(c.state.presets.message?.level).toBe('warn');
  });

  it('preset load while muted only applies the parameters', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    c.savePreset('Cucina');
    const id = c.state.presets.list[0]!.id;
    c.setMuted(true);
    e.sent = [];
    c.loadPreset(id);
    expect(e.types()).toEqual(['params']);
    expect(c.state.presets.activeId).toBe(id);
  });

  it('saved presets carry the device key and sample rate, and calibration data as plain JSON', async () => {
    const { c, e, storage } = setup();
    await openUsb(c, e);
    c.setCalibration({ version: 1, fs: 48000, values: [1, 2, 3] });
    c.savePreset('Con calibrazione');
    const p = c.state.presets.list[0]!;
    expect(p.sampleRate).toBe(48000);
    expect(p.deviceKey).toMatch(/Scarlett/);
    expect(p.calibration).toEqual({ version: 1, fs: 48000, values: [1, 2, 3] });
    expect(storage.length).toBeGreaterThan(0);
    const exp = c.exportPresets(p.id);
    expect(exp?.filename).toMatch(/\.json$/);
    expect(JSON.parse(exp!.json).presets).toHaveLength(1);
  });

  it('applyParams keeps the unlock state and follows the headphones flag; softStart goes through mute', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    e.sent = [];
    c.applyParams({ ...c.state.params, gainUnlocked: true, masterDb: -10 }, { softStart: true, calibrated: true });
    expect(e.sent.map((m) => (m.type === 'mute' ? `mute:${m.muted}` : m.type))).toEqual(['mute:true', 'params', 'mute:false']);
    expect(e.lastParams().gainUnlocked).toBe(false);
    expect(e.lastParams().masterDb).toBe(-10);
    expect(c.state.calibratedChannels).not.toBeNull();
    e.sent = [];
    c.applyParams({ ...c.state.params, masterDb: -12 });
    expect(e.types()).toEqual(['params']);
  });

  it('single-speaker configuration turns the R output off', () => {
    const { c } = setup();
    c.setSpeakers('single');
    expect(c.state.params.channels[1].enabled).toBe(false);
    c.setSpeakers('shared');
    expect(c.state.params.channels[1].enabled).toBe(true);
  });

  it('feedback FIRs from calibration are applied, and cleared when no converter is configured', () => {
    const e = new FakeEngine();
    const taps = new Float64Array([0.1, -0.05]);
    const c = new AppController({
      engine: e,
      devices,
      presetStorage: null,
      prefsStorage: null,
      firsFromCalibration: () => [{ bulkDelay: 10, taps }, null],
    });
    controllers.push(c);
    c.setCalibration({ version: 1 });
    const fir = e.sent.find((m) => m.type === 'feedbackFir');
    expect(fir).toMatchObject({ type: 'feedbackFir', channel: 0, bulkDelay: 10 });
    expect(c.state.feedbackFirActive).toEqual([true, false]);
    e.sent = [];
    c.setCalibration(null);
    expect(e.sent).toContainEqual({ type: 'feedbackFir', channel: 0, bulkDelay: 0, taps: null });
    // Channel 1 had none: no needless message (it would trigger a reconfigure ramp).
    expect(e.sent.filter((m) => m.type === 'feedbackFir')).toHaveLength(1);
  });
});

// ------------------------------------------------------------------------------- tap analysis

describe('AppController: tap analysis', () => {
  async function armedWithErrorMic(): Promise<{ c: AppController; e: FakeEngine }> {
    const { c, e } = setup();
    await openUsb(c, e);
    c.setErrorChannel(1);
    e.emit({ type: 'channelCheck', status: 'ok' });
    expect(c.state.ch2).toBe('ok');
    return { c, e };
  }

  it('measures the attenuation against the bypass reference', async () => {
    const { c, e } = await armedWithErrorMic();
    // Disarmed: ANC off. err − ref = −6 dB.
    feedTaps(e, (PHASE_SETTLE_MS + 1500) / 1000, -30, -36);
    expect(c.arm().ok).toBe(true);
    e.emit(meters({ softStart: 1 }));
    // ANC on: the error drops 20 dB.
    feedTaps(e, 3.5, -30, -56);
    expect(c.state.attenuationDb).toBeCloseTo(20, 1);
    expect(c.state.muted).toBe(false);
    expect(c.spectra.ref).not.toBeNull();
    expect(c.spectra.err).not.toBeNull();
    expect(c.spectra.errBypass).not.toBeNull();
  });

  it('"ANC sta aumentando il rumore": error 6 dB above bypass for > 2 s mutes the output', async () => {
    const { c, e } = await armedWithErrorMic();
    feedTaps(e, (PHASE_SETTLE_MS + 1500) / 1000, -30, -36);
    c.arm();
    e.emit(meters({ softStart: 1 }));
    e.sent = [];
    feedTaps(e, 1.5, -30, -30);
    expect(c.state.muted).toBe(false);
    feedTaps(e, 1.5, -30, -30);
    expect(c.state.muted).toBe(true);
    expect(e.sent).toContainEqual({ type: 'mute', muted: true });
    expect(c.state.message?.text).toMatch(/ANC sta aumentando il rumore/);
  });

  it('capture() records the ambient with the anti-noise muted, then soft-starts it again', async () => {
    vi.useFakeTimers();
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    e.sent = [];
    const p = c.capture(0.1);
    expect(e.sent).toEqual([{ type: 'mute', muted: true }]);
    expect(c.state.capturing).toBe(true);
    // Blocks before the settling time are not recorded.
    feedTaps(e, 0.2, -10, -10);
    vi.advanceTimersByTime(350);
    feedTaps(e, 0.2, -30, -40);
    const r = await p;
    expect(r.ref.length).toBe(4800);
    expect(r.err).toBeNull();
    const rms = Math.sqrt(r.ref.reduce((a, v) => a + v * v, 0) / r.ref.length);
    expect(20 * Math.log10(rms)).toBeCloseTo(-30, 0);
    expect(e.sent[e.sent.length - 1]).toEqual({ type: 'mute', muted: false });
    expect(c.state.capturing).toBe(false);
  });

  it('capture() is rejected by PANIC and does not un-mute afterwards', async () => {
    vi.useFakeTimers();
    const { c, e } = setup();
    await openUsb(c, e);
    c.arm();
    const p = c.capture(1);
    c.panic();
    await expect(p).rejects.toThrow(/PANIC/);
    expect(c.state.muted).toBe(true);
    expect(e.sent[e.sent.length - 1]?.type).not.toBe('mute');
  });

  it('measure() needs an armed output', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    const r = await c.measure(new Float32Array(100), 0, 10);
    expect(r.aborted).toBe(true);
    expect(e.measure).not.toHaveBeenCalled();
    c.arm({ muted: true });
    const ok = await c.measure(new Float32Array(100), 0, 10);
    expect(ok.aborted).toBe(false);
    expect(e.measure).toHaveBeenCalledTimes(1);
    expect(c.state.measuring).toBe(false);
  });
});

// ------------------------------------------------------------------------------- review fixes

/** A calibration whose speaker → reference path is an electrical loopback (flat, pure delay). */
function loopbackCalibration(fs = 48000): Record<string, unknown> {
  const freqs: number[] = [];
  for (let k = 0; k < 64; k++) freqs.push(20 * Math.pow(50, k / 63));
  const d = 96 / fs;
  const path = {
    delaySamples: 96,
    sign: 1,
    qualityDb: 40,
    stdSamples: 0.05,
    freqs,
    re: freqs.map((f) => Math.cos(-2 * Math.PI * f * d)),
    im: freqs.map((f) => Math.sin(-2 * Math.PI * f * d)),
    ir: [],
    irOffset: 0,
  };
  return {
    version: 1,
    fs,
    band: [20, 1000],
    tempC: 20,
    createdAt: 1,
    feedback: [path, null],
    secondary: [null, null],
    primary: null,
    loopbackLatencyS: null,
    geometry: null,
    levels: { zoneDb: null, refDb: null },
    speakers: 'single',
  };
}

describe('AppController: review fixes', () => {
  it('changing the reference input while armed disarms; the FIRs only apply to the input they were made with', async () => {
    const e = new FakeEngine();
    const taps = new Float64Array([0.1, -0.05]);
    const c = new AppController({
      engine: e,
      devices,
      presetStorage: null,
      prefsStorage: null,
      firsFromCalibration: () => [{ bulkDelay: 10, taps }, null],
    });
    controllers.push(c);
    await openUsb(c, e);
    c.setCalibration({ version: 1, fs: 48000 });
    expect(c.state.feedbackFirActive).toEqual([true, false]);
    expect(c.arm().ok).toBe(true);
    e.sent = [];
    c.setRefChannel(1);
    // Disarmed before the new input reaches the speaker.
    expect(e.sent[0]).toEqual({ type: 'arm', armed: false });
    expect(c.state.armed).toBe(false);
    expect(c.state.message?.text).toMatch(/Canale del riferimento cambiato/);
    expect(e.sent).toContainEqual({ type: 'feedbackFir', channel: 0, bulkDelay: 0, taps: null });
    expect(c.state.feedbackFirActive).toEqual([false, false]);
    // Back on the calibrated input: the FIRs return; the same channel again changes nothing.
    c.setRefChannel(0);
    expect(c.state.feedbackFirActive).toEqual([true, false]);
    expect(c.arm().ok).toBe(true);
    e.sent = [];
    c.setRefChannel(0);
    expect(e.sent.some((m) => m.type === 'arm' || m.type === 'feedbackFir')).toBe(false);
    expect(c.state.armed).toBe(true);
  });

  it('a calibration made with the other reference input refuses every un-muted start until the input is restored', async () => {
    const e = new FakeEngine();
    const c = new AppController({
      engine: e,
      devices,
      presetStorage: null,
      prefsStorage: null,
      firsFromCalibration: () => [{ bulkDelay: 10, taps: new Float64Array([0.1]) }, null],
    });
    controllers.push(c);
    await openUsb(c, e);
    c.setCalibration({ version: 1, fs: 48000, refChannel: 0 });
    c.setRefChannel(1);
    e.sent = [];
    // «Avvio»/«Attiva uscita» un-muted: refused, nothing reaches the worklet.
    const r = c.arm();
    expect(r).toMatchObject({ ok: false, reason: 'ref-input' });
    expect(c.state.message?.text).toMatch(/calibrazione è stata fatta con il microfono di riferimento sull'ingresso 1/);
    expect(e.sent.some((m) => m.type === 'arm' || (m.type === 'mute' && !m.muted))).toBe(false);
    // Armed muted (calibration stimuli) is allowed, but the anti-noise stays muted.
    expect(c.arm({ muted: true }).ok).toBe(true);
    e.sent = [];
    expect(c.setMuted(false)).toBe(false);
    expect(c.state.muted).toBe(true);
    expect(e.sent).not.toContainEqual({ type: 'mute', muted: false });
    // A calibration set while audible for the other input mutes the output (changing the input disarmed).
    c.setRefChannel(0);
    expect(c.state.armed).toBe(false);
    expect(c.arm().ok).toBe(true);
    expect(c.state.muted).toBe(false);
    e.sent = [];
    c.setCalibration({ version: 1, fs: 48000, refChannel: 1 });
    expect(c.state.muted).toBe(true);
    expect(e.sent).toContainEqual({ type: 'mute', muted: true });
    // Back on the calibrated input: allowed again.
    c.setRefChannel(1);
    c.disarm();
    expect(c.arm().ok).toBe(true);
  });

  it('a lifecycle auto-mute aborts a running measurement and fails a capture', async () => {
    vi.useFakeTimers();
    const causes = ['devicechange', 'sinkchange', 'underruns', 'track-muted', 'context-state', 'latency-jump'] as const;
    for (const cause of causes) {
      const { c, e } = setup();
      await openUsb(c, e);
      expect(c.arm({ muted: true }).ok).toBe(true);
      e.holdMeasure = true;
      const result = c.measure(new Float32Array(48000), 0, 0);
      await Promise.resolve();
      expect(c.state.measuring).toBe(true);
      e.sent = [];
      e.emit({ type: 'lifecycle', reason: 'Uscita silenziata.', autoMuted: true, cause });
      expect(e.types()).toContain('abortMeasure');
      expect((await result).aborted).toBe(true);
      expect(c.state.measuring).toBe(false);

      e.holdMeasure = false;
      const cap = c.capture(1);
      e.emit({ type: 'lifecycle', reason: 'Uscita silenziata.', autoMuted: true, cause });
      await expect(cap).rejects.toThrow(/interrott/);
    }
  });

  it('an output that turns into headphones while armed is muted and needs the confirmation to un-mute', async () => {
    let defaultLabel = 'Predefinito - Scarlett 2i2 USB';
    let defaultGroup = 'g-usb';
    const provider: DeviceProvider = {
      ...devices,
      list: async () => ({
        inputs: [...INPUTS],
        outputs: [dev('default', 'audiooutput', defaultLabel, defaultGroup), ...OUTPUTS.slice(1)],
      }),
    };
    const e = new FakeEngine();
    const c = new AppController({ engine: e, devices: provider, presetStorage: null, prefsStorage: null, now: () => clockMs });
    controllers.push(c);
    await c.openMic();
    await c.setOutput('default');
    expect(c.arm().ok).toBe(true);
    // macOS moves the default output to the headphones just plugged in; Chrome follows it.
    defaultLabel = 'Predefinito - External Headphones';
    defaultGroup = 'g-jack';
    e.sent = [];
    e.emit({ type: 'lifecycle', reason: 'I dispositivi audio sono cambiati.', autoMuted: true, cause: 'devicechange' });
    await vi.waitFor(() => expect(c.state.validation.headphones).toBe(true));
    expect(c.state.headphonesConfirmed).toBe(false);
    expect(c.state.armed).toBe(true);
    expect(c.state.muted).toBe(true);
    expect(e.lastParams().ceilingDb).toBeLessThanOrEqual(HEADPHONES_CEILING_DB);
    e.sent = [];
    expect(c.setMuted(false)).toBe(false);
    expect(e.sent.some((m) => m.type === 'mute' && !m.muted)).toBe(false);
    expect(c.state.message?.text).toMatch(/cuffie/);
    c.confirmHeadphones(true);
    expect(c.setMuted(false)).toBe(true);
    expect(e.sent).toContainEqual({ type: 'mute', muted: false });
  });

  it('a calibration with a loopback signature refuses every un-mute (console, arm, preset)', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    expect(c.arm().ok).toBe(true);
    // Set while audible: muted at once.
    c.setCalibration(loopbackCalibration());
    expect(c.state.muted).toBe(true);
    expect(e.sent[e.sent.length - 1]).toEqual({ type: 'mute', muted: true });
    expect(c.state.message?.text).toMatch(/loopback/);
    e.sent = [];
    expect(c.setMuted(false)).toBe(false);
    c.disarm();
    const r = c.arm();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('loopback');
    // Armed muted: calibration stimuli can still play (e.g. to repeat the measurements).
    expect(c.arm({ muted: true }).ok).toBe(true);
    expect(e.sent.some((m) => m.type === 'mute' && !m.muted)).toBe(false);

    // A preset carrying it, loaded while audible, leaves the output muted.
    c.savePreset('Loopback');
    const id = c.state.presets.activeId!;
    c.setCalibration(null);
    expect(c.setMuted(false)).toBe(true);
    e.sent = [];
    c.loadPreset(id);
    expect(e.sent.filter((m) => m.type === 'mute')).toEqual([{ type: 'mute', muted: true }]);
    expect(c.state.muted).toBe(true);
    expect(c.state.presets.message?.text).toMatch(/loopback/);
    // An acoustic calibration (or none) un-mutes again.
    c.setCalibration(null);
    expect(c.setMuted(false)).toBe(true);
  });

  it('feedback FIRs are applied only at the calibration sample rate, and follow rate changes', async () => {
    const e = new FakeEngine();
    const taps = new Float64Array([0.1, -0.05]);
    const hook = vi.fn((_cal: unknown, _taps: number) => [{ bulkDelay: 10, taps }, null] as [{ bulkDelay: number; taps: Float64Array }, null]);
    const storage = new MemoryStorage();
    const c = new AppController({ engine: e, devices, presetStorage: storage, prefsStorage: null, firsFromCalibration: hook });
    controllers.push(c);
    await openUsb(c, e);
    // A preset calibrated at 44.1 kHz loaded at 48 kHz: no FIR, and a warning.
    c.setCalibration({ version: 1, fs: 44100 });
    expect(c.state.feedbackFirActive).toEqual([false, false]);
    expect(e.sent.some((m) => m.type === 'feedbackFir' && m.taps !== null)).toBe(false);
    expect(c.state.message?.text).toMatch(/44,1 kHz.*48 kHz.*neutralizzazione del feedback/);
    c.savePreset('44k');
    const id = c.state.presets.activeId!;
    c.setCalibration(null);
    e.sent = [];
    c.loadPreset(id);
    expect(e.sent.some((m) => m.type === 'feedbackFir' && m.taps !== null)).toBe(false);

    // Calibrated at 48 kHz: applied; the context moves to 44.1 kHz: cleared; back: applied again.
    c.setCalibration({ version: 1, fs: 48000 });
    expect(c.state.feedbackFirActive).toEqual([true, false]);
    e.sent = [];
    e.emit({ type: 'stats', stats: goodStats({ sampleRate: 44100 }) });
    expect(e.sent).toContainEqual({ type: 'feedbackFir', channel: 0, bulkDelay: 0, taps: null });
    expect(c.state.feedbackFirActive).toEqual([false, false]);
    e.emit({ type: 'ready', sampleRate: 48000 });
    expect(c.state.feedbackFirActive).toEqual([true, false]);
  });

  it('feedback taps: clamped, remembered, passed to the FIR design and re-applied', async () => {
    const e = new FakeEngine();
    const hook = vi.fn((_cal: unknown, n: number) => [{ bulkDelay: 10, taps: new Float64Array(n) }, null] as [{ bulkDelay: number; taps: Float64Array }, null]);
    const prefs = new MemoryStorage();
    const c = new AppController({ engine: e, devices, presetStorage: null, prefsStorage: prefs, firsFromCalibration: hook });
    controllers.push(c);
    expect(c.state.feedbackTaps).toBe(256);
    c.setCalibration({ version: 1 });
    expect(hook).toHaveBeenLastCalledWith({ version: 1 }, 256);
    c.setFeedbackTaps(5000);
    expect(c.state.feedbackTaps).toBe(1024);
    expect(hook).toHaveBeenLastCalledWith({ version: 1 }, 1024);
    c.setFeedbackTaps(10);
    expect(c.state.feedbackTaps).toBe(128);
    const again = new AppController({ engine: new FakeEngine(), devices, presetStorage: null, prefsStorage: prefs });
    controllers.push(again);
    expect(again.state.feedbackTaps).toBe(128);
  });

  it('a failed output switch keeps the output actually in use (shown, validated, remembered)', async () => {
    const e = new FakeEngine();
    const prefs = new MemoryStorage();
    const c = new AppController({ engine: e, devices, presetStorage: null, prefsStorage: prefs, now: () => clockMs });
    controllers.push(c);
    // Auto-selection at mic open fails: the context stays on the system default (MacBook speakers).
    e.failingSinks.add('usb-out');
    await c.openMic();
    expect(e.sink).toBe('');
    expect(c.state.outputId).toBeNull();
    // The panel and the validation follow the default output, not the one that failed.
    expect(c.state.validation.issues.some((i) => i.id === ISSUE.clockDrift)).toBe(true);
    e.failingSinks.clear();
    await c.setOutput('usb-out');
    expect(c.state.outputId).toBe('usb-out');
    // A user switch that fails keeps usb-out (validated, remembered).
    e.failingSinks.add('bt-out');
    await c.setOutput('bt-out');
    expect(e.sink).toBe('usb-out');
    expect(c.state.outputId).toBe('usb-out');
    expect(c.state.validation.ok).toBe(true);
    expect(c.state.message?.level).toBe('error');
    expect(JSON.parse(prefs.getItem('controfase.devices.v1')!).outputId).toBe('usb-out');
  });

  it('the «ANC attivo» error spectrum only averages audio with the anti-noise on', async () => {
    const { c, e } = setup();
    await openUsb(c, e);
    c.setErrorChannel(1);
    e.emit({ type: 'channelCheck', status: 'ok' });
    // Disarmed: only the bypass curve exists.
    feedTaps(e, 2, -30, -36);
    expect(c.spectra.errBypass).not.toBeNull();
    expect(c.spectra.err).toBeNull();
    const bypassPeak = Math.max(...c.spectra.errBypass!);
    expect(c.arm().ok).toBe(true);
    e.emit(meters({ softStart: 1 }));
    feedTaps(e, 2, -30, -56);
    const on = c.spectra.err!;
    expect(Math.max(...on)).toBeCloseTo(bypassPeak - 20, 0);
    // Bypass (B): the ANC-on curve keeps its last value, the bypass curve gets the new audio.
    c.setBypass(true);
    feedTaps(e, 2, -30, -40);
    expect(Math.max(...c.spectra.err!)).toBeCloseTo(bypassPeak - 20, 0);
    expect(Math.max(...c.spectra.errBypass!)).toBeCloseTo(bypassPeak - 4, 0);
  });

  it('the per-band reference spectra follow the ANC state, from x̂ (cond) while the neutraliser runs (plan-9)', async () => {
    const taps = new Float64Array(16);
    taps[0] = 0.5;
    const e = new FakeEngine();
    const c = new AppController({
      engine: e,
      devices,
      presetStorage: null,
      prefsStorage: null,
      now: () => clockMs,
      firsFromCalibration: () => [{ bulkDelay: 10, taps }, null],
      loopbackSuspects: () => [],
    });
    controllers.push(c);
    await openUsb(c, e);
    c.setErrorChannel(1);
    e.emit({ type: 'channelCheck', status: 'ok' });
    /** `seconds` of blocks with different levels on ref, cond and err. */
    const feed = (seconds: number, refDb: number, condDb: number, errDb: number): void => {
      const n = TAP_BLOCK_FRAMES;
      const a = (db: number): number => Math.SQRT2 * Math.pow(10, db / 20);
      const blockMs = (n / 48000) * 1000;
      for (let k = 0; k < Math.ceil((seconds * 1000) / blockMs); k++) {
        const b = new Float32Array(TAP_CHANNELS.length * n);
        for (let i = 0; i < n; i++) {
          const v = Math.sin((2 * Math.PI * 375 * (k * n + i)) / 48000);
          b[TAP_CHANNELS.indexOf('ref') * n + i] = a(refDb) * v;
          b[TAP_CHANNELS.indexOf('cond') * n + i] = a(condDb) * v;
          b[TAP_CHANNELS.indexOf('err') * n + i] = a(errDb) * v;
        }
        clockMs += blockMs;
        e.emit({ type: 'tap', block: b, frames: n });
      }
    };
    const peak = (x: Float32Array | null): number => Math.max(...x!);
    // No neutraliser: the raw reference, per state.
    feed(2, -30, -35, -40);
    expect(c.spectra.refOn).toBeNull();
    const off = peak(c.spectra.refOff);
    expect(c.arm().ok).toBe(true);
    e.emit(meters({ softStart: 1 }));
    // With the anti-noise on, the raw reference also carries the speaker's feedback (+3 dB here).
    feed(2, -27, -32, -50);
    expect(peak(c.spectra.refOn) - off).toBeCloseTo(3, 0);
    expect(peak(c.spectra.refOff)).toBeCloseTo(off, 3);
    // With a calibration the FIRs run: both states restart from x̂ ('cond'), which has no feedback.
    c.setCalibration({ fs: 48000 });
    expect(c.state.feedbackFirActive[0]).toBe(true);
    expect(c.spectra.refOn).toBeNull();
    expect(c.spectra.refOff).toBeNull();
    feed(2, -27, -35, -50);
    const onCond = peak(c.spectra.refOn);
    c.setBypass(true);
    feed(2, -30, -35, -40);
    expect(peak(c.spectra.refOff)).toBeCloseTo(onCond, 1);
  });

  it('a dismissed permission prompt (NotAllowedError) is not shown as "accesso negato"', async () => {
    let browserSays: MicPermission = 'prompt';
    const e = new FakeEngine();
    const c = new AppController({
      engine: e,
      devices: { ...devices, permission: async () => browserSays },
      presetStorage: null,
      prefsStorage: null,
    });
    controllers.push(c);
    e.failOpen = Object.assign(new Error('Accesso al microfono negato.'), { code: 'denied' });
    await c.openMic();
    await vi.waitFor(() => expect(c.state.permission).toBe('prompt'));
    // A real denial stays.
    browserSays = 'denied';
    await c.openMic();
    await Promise.resolve();
    await Promise.resolve();
    expect(c.state.permission).toBe('denied');
  });
});
