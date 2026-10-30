/**
 * The controller hooks the guided setup relies on: the anti-noise lock (un-mute only at step 7)
 * and the input-2 loopback mode (a silent loopback input must not disarm the output).
 */
import { describe, expect, it } from 'vitest';
import type { AudioEngineEvent, AudioStats, LatencyMode, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import type { DeviceOption } from '../../src/audio/devices.ts';
import type { WorkletInMsg } from '../../src/audio/protocol.ts';
import { AppController } from '../../src/app/controller.ts';
import type { DeviceProvider, EngineLike } from '../../src/app/controller.ts';
import { ISSUE } from '../../src/app/validation.ts';

const dev = (deviceId: string, kind: DeviceOption['kind'], label: string, groupId: string): DeviceOption => ({
  deviceId,
  kind,
  label,
  groupId,
  pseudo: false,
});

const INPUT = dev('usb-in', 'audioinput', 'Scarlett 2i2 USB', 'g-usb');
const OUTPUT = dev('usb-out', 'audiooutput', 'Scarlett 2i2 USB', 'g-usb');

const STATS: AudioStats = {
  sampleRate: 48000,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: 48000, channelCount: 2, deviceId: 'usb-in' },
  sinkId: '',
};

class FakeEngine implements EngineLike {
  sent: WorkletInMsg[] = [];
  state: 'idle' | 'mic' | 'error' = 'idle';
  error: string | null = null;
  stats: AudioStats = { ...STATS, sampleRate: 0, outputLatencyMs: null, trackSettings: null };
  trackInfo: TrackInfo | null = null;
  sinkSupported = true;
  private listeners = new Set<(e: AudioEngineEvent) => void>();
  async openMic(): Promise<void> {
    this.state = 'mic';
    this.stats = { ...STATS };
    this.trackInfo = { deviceId: 'usb-in', groupId: 'g-usb', label: INPUT.label, settings: STATS.trackSettings!, channelCount: 2 };
  }
  async setOutput(): Promise<void> {}
  async ensureContextMatchesTrack(): Promise<boolean> {
    return false;
  }
  async setLatencyMode(_m: LatencyMode): Promise<void> {}
  async measure(s: Float32Array, _c: 0 | 1, p: number): Promise<MeasureResult> {
    return { ref: new Float32Array(s.length + p), err: null, masterLin: 1, aborted: false };
  }
  panic(): void {
    this.send({ type: 'panic' });
  }
  async resume(): Promise<void> {}
  send(msg: WorkletInMsg): void {
    this.sent.push(msg);
  }
  on(cb: (e: AudioEngineEvent) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
  emit(e: AudioEngineEvent): void {
    for (const cb of [...this.listeners]) cb(e);
  }
  muteMsgs(): boolean[] {
    return this.sent.flatMap((m) => (m.type === 'mute' ? [m.muted] : []));
  }
}

const devices: DeviceProvider = {
  list: async () => ({ inputs: [INPUT], outputs: [OUTPUT] }),
  platform: () => 'mac',
  chromium: () => true,
};

async function setup(): Promise<{ c: AppController; e: FakeEngine }> {
  const e = new FakeEngine();
  const c = new AppController({ engine: e, devices, presetStorage: null, prefsStorage: null });
  await c.openMic();
  await c.setOutput('usb-out');
  expect(c.state.validation.ok).toBe(true);
  return { c, e };
}

const LOCK = 'Completa la configurazione guidata.';

describe('anti-noise lock', () => {
  it('is off by default and shows in the state', async () => {
    const { c } = await setup();
    expect(c.state.ancLock).toBeNull();
    c.setAncLock(LOCK);
    expect(c.state.ancLock).toBe(LOCK);
    c.setAncLock(null);
    expect(c.state.ancLock).toBeNull();
  });

  it('allows arming muted (stimuli) but never un-muted', async () => {
    const { c, e } = await setup();
    c.setAncLock(LOCK);
    const r = c.arm();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('locked');
    expect(c.state.armed).toBe(false);
    expect(c.arm({ muted: true }).ok).toBe(true);
    expect(c.state.armed).toBe(true);
    expect(c.state.muted).toBe(true);
    e.sent = [];
    expect(c.setMuted(false)).toBe(false);
    c.toggleMute();
    expect(e.muteMsgs()).not.toContain(false);
    expect(c.state.muted).toBe(true);
    expect(c.state.message?.text).toBe(LOCK);
  });

  it('mutes an audible output when engaged, and un-mutes again once released', async () => {
    const { c, e } = await setup();
    expect(c.arm().ok).toBe(true);
    expect(c.state.muted).toBe(false);
    e.sent = [];
    c.setAncLock(LOCK);
    expect(e.muteMsgs()).toEqual([true]);
    expect(c.state.muted).toBe(true);
    c.setAncLock(null);
    expect(c.setMuted(false)).toBe(true);
    expect(e.muteMsgs()).toEqual([true, false]);
  });

  it('still lets a disarmed PANIC resume the audio (the output stays at zero)', async () => {
    const { c } = await setup();
    c.setAncLock(LOCK);
    c.panic();
    expect(c.state.panic).toBe(true);
    c.setMuted(false);
    expect(c.state.panic).toBe(false);
    expect(c.state.armed).toBe(false);
  });

  it('leaves a PANIC armed muted through arm({ muted: true })', async () => {
    const { c } = await setup();
    c.setAncLock(LOCK);
    c.arm({ muted: true });
    c.panic();
    expect(c.arm().ok).toBe(false);
    expect(c.state.panic).toBe(true);
    expect(c.arm({ muted: true }).ok).toBe(true);
    expect(c.state.panic).toBe(false);
    expect(c.state.muted).toBe(true);
  });
});

describe('input-2 loopback mode', () => {
  it('does not fail the validation (nor disarm) while the loopback input is silent', async () => {
    const { c, e } = await setup();
    c.setErrorChannel(1);
    e.emit({ type: 'channelCheck', status: 'ok' });
    c.arm({ muted: true });
    c.setInput2Loopback(true);
    e.emit({ type: 'channelCheck', status: 'silent' });
    expect(c.state.validation.ok).toBe(true);
    expect(c.state.armed).toBe(true);
    expect(c.state.validation.issues.some((i) => i.id === ISSUE.errorMicUnavailable)).toBe(false);
    // Back to normal: a silent input 2 for the error mic fails the validation and disarms.
    c.setInput2Loopback(false);
    expect(c.state.validation.ok).toBe(false);
    expect(c.state.armed).toBe(false);
  });
});

describe('device key seen by the wizard', () => {
  it('is the key the controller stores in presets (reuse compares the two)', async () => {
    const { c } = await setup();
    c.savePreset('Sera');
    const p = c.state.presets.list.find((x) => x.name === 'Sera');
    expect(p?.deviceKey).toMatch(/Scarlett/);
    expect(c.state.deviceKey).toBe(p!.deviceKey);
  });
});
