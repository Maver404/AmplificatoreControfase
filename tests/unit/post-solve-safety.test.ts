/**
 * What the solve guaranteed must still hold when things change after it. The loop-gain stability margin
 * (neutralised loop ≤ LOOP_GAIN_DANGER_DB) for the step-6 autotune (v-safety S1) and for a shorter
 * feedback neutraliser or a loaded preset (v-safety S4, calib-ui-3); the reference input the paths were
 * measured with (v-safety S2, controller side; the wizard gates are in wizard-flow.test.ts).
 */
import { describe, expect, it } from 'vitest';
import type { AudioEngineEvent, AudioStats, LatencyMode, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import { runAutotune } from '../../src/audio/autotune.ts';
import type { AutotuneDriver } from '../../src/audio/autotune.ts';
import {
  calibrationFreqs,
  capLoopGain,
  checkLoop,
  createCalibration,
  feedbackFirs,
  LOOP_GAIN_DANGER_DB,
  pathResponse,
  sanitizeCalibration,
} from '../../src/audio/calibration.ts';
import type { CalibrationData, PathData } from '../../src/audio/calibration.ts';
import type { DeviceOption } from '../../src/audio/devices.ts';
import type { WorkletInMsg } from '../../src/audio/protocol.ts';
import { AppController } from '../../src/app/controller.ts';
import type { DeviceProvider, EngineLike } from '../../src/app/controller.ts';
import { clampParams } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';
import { loopGuard, mergedZonesLoop } from '../../src/ui/wizard/logic.ts';

const FS = 48000;
const BAND: [number, number] = [20, 1000];

/** Speaker → reference path with an echo too late for the neutraliser (beyond 1024 taps). */
function echoPath(lag: number, echoLag: number, gain: number): PathData {
  const ir = new Array<number>(lag + echoLag + 64).fill(0);
  ir[lag] = gain;
  ir[lag + echoLag] = gain;
  const p: PathData = { delaySamples: lag, sign: 1, qualityDb: 40, stdSamples: 0.05, freqs: [], re: [], im: [], ir, irOffset: 0 };
  const r = pathResponse(p, FS, calibrationFreqs(FS, BAND));
  return { ...p, freqs: Array.from(calibrationFreqs(FS, BAND)), re: Array.from(r.re), im: Array.from(r.im) };
}

/**
 * Error mic of a pure gain error: every enabled channel cancels exactly at `idealDb`, so the error keeps
 * falling as a capped gain rises towards it. Records every parameter set played.
 */
function gainErrorDriver(idealDb: number): AutotuneDriver & { played: EngineParams[] } {
  let last: EngineParams | null = null;
  const played: EngineParams[] = [];
  return {
    played,
    apply(p) {
      last = p;
      played.push(p);
    },
    settle: async () => {},
    async measureErrorDb() {
      let pow = 1e-4;
      for (const ch of [0, 1] as const) {
        if (!last!.channels[ch].enabled) continue;
        const g = Math.pow(10, (last!.channels[ch].gainDb - idealDb) / 20);
        pow += 0.5 * (1 - g) * (1 - g);
      }
      return 10 * Math.log10(pow);
    },
    measureBypassDb: async () => 0,
  };
}

describe('step-6 autotune keeps the loop-gain margin of the solve (v-safety S1)', () => {
  it('separate zones: gains capped by mergedZonesLoop never climb back over the cap, not even in a probe', async () => {
    const cal: CalibrationData = createCalibration(FS, BAND, 'separate');
    cal.feedback = [echoPath(900, 3000, 0.4), echoPath(900, 3000, 0.4)];
    const p = clampParams(undefined);
    p.masterDb = 0;
    p.channels[0].gainDb = 0;
    p.channels[1].gainDb = 0;
    const capped = mergedZonesLoop(cal, p, FS, 256, []);
    expect(capped.limitedDb).toBeGreaterThan(3);
    const drv = gainErrorDriver(0);
    // Without the guard the search raises gain L by ≈ 4 dB (the regression).
    const free = await runAutotune(gainErrorDriver(0), capped.params, { fs: FS, channels: [0], rt60S: 0.1, band: BAND });
    expect(checkLoop(cal, free.params, FS, 256)!.afterDb).toBeGreaterThan(LOOP_GAIN_DANGER_DB + 1);
    const r = await runAutotune(drv, capped.params, {
      fs: FS,
      channels: [0],
      rt60S: 0.1,
      band: BAND,
      allowed: loopGuard(cal, capped.params, FS, 256),
    });
    expect(r.aborted).toBe(false);
    for (const q of drv.played) expect(checkLoop(cal, q, FS, 256)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 0.05);
    expect(checkLoop(cal, r.params, FS, 256)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 0.05);
    // What step-align applies: the result, capped once more.
    expect(checkLoop(cal, capLoopGain(cal, r.params, FS, 256).params, FS, 256)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 1e-9);
  });

  it('one zone: a gain lowered by the finishSolve cap stays there; lowering it further is still allowed', async () => {
    const cal: CalibrationData = createCalibration(FS, BAND, 'single');
    cal.feedback = [echoPath(900, 3000, 2.5), null];
    const p = clampParams(undefined);
    p.masterDb = 0;
    p.channels[0].gainDb = -6;
    p.channels[1].enabled = false;
    const capped = capLoopGain(cal, p, FS, 256);
    expect(capped.limitedDb).toBeGreaterThan(5);
    expect(capped.loop!.afterDb).toBeCloseTo(LOOP_GAIN_DANGER_DB, 6);
    const guard = loopGuard(cal, capped.params, FS, 256);
    expect(guard(capped.params)).toBe(true);
    const lower = clampParams({ ...capped.params, channels: [{ ...capped.params.channels[0], gainDb: capped.params.channels[0].gainDb - 1 }, capped.params.channels[1]] });
    const higher = clampParams({ ...capped.params, channels: [{ ...capped.params.channels[0], gainDb: capped.params.channels[0].gainDb + 1 }, capped.params.channels[1]] });
    expect(guard(lower)).toBe(true);
    expect(guard(higher)).toBe(false);
    const drv = gainErrorDriver(-6);
    const r = await runAutotune(drv, capped.params, { fs: FS, channels: [0], rt60S: 0.1, band: BAND, allowed: guard });
    expect(r.params.channels[0].gainDb).toBeLessThanOrEqual(capped.params.channels[0].gainDb + 1e-9);
    for (const q of drv.played) expect(q.channels[0].gainDb).toBeLessThanOrEqual(capped.params.channels[0].gainDb + 0.01);
  });

  it('a cap lands at the margin, never a rounding error above it; a loop left at the margin is not capped again', () => {
    const cal: CalibrationData = createCalibration(FS, BAND, 'single');
    cal.feedback = [echoPath(900, 3000, 2.5), null];
    for (const g0 of [-6, -3.3, -1.7, 0]) {
      const p = clampParams(undefined);
      p.masterDb = 0;
      p.channels[0].gainDb = g0;
      p.channels[1].enabled = false;
      const c = capLoopGain(cal, p, FS, 256);
      expect(c.limitedDb).toBeGreaterThan(0);
      expect(c.loop!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB);
      expect(c.loop!.afterDb).toBeCloseTo(LOOP_GAIN_DANGER_DB, 6);
      expect(capLoopGain(cal, c.params, FS, 256).limitedDb).toBe(0);
      // Gains lowered by exactly the excess, as finishSolve does: within the margin, nothing to take off.
      const excess = checkLoop(cal, p, FS, 256)!.afterDb - LOOP_GAIN_DANGER_DB;
      const solved = clampParams({ ...p, channels: [{ ...p.channels[0], gainDb: g0 - excess }, p.channels[1]] });
      expect(capLoopGain(cal, solved, FS, 256).limitedDb).toBe(0);
    }
  });

  it('without feedback data there is nothing to limit', () => {
    const cal = createCalibration(FS, BAND, 'single');
    const p = clampParams(undefined);
    expect(loopGuard(cal, p, FS, 256)({ ...p, masterDb: 6 })).toBe(true);
    const c = capLoopGain(cal, p, FS, 256);
    expect(c.params).toBe(p);
    expect(c.limitedDb).toBe(0);
    expect(c.loop).toBeNull();
  });
});

// ------------------------------------------------------------------------------- controller

const STATS: AudioStats = {
  sampleRate: FS,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: FS, channelCount: 2, deviceId: 'usb-in' },
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
    this.trackInfo = { deviceId: 'usb-in', groupId: 'g', label: 'Scarlett 2i2 USB', settings: STATS.trackSettings!, channelCount: 2 };
  }
  async setOutput(): Promise<void> {}
  async ensureContextMatchesTrack(): Promise<boolean> {
    return false;
  }
  async setLatencyMode(_m: LatencyMode): Promise<void> {}
  async measure(s: Float32Array, _c: 0 | 1, p: number): Promise<MeasureResult> {
    return { ref: new Float32Array(s.length + p), err: null, masterLin: 1, aborted: false };
  }
  panic(): void {}
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
}

const dev = (deviceId: string, kind: DeviceOption['kind']): DeviceOption => ({ deviceId, kind, label: 'Scarlett 2i2 USB', groupId: 'g', pseudo: false });
const devices: DeviceProvider = {
  list: async () => ({ inputs: [dev('usb-in', 'audioinput')], outputs: [dev('usb-out', 'audiooutput')] }),
  platform: () => 'mac',
  chromium: () => true,
};

async function controller(): Promise<{ c: AppController; e: FakeEngine }> {
  const e = new FakeEngine();
  const c = new AppController({
    engine: e,
    devices,
    presetStorage: null,
    prefsStorage: null,
    firsFromCalibration: (raw, taps) => {
      const x = sanitizeCalibration(raw);
      return x ? feedbackFirs(x, taps) : [null, null];
    },
  });
  await c.openMic();
  await c.setOutput('usb-out');
  e.emit({ type: 'ready', sampleRate: FS } as AudioEngineEvent);
  c.setSpeakers('single');
  return { c, e };
}

/** Speaker → reference path: direct sound plus a 100 ms diffuse tail that only long FIRs model. */
function roomPath(lag: number, gain: number): PathData {
  const n = lag + Math.round(0.1 * FS);
  const ir = new Array<number>(n).fill(0);
  const rnd = mulberry32(3);
  ir[lag] = gain;
  const tau = 0.1 / 6.9;
  for (let i = lag + 20; i < n; i++) ir[i] = gain * 0.08 * (2 * rnd() - 1) * Math.exp(-(i - lag) / FS / tau);
  const p: PathData = { delaySamples: lag, sign: 1, qualityDb: 40, stdSamples: 0.05, freqs: [], re: [], im: [], ir, irOffset: 0 };
  const r = pathResponse(p, FS, calibrationFreqs(FS, BAND));
  return { ...p, freqs: Array.from(calibrationFreqs(FS, BAND)), re: Array.from(r.re), im: Array.from(r.im) };
}

function roomCase(): { cal: CalibrationData; p: EngineParams } {
  const cal = createCalibration(FS, BAND, 'single');
  cal.feedback = [roomPath(900, 1), null];
  const p = clampParams(undefined);
  p.masterDb = 0;
  p.channels[0].gainDb = 0;
  p.channels[1].enabled = false;
  // Solved with 1024 taps: within the margin. With 128 taps the same gains are unstable.
  expect(checkLoop(cal, p, FS, 1024)!.afterDb).toBeLessThan(LOOP_GAIN_DANGER_DB);
  expect(checkLoop(cal, p, FS, 128)!.afterDb).toBeGreaterThan(0);
  return { cal, p };
}

describe('a shorter neutraliser or a loaded preset keeps the loop-gain margin (v-safety S4, calib-ui-3)', () => {
  it('lowering the taps on an audible output lowers the channel gains before the shorter FIRs are sent', async () => {
    const { cal, p } = roomCase();
    const { c, e } = await controller();
    c.setFeedbackTaps(1024);
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    c.applyParams(p, { calibrated: true });
    expect(c.arm().ok).toBe(true);
    expect(c.state.muted).toBe(false);
    e.sent = [];
    c.setFeedbackTaps(128);
    const s = c.state;
    expect(s.feedbackTaps).toBe(128);
    expect(checkLoop(cal, s.params, FS, 128)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB);
    expect(checkLoop(cal, c.currentParams, FS, 128)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB);
    expect(s.params.channels[0].gainDb).toBeLessThan(-10);
    // The double-click value no longer undoes it.
    expect(s.calibratedChannels![0].gainDb).toBeCloseTo(s.params.channels[0].gainDb, 9);
    expect(s.message?.level).toBe('warn');
    expect(s.message?.text).toMatch(/^Neutralizzatore a 128 tap: .*guadagni dei canali ridotti di \d+,\d dB/);
    // Order: the lower gains reach the engine before the shorter FIR.
    const types = e.sent.map((m) => m.type);
    expect(types.indexOf('params')).toBeGreaterThanOrEqual(0);
    expect(types.indexOf('params')).toBeLessThan(types.indexOf('feedbackFir'));
    // Longer again: nothing changes.
    const before = c.state.params.channels[0].gainDb;
    c.setFeedbackTaps(1024);
    expect(c.state.params.channels[0].gainDb).toBe(before);
  });

  it('a preset saved after the reduction reloads with the same taps without another one', async () => {
    const { cal, p } = roomCase();
    const { c } = await controller();
    c.setFeedbackTaps(1024);
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    c.applyParams(p, { calibrated: true });
    c.setFeedbackTaps(128);
    const lowered = c.state.params.channels[0].gainDb;
    c.savePreset('Ridotto');
    const id = c.state.presets.activeId!;
    c.loadPreset(id);
    expect(c.state.params.channels[0].gainDb).toBe(lowered);
    expect(c.state.presets.message?.level).toBe('info');
    expect(c.state.presets.message?.text).not.toMatch(/ridotti/);
  });

  it('the per-harmonic attenuation that came with the lowered gains is dropped (R-L1)', async () => {
    const { cal, p } = roomCase();
    const { c } = await controller();
    c.setFeedbackTaps(1024);
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    c.applyParams(p, { calibrated: true, harmonics: [{ h: 1, freqHz: 100, attenuationDb: 24, source: 'measured' }] });
    expect(c.state.calibratedHarmonics).toHaveLength(1);
    c.setFeedbackTaps(128);
    expect(c.state.params.channels[0].gainDb).toBeLessThan(p.channels[0].gainDb - 1);
    // 24 dB was measured with gains ≈ 12 dB higher: step 7 must not expect it.
    expect(c.state.calibratedHarmonics).toBeNull();
    // Without a reduction it stays.
    c.applyParams(c.currentParams, { calibrated: true, harmonics: [{ h: 1, freqHz: 100, attenuationDb: 9, source: 'measured' }] });
    c.setFeedbackTaps(1024);
    expect(c.state.calibratedHarmonics).toHaveLength(1);
  });

  it('a gain within the margin for the new length is left alone', async () => {
    const { cal, p } = roomCase();
    const { c } = await controller();
    c.setFeedbackTaps(1024);
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    const low = clampParams({ ...p, channels: [{ ...p.channels[0], gainDb: -20 }, p.channels[1]] });
    expect(checkLoop(cal, low, FS, 128)!.afterDb).toBeLessThan(LOOP_GAIN_DANGER_DB);
    c.applyParams(low, { calibrated: true });
    c.setFeedbackTaps(128);
    expect(c.state.params.channels[0].gainDb).toBe(-20);
    expect(c.state.message).toBeNull();
  });

  it('a preset solved with more taps than the ones in use loads with the gains lowered, and says so', async () => {
    const { cal, p } = roomCase();
    const { c } = await controller();
    c.setFeedbackTaps(1024);
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    c.applyParams(p, { calibrated: true });
    c.savePreset('Sera');
    const id = c.state.presets.activeId!;
    c.setCalibration(null);
    c.setFeedbackTaps(256);
    c.loadPreset(id);
    const s = c.state;
    expect(checkLoop(cal, s.params, FS, 256)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 1e-6);
    expect(s.params.channels[0].gainDb).toBeLessThan(-5);
    expect(s.calibratedChannels![0].gainDb).toBe(s.params.channels[0].gainDb);
    expect(s.presets.message?.level).toBe('warn');
    expect(s.presets.message?.text).toMatch(/Con 256 tap del neutralizzatore il guadagno d'anello/);
  });
});

describe('the calibration records its reference input (v-safety S2)', () => {
  it('survives sanitising (saved with presets) and is optional', () => {
    const cal = createCalibration(FS, BAND, 'single');
    expect(sanitizeCalibration(JSON.parse(JSON.stringify(cal)))!.refChannel).toBeUndefined();
    cal.refChannel = 1;
    expect(sanitizeCalibration(JSON.parse(JSON.stringify(cal)))!.refChannel).toBe(1);
    expect(sanitizeCalibration({ ...JSON.parse(JSON.stringify(cal)), refChannel: 7 })!.refChannel).toBeUndefined();
  });

  it('the controller applies the feedback FIRs only on the input the paths were measured with', async () => {
    const { cal } = roomCase();
    cal.refChannel = 0;
    const { c } = await controller();
    c.setRefChannel(1);
    // Set while the reference is on input 2: the record says input 1, so no neutraliser.
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    expect(c.state.feedbackFirActive).toEqual([false, false]);
    c.setRefChannel(0);
    expect(c.state.feedbackFirActive[0]).toBe(true);
  });
});
