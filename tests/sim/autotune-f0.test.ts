/**
 * Closed-loop simulator: the tonal (per-harmonic) autotune over a run of about 100 s — the wizard's 4 s windows —
 * while the source changes under it (fs = 48 kHz, block 128).
 * A. f0 drifts 0.4 %: the wizard's driver (ops.ts) on a real AppController whose engine is the simulator
 *    (SimEngine), with the controller's live f0 follower running all along.
 * B. The noise level ramps by 6 dB: each harmonic's attenuation is referred to the bypass level at the time of
 *    the end window, not to the mean of the run's bypass windows.
 * C. The source's f0 steps mid-run (as A, the wizard's driver and the follower): within the search's tolerance
 *    the windows read the harmonics where the bank now is; beyond it the run stops and restores the start.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AppController } from '../../src/app/controller.ts';
import type { DeviceProvider } from '../../src/app/controller.ts';
import { inProcessAnalyzer } from '../../src/audio/analysis-client.ts';
import { MAX_WINDOW_MS, autotuneTiming, runAutotune } from '../../src/audio/autotune.ts';
import type { AutotuneDriver, AutotuneOptions, AutotuneResult } from '../../src/audio/autotune.ts';
import {
  createCalibration,
  defaultSweepBand,
  feedbackFirs,
  measurePath,
  measurePrimary,
  noiseBandwidthHz,
  sanitizeCalibration,
  solveFromCalibration,
} from '../../src/audio/calibration.ts';
import type { CalibrationData, MeasurePathOptions } from '../../src/audio/calibration.ts';
import { TAP_BLOCK_FRAMES } from '../../src/audio/protocol.ts';
import { clampParams } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { rt60ForAutotune } from '../../src/ui/wizard/logic.ts';
import { autotuneDriver, liveWithChannels } from '../../src/ui/wizard/ops.ts';
import { SimEngine, SimRig, toneLevelDb } from './harness.ts';
import type { PlantConfig } from './plant.ts';

const fs = 48000;
/** Short soft-start: the simulated seconds go to the measurements, not to waiting. */
const RIG = { engine: { softStartMs: 300 } };
const AMPS = [0.04, 0.025, 0.015, 0.01];
const PHASES = [0.3, 1.1, 2.0, 4.0];
const HS = [1, 2, 3, 4];

/** One speaker, tone at the reference 5 ms ahead of the zone (as closed-loop 6d). */
const plant = (noise: PlantConfig['noise']): PlantConfig => ({
  fs,
  noise,
  latencyS: 0.012,
  primary: { delayS: 0.005, gain: 0.7 },
  secondary: [{ delayS: 0.003, gain: 2, reflections: [{ delayS: 0.0025, gain: 0.15 }] }, null],
  feedback: [{ delayS: 0.004, gain: 0.3 }, null],
  sensorNoiseRms: 1e-4,
  seed: 21,
});

/** Calibration-time parameters: master 0 dB, error mic on input 2, anti-noise practically silent. */
function calibrationParams(): EngineParams {
  const p = clampParams(undefined);
  p.masterDb = 0;
  p.errorChannel = 1;
  p.channels[0].gainDb = -60;
  p.channels[1].gainDb = -60;
  p.channels[1].enabled = false;
  return p;
}

const pathOptions = (band: [number, number]): Omit<MeasurePathOptions, 'mic'> => ({
  fs,
  channel: 0,
  band,
  levelDbfs: -14,
  repetitions: 3,
  mode: 'acoustic',
  previewFirst: true,
  sweepS: 1,
});

/**
 * Calibrates on a steady 100 Hz source, solves in tonal mode and detunes the solution (+20° on harmonic 3,
 * −3 dB on harmonic 2): the autotune has something to find. Also the options the wizard passes (4 s windows).
 */
async function tonalStart(): Promise<{ cal: CalibrationData; solved: EngineParams; start: EngineParams; opts: AutotuneOptions }> {
  const rig = new SimRig(plant({ tone: { f0: 100, amplitudes: AMPS }, broadband: { rms: 0.002 } }), RIG);
  const p = calibrationParams();
  rig.start(p, 0.1);
  const band = defaultSweepBand(p, fs);
  const cal = createCalibration(fs, band, 'single');
  cal.feedback[0] = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(band), mic: 'ref' });
  cal.secondary[0] = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(band), mic: 'err' });
  cal.primary = await measurePrimary(rig.capture, inProcessAnalyzer, { fs, seconds: 2, band });
  p.phaseMode = 'tonal';
  p.tonal.f0 = cal.primary.tonalF0!;
  const solved = clampParams(solveFromCalibration(cal, p, fs).params);
  const start = clampParams(solved);
  start.channels[0].harmonicTrims[2].phaseDeg += 20;
  start.channels[0].harmonicTrims[1].gainDb -= 3;
  const eff = noiseBandwidthHz(cal) ?? undefined;
  const opts = { fs, channels: [0 as const], rt60S: rt60ForAutotune(cal), band: cal.band, ...(eff !== undefined ? { effectiveBandHz: eff } : {}) };
  expect(autotuneTiming(opts).windowMs).toBe(MAX_WINDOW_MS);
  return { cal, solved, start, opts };
}

/** Error-mic attenuation of each harmonic of `f(t)` now: anti-noise on, bypassed, on again (a level ramp cancels). */
function attenuationNow(rig: SimRig, f: (t: number) => number): number[] {
  const at = (x: Float32Array, t0: number): number[] => HS.map((h) => toneLevelDb(x, fs, h * f(t0 + 1)));
  const t1 = rig.timeS;
  const on1 = at(rig.run(2, ['err']).err!, t1);
  const t2 = rig.timeS + rig.settleS;
  const off = at(rig.bypassed(() => rig.run(2, ['err']).err!), t2);
  const t3 = rig.timeS;
  const on2 = at(rig.run(2, ['err']).err!, t3);
  return HS.map((_, i) => off[i] - (on1[i] + on2[i]) / 2);
}

afterEach(() => {
  vi.useRealTimers();
});

const devices: DeviceProvider = {
  list: async () => ({
    inputs: [{ deviceId: 'sim-in', kind: 'audioinput', label: 'Interfaccia simulata', groupId: 'sim', pseudo: false }],
    outputs: [{ deviceId: 'sim-out', kind: 'audiooutput', label: 'Interfaccia simulata', groupId: 'sim', pseudo: false }],
  }),
  platform: () => 'mac',
  chromium: () => true,
};

/** The four harmonics of a source whose fundamental's phase (cycles) is `cycles(t)`, plus broadband noise. */
const harmonicSource = (cycles: (t: number) => number): PlantConfig['noise'] => ({
  custom: (t) => {
    const phi = 2 * Math.PI * cycles(t);
    let v = 0;
    for (let h = 0; h < 4; h++) v += AMPS[h] * Math.sin((h + 1) * phi + PHASES[h]);
    return v;
  },
  broadband: { rms: 0.002 },
});

/**
 * A real AppController on the simulator (SimEngine) with the live f0 follower, calibrated with `cal` and armed on
 * `start`, then `body` with `until` (advances the rig and the controller's fake timers together).
 */
async function onController(
  rig: SimRig,
  cal: CalibrationData,
  start: EngineParams,
  body: (c: AppController, engine: SimEngine, until: (done: () => boolean, maxS: number) => Promise<void>) => Promise<void>,
): Promise<void> {
  vi.useFakeTimers();
  const engine = new SimEngine(rig);
  const c = new AppController({
    engine,
    devices,
    presetStorage: null,
    prefsStorage: null,
    now: () => rig.timeS * 1000,
    f0Analyzer: inProcessAnalyzer,
    firsFromCalibration: (raw, taps) => {
      const k = sanitizeCalibration(raw);
      return k ? feedbackFirs(k, taps) : [null, null];
    },
    // The simulated paths are flat delays: they carry the signature of a loopback cable.
    loopbackSuspects: () => [],
  });
  try {
    const TAP_S = TAP_BLOCK_FRAMES / fs;
    const until = async (done: () => boolean, maxS: number): Promise<void> => {
      for (let t = 0; !done() && t < maxS; t += TAP_S) {
        rig.run(TAP_S);
        await vi.advanceTimersByTimeAsync(TAP_S * 1000);
      }
    };
    await c.openMic();
    await c.setOutput('sim-out');
    c.applyParams(start);
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    await until(() => false, 1.5);
    expect(c.arm()).toEqual({ ok: true });
    await until(() => false, 6);
    await body(c, engine, until);
  } finally {
    c.dispose();
  }
}

/** runAutotune through the wizard's driver on `c`, the simulated time advancing until it ends. */
async function wizardRun(
  c: AppController,
  cal: CalibrationData,
  opts: AutotuneOptions,
  until: (done: () => boolean, maxS: number) => Promise<void>,
  wrap: (d: AutotuneDriver) => AutotuneDriver = (d) => d,
): Promise<AutotuneResult> {
  const p0 = c.currentParams;
  const driver = wrap(autotuneDriver(c, { fs, band: cal.band, signal: new AbortController().signal, onMeasure: () => {} }));
  const job = runAutotune(driver, p0, opts);
  let done = false;
  void job.then(() => (done = true));
  await until(() => done, 300);
  return job;
}

describe('A. tonal autotune through the wizard driver while the controller follows a drifting f0', () => {
  it('every probe keeps the f0 the follower set: no swing back to the start value, measurements in one regime', async () => {
    const { cal, start, opts } = await tonalStart();
    // The source rises by 0.004 Hz/s: 0.4 % over the ~100 s run.
    const DRIFT = 0.004;
    const fAt = (t: number): number => 100 + DRIFT * t;
    const rig = new SimRig(plant(harmonicSource((t) => 100 * t + (DRIFT * t * t) / 2)), RIG);
    await onController(rig, cal, start, async (c, engine, until) => {
      // The follower already moved f0 off the calibrated value.
      const p0 = c.currentParams;
      expect(p0.tonal.f0).not.toBe(start.tonal.f0);
      expect(Math.abs(p0.tonal.f0 / fAt(rig.timeS) - 1)).toBeLessThan(3e-4);

      const t0 = rig.timeS;
      const from = engine.sent.length;
      const r = await wizardRun(c, cal, opts, until);
      expect(r.aborted).toBe(false);
      const sent = engine.sent.slice(from).flatMap((m) => (m.msg.type === 'params' ? [{ t: m.t, p: m.msg.params }] : []));
      // ≈ 25 probes and 50 follower updates.
      expect(sent.length).toBeGreaterThan(50);
      expect(fAt(rig.timeS) / fAt(t0) - 1).toBeGreaterThan(0.0035);
      // f0 only ever follows the source: within 0.03 % of it (the follower's lag), never back down by more than
      // the estimate's own scatter. Each probe used to send the start value again (0.38 % below the source at
      // the end, 38 reversals): every window then ran half at one f0, half at the other.
      for (const m of sent) expect(Math.abs(m.p.tonal.f0 / fAt(m.t) - 1)).toBeLessThan(3e-4);
      for (let i = 1; i < sent.length; i++) expect(sent[i].p.tonal.f0 / sent[i - 1].p.tonal.f0 - 1).toBeGreaterThan(-5e-5);
      expect(sent.some((m) => m.p.tonal.f0 === p0.tonal.f0 && m.t > t0 + 10)).toBe(false);
      // The probes' trims reached the engine, f0 untouched by them: the last one is the result's.
      expect(sent[sent.length - 1].p.channels[0].harmonicTrims).toEqual(r.params.channels[0].harmonicTrims);

      // What step 6 then applies (the result's channels on the live parameters), and the real attenuation.
      c.applyParams(liveWithChannels(c, r.params), { calibrated: true });
      expect(c.state.params.tonal.f0).not.toBe(p0.tonal.f0);
      const actual = attenuationNow(rig, fAt);
      for (const a of actual) expect(a).toBeGreaterThan(15);
      // The reported figure (end window, ~6 s earlier: the drift keeps rotating the harmonics) is the real one.
      r.harmonics!.forEach((x, i) => expect(x.bypassDb - x.endDb, `harmonic ${x.h}`).toBeLessThan(actual[i] + 8));
      expect(rig.autoMuted).toBe(false);
    });
  }, 180000);
});

describe('B. tonal autotune while the noise level ramps by 6 dB', () => {
  it('each harmonic is referred to the bypass level at the end window (the mean of the run understated it by ≈ 2 dB)', async () => {
    const { cal, solved, start, opts } = await tonalStart();
    // +0.06 dB/s on the tone and the broadband noise alike: ≈ +6 dB over the ~100 s run.
    const RAMP = 0.06;
    const gain = (t: number): number => Math.pow(10, (RAMP * t) / 20);
    const rig = new SimRig(
      plant({ tone: { f0: 100, amplitudes: AMPS }, broadband: { rms: 0.002 }, toneGain: (t) => gain(t), broadbandGain: gain }),
      RIG,
    );
    rig.send({ type: 'arm', armed: false });
    const [fir] = feedbackFirs(cal, 256);
    rig.send({ type: 'feedbackFir', channel: 0, bulkDelay: fir?.bulkDelay ?? 1, taps: fir?.taps ?? null });
    rig.start(start, 0.3);
    const driver = rig.autotuneDriver(cal.band);
    const res = await runAutotune(driver, start, opts);
    expect(res.aborted).toBe(false);
    const windows = driver.log.harmonics;
    const byp = windows.filter((w) => w.bypass);
    expect(byp[byp.length - 1].levels[0] - byp[0].levels[0]).toBeGreaterThan(5);
    // The final window is followed by a bypass window: its reference is interpolated, never extrapolated.
    const close = windows[windows.length - 1];
    const end = windows[windows.length - 2];
    expect(close.bypass && !end.bypass).toBe(true);
    res.harmonics!.forEach((x, i) => {
      expect(x.endDb).toBe(end.levels[i]);
      // The bypassed level at the end window's time: the closing bypass window, back by the ramp between them.
      const truth = close.levels[i] - RAMP * (close.t - end.t);
      // The reference before: the power mean of every bypass window of the run.
      const mean = 10 * Math.log10(byp.reduce((a, w) => a + Math.pow(10, w.levels[i] / 10), 0) / byp.length);
      expect(truth - mean, `harmonic ${x.h}`).toBeGreaterThan(1.5);
      expect(Math.abs(x.bypassDb - truth), `harmonic ${x.h}`).toBeLessThan(0.25);
      // The start against the level at its own time (the first bypass window, ≈ 5 s earlier).
      expect(Math.abs(x.startBypassDb! - (byp[0].levels[i] + RAMP * (windows[1].t - byp[0].t))), `harmonic ${x.h}`).toBeLessThan(0.25);
    });
    // Deep cancellation: the figures are lower bounds (noise-limited windows), checked against the rig as such.
    const actual = attenuationNow(rig, () => 100);
    for (const [i, x] of res.harmonics!.entries()) {
      expect(x.bypassDb - x.endDb).toBeGreaterThan(30);
      expect(actual[i]).toBeGreaterThan(30);
    }
    expect(Math.abs(res.params.channels[0].harmonicTrims[2].phaseDeg - solved.channels[0].harmonicTrims[2].phaseDeg)).toBeLessThan(5);
    expect(rig.autoMuted).toBe(false);
  }, 120000);
});

describe('C. tonal autotune while the source steps to another f0 (the wizard driver, the follower on)', () => {
  /** A source at 100 Hz that steps to 100·k at STEP_T s (phase-continuous). */
  const STEP_T = 30;
  const stepped = (k: number) => (t: number): number => (t < STEP_T ? 100 * t : 100 * STEP_T + 100 * k * (t - STEP_T));

  it('+0.8 % (inside the search\'s tolerance): every window reads the harmonics where the bank now is', async () => {
    const { cal, start, opts } = await tonalStart();
    const K = 1.008;
    const rig = new SimRig(plant(harmonicSource(stepped(K))), RIG);
    await onController(rig, cal, start, async (c, _engine, until) => {
      // True bypass level of each harmonic at the error mic (primary 0.7, flat), dBFS RMS.
      const trueBypass = AMPS.map((a) => 20 * Math.log10((0.7 * a) / Math.SQRT2));
      const bypassReads: { t: number; levels: number[] }[] = [];
      const r = await wizardRun(c, cal, opts, until, (d) => ({
        ...d,
        measureHarmonicsDb: async (ms, freqs, bypass, tol) => {
          const t = rig.timeS;
          const v = await d.measureHarmonicsDb!(ms, freqs, bypass, tol);
          if (bypass) bypassReads.push({ t, levels: v });
          return v;
        },
      }));
      expect(r.aborted).toBe(false);
      expect(c.currentParams.tonal.f0 / 100).toBeCloseTo(K, 3);
      // Bypass windows after the step (and after the follower's next estimate) read the tone, not the noise floor
      // 40–50 dB under it (the lock-in stayed at the start f0 while the bank followed the source).
      const after = bypassReads.filter((w) => w.t > STEP_T + 4);
      expect(after.length).toBeGreaterThan(0);
      for (const w of after) w.levels.forEach((db, i) => expect(Math.abs(db - trueBypass[i]), `harmonic ${i + 1} at ${w.t.toFixed(0)} s`).toBeLessThan(1.5));
      expect(rig.autoMuted).toBe(false);
    });
  }, 180000);

  it('+2 % (beyond it): the run stops and restores the start values over the live f0, never a blind completed run', async () => {
    const { cal, start, opts } = await tonalStart();
    const K = 1.02;
    const rig = new SimRig(plant(harmonicSource(stepped(K))), RIG);
    await onController(rig, cal, start, async (c, engine, until) => {
      const p0 = c.currentParams;
      const from = engine.sent.length;
      const r = await wizardRun(c, cal, opts, until);
      // It used to complete, the second half reading the noise floor: «da −47,1 a −84,6 dB … valori migliorati».
      expect(r.aborted).toBe(true);
      expect(r.improved).toBe(false);
      expect(r.reason).toMatch(/^Autotune interrotto \(la fondamentale si è spostata da 100,\d\d a 102,\d\d Hz durante la misura/);
      expect(rig.timeS).toBeLessThan(STEP_T + 15);
      // Restored: the start's trims, at the f0 the follower holds (the source's), not the start's.
      const last = engine.sent.slice(from).filter((m) => m.msg.type === 'params').pop()!;
      const p = (last.msg as { params: EngineParams }).params;
      expect(p.channels[0].harmonicTrims).toEqual(p0.channels[0].harmonicTrims);
      expect(p.tonal.f0 / 100).toBeCloseTo(K, 3);
      expect(rig.autoMuted).toBe(false);
    });
  }, 180000);
});
