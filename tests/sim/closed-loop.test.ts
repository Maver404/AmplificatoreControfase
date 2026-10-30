/**
 * Closed-loop simulator: the real calibration, alignment, autotune and AntiPhaseEngine code against a
 * simulated room (tests/sim/plant.ts), fs = 48 kHz, block 128.
 */
import { describe, expect, it } from 'vitest';
import { inProcessAnalyzer } from '../../src/audio/analysis-client.ts';
import { runAutotune } from '../../src/audio/autotune.ts';
import {
  bandRmsDbfs,
  createCalibration,
  defaultSweepBand,
  feedbackFirs,
  looksLikeLoopback,
  measurePath,
  measurePrimary,
  solveFromCalibration,
  solveFromGeometry,
} from '../../src/audio/calibration.ts';
import type { CalibrationData, MeasurePathOptions, SolveOutcome } from '../../src/audio/calibration.ts';
import type { SpeakerConfig } from '../../src/dsp/alignment.ts';
import { clampParams, dbToLin } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { headStartSeconds, speedOfSound } from '../../src/dsp/physics.ts';
import { SimRig, bandLevelDb, toneLevelDb } from './harness.ts';
import type { PlantConfig } from './plant.ts';

const fs = 48000;
const BAND: [number, number] = [80, 400];
/** Short soft-start: the simulated seconds go to the measurements, not to waiting. */
const RIG = { engine: { softStartMs: 300 } };

/** Calibration-time parameters: master 0 dB, error mic on input 2, anti-noise practically silent. */
function calibrationParams(speakers: SpeakerConfig): EngineParams {
  const p = clampParams(undefined);
  p.masterDb = 0;
  p.errorChannel = 1;
  p.channels[0].gainDb = -60;
  p.channels[1].gainDb = -60;
  p.channels[1].enabled = speakers !== 'single';
  return p;
}

const pathOptions = (ch: 0 | 1, band: [number, number]): Omit<MeasurePathOptions, 'mic'> => ({
  fs,
  channel: ch,
  band,
  levelDbfs: -14,
  repetitions: 3,
  mode: 'acoustic',
  previewFirst: true,
  sweepS: 1,
});

/** measurePath(feedback & secondary) per channel + measurePrimary, all through the simulated engine. */
async function calibrate(rig: SimRig, p: EngineParams, speakers: SpeakerConfig): Promise<CalibrationData> {
  rig.start(p, 0.1);
  const band = defaultSweepBand(p, fs);
  const cal = createCalibration(fs, band, speakers);
  for (const ch of speakers === 'single' ? ([0] as const) : ([0, 1] as const)) {
    const fb = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(ch, band), mic: 'ref' });
    const sec = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(ch, band), mic: 'err' });
    expect(fb.reason).toBeUndefined();
    expect(sec.reason).toBeUndefined();
    cal.feedback[ch] = fb;
    cal.secondary[ch] = sec;
  }
  cal.primary = await measurePrimary(rig.capture, inProcessAnalyzer, { fs, seconds: 2, band });
  return cal;
}

/** Disarms, loads the solution and the feedback FIRs (if any), re-arms and settles. */
function apply(rig: SimRig, out: SolveOutcome, cal: CalibrationData | null): void {
  rig.send({ type: 'arm', armed: false });
  const firs = cal ? feedbackFirs(cal, 256) : [null, null];
  for (const ch of [0, 1] as const) {
    const f = firs[ch];
    rig.send({ type: 'feedbackFir', channel: ch, bulkDelay: f?.bulkDelay ?? 1, taps: f?.taps ?? null });
  }
  rig.start(out.params, 0.3);
}

/** Error-mic level drop (dB) in `band`, anti-noise on vs bypassed. */
function attenuationDb(rig: SimRig, band: [number, number] = BAND, seconds = 2): number {
  const on = rig.run(seconds, ['err']).err!;
  const off = rig.bypassed(() => rig.run(seconds, ['err']).err!);
  return bandLevelDb(off, fs, band[0], band[1]) - bandLevelDb(on, fs, band[0], band[1]);
}

/** Layout A, broadband: noise 25 ms ahead at the zone, L_e 12 ms, speaker 3 ms from the error mic. */
const causalBroadband = (over: Partial<PlantConfig> = {}): PlantConfig => ({
  fs,
  noise: { broadband: { rms: 0.02, lo: 20, hi: 1000 } },
  latencyS: 0.012,
  primary: { delayS: 0.025, gain: 0.7 },
  secondary: [{ delayS: 0.003, gain: 2, reflections: [{ delayS: 0.0025, gain: 0.15 }] }, null],
  feedback: [{ delayS: 0.004, gain: 0.3 }, null],
  sensorNoiseRms: 1e-4,
  seed: 3,
  ...over,
});

describe('closed loop: calibration → alignment → engine', () => {
  it('1. causal broadband: the calibrated solution cancels ≥ 10 dB in 80–400 Hz', async () => {
    const rig = new SimRig(causalBroadband(), RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 'single');
    // Paths as simulated: S = L_e + 3 ms = 720 samples, F = L_e + 4 ms = 768 samples.
    expect(cal.secondary[0]!.delaySamples).toBeCloseTo(720, 0);
    expect(cal.secondary[0]!.sign).toBe(1);
    expect(cal.feedback[0]!.delaySamples).toBeCloseTo(768, 0);
    // A dry simulated room: the decay is over within a few tens of ms.
    expect(cal.secondary[0]!.rt60S ?? 0).toBeLessThan(0.1);
    expect(cal.primary!.tonalF0).toBeNull();
    const out = solveFromCalibration(cal, p, fs);
    expect(out.result.causal).toBe(true);
    expect(out.params.channels[0].enabled).toBe(true);
    expect(out.params.channels[0].polarityInvert).toBe(true);
    expect(out.params.channels[1].enabled).toBe(false);
    // P 25 ms − S 15 ms − conditioning group delay (~0.4 ms).
    expect(out.params.channels[0].delayMs).toBeGreaterThan(9.3);
    expect(out.params.channels[0].delayMs).toBeLessThan(10);
    expect(out.loopGainDb).not.toBeNull();
    apply(rig, out, cal);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(10);
    expect(rig.autoMuted).toBe(false);
  }, 30000);

  it('2. tonal noise with a non-causal geometry: tonal mode with f0 from estimateF0 cancels the tone ≥ 15 dB', async () => {
    const cfg = causalBroadband({
      noise: { tone: { f0: 100, amplitudes: [0.04, 0.025, 0.015, 0.01] }, broadband: { rms: 0.002 } },
      primary: { delayS: 0.005, gain: 0.7 },
    });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 'single');
    const f0 = cal.primary!.tonalF0!;
    expect(Math.abs(f0 - 100)).toBeLessThan(0.01);
    // Delay mode cannot be causal for broadband noise here; the advice points to the tonal mode.
    const asDelay = solveFromCalibration(cal, p, fs);
    expect(asDelay.advice.some((a) => a.includes('modalità tonale'))).toBe(true);
    p.phaseMode = 'tonal';
    p.tonal.f0 = f0;
    const out = solveFromCalibration(cal, p, fs);
    expect(out.result.causal).toBe(true);
    // Whole periods are free: the tap delay stays within one period above D_MIN.
    expect(out.params.channels[0].delayMs).toBeLessThan(10);
    apply(rig, out, cal);
    const on = rig.run(2, ['err']).err!;
    const off = rig.bypassed(() => rig.run(2, ['err']).err!);
    const drop = (f: number): number => toneLevelDb(off, fs, f) - toneLevelDb(on, fs, f);
    expect(drop(100)).toBeGreaterThanOrEqual(15);
    const tonalPower = (x: Float32Array): number =>
      [100, 200, 300, 400].reduce((s, f) => s + Math.pow(10, toneLevelDb(x, fs, f) / 10), 0);
    expect(10 * Math.log10(tonalPower(off) / tonalPower(on))).toBeGreaterThanOrEqual(15);
  }, 30000);

  it('3. speaker wired backwards: the solution flips the polarity and still cancels', async () => {
    const cfg = causalBroadband();
    cfg.secondary[0] = { ...cfg.secondary[0]!, invert: true };
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 'single');
    expect(cal.secondary[0]!.sign).toBe(-1);
    const out = solveFromCalibration(cal, p, fs);
    expect(out.params.channels[0].polarityInvert).toBe(false);
    apply(rig, out, cal);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(10);
  }, 30000);

  it("4. two speakers on one zone: the joint 'shared' solution cancels, two independent ones do not", async () => {
    const cfg = causalBroadband({
      secondary: [
        { delayS: 0.003, gain: 2 },
        { delayS: 0.0042, gain: 1.6 },
      ],
      feedback: [
        { delayS: 0.004, gain: 0.4 },
        { delayS: 0.0035, gain: 0.4 },
      ],
    });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('shared');
    const cal = await calibrate(rig, p, 'shared');
    const joint = solveFromCalibration(cal, p, fs);
    expect(joint.params.channels[0].enabled && joint.params.channels[1].enabled).toBe(true);
    apply(rig, joint, cal);
    const jointDb = attenuationDb(rig);
    expect(jointDb).toBeGreaterThanOrEqual(10);
    // Each speaker solved alone against the whole noise: together they cancel it twice over.
    const separate = solveFromCalibration({ ...cal, speakers: 'separate' }, p, fs);
    apply(rig, separate, cal);
    const separateDb = attenuationDb(rig);
    expect(separateDb).toBeLessThan(3);
    expect(jointDb - separateDb).toBeGreaterThan(10);
  }, 30000);

  it('5. strong feedback: stable with the neutralisation FIRs, auto-muted (never above the ceiling) without', async () => {
    const cfg = causalBroadband({
      primary: { delayS: 0.025, gain: 1 },
      secondary: [{ delayS: 0.003, gain: 1.4, reflections: [{ delayS: 0.0025, gain: 0.1 }] }, null],
      feedback: [{ delayS: 0.002, gain: 2 }, null],
      seed: 5,
    });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 'single');
    const out = solveFromCalibration(cal, p, fs);
    // The open loop without neutralisation is above 0 dB: it would howl.
    expect(out.loopGainDb!).toBeGreaterThan(0);
    expect(out.advice.some((a) => a.includes("Guadagno d'anello") || a.includes('Rischio di innesco'))).toBe(true);
    apply(rig, out, cal);
    rig.run(2);
    expect(rig.autoMuted).toBe(false);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(10);
    expect(rig.autoMuted).toBe(false);

    const bare = new SimRig(cfg, RIG);
    bare.start(out.params, 0);
    const ceiling = dbToLin(out.params.ceilingDb);
    for (let t = 0; t < 4 && !bare.autoMuted; t += 0.1) bare.run(0.1);
    expect(bare.autoMuted).toBe(true);
    expect(bare.events.some((e) => e.type === 'howl')).toBe(true);
    expect(bare.maxAbsOut).toBeLessThanOrEqual(ceiling);
    // Muted for good: the output decays to silence and stays there.
    const tail = bare.run(0.5, ['outL']).outL!;
    expect(Math.max(...tail.subarray(tail.length >> 1).map(Math.abs))).toBe(0);
    expect(bare.maxAbsOut).toBeLessThanOrEqual(ceiling);
  }, 30000);

  describe('6. autotune', () => {
    const cfg = causalBroadband({ seed: 11 });
    let solved: { out: SolveOutcome; cal: CalibrationData } | null = null;
    /** A fresh rig of the same room running the calibrated solution detuned by +0.4 ms and −4 dB. */
    async function detunedRig(): Promise<{ rig: SimRig; start: EngineParams }> {
      if (solved === null) {
        const p = calibrationParams('single');
        const cal = await calibrate(new SimRig(cfg, RIG), p, 'single');
        solved = { out: solveFromCalibration(cal, p, fs), cal };
      }
      const start = clampParams(solved.out.params);
      start.channels[0].delayMs += 0.4;
      start.channels[0].gainDb -= 4;
      const rig = new SimRig(cfg, RIG);
      apply(rig, { ...solved.out, params: start }, solved.cal);
      return { rig, start };
    }
    const opts = { fs, channels: [0 as const], rt60S: 0.03, band: BAND };

    it('improves a detuned start by ≥ 6 dB and is never worse than bypass + 1 dB', async () => {
      const { rig, start } = await detunedRig();
      const driver = rig.autotuneDriver(BAND);
      const res = await runAutotune(driver, start, opts);
      expect(res.aborted).toBe(false);
      expect(res.startDb - res.endDb).toBeGreaterThanOrEqual(6);
      const bypass = Math.max(...driver.log.bypass);
      for (const e of driver.log.errors) expect(e).toBeLessThanOrEqual(bypass + 1);
      // The result moved back towards the calibrated delay and gain.
      expect(Math.abs(res.params.channels[0].delayMs - (start.channels[0].delayMs - 0.4))).toBeLessThan(0.15);
      expect(res.params.channels[0].gainDb).toBeGreaterThan(start.channels[0].gainDb + 2);
      // Independent check on the rig with the final parameters.
      expect(attenuationDb(rig)).toBeGreaterThanOrEqual(10);
    }, 30000);

    it('restores the start parameters when aborted', async () => {
      const { rig, start } = await detunedRig();
      const ctl = new AbortController();
      const driver = rig.autotuneDriver(BAND, ctl.signal);
      const measure = driver.measureErrorDb;
      driver.measureErrorDb = async (ms) => {
        if (driver.log.errors.length === 4) ctl.abort();
        return measure(ms);
      };
      const res = await runAutotune(driver, start, opts);
      expect(res.aborted).toBe(true);
      expect(res.reason).toMatch(/annullato/);
      expect(res.params).toEqual(clampParams(start));
      const last = driver.log.applied[driver.log.applied.length - 1];
      expect(last.jump).toBe(true);
      expect(last.p).toEqual(clampParams(start));
      expect(rig.currentParams).toEqual(clampParams(start));
    }, 30000);

    it('aborts and restores when the error exceeds the bypass level by more than 1 dB', async () => {
      const { rig, start } = await detunedRig();
      const driver = rig.autotuneDriver(BAND);
      const measure = driver.measureErrorDb;
      // A disturbance mid-search: one window comes back 3 dB above bypass.
      driver.measureErrorDb = async (ms) => {
        const db = await measure(ms);
        return driver.log.errors.length === 5 ? Math.max(...driver.log.bypass) + 3 : db;
      };
      const res = await runAutotune(driver, start, opts);
      expect(res.aborted).toBe(true);
      expect(res.reason).toMatch(/bypass/);
      expect(rig.currentParams).toEqual(clampParams(start));
    }, 30000);
  });

  it('6b. two speakers on one zone: the autotune tunes both channels (one at a time, the other active)', async () => {
    const cfg = causalBroadband({
      secondary: [
        { delayS: 0.003, gain: 2 },
        { delayS: 0.0042, gain: 1.6 },
      ],
      feedback: [
        { delayS: 0.004, gain: 0.4 },
        { delayS: 0.0035, gain: 0.4 },
      ],
      seed: 13,
    });
    const p = calibrationParams('shared');
    const cal = await calibrate(new SimRig(cfg, RIG), p, 'shared');
    const solved = solveFromCalibration(cal, p, fs);
    expect(solved.params.channels[0].enabled && solved.params.channels[1].enabled).toBe(true);
    const start = clampParams(solved.params);
    start.channels[0].delayMs += 0.4;
    start.channels[0].gainDb -= 4;
    start.channels[1].delayMs -= 0.3;
    start.channels[1].gainDb += 3;
    const rig = new SimRig(cfg, RIG);
    apply(rig, { ...solved, params: start }, cal);
    const driver = rig.autotuneDriver(BAND);
    const res = await runAutotune(driver, start, { fs, channels: [0, 1], rt60S: 0.03, band: BAND });
    expect(res.aborted).toBe(false);
    expect(res.startDb - res.endDb).toBeGreaterThanOrEqual(6);
    const bypass = Math.max(...driver.log.bypass);
    for (const e of driver.log.errors) expect(e).toBeLessThanOrEqual(bypass + 1);
    // Both channels were searched.
    for (const ch of [0, 1] as const) expect(res.params.channels[ch]).not.toEqual(start.channels[ch]);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(10);
    expect(rig.autoMuted).toBe(false);
  }, 60000);

  describe('7. geometry only (no error mic in the engine)', () => {
    const tempC = 20;
    const c = speedOfSound(tempC);
    const geometry = { micToZoneM: 10, sourceAngleDeg: 0, speakerToZoneM: [1, 1] as [number, number], tempC };

    it('the solution from distances and latency cancels ≥ 6 dB', async () => {
      const head = headStartSeconds({ micToZoneM: 10, sourceAngleDeg: 0, tempC });
      const cfg = causalBroadband({
        primary: { delayS: head, gain: 0.7 },
        secondary: [{ delayS: 1 / c, gain: 1 }, null],
        feedback: [{ delayS: 0.004, gain: 0.1 }, null],
      });
      const rig = new SimRig(cfg, { ...RIG, errorMic: false });
      const p = calibrationParams('single');
      p.errorChannel = -1;
      rig.start(p, 0.1);
      const band = defaultSweepBand(p, fs);
      const cal = createCalibration(fs, band, 'single', tempC);
      cal.geometry = geometry;
      // Two-position levels: the single mic in the zone, then back at the reference.
      const amb = rig.bypassed(() => rig.run(2, ['ref', 'err']));
      cal.levels = { zoneDb: bandRmsDbfs(amb.err!, fs, band), refDb: bandRmsDbfs(amb.ref!, fs, band) };
      const out = solveFromGeometry(cal, p, fs, cfg.latencyS);
      expect(out.result.causal).toBe(true);
      expect(out.advice.some((a) => a.includes('non è misurabile'))).toBe(true);
      apply(rig, out, null);
      expect(attenuationDb(rig)).toBeGreaterThanOrEqual(6);
    }, 30000);

    it('a source 80° off the mic → zone line gives a warning, not a causality claim', () => {
      const cal = createCalibration(fs, [20, 1000], 'single', tempC);
      cal.geometry = { ...geometry, sourceAngleDeg: 80 };
      cal.levels = { zoneDb: -40, refDb: -37 };
      const out = solveFromGeometry(cal, calibrationParams('single'), fs, 0.012);
      expect(out.result.causal).toBe(false);
      expect(out.advice.some((a) => a.includes('80°') && a.includes('causalità non è garantita'))).toBe(true);
    });
  });

  it('8. looksLikeLoopback: true for an electrical loopback, false for an acoustic path', async () => {
    const quiet = { noise: { broadband: { rms: 0.001 } }, latencyS: 0.005, feedback: [null, null] as [null, null] };
    const loop = new SimRig(causalBroadband({ ...quiet, secondary: [{ delayS: 0, gain: 0.5 }, null] }), RIG);
    const p = calibrationParams('single');
    loop.start(p, 0.1);
    const band = defaultSweepBand(p, fs);
    const opts = { ...pathOptions(0, band), mic: 'err' as const };
    const electric = await measurePath(loop.measure, inProcessAnalyzer, { ...opts, mode: 'loopback' });
    expect(electric.ok).toBe(true);
    expect(electric.delaySamples).toBeCloseTo(240, 1);
    expect(looksLikeLoopback(electric)).toBe(true);
    expect(looksLikeLoopback(electric, { seconds: 0.005, fs })).toBe(true);
    expect(looksLikeLoopback(electric, { seconds: 0.012, fs })).toBe(false);

    const room = new SimRig(
      causalBroadband({
        ...quiet,
        secondary: [
          {
            delayS: 0.003,
            gain: 1,
            highpassHz: 80,
            reflections: [
              { delayS: 0.0021, gain: 0.4 },
              { delayS: 0.0057, gain: -0.25 },
            ],
          },
          null,
        ],
      }),
      RIG,
    );
    room.start(p, 0.1);
    const acoustic = await measurePath(room.measure, inProcessAnalyzer, { ...opts, mode: 'acoustic' });
    expect(acoustic.ok).toBe(true);
    expect(looksLikeLoopback(acoustic)).toBe(false);
  }, 30000);
});
