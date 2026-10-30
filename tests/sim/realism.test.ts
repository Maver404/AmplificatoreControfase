/**
 * Closed-loop simulator, rooms closer to real hardware than closed-loop.test.ts: a direct sound weaker
 * than a later reflection, noise at the zone the reference does not hear, a tonal source far upstream,
 * an inverting amplifier with the geometry-only alignment, reverberant paths. fs = 48 kHz, block 128.
 */
import { describe, expect, it } from 'vitest';
import { inProcessAnalyzer } from '../../src/audio/analysis-client.ts';
import {
  LOOP_GAIN_DANGER_DB,
  bandRmsDbfs,
  createCalibration,
  defaultSweepBand,
  feedbackFirs,
  measurePath,
  measurePrimary,
  solveFromCalibration,
  solveFromGeometry,
} from '../../src/audio/calibration.ts';
import type { CalibrationData, MeasurePathOptions, SolveOutcome } from '../../src/audio/calibration.ts';
import type { SpeakerConfig } from '../../src/dsp/alignment.ts';
import { clampParams, withoutTrims } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { headStartSeconds, speedOfSound } from '../../src/dsp/physics.ts';
import { SimRig, bandLevelDb, toneLevelDb } from './harness.ts';
import type { PlantConfig } from './plant.ts';

const fs = 48000;
const BAND: [number, number] = [80, 400];
const RIG = { engine: { softStartMs: 300 } };

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

async function calibrate(rig: SimRig, p: EngineParams, seconds = 2): Promise<CalibrationData> {
  rig.start(p, 0.1);
  const band = defaultSweepBand(p, fs);
  const cal = createCalibration(fs, band, 'single');
  const fb = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(0, band), mic: 'ref' });
  const sec = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(0, band), mic: 'err' });
  expect(fb.reason).toBeUndefined();
  expect(sec.reason).toBeUndefined();
  cal.feedback[0] = fb;
  cal.secondary[0] = sec;
  cal.primary = await measurePrimary(rig.capture, inProcessAnalyzer, { fs, seconds, band });
  return cal;
}

function apply(rig: SimRig, out: SolveOutcome, cal: CalibrationData | null): void {
  rig.send({ type: 'arm', armed: false });
  const firs = cal ? feedbackFirs(cal, 256) : [null, null];
  for (const ch of [0, 1] as const) {
    const f = firs[ch];
    rig.send({ type: 'feedbackFir', channel: ch, bulkDelay: f?.bulkDelay ?? 1, taps: f?.taps ?? null });
  }
  rig.start(out.params, 0.3);
}

function attenuationDb(rig: SimRig, band: [number, number] = BAND, seconds = 2): number {
  const on = rig.run(seconds, ['err']).err!;
  const off = rig.bypassed(() => rig.run(seconds, ['err']).err!);
  return bandLevelDb(off, fs, band[0], band[1]) - bandLevelDb(on, fs, band[0], band[1]);
}

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

describe('closed loop, realistic rooms', () => {
  it('9. a weak direct sound before a strong reflection on the feedback path: the FIR neutralises both', async () => {
    // Speaker → reference: direct sound 0.6 at L_e + 2 ms, a reflection 3.0 (+14 dB) 6 ms later.
    const cfg = causalBroadband({ feedback: [{ delayS: 0.002, gain: 0.6, reflections: [{ delayS: 0.006, gain: 3 }] }, null] });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p);
    const direct = Math.round((0.012 + 0.002) * fs);
    const reflection = Math.round((0.012 + 0.008) * fs);
    // The estimator's first arrival (−6 dB rule) is the reflection…
    expect(cal.feedback[0]!.delaySamples).toBeCloseTo(reflection, 0);
    // …but the FIR starts before the direct sound and reaches past the reflection.
    const fir = feedbackFirs(cal, 256)[0]!;
    expect(fir.bulkDelay).toBeLessThan(direct);
    expect(fir.bulkDelay + fir.taps.length).toBeGreaterThan(reflection + 100);
    const out = solveFromCalibration(cal, p, fs);
    // Without neutralisation the loop would howl; what the FIR leaves is the low band edge a 10 ms FIR
    // cannot model, kept inside the 6 dB margin with at most a small gain reduction.
    expect(out.loopGainDb! + out.loopLimitedDb!).toBeGreaterThan(0);
    expect(out.loopGainAfterDb!).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 1e-9);
    expect(out.loopLimitedDb!).toBeLessThan(2);
    apply(rig, out, cal);
    rig.run(1);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(10);
    expect(rig.autoMuted).toBe(false);
  }, 60000);

  it('10. noise at the zone the reference does not hear: the prediction stays within the coherence bound', async () => {
    // Primary noise at the zone ≈ 0.007 RMS; a local source of half that (−6 dB) only at the error mic.
    const cfg = causalBroadband({ noise: { broadband: { rms: 0.01, lo: 20, hi: 1000 } }, zoneNoise: { rms: 0.0035, lo: 20, hi: 1000 }, seed: 9 });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 4);
    const out = solveFromCalibration(cal, p, fs);
    // γ² ≈ 0.8 in band: at most ≈ 7 dB can be cancelled.
    const bound = -10 * Math.log10(1 - 0.8);
    expect(out.result.predictedAttenuationDb).toBeLessThan(bound + 0.5);
    apply(rig, out, cal);
    const measured = attenuationDb(rig, cal.band, 4);
    expect(measured).toBeGreaterThan(3);
    expect(Math.abs(out.result.predictedAttenuationDb - measured)).toBeLessThan(2);
  }, 60000);

  it('11. a tonal source 0.2 s upstream: the tonal solution cancels each harmonic ≥ 15 dB', async () => {
    const cfg = causalBroadband({
      noise: { tone: { f0: 100, amplitudes: [0.04, 0.025, 0.015, 0.01] }, broadband: { rms: 0.002 } },
      primary: { delayS: 0.2, gain: 0.7 },
      seed: 13,
    });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 3);
    const f0 = cal.primary!.tonalF0!;
    expect(Math.abs(f0 - 100)).toBeLessThan(0.01);
    p.phaseMode = 'tonal';
    p.tonal.f0 = f0;
    const out = solveFromCalibration(cal, p, fs);
    expect(out.params.channels[0].enabled).toBe(true);
    apply(rig, out, cal);
    rig.run(1);
    const on = rig.run(2, ['err']).err!;
    const off = rig.bypassed(() => rig.run(2, ['err']).err!);
    // Per harmonic: 16.4 / 19.9 / 17.2 / 10.4 dB with the delay alone, ≥ 40 dB with the harmonic trims.
    for (const h of [1, 2, 3, 4]) expect(toneLevelDb(off, fs, 100 * h) - toneLevelDb(on, fs, 100 * h)).toBeGreaterThanOrEqual(15);
    const tonalPower = (x: Float32Array): number =>
      [100, 200, 300, 400].reduce((s, f) => s + Math.pow(10, toneLevelDb(x, fs, f) / 10), 0);
    expect(10 * Math.log10(tonalPower(off) / tonalPower(on))).toBeGreaterThanOrEqual(15);
  }, 60000);

  it('14. tonal stress: HP 40 Hz, barrier on, a wall-like primary and a room-like speaker path: each harmonic ≥ 12 dB', async () => {
    const cfg = causalBroadband({
      noise: { tone: { f0: 100, amplitudes: [0.04, 0.025, 0.015, 0.01] }, broadband: { rms: 0.002 } },
      // Through a wall (1st-order low-pass) with two reflections, 0.2 s upstream.
      primary: { delayS: 0.2, gain: 0.7, lowpassHz: 150, reflections: [{ delayS: 0.0037, gain: 0.3 }, { delayS: 0.0091, gain: -0.2 }] },
      // A small speaker (roll-off below 90 Hz) in a room: strong early reflections and a diffuse tail.
      secondary: [
        {
          delayS: 0.003,
          gain: 2,
          highpassHz: 90,
          reflections: [{ delayS: 0.0025, gain: 0.5 }, { delayS: 0.0061, gain: -0.4 }],
          tail: { rt60S: 0.08, gain: 0.08, seed: 5 },
        },
        null,
      ],
      feedback: [{ delayS: 0.004, gain: 0.3, tail: { rt60S: 0.08, gain: 0.02, seed: 6 } }, null],
      seed: 17,
    });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    p.filters = { ...p.filters, hpFreq: 40, hpOrder: 2, barrierEnabled: true };
    const cal = await calibrate(rig, p, 3);
    p.phaseMode = 'tonal';
    p.tonal.f0 = cal.primary!.tonalF0!;
    expect(Math.abs(p.tonal.f0 - 100)).toBeLessThan(0.01);
    const out = solveFromCalibration(cal, p, fs);
    expect(out.params.channels[0].enabled).toBe(true);
    const drops = (params: EngineParams): number[] => {
      apply(rig, { ...out, params }, cal);
      rig.run(1);
      const on = rig.run(2, ['err']).err!;
      const off = rig.bypassed(() => rig.run(2, ['err']).err!);
      return [1, 2, 3, 4].map((h) => toneLevelDb(off, fs, 100 * h) - toneLevelDb(on, fs, 100 * h));
    };
    // Measured: the delay alone 9.0 / 13.5 / 0.7 / −6.0 dB (the 4th harmonic louder than without anti-noise),
    // with the harmonic trims 36 / 48 / 42 / 68 dB.
    expect(Math.min(...drops(withoutTrims(out.params)))).toBeLessThan(6);
    for (const d of drops(out.params)) expect(d).toBeGreaterThanOrEqual(12);
    expect(rig.autoMuted).toBe(false);
  }, 60000);

  it('12. geometry only with an inverting amplifier: the polarity measured on F keeps the cancellation', async () => {
    const tempC = 20;
    const c = speedOfSound(tempC);
    const head = headStartSeconds({ micToZoneM: 10, sourceAngleDeg: 0, tempC });
    // The amplifier inverts: both speaker paths arrive with a negative sign.
    const cfg = causalBroadband({
      primary: { delayS: head, gain: 0.7 },
      secondary: [{ delayS: 1 / c, gain: 1, invert: true }, null],
      feedback: [{ delayS: 0.004, gain: 0.1, invert: true }, null],
    });
    const rig = new SimRig(cfg, { ...RIG, errorMic: false });
    const p = calibrationParams('single');
    p.errorChannel = -1;
    rig.start(p, 0.1);
    const band = defaultSweepBand(p, fs);
    const cal = createCalibration(fs, band, 'single', tempC);
    cal.geometry = { micToZoneM: 10, sourceAngleDeg: 0, speakerToZoneM: [1, 1], tempC };
    cal.feedback[0] = await measurePath(rig.measure, inProcessAnalyzer, { ...pathOptions(0, band), mic: 'ref' });
    expect(cal.feedback[0].sign).toBe(-1);
    const amb = rig.bypassed(() => rig.run(2, ['ref', 'err']));
    cal.levels = { zoneDb: bandRmsDbfs(amb.err!, fs, band), refDb: bandRmsDbfs(amb.ref!, fs, band) };
    const out = solveFromGeometry(cal, p, fs, cfg.latencyS);
    expect(out.params.channels[0].polarityInvert).toBe(false);
    apply(rig, out, cal);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(6);
  }, 60000);

  it('13. reverberant speaker paths (RT60 80 ms): still ≥ 8 dB in 80–400 Hz, stable', async () => {
    const cfg = causalBroadband({
      noise: { broadband: { rms: 0.01, lo: 20, hi: 1000 } },
      secondary: [{ delayS: 0.003, gain: 2, reflections: [{ delayS: 0.0025, gain: 0.15 }], tail: { rt60S: 0.08, gain: 0.08, seed: 5 } }, null],
      feedback: [{ delayS: 0.004, gain: 0.3, tail: { rt60S: 0.08, gain: 0.02, seed: 6 } }, null],
    });
    const rig = new SimRig(cfg, RIG);
    const p = calibrationParams('single');
    const cal = await calibrate(rig, p, 3);
    expect(cal.secondary[0]!.rt60S ?? 0).toBeGreaterThan(0.03);
    const out = solveFromCalibration(cal, p, fs);
    apply(rig, out, cal);
    rig.run(1);
    expect(attenuationDb(rig)).toBeGreaterThanOrEqual(8);
    expect(rig.autoMuted).toBe(false);
  }, 60000);
});
