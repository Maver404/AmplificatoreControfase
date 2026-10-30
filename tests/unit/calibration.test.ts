import { afterEach, describe, expect, it } from 'vitest';
import { F0Window, createAnalyzer, inProcessAnalyzer } from '../../src/audio/analysis-client.ts';
import type { AnalysisRequest, AnalysisResponse } from '../../src/audio/analysis-client.ts';
import {
  AMP_ADVICE_DB,
  CalibrationError,
  amplifierAdvice,
  amplifierLevelNoise,
  bandRmsDbfs,
  calibrationFreqs,
  createCalibration,
  defaultSweepBand,
  estimateRt60,
  LOOP_GAIN_DANGER_DB,
  feedbackFirs,
  looksLikeLoopback,
  loopbackSuspectPaths,
  measurePath,
  measurePrimary,
  pathResponse,
  sanitizeCalibration,
  solutionToParams,
  solveFromCalibration,
  solveFromGeometry,
  welchTransfer,
} from '../../src/audio/calibration.ts';
import type { CalibrationData, MeasureFn, MeasurePathOptions, PathData } from '../../src/audio/calibration.ts';
import type { AlignmentResult } from '../../src/dsp/alignment.ts';
import { FEEDBACK_FIR_MARGIN } from '../../src/dsp/feedback-canceller.ts';
import { F0Tracker, estimateF0 } from '../../src/dsp/f0-tracker.ts';
import { irToResponse } from '../../src/dsp/latency-estimator.ts';
import { D_MIN_SAMPLES, clampParams, computeChannelDelay, dbToLin } from '../../src/dsp/params.ts';
import { channelResponse } from '../../src/dsp/response.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';

const fs = 48000;

function gaussian(seed: number): () => number {
  const u = mulberry32(seed);
  return () => Math.sqrt(-2 * Math.log(Math.max(u(), 1e-12))) * Math.cos(2 * Math.PI * u());
}

/** A small but complete calibration record. */
function sampleCalibration(): CalibrationData {
  const path = (d: number): PathData => ({
    delaySamples: d,
    sign: 1,
    qualityDb: 35,
    stdSamples: 0.05,
    freqs: [100, 200, 300],
    re: [0.5, -0.4, 0.1],
    im: [0.1, 0.2, -0.45],
    ir: [0, 0.01, 0.5, 0.2, -0.05, 0],
    irOffset: d - 2,
  });
  return {
    version: 1,
    fs,
    band: [20, 1000],
    tempC: 21,
    createdAt: 1_700_000_000_000,
    feedback: [path(700), null],
    secondary: [path(720), path(760)],
    primary: { freqs: [100, 200, 300], re: [0.7, 0, -0.7], im: [0, 0.7, 0], coherence: [0.99, 0.98, 0.97], tonalF0: null, psd: [1e-6, 5e-7, 2e-7] },
    loopbackLatencyS: 0.012,
    geometry: { micToZoneM: 8, sourceAngleDeg: 10, speakerToZoneM: [1, 1.2], tempC: 21 },
    levels: { zoneDb: -40, refDb: -37 },
    speakers: 'shared',
  };
}

describe('sanitizeCalibration', () => {
  it('accepts a valid record and returns a deep plain copy', () => {
    const cal = sampleCalibration();
    const out = sanitizeCalibration(cal);
    expect(out).toEqual(cal);
    expect(out).not.toBe(cal);
    expect(out!.secondary[0]).not.toBe(cal.secondary[0]);
    expect(sanitizeCalibration(JSON.parse(JSON.stringify(cal)))).toEqual(cal);
  });

  it('converts typed arrays to plain number arrays', () => {
    const cal = sampleCalibration() as unknown as Record<string, unknown>;
    const s0 = (cal.secondary as Record<string, unknown>[])[0];
    s0.ir = Float32Array.from(s0.ir as number[]);
    s0.re = Float64Array.from(s0.re as number[]);
    cal.band = Float64Array.of(20, 1000);
    (cal.primary as Record<string, unknown>).coherence = Float32Array.of(0.5, 0.25, 0.125);
    (cal.geometry as Record<string, unknown>).speakerToZoneM = Float32Array.of(1, 2);
    const out = sanitizeCalibration(cal)!;
    expect(out).not.toBeNull();
    expect(Array.isArray(out.secondary[0]!.ir)).toBe(true);
    expect(Array.isArray(out.secondary[0]!.re)).toBe(true);
    expect(Array.isArray(out.band)).toBe(true);
    expect(Array.isArray(out.primary!.coherence)).toBe(true);
    expect(out.geometry!.speakerToZoneM).toEqual([1, 2]);
    expect(out.secondary[0]!.ir).toEqual(Array.from(Float32Array.from([0, 0.01, 0.5, 0.2, -0.05, 0])));
    expect(out.primary!.coherence).toEqual([0.5, 0.25, 0.125]);
    // Nothing typed is left: a JSON round trip is lossless.
    expect(JSON.parse(JSON.stringify(out))).toEqual(out);
  });

  const mutations: [string, (c: CalibrationData) => void][] = [
    ['fs', (c) => (c.fs = Number.NaN)],
    ['tempC', (c) => (c.tempC = Number.NaN)],
    ['band', (c) => (c.band[1] = Number.NaN)],
    ['createdAt', (c) => (c.createdAt = Number.POSITIVE_INFINITY)],
    ['feedback re', (c) => (c.feedback[0]!.re[1] = Number.NaN)],
    ['secondary ir', (c) => (c.secondary[1]!.ir[3] = Number.NaN)],
    ['secondary delay', (c) => (c.secondary[0]!.delaySamples = Number.NaN)],
    ['secondary quality', (c) => (c.secondary[0]!.qualityDb = Number.NaN)],
    ['primary coherence', (c) => (c.primary!.coherence[0] = Number.NaN)],
    ['primary psd', (c) => (c.primary!.psd![2] = Number.NaN)],
    ['primary tonalF0', (c) => (c.primary!.tonalF0 = Number.NaN)],
    ['loopback latency', (c) => (c.loopbackLatencyS = Number.NaN)],
    ['geometry', (c) => (c.geometry!.micToZoneM = Number.NaN)],
    ['geometry speaker', (c) => (c.geometry!.speakerToZoneM[1] = Number.NaN)],
    ['levels', (c) => (c.levels.zoneDb = Number.NaN)],
  ];
  for (const [name, mutate] of mutations) {
    it(`rejects NaN/Infinity in ${name}`, () => {
      const cal = sampleCalibration();
      mutate(cal);
      expect(sanitizeCalibration(cal)).toBeNull();
    });
  }

  it('rejects a wrong or missing version and non-objects', () => {
    for (const version of [2, 0, '1', undefined, null]) {
      expect(sanitizeCalibration({ ...sampleCalibration(), version })).toBeNull();
    }
    for (const raw of [null, undefined, 42, 'x', [], [sampleCalibration()]]) expect(sanitizeCalibration(raw)).toBeNull();
  });

  it('rejects inconsistent data', () => {
    const broken: ((c: CalibrationData) => void)[] = [
      (c) => c.secondary[0]!.re.pop(),
      (c) => (c.secondary[0]!.freqs = [100, 100, 300]),
      (c) => (c.secondary[0]!.sign = 0 as unknown as 1),
      (c) => (c.secondary[0]!.irOffset = 1.5),
      (c) => (c.band = [1000, 20]),
      (c) => (c.band = [20, 30000]),
      (c) => (c.speakers = 'mono' as 'single'),
      (c) => ((c as unknown as Record<string, unknown>).feedback = [null]),
      (c) => c.primary!.coherence.pop(),
      (c) => (c.primary!.psd = [1, -1, 1]),
    ];
    for (const b of broken) {
      const cal = sampleCalibration();
      b(cal);
      expect(sanitizeCalibration(cal)).toBeNull();
    }
  });

  it('reads absent nullable fields as null and clamps coherence rounding to [0, 1]', () => {
    const cal = sampleCalibration() as unknown as Record<string, unknown>;
    delete cal.loopbackLatencyS;
    delete cal.geometry;
    delete cal.levels;
    (cal.primary as Record<string, unknown>).coherence = [1.0000001, 0.5, 0];
    const out = sanitizeCalibration(cal)!;
    expect(out.loopbackLatencyS).toBeNull();
    expect(out.geometry).toBeNull();
    expect(out.levels).toEqual({ zoneDb: null, refDb: null });
    expect(out.primary!.coherence[0]).toBe(1);
  });
});

const unused = { use: false, delaySamples: D_MIN_SAMPLES, gainLin: 0, polarityInvert: true };
const result = (ch0: AlignmentResult['channels'][0], ch1 = unused): AlignmentResult => ({
  channels: [ch0, ch1],
  predictedAttenuationDb: 15,
  causal: true,
  requiredExtraLeadSeconds: 0,
});

describe('solutionToParams', () => {
  it('converts total delay and gain into ChannelParams, accounting for D_MIN and the master', () => {
    const p = clampParams(undefined);
    p.masterDb = -6;
    const { params, advice } = solutionToParams(p, result({ use: true, delaySamples: D_MIN_SAMPLES + 480.5, gainLin: 0.25, polarityInvert: true }), fs);
    const c = params.channels[0];
    expect(c.enabled).toBe(true);
    expect(c.delayMs).toBeCloseTo((480.5 / fs) * 1000, 9);
    expect(c.gainDb).toBeCloseTo(20 * Math.log10(0.25) + 6, 9);
    expect(c.polarityInvert).toBe(true);
    expect(c.phaseDeltaDeg).toBe(0);
    expect(params.channels[1].enabled).toBe(false);
    expect(params.masterDb).toBe(-6);
    // The engine's own delay and gain reproduce the totals exactly.
    expect(computeChannelDelay(params, fs, 0).samples).toBeCloseTo(D_MIN_SAMPLES + 480.5, 9);
    expect(dbToLin(c.gainDb) * dbToLin(params.masterDb)).toBeCloseTo(0.25, 12);
    const h = channelResponse({ ...params, filters: { ...params.filters, hpEnabled: false, lpEnabled: false } }, fs, 200, 0);
    expect(Math.hypot(h.re, h.im)).toBeCloseTo(0.25, 3);
    expect(advice).toEqual([]);
  });

  it('maps a delay of exactly D_MIN to 0 ms and keeps the polarity of the solution', () => {
    const { params } = solutionToParams(clampParams(undefined), result({ use: true, delaySamples: D_MIN_SAMPLES, gainLin: 0.01, polarityInvert: false }), fs);
    expect(params.channels[0].delayMs).toBe(0);
    expect(params.channels[0].polarityInvert).toBe(false);
    // default master −30 dB: channel gain = total + 30
    expect(params.channels[0].gainDb).toBeCloseTo(-10, 9);
  });

  it('in tonal mode the delay stays the tap delay modulo the period', () => {
    const p = clampParams(undefined);
    p.phaseMode = 'tonal';
    p.tonal.f0 = 100;
    const { params } = solutionToParams(p, result({ use: true, delaySamples: D_MIN_SAMPLES + 300.25, gainLin: 0.03, polarityInvert: true }), fs);
    expect(computeChannelDelay(params, fs, 0).samples).toBeCloseTo(D_MIN_SAMPLES + 300.25, 9);
  });

  it('clamps gain and delay to the parameter ranges with advice', () => {
    const p = clampParams(undefined);
    p.masterDb = 0;
    const r = result({ use: true, delaySamples: D_MIN_SAMPLES + 0.6 * fs, gainLin: 2, polarityInvert: true });
    const { params, advice } = solutionToParams(p, r, fs);
    expect(params.channels[0].gainDb).toBe(0);
    expect(params.channels[0].delayMs).toBe(500);
    expect(advice.some((a) => a.includes('limitato a 0 dB'))).toBe(true);
    expect(advice.some((a) => a.includes('500 ms'))).toBe(true);
    const unlocked = solutionToParams({ ...p, gainUnlocked: true }, r, fs).params;
    expect(unlocked.channels[0].gainDb).toBeCloseTo(20 * Math.log10(2), 9);
  });

  it('keeps the previous gains when the master is at −∞', () => {
    const p = clampParams(undefined);
    p.masterDb = -120;
    p.channels[0].gainDb = -17;
    const { params, advice } = solutionToParams(p, result({ use: true, delaySamples: 100, gainLin: 0.3, polarityInvert: true }), fs);
    expect(params.channels[0].gainDb).toBe(-17);
    expect(params.channels[0].delayMs).toBeCloseTo((99 / fs) * 1000, 9);
    expect(advice.some((a) => a.includes('master'))).toBe(true);
  });
});

describe('amplifierAdvice', () => {
  it('is silent inside −30…−6 dB', () => {
    for (const g of [AMP_ADVICE_DB.min, -18, AMP_ADVICE_DB.max, -6.0001]) expect(amplifierAdvice(0, g)).toBeNull();
    expect(amplifierAdvice(0, Number.NaN)).toBeNull();
  });

  it('above −6 dB asks for more reference preamp gain, never a louder amplifier', () => {
    for (const [g, by] of [[-3, 15], [4.4, 22], [12, 30]] as const) {
      const up = amplifierAdvice(1, g)!;
      expect(up).toMatch(/^Canale R/);
      expect(up).toContain(`Alza di circa ${by} dB il guadagno d'ingresso (preamplificatore) del microfono di riferimento`);
      expect(up).toMatch(/ripeti la misura dei percorsi e l'allineamento/);
      expect(up).toMatch(/Non alzare l'amplificatore/);
      expect(up).not.toMatch(/Alza il volume dell'amplificatore/);
    }
    // Barely above the limit: at least 1 dB.
    expect(amplifierAdvice(0, -5.9)).toMatch(/Alza di circa 12 dB/);
  });

  it('below −30 dB asks to turn the amplifier down', () => {
    const down = amplifierAdvice(0, -40)!;
    expect(down).toMatch(/^Canale L/);
    expect(down).toMatch(/l'amplificatore è troppo alto\. Abbassalo di circa 22 dB/);
    expect(down).not.toMatch(/preamplificatore/);
  });

  it('is part of the solution advice', () => {
    const p = clampParams(undefined);
    p.masterDb = 0;
    const loud = solutionToParams(p, result({ use: true, delaySamples: 10, gainLin: dbToLin(-2), polarityInvert: true }), fs);
    expect(loud.advice.some((a) => a.includes('preamplificatore'))).toBe(true);
    // A quiet reference: the solved gain is above 0 dB (clamped), still the preamp, not the amplifier.
    const quiet = solutionToParams(p, result({ use: true, delaySamples: 10, gainLin: dbToLin(8), polarityInvert: true }), fs);
    expect(quiet.params.channels[0].gainDb).toBe(0);
    expect(quiet.advice.some((a) => a.includes('Alza di circa 26 dB') && a.includes('preamplificatore'))).toBe(true);
    expect(quiet.advice.some((a) => a.includes('limitato a 0 dB'))).toBe(true);
    const soft = solutionToParams(p, result({ use: true, delaySamples: 10, gainLin: dbToLin(-45), polarityInvert: true }), fs);
    expect(soft.advice.some((a) => a.includes('Abbassalo'))).toBe(true);
    const fine = solutionToParams(p, result({ use: true, delaySamples: 10, gainLin: dbToLin(-15), polarityInvert: true }), fs);
    expect(fine.advice).toEqual([]);
  });
});

describe('sweep band and grid', () => {
  it('defaults to [max(0.8·hp, 20), min(2·lp, 0.45·fs)]', () => {
    const p = clampParams(undefined);
    expect(defaultSweepBand(p, fs)).toEqual([20, 1000]);
    p.filters.hpFreq = 50;
    p.filters.lpFreq = 3000;
    expect(defaultSweepBand(p, fs)).toEqual([40, 6000]);
    expect(defaultSweepBand(p, 8000)).toEqual([40, 3600]);
  });

  it('uses one shared, deterministic, stratified grid: one point per cell of ≥ 2 Hz, ≤ 800 points', () => {
    const g = calibrationFreqs(fs, [20, 1000]);
    expect(g.length).toBe(490);
    for (let k = 0; k < g.length; k++) {
      expect(g[k]).toBeGreaterThanOrEqual(20 + 2 * k);
      expect(g[k]).toBeLessThan(20 + 2 * (k + 1));
    }
    expect(calibrationFreqs(fs, [20, 1000])).toEqual(g);
    const wide = calibrationFreqs(fs, [20, 8000]);
    expect(wide.length).toBeLessThanOrEqual(800);
    expect(calibrationFreqs(fs, [0, 1000]).length).toBe(0);
  });

  it('does not alias long delays: a 0.5 s misalignment is not a solution', () => {
    // On a uniform 2 Hz grid e^{-jω·0.5 s} = 1 at every point; on the stratified grid it averages out.
    const g = calibrationFreqs(fs, [20, 1000]);
    let re = 0;
    let im = 0;
    for (const f of g) {
      re += Math.cos(2 * Math.PI * f * 0.5);
      im += Math.sin(2 * Math.PI * f * 0.5);
    }
    expect(Math.hypot(re, im) / g.length).toBeLessThan(0.1);
  });
});

describe('levels and decay', () => {
  it('bandRmsDbfs measures the RMS level inside a band', () => {
    const x = Float32Array.from({ length: fs }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 200 * i) / fs));
    expect(bandRmsDbfs(x, fs, [100, 300])).toBeCloseTo(20 * Math.log10(0.5 / Math.SQRT2), 1);
    expect(bandRmsDbfs(x, fs, [400, 800])).toBeLessThan(-60);
    expect(bandRmsDbfs(new Float32Array(fs), fs, [100, 300])).toBe(-120);
  });

  it('amplifierLevelNoise is band-limited pink noise at the requested RMS', () => {
    const x = amplifierLevelNoise(fs, [20, 1000], 2);
    const rms = Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length);
    expect(20 * Math.log10(rms)).toBeCloseTo(-20, 1);
    expect(x.reduce((a, v) => Math.max(a, Math.abs(v)), 0)).toBeLessThanOrEqual(1);
    expect(bandRmsDbfs(x, fs, [20, 1000])).toBeCloseTo(-20, 0);
    expect(bandRmsDbfs(x, fs, [2000, 8000])).toBeLessThan(-60);
    // Pink: equal power per octave.
    expect(bandRmsDbfs(x, fs, [100, 200]) - bandRmsDbfs(x, fs, [400, 800])).toBeCloseTo(0, 0);
  });

  it('estimateRt60 recovers the decay time of an exponential tail and refuses a bare impulse', () => {
    const g = gaussian(4);
    const ir = new Float64Array(1 << 17);
    ir[500] = 1;
    for (let i = 501; i < ir.length >> 1; i++) ir[i] = 0.05 * g() * Math.exp((-Math.log(1000) * (i - 500)) / (0.4 * fs)) + 1e-6 * g();
    expect(estimateRt60(ir, 500, fs)!).toBeGreaterThan(0.36);
    expect(estimateRt60(ir, 500, fs)!).toBeLessThan(0.44);
    const dry = new Float64Array(1 << 17);
    dry[500] = 1;
    expect(estimateRt60(dry, 500, fs)).toBeNull();
  });
});

describe('welchTransfer', () => {
  const n = 3 * fs;
  const g = gaussian(7);
  const x = Float32Array.from({ length: n }, () => 0.1 * g());

  it('estimates gain, long delay and coherence without bias', () => {
    const d = 1200;
    const gn = gaussian(8);
    const y = Float32Array.from({ length: n }, (_, i) => (i >= d ? -0.5 * x[i - d] : 0) + 1e-4 * gn());
    const freqs = calibrationFreqs(fs, [50, 1000]);
    const t = welchTransfer(x, y, fs, freqs);
    for (let k = 0; k < freqs.length; k += 25) {
      const w = (2 * Math.PI * freqs[k]) / fs;
      expect(Math.hypot(t.re[k], t.im[k])).toBeCloseTo(0.5, 2);
      const err = Math.atan2(t.im[k], t.re[k]) - Math.atan2(-0.5 * Math.sin(-w * d), -0.5 * Math.cos(-w * d));
      expect(Math.abs(Math.atan2(Math.sin(err), Math.cos(err)))).toBeLessThan((1 * Math.PI) / 180);
      expect(t.coherence[k]).toBeGreaterThan(0.99);
      expect(t.pxx[k]).toBeGreaterThan(0);
    }
  });

  it('is exact at the harmonics of a tone', () => {
    const tone = Float32Array.from({ length: n }, (_, i) => 0.3 * Math.sin((2 * Math.PI * 100 * i) / fs) + 0.1 * Math.sin((2 * Math.PI * 200 * i) / fs + 1));
    const y = Float32Array.from({ length: n }, (_, i) => (i >= 240 ? 0.7 * tone[i - 240] : 0));
    const t = welchTransfer(tone, y, fs, Float64Array.of(100, 200));
    for (const [k, f] of [[0, 100], [1, 200]]) {
      const w = (2 * Math.PI * f) / fs;
      expect(t.re[k]).toBeCloseTo(0.7 * Math.cos(-w * 240), 3);
      expect(t.im[k]).toBeCloseTo(0.7 * Math.sin(-w * 240), 3);
    }
  });

  it('returns zeros for unusable input', () => {
    const t = welchTransfer(new Float32Array(100), new Float32Array(100), fs, Float64Array.of(100));
    expect(t.re[0]).toBe(0);
    expect(t.coherence[0]).toBe(0);
  });
});

/** A MeasureFn for a known output → mic path: taps (lag, gain), master gain, ambient noise and tone. */
function fakeMeasure(o: {
  taps: [number, number][];
  masterLin?: number;
  noise?: number;
  tone?: { f: number; a: number }[];
  noErr?: boolean;
  aborted?: boolean;
}): MeasureFn & { calls: Float32Array[] } {
  const calls: Float32Array[] = [];
  const g = gaussian(99);
  let clock = 0;
  const fn = (async (stimulus: Float32Array, _channel: 0 | 1, post: number) => {
    calls.push(stimulus);
    const m = o.masterLin ?? 1;
    const n = stimulus.length + post;
    const rec = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (const [lag, gain] of o.taps) {
        const j = i - lag;
        if (j >= 0 && j < stimulus.length) v += gain * m * stimulus[j];
      }
      v += (o.noise ?? 0) * g();
      for (const t of o.tone ?? []) v += t.a * Math.sin((2 * Math.PI * t.f * (clock + i)) / fs);
      rec[i] = v;
    }
    // Repetitions start at unrelated times relative to the ambient tone.
    clock += n + 12345;
    return { ref: rec, err: o.noErr ? null : rec, masterLin: m, aborted: o.aborted === true };
  }) as MeasureFn & { calls: Float32Array[] };
  fn.calls = calls;
  return fn;
}

const opts = (over: Partial<MeasurePathOptions> = {}): MeasurePathOptions => ({
  fs,
  channel: 0,
  band: [20, 1000],
  levelDbfs: -20,
  repetitions: 3,
  mic: 'err',
  mode: 'acoustic',
  previewFirst: true,
  sweepS: 1,
  postRollS: 0.4,
  ...over,
});

/** True response of a tap list at f. */
const tapsAt = (taps: [number, number][], f: number): { re: number; im: number } => {
  const w = (2 * Math.PI * f) / fs;
  let re = 0;
  let im = 0;
  for (const [lag, gain] of taps) {
    re += gain * Math.cos(-w * lag);
    im += gain * Math.sin(-w * lag);
  }
  return { re, im };
};

describe('measurePath (synthetic MeasureFn)', () => {
  // A reflection 4 ms after the opposite-polarity main arrival (closer ones bias the sub-sample peak).
  const taps: [number, number][] = [
    [700, -0.8],
    [892, 0.2],
  ];

  it('measures delay, polarity and the response divided by the master gain', async () => {
    const m = fakeMeasure({ taps, masterLin: 0.5, noise: 1e-4 });
    const r = await measurePath(m, inProcessAnalyzer, opts());
    expect(r.ok).toBe(true);
    expect(r.reason).toBeUndefined();
    expect(r.delaySamples).toBeCloseTo(700, 1);
    expect(r.sign).toBe(-1);
    expect(r.stdSamples).toBeLessThan(0.1);
    // Preview first, 20 dB lower, then the repetitions.
    expect(m.calls.length).toBe(4);
    const peak = (x: Float32Array): number => x.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
    expect(peak(m.calls[1])).toBeCloseTo(0.1, 3);
    expect(peak(m.calls[0]) / peak(m.calls[1])).toBeCloseTo(0.1, 5);
    // The stored window starts before the arrival (possibly at a negative lag) and is plain JSON.
    expect(r.irOffset).toBeLessThan(700);
    expect(Array.isArray(r.ir) && Array.isArray(r.freqs) && Array.isArray(r.re)).toBe(true);
    expect(r.freqs).toEqual(Array.from(calibrationFreqs(fs, [20, 1000]), (f) => Number(f.toPrecision(7))));
    // Inside the band (away from the edge tapers) the stored IR reproduces the path within 1 %.
    const fr = Float64Array.of(60, 100, 250, 500, 800);
    const h = pathResponse(r, fs, fr);
    for (let k = 0; k < fr.length; k++) {
      const t = tapsAt(taps, fr[k]);
      expect(Math.hypot(h.re[k] - t.re, h.im[k] - t.im) / Math.hypot(t.re, t.im)).toBeLessThan(fr[k] < 100 ? 0.01 : 0.001);
    }
    const cal = createCalibration(fs, [20, 1000], 'single');
    cal.secondary[0] = r;
    expect(sanitizeCalibration(JSON.parse(JSON.stringify(cal)))).not.toBeNull();
  });

  it('removes a stationary ambient tone from the recordings', async () => {
    const tone = [
      { f: 100, a: 0.03 },
      { f: 200, a: 0.02 },
    ];
    const fr = Float64Array.of(100, 200);
    const relErr = async (removeTones: boolean): Promise<number> => {
      const r = await measurePath(fakeMeasure({ taps, noise: 1e-4, tone }), inProcessAnalyzer, opts({ removeTones, previewFirst: false }));
      expect(r.ok).toBe(true);
      const h = pathResponse(r, fs, fr);
      let worst = 0;
      for (let k = 0; k < fr.length; k++) {
        const t = tapsAt(taps, fr[k]);
        worst = Math.max(worst, Math.hypot(h.re[k] - t.re, h.im[k] - t.im) / Math.hypot(t.re, t.im));
      }
      return worst;
    };
    const without = await relErr(false);
    const withRemoval = await relErr(true);
    expect(without).toBeGreaterThan(0.1);
    expect(withRemoval).toBeLessThan(0.01);
  });

  it('stops after the preview when the real sweep would saturate the input', async () => {
    const m = fakeMeasure({ taps: [[300, 6]] });
    const r = await measurePath(m, inProcessAnalyzer, opts({ levelDbfs: -14 }));
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/saturazione/);
    expect(m.calls.length).toBe(1);
  });

  it('reports silence, a missing error mic, a zero master and aborts', async () => {
    const silent = await measurePath(fakeMeasure({ taps: [] }), inProcessAnalyzer, opts());
    expect(silent.ok).toBe(false);
    expect(silent.reason).toMatch(/Nessun segnale/);
    const noErr = await measurePath(fakeMeasure({ taps, noErr: true }), inProcessAnalyzer, opts());
    expect(noErr.reason).toMatch(/Microfono d'errore assente: collegalo all'ingresso 2/);
    // With the reference on input 2 the error mic goes on input 1 (calib-ui-6).
    const noErr1 = await measurePath(fakeMeasure({ taps, noErr: true }), inProcessAnalyzer, opts({ errorInput: 1 }));
    expect(noErr1.reason).toMatch(/Microfono d'errore assente: collegalo all'ingresso 1/);
    const ref = await measurePath(fakeMeasure({ taps, noErr: true }), inProcessAnalyzer, opts({ mic: 'ref', previewFirst: false, repetitions: 1 }));
    expect(ref.ok).toBe(true);
    const zero = await measurePath(fakeMeasure({ taps, masterLin: 0 }), inProcessAnalyzer, opts());
    expect(zero.reason).toMatch(/master/);
    const engineAbort = await measurePath(fakeMeasure({ taps, aborted: true }), inProcessAnalyzer, opts());
    expect(engineAbort.aborted).toBe(true);
    const ctl = new AbortController();
    ctl.abort();
    const m = fakeMeasure({ taps });
    const userAbort = await measurePath(m, inProcessAnalyzer, opts({ signal: ctl.signal }));
    expect(userAbort.aborted).toBe(true);
    expect(m.calls.length).toBe(0);
    const badBand = await measurePath(fakeMeasure({ taps }), inProcessAnalyzer, opts({ band: [1000, 1000] }));
    expect(badBand.reason).toMatch(/Banda/);
  });
});

describe('measurePrimary', () => {
  const n = 2 * fs;
  const g = gaussian(5);
  const ref = Float32Array.from({ length: n }, () => 0.05 * g());
  const gn = gaussian(6);
  const err = Float32Array.from({ length: n }, (_, i) => (i >= 240 ? 0.7 * ref[i - 240] : 0) + 1e-4 * gn());

  it('measures P = reference → error with coherence and reference PSD on the calibration grid', async () => {
    const p = await measurePrimary(async () => ({ ref, err }), inProcessAnalyzer, { fs, seconds: 2, band: [20, 1000] });
    expect(p.freqs.length).toBe(calibrationFreqs(fs, [20, 1000]).length);
    expect(p.psd!.length).toBe(p.freqs.length);
    expect(p.tonalF0).toBeNull();
    for (let k = 20; k < p.freqs.length; k += 50) {
      const w = (2 * Math.PI * p.freqs[k]) / fs;
      expect(p.re[k]).toBeCloseTo(0.7 * Math.cos(-w * 240), 2);
      expect(p.im[k]).toBeCloseTo(0.7 * Math.sin(-w * 240), 2);
      expect(p.coherence[k]).toBeGreaterThan(0.99);
    }
  });

  it('finds the f0 of a tonal noise', async () => {
    const tone = Float32Array.from({ length: n }, (_, i) => 0.1 * Math.sin((2 * Math.PI * 120 * i) / fs) + 0.05 * Math.sin((2 * Math.PI * 240 * i) / fs) + ref[i] * 0.01);
    const p = await measurePrimary(async () => ({ ref: tone, err: tone }), inProcessAnalyzer, { fs, seconds: 2, band: [20, 1000] });
    expect(p.tonalF0).not.toBeNull();
    expect(Math.abs(p.tonalF0! - 120)).toBeLessThan(0.05);
  });

  it('throws an Italian CalibrationError without the error mic', async () => {
    await expect(measurePrimary(async () => ({ ref, err: null }), inProcessAnalyzer, { fs, seconds: 2, band: [20, 1000] })).rejects.toBeInstanceOf(CalibrationError);
    await expect(
      measurePrimary(async () => ({ ref, err: new Float32Array(n) }), inProcessAnalyzer, { fs, seconds: 2, band: [20, 1000] }),
    ).rejects.toThrow(/errore/);
  });
});

/** PathData of a synthetic response on the calibration grid. */
function syntheticPath(h: (f: number) => { re: number; im: number }, delay: number, over: Partial<PathData> = {}): PathData {
  const freqs = Array.from(calibrationFreqs(fs, [20, 1000]));
  return {
    delaySamples: delay,
    sign: 1,
    qualityDb: 40,
    stdSamples: 0.02,
    freqs,
    re: freqs.map((f) => h(f).re),
    im: freqs.map((f) => h(f).im),
    ir: [],
    irOffset: 0,
    ...over,
  };
}

describe('looksLikeLoopback', () => {
  const flat = syntheticPath((f) => tapsAt([[240.3, 0.5]], f), 240.3);

  it('recognises a flat, pure-delay, clean path', () => {
    expect(looksLikeLoopback(flat)).toBe(true);
    expect(looksLikeLoopback(flat, { seconds: 240.3 / fs, fs })).toBe(true);
    expect(looksLikeLoopback(flat, { seconds: 0.012, fs })).toBe(false);
  });

  it('handles long latencies when fs is known', () => {
    const slow = syntheticPath((f) => tapsAt([[9600.4, 0.5]], f), 9600.4);
    expect(looksLikeLoopback(slow, { seconds: 0.2, fs })).toBe(true);
  });

  it('rejects reflections, a speaker roll-off and poor quality', () => {
    expect(looksLikeLoopback(syntheticPath((f) => tapsAt([[240, 0.5], [330, 0.15]], f), 240))).toBe(false);
    const hp = (f: number): { re: number; im: number } => {
      const d = tapsAt([[240, 0.5]], f);
      // 1st-order high-pass at 80 Hz: jf/(f0 + jf)
      const den = 80 * 80 + f * f;
      const hr = (f * f) / den;
      const hi = (80 * f) / den;
      return { re: d.re * hr - d.im * hi, im: d.re * hi + d.im * hr };
    };
    expect(looksLikeLoopback(syntheticPath(hp, 240))).toBe(false);
    expect(looksLikeLoopback({ ...flat, qualityDb: 12 })).toBe(false);
    expect(looksLikeLoopback({ ...flat, stdSamples: 0.8 })).toBe(false);
  });
});

describe('feedbackFirs', () => {
  it('rebuilds the lag-indexed IR (negative window start allowed) and designs bulk delay + taps', () => {
    const ir = Array.from({ length: 4000 }, (_, i) => Math.sin(i / 7) * Math.exp(-i / 900));
    const cal = createCalibration(fs, [20, 1000], 'single');
    cal.feedback[0] = { ...syntheticPath(() => ({ re: 1, im: 0 }), 300.4), ir, irOffset: -1000 };
    const [f0, f1] = feedbackFirs(cal, 256);
    expect(f1).toBeNull();
    expect(f0!.bulkDelay).toBe(300 - FEEDBACK_FIR_MARGIN);
    expect(f0!.taps.length).toBe(256);
    // Away from the fades the taps are the IR at lag bulkDelay + k.
    for (const k of [40, 100, 200]) expect(f0!.taps[k]).toBeCloseTo(ir[f0!.bulkDelay + k + 1000], 12);
    expect(feedbackFirs(cal, 1e6)[0]!.taps.length).toBe(1024);
  });
});

describe('solveFromCalibration', () => {
  /** P = 0.7·e^{-jω·1200}, S = 2·e^{-jω·720} (IR-backed), PSD flat, coherence 1. */
  function perfect(): CalibrationData {
    const cal = createCalibration(fs, [20, 1000], 'single');
    const freqs = Array.from(calibrationFreqs(fs, [20, 1000]));
    const p = freqs.map((f) => tapsAt([[1200, 0.7]], f));
    cal.primary = { freqs, re: p.map((v) => v.re), im: p.map((v) => v.im), coherence: freqs.map(() => 1), tonalF0: null, psd: freqs.map(() => 1e-6) };
    const ir = new Array<number>(400).fill(0);
    ir[100] = 2;
    cal.secondary[0] = { ...syntheticPath((f) => tapsAt([[720, 2]], f), 720), ir, irOffset: 620 };
    return cal;
  }

  it('aligns delay, gain and polarity so that the modelled residual is small', () => {
    const params = clampParams(undefined);
    params.masterDb = 0;
    // Flat conditioning (DC blocker only): a pure delay and gain can match P exactly.
    params.filters.hpEnabled = false;
    params.filters.lpEnabled = false;
    const out = solveFromCalibration(perfect(), params, fs);
    const c = out.params.channels[0];
    expect(out.result.causal).toBe(true);
    expect(c.enabled).toBe(true);
    expect(c.polarityInvert).toBe(true);
    expect(out.params.channels[1].enabled).toBe(false);
    expect(c.delayMs).toBeCloseTo(((1200 - 720 - D_MIN_SAMPLES) / fs) * 1000, 2);
    expect(c.gainDb).toBeCloseTo(20 * Math.log10(0.35), 1);
    expect(out.loopGainDb).toBeNull();
    // Modelled residual |P + G·S| at 200 Hz.
    const g = channelResponse(out.params, fs, 200, 0);
    const s = tapsAt([[720, 2]], 200);
    const pp = tapsAt([[1200, 0.7]], 200);
    const rr = pp.re + g.re * s.re - g.im * s.im;
    const ri = pp.im + g.re * s.im + g.im * s.re;
    expect(20 * Math.log10(Math.hypot(rr, ri) / 0.7)).toBeLessThan(-30);
  });

  it('explains what is missing or inconsistent instead of solving', () => {
    const params = clampParams(undefined);
    const wrongFs = solveFromCalibration({ ...perfect(), fs: 44100 }, params, fs);
    expect(wrongFs.advice[0]).toMatch(/44,1 kHz/);
    expect(wrongFs.params).toEqual(params);
    expect(solveFromCalibration({ ...perfect(), primary: null }, params, fs).advice[0]).toMatch(/percorso primario/);
    expect(solveFromCalibration({ ...perfect(), secondary: [null, null] }, params, fs).advice[0]).toMatch(/cassa/);
  });

  it('suggests the tonal mode and flags a non-causal setup', () => {
    const cal = perfect();
    cal.primary!.tonalF0 = 100;
    const p = cal.primary!;
    // Noise reaches the zone only 5 ms after the reference: broadband cancellation is not causal.
    const late = p.freqs.map((f) => tapsAt([[240, 0.7]], f));
    p.re = late.map((v) => v.re);
    p.im = late.map((v) => v.im);
    const out = solveFromCalibration(cal, clampParams(undefined), fs);
    expect(out.result.causal).toBe(false);
    expect(out.advice.some((a) => a.includes('troppo tardi'))).toBe(true);
    expect(out.advice.some((a) => a.includes('modalità tonale'))).toBe(true);
  });
});

describe('analyzers', () => {
  it('inProcessAnalyzer calls the pure functions', async () => {
    const x = Float32Array.from({ length: fs }, (_, i) => Math.sin((2 * Math.PI * 150 * i) / fs));
    expect(await inProcessAnalyzer.estimateF0(x, fs, 50, 500)).toEqual(estimateF0(x, fs, 50, 500));
  });

  /**
   * The real worker module behind a fake Worker: messages cross with structuredClone and transfer lists,
   * so transferred buffers really get detached, as in the browser.
   */
  type Handler = ((e: { data: unknown }) => void) | null;
  const g = globalThis as unknown as { self?: unknown; Worker?: unknown };
  let workerScope: { onmessage: Handler } | null = null;
  let current: FakeWorker | null = null;
  class FakeWorker {
    onmessage: Handler = null;
    onerror: ((e: { preventDefault(): void }) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    readonly url: string;
    sent = 0;
    /** 'crash': the worker dies on the next message; 'reject': it answers ok=false. */
    next: 'normal' | 'crash' | 'reject' = 'normal';
    constructor(url: URL, opts: { type: string }) {
      this.url = String(url);
      expect(opts.type).toBe('module');
      current = this;
    }
    postMessage(msg: AnalysisRequest, transfer: Transferable[]): void {
      this.sent++;
      const data = structuredClone(msg, { transfer });
      const mode = this.next;
      setTimeout(() => {
        if (mode === 'crash') this.onerror?.({ preventDefault() {} });
        else if (mode === 'reject') this.onmessage!({ data: { id: data.id, ok: false, error: 'memoria esaurita' } });
        else workerScope!.onmessage!({ data });
      }, 0);
    }
    terminate(): void {}
  }

  async function installWorker(): Promise<void> {
    if (workerScope === null) {
      const scope = {
        onmessage: null as Handler,
        postMessage(msg: AnalysisResponse, transfer: Transferable[]) {
          const data = structuredClone(msg, { transfer });
          setTimeout(() => current!.onmessage!({ data }), 0);
        },
      };
      g.self = scope;
      await import('../../src/workers/analysis.worker.ts');
      workerScope = scope;
    }
    g.Worker = FakeWorker;
  }

  afterEach(() => {
    delete g.Worker;
  });

  const n = fs;
  const gn = gaussian(3);
  const x = Float32Array.from({ length: n }, () => 0.1 * gn());
  const y = Float32Array.from({ length: n }, (_, i) => (i >= 100 ? 0.5 * x[i - 100] : 0));

  it('the worker client gives the in-process results and leaves the caller buffers intact', async () => {
    await installWorker();
    const a = createAnalyzer();
    expect(a.usingWorker).toBe(true);
    expect(current!.url).toMatch(/workers\/analysis\.worker\.ts$/);
    const freqs = calibrationFreqs(fs, [50, 1000]);
    const [t, d, f] = await Promise.all([
      a.transfer(x, y, fs, freqs),
      a.estimateDelay(y, x, fs, 50, 1000, 'acoustic'),
      a.estimateF0(x, fs, 50, 500),
    ]);
    expect(current!.sent).toBe(3);
    expect(t).toEqual(await inProcessAnalyzer.transfer(x, y, fs, freqs));
    expect(d).toEqual(await inProcessAnalyzer.estimateDelay(y, x, fs, 50, 1000, 'acoustic'));
    expect(f).toEqual(await inProcessAnalyzer.estimateF0(x, fs, 50, 500));
    // Copies were transferred, not the caller's arrays.
    expect(x.length).toBe(n);
    expect(y.length).toBe(n);
    expect(freqs.length).toBeGreaterThan(0);
    a.terminate();
    expect(a.usingWorker).toBe(false);
  });

  it('answers malformed requests with ok=false', async () => {
    await installWorker();
    createAnalyzer();
    const replies: unknown[] = [];
    current!.onmessage = (e) => replies.push(e.data);
    workerScope!.onmessage!({ data: { id: 7, op: 'nope' } });
    await new Promise((r) => setTimeout(r, 5));
    expect(replies).toEqual([{ id: 7, ok: false, error: 'richiesta sconosciuta' }]);
  });

  it('rejects on a worker-side error and falls back in-process when the worker dies', async () => {
    await installWorker();
    const a = createAnalyzer();
    current!.next = 'reject';
    await expect(a.estimateF0(x, fs, 50, 500)).rejects.toThrow(/Analisi non riuscita: memoria esaurita/);
    current!.next = 'crash';
    const r = await a.estimateF0(x, fs, 50, 500);
    expect(r).toEqual(await inProcessAnalyzer.estimateF0(x, fs, 50, 500));
    expect(a.usingWorker).toBe(false);
    // Later requests run in-process.
    const sent = current!.sent;
    await a.estimateF0(x, fs, 50, 500);
    expect(current!.sent).toBe(sent);
  });

  it('without Worker support it runs in-process', async () => {
    delete g.Worker;
    const a = createAnalyzer();
    expect(a.usingWorker).toBe(false);
    expect(await a.estimateF0(x, fs, 50, 500)).toEqual(await inProcessAnalyzer.estimateF0(x, fs, 50, 500));
  });

  /** Blocks of a drifting two-tone signal (with one NaN), fed to both an F0Window and an F0Tracker. */
  function liveBlocks(count: number, frames: number): Float32Array[] {
    const noise = gaussian(11);
    let n = 0;
    return Array.from({ length: count }, (_, k) =>
      Float32Array.from({ length: frames }, (_, i) => {
        const t = n++ / fs;
        if (k === 1 && i === 5) return Number.NaN;
        return 0.3 * Math.sin(2 * Math.PI * (120 + 0.5 * t) * t) + 0.1 * Math.sin(2 * Math.PI * 240 * t) + 0.01 * noise();
      }),
    );
  }

  it('F0Window holds the same window as F0Tracker and gives the same estimate', async () => {
    const w = new F0Window(fs, 1, 50, 500);
    const tr = new F0Tracker(fs, 1, 50, 500);
    const blocks = liveBlocks(40, 2048);
    let checked = 0;
    for (const [k, b] of blocks.entries()) {
      w.push(b);
      tr.push(b);
      // Partly filled (no estimate yet), just full, and wrapped around several times.
      if (k === 3 || k === 23 || k === 24 || k === 39) {
        const snap = w.snapshot();
        expect(snap.length).toBe(Math.min(fs, (k + 1) * 2048));
        expect(await w.estimate(inProcessAnalyzer)).toEqual(tr.estimate());
        checked++;
      }
    }
    expect(checked).toBe(4);
    expect(tr.estimate()).not.toBeNull();
    // The snapshot is a copy: later pushes do not change it.
    const snap = w.snapshot();
    const copy = snap.slice();
    w.push(blocks[0]!);
    expect(snap).toEqual(copy);
    w.reset();
    expect(w.snapshot().length).toBe(0);
    expect(() => new F0Window(0)).toThrow(RangeError);
  });

  it('F0Window through the analysis worker gives the main-thread result', async () => {
    await installWorker();
    const a = createAnalyzer();
    const w = new F0Window(fs, 2, 20, 2000);
    const tr = new F0Tracker(fs, 2, 20, 2000);
    for (const b of liveBlocks(60, 2048)) {
      w.push(b);
      tr.push(b);
    }
    const viaWorker = await w.estimate(a);
    expect(current!.sent).toBe(1);
    expect(a.usingWorker).toBe(true);
    expect(viaWorker).toEqual(tr.estimate());
    expect(viaWorker!.f0).toBeGreaterThan(100);
    // The worker dies: the same request is answered in-process.
    current!.next = 'crash';
    expect(await w.estimate(a)).toEqual(tr.estimate());
    expect(a.usingWorker).toBe(false);
  });
});

// ------------------------------------------------------------------------------- review fixes

describe('review fixes', () => {
  const band: [number, number] = [20, 1000];

  /** A path of integer-lag impulses (lags ≥ 0) stored like measurePath (IR from lag 0, grid response). */
  function impulsePath(arrivals: [number, number][]): PathData {
    const first = Math.min(...arrivals.map((a) => a[0]));
    const last = Math.max(...arrivals.map((a) => a[0]));
    const ir = new Array<number>(last + 64).fill(0);
    for (const [lag, gain] of arrivals) ir[lag] += gain;
    const freqs = calibrationFreqs(fs, band);
    const base: PathData = {
      delaySamples: first,
      sign: arrivals.find((a) => a[0] === first)![1] >= 0 ? 1 : -1,
      qualityDb: 40,
      stdSamples: 0.05,
      freqs: [],
      re: [],
      im: [],
      ir,
      irOffset: 0,
    };
    const r = pathResponse(base, fs, freqs);
    return { ...base, freqs: Array.from(freqs), re: Array.from(r.re), im: Array.from(r.im) };
  }

  /** Pure-delay primary path on the stored grid, coherence 1 (or `coherence`), flat PSD (or `psd`). */
  function delayCal(gain: number, delayS: number, over: { coherence?: number; psd?: (f: number) => number } = {}): CalibrationData {
    const cal = createCalibration(fs, band, 'single');
    const freqs = Array.from(calibrationFreqs(fs, band));
    cal.primary = {
      freqs,
      re: freqs.map((f) => gain * Math.cos(-2 * Math.PI * f * delayS)),
      im: freqs.map((f) => gain * Math.sin(-2 * Math.PI * f * delayS)),
      coherence: freqs.map(() => over.coherence ?? 1),
      tonalF0: null,
      psd: freqs.map((f) => (over.psd ? over.psd(f) : 1)),
    };
    cal.secondary[0] = impulsePath([[720, 1]]);
    return cal;
  }

  const solveParams = (): ReturnType<typeof clampParams> => {
    const p = clampParams(undefined);
    p.masterDb = 0;
    p.channels[1].enabled = false;
    return p;
  };

  /** Loop gain of `p` through `cal.feedback` after neutralisation with feedbackFirs(cal, taps), dB. */
  function neutralisedLoopDb(cal: CalibrationData, p: ReturnType<typeof clampParams>, taps = 256): number {
    const freqs = calibrationFreqs(fs, cal.band);
    const fir = feedbackFirs(cal, taps)[0]!;
    const fb = pathResponse(cal.feedback[0]!, fs, freqs);
    const h = irToResponse(fir.taps, fs, freqs);
    let max = 0;
    for (let k = 0; k < freqs.length; k++) {
      const a = (-2 * Math.PI * freqs[k] * fir.bulkDelay) / fs;
      const rr = fb.re[k] - (h.re[k] * Math.cos(a) - h.im[k] * Math.sin(a));
      const ri = fb.im[k] - (h.re[k] * Math.sin(a) + h.im[k] * Math.cos(a));
      const g = channelResponse(p, fs, freqs[k], 0);
      max = Math.max(max, Math.hypot(g.re * rr - g.im * ri, g.re * ri + g.im * rr));
    }
    return 20 * Math.log10(max);
  }

  it('the feedback FIR covers a weak direct sound before a stronger reflection (measured)', async () => {
    // Direct sound 0.3 at 768, a reflection 0.9 at 1068: the −6 dB "first arrival" is the reflection.
    const m = fakeMeasure({ taps: [[768, 0.3], [1068, 0.9]], noise: 1e-5 });
    const r = await measurePath(m, inProcessAnalyzer, opts({ mic: 'ref', previewFirst: false }));
    expect(r.ok).toBe(true);
    expect(r.delaySamples).toBeCloseTo(1068, 0);
    const cal = createCalibration(fs, band, 'single');
    cal.feedback[0] = r;
    const fir = feedbackFirs(cal, 256)[0]!;
    expect(fir.bulkDelay).toBeLessThanOrEqual(768 - 32);
    expect(fir.bulkDelay + fir.taps.length).toBeGreaterThan(1068 + 128);
    const freqs = calibrationFreqs(fs, band);
    const F = pathResponse(r, fs, freqs);
    const H = irToResponse(fir.taps, fs, freqs);
    let eF = 0;
    let eR = 0;
    for (let k = 0; k < freqs.length; k++) {
      if (freqs[k] < 60 || freqs[k] > 500) continue;
      const a = (-2 * Math.PI * freqs[k] * fir.bulkDelay) / fs;
      eF += F.re[k] ** 2 + F.im[k] ** 2;
      eR += (F.re[k] - (H.re[k] * Math.cos(a) - H.im[k] * Math.sin(a))) ** 2 + (F.im[k] - (H.re[k] * Math.sin(a) + H.im[k] * Math.cos(a))) ** 2;
    }
    expect(10 * Math.log10(eF / eR)).toBeGreaterThan(20);
    // A single arrival keeps the plain design (taps, bulk = arrival − margin).
    cal.feedback[0] = impulsePath([[700, 0.5]]);
    const plain = feedbackFirs(cal, 256)[0]!;
    expect(plain.taps.length).toBe(256);
    expect(plain.bulkDelay).toBe(700 - FEEDBACK_FIR_MARGIN);
  });

  /** Noise where the default conditioning (HP 20 Hz, LP 500 Hz) passes the anti-noise unchanged. */
  const midBand = (f: number): number => (f >= 60 && f <= 400 ? 1 : 1e-4);

  it('a reflection after the FIR window lengthens it (up to 1024 taps) instead of being left in the loop', () => {
    const cal = delayCal(0.7, 0.03, { psd: midBand });
    // Direct sound at 768, a strong reflection 12 ms later (beyond 256 taps).
    cal.feedback[0] = impulsePath([[768, 1], [768 + 576, 0.9]]);
    const fir = feedbackFirs(cal, 256)[0]!;
    expect(fir.taps.length).toBe(576 + 256);
    const out = solveFromCalibration(cal, solveParams(), fs);
    expect(out.params.channels[0].enabled).toBe(true);
    expect(out.loopLimitedDb).toBe(0);
    expect(neutralisedLoopDb(cal, out.params)).toBeLessThan(-40);
    expect(out.result.predictedAttenuationDb).toBeGreaterThan(12);
  });

  it('a loop left above −6 dB after neutralisation lowers the channel gains to keep the 6 dB margin', () => {
    const cal = delayCal(0.7, 0.03, { psd: midBand });
    // A reflection 40 ms after the direct sound: out of reach of even 1024 taps.
    cal.feedback[0] = impulsePath([[768, 0.5], [768 + 1920, 0.9]]);
    const full = solveFromCalibration(createCalibrationLike(cal), solveParams(), fs);
    const out = solveFromCalibration(cal, solveParams(), fs);
    const before = neutralisedLoopDb(cal, full.params);
    expect(before).toBeGreaterThan(LOOP_GAIN_DANGER_DB);
    expect(out.loopLimitedDb).toBeCloseTo(before - LOOP_GAIN_DANGER_DB, 1);
    // Reported for the params returned (the loop scales with the gains).
    expect(out.loopGainAfterDb).toBeCloseTo(LOOP_GAIN_DANGER_DB, 1);
    expect(out.params.channels[0].gainDb).toBeCloseTo(full.params.channels[0].gainDb - out.loopLimitedDb!, 6);
    expect(neutralisedLoopDb(cal, out.params)).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 0.01);
    expect(out.advice.some((a) => a.includes('ridotti di'))).toBe(true);
    // The prediction is for the lowered gains (less than the unconstrained one, still useful).
    expect(out.result.predictedAttenuationDb).toBeLessThan(full.result.predictedAttenuationDb - 1);
    expect(out.result.predictedAttenuationDb).toBeGreaterThan(1);

    // A loop that would need the gains far down: not applied.
    cal.feedback[0] = impulsePath([[768, 1], [768 + 1920, 10]]);
    const hopeless = solveFromCalibration(cal, solveParams(), fs);
    expect(hopeless.params).toEqual(solveParams());
    expect(hopeless.result.predictedAttenuationDb).toBeLessThan(1);
    expect(hopeless.advice.some((a) => a.startsWith('Rischio di innesco') && a.includes('non sono stati cambiati'))).toBe(true);
  });

  /** The same calibration without feedback paths (the unconstrained solution). */
  function createCalibrationLike(cal: CalibrationData): CalibrationData {
    return { ...cal, feedback: [null, null] };
  }

  it('the predicted attenuation includes the incoherent part of the zone noise', () => {
    const coherent = solveFromCalibration(delayCal(0.7, 0.03, { psd: midBand }), solveParams(), fs);
    expect(coherent.result.predictedAttenuationDb).toBeGreaterThan(12);
    const cal = delayCal(0.7, 0.03, { coherence: 0.75, psd: midBand });
    const out = solveFromCalibration(cal, solveParams(), fs);
    const bound = -10 * Math.log10(1 - 0.75);
    expect(out.result.predictedAttenuationDb).toBeLessThanOrEqual(bound);
    expect(out.result.predictedAttenuationDb).toBeGreaterThan(bound - 1);
  });

  it('tonal mode: a 0.2 s primary delay gives the same per-harmonic cancellation as a 0.05 s one', () => {
    const f0 = 101.37;
    const attenuations = (delayS: number): number[] => {
      const cal = delayCal(0.7, delayS, {
        psd: (f) => {
          const h = Math.round(f / f0);
          return h >= 1 && h <= 4 && Math.abs(f - h * f0) < 6 ? 1 : 1e-3;
        },
      });
      const p = solveParams();
      p.phaseMode = 'tonal';
      p.tonal = { f0, harmonics: 4, q: 30 };
      const out = solveFromCalibration(cal, p, fs);
      return [1, 2, 3, 4].map((h) => {
        const f = h * f0;
        const pr = 0.7 * Math.cos(-2 * Math.PI * f * delayS);
        const pi = 0.7 * Math.sin(-2 * Math.PI * f * delayS);
        const g = channelResponse(out.params, fs, f, 0);
        const s = tapsAt([[720, 1]], f);
        return -20 * Math.log10(Math.hypot(pr + g.re * s.re - g.im * s.im, pi + g.re * s.im + g.im * s.re) / 0.7);
      });
    };
    const near = attenuations(0.05);
    const far = attenuations(0.2);
    for (let h = 0; h < 4; h++) {
      expect(near[h]).toBeGreaterThan(10);
      expect(far[h]).toBeGreaterThan(near[h] - 1);
    }
  });

  it('tonal mode: a tone measured off tonal.f0 beyond the resonator bandwidth is not solved', () => {
    const cal = delayCal(0.7, 0.006);
    const p = solveParams();
    p.phaseMode = 'tonal';
    p.tonal = { f0: 100, harmonics: 4, q: 30 };
    cal.primary!.tonalF0 = 104;
    const off = solveFromCalibration(cal, p, fs);
    expect(off.params).toEqual(clampParams(p));
    expect(off.advice[0]).toMatch(/104,00 Hz.*100,00 Hz/);
    // Within the half bandwidth (100/60 Hz), or a harmonic of f0: solved.
    cal.primary!.tonalF0 = 101;
    expect(solveFromCalibration(cal, p, fs).params.channels[0].enabled).toBe(true);
    cal.primary!.tonalF0 = 200.5;
    const harmonic = solveFromCalibration(cal, p, fs);
    expect(harmonic.advice.some((a) => a.includes('tono misurato'))).toBe(false);
  });

  it('two-level check: a path whose gain depends on the level is flagged, a linear noisy one is not', async () => {
    // AGC-like: quiet stimuli (the preview) come back 10 dB louder.
    const agc: MeasureFn = async (stimulus, _ch, post) => {
      const peak = stimulus.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
      const gain = peak > 0.03 ? 0.3 : 0.3 * Math.pow(10, 10 / 20);
      const rec = new Float32Array(stimulus.length + post);
      for (let i = 0; i < stimulus.length && i + 700 < rec.length; i++) rec[i + 700] = gain * stimulus[i];
      return { ref: rec, err: rec, masterLin: 1, aborted: false };
    };
    const flagged = await measurePath(agc, inProcessAnalyzer, opts());
    expect(flagged.ok).toBe(false);
    expect(flagged.reason).toMatch(/dipende dal livello.*10,0 dB.*Miglioramenti audio/);
    // Linear but with loud ambient noise: the preview is noisy, the check must not cry wolf.
    for (const seed of [1, 2, 3]) {
      const g = gaussian(seed);
      const noisy: MeasureFn = async (stimulus, _ch, post) => {
        const rec = new Float32Array(stimulus.length + post);
        for (let i = 0; i < rec.length; i++) rec[i] = 3e-3 * g() + (i >= 700 && i - 700 < stimulus.length ? 0.3 * stimulus[i - 700] : 0);
        return { ref: rec, err: rec, masterLin: 1, aborted: false };
      };
      const r = await measurePath(noisy, inProcessAnalyzer, opts());
      expect(r.reason).toBeUndefined();
      expect(r.ok).toBe(true);
    }
  });

  it('geometry only: the synthetic speaker → zone path takes the chain polarity measured on F', () => {
    const geometryCal = (sign: 1 | -1 | 0): CalibrationData => {
      const cal = createCalibration(fs, band, 'single', 20);
      if (sign !== 0) cal.feedback[0] = impulsePath([[600, 0.05 * sign]]);
      cal.geometry = { micToZoneM: 10, sourceAngleDeg: 0, speakerToZoneM: [1, 1], tempC: 20 };
      return cal;
    };
    const p = solveParams();
    const normal = solveFromGeometry(geometryCal(1), p, fs, 0.01);
    const inverted = solveFromGeometry(geometryCal(-1), p, fs, 0.01);
    expect(normal.params.channels[0].enabled).toBe(true);
    expect(inverted.params.channels[0].polarityInvert).toBe(!normal.params.channels[0].polarityInvert);
    expect(normal.advice.some((a) => a.includes('Polarità della catena non verificata'))).toBe(false);
    const unknown = solveFromGeometry(geometryCal(0), p, fs, 0.01);
    expect(unknown.params.channels[0].polarityInvert).toBe(normal.params.channels[0].polarityInvert);
    expect(unknown.advice.some((a) => a.includes('Polarità della catena non verificata'))).toBe(true);
  });

  it('loopbackSuspectPaths names the paths with a cable signature', () => {
    const cal = createCalibration(fs, band, 'shared');
    cal.feedback[1] = syntheticPath((f) => tapsAt([[96, 1]], f), 96, { qualityDb: 40, stdSamples: 0.05 });
    cal.secondary[0] = impulsePath([[720, 1], [900, 0.4]]);
    expect(loopbackSuspectPaths(cal)).toEqual(['cassa R → microfono di riferimento']);
    expect(loopbackSuspectPaths(null)).toEqual([]);
  });
});

describe('two-level check with an ambient tone', () => {
  it('a loud stationary hum does not read as a level-dependent path', async () => {
    // A machine hum with 8 harmonics, each well above the quiet preview's sweep in its band.
    const tone = [1, 2, 3, 4, 5, 6, 7, 8].map((h) => ({ f: 97.3 * h, a: 0.05 / h }));
    const m = fakeMeasure({ taps: [[700, 0.3]], noise: 1e-4, tone });
    const r = await measurePath(m, inProcessAnalyzer, opts());
    expect(r.reason).toBeUndefined();
    expect(r.ok).toBe(true);
  });
});
