import { describe, expect, it } from 'vitest';
import type { AudioStats } from '../../src/audio/audio-engine.ts';
import { calibrationFreqs, createCalibration } from '../../src/audio/calibration.ts';
import type { PathData } from '../../src/audio/calibration.ts';
import type { Preset } from '../../src/app/presets.ts';
import { DEFAULT_PARAMS, clampParams } from '../../src/dsp/params.ts';
import { speedOfSound } from '../../src/dsp/physics.ts';
import {
  appendTapChannel,
  causality,
  conditioningDelayS,
  defaultPresetName,
  estimatedLatencyS,
  fadeEdges,
  geometryToDraft,
  loopbackSuspects,
  mergeZoneSolutions,
  parseGeometry,
  parseWizardPrefs,
  pathSummary,
  powerMeanDb,
  presetsWithCalibration,
  refLevelAdvice,
  rt60ForAutotune,
  serializeWizardPrefs,
  stimulusLevels,
  toPathData,
} from '../../src/ui/wizard/logic.ts';

const FS = 48000;

const stats = (over: Partial<AudioStats> = {}): AudioStats => ({
  sampleRate: FS,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: null,
  sinkId: '',
  ...over,
});

function flatPath(delay: number): PathData {
  const freqs = [...calibrationFreqs(FS, [20, 1000])];
  return {
    delaySamples: delay,
    sign: 1,
    qualityDb: 40,
    stdSamples: 0.05,
    freqs,
    re: freqs.map((f) => Math.cos((-2 * Math.PI * f * delay) / FS)),
    im: freqs.map((f) => Math.sin((-2 * Math.PI * f * delay) / FS)),
    ir: [],
    irOffset: 0,
  };
}

describe('levels', () => {
  it('plays the stimuli at −20 dBFS, lower with a low ceiling', () => {
    expect(stimulusLevels(-12)).toEqual({ noiseRmsDbfs: -20, sweepPeakDbfs: -20 });
    expect(stimulusLevels(-1).noiseRmsDbfs).toBe(-20);
    expect(stimulusLevels(-24).noiseRmsDbfs).toBe(-32);
    expect(stimulusLevels(Number.NaN).noiseRmsDbfs).toBe(-20);
  });

  it('fades both ends of a noise chunk without touching the middle', () => {
    const x = new Float32Array(4800).fill(1);
    fadeEdges(x, FS, 30);
    expect(x[0]).toBeLessThan(0.01);
    expect(x[x.length - 1]).toBeLessThan(0.01);
    expect(x[2400]).toBe(1);
    expect(x[1439]).toBeGreaterThan(0.99);
    const tiny = fadeEdges(new Float32Array(3).fill(1), FS, 30);
    expect([...tiny].every((v) => Number.isFinite(v) && v <= 1)).toBe(true);
  });

  it('advises on the reference input gain', () => {
    expect(refLevelAdvice(null, null).level).toBe('none');
    expect(refLevelAdvice(-120, -120).level).toBe('none');
    expect(refLevelAdvice(-60, -48).level).toBe('low');
    expect(refLevelAdvice(-20, -1).level).toBe('high');
    expect(refLevelAdvice(-30, -18).level).toBe('ok');
    expect(refLevelAdvice(-60, -48).text).toContain('guadagno d’ingresso');
  });

  it('averages levels in power', () => {
    expect(powerMeanDb([-20, -20])).toBeCloseTo(-20, 9);
    expect(powerMeanDb([-20, null])).toBeCloseTo(-20, 9);
    expect(powerMeanDb([0, -100])).toBeCloseTo(10 * Math.log10(0.5), 6);
    expect(powerMeanDb([])).toBeNull();
    expect(powerMeanDb([null, Number.NaN])).toBeNull();
  });
});

describe('latency and causality', () => {
  it('sums the browser latencies, null without an output latency', () => {
    expect(estimatedLatencyS(stats())).toBeCloseTo(0.0197, 9);
    expect(estimatedLatencyS(stats({ outputLatencyMs: null }))).toBeNull();
    expect(estimatedLatencyS(stats({ baseLatencyMs: null, trackLatencyMs: null }))).toBeCloseTo(0.012, 9);
  });

  it('compares the acoustic head start with latency + speaker distance + filters', () => {
    const c = speedOfSound(20);
    const r = causality({ micToZoneM: 10, sourceAngleDeg: 0, speakerToZoneM: 1, tempC: 20, latencyS: 0.02, conditioningS: 0.001 });
    expect(r.headStartS).toBeCloseTo(10 / c, 9);
    expect(r.neededS).toBeCloseTo(0.02 + 1 / c + 0.001, 9);
    expect(r.causal).toBe(true);
    expect(r.marginS).toBeCloseTo(r.headStartS - r.neededS, 12);
    expect(r.extraLeadM).toBe(0);
    expect(r.warnAngle).toBe(false);

    const late = causality({ micToZoneM: 3, sourceAngleDeg: 0, speakerToZoneM: 1, tempC: 20, latencyS: 0.02, conditioningS: 0.001 });
    expect(late.causal).toBe(false);
    expect(late.extraLeadM).toBeCloseTo(-late.marginS * c, 6);
  });

  it('warns when the source is more than ~45° off the mic → zone line', () => {
    const r = causality({ micToZoneM: 30, sourceAngleDeg: 50, speakerToZoneM: 1, tempC: 20, latencyS: 0.02, conditioningS: 0 });
    expect(r.cosTheta).toBeCloseTo(Math.cos((50 * Math.PI) / 180), 9);
    expect(r.warnAngle).toBe(true);
  });

  it('includes a positive conditioning group delay for the default filters', () => {
    const d = conditioningDelayS(clampParams(DEFAULT_PARAMS), FS, [20, 1000]);
    expect(d).toBeGreaterThan(0);
    expect(d).toBeLessThan(0.01);
  });
});

describe('geometry input', () => {
  it('parses Italian decimals and ignores R for one speaker', () => {
    const g = parseGeometry({ micToZoneM: '6,5', speakerToZoneM: ['1,25', ''], sourceAngleDeg: '15', tempC: '−5' }, [0]);
    expect(g.ok).toBe(true);
    if (g.ok) {
      expect(g.value.micToZoneM).toBe(6.5);
      expect(g.value.speakerToZoneM).toEqual([1.25, 1.25]);
      expect(g.value.tempC).toBe(-5);
    }
  });

  it('lists every invalid field in Italian', () => {
    const g = parseGeometry({ micToZoneM: '', speakerToZoneM: ['0', 'abc'], sourceAngleDeg: '200', tempC: '99' }, [0, 1]);
    expect(g.ok).toBe(false);
    if (!g.ok) {
      expect(g.errors).toHaveLength(5);
      expect(g.errors[0]).toContain('microfono');
      expect(g.errors.join(' ')).toContain('cassa R');
    }
  });

  it('round-trips stored numbers through the text draft', () => {
    const v = { micToZoneM: 7.25, speakerToZoneM: [1.5, 2] as [number, number], sourceAngleDeg: 12.5, tempC: 21 };
    const g = parseGeometry(geometryToDraft(v), [0, 1]);
    expect(g.ok && g.value).toEqual(v);
  });
});

describe('saved calibrations and paths', () => {
  const preset = (id: string, createdAt: number, calibration?: unknown): Preset => ({
    id,
    name: id,
    createdAt,
    deviceKey: '',
    sampleRate: FS,
    params: clampParams(DEFAULT_PARAMS),
    ...(calibration !== undefined ? { calibration } : {}),
  });

  it('keeps only presets with a valid calibration, most recent first', () => {
    const cal = createCalibration(FS, [20, 1000], 'single');
    const list = [preset('a', 1, cal), preset('b', 3, cal), preset('c', 2), preset('d', 4, { version: 9 })];
    expect(presetsWithCalibration(list).map((x) => x.preset.id)).toEqual(['b', 'a']);
  });

  it('names presets with the date', () => {
    expect(defaultPresetName(new Date(2026, 9, 30, 9, 5))).toBe('Calibrazione 30/10/2026 09:05');
  });

  it('flags paths that look like an electrical loopback', () => {
    const cal = createCalibration(FS, [20, 1000], 'shared');
    expect(loopbackSuspects(cal)).toEqual([]);
    expect(loopbackSuspects(null)).toEqual([]);
    cal.secondary[1] = flatPath(500);
    expect(loopbackSuspects(cal)).toEqual(['cassa R → microfono nella zona']);
    // With the loopback latency known, a different delay is not the cable.
    cal.loopbackLatencyS = 100 / FS;
    expect(loopbackSuspects(cal)).toEqual([]);
  });

  it('stores only the plain PathData fields', () => {
    const r = { ...flatPath(10), ok: true, reason: 'x', aborted: false, rt60S: 0.4 };
    const p = toPathData(r);
    expect(Object.keys(p).sort()).toEqual(['delaySamples', 'freqs', 'im', 'ir', 'irOffset', 'qualityDb', 're', 'rt60S', 'sign', 'stdSamples']);
    expect(JSON.parse(JSON.stringify(p))).toEqual(p);
  });

  it('summarises a path', () => {
    const s = pathSummary({ ...flatPath(480), sign: -1 }, FS);
    expect(s.delayMs).toBeCloseTo(10, 9);
    expect(s.inverted).toBe(true);
    expect(s.peakDb).toBeCloseTo(0, 6);
  });

  it('settles the autotune on the longest measured decay', () => {
    expect(rt60ForAutotune(null)).toBe(0.3);
    const cal = createCalibration(FS, [20, 1000], 'shared');
    cal.secondary = [{ ...flatPath(1), rt60S: 0.4 }, { ...flatPath(1), rt60S: 0.7 }];
    expect(rt60ForAutotune(cal)).toBe(0.7);
    cal.secondary = [{ ...flatPath(1), rt60S: 9 }, null];
    expect(rt60ForAutotune(cal)).toBe(2);
  });

  it('merges per-zone solutions channel by channel', () => {
    const base = clampParams(DEFAULT_PARAMS);
    const a = clampParams({ ...base, channels: [{ ...base.channels[0], delayMs: 3 }, { ...base.channels[1], enabled: false }] });
    const b = clampParams({ ...base, channels: [{ ...base.channels[0], enabled: false }, { ...base.channels[1], delayMs: 7 }] });
    const m = mergeZoneSolutions(base, [a, b]);
    expect(m.channels[0].delayMs).toBe(3);
    expect(m.channels[1].delayMs).toBe(7);
    expect(m.channels[0].enabled && m.channels[1].enabled).toBe(true);
    // A zone without a useful solution keeps its speaker off, never the unsolved base values (calib-ui-2).
    const off = mergeZoneSolutions(base, [a, null]);
    expect(off.channels[0].delayMs).toBe(3);
    expect(off.channels[1]).toEqual({ ...base.channels[1], enabled: false });
    expect(mergeZoneSolutions(base, [null, null]).channels.map((c) => c.enabled)).toEqual([false, false]);
  });

  it('copies one channel of planar tap blocks', () => {
    const frames = 4;
    const block = Float32Array.from({ length: 5 * frames }, (_, i) => i);
    const dst = new Float32Array(6);
    let n = appendTapChannel(block, frames, 4, dst, 0);
    n = appendTapChannel(block, frames, 4, dst, n);
    expect(n).toBe(6);
    expect([...dst]).toEqual([16, 17, 18, 19, 16, 17]);
    expect(appendTapChannel(block, frames, 4, dst, n)).toBe(6);
  });
});

describe('wizard preferences', () => {
  it('falls back to defaults on missing or broken data', () => {
    expect(parseWizardPrefs(null).layout).toBe('A');
    expect(parseWizardPrefs('{not json').layout).toBe('A');
    const odd = parseWizardPrefs(JSON.stringify({ layout: 'Z', geometry: { micToZoneM: 5, speakerToZoneM: 'x' } }));
    expect(odd.layout).toBe('A');
    expect(odd.geometry.micToZoneM).toBe('');
    expect(odd.geometry.speakerToZoneM).toEqual(['', '']);
  });

  it('round-trips', () => {
    const p = { layout: 'B' as const, geometry: { micToZoneM: '6,5', speakerToZoneM: ['1', '2'] as [string, string], sourceAngleDeg: '0', tempC: '20' } };
    expect(parseWizardPrefs(serializeWizardPrefs(p))).toEqual(p);
  });
});
