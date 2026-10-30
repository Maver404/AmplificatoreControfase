/**
 * Howl protection in the closed-loop simulator (fs = 48 kHz, block 128): the real AntiPhaseEngine with
 * default parameters (master and channel gain set for the wanted loop gain, right channel off) and a
 * speaker → reference path that is closed at a known time. The loop delay is the plant's L_e plus the
 * acoustic delay plus the engine's one-sample minimum (the conditioning adds its ~0.5 ms group delay);
 * the loop gain is the peak of |conditioning · F| times the commanded gain. Ambient broadband noise
 * starts ~40 dB below the ceiling. Requirements:
 *  (a) loop gain 1.5 with a 3–100 ms loop: howl event and mute within 1 s of closing the loop;
 *  (b) any loop gain > 1 (1.1–2) with a loop up to 500 ms: muted within 400 ms of the limiter starting to
 *      reduce gain in a sustained way, i.e. by > 3 dB (as required) or by > 0.5 dB (stricter: loop gains
 *      ≤ 1.2 never reach 3 dB) in ≥ 80 % of a 200 ms window;
 *  (b′) loops longer than 600 ms (the 500 ms channel delay plus device latency; v-dsp H3), loop gain
 *      1.3–1.5: muted within two trips around the loop of sustained limiting (a long loop's howl is held
 *      unevenly at first: only the loudest part of each trip reaches the ceiling);
 *  (c) the output never exceeds the ceiling;
 *  (d) a chain wider than ~2 kHz (Bessel LP at 4 kHz, or off) with a speaker → reference path that passes
 *      only 2.7–4.3 kHz (the loop peaking at ~3.3–3.5 kHz), so that the howl is a dense comb above the ~6 kHz
 *      periodicity analysis' band, under low-frequency anti-noise: loop gain 1.2 or 1.5, 12.5–200 ms, muted
 *      within 400 ms of sustained limiting, and within 1.5 s of closing the loop for loop gain 1.5 and loops
 *      up to 100 ms (the detector analyses a wide chain at ~12 kHz);
 * and no trigger in 10 s of legitimate signals, some of them held at the ceiling by the limiter, with the
 * default chain and with wide ones (and while the chain switches between narrow and wide).
 * HOWL_SIM_REPORT=1 prints the measured times.
 */
import { afterAll, describe, expect, it } from 'vitest';
import { biquadResponse, highpass, lowpass, onePoleLowpass } from '../../src/dsp/biquad.ts';
import { cAbs, cMul } from '../../src/dsp/complex.ts';
import type { Complex } from '../../src/dsp/complex.ts';
import { clampParams, dbToLin } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { ConditioningChain, conditioningResponse } from '../../src/dsp/response.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';
import { SimRig } from './harness.ts';
import type { NoiseSpec, PathSpec, PlantConfig } from './plant.ts';

const fs = 48000;
const BLOCK = 128;
/** Plant L_e: one block plus 15 samples, so the shortest loop is 144 samples = 3 ms. */
const LE_SAMPLES = 143;
const REPORT = process.env.HOWL_SIM_REPORT === '1';
const report: string[] = [];
afterAll(() => {
  if (REPORT) console.log(report.join('\n'));
});

/** Default parameters with master −10 dB and channel gain `gainDb` on the left channel only. */
function engineParams(gainDb = -10): EngineParams {
  const p = clampParams(undefined);
  p.masterDb = -10;
  p.channels[0].gainDb = gainDb;
  p.channels[1].enabled = false;
  return p;
}

/** engineParams() with the Bessel LP at `lpHz` (null: off), a chain wider than HOWL_WIDE_BAND_HZ. */
function wideParams(lpHz: number | null, gainDb = -10): EngineParams {
  const p = engineParams(gainDb);
  if (lpHz === null) p.filters.lpEnabled = false;
  else p.filters.lpFreq = lpHz;
  return p;
}

/**
 * A speaker → reference path that passes only 2.7–4.3 kHz, flat within ~1 dB over 3–3.8 kHz: a 41-tap
 * band-pass FIR (Hamming-windowed sinc, 0.42 ms of group delay) as arrivals 1/fs apart. With the chain the
 * loop peaks at ~3.3 kHz (LP 4 kHz) or ~3.5 kHz (LP off); loop gain 1.2 is above 1 at ~2.7–4.0 / 2.9–4.1 kHz
 * only, 1.5 at ~2.4–4.2 / 2.6–4.4 kHz, and ≤ 0.6 below 2 kHz and above 6 kHz. Flat, so the howl stays a dense
 * comb (no narrow peak) for a long time.
 */
const BAND_PATH: Omit<PathSpec, 'delayS' | 'gain'> = (() => {
  const m = 41;
  const h = Array.from({ length: m }, (_, k) => {
    const t = k - (m - 1) / 2;
    const sinc = (f: number): number => (t === 0 ? (2 * f) / fs : Math.sin((2 * Math.PI * f * t) / fs) / (Math.PI * t));
    return (sinc(4300) - sinc(2700)) * (0.54 - 0.46 * Math.cos((2 * Math.PI * k) / (m - 1)));
  });
  // The first tap is the main arrival (gain 1): the others relative to it.
  return { reflections: h.slice(1).map((v, k) => ({ delayS: (k + 1) / fs, gain: v / h[0] })) };
})();

const ceilingOf = (p: EngineParams): number => dbToLin(p.ceilingDb);
const commanded = (p: EngineParams): number => dbToLin(p.masterDb + p.channels[0].gainDb);

/** Response of a plant path (delays excluded: only the shape matters for the loop-gain peak). */
function pathShape(spec: PathSpec, f: number): Complex {
  let re = spec.gain;
  let im = 0;
  for (const r of spec.reflections ?? []) {
    re += r.gain * Math.cos(2 * Math.PI * f * r.delayS);
    im -= r.gain * Math.sin(2 * Math.PI * f * r.delayS);
  }
  let h: Complex = { re: spec.invert === true ? -re : re, im: spec.invert === true ? -im : im };
  if (spec.lowpassHz !== undefined) h = cMul(h, biquadResponse(onePoleLowpass(fs, spec.lowpassHz), f, fs));
  return h;
}

/** Peak over frequency of |conditioning(f)·F(f)| for a unit-gain F of the given shape. */
function loopPeak(p: EngineParams, shape: Omit<PathSpec, 'delayS'>): number {
  let m = 0;
  for (let f = 5; f < 0.45 * fs; f *= 1.001) {
    m = Math.max(m, cAbs(cMul(conditioningResponse(p, fs, f), pathShape({ ...shape, delayS: 0 }, f))));
  }
  return m;
}

/** Power gain of the conditioning for the plant's band-passed noise (2nd-order Butterworth lo…hi). */
function noiseGain(p: EngineParams, lo: number, hi: number): number {
  const hp = highpass(fs, lo, Math.SQRT1_2);
  const lp = lowpass(fs, Math.min(hi, 0.45 * fs), Math.SQRT1_2);
  let num = 0;
  let den = 0;
  for (let f = 1; f < fs / 2; f += 2) {
    const b = (cAbs(biquadResponse(hp, f, fs)) * cAbs(biquadResponse(lp, f, fs))) ** 2;
    num += b * cAbs(conditioningResponse(p, fs, f)) ** 2;
    den += b;
  }
  return Math.sqrt(num / den);
}

/** Reference-mic RMS of plant noise (band lo…hi) giving an output RMS `relDb` re the ceiling. */
function noiseRmsFor(p: EngineParams, relDb: number, lo = 20, hi = 2000): number {
  return (ceilingOf(p) * dbToLin(relDb)) / (commanded(p) * noiseGain(p, lo, hi));
}

/** Reference-mic amplitude of a tone at f giving an output amplitude `relDb` re the ceiling. */
function toneAmpFor(p: EngineParams, f: number, relDb: number): number {
  return (ceilingOf(p) * dbToLin(relDb)) / (commanded(p) * cAbs(conditioningResponse(p, fs, f)));
}

interface LoopSpec {
  loopGain: number;
  loopMs: number;
  noise: NoiseSpec;
  /** Shape of the speaker → reference path (default: a single arrival). */
  shape?: Omit<PathSpec, 'delayS' | 'gain'>;
  seed?: number;
  /** Engine parameters (default engineParams()). */
  params?: EngineParams;
}

function loopPlant(p: EngineParams, s: LoopSpec): PlantConfig {
  const shape = { gain: 1, ...s.shape };
  const kF = s.loopGain > 0 ? s.loopGain / (commanded(p) * loopPeak(p, shape)) : 0;
  const scale = (r: { delayS: number; gain: number }) => ({ delayS: r.delayS, gain: r.gain * kF });
  const acoustic = s.loopMs / 1000 - (LE_SAMPLES + 1) / fs;
  if (acoustic < -1e-12) throw new RangeError('loop shorter than L_e + 1 sample');
  return {
    fs,
    noise: s.noise,
    latencyS: LE_SAMPLES / fs,
    primary: { delayS: 0.01, gain: 1 },
    secondary: [{ delayS: 0.002, gain: 1 }, null],
    feedback: [{ ...shape, delayS: Math.max(0, acoustic), gain: kF, reflections: shape.reflections?.map(scale) }, null],
    sensorNoiseRms: 1e-4,
    seed: s.seed ?? 1,
  };
}

interface LoopOutcome {
  /** Seconds from closing the loop to the howl event (null: none within the run). */
  eventS: number | null;
  reason: string | null;
  freqHz: number | null;
  /** Seconds from closing the loop to sustained limiting (GR > 0.5 dB / > 3 dB), null if never before the event. */
  lim05S: number | null;
  lim3S: number | null;
  /** Output RMS before closing the loop, dB re the ceiling. */
  ambientDb: number;
  maxAbsOut: number;
  /** Largest |output| over the last 30 ms of the 60 ms that follow the event. */
  tailPeak: number;
  autoMuted: boolean;
}

/** First time the per-block GR exceeds `db` in ≥ 80 % of the blocks of a 200 ms window (s from close). */
function sustainedGr(gr: number[], db: number): number | null {
  const w = Math.round((0.2 * fs) / BLOCK);
  let count = 0;
  for (let i = 0; i < gr.length; i++) {
    if (gr[i] > db) count++;
    if (i >= w && gr[i - w] > db) count--;
    if (i >= w - 1 && count >= 0.8 * w) return ((i - w + 1) * BLOCK) / fs;
  }
  return null;
}

function runLoop(s: LoopSpec, maxS: number): LoopOutcome {
  const p = s.params ?? engineParams();
  const gr: number[] = [];
  let closed = false;
  const rig = new SimRig(loopPlant(p, s), {
    engine: { meterHz: fs / BLOCK },
    onMessage: (m) => {
      if (closed && m.type === 'meters') gr.push(m.limiterGrDb[0]);
    },
  });
  rig.plant.feedbackScale[0] = 0;
  rig.start(p, 0.3);
  const before = rig.run(0.3, ['outL']).outL!;
  let e = 0;
  for (const v of before) e += v * v;
  const ambientDb = 10 * Math.log10(e / before.length) - p.ceilingDb;
  rig.maxAbsOut = 0;
  closed = true;
  rig.plant.feedbackScale[0] = 1;
  const blocks = Math.ceil((maxS * fs) / BLOCK);
  let eventS: number | null = null;
  let reason: string | null = null;
  let freqHz: number | null = null;
  for (let b = 0; b < blocks && eventS === null; b++) {
    rig.run(BLOCK / fs);
    const h = rig.events.find((m) => m.type === 'howl');
    if (h !== undefined && h.type === 'howl') {
      eventS = ((b + 1) * BLOCK) / fs;
      reason = h.reason;
      freqHz = h.freqHz;
    }
  }
  const upTo = eventS === null ? gr.length : Math.round((eventS * fs) / BLOCK);
  const head = gr.slice(0, upTo);
  let tailPeak = 0;
  if (eventS !== null) {
    const tail = rig.run(0.06, ['outL']).outL!;
    for (let i = Math.round(0.03 * fs); i < tail.length; i++) tailPeak = Math.max(tailPeak, Math.abs(tail[i]));
  }
  return {
    eventS,
    reason,
    freqHz,
    lim05S: sustainedGr(head, 0.5),
    lim3S: sustainedGr(head, 3),
    ambientDb,
    maxAbsOut: rig.maxAbsOut,
    tailPeak,
    autoMuted: rig.autoMuted,
  };
}

const ms = (s: number | null): string => (s === null ? '—' : `${Math.round(s * 1000)} ms`);

function describeLoop(name: string, o: LoopOutcome): string {
  const f = o.freqHz === null ? '' : ` ${Math.round(o.freqHz)} Hz`;
  const event = o.eventS === null ? 'no trigger' : `${o.reason}${f} at ${ms(o.eventS)}`;
  const limited = `sustained GR > 0.5 dB at ${ms(o.lim05S)}, > 3 dB at ${ms(o.lim3S)}`;
  return `${name}: ${event}; ${limited}; ambient ${o.ambientDb.toFixed(1)} dB re ceiling`;
}

function expectMutedHowl(o: LoopOutcome, ceiling: number, broadbandOnly = true): void {
  expect(o.eventS).not.toBeNull();
  expect(o.autoMuted).toBe(true);
  // (c) never above the ceiling; muted (10 ms panic ramp) right after the event.
  expect(o.maxAbsOut).toBeLessThanOrEqual(ceiling);
  expect(o.tailPeak).toBe(0);
  if (broadbandOnly) {
    // The howl starts from ambient noise about 40 dB below the ceiling.
    expect(o.ambientDb).toBeGreaterThan(-46);
    expect(o.ambientDb).toBeLessThan(-36);
  }
}

describe('howl simulator: (a) loop gain 1.5, 3–100 ms, muted within 1 s of closing the loop', () => {
  const p = engineParams();
  const ambient = noiseRmsFor(p, -42);
  const sources: [string, NoiseSpec][] = [
    ['broadband', { broadband: { rms: ambient, lo: 20, hi: 2000 } }],
    [
      'tonal source',
      {
        broadband: { rms: ambient, lo: 20, hi: 2000 },
        tone: { f0: 100, amplitudes: [toneAmpFor(p, 100, -20), toneAmpFor(p, 200, -26), toneAmpFor(p, 300, -30)] },
      },
    ],
  ];
  for (const [src, noise] of sources) {
    for (const loopMs of [3, 12.5, 24, 50, 100]) {
      it(`${src}, ${loopMs} ms loop`, () => {
        const o = runLoop({ loopGain: 1.5, loopMs, noise, seed: 7 }, 3);
        report.push(describeLoop(`(a) G 1.5, ${loopMs} ms, ${src}`, o));
        expectMutedHowl(o, ceilingOf(p), noise.tone === undefined);
        expect(o.eventS!).toBeLessThan(1);
      });
    }
  }

  it('a room-like feedback path (two reflections, 1st-order low-pass at 2 kHz), 24 and 50 ms', () => {
    const shape = { reflections: [{ delayS: 0.0031, gain: 0.5 }, { delayS: 0.0117, gain: -0.3 }], lowpassHz: 2000 };
    for (const loopMs of [24, 50]) {
      const o = runLoop({ loopGain: 1.5, loopMs, noise: sources[0][1], shape, seed: 9 }, 3);
      report.push(describeLoop(`(a) G 1.5, ${loopMs} ms, reflections`, o));
      expectMutedHowl(o, ceilingOf(p));
      expect(o.eventS!).toBeLessThan(1);
    }
  });
});

describe('howl simulator: (b) any loop gain > 1, loops up to 500 ms, muted within 400 ms of sustained limiting', () => {
  const p = engineParams();
  const noise: NoiseSpec = { broadband: { rms: noiseRmsFor(p, -42), lo: 20, hi: 2000 } };
  for (const loopGain of [1.1, 1.2, 1.5, 2]) {
    for (const loopMs of [12.5, 50, 200, 500]) {
      it(`loop gain ${loopGain}, ${loopMs} ms`, () => {
        // Growth is 20·log10(G) dB per loop: 0.83 dB per 500 ms at G = 1.1 needs ~25 s to reach the ceiling.
        const growthS = 45 / ((20 * Math.log10(loopGain)) / (loopMs / 1000));
        const o = runLoop({ loopGain, loopMs, noise, seed: 11 }, growthS + 5);
        report.push(describeLoop(`(b) G ${loopGain}, ${loopMs} ms`, o));
        expectMutedHowl(o, ceilingOf(p));
        if (o.lim05S !== null) expect(o.eventS! - o.lim05S).toBeLessThanOrEqual(0.4);
        if (o.lim3S !== null) expect(o.eventS! - o.lim3S).toBeLessThanOrEqual(0.4);
      }, 30000);
    }
  }
});

describe('howl simulator: (b′) loops longer than 600 ms, muted within two trips of sustained limiting (v-dsp H3)', () => {
  const p = engineParams();
  const noise: NoiseSpec = { broadband: { rms: noiseRmsFor(p, -42), lo: 20, hi: 2000 } };
  for (const loopGain of [1.3, 1.5]) {
    for (const loopMs of [650, 1000]) {
      it(`loop gain ${loopGain}, ${loopMs} ms`, () => {
        const growthS = 45 / ((20 * Math.log10(loopGain)) / (loopMs / 1000));
        const o = runLoop({ loopGain, loopMs, noise, seed: 11 }, growthS + 5);
        report.push(describeLoop(`(b′) G ${loopGain}, ${loopMs} ms`, o));
        expectMutedHowl(o, ceilingOf(p));
        expect(o.reason).toBe('limiter-saturated');
        if (o.lim05S !== null) expect(o.eventS! - o.lim05S).toBeLessThanOrEqual((2 * loopMs) / 1000);
      }, 30000);
    }
  }
});

describe('howl simulator: (d) a wide chain and a howl at 3–3.8 kHz, muted within 400 ms of sustained limiting', () => {
  for (const [chain, lpHz] of [
    ['LP 4 kHz', 4000],
    ['LP off', null],
  ] as const) {
    const p = wideParams(lpHz);
    // Anti-noise of a low-frequency source 25 dB below the ceiling, which also seeds the howl band ~50 dB
    // below it. Below ~2.4 kHz it dilutes the howl, which the ~6 kHz analysis sees only through its
    // anti-alias filter (−13 dB at 3.4 kHz): analysed there, 11 of these 16 howls stayed pinned at the
    // ceiling and the other 5 were muted 0.55–4.9 s after sustained limiting.
    const noise: NoiseSpec = { broadband: { rms: noiseRmsFor(p, -25, 20, 1000), lo: 20, hi: 1000 } };
    for (const loopGain of [1.2, 1.5]) {
      for (const loopMs of [12.5, 50, 100, 200]) {
        it(`${chain}, loop gain ${loopGain}, ${loopMs} ms`, () => {
          const growthS = 50 / ((20 * Math.log10(loopGain)) / (loopMs / 1000));
          const o = runLoop({ loopGain, loopMs, noise, shape: BAND_PATH, params: p, seed: 13 }, growthS + 5);
          report.push(describeLoop(`(d) ${chain}, G ${loopGain}, ${loopMs} ms`, o));
          expectMutedHowl(o, ceilingOf(p), false);
          expect(o.ambientDb).toBeGreaterThan(-28);
          expect(o.ambientDb).toBeLessThan(-22);
          expect(o.freqHz!).toBeGreaterThan(2400);
          if (o.lim05S !== null) expect(o.eventS! - o.lim05S).toBeLessThanOrEqual(0.4);
          if (o.lim3S !== null) expect(o.eventS! - o.lim3S).toBeLessThanOrEqual(0.4);
          if (loopGain === 1.5 && loopMs <= 100) expect(o.eventS!).toBeLessThanOrEqual(1.5);
        }, 30000);
      }
    }
  }
});

/** Plant time (s) at which the 10 s of expectQuiet() begin; source envelopes are written against it. */
let quietStart = Infinity;

interface QuietSpec {
  noise: NoiseSpec;
  /** Stable loop gain (0 = no speaker → reference path). */
  loopGain?: number;
  /** Shape of that path (default: a single arrival). */
  shape?: Omit<PathSpec, 'delayS' | 'gain'>;
  /** Bessel LP of the chain (null: off; default: the default 500 Hz). */
  lpHz?: number | null;
  gainDb?: number;
  /** Arm and record from the first sample (the soft-start included). */
  fromArm?: boolean;
  /** Called before each block with the time since the start of the 10 s. */
  during?: (rig: SimRig, t: number) => void;
  seed?: number;
}

/** 10 s of a legitimate signal: no howl event, no auto-mute, never above the ceiling. */
function expectQuiet(name: string, s: QuietSpec): void {
  const p = s.lpHz === undefined ? engineParams(s.gainDb ?? -10) : wideParams(s.lpHz, s.gainDb ?? -10);
  const cfg = loopPlant(p, { loopGain: s.loopGain ?? 0, loopMs: 30, noise: s.noise, shape: s.shape, seed: s.seed ?? 5 });
  let maxGr = 0;
  const rig = new SimRig(cfg, {
    engine: { meterHz: fs / BLOCK },
    onMessage: (m) => {
      if (m.type === 'meters') maxGr = Math.max(maxGr, m.limiterGrDb[0]);
    },
  });
  quietStart = Infinity;
  if (s.fromArm === true) {
    rig.send({ type: 'params', params: p });
    rig.send({ type: 'arm', armed: true });
  } else {
    rig.start(p, 0.3);
  }
  quietStart = rig.plant.time / fs;
  rig.maxAbsOut = 0;
  maxGr = 0;
  let sumSq = 0;
  const n = Math.round((10 * fs) / BLOCK);
  for (let b = 0; b < n; b++) {
    s.during?.(rig, (b * BLOCK) / fs);
    const out = rig.run(BLOCK / fs, ['outL']).outL!;
    for (const v of out) sumSq += v * v;
  }
  const rms = 10 * Math.log10(sumSq / (n * BLOCK)) - p.ceilingDb;
  const levels = `output RMS ${rms.toFixed(1)} dB re ceiling, max GR ${maxGr.toFixed(1)} dB`;
  report.push(`no false trigger — ${name}: ${rig.autoMuted ? 'TRIGGERED' : 'no trigger'} in 10 s (${levels})`);
  expect(rig.events.filter((m) => m.type === 'howl' || m.type === 'autoMute')).toEqual([]);
  expect(rig.maxAbsOut).toBeLessThanOrEqual(ceilingOf(p));
}

describe('howl simulator: no false trigger in 10 s', () => {
  const p = engineParams();
  const bb = (relDb: number, lo = 20, hi = 2000) => ({ rms: noiseRmsFor(p, relDb, lo, hi), lo, hi });

  it('stationary broadband noise, 20 dB and 8 dB below the ceiling (the latter keeps the limiter busy)', () => {
    expectQuiet('stationary broadband noise −20 dB', { noise: { broadband: bb(-20) } });
    expectQuiet('stationary broadband noise −8 dB', { noise: { broadband: bb(-8) } });
  });

  it('broadband source stepping +20 dB at constant commanded gain, up to 6 dB below the ceiling', () => {
    expectQuiet('broadband step +20 dB to −6 dB', {
      noise: { broadband: bb(-6), broadbandGain: (t) => (t < quietStart + 3 ? 0.1 : 1) },
    });
  });

  it('a 100 Hz machine tone with harmonics switching on with a 50 ms ramp', () => {
    // Seeds 8, 11 and 23 trigger the band-level growth rule without its steady-growth test.
    const amps = [toneAmpFor(p, 100, -8), toneAmpFor(p, 200, -12), toneAmpFor(p, 300, -16), toneAmpFor(p, 400, -20)];
    for (const seed of [5, 8, 11, 23]) {
      expectQuiet(`100 Hz machine tone on in 50 ms (seed ${seed})`, {
        seed,
        noise: {
          broadband: bb(-40),
          tone: { f0: 100, amplitudes: amps },
          toneGain: (t) => Math.min(1, Math.max(0, (t - quietStart - 3) / 0.05)),
        },
      });
    }
  });

  it('a 100 Hz machine hum switching on over 150–200 ms over broadband noise ~20 dB below it (v-dsp H4)', () => {
    // The band level is averaged over ~110 ms: an onset this short must not pass for a 300 ms growth run.
    const amps = [toneAmpFor(p, 100, -20), toneAmpFor(p, 200, -24), toneAmpFor(p, 300, -28), toneAmpFor(p, 400, -32)];
    const onsets: [number, 'lin' | 'exp', number][] = [
      [0.15, 'lin', 3],
      [0.2, 'lin', 5],
      [0.2, 'exp', 7],
    ];
    for (const [rampS, shape, seed] of onsets) {
      expectQuiet(`100 Hz hum on in ${Math.round(rampS * 1000)} ms (${shape}) over noise −40 dB`, {
        seed,
        noise: {
          broadband: bb(-40),
          tone: { f0: 100, amplitudes: amps },
          toneGain: (t) => {
            const u = Math.min(1, Math.max(0, (t - quietStart - 3) / rampS));
            return shape === 'lin' ? u : u <= 0 ? 0 : dbToLin(-30 * (1 - u));
          },
        },
      });
    }
  });

  it('a steady machine hum whose peaks exceed the ceiling by 0.3 dB (the limiter takes off < 0.5 dB) (v-dsp H1)', () => {
    const amps = [1, 0.7, 0.5, 0.4];
    const wave = (t: number): number => amps.reduce((a, v, h) => a + v * Math.cos(2 * Math.PI * (h + 1) * 100 * t), 0);
    // Peak of the conditioned waveform per unit amplitude at the reference mic.
    const chain = new ConditioningChain(fs, p);
    let peak = 0;
    for (let i = 0; i < fs; i++) {
      const y = chain.process(wave(i / fs));
      if (i >= fs / 2) peak = Math.max(peak, Math.abs(y));
    }
    const k = (ceilingOf(p) * dbToLin(0.3)) / (commanded(p) * peak);
    expectQuiet('steady hum 0.3 dB into the limiter', { noise: { broadband: bb(-50), custom: (t) => k * wave(t) } });
  });

  it('a tone with ±6 dB slow beating', () => {
    for (const beatHz of [0.2, 0.4]) {
      const a = toneAmpFor(p, 150, -6) / 1.6;
      expectQuiet(`tone beating ±6 dB at ${beatHz} Hz`, {
        noise: {
          broadband: bb(-40),
          custom: (t) => a * (Math.sin(2 * Math.PI * 150 * t) + 0.6 * Math.sin(2 * Math.PI * (150 + beatHz) * t)),
        },
      });
    }
  });

  it('the 1.5 s soft-start with a stable loop (loop gain 0.5)', () => {
    expectQuiet('soft-start, loop gain 0.5', {
      fromArm: true,
      loopGain: 0.5,
      noise: { broadband: bb(-20), tone: { f0: 100, amplitudes: [toneAmpFor(p, 100, -10)] } },
    });
  });

  it('a +20 dB commanded gain change (rate-limited at 6 dB/s), loop gain 0.05 → 0.5', () => {
    let done = false;
    expectQuiet('+20 dB gain change', {
      gainDb: -30,
      loopGain: 0.05,
      // Levels at −10 dB of channel gain, i.e. after the change (20 dB lower before it).
      noise: { broadband: bb(-20), tone: { f0: 100, amplitudes: [toneAmpFor(p, 100, -10)] } },
      during: (rig, t) => {
        if (done || t < 2) return;
        done = true;
        const q = rig.currentParams;
        q.channels[0].gainDb = -10;
        rig.send({ type: 'params', params: q });
      },
    });
  });

  it('an impulsive source: identical bursts every 0.5 s whose peaks touch the ceiling', () => {
    // A 30 ms noise burst with a 6 ms decay, the same waveform every time (periodic at 500 ms),
    // scaled so that its peak at the output is 1 dB over the ceiling.
    const rnd = mulberry32(77);
    const decay = (i: number): number => Math.exp(-i / (0.006 * fs));
    const burst = Float64Array.from({ length: Math.round(0.03 * fs) }, (_, i) => (2 * rnd() - 1) * decay(i));
    const chain = new ConditioningChain(fs, p);
    let peak = 0;
    for (let i = 0; i < 0.1 * fs; i++) peak = Math.max(peak, Math.abs(chain.process(i < burst.length ? burst[i] : 0)));
    const amp = (ceilingOf(p) * dbToLin(1)) / (commanded(p) * peak);
    expectQuiet('impulsive bursts every 0.5 s', {
      noise: {
        broadband: bb(-40),
        custom: (t) => {
          const i = Math.round((t % 0.5) * fs);
          return i < burst.length ? amp * burst[i] : 0;
        },
      },
    });
  });

  it('speech-like modulated noise', () => {
    // Syllables at 4 Hz (sin² envelope) with random levels over 12 dB, 20 % silent, a pause every 2.5 s.
    const rnd = mulberry32(5);
    const syl = Array.from({ length: 64 }, () => (rnd() < 0.2 ? 0 : dbToLin(-12 * rnd())));
    expectQuiet('speech-like modulated noise', {
      noise: {
        broadband: bb(-15, 100, 2000),
        broadbandGain: (t) => {
          const k = Math.floor(t * 4);
          const env = Math.sin(Math.PI * (t * 4 - k)) ** 2;
          return t % 2.5 > 2.1 ? 0 : syl[k % syl.length] * env;
        },
      },
    });
  });

  it('normal feed-forward operation with no loop', () => {
    expectQuiet('feed-forward, no loop', {
      noise: { broadband: bb(-15), tone: { f0: 120, amplitudes: [toneAmpFor(p, 120, -10), toneAmpFor(p, 240, -16)] } },
    });
  });

  it('broadband noise fading in (−35 → −10 dB in 1 s) over a steady 120 Hz tone', () => {
    // The tone keeps the periodicity high while the noise grows: what grows is not periodic.
    expectQuiet('noise fade-in over a steady tone', {
      noise: {
        broadband: bb(-10),
        broadbandGain: (t) => Math.max(dbToLin(-25), Math.min(1, t - quietStart - 2)),
        tone: { f0: 120, amplitudes: [toneAmpFor(p, 120, -25)] },
      },
    });
  });

  it('a broadband source rising smoothly by 20 dB at 20 dB/s (a passing vehicle)', () => {
    expectQuiet('broadband rise 20 dB at 20 dB/s to −10 dB', {
      noise: {
        broadband: bb(-10),
        broadbandGain: (t) => {
          const u = t - quietStart - 2;
          const db = u < 0 ? -20 : u < 1 ? -20 + 20 * u : u < 3 ? 0 : Math.max(-20, -20 * (u - 3));
          return dbToLin(db);
        },
      },
    });
  });
});

describe('howl simulator: no false trigger in 10 s with a wide chain (periodicity at ~12 kHz)', () => {
  const p = wideParams(4000);
  const bb = (relDb: number, lo = 20, hi = 8000) => ({ rms: noiseRmsFor(p, relDb, lo, hi), lo, hi });

  it('broadband noise up to 8 kHz, 20 dB and 8 dB below the ceiling (LP 4 kHz), and 8 dB below with the LP off', () => {
    expectQuiet('LP 4 kHz, broadband noise −20 dB', { lpHz: 4000, noise: { broadband: bb(-20) } });
    expectQuiet('LP 4 kHz, broadband noise −8 dB', { lpHz: 4000, noise: { broadband: bb(-8) } });
    const off = wideParams(null);
    expectQuiet('LP off, broadband noise −8 dB', { lpHz: null, noise: { broadband: { rms: noiseRmsFor(off, -8, 20, 8000), lo: 20, hi: 8000 } } });
  });

  it('a 3 kHz tone switching on with a 50 ms ramp, 8 dB below the ceiling (LP 4 kHz)', () => {
    for (const seed of [5, 8, 11, 23]) {
      expectQuiet(`LP 4 kHz, 3 kHz tone on in 50 ms (seed ${seed})`, {
        lpHz: 4000,
        seed,
        noise: {
          broadband: bb(-40),
          tone: { f0: 3000, amplitudes: [toneAmpFor(p, 3000, -8)] },
          toneGain: (t) => Math.min(1, Math.max(0, (t - quietStart - 3) / 0.05)),
        },
      });
    }
  });

  it('speech-like modulated noise, 100–4000 Hz (LP 4 kHz)', () => {
    const rnd = mulberry32(5);
    const syl = Array.from({ length: 64 }, () => (rnd() < 0.2 ? 0 : dbToLin(-12 * rnd())));
    expectQuiet('LP 4 kHz, speech-like modulated noise', {
      lpHz: 4000,
      noise: {
        broadband: bb(-15, 100, 4000),
        broadbandGain: (t) => {
          const k = Math.floor(t * 4);
          const env = Math.sin(Math.PI * (t * 4 - k)) ** 2;
          return t % 2.5 > 2.1 ? 0 : syl[k % syl.length] * env;
        },
      },
    });
  });

  it('a stable loop (gain 0.7) on the 2.7–4.3 kHz band path, broadband noise 15 and 8 dB below the ceiling (LP 4 kHz)', () => {
    for (const relDb of [-15, -8]) {
      expectQuiet(`LP 4 kHz, stable band-path loop 0.7, noise ${relDb} dB`, { lpHz: 4000, loopGain: 0.7, shape: BAND_PATH, noise: { broadband: bb(relDb) } });
    }
  });

  it('broadband noise fading in over a steady 120 Hz tone, and rising smoothly at 20 dB/s (LP 4 kHz)', () => {
    expectQuiet('LP 4 kHz, noise fade-in over a steady tone', {
      lpHz: 4000,
      noise: {
        broadband: bb(-10),
        broadbandGain: (t) => Math.max(dbToLin(-25), Math.min(1, t - quietStart - 2)),
        tone: { f0: 120, amplitudes: [toneAmpFor(p, 120, -25)] },
      },
    });
    expectQuiet('LP 4 kHz, broadband rise 20 dB at 20 dB/s to −10 dB', {
      lpHz: 4000,
      noise: {
        broadband: bb(-10),
        broadbandGain: (t) => {
          const u = t - quietStart - 2;
          return dbToLin(u < 0 ? -20 : u < 1 ? -20 + 20 * u : u < 3 ? 0 : Math.max(-20, -20 * (u - 3)));
        },
      },
    });
  });

  it('noise fading in over a steady tone while the chain switches between narrow and wide (LP 1 ↔ 4 kHz)', () => {
    // Every switch restarts the periodicity analysis (and the reconfigure envelope dips the output).
    let next = 1;
    let wide = true;
    expectQuiet('noise fade-in over a tone, LP 1 ↔ 4 kHz every 0.7 s', {
      lpHz: 4000,
      noise: {
        broadband: bb(-10),
        broadbandGain: (t) => Math.max(dbToLin(-25), Math.min(1, (t - quietStart - 2) / 3)),
        tone: { f0: 120, amplitudes: [toneAmpFor(p, 120, -25)] },
      },
      during: (rig, t) => {
        if (t < next) return;
        next += 0.7;
        wide = !wide;
        const q = rig.currentParams;
        q.filters.lpFreq = wide ? 4000 : 1000;
        rig.send({ type: 'params', params: q });
      },
    });
  });
});
