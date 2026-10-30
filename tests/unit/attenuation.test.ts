import { describe, expect, it } from 'vitest';
import { AttenuationEstimator } from '../../src/app/attenuation.ts';
import type { AncPhase } from '../../src/app/attenuation.ts';

const FS = 48000;
const BLOCK = 2048;
const ms = (db: number): number => Math.pow(10, db / 10);

/** Pushes `seconds` of blocks with constant levels; returns true if the guard tripped. */
function feed(
  e: AttenuationEstimator,
  phase: AncPhase,
  seconds: number,
  levels: { err: number; ref: number; cond?: number },
): boolean {
  let tripped = false;
  const blocks = Math.round((seconds * FS) / BLOCK);
  for (let i = 0; i < blocks; i++) {
    if (e.push(phase, BLOCK, ms(levels.err), ms(levels.ref), ms(levels.cond ?? levels.ref))) tripped = true;
  }
  return tripped;
}

describe('AttenuationEstimator', () => {
  it('A = (err − ref)_off − (err − ref)_on, with the "on" reference taken from the bypass windows', () => {
    const e = new AttenuationEstimator(FS);
    feed(e, 'off', 2, { err: -40, ref: -34 });
    expect(e.attenuationDb).toBeNull();
    // ANC on: the error drops by 18 dB; the raw reference (polluted by the anti-noise) rises,
    // but it is not used.
    feed(e, 'on', 3, { err: -58, ref: -20 });
    expect(e.attenuationDb).toBeCloseTo(18, 6);
    // The readout survives a bypass window.
    feed(e, 'off', 1, { err: -40, ref: -34 });
    expect(e.attenuationDb).toBeCloseTo(18, 6);
  });

  it("'cond' source: the conditioned x̂ level is used on both sides", () => {
    const e = new AttenuationEstimator(FS);
    e.setRefSource('cond');
    feed(e, 'off', 2, { err: -40, ref: -34, cond: -36 });
    // The noise source got 6 dB louder while ANC is on: x̂ follows it, so A stays 15 dB.
    feed(e, 'on', 3, { err: -49, ref: -10, cond: -30 });
    expect(e.attenuationDb).toBeCloseTo(15, 6);
  });

  it('no bypass reference → no attenuation and the guard cannot trip', () => {
    const e = new AttenuationEstimator(FS);
    expect(feed(e, 'on', 5, { err: -10, ref: -40 })).toBe(false);
    expect(e.attenuationDb).toBeNull();
  });

  it('transition audio is ignored', () => {
    const e = new AttenuationEstimator(FS);
    feed(e, 'off', 1, { err: -40, ref: -34 });
    feed(e, 'transition', 2, { err: 0, ref: 0 });
    feed(e, 'on', 1, { err: -50, ref: -34 });
    expect(e.attenuationDb).toBeCloseTo(10, 6);
  });

  it('guard: error 3+ dB above the bypass reference for more than 2 s trips once', () => {
    const e = new AttenuationEstimator(FS);
    feed(e, 'off', 2, { err: -40, ref: -34 });
    // 4 dB worse for 1.75 s: not yet.
    expect(feed(e, 'on', 1.75, { err: -36, ref: -34 })).toBe(false);
    expect(e.excessDb).toBeCloseTo(4, 6);
    // Keeps going past 2 s: trips.
    expect(feed(e, 'on', 0.5, { err: -36, ref: -34 })).toBe(true);
    expect(e.attenuationDb).toBeLessThan(0);
  });

  it('guard: 2.5 dB worse never trips, and a bypass window restarts the count', () => {
    const e = new AttenuationEstimator(FS);
    feed(e, 'off', 2, { err: -40, ref: -34 });
    expect(feed(e, 'on', 10, { err: -37.5, ref: -34 })).toBe(false);
    expect(feed(e, 'on', 1.5, { err: -30, ref: -34 })).toBe(false);
    feed(e, 'off', 0.5, { err: -40, ref: -34 });
    expect(feed(e, 'on', 1.5, { err: -30, ref: -34 })).toBe(false);
    expect(feed(e, 'on', 1, { err: -30, ref: -34 })).toBe(true);
  });

  it('silent or non-finite blocks carry no information', () => {
    const e = new AttenuationEstimator(FS);
    feed(e, 'off', 2, { err: -200, ref: -200 });
    feed(e, 'on', 2, { err: -40, ref: -40 });
    expect(e.attenuationDb).toBeNull();
    expect(e.push('on', BLOCK, Number.NaN, 1, 1)).toBe(false);
    expect(e.push('on', 0, 1, 1, 1)).toBe(false);
  });

  it('a source that gets quieter while the ANC is on does not hide an ANC that adds noise', () => {
    const e = new AttenuationEstimator(FS);
    // Bypass: error −40, raw reference −30.
    feed(e, 'off', 5, { err: -40, ref: -30 });
    // ANC on, 2 dB of attenuation; the anti-noise it picks up adds 1 dB to the raw reference.
    expect(feed(e, 'on', 2, { err: -42, ref: -29 })).toBe(false);
    expect(e.attenuationDb).toBeCloseTo(2, 6);
    // Then the source gets 6 dB quieter and a mis-tuned ANC adds 5 dB at the zone (err −41): against the
    // stored −30 it would read +1 dB. The raw reference fell 6 dB during the run: the ambient did.
    expect(feed(e, 'on', 6, { err: -41, ref: -35 })).toBe(true);
    expect(e.attenuationDb).toBeCloseTo(-5, 6);
    expect(e.excessDb).toBeCloseTo(5, 6);
    // A louder raw reference (feedback, or a louder source) never raises the stored level.
    const f = new AttenuationEstimator(FS);
    feed(f, 'off', 2, { err: -40, ref: -34 });
    feed(f, 'on', 3, { err: -58, ref: -20 });
    expect(f.attenuationDb).toBeCloseTo(18, 6);
  });

  it('a tone partly cancelled at the raw reference by its own anti-noise is no ambient drop (calib-ui-4)', () => {
    const e = new AttenuationEstimator(FS);
    feed(e, 'off', 5, { err: -40, ref: -30 });
    // ANC on: the error 0.5 dB lower; from the first window the anti-noise reaching the raw reference
    // cancels 4 dB of the tone there (|1 − 0.37|²). Not "the ambient dropped by 4 dB": no guard trip.
    expect(feed(e, 'on', 5, { err: -40.5, ref: -34 })).toBe(false);
    expect(e.attenuationDb).toBeCloseTo(0.5, 6);
    expect(e.excessDb).toBeCloseTo(-0.5, 6);
  });

  it('raw reference windows scattering around their level do not bias the readout (calib-ui-4)', () => {
    const e = new AttenuationEstimator(FS);
    const win = (6 * BLOCK) / FS; // one 0.25 s window per call
    const d = (i: number): number => (i % 2 === 0 ? 1.5 : -1.5);
    for (let i = 0; i < 16; i++) feed(e, 'off', win, { err: -40 + d(i), ref: -30 + d(i) });
    // ANC on, 10 dB of attenuation; the raw reference keeps scattering ±1.5 dB window by window.
    for (let i = 0; i < 16; i++) feed(e, 'on', win, { err: -50, ref: -30 + d(i) });
    expect(e.attenuationDb).toBeCloseTo(10, 6);
    // A real drop still counts once its mean is beyond the scatter (> 3 standard errors): 4 dB here.
    for (let i = 0; i < 32; i++) feed(e, 'on', win, { err: -54, ref: -34 + d(i) });
    expect(e.attenuationDb).toBeCloseTo(10, 0);
  });

  it('referenceAgeS: age of the newest bypass window', () => {
    const e = new AttenuationEstimator(FS);
    expect(e.referenceAgeS).toBeNull();
    feed(e, 'off', 1, { err: -40, ref: -34 });
    feed(e, 'on', 10, { err: -50, ref: -34 });
    expect(e.referenceAgeS).toBeGreaterThan(9.9);
    expect(e.referenceAgeS).toBeLessThan(10.3);
  });

  it('an old bypass reference expires', () => {
    const e = new AttenuationEstimator(FS, { maxOffAgeS: 5 });
    feed(e, 'off', 1, { err: -40, ref: -34 });
    feed(e, 'transition', 10, { err: 0, ref: 0 });
    feed(e, 'on', 2, { err: -60, ref: -34 });
    expect(e.attenuationDb).toBeNull();
  });
});
