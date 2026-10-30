import { describe, expect, it } from 'vitest';
import { LevelMeter } from '../../src/dsp/meters.ts';

describe('LevelMeter', () => {
  it('reads -120 dBFS on silence', () => {
    const m = new LevelMeter(48000);
    expect(m.rmsDb).toBe(-120);
    expect(m.peakDb).toBe(-120);
    for (let i = 0; i < 1000; i++) m.push(0);
    expect(m.rmsDb).toBe(-120);
    expect(m.peakDb).toBe(-120);
  });

  it('measures the RMS and peak of a sine', () => {
    const fs = 48000;
    const m = new LevelMeter(fs);
    const amp = 0.5;
    const w = (2 * Math.PI * 1000) / fs;
    for (let n = 0; n < 3 * fs; n++) m.push(amp * Math.sin(w * n));
    expect(Math.abs(m.rmsDb - 20 * Math.log10(amp / Math.SQRT2))).toBeLessThan(0.05);
    expect(Math.abs(m.peakDb - 20 * Math.log10(amp))).toBeLessThan(0.01);
  });

  it('holds the peak for 1.5 s, then falls at 20 dB/s', () => {
    const fs = 48000;
    const m = new LevelMeter(fs);
    m.push(1);
    for (let n = 0; n < 1.4 * fs; n++) m.push(0);
    expect(m.peakDb).toBe(0);
    for (let n = 0; n < 0.1 * fs + 1; n++) m.push(0);
    // hold expired: now falling
    for (let n = 0; n < 0.5 * fs; n++) m.push(0);
    expect(m.peakDb).toBeLessThan(-9.5);
    expect(m.peakDb).toBeGreaterThan(-10.5);
  });

  it('RMS time constant ≈ 300 ms', () => {
    const fs = 48000;
    const m = new LevelMeter(fs);
    for (let n = 0; n < 0.3 * fs; n++) m.push(1);
    // mean square after one time constant: 1 − 1/e
    expect(m.rmsDb).toBeCloseTo(10 * Math.log10(1 - Math.exp(-1)), 2);
  });

  it('ignores non-finite samples and resets', () => {
    const m = new LevelMeter(48000);
    m.push(0.1);
    m.push(Number.NaN);
    m.push(Number.POSITIVE_INFINITY);
    expect(Number.isFinite(m.rmsDb)).toBe(true);
    expect(Number.isFinite(m.peakDb)).toBe(true);
    expect(m.peakDb).toBeCloseTo(-20, 6);
    m.reset();
    expect(m.rmsDb).toBe(-120);
    expect(m.peakDb).toBe(-120);
  });
});
