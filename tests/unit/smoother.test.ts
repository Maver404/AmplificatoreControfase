import { describe, expect, it } from 'vitest';
import { LinearEnvelope, OnePole } from '../../src/dsp/smoother.ts';

describe('OnePole', () => {
  it('follows an exponential with the given time constant', () => {
    const fs = 48000;
    const timeMs = 20;
    const p = new OnePole(fs, timeMs, 0);
    p.setTarget(1);
    const a = Math.exp(-1000 / (timeMs * fs));
    for (let n = 1; n <= 2000; n++) {
      const v = p.next();
      expect(Math.abs(v - (1 - Math.pow(a, n)))).toBeLessThan(1e-12);
    }
    // after one time constant: 1 − 1/e
    const q = new OnePole(fs, timeMs, 0);
    q.setTarget(1);
    let v = 0;
    for (let n = 0; n < (timeMs * fs) / 1000; n++) v = q.next();
    expect(v).toBeCloseTo(1 - Math.exp(-1), 6);
  });

  it('lands exactly on the target', () => {
    const p = new OnePole(48000, 5, -3);
    p.setTarget(7);
    let v = 0;
    for (let n = 0; n < 48000; n++) v = p.next();
    expect(v).toBe(7);
    expect(p.value).toBe(7);
  });

  it('set() jumps and non-finite targets are ignored', () => {
    const p = new OnePole(48000, 20, 0);
    p.set(2);
    expect(p.next()).toBe(2);
    p.setTarget(Number.NaN);
    p.setTarget(Number.POSITIVE_INFINITY);
    expect(p.next()).toBe(2);
    p.set(Number.NaN);
    expect(p.value).toBe(2);
  });

  it('zero time is immediate', () => {
    const p = new OnePole(48000, 0, 0);
    p.setTarget(1);
    expect(p.next()).toBe(1);
  });
});

describe('LinearEnvelope', () => {
  it('ramps linearly and lands exactly on the target', () => {
    const fs = 48000;
    const e = new LinearEnvelope(fs, 0);
    expect(e.done).toBe(true);
    e.rampTo(1, 10);
    expect(e.done).toBe(false);
    const n = 480;
    for (let i = 1; i <= n; i++) {
      const v = e.next();
      if (i < n) expect(Math.abs(v - i / n)).toBeLessThan(1e-12);
      else expect(v).toBe(1);
    }
    expect(e.done).toBe(true);
    expect(e.next()).toBe(1);
  });

  it('soft-start 1.5 s and mute 15 ms durations', () => {
    const fs = 44100;
    const e = new LinearEnvelope(fs, 0);
    e.rampTo(1, 1500);
    let steps = 0;
    while (!e.done) {
      e.next();
      steps++;
    }
    expect(steps).toBe(Math.round(1.5 * fs));
    e.rampTo(0, 15);
    steps = 0;
    while (!e.done) {
      e.next();
      steps++;
    }
    expect(steps).toBe(Math.round(0.015 * fs));
    expect(e.value).toBe(0);
  });

  it('can be retargeted mid-ramp without a jump', () => {
    const e = new LinearEnvelope(48000, 0);
    e.rampTo(1, 100);
    let prev = 0;
    for (let i = 0; i < 1000; i++) prev = e.next();
    e.rampTo(0, 10);
    const first = e.next();
    expect(Math.abs(first - prev)).toBeLessThan(prev / 400);
    for (let i = 0; i < 1000; i++) e.next();
    expect(e.value).toBe(0);
  });

  it('zero time and set() are immediate; NaN is ignored', () => {
    const e = new LinearEnvelope(48000, 0.5);
    e.rampTo(0, 0);
    expect(e.value).toBe(0);
    expect(e.done).toBe(true);
    e.set(0.25);
    expect(e.next()).toBe(0.25);
    e.rampTo(Number.NaN, 10);
    expect(e.next()).toBe(0.25);
  });
});
