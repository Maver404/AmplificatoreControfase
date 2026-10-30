import { describe, expect, it } from 'vitest';
import { DelayLine, SmoothDelayTap } from '../../src/dsp/fractional-delay.ts';
import { D_MIN_SAMPLES, DELAY_CROSSFADE_MS, DELAY_SLEW, delayCapacity } from '../../src/dsp/params.ts';

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Steady-state complex response for input sin(ωn), by least squares over [start, end). */
function fitSine(y: Float64Array, w: number, start: number, end: number) {
  let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
  for (let n = start; n < end; n++) {
    const s = Math.sin(w * n);
    const c = Math.cos(w * n);
    ss += s * s; sc += s * c; cc += c * c; ys += y[n] * s; yc += y[n] * c;
  }
  const det = ss * cc - sc * sc;
  return { re: (ys * cc - yc * sc) / det, im: (yc * ss - ys * sc) / det };
}

describe('DelayLine', () => {
  it('integer delays are exact: y[n] = x[n − D]', () => {
    const rnd = mulberry32(1);
    const line = new DelayLine(delayCapacity(48000));
    const x = new Float64Array(5000);
    for (let i = 0; i < x.length; i++) x[i] = rnd() * 2 - 1;
    const delays = [1, 2, 3, 5, 100, 1000, 4096];
    for (let n = 0; n < x.length; n++) {
      line.write(x[n]);
      for (const d of delays) expect(line.read(d)).toBe(n - d >= 0 ? x[n - d] : 0);
    }
  });

  it('D = 12.37 samples: amplitude and phase at 100 Hz and 1 kHz within Lagrange-3 accuracy', () => {
    const fs = 48000;
    const D = 12.37;
    for (const f of [100, 1000]) {
      const line = new DelayLine(1024);
      const w = (2 * Math.PI * f) / fs;
      const n = fs;
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        line.write(Math.sin(w * i));
        y[i] = line.read(D);
      }
      const h = fitSine(y, w, 1000, n);
      const magDb = 20 * Math.log10(Math.hypot(h.re, h.im));
      // phase delay in samples: −arg(H)/ω
      const phaseDelay = -Math.atan2(h.im, h.re) / w;
      expect(Math.abs(magDb)).toBeLessThan(0.01);
      expect(Math.abs(phaseDelay - D)).toBeLessThan(0.001);
    }
  });

  it('D = 12.37 samples: group delay error < 0.001 samples at 1 kHz', () => {
    const fs = 48000;
    const D = 12.37;
    // impulse response of the interpolator
    const line = new DelayLine(64);
    const ir = new Float64Array(32);
    for (let n = 0; n < ir.length; n++) {
      line.write(n === 0 ? 1 : 0);
      ir[n] = line.read(D);
    }
    const resp = (f: number) => {
      const w = (2 * Math.PI * f) / fs;
      let re = 0, im = 0;
      for (let n = 0; n < ir.length; n++) {
        re += ir[n] * Math.cos(w * n);
        im -= ir[n] * Math.sin(w * n);
      }
      return Math.atan2(im, re);
    };
    for (const f of [100, 1000]) {
      const df = 0.05;
      let dphi = resp(f + df) - resp(f - df);
      dphi = Math.atan2(Math.sin(dphi), Math.cos(dphi));
      const gd = (-dphi / (2 * Math.PI * 2 * df)) * fs;
      expect(Math.abs(gd - D)).toBeLessThan(0.001);
    }
    // exactly four non-zero taps around the delay
    const nz = Array.from(ir.keys()).filter((i) => ir[i] !== 0);
    expect(nz).toEqual([11, 12, 13, 14]);
  });

  it('the minimum delay is 1 sample', () => {
    const line = new DelayLine(64);
    for (let n = 0; n < 10; n++) line.write(n + 1);
    // newest sample is 10, x[n − 1] = 9
    for (const d of [D_MIN_SAMPLES, 0.999, 0.3, 0, -5, Number.NaN, Number.NEGATIVE_INFINITY]) {
      expect(line.read(d)).toBe(9);
    }
  });

  it('clamps delays beyond the capacity', () => {
    const line = new DelayLine(64);
    for (let n = 0; n < 200; n++) line.write(Math.sin(n));
    expect(line.read(1e9)).toBe(line.read(line.maxDelay));
    expect(Number.isFinite(line.read(Number.POSITIVE_INFINITY))).toBe(true);
  });

  it('stores non-finite input as 0 and resets', () => {
    const line = new DelayLine(16);
    line.write(Number.NaN);
    line.write(1);
    expect(line.read(1)).toBe(0);
    line.reset();
    line.write(0);
    expect(line.read(1)).toBe(0);
  });
});

describe('SmoothDelayTap', () => {
  const fs = 48000;

  it('slews small changes at ≤ 0.003 samples/sample', () => {
    const line = new DelayLine(delayCapacity(fs));
    const tap = new SmoothDelayTap(fs, 100);
    tap.setTarget(110); // 10 samples < 1 ms threshold: slew, no crossfade
    let prev = tap.current;
    let steps = 0;
    for (let n = 0; n < 5000; n++) {
      line.write(Math.sin(0.01 * n));
      const y = tap.next(line);
      expect(tap.crossfading).toBe(false);
      expect(y).toBe(line.read(tap.current));
      const d = tap.current - prev;
      expect(Math.abs(d)).toBeLessThanOrEqual(DELAY_SLEW + 1e-12);
      if (d !== 0) steps++;
      prev = tap.current;
    }
    expect(tap.current).toBe(110);
    expect(steps).toBe(Math.ceil(10 / DELAY_SLEW));
  });

  it('jumps with a click-free crossfade (100 Hz sine)', () => {
    const f = 100;
    const w = (2 * Math.PI * f) / fs;
    const naturalStep = 2 * Math.sin(w / 2); // max |Δx| of a unit sine
    const run = (smooth: boolean) => {
      const line = new DelayLine(delayCapacity(fs));
      const tap = new SmoothDelayTap(fs, 100);
      let prev = 0;
      let maxStep = 0;
      for (let n = 0; n < fs; n++) {
        // half a period: the worst case for a hard jump
        if (n === 10000) tap.setTarget(100 + fs / f / 2);
        line.write(Math.sin(w * n));
        const y = smooth ? tap.next(line) : line.read(n >= 10000 ? 100 + fs / f / 2 : 100);
        if (n > 1000) maxStep = Math.max(maxStep, Math.abs(y - prev));
        prev = y;
      }
      return { maxStep, tap };
    };
    const hard = run(false);
    const soft = run(true);
    expect(hard.maxStep).toBeGreaterThan(1);
    // crossfade adds at most π/N (N = fade length) to the natural slope
    const n = Math.round((DELAY_CROSSFADE_MS / 1000) * fs);
    expect(soft.maxStep).toBeLessThanOrEqual(naturalStep + Math.PI / n + 1e-9);
    expect(soft.maxStep).toBeLessThan(0.016);
    expect(soft.tap.current).toBe(100 + fs / f / 2);
  });

  it('crossfade lasts 30 ms and ends exactly on the new tap', () => {
    const line = new DelayLine(delayCapacity(fs));
    const tap = new SmoothDelayTap(fs, 10);
    const rnd = mulberry32(7);
    tap.setTarget(500);
    let fading = 0;
    for (let n = 0; n < 4000; n++) {
      line.write(rnd() - 0.5);
      const y = tap.next(line);
      if (tap.crossfading) fading++;
      else if (n > 10) expect(y).toBe(line.read(500));
    }
    expect(fading).toBe(Math.round((DELAY_CROSSFADE_MS / 1000) * fs) - 1);
  });

  it('forceJump crossfades even a small change', () => {
    const line = new DelayLine(delayCapacity(fs));
    const tap = new SmoothDelayTap(fs, 10);
    tap.setTarget(12, true);
    line.write(1);
    tap.next(line);
    expect(tap.crossfading).toBe(true);
    expect(tap.current).toBe(12);
  });

  it('a jump requested during a crossfade starts after it', () => {
    const f = 100;
    const w = (2 * Math.PI * f) / fs;
    const line = new DelayLine(delayCapacity(fs));
    const tap = new SmoothDelayTap(fs, 100);
    let prev = 0;
    let maxStep = 0;
    for (let n = 0; n < fs; n++) {
      if (n === 5000) tap.setTarget(400);
      if (n === 5500) tap.setTarget(2000);
      line.write(Math.sin(w * n));
      const y = tap.next(line);
      if (n > 1000) maxStep = Math.max(maxStep, Math.abs(y - prev));
      prev = y;
    }
    expect(tap.current).toBe(2000);
    expect(tap.target).toBe(2000);
    expect(maxStep).toBeLessThan(0.016);
  });

  it('reach covers the read position, the position fading out and the target', () => {
    const line = new DelayLine(delayCapacity(fs));
    const tap = new SmoothDelayTap(fs, 500);
    expect(tap.reach).toBe(500);
    // slewing towards a longer delay: the target counts at once
    tap.setTarget(510);
    expect(tap.reach).toBe(510);
    tap.snap();
    // crossfade to a shorter delay: the old position is read until the fade ends
    tap.setTarget(100);
    line.write(0);
    tap.next(line);
    expect(tap.crossfading).toBe(true);
    expect(tap.current).toBe(100);
    expect(tap.reach).toBe(510);
    for (let n = 0; n < fs / 10; n++) {
      line.write(0);
      tap.next(line);
    }
    expect(tap.reach).toBe(100);
  });

  it('clamps targets to the valid range and ignores NaN', () => {
    const tap = new SmoothDelayTap(fs, 50);
    tap.setTarget(Number.NaN);
    expect(tap.target).toBe(50);
    tap.setTarget(-3);
    expect(tap.target).toBe(D_MIN_SAMPLES);
    tap.setTarget(1e12);
    expect(tap.target).toBeLessThan(delayCapacity(fs) - 3);
  });

  it('output is identical regardless of how samples are grouped into blocks', () => {
    const total = 20000;
    const rnd = mulberry32(42);
    const x = Float64Array.from({ length: total }, () => rnd() * 2 - 1);
    // parameter changes at fixed sample positions, applied by the block driver
    const schedule = new Map<number, [number, boolean]>([
      [0, [37.25, false]],
      [3000, [39.5, false]],
      [7000, [900.125, false]],
      [7100, [12.5, true]],
      [15000, [12.75, false]],
    ]);
    const runBlocks = (sizes: () => number) => {
      const line = new DelayLine(delayCapacity(fs));
      const tap = new SmoothDelayTap(fs, 20);
      const out = new Float64Array(total);
      let n = 0;
      while (n < total) {
        const len = Math.min(sizes(), total - n);
        for (let i = 0; i < len; i++, n++) {
          const ev = schedule.get(n);
          if (ev) tap.setTarget(ev[0], ev[1]);
          line.write(x[n]);
          out[n] = tap.next(line);
        }
      }
      return out;
    };
    const ref = runBlocks(() => 128);
    for (const size of [1, 64, 256, 4096]) expect(runBlocks(() => size)).toEqual(ref);
    const r = mulberry32(3);
    expect(runBlocks(() => 1 + Math.floor(r() * 300))).toEqual(ref);
  });
});
