import { describe, expect, it } from 'vitest';
import { GainRamp, RISE_FLOOR_DB } from '../../src/dsp/gain-ramp.ts';
import { BIG_JUMP_DB, GAIN_SMOOTH_MS, MASTER_MIN_DB, MAX_RISE_DB_PER_S } from '../../src/dsp/params.ts';

const fs = 48000;
const a = Math.exp(-1000 / (GAIN_SMOOTH_MS * fs));

/** Max rise (dB) of `db` over any window of `win` samples. */
function maxRiseOverWindow(db: Float64Array, win: number): number {
  let worst = 0;
  for (let i = win; i < db.length; i++) worst = Math.max(worst, db[i] - db[i - win]);
  return worst;
}

describe('GainRamp', () => {
  it('starts at the initial gain', () => {
    const g = new GainRamp(fs, -6);
    expect(g.next()).toBeCloseTo(Math.pow(10, -6 / 20), 15);
    expect(g.currentDb).toBe(-6);
    expect(g.targetDb).toBe(-6);
  });

  it('decrease follows a 20 ms one-pole in dB', () => {
    const g = new GainRamp(fs, 0);
    g.setTargetDb(-30);
    for (let n = 1; n <= 5000; n++) {
      const lin = g.next();
      const expectedDb = -30 + 30 * Math.pow(a, n);
      expect(Math.abs(g.currentDb - expectedDb)).toBeLessThan(1e-9);
      expect(Math.abs(lin - Math.pow(10, g.currentDb / 20))).toBeLessThan(1e-12);
    }
  });

  it('a +3 dB step is a plain one-pole', () => {
    const g = new GainRamp(fs, -20);
    g.setTargetDb(-17);
    for (let n = 1; n <= 5000; n++) {
      g.next();
      const expectedDb = -17 - 3 * Math.pow(a, n);
      expect(Math.abs(g.currentDb - expectedDb)).toBeLessThan(1e-9);
    }
  });

  it('-40 → 0 dB rises at ≤ 6 dB/s while the gap exceeds 6 dB', () => {
    const g = new GainRamp(fs, -40);
    g.setTargetDb(0);
    const n = 8 * fs;
    const db = new Float64Array(n + 1);
    db[0] = g.currentDb;
    let crossedAt = -1;
    for (let i = 1; i <= n; i++) {
      g.next();
      db[i] = g.currentDb;
      const step = db[i] - db[i - 1];
      expect(step).toBeGreaterThanOrEqual(0);
      if (0 - db[i - 1] > BIG_JUMP_DB) expect(step).toBeLessThanOrEqual(MAX_RISE_DB_PER_S / fs + 1e-12);
      else if (crossedAt < 0) crossedAt = i;
    }
    // 34 dB at 6 dB/s before the final one-pole takes over
    expect(crossedAt / fs).toBeGreaterThan((40 - BIG_JUMP_DB) / MAX_RISE_DB_PER_S - 0.01);
    expect(crossedAt / fs).toBeLessThan((40 - BIG_JUMP_DB) / MAX_RISE_DB_PER_S + 0.01);
    expect(db[n]).toBe(0);
    // over any 1 s window the rise never exceeds BIG_JUMP_DB + MAX_RISE_DB_PER_S·1 s
    expect(maxRiseOverWindow(db, fs)).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S + 1e-6);
  });

  it('repeated +5 dB increments cannot bypass the rate limit', () => {
    const g = new GainRamp(fs, -60);
    const n = 14 * fs;
    const db = new Float64Array(n + 1);
    db[0] = g.currentDb;
    let target = -60;
    let reached = -1;
    for (let i = 1; i <= n; i++) {
      // the user clicks +5 dB every 100 ms until 0 dB
      if (i % (fs / 10) === 1 && target < 0) {
        target = Math.min(0, target + 5);
        g.setTargetDb(target);
      }
      g.next();
      db[i] = g.currentDb;
      if (reached < 0 && db[i] >= -1) reached = i;
    }
    for (const winS of [0.1, 0.25, 0.5, 1, 2, 5]) {
      const win = Math.round(winS * fs);
      expect(maxRiseOverWindow(db, win)).toBeLessThanOrEqual(BIG_JUMP_DB + MAX_RISE_DB_PER_S * winS + 1e-6);
    }
    // 59 dB of rise needs at least (59 − 6)/6 s
    expect(reached / fs).toBeGreaterThanOrEqual((59 - BIG_JUMP_DB) / MAX_RISE_DB_PER_S);
    expect(db[n]).toBe(0);
  });

  it('≤ -120 dB gives exactly 0', () => {
    const g = new GainRamp(fs, MASTER_MIN_DB);
    expect(g.next()).toBe(0);
    const h = new GainRamp(fs, -1000);
    expect(h.next()).toBe(0);
    expect(h.currentDb).toBe(MASTER_MIN_DB);
    for (const t of [MASTER_MIN_DB, -130, Number.NEGATIVE_INFINITY]) {
      const r = new GainRamp(fs, -30);
      r.setTargetDb(t);
      let lin = 1;
      let zeroAt = -1;
      for (let i = 0; i < fs; i++) {
        lin = r.next();
        if (zeroAt < 0 && lin === 0) zeroAt = i;
      }
      expect(lin).toBe(0);
      expect(r.currentDb).toBe(MASTER_MIN_DB);
      // silence is reached in finite time (≈ 1.7 τ), not asymptotically
      expect(zeroAt).toBeGreaterThan(0);
      expect(zeroAt / fs).toBeLessThan(0.1);
    }
  });

  it('rising from silence jumps to RISE_FLOOR_DB, then obeys the same rate limit', () => {
    const g = new GainRamp(fs, MASTER_MIN_DB);
    g.setTargetDb(-20);
    let prev = g.currentDb;
    let floorAt = -1;
    for (let i = 1; i <= 2 * fs; i++) {
      g.next();
      const step = g.currentDb - prev;
      expect(step).toBeGreaterThanOrEqual(0);
      if (prev >= RISE_FLOOR_DB) expect(step).toBeLessThanOrEqual(MAX_RISE_DB_PER_S / fs + 1e-12);
      else expect(g.currentDb).toBeLessThanOrEqual(RISE_FLOOR_DB);
      if (floorAt < 0 && g.currentDb >= RISE_FLOOR_DB) floorAt = i;
      prev = g.currentDb;
    }
    // the inaudible part is a one-pole (≈ 1.4 τ), not 10 s at 6 dB/s
    expect(floorAt / fs).toBeGreaterThan(0.02);
    expect(floorAt / fs).toBeLessThan(0.05);
    // from the floor on: linear at 6 dB/s
    expect(g.currentDb).toBeCloseTo(RISE_FLOOR_DB + MAX_RISE_DB_PER_S * (2 - floorAt / fs), 6);
  });

  it('rises entirely below RISE_FLOOR_DB are a plain one-pole', () => {
    const g = new GainRamp(fs, -110);
    g.setTargetDb(-70);
    for (let n = 1; n <= 2000; n++) {
      g.next();
      expect(Math.abs(g.currentDb - (-70 - 40 * Math.pow(a, n)))).toBeLessThan(1e-9);
    }
  });

  it('un-muting a master at -∞ to -30 dB takes ≈ 5 s, not 14 s', () => {
    const g = new GainRamp(fs, Number.NEGATIVE_INFINITY);
    g.setTargetDb(-30);
    let reached = -1;
    for (let i = 1; i <= 8 * fs && reached < 0; i++) {
      g.next();
      if (g.currentDb >= -30.01) reached = i;
    }
    expect(reached / fs).toBeGreaterThan((30 - BIG_JUMP_DB) / MAX_RISE_DB_PER_S);
    expect(reached / fs).toBeLessThan((30 - BIG_JUMP_DB) / MAX_RISE_DB_PER_S + 1);
  });

  it('holdRise defers increases (the gain stays where it was last heard), decreases go on', () => {
    const g = new GainRamp(fs, -40);
    g.setTargetDb(0);
    for (let i = 0; i < 3 * fs; i++) g.next(true);
    expect(g.currentDb).toBe(-40);
    // released: the usual 6 dB/s from where it was, not a 40 dB jump
    for (let i = 0; i < fs; i++) g.next();
    expect(g.currentDb).toBeCloseTo(-34, 6);
    g.setTargetDb(-60);
    for (let i = 0; i < fs; i++) g.next(true);
    expect(g.currentDb).toBeCloseTo(-60, 6);
    // the floor rule waits too
    const s = new GainRamp(fs, MASTER_MIN_DB);
    s.setTargetDb(-20);
    for (let i = 0; i < fs; i++) expect(s.next(true)).toBe(0);
  });

  it('setRiseFloorDb moves the free range: below the floor a plain one-pole, above it 6 dB/s', () => {
    // channel −20 dB + master: free while the master part is below −60 dB, i.e. the total below −80 dB
    const g = new GainRamp(fs, -130);
    g.setRiseFloorDb(-20 + RISE_FLOOR_DB);
    g.setTargetDb(-20);
    let floorAt = -1;
    let prev = g.currentDb;
    for (let i = 1; i <= 3 * fs; i++) {
      g.next();
      if (prev >= -80) expect(g.currentDb - prev).toBeLessThanOrEqual(MAX_RISE_DB_PER_S / fs + 1e-12);
      else expect(g.currentDb).toBeLessThanOrEqual(-80);
      if (floorAt < 0 && g.currentDb >= -80) floorAt = i;
      prev = g.currentDb;
    }
    expect(floorAt / fs).toBeLessThan(0.05);
    expect(g.currentDb).toBeCloseTo(-80 + MAX_RISE_DB_PER_S * (3 - floorAt / fs), 6);
    // a gain above the floor is rate-limited even below RISE_FLOOR_DB
    const h = new GainRamp(fs, -75);
    h.setRiseFloorDb(-90);
    h.setTargetDb(-20);
    for (let i = 0; i < fs; i++) h.next();
    expect(h.currentDb).toBeCloseTo(-69, 6);
    h.setRiseFloorDb(Number.NaN);
    h.setTargetDb(-75);
    for (let i = 0; i < fs; i++) h.next();
    h.setTargetDb(-20);
    for (let i = 0; i < fs; i++) h.next();
    expect(h.currentDb).toBeCloseTo(-69, 6);
  });

  it('ignores NaN, caps at +12 dB, jumpTo is immediate', () => {
    const g = new GainRamp(fs, -10);
    g.setTargetDb(Number.NaN);
    expect(g.targetDb).toBe(-10);
    g.setTargetDb(40);
    expect(g.targetDb).toBe(12);
    g.jumpTo(-3);
    expect(g.next()).toBeCloseTo(Math.pow(10, -3 / 20), 15);
    expect(g.currentDb).toBe(-3);
    const bad = new GainRamp(fs, Number.NaN);
    expect(bad.next()).toBe(0);
  });
});
