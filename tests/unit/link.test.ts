import { describe, expect, it } from 'vitest';
import { applyChannelChange } from '../../src/app/link.ts';
import type { ChannelPair } from '../../src/app/link.ts';
import type { ChannelParams } from '../../src/dsp/params.ts';

const ch = (p: Partial<ChannelParams>): ChannelParams => ({
  enabled: true,
  polarityInvert: true,
  gainDb: -20,
  delayMs: 0,
  phaseDeltaDeg: 0,
  ...p,
});

const pair = (l: Partial<ChannelParams>, r: Partial<ChannelParams>): ChannelPair => [ch(l), ch(r)];

describe('applyChannelChange', () => {
  it('unlinked: only the edited channel changes', () => {
    const before = pair({ gainDb: -20 }, { gainDb: -26 });
    const out = applyChannelChange(before, 0, ch({ gainDb: -14 }), 'gainDb', false);
    expect(out[0].gainDb).toBe(-14);
    expect(out[1].gainDb).toBe(-26);
    // Inputs are not mutated.
    expect(before[0].gainDb).toBe(-20);
  });

  it('linked: numeric fields move together and keep the L/R offset', () => {
    const before = pair({ gainDb: -20, delayMs: 2, phaseDeltaDeg: 10 }, { gainDb: -26, delayMs: 3.5, phaseDeltaDeg: -5 });
    let out = applyChannelChange(before, 0, ch({ ...before[0], gainDb: -17 }), 'gainDb', true);
    expect(out[0].gainDb).toBeCloseTo(-17, 12);
    expect(out[1].gainDb).toBeCloseTo(-23, 12);
    out = applyChannelChange(out, 1, ch({ ...out[1], delayMs: 4 }), 'delayMs', true);
    expect(out[1].delayMs).toBeCloseTo(4, 12);
    expect(out[0].delayMs).toBeCloseTo(2.5, 12);
    out = applyChannelChange(out, 0, ch({ ...out[0], phaseDeltaDeg: 0 }), 'phaseDeltaDeg', true);
    expect(out[1].phaseDeltaDeg - out[0].phaseDeltaDeg).toBeCloseTo(-15, 12);
  });

  it('linked: the shared change stops where one channel reaches its limit, so the offset survives', () => {
    const before = pair({ gainDb: -6 }, { gainDb: -3 });
    // L asks for -1 dB: R would go to +2 dB (above 0) → both stop at R = 0, L = -3.
    const out = applyChannelChange(before, 0, ch({ gainDb: -1 }), 'gainDb', true);
    expect(out[1].gainDb).toBe(0);
    expect(out[0].gainDb).toBe(-3);
    // With the +12 dB unlock the same request fits.
    const unlocked = applyChannelChange(before, 0, ch({ gainDb: -1 }), 'gainDb', true, true);
    expect(unlocked[0].gainDb).toBe(-1);
    expect(unlocked[1].gainDb).toBe(2);
    // Delay cannot go below 0 for either channel.
    const d = applyChannelChange(pair({ delayMs: 5 }, { delayMs: 1 }), 0, ch({ delayMs: 0 }), 'delayMs', true);
    expect(d[1].delayMs).toBe(0);
    expect(d[0].delayMs).toBe(4);
  });

  it('linked: toggles are copied to the other channel', () => {
    const before = pair({ polarityInvert: true }, { polarityInvert: true, enabled: false });
    const out = applyChannelChange(before, 0, ch({ polarityInvert: false }), 'polarityInvert', true);
    expect(out[0].polarityInvert).toBe(false);
    expect(out[1].polarityInvert).toBe(false);
    const en = applyChannelChange(before, 1, ch({ enabled: true }), 'enabled', true);
    expect(en[0].enabled).toBe(true);
    expect(en[1].enabled).toBe(true);
  });

  it('non-finite requests are ignored and out-of-range ones clamped', () => {
    const before = pair({ gainDb: -20 }, { gainDb: -26 });
    expect(applyChannelChange(before, 0, ch({ gainDb: Number.NaN }), 'gainDb', true)).toEqual(before);
    const out = applyChannelChange(before, 1, ch({ gainDb: 40 }), 'gainDb', false);
    expect(out[1].gainDb).toBe(0);
    const low = applyChannelChange(before, 1, ch({ gainDb: -400 }), 'gainDb', false);
    expect(low[1].gainDb).toBe(-60);
  });
});
