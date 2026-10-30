import { describe, expect, it } from 'vitest';
import { applyChannelChange, clearTrims, wrapPhaseDeg } from '../../src/app/link.ts';
import type { ChannelPair } from '../../src/app/link.ts';
import { neutralTrims } from '../../src/dsp/params.ts';
import type { ChannelParams } from '../../src/dsp/params.ts';

const ch = (p: Partial<ChannelParams>): ChannelParams => ({
  enabled: true,
  polarityInvert: true,
  gainDb: -20,
  delayMs: 0,
  phaseDeltaDeg: 0,
  harmonicTrims: neutralTrims(),
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

  it('harmonic trims, unlinked: the edited channel takes them (clamped, a copy), the other keeps its own', () => {
    const before = pair({}, { harmonicTrims: neutralTrims().map((_, k) => ({ gainDb: -k, phaseDeg: 10 * k })) });
    const want = neutralTrims();
    want[2] = { gainDb: 9, phaseDeg: -200 };
    want[3] = { gainDb: Number.NaN, phaseDeg: 15 };
    const out = applyChannelChange(before, 0, ch({ harmonicTrims: want }), 'harmonicTrims', false);
    expect(out[0].harmonicTrims[2]).toEqual({ gainDb: 6, phaseDeg: -180 });
    // A non-finite field is left as it was.
    expect(out[0].harmonicTrims[3]).toEqual({ gainDb: 0, phaseDeg: 15 });
    expect(out[0].harmonicTrims).toHaveLength(8);
    expect(out[0].harmonicTrims).not.toBe(want);
    expect(out[1].harmonicTrims).toEqual(before[1].harmonicTrims);
    expect(out[0].gainDb).toBe(before[0].gainDb);
  });

  it('harmonic trims, linked: every entry moves by the same amount on both channels, offsets kept within range', () => {
    const l = neutralTrims();
    const r = neutralTrims();
    l[1] = { gainDb: -2, phaseDeg: 20 };
    r[1] = { gainDb: 3, phaseDeg: -10 };
    const before = pair({ harmonicTrims: l }, { harmonicTrims: r });
    const want = l.map((t) => ({ ...t }));
    want[1] = { gainDb: 0, phaseDeg: 50 };
    const out = applyChannelChange(before, 0, ch({ harmonicTrims: want }), 'harmonicTrims', true);
    expect(out[0].harmonicTrims[1]).toEqual({ gainDb: 0, phaseDeg: 50 });
    expect(out[1].harmonicTrims[1].gainDb).toBeCloseTo(5, 12);
    expect(out[1].harmonicTrims[1].phaseDeg).toBeCloseTo(20, 12);
    // R would pass +6 dB: the shared rise stops there (L +1 dB, R +6 dB), the offset survives.
    want[1] = { gainDb: 4, phaseDeg: 50 };
    const capped = applyChannelChange(out, 0, ch({ harmonicTrims: want }), 'harmonicTrims', true);
    expect(capped[1].harmonicTrims[1].gainDb).toBeCloseTo(6, 12);
    expect(capped[0].harmonicTrims[1].gainDb).toBeCloseTo(1, 12);
    // Untouched entries stay as they were on both channels; the inputs are not mutated.
    for (const k of [0, 2, 3, 4, 5, 6, 7]) {
      expect(out[0].harmonicTrims[k]).toEqual(l[k]);
      expect(out[1].harmonicTrims[k]).toEqual(r[k]);
    }
    expect(before[1].harmonicTrims[1]).toEqual({ gainDb: 3, phaseDeg: -10 });
    // An edit on R moves L the same way.
    const fromR = applyChannelChange(before, 1, ch({ harmonicTrims: r.map((t, k) => (k === 1 ? { gainDb: 1, phaseDeg: -10 } : t)) }), 'harmonicTrims', true);
    expect(fromR[1].harmonicTrims[1]).toEqual({ gainDb: 1, phaseDeg: -10 });
    expect(fromR[0].harmonicTrims[1]).toEqual({ gainDb: -4, phaseDeg: 20 });
    const alone = applyChannelChange(before, 1, ch({ harmonicTrims: neutralTrims() }), 'harmonicTrims', false);
    expect(alone[1].harmonicTrims).toEqual(neutralTrims());
    expect(alone[0].harmonicTrims).toEqual(l);
  });

  it('harmonic trims, linked: phases are angles, moved by the same delta and wrapped, never limited by ±180°', () => {
    const l = neutralTrims();
    const r = neutralTrims();
    r[0] = { gainDb: 0, phaseDeg: 175 };
    const before = pair({ harmonicTrims: l }, { harmonicTrims: r });
    const want = l.map((t) => ({ ...t }));
    want[0] = { gainDb: 0, phaseDeg: 20 };
    const out = applyChannelChange(before, 0, ch({ harmonicTrims: want }), 'harmonicTrims', true);
    // +20° on L (not the +5° a ±180 limit would allow), 175° + 20° = 195° ≡ −165° on R
    expect(out[0].harmonicTrims[0].phaseDeg).toBe(20);
    expect(out[1].harmonicTrims[0].phaseDeg).toBeCloseTo(-165, 9);
    // −170° / +170° (20° apart on the circle): an edit moves both
    const close = pair({ harmonicTrims: neutralTrims().map((t, k) => (k === 0 ? { gainDb: 0, phaseDeg: -170 } : t)) }, {
      harmonicTrims: neutralTrims().map((t, k) => (k === 0 ? { gainDb: 0, phaseDeg: 170 } : t)),
    });
    const w2 = close[0].harmonicTrims.map((t) => ({ ...t }));
    w2[0].phaseDeg = -150;
    const moved = applyChannelChange(close, 0, ch({ harmonicTrims: w2 }), 'harmonicTrims', true);
    expect(moved[0].harmonicTrims[0].phaseDeg).toBe(-150);
    expect(moved[1].harmonicTrims[0].phaseDeg).toBeCloseTo(-170, 9);
    // an unchanged phase leaves the other channel's exactly as it is (−180° is not rewritten as 180°)
    const edge = pair({}, { harmonicTrims: neutralTrims().map((t, k) => (k === 0 ? { gainDb: 0, phaseDeg: -180 } : t)) });
    const g = neutralTrims();
    g[0] = { gainDb: 2, phaseDeg: 0 };
    expect(applyChannelChange(edge, 0, ch({ harmonicTrims: g }), 'harmonicTrims', true)[1].harmonicTrims[0]).toEqual({ gainDb: 2, phaseDeg: -180 });
    expect([wrapPhaseDeg(195), wrapPhaseDeg(-180), wrapPhaseDeg(180), wrapPhaseDeg(-540), wrapPhaseDeg(30)]).toEqual([-165, 180, 180, 180, 30]);
  });
});

describe('clearTrims («Azzera armoniche»)', () => {
  const trimmed = (k: number, gainDb: number, phaseDeg: number) => neutralTrims().map((t, i) => (i === k ? { gainDb, phaseDeg } : t));

  it('unlinked: that channel to 0, the other untouched (as copies)', () => {
    const before = pair({ harmonicTrims: trimmed(1, -6, 30) }, { harmonicTrims: trimmed(3, 4, -90) });
    const out = clearTrims(before, 0, false);
    expect(out[0].harmonicTrims).toEqual(neutralTrims());
    expect(out[1].harmonicTrims).toEqual(before[1].harmonicTrims);
    expect(out[1].harmonicTrims).not.toBe(before[1].harmonicTrims);
    expect(before[0].harmonicTrims[1]).toEqual({ gainDb: -6, phaseDeg: 30 });
    expect(out[0].gainDb).toBe(before[0].gainDb);
  });

  it('linked: both channels to 0 — never the other one moved by the cleared amounts', () => {
    // Cases where a linked delta failed: −170°/+170° (L not cleared), −28/+4 dB (R raised to +6 dB), L only trimmed
    // (R given −6 dB → +6 dB and −30°).
    for (const [l, r] of [
      [trimmed(0, 0, -170), trimmed(0, 0, 170)],
      [trimmed(1, -28, 0), trimmed(1, 4, 0)],
      [trimmed(1, -6, 30), neutralTrims()],
    ]) {
      const out = clearTrims(pair({ harmonicTrims: l }, { harmonicTrims: r }), 0, true);
      expect(out[0].harmonicTrims).toEqual(neutralTrims());
      expect(out[1].harmonicTrims).toEqual(neutralTrims());
    }
  });
});
