import { describe, expect, it } from 'vitest';
import { biquadResponse, lowpass } from '../../src/dsp/biquad.ts';
import {
  besselGroupDelayMs,
  clampFinite,
  cleanFloat,
  dbToFraction,
  decimalsForStep,
  delayUnits,
  driftDegPerHz,
  formatAdaptive,
  formatDb,
  formatNumber,
  formatSigned,
  keySteps,
  modifierMultiplier,
  parseNumber,
  phaseDeltaToMs,
  positionToValue,
  snapToStep,
  stepInteger,
  stepValue,
  totalPhaseDeg,
  valueToPosition,
} from '../../src/ui/units.ts';

const NONE = { shiftKey: false, altKey: false };
const SHIFT = { shiftKey: true, altKey: false };
const ALT = { shiftKey: false, altKey: true };

describe('keyboard stepping', () => {
  it('maps keys to fine steps with modifiers', () => {
    expect(keySteps('ArrowUp', NONE)).toBe(1);
    expect(keySteps('ArrowRight', NONE)).toBe(1);
    expect(keySteps('ArrowDown', NONE)).toBe(-1);
    expect(keySteps('ArrowLeft', NONE)).toBe(-1);
    expect(keySteps('ArrowUp', SHIFT)).toBe(10);
    expect(keySteps('ArrowDown', ALT)).toBeCloseTo(-0.1, 12);
    expect(keySteps('PageUp', NONE)).toBe(100);
    expect(keySteps('PageDown', NONE)).toBe(-100);
    expect(keySteps('Home', NONE)).toBe(0);
    expect(keySteps('End', NONE)).toBe(0);
    expect(keySteps('a', NONE)).toBe(0);
  });

  it('ignores horizontal arrows in text fields', () => {
    expect(keySteps('ArrowLeft', NONE, false)).toBe(0);
    expect(keySteps('ArrowRight', SHIFT, false)).toBe(0);
    expect(keySteps('ArrowUp', NONE, false)).toBe(1);
  });

  it('combines Shift and Alt multiplicatively', () => {
    expect(modifierMultiplier({ shiftKey: true, altKey: true })).toBeCloseTo(1, 12);
  });
});

describe('stepValue', () => {
  it('removes floating-point noise', () => {
    let v = 0;
    for (let i = 0; i < 3; i++) v = stepValue(v, 1, 0.1, 1, -10, 10);
    expect(v).toBe(0.3);
  });

  it('clamps to the range and never returns NaN', () => {
    expect(stepValue(-0.005, 1, 0.01, 1, -60, 0)).toBe(0);
    expect(stepValue(-0.001, 10, 0.01, 1, -60, 0)).toBe(0);
    expect(stepValue(-59.99, -100, 0.01, 1, -60, 0)).toBe(-60);
    expect(stepValue(Number.NaN, 1, 0.01, 1, -60, 0)).toBe(-59.99);
    expect(stepValue(-20, Number.NaN, 0.01, 1, -60, 0)).toBe(-20);
    expect(Number.isNaN(stepValue(-20, 1, 0.01, 0, -60, 0))).toBe(false);
  });

  it('steps exactly in the display unit (0.01 campioni at 48 kHz)', () => {
    const fs = 48000;
    const samples = delayUnits(fs, 343).find((u) => u.id === 'samples');
    expect(samples).toBeDefined();
    const factor = samples!.factor;
    let ms = 10; // = 480 samples
    for (let i = 0; i < 37; i++) ms = stepValue(ms, 1, samples!.step!, factor, 0, 500);
    expect(cleanFloat(ms * factor)).toBe(480.37);
  });

  it('steps cm through the speed of sound', () => {
    const cm = delayUnits(48000, 343.2).find((u) => u.id === 'cm')!;
    expect(cm.factor).toBeCloseTo(34.32, 10);
    const ms = stepValue(1, 10, cm.step!, cm.factor, 0, 500); // +1 cm
    expect(ms * cm.factor).toBeCloseTo(35.32, 9);
  });

  it('integer stepping always moves by at least one', () => {
    expect(stepInteger(4, 0.1, 1, 8)).toBe(5);
    expect(stepInteger(4, -0.1, 1, 8)).toBe(3);
    expect(stepInteger(4, 10, 1, 8)).toBe(8);
    expect(stepInteger(1, -1, 1, 8)).toBe(1);
  });
});

describe('slider mapping', () => {
  it('round-trips in every scale', () => {
    for (const scale of ['linear', 'log', 'cubic'] as const) {
      for (const v of [20, 37.5, 100, 999.9, 2000]) {
        const p = valueToPosition(v, 20, 2000, scale);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
        expect(positionToValue(p, 20, 2000, scale)).toBeCloseTo(v, 6);
      }
    }
  });

  it('log puts the geometric mean in the middle, cubic expands the low end', () => {
    expect(valueToPosition(200, 20, 2000, 'log')).toBeCloseTo(0.5, 12);
    expect(valueToPosition(10, 0, 500, 'cubic')).toBeGreaterThan(0.25);
  });

  it('clamps out-of-range and non-finite input', () => {
    expect(valueToPosition(-5, 0, 10)).toBe(0);
    expect(valueToPosition(50, 0, 10)).toBe(1);
    expect(valueToPosition(Number.NaN, 0, 10)).toBe(0);
    expect(positionToValue(Number.NaN, 0, 10)).toBe(0);
    expect(positionToValue(2, 0, 10)).toBe(10);
  });

  it('snaps to the step', () => {
    expect(snapToStep(12.3456, 0.01)).toBe(12.35);
    expect(snapToStep(7, 0)).toBe(7);
  });
});

describe('parseNumber', () => {
  it.each([
    ['12', 12],
    ['12,5', 12.5],
    ['12.5', 12.5],
    ['−3,25', -3.25],
    ['-3.25', -3.25],
    ['+4', 4],
    [' 1.234,5 ', 1234.5],
    [',5', 0.5],
    ['12 ms', 12],
    ['-6dB', -6],
    ['90°', 90],
    ['1e3', 1000],
  ])('parses %s', (text, expected) => {
    expect(parseNumber(text)).toBe(expected);
  });

  it.each(['', 'abc', '-', '.', '1,2,3', '12.3.4', '1-2', 'NaN'])('rejects %s', (text) => {
    expect(parseNumber(text)).toBeNull();
  });

  it('understands infinities (the caller clamps them)', () => {
    expect(parseNumber('−∞')).toBe(-Infinity);
    expect(parseNumber('-inf')).toBe(-Infinity);
    expect(parseNumber('∞ dB')).toBe(Infinity);
  });
});

describe('formatting', () => {
  it('uses Italian decimals and a typographic minus', () => {
    expect(formatNumber(1234.5, 1)).toBe('1234,5');
    expect(formatNumber(-1.5, 2)).toBe('−1,50');
    expect(formatNumber(-0.004, 2)).toBe('0,00');
    expect(formatNumber(Number.NaN, 2)).toBe('—');
    expect(formatNumber(-Infinity, 2)).toBe('−∞');
  });

  it('formats signs and dB floors', () => {
    expect(formatSigned(0.833, 2)).toBe('+0,83');
    expect(formatSigned(-0.833, 2)).toBe('−0,83');
    expect(formatSigned(0, 1)).toBe('0,0');
    expect(formatDb(-120, 1)).toBe('−∞');
    expect(formatDb(-12.34, 1)).toBe('−12,3');
  });

  it('shows one extra decimal only when it is not zero', () => {
    expect(formatAdaptive(45, 1)).toBe('45,0');
    expect(formatAdaptive(45.01, 1)).toBe('45,01');
    expect(formatAdaptive(-6.5, 2)).toBe('\u22126,50');
    expect(formatAdaptive(-6.505, 2)).toBe('\u22126,505');
    expect(formatAdaptive(100, 0)).toBe('100');
    expect(formatAdaptive(100.5, 0)).toBe('100,5');
    expect(formatAdaptive(-0.00001, 2)).toBe('0,00');
    expect(formatAdaptive(-Infinity, 1)).toBe('\u2212∞');
    expect(parseNumber(formatAdaptive(-6.505, 2))).toBe(-6.505);
  });

  it('derives decimals from the step', () => {
    expect(decimalsForStep(1)).toBe(0);
    expect(decimalsForStep(0.1)).toBe(1);
    expect(decimalsForStep(0.01)).toBe(2);
    expect(decimalsForStep(0.25)).toBe(2);
    expect(decimalsForStep(0.001)).toBe(3);
  });

  it('round-trips through parse for every delay unit', () => {
    const units = delayUnits(44100, 343);
    for (const u of units) {
      const text = formatNumber(12.3456 * u.factor, u.decimals ?? 2);
      const back = parseNumber(text);
      expect(back).not.toBeNull();
      expect(Math.abs((back as number) / u.factor - 12.3456)).toBeLessThan(10 ** -(u.decimals ?? 2) / u.factor);
    }
  });
});

describe('phase helpers', () => {
  it('total phase = polarity + Δ, wrapped to [0, 360)', () => {
    expect(totalPhaseDeg(true, 0)).toBe(180);
    expect(totalPhaseDeg(true, 30)).toBe(210);
    expect(totalPhaseDeg(true, 180)).toBe(0);
    expect(totalPhaseDeg(true, -180)).toBe(0);
    expect(totalPhaseDeg(false, -30)).toBe(330);
    expect(totalPhaseDeg(false, 0)).toBe(0);
    expect(totalPhaseDeg(true, Number.NaN)).toBe(180);
  });

  it('converts Δ to time at the reference frequency', () => {
    expect(phaseDeltaToMs(36, 100)).toBeCloseTo(1, 12);
    expect(phaseDeltaToMs(-90, 250)).toBeCloseTo(-1, 12);
    expect(phaseDeltaToMs(30, 0)).toBe(0);
  });

  it('drift sensitivity: 1 Hz across 10 ms is 3.6° per harmonic', () => {
    expect(driftDegPerHz(1, 0.01)).toBeCloseTo(3.6, 12);
    expect(driftDegPerHz(4, 0.01)).toBeCloseTo(14.4, 12);
  });
});

describe('besselGroupDelayMs', () => {
  it('matches the 2nd-order section of the contract (Q = 1/√3, f0 = 1.272·fc)', () => {
    const fs = 48000;
    const fc = 500;
    const c = lowpass(fs, 1.272 * fc, 1 / Math.sqrt(3));
    const f1 = 5;
    const f2 = 10;
    const p1 = Math.atan2(biquadResponse(c, f1, fs).im, biquadResponse(c, f1, fs).re);
    const p2 = Math.atan2(biquadResponse(c, f2, fs).im, biquadResponse(c, f2, fs).re);
    const tauMs = (-(p2 - p1) / (2 * Math.PI * (f2 - f1))) * 1000;
    expect(besselGroupDelayMs(fc, 2)).toBeCloseTo(tauMs, 2);
  });

  it('is longer for the 4th order and zero for invalid input', () => {
    expect(besselGroupDelayMs(500, 4)).toBeGreaterThan(besselGroupDelayMs(500, 2));
    expect(besselGroupDelayMs(0, 2)).toBe(0);
  });
});

describe('meters and clamps', () => {
  it('maps dB to the −60…0 meter scale', () => {
    expect(dbToFraction(0)).toBe(1);
    expect(dbToFraction(-60)).toBe(0);
    expect(dbToFraction(-30)).toBe(0.5);
    expect(dbToFraction(6)).toBe(1);
    expect(dbToFraction(-Infinity)).toBe(0);
    expect(dbToFraction(Number.NaN)).toBe(0);
  });

  it('clampFinite never returns NaN', () => {
    expect(clampFinite(Number.NaN, 0, 1)).toBe(0);
    expect(clampFinite(Infinity, 0, 1, 0.5)).toBe(0.5);
    expect(clampFinite(2, 0, 1)).toBe(1);
  });
});
