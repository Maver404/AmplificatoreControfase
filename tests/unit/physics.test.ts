import { describe, expect, it } from 'vitest';
import {
  attenuationDb,
  gainToleranceDb,
  geometryDelay,
  headStartSeconds,
  minMicLeadMeters,
  phaseToleranceDeg,
  speedOfSound,
  timeToleranceSeconds,
} from '../../src/dsp/physics.ts';

describe('physics', () => {
  it('speed of sound follows 331.3·√(1 + T/273.15)', () => {
    expect(speedOfSound(20)).toBeCloseTo(343.2, 1);
    expect(speedOfSound(0)).toBeCloseTo(331.3, 6);
    expect(speedOfSound(-10)).toBeLessThan(speedOfSound(0));
  });

  it('phase and gain tolerances for a residual target', () => {
    expect(phaseToleranceDeg(-20)).toBeCloseTo(5.73, 2);
    expect(attenuationDb(1, (phaseToleranceDeg(-20) * Math.PI) / 180)).toBeCloseTo(-20, 9);
    expect(phaseToleranceDeg(-10)).toBeCloseTo(18.19, 1);
    expect(phaseToleranceDeg(10)).toBe(180);
    expect(gainToleranceDb(-20)).toBeCloseTo(0.915, 2);
    expect(attenuationDb(Math.pow(10, -gainToleranceDb(-20) / 20), 0)).toBeCloseTo(-20, 9);
    expect(timeToleranceSeconds(-20, 100) * 1e6).toBeCloseTo(159.2, 1);
  });

  it('attenuation is 0 dB at 90° with unit gain and +6 dB when polarity is wrong', () => {
    expect(attenuationDb(1, Math.PI / 3)).toBeCloseTo(0, 9);
    expect(attenuationDb(1, Math.PI)).toBeCloseTo(6.02, 2);
    expect(attenuationDb(0, 0)).toBeCloseTo(0, 9);
  });

  it('head start: aligned, at an angle, and with a near source', () => {
    const d = speedOfSound(20) * 0.01; // 10 ms of travel
    expect(headStartSeconds({ micToZoneM: d, sourceAngleDeg: 0, tempC: 20 })).toBeCloseTo(0.01, 9);
    expect(headStartSeconds({ micToZoneM: d, sourceAngleDeg: 60, tempC: 20 })).toBeCloseTo(0.005, 9);
    expect(headStartSeconds({ micToZoneM: d, sourceAngleDeg: 90, tempC: 20 })).toBeCloseTo(0, 9);
    expect(headStartSeconds({ micToZoneM: d, sourceAngleDeg: 180, tempC: 20 })).toBeCloseTo(-0.01, 9);
    // Aligned near source: exactly d/c regardless of the distance.
    expect(headStartSeconds({ micToZoneM: d, sourceAngleDeg: 0, sourceDistanceM: 0.5, tempC: 20 })).toBeCloseTo(0.01, 9);
    // Off-axis near source: (|zone − src| − r)/c, never below the far-field value d·cosθ/c.
    const near = headStartSeconds({ micToZoneM: 3, sourceAngleDeg: 45, sourceDistanceM: 2, tempC: 20 });
    const exact = (Math.hypot(3 + 2 * Math.SQRT1_2, 2 * Math.SQRT1_2) - 2) / speedOfSound(20);
    expect(near).toBeCloseTo(exact, 12);
    expect(near).toBeGreaterThan((3 * Math.SQRT1_2) / speedOfSound(20));
    // Very far source converges to the far-field formula.
    const far = headStartSeconds({ micToZoneM: 3, sourceAngleDeg: 45, sourceDistanceM: 1e6, tempC: 20 });
    expect(far).toBeCloseTo((3 * Math.SQRT1_2) / speedOfSound(20), 7);
  });

  it('geometry delay and causality', () => {
    const ok = geometryDelay(0.01, 0.006, 0.001, 20, 30);
    expect(ok.causal).toBe(true);
    expect(ok.delaySeconds).toBeCloseTo(0.003, 12);
    expect(ok.requiredExtraLeadM).toBe(0);
    expect(ok.cosTheta).toBeCloseTo(Math.cos(Math.PI / 6), 12);
    expect(ok.warnAngle).toBe(false);

    const late = geometryDelay(0.005, 0.006, 0.001, 20, 60);
    expect(late.causal).toBe(false);
    expect(late.delaySeconds).toBe(0);
    expect(late.requiredExtraLeadM).toBeCloseTo(0.002 * speedOfSound(20), 9);
    expect(late.warnAngle).toBe(true);

    const exact = geometryDelay(0.007, 0.006, 0.001, 20);
    expect(exact.causal).toBe(true);
    expect(exact.cosTheta).toBe(1);
  });

  it('minimum mic lead ≈ 3.4 m per 10 ms', () => {
    expect(minMicLeadMeters(0.01)).toBeCloseTo(3.432, 3);
    expect(minMicLeadMeters(0.01, 0)).toBeCloseTo(3.313, 3);
    expect(minMicLeadMeters(-1)).toBe(0);
  });
});
