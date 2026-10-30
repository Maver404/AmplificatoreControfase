/**
 * Acoustic rules of thumb used by the geometry assistant, the causality check and the UI.
 */

const DEG = Math.PI / 180;
const finiteOr = (v: number | undefined, d: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : d);

/** Speed of sound in air (m/s): 331.3·√(1 + T/273.15); T clamped above absolute zero. */
export const speedOfSound = (tempC: number): number => 331.3 * Math.sqrt(Math.max(0.01, 1 + finiteOr(tempC, 20) / 273.15));

/** Residual of anti-noise with amplitude ratio g and phase error Δφ: 10·log10(1 + g² − 2g·cosΔφ) dB. */
export const attenuationDb = (gainRatio: number, phaseErrRad: number): number =>
  10 * Math.log10(Math.max(1e-30, 1 + gainRatio * gainRatio - 2 * gainRatio * Math.cos(phaseErrRad)));

/** Largest phase error (degrees, exact gain) that still reaches `targetDb` (e.g. -20 dB → 5.73°). */
export const phaseToleranceDeg = (targetDb: number): number => {
  // 2 − 2cosΔφ = 10^(A/10)  ⇔  Δφ = 2·asin(√(10^(A/10)) / 2)
  const r = Math.sqrt(Math.pow(10, targetDb / 10)) / 2;
  return r >= 1 ? 180 : (2 * Math.asin(r)) / DEG;
};

/** Largest gain error (± dB, exact phase) that still reaches `targetDb` (e.g. -20 dB → ≈ 0.83 dB). */
export const gainToleranceDb = (targetDb: number): number => {
  const e = Math.pow(10, targetDb / 20);
  return e >= 1 ? Infinity : -20 * Math.log10(1 - e);
};

/** Timing error (seconds) equivalent to the phase tolerance at frequency f (−20 dB @ 100 Hz → 159 µs). */
export const timeToleranceSeconds = (targetDb: number, f: number): number => phaseToleranceDeg(targetDb) / (360 * f);

/**
 * Acoustic head start of the noise at the zone relative to the reference mic:
 * (d_src→zone − d_src→mic)/c. θ is the angle between the direction the sound travels at the mic
 * (source → mic) and the mic → zone line: θ = 0 means source, mic and zone are aligned.
 * Without `sourceDistanceM` the source is far: ≈ d_mic→zone·cosθ/c.
 */
export function headStartSeconds(g: {
  micToZoneM: number;
  sourceAngleDeg: number;
  sourceDistanceM?: number;
  tempC: number;
}): number {
  const c = speedOfSound(g.tempC);
  const d = Math.max(0, finiteOr(g.micToZoneM, 0));
  const th = finiteOr(g.sourceAngleDeg, 0) * DEG;
  const r = g.sourceDistanceM;
  if (r === undefined || !Number.isFinite(r) || r <= 0) return (d * Math.cos(th)) / c;
  // Mic at the origin, zone at (d, 0), source at −r·(cosθ, sinθ).
  const toZone = Math.hypot(d + r * Math.cos(th), r * Math.sin(th));
  return (toZone - r) / c;
}

export interface GeometryDelay {
  /** Delay the tap must add (≥ 0; 0 when the setup is not causal). */
  delaySeconds: number;
  causal: boolean;
  /** How much farther upstream the mic must go (m) to become causal; 0 if causal. */
  requiredExtraLeadM: number;
  cosTheta: number;
  /** cosθ < 0.7: the source is too far off the mic → zone line for a reliable head start. */
  warnAngle: boolean;
}

/** Below this cosθ the head start depends too much on the source direction. */
export const COS_THETA_WARN = 0.7;

/**
 * Delay the tap must add: head start − τ_S (latency + speaker→zone) − conditioning group delay.
 * `sourceAngleDeg` (optional, an extension of the contract) fills cosTheta/warnAngle; without it
 * cosTheta is 1 and no angle warning is raised.
 */
export function geometryDelay(
  headStartS: number,
  secondaryDelayS: number,
  conditioningGroupDelayS: number,
  tempC: number,
  sourceAngleDeg?: number,
): GeometryDelay {
  const raw = finiteOr(headStartS, 0) - finiteOr(secondaryDelayS, 0) - finiteOr(conditioningGroupDelayS, 0);
  const causal = raw >= 0;
  const cosTheta = sourceAngleDeg === undefined || !Number.isFinite(sourceAngleDeg) ? 1 : Math.cos(sourceAngleDeg * DEG);
  return {
    delaySeconds: causal ? raw : 0,
    causal,
    requiredExtraLeadM: causal ? 0 : -raw * speedOfSound(tempC),
    cosTheta,
    warnAngle: cosTheta < COS_THETA_WARN,
  };
}

/** Distance (m) sound travels during `latencySeconds`: the minimum mic lead for broadband ANC. */
export const minMicLeadMeters = (latencySeconds: number, tempC = 20): number =>
  Math.max(0, finiteOr(latencySeconds, 0)) * speedOfSound(tempC);
