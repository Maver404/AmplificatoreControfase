/** Minimal complex-number helpers used by frequency-response and calibration code. */
export interface Complex {
  re: number;
  im: number;
}

export const complex = (re: number, im = 0): Complex => ({ re, im });

export const cMul = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});

export const cDiv = (a: Complex, b: Complex): Complex => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};

export const cAdd = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });
export const cSub = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });
export const cScale = (a: Complex, k: number): Complex => ({ re: a.re * k, im: a.im * k });
export const cAbs = (a: Complex): number => Math.hypot(a.re, a.im);
export const cArg = (a: Complex): number => Math.atan2(a.im, a.re);

/** e^{-j·ω·d}: response of a pure delay of `delaySamples` at frequency `f`. */
export const cDelay = (f: number, fs: number, delaySamples: number): Complex => {
  const w = (-2 * Math.PI * f * delaySamples) / fs;
  return { re: Math.cos(w), im: Math.sin(w) };
};

/** Wrap an angle in radians to (-π, π]. */
export const wrapPi = (a: number): number => {
  const t = (a + Math.PI) % (2 * Math.PI);
  return (t < 0 ? t + 2 * Math.PI : t) - Math.PI;
};
