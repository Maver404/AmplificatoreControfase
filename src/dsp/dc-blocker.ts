import type { Complex } from './complex.ts';

export const DC_BLOCKER_HZ = 3;

const DENORMAL = 1e-30;

/** Pole radius of the DC blocker: R = e^{-2π·fc/fs}. */
const poleRadius = (fs: number, fc: number): number => {
  const safeFc = Number.isFinite(fc) && fc > 0 ? fc : DC_BLOCKER_HZ;
  return Math.exp((-2 * Math.PI * safeFc) / fs);
};

/** First-order DC blocker y[n] = x[n] − x[n−1] + R·y[n−1], i.e. H(z) = (1 − z⁻¹)/(1 − R z⁻¹). */
export class DcBlocker {
  private readonly r: number;
  private x1 = 0;
  private y1 = 0;

  constructor(fs: number, fc: number = DC_BLOCKER_HZ) {
    this.r = poleRadius(fs, fc);
  }

  reset(): void {
    this.x1 = 0;
    this.y1 = 0;
  }

  process(x: number): number {
    const y = x - this.x1 + this.r * this.y1;
    if (!Number.isFinite(y)) {
      this.reset();
      return 0;
    }
    this.x1 = x;
    this.y1 = Math.abs(y) < DENORMAL ? 0 : y;
    return y;
  }
}

/** Complex response of DcBlocker at frequency f: (1 − z⁻¹)/(1 − R z⁻¹), z = e^{jω}. */
export function dcBlockerResponse(fs: number, f: number, fc: number = DC_BLOCKER_HZ): Complex {
  const r = poleRadius(fs, fc);
  const w = (2 * Math.PI * f) / fs;
  const c = Math.cos(w);
  const s = Math.sin(w);
  // numerator 1 − e^{-jω} = (1 − cos ω) + j sin ω ; denominator 1 − R e^{-jω} = (1 − R cos ω) + j R sin ω
  const nr = 1 - c;
  const ni = s;
  const dr = 1 - r * c;
  const di = r * s;
  const d = dr * dr + di * di;
  return { re: (nr * dr + ni * di) / d, im: (ni * dr - nr * di) / d };
}
