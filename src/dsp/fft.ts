/**
 * In-place iterative radix-2 complex FFT in double precision.
 * forward/inverse allocate nothing (tables are built in the constructor), so the howl detector can
 * use it inside the AudioWorklet.
 */

/** Smallest power of two ≥ n (1 for n ≤ 1). */
export const nextPow2 = (n: number): number => {
  if (!(n > 1)) return 1;
  let p = 1;
  while (p < n) p *= 2;
  return p;
};

/** Largest size accepted (2^26 points = 1 GiB of complex doubles): a guard against absurd requests. */
const MAX_FFT_SIZE = 2 ** 26;

export class FFT {
  readonly n: number;
  private readonly rev: Uint32Array;
  private readonly cosT: Float64Array;
  private readonly sinT: Float64Array;

  constructor(n: number) {
    if (!Number.isInteger(n) || n < 1 || n > MAX_FFT_SIZE || (n & (n - 1)) !== 0) {
      throw new RangeError(`FFT size must be a power of two between 1 and ${MAX_FFT_SIZE}, got ${n}`);
    }
    this.n = n;
    const half = Math.max(1, n >>> 1);
    this.cosT = new Float64Array(half);
    this.sinT = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      // Direct evaluation (no recurrence) keeps twiddles accurate for very long transforms.
      const a = (2 * Math.PI * k) / n;
      this.cosT[k] = Math.cos(a);
      this.sinT[k] = Math.sin(a);
    }
    let bits = 0;
    while (1 << bits < n) bits++;
    this.rev = new Uint32Array(n);
    for (let i = 1; i < n; i++) this.rev[i] = (this.rev[i >>> 1] >>> 1) | ((i & 1) << (bits - 1));
  }

  /** X[k] = Σ x[m]·e^{-j2πkm/n}, in place. */
  forward(re: Float64Array, im: Float64Array): void {
    this.run(re, im, -1);
  }

  /** x[m] = (1/n)·Σ X[k]·e^{+j2πkm/n}, in place. */
  inverse(re: Float64Array, im: Float64Array): void {
    this.run(re, im, 1);
    const s = 1 / this.n;
    for (let i = 0; i < this.n; i++) {
      re[i] *= s;
      im[i] *= s;
    }
  }

  private run(re: Float64Array, im: Float64Array, sign: 1 | -1): void {
    const n = this.n;
    if (re.length !== n || im.length !== n) {
      throw new RangeError(`FFT of size ${n} needs arrays of length ${n} (got ${re.length}/${im.length})`);
    }
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    const cosT = this.cosT;
    const sinT = this.sinT;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >>> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const wr = cosT[k];
          const wi = sign * sinT[k];
          const a = start + j;
          const b = a + half;
          const xr = re[b];
          const xi = im[b];
          const tr = xr * wr - xi * wi;
          const ti = xr * wi + xi * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }
}

const cache: FFT[] = [];

/**
 * Returns a cached FFT of size n: tables for long transforms are expensive to rebuild for every
 * calibration measurement. An FFT holds no state besides its read-only tables, so sharing is safe.
 */
export function sharedFFT(n: number): FFT {
  for (const f of cache) if (f.n === n) return f;
  const f = new FFT(n);
  cache.unshift(f);
  if (cache.length > 3) cache.pop();
  return f;
}
