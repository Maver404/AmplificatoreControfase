/**
 * Feedback neutralisation: removes from the reference mic the part of the speakers' own output
 * that reaches it, using a measured FIR per channel (bulk delay + taps).
 *   x̂[n] = x[n] − Σ_ch Σ_k h_ch[k]·y_ch[n − bulk_ch − k]
 * Per-sample calls allocate nothing; the history is a mirrored ring so each FIR is one contiguous
 * dot product.
 */

/** Samples of the measured IR kept before the first arrival (band-limited IRs start early). */
export const FEEDBACK_FIR_MARGIN = 64;
/** Raised-cosine fade at the end of the FIR window, at most this many samples. */
export const FEEDBACK_FIR_FADE = 32;

export class FeedbackCanceller {
  private readonly size: number;
  private readonly mask: number;
  private readonly maxBulk: number;
  private readonly maxTaps: number;
  /** Mirrored output histories: sample i is stored at i and i + size. */
  private readonly hist: [Float64Array, Float64Array];
  /** Reversed taps (oldest history sample first) per channel. */
  private readonly coef: [Float64Array, Float64Array];
  private readonly nTaps = [0, 0];
  private readonly bulk = [1, 1];
  /** Index of the newest output sample in the ring. */
  private w: number;

  constructor(fs: number, maxBulkDelay: number, maxTaps: number) {
    if (!(fs > 0)) throw new RangeError('FeedbackCanceller: fs must be positive');
    this.maxBulk = Math.max(1, Math.floor(maxBulkDelay));
    this.maxTaps = Math.max(1, Math.floor(maxTaps));
    let size = 1;
    while (size < this.maxBulk + this.maxTaps + 1) size <<= 1;
    this.size = size;
    this.mask = size - 1;
    this.hist = [new Float64Array(2 * size), new Float64Array(2 * size)];
    this.coef = [new Float64Array(this.maxTaps), new Float64Array(this.maxTaps)];
    this.w = size - 1;
  }

  /**
   * Sets the FIR of one channel. bulkDelay is rounded and clamped to [1, maxBulkDelay] (the current
   * output sample is not known when predict() runs); taps beyond maxTaps are dropped; a FIR with
   * any non-finite value disables the channel instead of injecting garbage into the reference.
   */
  setFir(ch: 0 | 1, bulkDelay: number, taps: Float64Array | null): void {
    const c = this.coef[ch];
    c.fill(0);
    this.nTaps[ch] = 0;
    if (!taps || taps.length === 0 || !Number.isFinite(bulkDelay)) return;
    const n = Math.min(taps.length, this.maxTaps);
    for (let k = 0; k < n; k++) if (!Number.isFinite(taps[k])) return;
    for (let k = 0; k < n; k++) c[n - 1 - k] = taps[k];
    this.nTaps[ch] = n;
    this.bulk[ch] = Math.min(this.maxBulk, Math.max(1, Math.round(bulkDelay)));
  }

  /** Prediction of the feedback component in the current reference sample (call before computing outputs). */
  predict(): number {
    let acc = 0;
    for (let ch = 0; ch < 2; ch++) {
      const n = this.nTaps[ch];
      if (n === 0) continue;
      // Oldest needed sample: y[n − bulk − (N−1)]; newest: y[n − bulk] = ring[w − bulk + 1].
      const start = (this.w - this.bulk[ch] - n + 2 + this.size) & this.mask;
      const h = this.hist[ch];
      const c = this.coef[ch];
      for (let j = 0; j < n; j++) acc += c[j] * h[start + j];
    }
    if (!Number.isFinite(acc)) {
      this.reset();
      return 0;
    }
    return acc;
  }

  /** Push this sample's final outputs (call after computing outputs). */
  pushOutputs(yL: number, yR: number): void {
    const w = (this.w + 1) & this.mask;
    this.w = w;
    const l = Number.isFinite(yL) ? yL : 0;
    const r = Number.isFinite(yR) ? yR : 0;
    const hl = this.hist[0];
    const hr = this.hist[1];
    hl[w] = l;
    hl[w + this.size] = l;
    hr[w] = r;
    hr[w + this.size] = r;
  }

  /** Clears the output history (FIRs are kept). */
  reset(): void {
    this.hist[0].fill(0);
    this.hist[1].fill(0);
    this.w = this.size - 1;
  }
}

/**
 * Builds the FIR from a measured speaker→reference impulse response: bulkDelay = first arrival −
 * margin (≥ 1), taps = IR window of `taps` length starting there, with a short raised-cosine fade
 * at the end (and a half-margin fade-in, so truncated pre-ringing does not end in a step).
 */
export function designFeedbackFir(ir: Float64Array, arrivalSamples: number, taps: number): { bulkDelay: number; taps: Float64Array } {
  const n = Math.max(1, Math.floor(taps));
  const margin = Math.min(FEEDBACK_FIR_MARGIN, Math.floor(n / 4));
  const arrival = Number.isFinite(arrivalSamples) ? Math.floor(arrivalSamples) : 0;
  const bulkDelay = Math.max(1, arrival - margin);
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const i = bulkDelay + k;
    out[k] = i < ir.length && Number.isFinite(ir[i]) ? ir[i] : 0;
  }
  const fadeIn = Math.floor(margin / 2);
  for (let k = 0; k < fadeIn; k++) out[k] *= 0.5 - 0.5 * Math.cos((Math.PI * (k + 1)) / (fadeIn + 1));
  const fade = Math.min(FEEDBACK_FIR_FADE, Math.floor(n / 8));
  for (let j = 0; j < fade; j++) out[n - fade + j] *= 0.5 + 0.5 * Math.cos((Math.PI * (j + 1)) / (fade + 1));
  return { bulkDelay, taps: out };
}
