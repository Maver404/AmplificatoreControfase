/**
 * Collects `channels` values per frame into planar Float32Array blocks of `frames` frames
 * (channel k occupies [k·frames, (k+1)·frames)).
 * push() returns a full block (ownership passes to the caller, who transfers it) or null.
 * Buffers come from a pool refilled by recycle(); if the pool is empty the block is dropped (counted).
 */
export class TapCollector {
  readonly channels: number;
  readonly frames: number;
  private readonly pool: (Float32Array | null)[];
  private poolCount = 0;
  private cur: Float32Array | null = null;
  private idx = 0;
  /** True while the block being collected has no buffer and will be dropped. */
  private dropping = false;
  private droppedBlocks = 0;

  constructor(channels: number, frames: number, poolSize = 4) {
    const int = (v: number, min: number, fallback: number) => (Number.isFinite(v) ? Math.max(min, Math.floor(v)) : fallback);
    this.channels = int(channels, 1, 1);
    this.frames = int(frames, 1, 1);
    const size = int(poolSize, 0, 4);
    this.pool = new Array<Float32Array | null>(size).fill(null);
    for (let i = 0; i < size; i++) this.pool[i] = new Float32Array(this.channels * this.frames);
    this.poolCount = size;
  }

  push(values: Float64Array): Float32Array | null {
    if (this.idx === 0) {
      // Start of a new block: take a buffer, or drop the whole block if none is available.
      if (this.poolCount > 0) {
        this.poolCount--;
        this.cur = this.pool[this.poolCount];
        this.pool[this.poolCount] = null;
        this.dropping = false;
      } else {
        this.cur = null;
        this.dropping = true;
      }
    }
    const buf = this.cur;
    if (buf !== null) {
      const n = values.length < this.channels ? values.length : this.channels;
      const frames = this.frames;
      const i = this.idx;
      for (let c = 0; c < n; c++) {
        const v = values[c];
        buf[c * frames + i] = Number.isFinite(v) ? v : 0;
      }
      for (let c = n; c < this.channels; c++) buf[c * frames + i] = 0;
    }
    this.idx++;
    if (this.idx < this.frames) return null;
    this.idx = 0;
    if (this.dropping) {
      this.dropping = false;
      this.droppedBlocks++;
      return null;
    }
    this.cur = null;
    return buf;
  }

  /** Returns a buffer to the pool (ignored if it has the wrong size or the pool is full). */
  recycle(buf: Float32Array): void {
    if (buf.length !== this.channels * this.frames) return;
    if (this.poolCount >= this.pool.length) return;
    for (let i = 0; i < this.poolCount; i++) if (this.pool[i] === buf) return;
    if (buf === this.cur) return;
    this.pool[this.poolCount++] = buf;
  }

  /** Number of whole blocks dropped because the pool was empty. */
  get dropped(): number {
    return this.droppedBlocks;
  }

  /** Discards the partially collected block (its buffer goes back to the pool) and clears the drop count. */
  reset(): void {
    if (this.cur !== null) {
      const c = this.cur;
      this.cur = null;
      this.recycle(c);
    }
    this.idx = 0;
    this.dropping = false;
    this.droppedBlocks = 0;
  }
}
