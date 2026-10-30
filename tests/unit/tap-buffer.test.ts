import { describe, expect, it } from 'vitest';
import { TapCollector } from '../../src/dsp/tap-buffer.ts';

const frame = (...v: number[]) => Float64Array.from(v);

describe('TapCollector', () => {
  it('writes planar blocks and returns them only when full', () => {
    const t = new TapCollector(3, 4, 2);
    const v = new Float64Array(3);
    let out: Float32Array | null = null;
    for (let f = 0; f < 4; f++) {
      v[0] = f;
      v[1] = 10 + f;
      v[2] = 20 + f;
      out = t.push(v);
      if (f < 3) expect(out).toBeNull();
    }
    expect(out).not.toBeNull();
    expect(Array.from(out!)).toEqual([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]);
  });

  it('drops whole blocks when the pool is empty and counts them', () => {
    const t = new TapCollector(1, 2, 2);
    const blocks: (Float32Array | null)[] = [];
    for (let i = 0; i < 10; i++) {
      const b = t.push(frame(i));
      if (i % 2 === 1) blocks.push(b);
    }
    // 5 blocks: the first two have buffers, the other three are dropped
    expect(blocks[0]).not.toBeNull();
    expect(blocks[1]).not.toBeNull();
    expect(blocks.slice(2)).toEqual([null, null, null]);
    expect(t.dropped).toBe(3);
    expect(Array.from(blocks[0]!)).toEqual([0, 1]);
    expect(Array.from(blocks[1]!)).toEqual([2, 3]);
  });

  it('reuses recycled buffers (no allocation after construction)', () => {
    const t = new TapCollector(2, 8, 3);
    const seen = new Set<Float32Array>();
    const v = new Float64Array(2);
    for (let n = 0; n < 8 * 50; n++) {
      v[0] = n;
      v[1] = -n;
      const b = t.push(v);
      if (b) {
        seen.add(b);
        expect(b[0]).toBe(n - 7);
        expect(b[8 + 7]).toBe(-n);
        t.recycle(b);
      }
    }
    // only the 3 pooled buffers ever come out
    expect(seen.size).toBeLessThanOrEqual(3);
    expect(t.dropped).toBe(0);
  });

  it('a recycled buffer is the next one filled', () => {
    const t = new TapCollector(1, 1, 1);
    const a = t.push(frame(1));
    expect(a).not.toBeNull();
    expect(t.push(frame(2))).toBeNull(); // pool empty: dropped
    expect(t.dropped).toBe(1);
    t.recycle(a!);
    const b = t.push(frame(3));
    expect(b).toBe(a);
    expect(b![0]).toBe(3);
  });

  it('ignores wrong-size, duplicate and excess buffers in recycle', () => {
    const t = new TapCollector(1, 4, 1);
    t.recycle(new Float32Array(3));
    t.recycle(new Float32Array(4)); // pool already full
    let b: Float32Array | null = null;
    for (let i = 0; i < 4; i++) b = t.push(frame(i));
    expect(b).not.toBeNull();
    t.recycle(b!);
    t.recycle(b!);
    // only one buffer available: the second block is dropped
    for (let i = 0; i < 8; i++) t.push(frame(i));
    expect(t.dropped).toBe(1);
  });

  it('sanitises non-finite values and pads missing channels', () => {
    const t = new TapCollector(3, 1, 1);
    const b = t.push(frame(Number.NaN, Number.POSITIVE_INFINITY));
    expect(Array.from(b!)).toEqual([0, 0, 0]);
  });

  it('reset discards the partial block and returns its buffer to the pool', () => {
    const t = new TapCollector(1, 4, 1);
    t.push(frame(1));
    t.push(frame(2));
    t.reset();
    let b: Float32Array | null = null;
    for (let i = 0; i < 4; i++) b = t.push(frame(10 + i));
    expect(Array.from(b!)).toEqual([10, 11, 12, 13]);
    expect(t.dropped).toBe(0);
  });
});
