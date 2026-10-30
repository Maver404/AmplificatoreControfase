import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { PROCESSOR_NAME, TAP_BLOCK_FRAMES } from '../../src/audio/protocol.ts';
import type { WorkletInMsg, WorkletOutMsg } from '../../src/audio/protocol.ts';
import { AntiPhaseEngine } from '../../src/dsp/antiphase-engine.ts';
import { clampParams } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';

const FS = 48000;
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const WORKLET_FILE = resolve(ROOT, 'src/worklet/antiphase.worklet.ts');

class FakePort {
  onmessage: ((e: { data: unknown }) => void) | null = null;
  readonly sent: { msg: WorkletOutMsg; transfer: unknown }[] = [];
  postMessage(msg: WorkletOutMsg, transfer?: unknown): void {
    this.sent.push({ msg, transfer });
  }
  deliver(data: WorkletInMsg): void {
    this.onmessage?.({ data });
  }
  of<T extends WorkletOutMsg['type']>(type: T): Extract<WorkletOutMsg, { type: T }>[] {
    return this.sent.map((s) => s.msg).filter((m): m is Extract<WorkletOutMsg, { type: T }> => m.type === type);
  }
}

/** Stand-in for AudioWorkletProcessor: only `port` is used by the adapter. */
class FakeProcessorBase {
  readonly port = new FakePort();
}

interface Processor {
  readonly port: FakePort;
  process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean;
}
type ProcessorCtor = new () => Processor;

const registered = new Map<string, ProcessorCtor>();

beforeAll(async () => {
  vi.stubGlobal('AudioWorkletProcessor', FakeProcessorBase);
  vi.stubGlobal('sampleRate', FS);
  vi.stubGlobal('registerProcessor', (name: string, ctor: ProcessorCtor) => registered.set(name, ctor));
  // A non-literal specifier keeps tsc (node config, no audioworklet types) out of the worklet source.
  const spec = '../../src/worklet/antiphase.worklet.ts';
  await import(/* @vite-ignore */ spec);
});

afterEach(() => {
  vi.restoreAllMocks();
});

afterAll(() => {
  vi.unstubAllGlobals();
});

const create = (): Processor => {
  const Ctor = registered.get(PROCESSOR_NAME);
  expect(Ctor).toBeDefined();
  return new Ctor!();
};

const sine = (n: number, f: number, amp: number): Float32Array => {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * f * i) / FS);
  return x;
};

const testParams = (): EngineParams => {
  const p = clampParams(undefined);
  p.masterDb = -6;
  p.ceilingDb = -6;
  p.channels[0].gainDb = 0;
  p.channels[1].gainDb = -3;
  p.channels[1].polarityInvert = false;
  p.errorChannel = 1;
  return p;
};

/** Runs `n` frames through the processor in render quanta of `q`; returns what was written to each output channel. */
function drive(
  proc: Processor,
  n: number,
  q: number,
  inputs: (Float32Array | null)[],
  outChannels: number,
): Float32Array[] {
  const out = Array.from({ length: outChannels }, () => new Float32Array(n));
  for (let off = 0; off < n; off += q) {
    const m = Math.min(q, n - off);
    const ins = inputs.filter((x): x is Float32Array => x !== null).map((x) => x.slice(off, off + m));
    const outs = Array.from({ length: outChannels }, () => new Float32Array(m));
    expect(proc.process([ins], outChannels === 0 ? [] : [outs], {})).toBe(true);
    for (let c = 0; c < outChannels; c++) out[c].set(outs[c], off);
  }
  return out;
}

describe('antiphase worklet adapter', () => {
  it(`registers '${PROCESSOR_NAME}' and posts 'ready' once with the sample rate`, () => {
    const p = create();
    expect(p.port.of('ready')).toEqual([{ type: 'ready', sampleRate: FS }]);
    drive(p, 1024, 128, [], 2);
    expect(p.port.of('ready').length).toBe(1);
  });

  it('maps inputs[0][0..1] and outputs[0][0..1] onto the engine exactly', () => {
    const proc = create();
    const direct = new AntiPhaseEngine(FS);
    const msgs: WorkletInMsg[] = [
      { type: 'params', params: testParams() },
      { type: 'arm', armed: true },
    ];
    for (const m of msgs) {
      proc.port.deliver(m);
      direct.handleMessage(m);
    }
    const n = 2 * FS;
    const x0 = sine(n, 120, 0.3);
    const x1 = sine(n, 330, 0.2);
    const [L, R] = drive(proc, n, 128, [x0, x1], 2);
    const dl = new Float32Array(n);
    const dr = new Float32Array(n);
    for (let off = 0; off < n; off += 128) {
      direct.process(x0.subarray(off, off + 128), x1.subarray(off, off + 128), dl.subarray(off, off + 128), dr.subarray(off, off + 128));
    }
    expect(L).toEqual(dl);
    expect(R).toEqual(dr);
    let energy = 0;
    for (const v of L) energy += v * v;
    expect(energy).toBeGreaterThan(1);
    const meters = proc.port.of('meters');
    expect(meters.length).toBeGreaterThan(50);
    const last = meters[meters.length - 1];
    expect(last.armed).toBe(true);
    expect(last.err).not.toBeNull();
    expect(last.blockSize).toBe(128);
  });

  it('no input connected (inputs[0] empty) → silence, no throw, ch2 absent', () => {
    const proc = create();
    proc.port.deliver({ type: 'arm', armed: true });
    const [L, R] = drive(proc, FS + 4096, 128, [], 2);
    expect(L.every((v) => v === 0) && R.every((v) => v === 0)).toBe(true);
    expect(proc.port.of('channelCheck').map((m) => m.status)).toEqual(['absent']);
    expect(proc.port.of('fault')).toEqual([]);
  });

  it('a mono output (outputs[0] with 1 channel) and a missing output do not throw', () => {
    const proc = create();
    proc.port.deliver({ type: 'params', params: testParams() });
    proc.port.deliver({ type: 'arm', armed: true });
    const n = 2 * FS;
    const x = sine(n, 200, 0.3);
    const [L] = drive(proc, n, 128, [x], 1);
    let energy = 0;
    for (const v of L) energy += v * v;
    expect(energy).toBeGreaterThan(1);
    expect(L.every(Number.isFinite)).toBe(true);
    expect(() => drive(proc, 1024, 128, [x], 0)).not.toThrow();
    expect(proc.port.of('fault')).toEqual([]);
  });

  it('works with any render quantum (64, 256) and mismatched input lengths', () => {
    const proc = create();
    proc.port.deliver({ type: 'arm', armed: true });
    drive(proc, 4096, 64, [sine(4096, 100, 0.1)], 2);
    drive(proc, 4096, 256, [sine(4096, 100, 0.1)], 2);
    const outs = [new Float32Array(128), new Float32Array(128)];
    expect(() => proc.process([[new Float32Array(32)]], [outs], {})).not.toThrow();
    expect(outs[0].every(Number.isFinite)).toBe(true);
    const meters = proc.port.of('meters');
    expect(meters.some((m) => m.blockSize === 64) && meters.some((m) => m.blockSize === 256)).toBe(true);
  });

  it('port messages reach the engine; tap blocks are posted with their buffer transferred', () => {
    const proc = create();
    proc.port.deliver({ type: 'tap', enabled: true });
    drive(proc, 3 * TAP_BLOCK_FRAMES, 128, [sine(3 * TAP_BLOCK_FRAMES, 100, 0.1)], 2);
    const taps = proc.port.sent.filter((s) => s.msg.type === 'tap');
    expect(taps.length).toBe(3);
    for (const t of taps) {
      const m = t.msg as Extract<WorkletOutMsg, { type: 'tap' }>;
      expect(t.transfer).toEqual([m.block.buffer]);
    }
    const others = proc.port.sent.filter((s) => s.msg.type === 'meters');
    expect(others.every((s) => Array.isArray(s.transfer) && s.transfer.length === 0)).toBe(true);
  });

  it('an error thrown inside the engine → zeros + one fault, silent until un-muted', () => {
    const proc = create();
    proc.port.deliver({ type: 'params', params: testParams() });
    proc.port.deliver({ type: 'arm', armed: true });
    const n = 2 * FS;
    const x = sine(n, 150, 0.3);
    drive(proc, n, 128, [x], 2);
    const spy = vi.spyOn(AntiPhaseEngine.prototype, 'process').mockImplementation(function (
      this: AntiPhaseEngine,
      _a,
      _b,
      outL,
      outR,
    ) {
      outL.fill(0.9);
      outR.fill(-0.9);
      throw new Error('boom');
    });
    const outs = [new Float32Array(128).fill(0.5), new Float32Array(128).fill(0.5), new Float32Array(128).fill(0.5)];
    expect(proc.process([[x.slice(0, 128)]], [outs], {})).toBe(true);
    for (const o of outs) expect(o.every((v) => v === 0)).toBe(true);
    // repeated failures are reported once per second, not per quantum
    drive(proc, 4096, 128, [x], 2);
    const faults = proc.port.of('fault');
    expect(faults.length).toBe(1);
    expect(faults[0].message).toMatch(/boom/);
    spy.mockRestore();
    // the engine works again but the output stays muted until the main thread un-mutes
    const [L1] = drive(proc, FS, 128, [x], 2);
    expect(L1.every((v) => v === 0)).toBe(true);
    expect(proc.port.of('meters').at(-1)!.muted).toBe(true);
    proc.port.deliver({ type: 'mute', muted: false });
    const [L2] = drive(proc, 2 * FS, 128, [x], 2);
    // soft-start from zero, then full level
    expect(Math.abs(L2[64])).toBeLessThan(0.01);
    let late = 0;
    for (let i = FS + FS / 2; i < 2 * FS; i++) late = Math.max(late, Math.abs(L2[i]));
    expect(late).toBeGreaterThan(0.1);
  });

  it('an error in a message handler silences the output and is reported', () => {
    const proc = create();
    proc.port.deliver({ type: 'arm', armed: true });
    const spy = vi.spyOn(AntiPhaseEngine.prototype, 'handleMessage').mockImplementationOnce(() => {
      throw new Error('bad message');
    });
    proc.port.deliver({ type: 'bypass', bypass: true });
    spy.mockRestore();
    expect(proc.port.of('fault')[0].message).toMatch(/bad message/);
    const [L] = drive(proc, FS, 128, [sine(FS, 100, 0.5)], 2);
    expect(L.every((v) => v === 0)).toBe(true);
  });
});

describe('worklet bundle constraints', () => {
  /** Source without comments, so that prose mentioning `self` or `import.meta` does not count. */
  const code = (file: string): string =>
    readFileSync(file, 'utf8')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/(^|[^:'"`])\/\/.*$/gm, '$1');

  /** All modules reachable from the worklet through relative imports. */
  function graph(entry: string): string[] {
    const seen = new Set<string>();
    const stack = [entry];
    while (stack.length > 0) {
      const file = stack.pop()!;
      if (seen.has(file)) continue;
      seen.add(file);
      for (const m of code(file).matchAll(/(?:import|export)[^'"]*?from\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]/g)) {
        const spec = m[1] ?? m[2];
        expect(spec.startsWith('.'), `${relative(ROOT, file)} imports '${spec}'`).toBe(true);
        expect(spec.endsWith('.ts'), `${relative(ROOT, file)} imports '${spec}'`).toBe(true);
        stack.push(resolve(dirname(file), spec));
      }
    }
    return [...seen].map((f) => relative(ROOT, f).split('\\').join('/'));
  }

  it('the worklet only reaches src/dsp and src/audio/protocol.ts, without self/window/import.meta', () => {
    const files = graph(WORKLET_FILE);
    expect(files).toContain('src/dsp/antiphase-engine.ts');
    for (const f of files) {
      const allowed = f === 'src/worklet/antiphase.worklet.ts' || f === 'src/audio/protocol.ts' || f.startsWith('src/dsp/');
      expect(allowed, f).toBe(true);
      const src = code(resolve(ROOT, f));
      expect(/\bself\b/.test(src), `${f} uses self`).toBe(false);
      expect(/\bwindow\b/.test(src), `${f} uses window`).toBe(false);
      expect(/\bdocument\b/.test(src), `${f} uses document`).toBe(false);
      expect(/import\.meta/.test(src), `${f} uses import.meta`).toBe(false);
    }
  });
});
