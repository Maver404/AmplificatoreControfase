/**
 * Analyzer implementations: in-process (tests, simulator, fallback) and the client of the analysis
 * worker (browser), which runs the same pure functions off the main thread. Also the live f0 window
 * whose periodic estimate the controller runs through the worker.
 */
import { estimateF0 } from '../dsp/f0-tracker.ts';
import type { F0Estimate } from '../dsp/f0-tracker.ts';
import { estimateDelay } from '../dsp/latency-estimator.ts';
import type { DelayEstimate } from '../dsp/latency-estimator.ts';
import { welchTransfer } from './calibration.ts';
import type { Analyzer, TransferEstimate } from './calibration.ts';

/** Requests to src/workers/analysis.worker.ts (input buffers are transferred copies). */
export type AnalysisRequest =
  | {
      id: number;
      op: 'estimateDelay';
      recorded: Float32Array;
      stimulus: Float32Array;
      fs: number;
      f1: number;
      f2: number;
      mode: 'acoustic' | 'loopback';
    }
  | { id: number; op: 'transfer'; x: Float32Array; y: Float32Array; fs: number; freqs: Float64Array }
  | { id: number; op: 'estimateF0'; x: Float32Array; fs: number; fmin: number; fmax: number };

/** Responses from the worker (result buffers are transferred). */
export type AnalysisResponse =
  | { id: number; ok: true; op: 'estimateDelay'; result: DelayEstimate & { ir: Float64Array } }
  | { id: number; ok: true; op: 'transfer'; result: Required<TransferEstimate> }
  | { id: number; ok: true; op: 'estimateF0'; result: F0Estimate | null }
  | { id: number; ok: false; error: string };

/** Calls the pure analysis functions directly on the calling thread. */
export const inProcessAnalyzer: Analyzer = {
  estimateDelay: async (recorded, stimulus, fs, f1, f2, mode) => estimateDelay(recorded, stimulus, fs, f1, f2, mode),
  transfer: async (x, y, fs, freqs) => welchTransfer(x, y, fs, freqs),
  estimateF0: async (x, fs, fmin, fmax) => estimateF0(x, fs, fmin, fmax),
};

export interface WorkerAnalyzer extends Analyzer {
  /** False when the worker could not be started or has failed: requests then run in-process. */
  readonly usingWorker: boolean;
  /** Stops the worker; later requests run in-process. */
  terminate(): void;
}

interface Pending {
  resolve: (value: never) => void;
  reject: (err: Error) => void;
  /** Same computation in-process, used if the worker dies before answering. */
  fallback: () => Promise<unknown>;
}

/**
 * Analyzer backed by the analysis worker (same-origin module worker bundled by Vite, so it works
 * under a strict CSP). Inputs are copied and the copies transferred, so the caller's buffers stay
 * usable (measurePath reuses its stimulus); results come back as transferred buffers. If the worker
 * cannot start or fails, pending and later requests run in-process instead.
 */
export function createAnalyzer(): WorkerAnalyzer {
  let worker: Worker | null = null;
  try {
    if (typeof Worker === 'function') {
      worker = new Worker(new URL('../workers/analysis.worker.ts', import.meta.url), { type: 'module' });
    }
  } catch {
    worker = null;
  }
  const pending = new Map<number, Pending>();
  let nextId = 1;

  const fail = (): void => {
    if (worker !== null) worker.terminate();
    worker = null;
    const list = [...pending.values()];
    pending.clear();
    for (const p of list) p.fallback().then(p.resolve as (v: unknown) => void, p.reject);
  };

  if (worker !== null) {
    worker.onmessage = (e: MessageEvent<AnalysisResponse>) => {
      const res = e.data;
      const p = pending.get(res.id);
      if (p === undefined) return;
      pending.delete(res.id);
      if (res.ok) (p.resolve as (v: unknown) => void)(res.result);
      else p.reject(new Error(`Analisi non riuscita: ${res.error}`));
    };
    worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      fail();
    };
    worker.onmessageerror = () => fail();
  }

  function call<T>(make: (id: number) => { req: AnalysisRequest; transfer: Transferable[] }, fallback: () => Promise<T>): Promise<T> {
    const w = worker;
    if (w === null) return fallback();
    const id = nextId++;
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve: resolve as (v: never) => void, reject, fallback });
      try {
        const { req, transfer } = make(id);
        w.postMessage(req, transfer);
      } catch {
        pending.delete(id);
        fallback().then(resolve, reject);
      }
    });
  }

  return {
    get usingWorker() {
      return worker !== null;
    },
    terminate() {
      fail();
    },
    estimateDelay(recorded, stimulus, fs, f1, f2, mode) {
      return call(
        (id) => {
          const r = recorded.slice();
          const s = stimulus.slice();
          return { req: { id, op: 'estimateDelay', recorded: r, stimulus: s, fs, f1, f2, mode }, transfer: [r.buffer, s.buffer] };
        },
        () => inProcessAnalyzer.estimateDelay(recorded, stimulus, fs, f1, f2, mode),
      );
    },
    transfer(x, y, fs, freqs) {
      return call(
        (id) => {
          const a = x.slice();
          const b = y.slice();
          const f = freqs.slice();
          return { req: { id, op: 'transfer', x: a, y: b, fs, freqs: f }, transfer: [a.buffer, b.buffer, f.buffer] };
        },
        () => inProcessAnalyzer.transfer(x, y, fs, freqs),
      );
    },
    estimateF0(x, fs, fmin, fmax) {
      return call(
        (id) => {
          const a = x.slice();
          return { req: { id, op: 'estimateF0', x: a, fs, fmin, fmax }, transfer: [a.buffer] };
        },
        () => inProcessAnalyzer.estimateF0(x, fs, fmin, fmax),
      );
    },
  };
}

/** What the live f0 estimate needs from an Analyzer. */
export type F0Analyzer = Pick<Analyzer, 'estimateF0'>;

/**
 * The last `windowS` seconds of a live stream (the 'cond' tap) for the periodic f0 estimate, which
 * runs through an Analyzer: in the browser the analysis worker (≈ 46 ms per 2 s window that the
 * main thread no longer spends), in-process when workers are unavailable or the worker fails.
 * Same window length, samples and search range as F0Tracker, so the estimate is identical: the
 * blocks are float32, so the float32 window holds exactly the values F0Tracker's float64 ring does.
 */
export class F0Window {
  private readonly fs: number;
  private readonly fmin: number;
  private readonly fmax: number;
  private readonly ring: Float32Array;
  private pos = 0;
  private filled = 0;

  constructor(fs: number, windowS = 2, fmin = 20, fmax = 2000) {
    if (!(fs > 0)) throw new RangeError('F0Window: fs must be positive');
    this.fs = fs;
    this.fmin = fmin;
    this.fmax = fmax;
    this.ring = new Float32Array(Math.max(256, Math.round(Math.max(0.05, windowS) * fs)));
  }

  /** Appends a block (allocation-free; non-finite samples count as 0). */
  push(block: Float32Array): void {
    const n = this.ring.length;
    for (let i = 0; i < block.length; i++) {
      const v = block[i];
      this.ring[this.pos] = Number.isFinite(v) ? v : 0;
      this.pos = this.pos + 1 === n ? 0 : this.pos + 1;
    }
    this.filled = Math.min(n, this.filled + block.length);
  }

  /** The window in time order, as a new array (later pushes do not change it). */
  snapshot(): Float32Array {
    const n = this.ring.length;
    const m = this.filled;
    const out = new Float32Array(m);
    const start = (this.pos - m + n) % n;
    const head = Math.min(m, n - start);
    out.set(this.ring.subarray(start, start + head));
    if (head < m) out.set(this.ring.subarray(0, m - head), head);
    return out;
  }

  /** f0 of the current window (taken now) through `analyzer`. */
  estimate(analyzer: F0Analyzer): Promise<F0Estimate | null> {
    return analyzer.estimateF0(this.snapshot(), this.fs, this.fmin, this.fmax);
  }

  reset(): void {
    this.ring.fill(0);
    this.pos = 0;
    this.filled = 0;
  }
}
