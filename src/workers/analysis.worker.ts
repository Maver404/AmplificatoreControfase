/**
 * Analysis worker: sweep deconvolution / delay estimation, Welch transfer functions and f0 estimation
 * off the main thread. It only dispatches to the pure functions; the protocol lives in
 * src/audio/analysis-client.ts. Inputs arrive as transferred copies, results leave as transferred buffers.
 */
import type { AnalysisRequest, AnalysisResponse } from '../audio/analysis-client.ts';
import { welchTransfer } from '../audio/calibration.ts';
import { estimateF0 } from '../dsp/f0-tracker.ts';
import { estimateDelay } from '../dsp/latency-estimator.ts';

/** The parts of the dedicated worker scope used here. */
interface Scope {
  onmessage: ((e: MessageEvent<AnalysisRequest>) => void) | null;
  postMessage(msg: AnalysisResponse, transfer: Transferable[]): void;
}
const scope = self as unknown as Scope;

function handle(req: AnalysisRequest): { res: AnalysisResponse; transfer: Transferable[] } {
  switch (req.op) {
    case 'estimateDelay': {
      const result = estimateDelay(req.recorded, req.stimulus, req.fs, req.f1, req.f2, req.mode);
      return { res: { id: req.id, ok: true, op: req.op, result }, transfer: [result.ir.buffer] };
    }
    case 'transfer': {
      const result = welchTransfer(req.x, req.y, req.fs, req.freqs);
      return {
        res: { id: req.id, ok: true, op: req.op, result },
        transfer: [result.re.buffer, result.im.buffer, result.coherence.buffer, result.pxx.buffer],
      };
    }
    case 'estimateF0': {
      const result = estimateF0(req.x, req.fs, req.fmin, req.fmax);
      return { res: { id: req.id, ok: true, op: req.op, result }, transfer: [] };
    }
    default:
      throw new Error('richiesta sconosciuta');
  }
}

scope.onmessage = (e) => {
  const req = e.data;
  const id = typeof req === 'object' && req !== null && typeof req.id === 'number' ? req.id : -1;
  try {
    const { res, transfer } = handle(req);
    scope.postMessage(res, transfer);
  } catch (err) {
    scope.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) }, []);
  }
};
