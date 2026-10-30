/**
 * AudioWorklet adapter for AntiPhaseEngine: all DSP lives in the engine, this file only maps the
 * render quantum and the port onto it. AudioWorkletGlobalScope has no `self`/`window` and the module is
 * bundled as a worker chunk, so nothing here may use them or `import.meta`.
 */
import { PROCESSOR_NAME } from '../audio/protocol.ts';
import type { WorkletInMsg, WorkletOutMsg } from '../audio/protocol.ts';
import { AntiPhaseEngine } from '../dsp/antiphase-engine.ts';

const RENDER_QUANTUM = 128;
/** At most one 'fault' message per this many seconds of audio while the engine keeps failing. */
const FAULT_INTERVAL_S = 1;

const errorText = (err: unknown): string => (err instanceof Error ? err.message : String(err));

class AntiPhaseProcessor extends AudioWorkletProcessor {
  private engine: AntiPhaseEngine | null = null;
  /** Stand-ins for a missing output channel (mono output) or missing output. */
  private scratchL = new Float32Array(RENDER_QUANTUM);
  private scratchR = new Float32Array(RENDER_QUANTUM);
  /** After a fault the output stays at zero until the main thread un-mutes (the engine soft-starts then). */
  private silenced = false;
  private faultCooldown = 0;
  private readonly post = (msg: WorkletOutMsg, transfer?: ArrayBuffer[]): void => {
    this.port.postMessage(msg, transfer ?? []);
  };

  constructor() {
    super();
    try {
      this.engine = new AntiPhaseEngine(sampleRate);
    } catch (err) {
      this.onFault(err);
    }
    this.port.onmessage = (e: MessageEvent) => this.onMessage(e.data as WorkletInMsg);
    const ready: WorkletOutMsg = { type: 'ready', sampleRate };
    this.port.postMessage(ready);
  }

  private onMessage(msg: WorkletInMsg): void {
    try {
      if (this.engine === null) return;
      this.engine.handleMessage(msg);
      if (msg !== null && typeof msg === 'object' && msg.type === 'mute' && msg.muted === false) this.silenced = false;
    } catch (err) {
      this.onFault(err);
    }
  }

  private onFault(err: unknown): void {
    this.silenced = true;
    try {
      this.engine?.handleMessage({ type: 'panic' });
    } catch {
      // The engine is unusable: `silenced` alone keeps the output at zero.
    }
    if (this.faultCooldown > 0) return;
    this.faultCooldown = Math.max(1, Math.round(FAULT_INTERVAL_S * sampleRate));
    try {
      this.post({ type: 'fault', message: `Errore nel processore audio: ${errorText(err)}. Uscita silenziata.` });
    } catch {
      // Nothing else can be reported from here.
    }
  }

  private scratch(which: 0 | 1, n: number): Float32Array {
    let b = which === 0 ? this.scratchL : this.scratchR;
    if (b.length !== n) {
      b = new Float32Array(n);
      if (which === 0) this.scratchL = b;
      else this.scratchR = b;
    }
    return b;
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    try {
      const inp = inputs[0];
      const in0 = inp?.[0] ?? null;
      const in1 = inp?.[1] ?? null;
      const n = out?.[0]?.length ?? in0?.length ?? RENDER_QUANTUM;
      if (this.faultCooldown > 0) this.faultCooldown -= n;
      const outL = out?.[0] ?? this.scratch(0, n);
      const outR = out !== undefined && out.length > 1 ? out[1] : this.scratch(1, n);
      if (this.engine === null) {
        outL.fill(0);
        outR.fill(0);
      } else {
        this.engine.process(in0, in1, outL, outR);
        if (this.silenced) {
          outL.fill(0);
          outR.fill(0);
        }
      }
      if (out !== undefined) for (let c = 2; c < out.length; c++) out[c].fill(0);
      this.engine?.drain(this.post);
    } catch (err) {
      if (out !== undefined) for (let c = 0; c < out.length; c++) out[c].fill(0);
      this.onFault(err);
    }
    return true;
  }
}

registerProcessor(PROCESSOR_NAME, AntiPhaseProcessor);
