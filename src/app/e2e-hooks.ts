/**
 * Test hooks for the e2e build only (`vite build --mode e2e`): main.ts imports this module behind
 * `import.meta.env.MODE === 'e2e'`, so production bundles contain neither the module nor the global.
 *
 * window.__anc = {
 *   controller, engine,
 *   lastTap(blocks = 1): the last `blocks` tap blocks (≤ 16) concatenated per channel, as plain arrays,
 *   events: engine events (no meters/taps/stats/progress) and controller events, with timestamps,
 *   allowOutputLatency(): overrides the output-latency validation error,
 * }
 */
import type { AudioEngine, AudioEngineEvent } from '../audio/audio-engine.ts';
import { TAP_CHANNELS } from '../audio/protocol.ts';
import type { AppController } from './controller.ts';
import { ISSUE } from './validation.ts';

const MAX_BLOCKS = 16;
const MAX_EVENTS = 1000;

export interface E2eTap {
  /** Number of tap blocks received so far (increases by one per block). */
  seq: number;
  frames: number;
  sampleRate: number | null;
  ref: number[];
  cond: number[];
  outL: number[];
  outR: number[];
  err: number[];
}

export interface E2eEvent {
  t: number;
  source: 'engine' | 'app';
  type: string;
  detail?: unknown;
}

export interface E2eHooks {
  controller: AppController;
  engine: AudioEngine;
  lastTap(blocks?: number): E2eTap | null;
  events: E2eEvent[];
  allowOutputLatency(): void;
}

function summarize(e: AudioEngineEvent): unknown {
  switch (e.type) {
    case 'measureResult':
      return { id: e.id, aborted: e.aborted, frames: e.ref.length, err: e.err !== null, masterLin: e.masterLin };
    case 'howl':
      return { reason: e.reason, freqHz: e.freqHz };
    case 'autoMute':
      return e.reason;
    case 'fault':
      return e.message;
    case 'channelCheck':
      return e.status;
    case 'lifecycle':
      return { cause: e.cause ?? null, reason: e.reason, autoMuted: e.autoMuted };
    case 'ready':
      return e.sampleRate;
    default:
      return undefined;
  }
}

export function installE2eHooks(controller: AppController, engine: AudioEngine): void {
  const blocks: Array<{ data: Float32Array; frames: number }> = [];
  let seq = 0;
  controller.onTap((block, frames) => {
    blocks.push({ data: block.slice(0, TAP_CHANNELS.length * frames), frames });
    if (blocks.length > MAX_BLOCKS) blocks.shift();
    seq++;
  });

  const events: E2eEvent[] = [];
  const push = (ev: E2eEvent): void => {
    events.push(ev);
    if (events.length > MAX_EVENTS) events.splice(0, events.length - MAX_EVENTS);
  };
  engine.on((e) => {
    if (e.type === 'meters' || e.type === 'tap' || e.type === 'stats' || e.type === 'measureProgress') return;
    push({ t: performance.now(), source: 'engine', type: e.type, detail: summarize(e) });
  });
  controller.onEvent((e) => push({ t: performance.now(), source: 'app', type: e.type, detail: e.detail }));

  const lastTap = (n = 1): E2eTap | null => {
    const count = Math.max(1, Math.min(MAX_BLOCKS, Math.floor(n)));
    const sel = blocks.slice(-count);
    if (sel.length === 0) return null;
    const total = sel.reduce((a, b) => a + b.frames, 0);
    const channel = (k: number): number[] => {
      const out = new Array<number>(total);
      let o = 0;
      for (const b of sel) {
        for (let i = 0; i < b.frames; i++) out[o++] = b.data[k * b.frames + i] as number;
      }
      return out;
    };
    return {
      seq,
      frames: total,
      sampleRate: controller.sampleRate,
      ref: channel(TAP_CHANNELS.indexOf('ref')),
      cond: channel(TAP_CHANNELS.indexOf('cond')),
      outL: channel(TAP_CHANNELS.indexOf('outL')),
      outR: channel(TAP_CHANNELS.indexOf('outR')),
      err: channel(TAP_CHANNELS.indexOf('err')),
    };
  };

  const hooks: E2eHooks = {
    controller,
    engine,
    lastTap,
    events,
    allowOutputLatency: () => controller.setOverride(ISSUE.outputLatency, true),
  };
  (window as unknown as { __anc?: E2eHooks }).__anc = hooks;
}
