/**
 * Messages between the main thread and the 'antiphase' AudioWorkletProcessor.
 * Types only (plus a few constants), so it compiles under the app, worklet and worker tsconfigs.
 */
import type { EngineParams } from '../dsp/params.ts';

export const PROCESSOR_NAME = 'antiphase';

/** Channel order inside a tap block (planar: channel k occupies frames [k·N, (k+1)·N)). */
export const TAP_CHANNELS = ['ref', 'cond', 'outL', 'outR', 'err'] as const;
export type TapChannel = (typeof TAP_CHANNELS)[number];
export const TAP_BLOCK_FRAMES = 2048;

/**
 * Check of the error-mic input (input `errorChannel + 1`; input 2 while no error mic is set): 'absent' = mono
 * track, 'silent' = that input is silent, 'duplicate' = both inputs carry the same signal.
 */
export type Ch2Status = 'pending' | 'ok' | 'absent' | 'silent' | 'duplicate';

export type HowlReason = 'peak-growth' | 'limiter-saturated' | 'output-overload' | 'anc-worse';

export interface LevelReading {
  rmsDb: number;
  peakDb: number;
}

export interface MeterSnapshot {
  ref: LevelReading;
  err: LevelReading | null;
  outL: LevelReading;
  outR: LevelReading;
  /** Current limiter gain reduction per channel, dB (≥ 0). */
  limiterGrDb: [number, number];
  armed: boolean;
  muted: boolean;
  bypass: boolean;
  /** Soft-start envelope 0..1. */
  softStart: number;
  delaySamples: [number, number];
  delayClamped: [boolean, boolean];
  blockSize: number;
  ch2: Ch2Status;
  measuring: boolean;
  /**
   * Master gain applied right now, dB: the engine's master ramp after smoothing, rise limiting and the
   * stimulus override (it jumps to the target when a calibration stimulus starts), i.e. what a stimulus sample
   * is multiplied by now; MASTER_MIN_DB (−120) when the master is silent.
   */
  masterDb: number;
}

export type WorkletInMsg =
  | { type: 'params'; params: EngineParams }
  /** Validation gate: while false the processor outputs exact zeros. */
  | { type: 'arm'; armed: boolean }
  | { type: 'mute'; muted: boolean }
  | { type: 'bypass'; bypass: boolean }
  | { type: 'panic' }
  | { type: 'softStart' }
  | {
      type: 'measure';
      id: number;
      /** Played on `channel`, injected before master volume and limiter; anti-noise is muted meanwhile. */
      stimulus: Float32Array;
      channel: 0 | 1;
      postRollFrames: number;
    }
  | { type: 'abortMeasure' }
  /** Feedback neutralisation FIR (final output of `channel` → reference mic); null disables it. */
  | { type: 'feedbackFir'; channel: 0 | 1; bulkDelay: number; taps: Float64Array | null }
  | { type: 'tap'; enabled: boolean }
  /** Return a tap block buffer to the processor's pool. */
  | { type: 'recycle'; buffer: Float32Array }
  | { type: 'resetHowl' };

export type WorkletOutMsg =
  | { type: 'ready'; sampleRate: number }
  | ({ type: 'meters' } & MeterSnapshot)
  | { type: 'tap'; block: Float32Array; frames: number }
  | { type: 'measureProgress'; id: number; fraction: number }
  | {
      type: 'measureResult';
      id: number;
      /** Sample-aligned recordings of the reference and error inputs (error null if absent). */
      ref: Float32Array;
      err: Float32Array | null;
      /**
       * Linear master gain applied to every stimulus sample: the master is fixed when the stimulus starts and a
       * master change while it plays aborts the measurement.
       */
      masterLin: number;
      aborted: boolean;
    }
  | { type: 'howl'; reason: HowlReason; freqHz: number | null }
  | { type: 'autoMute'; reason: string }
  | { type: 'channelCheck'; status: Ch2Status }
  | { type: 'fault'; message: string };
