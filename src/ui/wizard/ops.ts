/**
 * Audio operations of the wizard, built only on the controller's public API: the looped
 * amplifier-level noise, band levels from the tap stream and the autotune driver.
 */
import type { AppController, ControllerState } from '../../app/controller.ts';
import { amplifierLevelNoise, bandRmsDbfs } from '../../audio/calibration.ts';
import type { AutotuneDriver } from '../../audio/autotune.ts';
import { TAP_CHANNELS } from '../../audio/protocol.ts';
import type { TapChannel } from '../../audio/protocol.ts';
import type { EngineParams } from '../../dsp/params.ts';
import { sleep } from './dom.ts';
import { appendTapChannel, fadeEdges, harmonicLevelDbfs } from './logic.ts';

/** Length of one noise chunk (each is one measure() call; ≤ the engine's MAX_MEASURE_S). */
export const NOISE_CHUNK_S = 10;
/** The amplifier-level noise stops by itself after this long. */
export const NOISE_MAX_S = 180;
const NOISE_FADE_MS = 30;
const SOFT_START_TIMEOUT_MS = 5000;
/** The engine also aborts a stimulus when the master changes while it plays. */
export const NOISE_INTERRUPTED_TEXT =
  'Il rumore di prova si è interrotto (uscita disabilitata, PANIC, volume master cambiato o misura rifiutata).';

/**
 * Plays band-limited pink noise on `ch` (through master and limiter: it is a measurement stimulus)
 * in chunks until `signal` aborts or NOISE_MAX_S elapse. `onPlayed` gets the seconds played.
 * Resolves with an Italian reason when the engine refused or aborted the stimulus, else null.
 */
export async function playLevelNoise(
  controller: AppController,
  ch: 0 | 1,
  o: { fs: number; band: [number, number]; rmsDbfs: number; signal: AbortSignal; onPlayed: (s: number) => void },
): Promise<string | null> {
  let played = 0;
  for (let k = 0; played < NOISE_MAX_S; k++) {
    if (o.signal.aborted) return null;
    const chunk = fadeEdges(amplifierLevelNoise(o.fs, o.band, NOISE_CHUNK_S, o.rmsDbfs, 1 + k), o.fs, NOISE_FADE_MS);
    const t0 = performance.now();
    const tick = setInterval(() => o.onPlayed(played + (performance.now() - t0) / 1000), 250);
    let r: Awaited<ReturnType<AppController['measure']>>;
    try {
      r = await controller.measure(chunk, ch, 0);
    } finally {
      clearInterval(tick);
    }
    if (o.signal.aborted) return null;
    if (r.aborted) return NOISE_INTERRUPTED_TEXT;
    played += NOISE_CHUNK_S;
    o.onPlayed(played);
  }
  return null;
}

/** Records `ms` of one tap channel (requires taps flowing: the microphone is open). */
export function recordTap(
  controller: AppController,
  channel: TapChannel,
  ms: number,
  fs: number,
  signal?: AbortSignal,
): Promise<Float32Array> {
  const idx = TAP_CHANNELS.indexOf(channel);
  const dst = new Float32Array(Math.max(1, Math.round((ms / 1000) * fs)));
  return new Promise<Float32Array>((resolve, reject) => {
    let filled = 0;
    const done = (err: Error | null): void => {
      off();
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      if (err) reject(err);
      else resolve(dst);
    };
    const off = controller.onTap((block, frames) => {
      filled = appendTapChannel(block, frames, idx, dst, filled);
      if (filled >= dst.length) done(null);
    });
    const timer = setTimeout(() => done(new Error('Nessun audio ricevuto dal microfono.')), ms + 3000);
    const onAbort = (): void => done(new Error('Interrotto.'));
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Resolves when `cond(state)` holds; rejects after `timeoutMs` or on abort. */
export function waitForState(
  controller: AppController,
  cond: (s: ControllerState) => boolean,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (cond(controller.state)) {
      resolve();
      return;
    }
    const finish = (err: Error | null): void => {
      off();
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      if (err) reject(err);
      else resolve();
    };
    const off = controller.subscribe((s) => {
      if (cond(s)) finish(null);
    });
    const timer = setTimeout(() => finish(new Error('Tempo scaduto.')), timeoutMs);
    const onAbort = (): void => finish(new Error('Interrotto.'));
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Waits for the 1.5 s soft-start of a just un-muted anti-noise. */
export const waitSoftStart = (controller: AppController, signal?: AbortSignal): Promise<void> =>
  waitForState(controller, (s) => s.ancOn && (s.meters?.softStart ?? 0) >= 0.999, SOFT_START_TIMEOUT_MS, signal);

/**
 * Error level of one autotune window: in tonal mode synchronous detection at the harmonics of f0
 * (the anti-noise only acts there; the broadband level would be steered by the noise around the
 * tone), otherwise the band RMS. Bypass windows use the same estimator.
 */
export function errorLevelDbfs(x: Float32Array, fs: number, band: [number, number], p: EngineParams): number {
  if (p.phaseMode === 'tonal') {
    const v = harmonicLevelDbfs(x, fs, p.tonal.f0, p.tonal.harmonics, band);
    if (v !== null) return v;
  }
  return bandRmsDbfs(x, fs, band);
}

/**
 * Autotune driver on the controller: parameters through applyParams, error level from the
 * 'err' tap in the band (errorLevelDbfs), bypass windows with the A/B bypass (15 ms ramp) plus
 * a settling pause.
 */
export function autotuneDriver(
  controller: AppController,
  o: { fs: number; band: [number, number]; signal: AbortSignal; onMeasure: () => void },
): AutotuneDriver {
  const level = async (ms: number): Promise<number> => {
    const x = await recordTap(controller, 'err', ms, o.fs, o.signal);
    o.onMeasure();
    return errorLevelDbfs(x, o.fs, o.band, controller.state.params);
  };
  return {
    signal: o.signal,
    apply(p: EngineParams) {
      controller.applyParams(p);
    },
    settle: (ms) => sleep(ms, o.signal),
    measureErrorDb: (ms) => level(ms),
    async measureBypassDb(ms) {
      controller.setBypass(true);
      try {
        // Mute ramp, delay line and the room's decay before the window counts as "off".
        await sleep(400, o.signal);
        return await level(ms);
      } finally {
        controller.setBypass(false);
      }
    },
  };
}
