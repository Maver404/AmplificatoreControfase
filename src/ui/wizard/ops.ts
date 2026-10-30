/**
 * Audio operations of the wizard, built only on the controller's public API: the looped
 * amplifier-level noise, band levels from the tap stream and the autotune driver.
 */
import type { AppController, ControllerState } from '../../app/controller.ts';
import { harmonicLevelsDb, harmonicTolerancesHz } from '../../audio/autotune.ts';
import type { AutotuneDriver, AutotuneResult } from '../../audio/autotune.ts';
import { amplifierLevelNoise, bandRmsDbfs, capLoopGain } from '../../audio/calibration.ts';
import type { CalibrationData } from '../../audio/calibration.ts';
import { TAP_CHANNELS } from '../../audio/protocol.ts';
import type { TapChannel } from '../../audio/protocol.ts';
import type { EngineParams } from '../../dsp/params.ts';
import { formatNumber } from '../units.ts';
import { sleep } from './dom.ts';
import { appendTapChannel, calibratedMeasurements, fadeEdges, harmonicLevelDbfs } from './logic.ts';

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
 * `p`'s channels (all an autotune changes: delay, Δ, gain, trims) over the controller's parameters as they
 * are now. The rest of `p` is what the run copied at its start: applied whole, every probe set tonal.f0 back
 * to that value and the live follower moved it again within 2 s, so with a drifting source f0 swung by the
 * whole run's drift in every window (and the trims were measured half at one f0, half at the other).
 */
export function liveWithChannels(controller: AppController, p: EngineParams): EngineParams {
  return { ...controller.currentParams, channels: [p.channels[0], p.channels[1]] };
}

/**
 * Step 6's autotune result, applied as the calibrated values when it improved on its start (`improved`; an
 * aborted or unconfirmed run left the start in place): only its channels, over the live params (liveWithChannels —
 * its own params carry the f0 and the master of the run's start, which the follower or the user moved since: f0
 * 4 % back, out of the follower's reach, and a master 15 dB above the user's), within the solve's stability
 * margin (capLoopGain), with its per-harmonic measurement (calibratedMeasurements: one zone only). Returns
 * whether it applied.
 */
export function applyAutotuneResult(
  controller: AppController,
  cal: CalibrationData,
  r: Pick<AutotuneResult, 'params' | 'aborted' | 'improved' | 'harmonics'>,
  o: { fs: number; taps: number; separateZones: boolean },
): boolean {
  if (r.aborted || !r.improved) return false;
  const capped = capLoopGain(cal, liveWithChannels(controller, r.params), o.fs, o.taps);
  const measured = o.separateZones ? null : calibratedMeasurements(r, capped.limitedDb);
  controller.applyParams(capped.params, { calibrated: true, harmonics: measured });
  return true;
}

/**
 * Autotune driver on the controller: the probes' channels through applyParams (liveWithChannels: f0
 * keeps following the source), error level from the 'err' tap in the band (errorLevelDbfs), bypass
 * windows with the A/B bypass (15 ms ramp) plus a settling pause. In tonal mode it also reads each
 * harmonic on its own (measureHarmonicsDb, synchronous detection), so the search refines the
 * per-harmonic trims. Create it right before runAutotune: the f0 it finds then is the run's start f0.
 *
 * The search asks for its harmonics at the start f0, while the bank follows the live one: each window reads
 * them where the bank is, freqs and tolerances × live f0 / start f0 (the live f0 halfway through the window).
 * Before that a +2 % step of the source left every later window reading the noise floor 40–50 dB under the
 * tone, and the run completed blind with a false improvement. A live f0 further from the start than the
 * search's drift tolerance (harmonicTolerancesHz: the trims' optimum moves with f0 too) stops the run, which
 * then restores the start values (over the live f0).
 */
export function autotuneDriver(
  controller: AppController,
  o: { fs: number; band: [number, number]; signal: AbortSignal; onMeasure: () => void },
): AutotuneDriver {
  const f0Run = controller.state.params.tonal.f0;
  /** Live f0 / the run's start f0, refused beyond the search's tolerance. */
  const f0Ratio = (): number => {
    const p = controller.state.params;
    const k = p.tonal.f0 / f0Run;
    if (!(Math.abs(k - 1) <= harmonicTolerancesHz(p, [1])[0])) {
      throw new Error(
        `la fondamentale si è spostata da ${formatNumber(f0Run, 2)} a ${formatNumber(p.tonal.f0, 2)} Hz durante la misura, ` +
          'più di quanto le correzioni per armonica tollerano',
      );
    }
    return k;
  };
  const level = async (ms: number): Promise<number> => {
    const x = await recordTap(controller, 'err', ms, o.fs, o.signal);
    o.onMeasure();
    return errorLevelDbfs(x, o.fs, o.band, controller.state.params);
  };
  /** Runs `measure` with the anti-noise bypassed (mute ramp, delay line and the room's decay first). */
  const bypassed = async <T>(measure: () => Promise<T>): Promise<T> => {
    controller.setBypass(true);
    try {
      await sleep(400, o.signal);
      return await measure();
    } finally {
      controller.setBypass(false);
    }
  };
  return {
    signal: o.signal,
    apply(p: EngineParams) {
      controller.applyParams(liveWithChannels(controller, p));
    },
    settle: (ms) => sleep(ms, o.signal),
    measureErrorDb: (ms) => level(ms),
    measureBypassDb: (ms) => bypassed(() => level(ms)),
    measureHarmonicsDb(ms, freqs, bypass, tolHz) {
      const levels = async (): Promise<number[]> => {
        const before = f0Ratio();
        const x = await recordTap(controller, 'err', ms, o.fs, o.signal);
        o.onMeasure();
        const k = (before + f0Ratio()) / 2;
        return harmonicLevelsDb(x, o.fs, freqs.map((f) => f * k), tolHz?.map((t) => t * k));
      };
      return bypass ? bypassed(levels) : levels();
    },
  };
}
