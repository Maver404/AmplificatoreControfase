/**
 * Browser side of the audio graph:
 *   getUserMedia(interface) → MediaStreamAudioSourceNode → AudioWorkletNode 'antiphase' → destination
 * plus the lifecycle safety net. All DSP and every safety envelope (arm gate, mute, soft-start, panic,
 * limiter, howl detector) live in the worklet; this class builds the graph, keeps a shadow copy of
 * the control state so a re-created processor resumes exactly where it was (always muted), and mutes
 * the output whenever the platform does something that invalidates the alignment. Every such
 * auto-mute also aborts a running measurement (its stimulus is played after the mute envelope).
 */
// The `?worker&url` import is Vite's; the reference keeps it typed in any project that includes this
// file (the tests' tsconfig reaches it through type imports).
/// <reference types="vite/client" />
import workletUrl from '../worklet/antiphase.worklet.ts?worker&url';
import { onDeviceChange, supportsSetSinkId } from './devices.ts';
import { PROCESSOR_NAME } from './protocol.ts';
import type { WorkletInMsg, WorkletOutMsg } from './protocol.ts';
import { PANIC_RAMP_MS } from '../dsp/params.ts';
import type { EngineParams } from '../dsp/params.ts';

export type LatencyMode = 'normal' | 'ultra';

/** Why a lifecycle event was emitted (extension of the contract's event, for the controller). */
export type LifecycleCause =
  | 'context-state'
  | 'context-error'
  | 'devicechange'
  | 'sinkchange'
  | 'track-ended'
  | 'track-muted'
  | 'processorerror'
  | 'underruns'
  | 'latency-jump'
  | 'context-recreated';

export type AudioEngineEvent =
  | WorkletOutMsg
  | { type: 'lifecycle'; reason: string; autoMuted: boolean; cause?: LifecycleCause }
  | { type: 'stats'; stats: AudioStats };

export interface AudioStats {
  sampleRate: number;
  renderQuantum: number;
  baseLatencyMs: number | null;
  outputLatencyMs: number | null;
  trackLatencyMs: number | null;
  underruns: number | null;
  trackSettings: MediaTrackSettings | null;
  sinkId: string | null;
}

export interface MeasureResult {
  ref: Float32Array;
  err: Float32Array | null;
  masterLin: number;
  aborted: boolean;
}

export interface TrackInfo {
  deviceId: string | null;
  groupId: string | null;
  label: string;
  settings: MediaTrackSettings;
  /** Channels actually delivered (getSettings().channelCount), null if unknown. */
  channelCount: number | null;
}

export type AudioEngineState = 'idle' | 'mic' | 'error';

export type MicErrorCode =
  | 'unsupported'
  | 'insecure'
  | 'denied'
  | 'not-found'
  | 'busy'
  | 'overconstrained'
  | 'worklet'
  | 'aborted'
  | 'unknown';

/** Rejection of openMic(): `message` is Italian and meant for the user. */
export class AudioEngineError extends Error {
  readonly code: MicErrorCode;
  constructor(code: MicErrorCode, message: string) {
    super(message);
    this.name = 'AudioEngineError';
    this.code = code;
  }
}

/** Members Chrome has and the bundled DOM typings do not (yet). */
interface ChromeAudioContext extends AudioContext {
  readonly sinkId?: string | { type: string };
  setSinkId?(sinkId: string): Promise<void>;
  readonly renderQuantumSize?: number;
  readonly playbackStats?: { readonly underrunEvents?: number };
}

const STATS_INTERVAL_MS = 500;
const UNDERRUN_WINDOW_MS = 10_000;
/** More than this many underruns inside the window mutes the output; fewer are only counted. */
const UNDERRUN_LIMIT = 3;
/** An output-latency change larger than this invalidates the calibrated delay. */
export const LATENCY_JUMP_MS = 5;
/** Latency readings settle after a context or sink change; jumps are ignored meanwhile. */
const LATENCY_SETTLE_MS = 2000;
/** Panic ramp plus a few render quanta and message latency, before suspending the context. */
const PANIC_SUSPEND_DELAY_MS = PANIC_RAMP_MS + 40;
const MEASURE_TIMEOUT_MARGIN_MS = 5000;

const MSG = {
  unsupported: 'Questo browser non supporta Web Audio o AudioWorklet: usa una versione recente di Google Chrome.',
  insecure:
    "Il microfono è disponibile solo su una connessione sicura: apri l'app da https:// oppure da http://localhost.",
  denied:
    "Accesso al microfono negato: consentilo dalle impostazioni del sito (icona accanto all'indirizzo) e dalle impostazioni privacy del sistema, poi riprova.",
  notFound:
    "Nessun microfono trovato: collega l'interfaccia audio, controlla che il sistema la veda e riprova.",
  busy: "Il microfono è occupato da un'altra applicazione o non risponde: chiudi le altre app audio e riprova.",
  overconstrained: 'Il dispositivo selezionato non è più disponibile: scegline un altro.',
  worklet: 'Impossibile avviare il processore audio (AudioWorklet): ricarica la pagina.',
  aborted: 'Apertura del microfono annullata.',
  unknown: 'Impossibile aprire il microfono.',
  suspended: "Il sistema ha sospeso l'audio del browser: uscita silenziata.",
  interrupted:
    "L'audio è stato interrotto dal sistema (un'altra applicazione ha preso il dispositivo?): uscita silenziata.",
  closed: 'Il motore audio è stato chiuso: uscita silenziata. Ricarica la pagina.',
  ctxError: "Errore del dispositivo audio: uscita silenziata. Controlla l'interfaccia e riapri il microfono.",
  devicechange: 'I dispositivi audio sono cambiati: uscita silenziata. Controlla la selezione e la validazione.',
  sinkchange: "Il dispositivo d'uscita è cambiato: uscita silenziata.",
  trackEnded: 'Il microfono si è scollegato o è stato chiuso: uscita silenziata. Riapri il microfono.',
  trackMuted: 'Il microfono non invia più audio (silenziato dal sistema): uscita silenziata.',
  processorError: 'Il processore audio si è bloccato ed è stato riavviato: uscita silenziata.',
  underruns: `Troppe interruzioni del flusso audio (più di ${UNDERRUN_LIMIT} underrun in ${UNDERRUN_WINDOW_MS / 1000} s): uscita silenziata. Chiudi le altre app o usa la latenza Normale.`,
  recreated: 'Il motore audio è stato riavviato con le nuove impostazioni: uscita silenziata.',
} as const;

const EMPTY_STATS: AudioStats = {
  sampleRate: 0,
  renderQuantum: 0,
  baseLatencyMs: null,
  outputLatencyMs: null,
  trackLatencyMs: null,
  underruns: null,
  trackSettings: null,
  sinkId: null,
};

const toMs = (s: unknown): number | null => (typeof s === 'number' && Number.isFinite(s) && s >= 0 ? s * 1000 : null);

function errorName(err: unknown): string {
  return err !== null && typeof err === 'object' && typeof (err as { name?: unknown }).name === 'string'
    ? (err as { name: string }).name
    : '';
}

function micError(err: unknown): AudioEngineError {
  if (err instanceof AudioEngineError) return err;
  switch (errorName(err)) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return new AudioEngineError('denied', MSG.denied);
    case 'SecurityError':
      return new AudioEngineError('insecure', MSG.insecure);
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return new AudioEngineError('not-found', MSG.notFound);
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return new AudioEngineError('busy', MSG.busy);
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return new AudioEngineError('overconstrained', MSG.overconstrained);
    default: {
      const detail = err instanceof Error && err.message ? ` (${err.message})` : '';
      return new AudioEngineError('unknown', `${MSG.unknown}${detail}`);
    }
  }
}

/** Exactly the plan's constraints: every browser processing off, lowest latency. */
function micConstraints(deviceId: string | null, stereo: boolean): MediaTrackConstraints {
  const c: Record<string, unknown> = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    voiceIsolation: false,
    latency: 0,
  };
  if (deviceId !== null && deviceId !== '') c['deviceId'] = { exact: deviceId };
  if (stereo) c['channelCount'] = { ideal: 2 };
  return c as MediaTrackConstraints;
}

const stopStream = (s: MediaStream | null): void => {
  if (s) for (const t of s.getTracks()) t.stop();
};

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

interface Shadow {
  params: EngineParams | null;
  armed: boolean;
  muted: boolean;
  bypass: boolean;
  tap: boolean;
  firs: [{ bulkDelay: number; taps: Float64Array } | null, { bulkDelay: number; taps: Float64Array } | null];
  /** Sample rate of the context each FIR was posted to (null: none yet); FIRs are sample-indexed. */
  firFs: [number | null, number | null];
}

interface PendingMeasure {
  resolve: (r: MeasureResult) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class AudioEngine {
  private latencyMode: LatencyMode;
  private ctx: ChromeAudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private track: MediaStreamTrack | null = null;
  private readonly modules = new WeakSet<BaseAudioContext>();
  private readonly listeners = new Set<(e: AudioEngineEvent) => void>();
  private _state: AudioEngineState = 'idle';
  private _error: string | null = null;
  /** Requested output device ('' / 'default' = system default), null = never chosen. */
  private sinkId: string | null = null;
  private readonly shadow: Shadow = {
    params: null,
    armed: false,
    muted: false,
    bypass: false,
    tap: false,
    firs: [null, null],
    firFs: [null, null],
  };
  private openSeq = 0;
  private measureSeq = 0;
  private readonly pending = new Map<number, PendingMeasure>();
  private _stats: AudioStats = { ...EMPTY_STATS };
  private statsTimer: ReturnType<typeof setInterval> | null = null;
  private panicTimer: ReturnType<typeof setTimeout> | null = null;
  private expectSuspend = false;
  private expectSinkChangeUntil = 0;
  /** Our own setSinkId is switching the device: Chrome suspends the context meanwhile (not the system). */
  private sinkSwitching = false;
  /** PANIC holds the engine suspended (also a context created meanwhile) until resume(). */
  private panicked = false;
  private closing = false;
  private lastBlockSize = 0;
  private lastUnderrunCount: number | null = null;
  private underrunTimes: number[] = [];
  /** Output latency the alignment is valid for: the last one heard (or measured). */
  private latencyRef: number | null = null;
  /** Latest settled output-latency reading. */
  private latencyNow: number | null = null;
  private latencySettleUntil = 0;
  /** Track sample rate a context was last re-created for (no endless retries if the device refuses it). */
  private triedSampleRate = 0;
  private trackEndedHandled = false;
  private unsubDeviceChange: (() => void) | null = null;
  private ctxCleanup: (() => void) | null = null;
  private trackCleanup: (() => void) | null = null;

  constructor(opts?: { latencyMode?: LatencyMode }) {
    this.latencyMode = opts?.latencyMode === 'ultra' ? 'ultra' : 'normal';
  }

  // ---------------------------------------------------------------- public API

  get state(): AudioEngineState {
    return this._state;
  }

  /** Italian text of the last failure (openMic, track ended, context closed), null when fine. */
  get error(): string | null {
    return this._error;
  }

  get stats(): AudioStats {
    return this._stats;
  }

  get sinkSupported(): boolean {
    return supportsSetSinkId();
  }

  get contextState(): AudioContextState | 'none' {
    return this.ctx ? this.ctx.state : 'none';
  }

  get trackInfo(): TrackInfo | null {
    const t = this.track;
    if (!t) return null;
    const settings = t.getSettings();
    return {
      deviceId: typeof settings.deviceId === 'string' ? settings.deviceId : null,
      groupId: typeof settings.groupId === 'string' ? settings.groupId : null,
      label: t.label,
      settings,
      channelCount: typeof settings.channelCount === 'number' ? settings.channelCount : null,
    };
  }

  /** The live microphone track (tests use it to simulate a disconnection). */
  get mediaTrack(): MediaStreamTrack | null {
    return this.track;
  }

  on(cb: (e: AudioEngineEvent) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /**
   * MUST be called synchronously inside a click handler: the AudioContext is created (or resumed)
   * before the first await, so Chrome's autoplay policy lets it run. The output stays at exact zeros
   * (the processor starts disarmed).
   */
  openMic(inputDeviceId: string | null, opts: { wantErrorMic: boolean }): Promise<void> {
    let ctx: ChromeAudioContext;
    try {
      ctx = this.ensureContext();
    } catch {
      return Promise.reject(this.fail(new AudioEngineError('unsupported', MSG.unsupported)));
    }
    if (ctx.state === 'suspended' && !this.panicked) {
      this.expectSuspend = false;
      ctx.resume().catch(() => undefined);
    }
    const seq = ++this.openSeq;
    return this.openMicAsync(ctx, seq, inputDeviceId, opts.wantErrorMic === true);
  }

  /**
   * Routes the output to `sinkId` (setSinkId, or a new context where unsupported). Mutes first. A
   * rejected setSinkId leaves the previous sink requested (the context still plays there), so a
   * later re-creation keeps the working device.
   */
  async setOutput(sinkId: string): Promise<void> {
    const prev = this.sinkId;
    this.sinkId = sinkId;
    const ctx = this.ctx;
    if (!ctx) return;
    if (this.currentSink(ctx) === this.normalizedSink()) return;
    this.send({ type: 'mute', muted: true });
    if (typeof ctx.setSinkId === 'function') {
      this.expectSinkChangeUntil = now() + LATENCY_SETTLE_MS;
      // Chrome suspends the context while it switches (statechange 'suspended', then 'running').
      this.sinkSwitching = true;
      try {
        await ctx.setSinkId(this.normalizedSink());
      } catch (err) {
        this.sinkId = prev;
        throw err;
      } finally {
        if (ctx === this.ctx && ctx.state === 'suspended' && !this.panicked && !this.expectSuspend) {
          // The 'running' that ends the switch clears sinkSwitching (see onContextState).
          ctx.resume().catch(() => undefined);
        } else {
          this.sinkSwitching = false;
        }
      }
      this.resetLatencyBaseline();
    } else {
      await this.recreateContext();
    }
    this.pollStats();
  }

  /** Re-creates the context when its sample rate or sink differ from the track / requested output. */
  async ensureContextMatchesTrack(): Promise<boolean> {
    const ctx = this.ctx;
    const track = this.track;
    if (!ctx || !track) return false;
    const trackSr = track.getSettings().sampleRate;
    const srMismatch =
      typeof trackSr === 'number' && trackSr > 0 && trackSr !== ctx.sampleRate && trackSr !== this.triedSampleRate;
    const sinkMismatch =
      this.sinkId !== null && typeof ctx.setSinkId === 'function' && this.currentSink(ctx) !== this.normalizedSink();
    if (!srMismatch && !sinkMismatch) return false;
    if (srMismatch) this.triedSampleRate = trackSr;
    await this.recreateContext(srMismatch ? trackSr : ctx.sampleRate);
    return true;
  }

  /** Re-creates the context with another render quantum (64 = ultra-low, experimental). */
  async setLatencyMode(mode: LatencyMode): Promise<void> {
    const m: LatencyMode = mode === 'ultra' ? 'ultra' : 'normal';
    if (m === this.latencyMode) return;
    this.latencyMode = m;
    if (this.ctx) await this.recreateContext(this.ctx.sampleRate);
  }

  /** Posts a control message; the control state is remembered and replayed on a new processor. */
  send(msg: WorkletInMsg, transfer?: Transferable[]): void {
    const wasAudible = this.audible();
    switch (msg.type) {
      case 'params':
        this.shadow.params = msg.params;
        break;
      case 'arm':
        this.shadow.armed = msg.armed;
        break;
      case 'mute':
        this.shadow.muted = msg.muted;
        break;
      case 'panic':
        this.shadow.muted = true;
        break;
      case 'bypass':
        this.shadow.bypass = msg.bypass;
        break;
      case 'tap':
        this.shadow.tap = msg.enabled;
        break;
      case 'feedbackFir':
        this.shadow.firs[msg.channel] = msg.taps ? { bulkDelay: msg.bulkDelay, taps: Float64Array.from(msg.taps) } : null;
        this.shadow.firFs[msg.channel] = this.ctx ? this.ctx.sampleRate : null;
        break;
      default:
        break;
    }
    this.post(msg, transfer);
    if (!wasAudible && this.audible()) this.onAudible();
  }

  /**
   * PANIC: 10 ms ramp to zero in the processor, then the context is suspended. A context created
   * before resume() (latency mode, output or sample-rate change in progress) is suspended too.
   */
  panic(): void {
    this.panicked = true;
    this.send({ type: 'panic' });
    const ctx = this.ctx;
    if (!ctx) return;
    if (this.panicTimer !== null) clearTimeout(this.panicTimer);
    this.panicTimer = setTimeout(() => {
      this.panicTimer = null;
      if (ctx !== this.ctx || ctx.state !== 'running' || !this.shadow.muted) return;
      this.expectSuspend = true;
      ctx.suspend().catch(() => undefined);
    }, PANIC_SUSPEND_DELAY_MS);
  }

  /** Resumes a suspended context (after PANIC). Does not unmute. */
  async resume(): Promise<void> {
    this.panicked = false;
    if (this.panicTimer !== null) {
      clearTimeout(this.panicTimer);
      this.panicTimer = null;
    }
    const ctx = this.ctx;
    if (!ctx || ctx.state === 'running' || ctx.state === 'closed') return;
    this.expectSuspend = false;
    await ctx.resume();
  }

  /**
   * Plays `stimulus` on `channel` through master and limiter (the processor must be armed) and
   * returns sample-aligned recordings. The stimulus is copied, so the caller keeps its array.
   */
  measure(stimulus: Float32Array, channel: 0 | 1, postRollFrames: number): Promise<MeasureResult> {
    const ctx = this.ctx;
    if (!this.node || !ctx) return Promise.resolve(abortedResult());
    const id = ++this.measureSeq;
    const copy = new Float32Array(stimulus);
    const post = Number.isFinite(postRollFrames) ? Math.max(0, Math.floor(postRollFrames)) : 0;
    return new Promise<MeasureResult>((resolve) => {
      const ms = ((copy.length + post) / ctx.sampleRate) * 1000 + MEASURE_TIMEOUT_MARGIN_MS;
      const timer = setTimeout(() => {
        this.post({ type: 'abortMeasure' });
        this.settleMeasure(id, abortedResult());
      }, ms);
      this.pending.set(id, { resolve, timer });
      this.post({ type: 'measure', id, stimulus: copy, channel, postRollFrames: post }, [copy.buffer]);
    });
  }

  async close(): Promise<void> {
    this.closing = true;
    this.openSeq++;
    if (this.statsTimer !== null) clearInterval(this.statsTimer);
    this.statsTimer = null;
    if (this.panicTimer !== null) clearTimeout(this.panicTimer);
    this.panicTimer = null;
    this.unsubDeviceChange?.();
    this.unsubDeviceChange = null;
    this.abortPendingMeasures();
    this.releaseInput();
    const ctx = this.ctx;
    this.ctx = null;
    if (ctx) await this.detachContext(ctx);
    this._state = 'idle';
    this._stats = { ...EMPTY_STATS };
    this.closing = false;
  }

  // ---------------------------------------------------------------- microphone

  private async openMicAsync(
    ctx: ChromeAudioContext,
    seq: number,
    inputDeviceId: string | null,
    wantErrorMic: boolean,
  ): Promise<void> {
    const md = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    if (!md || typeof md.getUserMedia !== 'function') throw this.fail(new AudioEngineError('insecure', MSG.insecure));
    let stream: MediaStream;
    try {
      stream = await md.getUserMedia({ audio: micConstraints(inputDeviceId, false), video: false });
      const first = stream.getAudioTracks()[0];
      const channels = first?.getSettings().channelCount ?? 1;
      if (wantErrorMic && channels < 2 && seq === this.openSeq) {
        // Mono track but the error mic needs input 2: ask once more for a stereo track.
        const stereo = await md
          .getUserMedia({ audio: micConstraints(inputDeviceId, true), video: false })
          .catch(() => null);
        const ch2 = stereo?.getAudioTracks()[0]?.getSettings().channelCount ?? 1;
        if (stereo && ch2 >= 2) {
          stopStream(stream);
          stream = stereo;
        } else {
          stopStream(stereo);
        }
      }
    } catch (err) {
      if (seq !== this.openSeq) throw new AudioEngineError('aborted', MSG.aborted);
      throw this.fail(micError(err));
    }
    if (seq !== this.openSeq || this.ctx !== ctx) {
      stopStream(stream);
      throw new AudioEngineError('aborted', MSG.aborted);
    }
    const track = stream.getAudioTracks()[0];
    if (!track) {
      stopStream(stream);
      throw this.fail(new AudioEngineError('not-found', MSG.notFound));
    }
    this.releaseInput();
    this.stream = stream;
    this.track = track;
    this.attachTrack(track);
    try {
      await this.ensureGraph(ctx);
    } catch (err) {
      if (err instanceof AudioEngineError && err.code === 'aborted') throw err;
      throw this.fail(new AudioEngineError('worklet', MSG.worklet));
    }
    if (seq !== this.openSeq) throw new AudioEngineError('aborted', MSG.aborted);
    this.connectSource();
    this.holdPanic(ctx);
    this._state = 'mic';
    this._error = null;
    this.pollStats();
  }

  private attachTrack(track: MediaStreamTrack): void {
    this.trackEndedHandled = false;
    const onEnded = (): void => this.onTrackEnded(track);
    const onMute = (): void => {
      if (track === this.track) this.lifecycle('track-muted', MSG.trackMuted);
    };
    track.addEventListener('ended', onEnded);
    track.addEventListener('mute', onMute);
    this.trackCleanup = () => {
      track.removeEventListener('ended', onEnded);
      track.removeEventListener('mute', onMute);
    };
  }

  private onTrackEnded(track: MediaStreamTrack): void {
    if (track !== this.track || this.trackEndedHandled) return;
    this.trackEndedHandled = true;
    this._state = 'error';
    this._error = MSG.trackEnded;
    this.lifecycle('track-ended', MSG.trackEnded);
  }

  private releaseInput(): void {
    this.trackCleanup?.();
    this.trackCleanup = null;
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    this.source = null;
    stopStream(this.stream);
    this.stream = null;
    this.track = null;
  }

  private connectSource(): void {
    const ctx = this.ctx;
    if (!ctx || !this.node || !this.stream) return;
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    this.source = ctx.createMediaStreamSource(this.stream);
    this.source.connect(this.node);
  }

  /** Records a failure; a live track from an earlier openMic keeps the engine in 'mic'. */
  private fail(err: AudioEngineError): AudioEngineError {
    if (err.code !== 'aborted') {
      this._state = this.track !== null && this.track.readyState === 'live' && this.node !== null ? 'mic' : 'error';
      this._error = err.message;
    }
    return err;
  }

  // ---------------------------------------------------------------- context and graph

  private ensureContext(): ChromeAudioContext {
    if (this.ctx && this.ctx.state !== 'closed') return this.ctx;
    const ctx = this.createContext();
    this.adoptContext(ctx);
    return ctx;
  }

  private normalizedSink(): string {
    return this.sinkId === null || this.sinkId === 'default' ? '' : this.sinkId;
  }

  private currentSink(ctx: ChromeAudioContext): string {
    const s = ctx.sinkId;
    return typeof s === 'string' ? (s === 'default' ? '' : s) : '';
  }

  /** AudioContext({ latencyHint: 0, sinkId, renderSizeHint }) with progressively simpler fallbacks. */
  private createContext(sampleRate?: number): ChromeAudioContext {
    if (typeof AudioContext === 'undefined') throw new Error('AudioContext unavailable');
    const sink = this.normalizedSink();
    const quantum = this.latencyMode === 'ultra' ? 64 : 128;
    const attempts: Record<string, unknown>[] = [];
    for (const sr of sampleRate ? [sampleRate, 0] : [0]) {
      const base: Record<string, unknown> = { latencyHint: 0 };
      if (sr > 0) base['sampleRate'] = sr;
      if (sink !== '') attempts.push({ ...base, sinkId: sink, renderSizeHint: quantum }, { ...base, sinkId: sink });
      attempts.push({ ...base, renderSizeHint: quantum }, base);
    }
    attempts.push({ latencyHint: 'interactive' }, {});
    let last: unknown = null;
    for (const o of attempts) {
      try {
        return new AudioContext(o as AudioContextOptions) as ChromeAudioContext;
      } catch (err) {
        last = err;
      }
    }
    throw last instanceof Error ? last : new Error('AudioContext creation failed');
  }

  private adoptContext(ctx: ChromeAudioContext): void {
    this.ctx = ctx;
    this.expectSuspend = false;
    this.lastUnderrunCount = null;
    this.underrunTimes = [];
    this.lastBlockSize = 0;
    this.resetLatencyBaseline();
    const onState = (): void => this.onContextState(ctx);
    const onError = (): void => {
      if (ctx !== this.ctx) return;
      this.lifecycle('context-error', MSG.ctxError);
    };
    const onSink = (): void => {
      if (ctx !== this.ctx) return;
      this.resetLatencyBaseline();
      // Our own setSinkId() already muted; only an external change is reported.
      if (now() < this.expectSinkChangeUntil) return;
      this.lifecycle('sinkchange', MSG.sinkchange);
    };
    ctx.addEventListener('statechange', onState);
    ctx.addEventListener('error', onError);
    ctx.addEventListener('sinkchange', onSink);
    this.ctxCleanup = () => {
      ctx.removeEventListener('statechange', onState);
      ctx.removeEventListener('error', onError);
      ctx.removeEventListener('sinkchange', onSink);
    };
    if (this.statsTimer === null) this.statsTimer = setInterval(() => this.pollStats(), STATS_INTERVAL_MS);
    if (this.unsubDeviceChange === null) {
      this.unsubDeviceChange = onDeviceChange(() => this.lifecycle('devicechange', MSG.devicechange));
    }
  }

  private onContextState(ctx: ChromeAudioContext): void {
    if (ctx !== this.ctx) return;
    const s = ctx.state;
    if (s === 'running') {
      this.expectSuspend = false;
      this.sinkSwitching = false;
      this.resetLatencyBaseline();
      this.pollStats();
      return;
    }
    if (s === 'suspended') {
      if (this.expectSuspend || this.sinkSwitching) return;
      this.lifecycle('context-state', MSG.suspended);
    } else if (s === 'interrupted') {
      this.lifecycle('context-state', MSG.interrupted);
    } else if (s === 'closed') {
      if (this.closing) return;
      this._state = 'error';
      this._error = MSG.closed;
      this.lifecycle('context-state', MSG.closed);
    }
    this.pollStats();
  }

  /** Disconnects everything of `ctx` and closes it. */
  private async detachContext(ctx: ChromeAudioContext): Promise<void> {
    this.ctxCleanup?.();
    this.ctxCleanup = null;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.onprocessorerror = null;
      try {
        this.node.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    this.node = null;
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    this.source = null;
    if (ctx.state !== 'closed') await ctx.close().catch(() => undefined);
  }

  private async recreateContext(sampleRate?: number): Promise<void> {
    const wasAudible = this.shadow.armed && !this.shadow.muted;
    this.send({ type: 'mute', muted: true });
    this.abortPendingMeasures();
    const old = this.ctx;
    this.ctx = null;
    if (old) await this.detachContext(old);
    const ctx = this.createContext(sampleRate);
    this.adoptContext(ctx);
    // Not awaited: outside a user gesture the promise may stay pending until the next one.
    if (ctx.state === 'suspended' && !this.panicked) ctx.resume().catch(() => undefined);
    if (this.stream) {
      await this.ensureGraph(ctx);
      this.connectSource();
    }
    this.holdPanic(ctx);
    if (wasAudible) this.lifecycle('context-recreated', MSG.recreated);
    this.pollStats();
  }

  /** While PANIC holds, a running context (e.g. one just created) is suspended. */
  private holdPanic(ctx: ChromeAudioContext): void {
    if (!this.panicked || ctx !== this.ctx || ctx.state !== 'running') return;
    this.expectSuspend = true;
    ctx.suspend().catch(() => undefined);
  }

  private async ensureGraph(ctx: ChromeAudioContext): Promise<void> {
    if (!this.modules.has(ctx)) {
      if (!ctx.audioWorklet) throw new AudioEngineError('unsupported', MSG.unsupported);
      await ctx.audioWorklet.addModule(workletUrl);
      this.modules.add(ctx);
    }
    if (ctx !== this.ctx) throw new AudioEngineError('aborted', MSG.aborted);
    if (!this.node) this.createNode(ctx);
  }

  private createNode(ctx: ChromeAudioContext): void {
    const node = new AudioWorkletNode(ctx, PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      channelCount: 2,
      channelCountMode: 'explicit',
      channelInterpretation: 'discrete',
    });
    node.port.onmessage = (e: MessageEvent) => this.onPortMessage(node, e.data as WorkletOutMsg);
    node.onprocessorerror = () => this.onProcessorError(node);
    const dest = ctx.destination;
    try {
      dest.channelCount = Math.max(1, Math.min(2, dest.maxChannelCount || 2));
      dest.channelCountMode = 'explicit';
      dest.channelInterpretation = 'discrete';
    } catch {
      // Some devices refuse the configuration: the default (stereo, speakers) is still 2 channels.
    }
    node.connect(dest);
    this.node = node;
    this.replayShadow(ctx.sampleRate);
  }

  /**
   * A fresh processor starts disarmed with default params: bring it to the remembered state. FIRs
   * posted to a context of another sample rate are dropped (they are sample-indexed; the controller
   * re-sends them for the new rate when they fit).
   */
  private replayShadow(sampleRate: number): void {
    const s = this.shadow;
    if (s.params) this.post({ type: 'params', params: s.params });
    s.firs.forEach((f, ch) => {
      const fs = s.firFs[ch];
      if (f && fs !== null && fs !== sampleRate) s.firs[ch] = null;
      else if (f) this.post({ type: 'feedbackFir', channel: ch as 0 | 1, bulkDelay: f.bulkDelay, taps: f.taps });
    });
    // Mute before arming: a re-created processor never comes back audible by itself.
    if (s.armed) s.muted = true;
    this.post({ type: 'mute', muted: s.muted });
    this.post({ type: 'bypass', bypass: s.bypass });
    this.post({ type: 'arm', armed: s.armed });
    this.post({ type: 'tap', enabled: s.tap });
  }

  private onProcessorError(node: AudioWorkletNode): void {
    if (node !== this.node) return;
    const ctx = this.ctx;
    this.shadow.muted = true;
    node.port.onmessage = null;
    node.onprocessorerror = null;
    try {
      node.disconnect();
    } catch {
      // Already disconnected.
    }
    this.node = null;
    this.abortPendingMeasures();
    if (ctx) {
      try {
        this.createNode(ctx);
        this.connectSource();
      } catch {
        this._state = 'error';
        this._error = MSG.worklet;
      }
    }
    this.lifecycle('processorerror', MSG.processorError, false);
  }

  private post(msg: WorkletInMsg, transfer?: Transferable[]): void {
    const node = this.node;
    if (!node) return;
    try {
      node.port.postMessage(msg, transfer ?? []);
    } catch {
      // The port is gone (context closing): the shadow state is replayed on the next node.
    }
  }

  private onPortMessage(node: AudioWorkletNode, msg: WorkletOutMsg): void {
    if (node !== this.node || msg === null || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'meters':
        this.lastBlockSize = msg.blockSize;
        break;
      case 'measureResult':
        this.settleMeasure(msg.id, { ref: msg.ref, err: msg.err, masterLin: msg.masterLin, aborted: msg.aborted });
        break;
      case 'howl':
      case 'autoMute':
      case 'fault':
        // The processor muted itself; leaving that state needs an explicit unmute.
        this.shadow.muted = true;
        break;
      default:
        break;
    }
    this.emit(msg);
    if (msg.type === 'tap') {
      // Listeners copy what they need during emit(); the buffer goes back to the processor's pool.
      this.post({ type: 'recycle', buffer: msg.block }, [msg.block.buffer as ArrayBuffer]);
    }
  }

  private settleMeasure(id: number, r: MeasureResult): void {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);
    clearTimeout(p.timer);
    // A completed measurement includes the current latency: the calibration made from it is valid for it.
    if (!r.aborted && this.latencyNow !== null) this.latencyRef = this.latencyNow;
    p.resolve(r);
  }

  private abortPendingMeasures(): void {
    for (const id of [...this.pending.keys()]) this.settleMeasure(id, abortedResult());
  }

  // ---------------------------------------------------------------- lifecycle and stats

  private lifecycle(cause: LifecycleCause, reason: string, mute = true): void {
    if (mute) {
      this.send({ type: 'mute', muted: true });
      // The stimulus is played after the mute envelope: stop it (it fades) and report it aborted.
      if (this.pending.size > 0) {
        this.post({ type: 'abortMeasure' });
        this.abortPendingMeasures();
      }
    }
    this.emit({ type: 'lifecycle', reason, autoMuted: true, cause });
  }

  private emit(e: AudioEngineEvent): void {
    for (const cb of [...this.listeners]) {
      try {
        cb(e);
      } catch (err) {
        // A failing view must not stop the others (or the recycling of tap buffers).
        console.error(err);
      }
    }
  }

  private resetLatencyBaseline(): void {
    this.latencyRef = null;
    this.latencyNow = null;
    this.latencySettleUntil = now() + LATENCY_SETTLE_MS;
  }

  private audible(): boolean {
    return this.shadow.armed && !this.shadow.muted;
  }

  private pollStats(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const track = this.track;
    // MediaStreamTrack.stop() ends the track without firing 'ended'.
    if (track && track.readyState === 'ended') this.onTrackEnded(track);
    const settings = track ? track.getSettings() : null;
    const underrunEvents = ctx.playbackStats?.underrunEvents;
    const underruns = typeof underrunEvents === 'number' && Number.isFinite(underrunEvents) ? underrunEvents : null;
    const sink = ctx.sinkId;
    const stats: AudioStats = {
      sampleRate: ctx.sampleRate,
      renderQuantum: ctx.renderQuantumSize ?? (this.lastBlockSize || 128),
      baseLatencyMs: toMs(ctx.baseLatency),
      outputLatencyMs: toMs(ctx.outputLatency),
      trackLatencyMs: toMs(settings ? (settings as { latency?: unknown }).latency : null),
      underruns,
      trackSettings: settings,
      sinkId: typeof sink === 'string' ? sink : sink && typeof sink === 'object' ? 'none' : null,
    };
    this._stats = stats;
    if (ctx.state === 'running') {
      this.checkUnderruns(underruns);
      this.checkLatency(stats.outputLatencyMs);
    }
    this.emit({ type: 'stats', stats });
  }

  private checkUnderruns(count: number | null): void {
    if (count === null) return;
    const prev = this.lastUnderrunCount;
    this.lastUnderrunCount = count;
    if (prev === null || count <= prev) return;
    const t = now();
    for (let i = 0; i < Math.min(count - prev, 16); i++) this.underrunTimes.push(t);
    this.underrunTimes = this.underrunTimes.filter((x) => t - x <= UNDERRUN_WINDOW_MS);
    if (this.underrunTimes.length > UNDERRUN_LIMIT) {
      this.underrunTimes = [];
      this.lifecycle('underruns', MSG.underruns);
    }
  }

  /**
   * Audible: a jump between two readings auto-mutes, slow drifts are followed. Silent (muted or
   * disarmed): the reference stays at the latency last heard (or measured), and a jump that happened
   * meanwhile is reported at the next un-mute / arm (onAudible).
   */
  private checkLatency(ms: number | null): void {
    if (ms === null || ms <= 0 || now() < this.latencySettleUntil) return;
    this.latencyNow = ms;
    const ref = this.latencyRef;
    if (ref === null) {
      this.latencyRef = ms;
      return;
    }
    if (!this.audible()) return;
    this.latencyRef = ms;
    if (Math.abs(ms - ref) <= LATENCY_JUMP_MS) return;
    const delta = `${(ms - ref).toFixed(1).replace('.', ',')} ms`;
    this.lifecycle(
      'latency-jump',
      `La latenza d'uscita è cambiata di ${delta}: l'allineamento non è più valido, uscita silenziata. Ricalibra se il problema si ripete.`,
    );
  }

  /** The output just became audible: a latency jump while it was silent invalidates the alignment. */
  private onAudible(): void {
    const ref = this.latencyRef;
    const cur = this.latencyNow;
    if (ref === null || cur === null) return;
    this.latencyRef = cur;
    if (Math.abs(cur - ref) <= LATENCY_JUMP_MS) return;
    // Reported after the caller's own bookkeeping (it is inside a send()).
    void Promise.resolve().then(() => {
      if (!this.audible()) return;
      const delta = `${(cur - ref).toFixed(1).replace('.', ',')} ms`;
      this.lifecycle(
        'latency-jump',
        `La latenza d'uscita è cambiata di ${delta} mentre l'anti-rumore era spento: l'allineamento non è più ` +
          "valido, uscita silenziata. Ripeti la misura dei percorsi; riattiva l'uscita solo se il problema non si ripete.",
      );
    });
  }
}

const abortedResult = (): MeasureResult => ({ ref: new Float32Array(0), err: null, masterLin: 0, aborted: true });
