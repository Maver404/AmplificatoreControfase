/**
 * Application controller: the single owner of the app state. It holds the EngineParams (with the L/R
 * link), drives the audio engine through a small interface (so the logic is testable without a
 * browser), validates the setup, keeps the output disarmed until validation passes AND the user
 * explicitly arms it, loads presets safely (mute → apply → soft-start), analyses the tap stream
 * (spectra, f0, attenuation, "ANC sta aumentando il rumore" guard) and binds the keyboard.
 * main.ts only mounts the views and forwards their callbacks here; the wizard uses the same API.
 */
import { F0Window, createAnalyzer } from '../audio/analysis-client.ts';
import type { F0Analyzer, WorkerAnalyzer } from '../audio/analysis-client.ts';
import type {
  AudioEngineEvent,
  AudioStats,
  LatencyMode,
  LifecycleCause,
  MeasureResult,
  TrackInfo,
} from '../audio/audio-engine.ts';
import {
  capLoopGain,
  DEFAULT_FEEDBACK_TAPS,
  FEEDBACK_TAPS_RANGE,
  LOOP_GAIN_DANGER_DB,
  loopbackSuspectPaths,
  sanitizeCalibration,
} from '../audio/calibration.ts';
import type { CaptureFn, MeasureFn } from '../audio/calibration.ts';
import { classifyLabel, deviceKey, isPseudoId, resolvePseudo } from '../audio/devices.ts';
import type { DeviceOption, Platform } from '../audio/devices.ts';
import { TAP_CHANNELS } from '../audio/protocol.ts';
import type { Ch2Status, MeterSnapshot, WorkletInMsg } from '../audio/protocol.ts';
import type { SpeakerConfig } from '../dsp/alignment.ts';
import type { F0Estimate } from '../dsp/f0-tracker.ts';
import { DEFAULT_PARAMS, MAX_HARMONICS, clampParams } from '../dsp/params.ts';
import type { ChannelParams, EngineParams, FilterParams, HarmonicTrim, PhaseMode, TonalParams } from '../dsp/params.ts';
import { AttenuationEstimator } from './attenuation.ts';
import type { AncPhase } from './attenuation.ts';
import { bindKeyboard, isEditableTarget, keyAction } from './keyboard.ts';
import type { KeyLike } from './keyboard.ts';
import { applyChannelChange, clearTrims } from './link.ts';
import { PresetStore, mismatchWarnings } from './presets.ts';
import type { Preset } from './presets.ts';
import { SPECTRUM_FFT_SIZE, SpectrumAccumulator } from './spectrum.ts';
import { validateSetup } from './validation.ts';
import type { Issue, SetupInfo, TrackSettingsInfo, ValidationResult } from './validation.ts';

// ------------------------------------------------------------------------------------- types

export type MicPermission = 'unknown' | 'prompt' | 'granted' | 'denied';

export type { CaptureFn, MeasureFn };

export interface FeedbackFir {
  bulkDelay: number;
  taps: Float64Array;
}

/** What the controller needs from AudioEngine (src/audio/audio-engine.ts implements it). */
export interface EngineLike {
  openMic(inputDeviceId: string | null, opts: { wantErrorMic: boolean }): Promise<void>;
  setOutput(sinkId: string): Promise<void>;
  ensureContextMatchesTrack(): Promise<boolean>;
  setLatencyMode(mode: LatencyMode): Promise<void>;
  send(msg: WorkletInMsg, transfer?: Transferable[]): void;
  measure(stimulus: Float32Array, channel: 0 | 1, postRollFrames: number): Promise<MeasureResult>;
  panic(): void;
  resume(): Promise<void>;
  on(cb: (e: AudioEngineEvent) => void): () => void;
  readonly stats: AudioStats;
  readonly state: 'idle' | 'mic' | 'error';
  readonly error: string | null;
  readonly trackInfo: TrackInfo | null;
  readonly sinkSupported: boolean;
}

/** Device facts from the browser (src/audio/devices.ts in production, fakes in tests). */
export interface DeviceProvider {
  list(): Promise<{ inputs: DeviceOption[]; outputs: DeviceOption[] }>;
  platform(): Platform;
  chromium(): boolean;
  /** Microphone permission without prompting, if the Permissions API knows it. */
  permission?(): Promise<MicPermission>;
}

export interface ControllerMessage {
  level: 'info' | 'warn' | 'error';
  text: string;
}

export type ControllerChange = 'state' | 'meters' | 'spectrum' | 'stats';

export type EngineStatus = 'stopped' | 'starting' | 'listening' | 'running' | 'muted' | 'panic' | 'error';

export interface PresetsInfo {
  list: Preset[];
  activeId: string | null;
  message: ControllerMessage | null;
  /** False when presets only live in memory (storage blocked or full). */
  persistent: boolean;
  storageError: string | null;
}

export interface SpectraSnapshot {
  fs: number;
  fftSize: number;
  ref: Float32Array | null;
  /** Error spectrum averaged only while the anti-noise was on (null until it has been on). */
  err: Float32Array | null;
  /** Error spectrum averaged only while the anti-noise was off (bypass, muted, disarmed). */
  errBypass: Float32Array | null;
  /**
   * Reference spectra over the same windows as `err` / `errBypass`: the L_rif terms of the per-band
   * attenuation A = [L_err − L_rif]_off − [L_err − L_rif]_on. While the feedback neutraliser runs they
   * come from x̂ ('cond': the speaker's feedback removed), otherwise from the raw reference (which then
   * includes the feedback while the anti-noise plays). Null without the error mic or until seen.
   */
  refOn: Float32Array | null;
  refOff: Float32Array | null;
}

export interface ControllerState {
  // Microphone and devices
  engineState: 'idle' | 'mic' | 'error';
  micOpen: boolean;
  /** openMic / device switch in progress. */
  opening: boolean;
  permission: MicPermission;
  micError: string | null;
  inputs: readonly DeviceOption[];
  outputs: readonly DeviceOption[];
  inputId: string | null;
  outputId: string | null;
  sinkSupported: boolean;
  latencyMode: LatencyMode;
  trackChannels: number | null;
  /** Input-2 status as used by validation (a mono track is 'absent'). */
  ch2: Ch2Status;
  fs: number | null;
  /**
   * Device key of the open input and the chosen output, exactly as saved in presets ('' while the
   * microphone is closed): the wizard compares it with a reused calibration's key.
   */
  deviceKey: string;
  // Validation and output gate
  validation: ValidationResult;
  overrides: readonly string[];
  /** Output can be armed now (mic open, validation ok, no measurement running). */
  canArm: boolean;
  headphonesConfirmed: boolean;
  armed: boolean;
  muted: boolean;
  bypass: boolean;
  panic: boolean;
  status: EngineStatus;
  /** Anti-noise actually playing (armed, not muted/bypassed/panicked, soft-start done or running). */
  ancOn: boolean;
  // Parameters
  params: EngineParams;
  linked: boolean;
  /** Double-click values, harmonic trims included. */
  calibratedChannels: [ChannelParams, ChannelParams] | null;
  /**
   * Per-harmonic attenuation that came with the calibrated values (tonal mode): the solve's prediction or the
   * autotune's measurement (applyParams `harmonics`); null when the calibrated values came without one.
   */
  calibratedHarmonics: CalibratedHarmonic[] | null;
  speakers: SpeakerConfig;
  hasCalibration: boolean;
  feedbackFirActive: [boolean, boolean];
  /** Feedback-neutraliser length ("Avanzate"), FEEDBACK_TAPS_RANGE. */
  feedbackTaps: number;
  // Live data
  stats: AudioStats;
  meters: MeterSnapshot | null;
  f0: F0Estimate | null;
  f0Follow: boolean;
  tonalSuggested: boolean;
  attenuationDb: number | null;
  /** Age of the bypass windows the attenuation is measured against (s), null without any. */
  attenuationRefAgeS: number | null;
  measuring: boolean;
  measureProgress: number | null;
  capturing: boolean;
  message: ControllerMessage | null;
  presets: PresetsInfo;
  /** Guided-setup lock (Italian reason): while set the anti-noise cannot be un-muted. */
  ancLock: string | null;
}

/** One harmonic's attenuation attached to the calibrated values (ControllerState.calibratedHarmonics). */
export interface CalibratedHarmonic {
  /** Harmonic number (ChannelParams.harmonicTrims[h − 1]) and its frequency, Hz. */
  h: number;
  freqHz: number;
  /** dB, positive = quieter: calibration.ts HarmonicSolution.predictedDb, or autotune bypassDb − endDb. */
  attenuationDb: number;
  source: 'predicted' | 'measured';
  /** The solve could not null this harmonic within the trims' range (HarmonicSolution.capped). */
  capped?: boolean;
}

export type ArmFailure = 'no-mic' | 'validation' | 'headphones-confirm' | 'measuring' | 'locked' | 'ref-input' | 'loopback';
export type ArmResult = { ok: true } | { ok: false; reason: ArmFailure; message: string };

export interface ControllerEvent {
  type:
    | 'mic-opened'
    | 'mic-error'
    | 'armed'
    | 'disarmed'
    | 'muted'
    | 'unmuted'
    | 'panic'
    | 'bypass'
    | 'preset-loaded'
    | 'anc-worse'
    | 'lifecycle'
    | 'howl'
    | 'fault'
    | 'validation';
  detail?: string;
}

export interface ControllerOptions {
  engine: EngineLike;
  devices: DeviceProvider;
  /** Preset storage: undefined = localStorage if usable, null = memory only. */
  presetStorage?: Storage | null;
  /** Remembered device choices: undefined = localStorage if usable, null = none. */
  prefsStorage?: Storage | null;
  /**
   * Turns stored calibration data into feedback-neutralisation FIRs of `taps` taps (calibration.ts
   * feedbackFirs). Without it, loading a preset or setting calibration data clears the FIRs (they
   * would be stale). The controller only applies them while the calibration's sample rate (when it
   * has one) is the running one and the reference input is the one it was set with.
   */
  firsFromCalibration?: (calibration: unknown, taps: number) => [FeedbackFir | null, FeedbackFir | null];
  /**
   * Paths of a calibration that look like an electrical loopback (Italian names). Every un-mute is
   * refused while it is not empty. Default: calibration.ts loopbackSuspectPaths on the sanitised data.
   */
  loopbackSuspects?: (calibration: unknown) => string[];
  /** Milliseconds clock (tests); default performance.now. */
  now?: () => number;
  /**
   * Overridable validation issues treated as overridden from the start and never reset. Only the
   * e2e build passes one (output latency with a fake microphone and the host's speakers).
   */
  forcedOverrides?: readonly string[];
  /**
   * Runs the periodic f0 estimate of the tap stream. Default: the analysis worker, created on first
   * use (in-process when workers are unavailable or the worker fails).
   */
  f0Analyzer?: F0Analyzer;
}

// ------------------------------------------------------------------------------------- constants

const TAP_N = TAP_CHANNELS.length;
const T_REF = TAP_CHANNELS.indexOf('ref');
const T_COND = TAP_CHANNELS.indexOf('cond');
const T_ERR = TAP_CHANNELS.indexOf('err');

const PREFS_KEY = 'controfase.devices.v1';
/**
 * Where the UI used to keep the neutraliser length on its own (a plain number). Read once into the prefs
 * (DevicePrefs.feedbackTaps) when they have none, then removed; ignored otherwise.
 */
export const LEGACY_FEEDBACK_TAPS_KEY = 'controfase.feedbackTaps.v1';
const EXPORT_FORMAT = 'controfase-presets';
/** Before a window counts as ANC on/off: mute ramp, delay line and the acoustic decay. */
export const PHASE_SETTLE_MS = 400;
/** The engine's input-2 check runs every ~1 s: earlier readings belong to the previous track. */
const CH2_STALE_MS = 1300;
const F0_INTERVAL_MS = 2000;
const F0_WINDOW_S = 2;
/** A dominant tone (peak vs median spectrum) above this suggests the tonal mode. */
export const TONAL_DOMINANCE_DB = 15;
const F0_FOLLOW_MIN_CONFIDENCE = 0.5;
/** f0 following only tracks drift; larger moves are the user's call ("Aggancia al picco"). */
const F0_FOLLOW_MAX_RATIO = 1.03;
const CAPTURE_SETTLE_MS = 300;
const CAPTURE_MAX_S = 120;
const REF_CHANNEL_ABSENT = 'ref-channel-absent';

const ANC_WORSE_TEXT =
  "L'ANC sta aumentando il rumore: nella zona l'errore supera di oltre 3 dB il livello misurato in bypass da più di 2 s. Uscita silenziata: controlla polarità, ritardo e guadagno (o il rumore è cambiato all'improvviso).";
const PANIC_TEXT =
  'PANIC: uscite azzerate in 10 ms e motore audio sospeso. Per ripartire premi Avvio o Mute (M): riparte con il soft-start.';
const HEADPHONES_CONFIRM_TEXT =
  "L'uscita sembra un paio di cuffie: conferma di non indossarle durante le prime prove (limite −24 dBFS).";

const defaultLoopbackSuspects = (raw: unknown): string[] => loopbackSuspectPaths(sanitizeCalibration(raw));

/** Reference input stored in a plain-JSON calibration (the one its paths were measured with), null when it has none. */
const calibrationRef = (raw: unknown): 0 | 1 | null => {
  const r = raw !== null && typeof raw === 'object' ? (raw as { refChannel?: unknown }).refChannel : undefined;
  return r === 0 || r === 1 ? r : null;
};

/** Sample rate stored in a plain-JSON calibration, null when it has none. */
const calibrationFs = (raw: unknown): number | null => {
  const fs = raw !== null && typeof raw === 'object' ? (raw as { fs?: unknown }).fs : undefined;
  return typeof fs === 'number' && Number.isFinite(fs) && fs > 0 ? fs : null;
};

/** Italian dB figure with one decimal ("−3,4"). */
const fmtDb = (db: number): string => db.toFixed(1).replace('.', ',').replace('-', '−');

/** What a stability reduction of `db` did (after "Neutralizzatore a N tap: " or "Con N tap del neutralizzatore "). */
const loopMarginText = (db: number): string =>
  `il guadagno d'anello dopo la neutralizzazione supererebbe ${fmtDb(LOOP_GAIN_DANGER_DB)} dB (margine di stabilità): ` +
  `guadagni dei canali ridotti di ${fmtDb(db)} dB. Per recuperarli aumenta i tap del neutralizzatore o ripeti l'allineamento.`;

const clampTaps = (n: number): number =>
  Number.isFinite(n) ? Math.round(Math.min(FEEDBACK_TAPS_RANGE.max, Math.max(FEEDBACK_TAPS_RANGE.min, n))) : DEFAULT_FEEDBACK_TAPS;

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** A channel's own copy: its harmonic trims are an array of objects (a spread would share them). */
const copyChannel = (c: ChannelParams): ChannelParams => ({ ...c, harmonicTrims: c.harmonicTrims.map((t) => ({ ...t })) });

/** Valid, copied CalibratedHarmonic entries (h in 1…MAX_HARMONICS, finite values); null when none. */
function cleanHarmonics(list: readonly CalibratedHarmonic[] | null | undefined): CalibratedHarmonic[] | null {
  if (!Array.isArray(list)) return null;
  const out = list
    .filter((x) => Number.isInteger(x?.h) && x.h >= 1 && x.h <= MAX_HARMONICS && Number.isFinite(x.freqHz) && Number.isFinite(x.attenuationDb))
    .map((x): CalibratedHarmonic => ({
      h: x.h,
      freqHz: x.freqHz,
      attenuationDb: x.attenuationDb,
      source: x.source === 'measured' ? 'measured' : 'predicted',
      ...(x.capped === true ? { capped: true } : {}),
    }));
  return out.length > 0 ? out : null;
}

function safeLocalStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}

const meanSquare = (b: Float32Array, off: number, n: number): number => {
  let s = 0;
  for (let i = 0; i < n; i++) {
    const v = b[off + i] as number;
    s += v * v;
  }
  return s / n;
};

const toTrackSettings = (s: MediaTrackSettings | null | undefined): TrackSettingsInfo => {
  if (!s) return {};
  const bool = (v: unknown): boolean | undefined =>
    typeof v === 'boolean' ? v : typeof v === 'string' ? v !== 'none' && v !== 'false' : undefined;
  const out: TrackSettingsInfo = {};
  const ec = bool(s.echoCancellation);
  const ns = bool(s.noiseSuppression);
  const agc = bool(s.autoGainControl);
  if (ec !== undefined) out.echoCancellation = ec;
  if (ns !== undefined) out.noiseSuppression = ns;
  if (agc !== undefined) out.autoGainControl = agc;
  if (typeof s.sampleRate === 'number') out.sampleRate = s.sampleRate;
  if (typeof s.channelCount === 'number') out.channelCount = s.channelCount;
  return out;
};

/** Short note and selectability of a device in the dropdowns. */
export function describeDevice(d: DeviceOption): { note?: string; blocked: boolean } {
  const c = classifyLabel(d.label);
  if (c.airplay) return { note: 'AirPlay: non utilizzabile', blocked: true };
  if (c.bluetooth) return { note: 'Bluetooth: non utilizzabile', blocked: true };
  if (c.headphones && d.kind === 'audiooutput') return { note: 'cuffie: limite −24 dBFS', blocked: false };
  if (c.builtin) return { note: d.kind === 'audioinput' ? 'integrato: sconsigliato' : 'integrate: sconsigliate', blocked: false };
  return { blocked: false };
}

/** Issue the user must deal with first (blocks, then non-overridden errors). */
export function firstProblem(v: ValidationResult): Issue | null {
  return v.issues.find((i) => i.level === 'block') ?? v.issues.find((i) => i.level === 'error' && !i.overridden) ?? null;
}

/**
 * Messages that load parameters safely: an audible output is muted (15 ms ramp), the parameters
 * applied, and the un-mute restarts it with the 1.5 s soft-start. A silent output only gets params.
 */
export function safeApplySequence(params: EngineParams, audible: boolean): WorkletInMsg[] {
  const p: WorkletInMsg = { type: 'params', params };
  return audible ? [{ type: 'mute', muted: true }, p, { type: 'mute', muted: false }] : [p];
}

interface CaptureJob {
  ref: Float32Array;
  err: Float32Array | null;
  filled: number;
  skipBlocks: number;
  resolve: (r: { ref: Float32Array; err: Float32Array | null }) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout> | null;
}

interface DevicePrefs {
  inputId: string | null;
  outputId: string | null;
  latencyMode: LatencyMode;
  feedbackTaps: number;
}

// ------------------------------------------------------------------------------------- controller

export class AppController {
  private readonly engine: EngineLike;
  private readonly devices: DeviceProvider;
  private readonly store: PresetStore;
  private readonly prefsStorage: Storage | null;
  private readonly firsFromCalibration: ControllerOptions['firsFromCalibration'];
  private readonly loopbackSuspects: (calibration: unknown) => string[];
  private readonly clock: () => number;
  private readonly platform: Platform;
  private readonly chromium: boolean;

  private listeners = new Set<(s: ControllerState, change: ControllerChange) => void>();
  private eventListeners = new Set<(e: ControllerEvent) => void>();
  private tapListeners = new Set<(block: Float32Array, frames: number) => void>();
  private snapshot: ControllerState | null = null;
  private unsubEngine: () => void;
  private f0Timer: ReturnType<typeof setInterval> | null = null;

  // Devices / mic
  private permission: MicPermission = 'unknown';
  private micError: string | null = null;
  private opening = false;
  private inputs: DeviceOption[] = [];
  private outputs: DeviceOption[] = [];
  private inputId: string | null = null;
  private outputId: string | null = null;
  private latencyMode: LatencyMode = 'normal';
  private prefs: DevicePrefs;
  private engineCh2: Ch2Status = 'pending';
  private ch2AcceptAfter = 0;
  private stats: AudioStats;
  private fs: number | null = null;

  // Gate
  private validation: ValidationResult;
  private readonly overrides = new Set<string>();
  private readonly forcedOverrides: readonly string[];
  private headphonesConfirmed = false;
  private armed = false;
  private muted = false;
  private bypass = false;
  private panicked = false;
  private ancLock: string | null = null;
  /** A loopback cable is on input 2 (guided measurement): its silence between stimuli is expected. */
  private input2Loopback = false;

  // Parameters
  private params: EngineParams = clampParams(DEFAULT_PARAMS);
  private linked = false;
  private calibrated: [ChannelParams, ChannelParams] | null = null;
  private calibratedHarmonics: CalibratedHarmonic[] | null = null;
  private speakers: SpeakerConfig = 'shared';
  private calibration: unknown = null;
  /** Reference input the calibration was set (or its preset saved) with: its FIRs only fit that one. */
  private calRefChannel: 0 | 1 = 0;
  private firActive: [boolean, boolean] = [false, false];
  private feedbackTaps = DEFAULT_FEEDBACK_TAPS;

  // Live data
  private meters: MeterSnapshot | null = null;
  private f0: F0Estimate | null = null;
  private f0Follow = true;
  private lastTapAt = -Infinity;
  private measuring = false;
  private measureProgress: number | null = null;
  private captureJob: CaptureJob | null = null;
  /** Set while capture() holds the mute it applied; any other mute/unmute takes it over. */
  private captureMuteToken = 0;
  private message: ControllerMessage | null = null;
  private activePresetId: string | null = null;
  private presetMessage: ControllerMessage | null = null;
  /** PresetStore.list() parses storage: cached, refreshed after every preset operation. */
  private presetList: Preset[] = [];

  // Analysis
  private analysisFs = 0;
  private specRef: SpectrumAccumulator | null = null;
  /** Error spectrum with the anti-noise on / off (phase 'on' / 'off' blocks only). */
  private specErr: SpectrumAccumulator | null = null;
  private specErrOff: SpectrumAccumulator | null = null;
  /** Reference (x̂ while the neutraliser runs) over the same windows as specErr / specErrOff. */
  private specRefOn: SpectrumAccumulator | null = null;
  private specRefOff: SpectrumAccumulator | null = null;
  private tracker: F0Window | null = null;
  private f0Analyzer: F0Analyzer | null;
  /** The worker analyzer this controller created (terminated by dispose). */
  private ownAnalyzer: WorkerAnalyzer | null = null;
  /** An f0 estimate is in flight: the next tick is skipped rather than queued. */
  private f0Busy = false;
  private att: AttenuationEstimator | null = null;
  private phaseKind: AncPhase = 'transition';
  private phaseSince = 0;

  constructor(opts: ControllerOptions) {
    this.engine = opts.engine;
    this.devices = opts.devices;
    this.store = new PresetStore(opts.presetStorage);
    this.prefsStorage = opts.prefsStorage === undefined ? safeLocalStorage() : opts.prefsStorage;
    this.firsFromCalibration = opts.firsFromCalibration;
    this.loopbackSuspects = opts.loopbackSuspects ?? defaultLoopbackSuspects;
    this.clock = opts.now ?? now;
    this.forcedOverrides = [...(opts.forcedOverrides ?? [])];
    this.f0Analyzer = opts.f0Analyzer ?? null;
    this.platform = this.devices.platform();
    this.chromium = this.devices.chromium();
    this.prefs = this.readPrefs();
    this.presetList = this.store.list();
    this.latencyMode = this.prefs.latencyMode;
    this.feedbackTaps = this.prefs.feedbackTaps;
    this.stats = this.engine.stats;
    this.validation = this.computeValidation();
    this.unsubEngine = this.engine.on((e) => this.onEngineEvent(e));
    // The engine starts with these params (replayed when its processor is created).
    this.pushParams();
    void this.initPermission();
  }

  // ===================================================================== subscription

  /** Current state (a fresh immutable snapshot after every change; cheap to read repeatedly). */
  get state(): ControllerState {
    if (this.snapshot === null) this.snapshot = this.buildState();
    return this.snapshot;
  }

  /** `change` says what moved: 'state' (anything), 'meters' (~30/s), 'spectrum' (~6/s), 'stats' (2/s). */
  subscribe(cb: (s: ControllerState, change: ControllerChange) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** Discrete events (arm, panic, howl, …), e.g. for logs or the wizard. */
  onEvent(cb: (e: ControllerEvent) => void): () => void {
    this.eventListeners.add(cb);
    return () => {
      this.eventListeners.delete(cb);
    };
  }

  /** Raw tap blocks (planar TAP_CHANNELS × frames). The block is only valid during the callback. */
  onTap(cb: (block: Float32Array, frames: number) => void): () => void {
    this.tapListeners.add(cb);
    return () => {
      this.tapListeners.delete(cb);
    };
  }

  dispose(): void {
    this.unsubEngine();
    if (this.f0Timer !== null) clearInterval(this.f0Timer);
    this.f0Timer = null;
    this.ownAnalyzer?.terminate();
    this.ownAnalyzer = null;
    this.failCapture('Acquisizione interrotta.');
    this.listeners.clear();
    this.eventListeners.clear();
    this.tapListeners.clear();
  }

  // ===================================================================== microphone and devices

  get sampleRate(): number | null {
    return this.fs;
  }

  /**
   * "Consenti microfono": MUST be called synchronously from the click handler (the engine creates
   * the AudioContext before its first await). Resolves when done; errors end up in state.micError.
   */
  openMic(): Promise<void> {
    const preferred = this.inputId ?? this.prefs.inputId;
    const wantErrorMic = this.params.errorChannel >= 0;
    let first: Promise<void>;
    try {
      first = this.engine.openMic(preferred, { wantErrorMic });
    } catch (err) {
      first = Promise.reject(err);
    }
    // Re-opening replaces the input: whatever was armed must be validated and armed again.
    if (this.armed) this.disarmForDeviceChange();
    this.opening = true;
    this.micError = null;
    this.changed('state');
    return first
      .catch((err: unknown) => {
        // A remembered device that is gone: fall back to the system default input.
        const code = (err as { code?: unknown } | null)?.code;
        if (preferred !== null && (code === 'overconstrained' || code === 'not-found')) {
          return this.engine.openMic(null, { wantErrorMic });
        }
        throw err;
      })
      .then(
        () => this.afterMicOpened(),
        (err: unknown) => this.onMicError(err),
      )
      .finally(() => {
        this.opening = false;
        this.changed('state');
      });
  }

  /** Re-reads the device lists (labels need the microphone permission). */
  async refreshDevices(): Promise<void> {
    try {
      const { inputs, outputs } = await this.devices.list();
      this.inputs = inputs;
      this.outputs = outputs;
    } catch {
      // Keep the previous lists.
    }
    if (this.outputId !== null && this.outputs.length > 0 && !this.outputs.some((d) => d.deviceId === this.outputId)) {
      this.outputId = null;
      if (this.armed) this.disarmInternal();
      this.setMessage('warn', "Il dispositivo d'uscita scelto non è più disponibile: scegli di nuovo l'uscita.");
    }
    this.revalidate();
    this.changed('state');
  }

  /** Switches the microphone input. Disarms: a new device must be validated and armed again. */
  async setInput(deviceId: string): Promise<void> {
    if (deviceId === this.inputId && this.engine.state === 'mic') return;
    this.disarmForDeviceChange();
    this.inputId = deviceId;
    this.savePrefs({ inputId: deviceId });
    this.opening = true;
    this.micError = null;
    this.changed('state');
    try {
      await this.engine.openMic(deviceId, { wantErrorMic: this.params.errorChannel >= 0 });
      await this.afterMicOpened();
    } catch (err) {
      this.onMicError(err);
      this.inputId = this.matchInputId(this.engine.trackInfo);
    } finally {
      this.opening = false;
      this.changed('state');
    }
  }

  /**
   * Routes the anti-noise to another output device. Disarms (new validation and arm needed). The
   * choice is taken (shown, validated, remembered) only once the engine has switched: when the
   * switch fails the audio still plays on the previous output, and that is the one validated.
   */
  async setOutput(deviceId: string): Promise<void> {
    this.disarmForDeviceChange();
    this.opening = true;
    this.changed('state');
    try {
      if (this.engine.sinkSupported) await this.engine.setOutput(deviceId);
      this.outputId = deviceId;
      this.savePrefs({ outputId: deviceId });
      await this.engine.ensureContextMatchesTrack();
    } catch (err) {
      const detail = err instanceof Error && err.message ? ` (${err.message})` : '';
      this.setMessage('error', `Impossibile usare l'uscita selezionata${detail}: scegline un'altra.`);
    } finally {
      this.opening = false;
      this.stats = this.engine.stats;
      this.revalidate();
      this.changed('state');
    }
  }

  /** Normale (128) / Ultra-bassa (64, experimental): re-creates the audio context, disarms. */
  async setLatencyMode(mode: LatencyMode): Promise<void> {
    if (mode === this.latencyMode) return;
    this.disarmForDeviceChange();
    this.latencyMode = mode;
    this.savePrefs({ latencyMode: mode });
    this.opening = true;
    this.changed('state');
    try {
      await this.engine.setLatencyMode(mode);
    } catch {
      this.setMessage('error', 'Impossibile cambiare la latenza del motore audio.');
    } finally {
      this.opening = false;
      this.stats = this.engine.stats;
      this.revalidate();
      this.changed('state');
    }
  }

  // ===================================================================== validation and arming

  /** "Continua comunque" on an overridable validation error (or undo it). */
  setOverride(issueId: string, on: boolean): void {
    if (on) this.overrides.add(issueId);
    else this.overrides.delete(issueId);
    this.revalidate();
    this.changed('state');
  }

  /** The user confirmed the headphones warning (not worn during the first tests, volume down). */
  confirmHeadphones(on: boolean): void {
    this.headphonesConfirmed = on;
    this.changed('state');
  }

  /**
   * Arms the output ("Attiva uscita"): only with the mic open, validation passed and, for headphones,
   * the confirmation. `muted: true` arms with the anti-noise muted (calibration stimuli can play);
   * otherwise the anti-noise starts with the 1.5 s soft-start. Also leaves a PANIC.
   */
  arm(opts: { muted?: boolean; headphonesConfirmed?: boolean } = {}): ArmResult {
    if (opts.headphonesConfirmed === true) this.headphonesConfirmed = true;
    if (this.engine.state !== 'mic') {
      return this.armFail('no-mic', 'Prima consenti il microfono e scegli i dispositivi.');
    }
    if (this.measuring) return this.armFail('measuring', 'Attendi la fine della misura in corso.');
    this.revalidate();
    if (!this.validation.ok) {
      const p = firstProblem(this.validation);
      return this.armFail('validation', `Validazione non superata: ${p ? p.text : 'controlla i dispositivi.'}`);
    }
    if (this.validation.headphones && !this.headphonesConfirmed) {
      return this.armFail('headphones-confirm', HEADPHONES_CONFIRM_TEXT);
    }
    if (this.ancLock !== null && opts.muted !== true) return this.armFail('locked', this.ancLock);
    const refInput = opts.muted === true ? null : this.refInputRefusal();
    if (refInput !== null) return this.armFail('ref-input', refInput);
    const loopback = opts.muted === true ? null : this.loopbackRefusal();
    if (loopback !== null) return this.armFail('loopback', loopback);
    const muted = opts.muted === true;
    const leavingPanic = this.panicked;
    this.panicked = false;
    this.engine.resume().catch(() => undefined);
    this.pushParams();
    this.engine.send({ type: 'mute', muted });
    this.engine.send({ type: 'arm', armed: true });
    this.armed = true;
    this.muted = muted;
    this.captureMuteToken = 0;
    this.setMessage(
      'info',
      muted
        ? "Uscita abilitata con l'anti-rumore muto: possono suonare solo gli stimoli di misura."
        : leavingPanic
          ? 'Uscita riattivata con il soft-start di 1,5 s.'
          : 'Uscita attiva: parte con il soft-start di 1,5 s. Tieni la mano sul volume dell’amplificatore.',
    );
    this.emitEvent({ type: 'armed', detail: muted ? 'muted' : 'soft-start' });
    this.changed('state');
    return { ok: true };
  }

  /**
   * Guided-setup lock (Italian reason, null releases it): while set, the anti-noise cannot be
   * un-muted nor armed un-muted; an output armed muted still plays calibration stimuli. Engaging
   * it mutes an audible output (15 ms ramp).
   */
  setAncLock(reason: string | null): void {
    this.ancLock = reason;
    if (reason !== null && this.armed && !this.muted) {
      this.engine.send({ type: 'mute', muted: true });
      this.muted = true;
      this.captureMuteToken = 0;
      this.emitEvent({ type: 'muted' });
    }
    this.changed('state');
  }

  /**
   * Guided loopback measurement with a cable on input 2: while on, a silent input 2 is expected
   * between stimuli and does not fail the validation (which would disarm the output).
   */
  setInput2Loopback(on: boolean): void {
    if (on === this.input2Loopback) return;
    this.input2Loopback = on;
    this.revalidate();
    this.changed('state');
  }

  /** Back to exact zeros ("Disattiva uscita"). */
  disarm(): void {
    if (!this.armed) return;
    this.disarmInternal();
    this.setMessage('info', 'Uscita disattivata: il processore scrive zeri.');
    this.changed('state');
  }

  // ===================================================================== safety actions

  /** PANIC: 10 ms ramp to zero, then the audio context is suspended. Never throws. */
  panic(): void {
    try {
      this.engine.panic();
    } catch {
      try {
        this.engine.send({ type: 'panic' });
      } catch {
        // Nothing else can be done from here; the state below still shows PANIC.
      }
    }
    this.muted = true;
    this.panicked = true;
    this.captureMuteToken = 0;
    this.failCapture('Acquisizione interrotta dal PANIC.');
    this.setMessage('error', PANIC_TEXT);
    this.emitEvent({ type: 'panic' });
    this.changed('state');
  }

  /** Mute (15 ms ramp) / un-mute (always a 1.5 s soft-start). Un-muting needs a valid armed setup. */
  setMuted(muted: boolean): boolean {
    this.captureMuteToken = 0;
    if (muted) {
      this.engine.send({ type: 'mute', muted: true });
      this.muted = true;
      this.emitEvent({ type: 'muted' });
      this.changed('state');
      return true;
    }
    if (!this.armed) {
      // Un-muting never arms: the output is enabled only by "Attiva uscita".
      this.engine.send({ type: 'mute', muted: false });
      this.muted = false;
      if (this.panicked) {
        this.panicked = false;
        this.engine.resume().catch(() => undefined);
      }
      this.setMessage('info', "L'uscita non è attiva: usa «Attiva uscita» dopo la validazione.");
      this.changed('state');
      return false;
    }
    if (this.ancLock !== null) {
      this.setMessage('warn', this.ancLock);
      this.changed('state');
      return false;
    }
    this.revalidate();
    if (!this.validation.ok) return false;
    // Headphones that appeared while armed (e.g. the default output moved to them) need the confirmation.
    const refusal =
      this.validation.headphones && !this.headphonesConfirmed
        ? HEADPHONES_CONFIRM_TEXT
        : (this.refInputRefusal() ?? this.loopbackRefusal());
    if (refusal !== null) {
      this.setMessage('warn', refusal);
      this.changed('state');
      return false;
    }
    this.failCapture('Acquisizione interrotta: anti-rumore riattivato.');
    // Also covers a context the system suspended (lifecycle auto-mute): un-muting is the user's go-ahead.
    this.engine.resume().catch(() => undefined);
    if (this.panicked) {
      this.panicked = false;
      this.setMessage('info', 'Uscita riattivata con il soft-start di 1,5 s.');
    }
    this.engine.send({ type: 'mute', muted: false });
    this.muted = false;
    this.emitEvent({ type: 'unmuted' });
    this.changed('state');
    return true;
  }

  /**
   * M and the status-bar Mute. Disarmed, every press only explains that «Attiva uscita» enables the
   * output (and leaves a PANIC): a mute has nothing to silence there, and arm() sets the mute itself.
   */
  toggleMute(): void {
    if (!this.armed) {
      this.setMuted(false);
      return;
    }
    this.setMuted(!(this.muted || this.panicked));
  }

  /** Bypass A/B: anti-noise off/on with the 15 ms mute ramp (not the soft-start). */
  setBypass(bypass: boolean): void {
    this.engine.send({ type: 'bypass', bypass });
    this.bypass = bypass;
    this.emitEvent({ type: 'bypass', detail: bypass ? 'on' : 'off' });
    this.changed('state');
  }

  toggleBypass(): void {
    this.setBypass(!this.bypass);
  }

  /** Global shortcuts; returns true when the key was handled. */
  handleKey(e: KeyLike & { target?: unknown }): boolean {
    const action = keyAction(e, isEditableTarget(e.target));
    if (action === 'panic') this.panic();
    else if (action === 'mute') this.toggleMute();
    else if (action === 'bypass') this.toggleBypass();
    return action !== null;
  }

  /** Escape (capture phase) → PANIC always; M / B outside form fields and without modifiers. */
  bindKeyboard(target: EventTarget): () => void {
    return bindKeyboard(target, {
      panic: () => this.panic(),
      mute: () => this.toggleMute(),
      bypass: () => this.toggleBypass(),
    });
  }

  dismissMessage(): void {
    this.message = null;
    this.changed('state');
  }

  // ===================================================================== parameters

  /** Copy of the current parameters. */
  get currentParams(): EngineParams {
    return structuredClone(this.params);
  }

  /**
   * Replaces all parameters (wizard, autotune), harmonic trims included. Clamped; the +12 dB unlock keeps its
   * current state and the headphones flag follows validation. `softStart`: an audible output is muted, updated
   * and soft-started (use it for big changes); `calibrated`: the channels (trims included) become the
   * double-click values, and `harmonics` (the solve's SolveOutcome.harmonics as 'predicted', or the autotune's
   * per-harmonic result as 'measured') their per-harmonic attenuation (calibratedHarmonics; null without).
   * In tonal mode the f0 follower changes tonal.f0 every 2 s, and the user may change the master meanwhile: a
   * caller that copied the parameters before an await (an autotune run, its final apply included) applies only
   * what it changed over currentParams (ops.ts liveWithChannels). Applied whole, the copy would set f0 back by
   * the drift since (beyond the follower's 3 %, for good) and the master to the copied value.
   */
  applyParams(
    p: EngineParams,
    opts: { softStart?: boolean; calibrated?: boolean; harmonics?: readonly CalibratedHarmonic[] | null } = {},
  ): void {
    const next = clampParams({ ...p, gainUnlocked: this.params.gainUnlocked, headphones: this.validation.headphones });
    this.sendSequence(next, opts.softStart === true && this.armed && !this.muted);
    if (opts.calibrated === true) {
      this.calibrated = [copyChannel(next.channels[0]), copyChannel(next.channels[1])];
      this.calibratedHarmonics = cleanHarmonics(opts.harmonics);
    }
    this.afterParamsChange();
  }

  /**
   * One channel strip edit; applies the L/R link (offsets kept) when linked. `changed` 'harmonicTrims': the
   * channel's trims become next.harmonicTrims (linked: each entry's gain and phase move by the same amount on
   * both channels, see link.ts).
   */
  setChannel(idx: 0 | 1, next: ChannelParams, changed: keyof ChannelParams): void {
    const channels = applyChannelChange(this.params.channels, idx, next, changed, this.linked, this.params.gainUnlocked);
    this.update({ channels });
  }

  /**
   * Harmonic trim h (1 … MAX_HARMONICS) of channel idx (tonal mode; clamped to RANGES.trimGainDb /
   * trimPhaseDeg, a non-finite field is left as it is), through setChannel and the L/R link. A rise of more
   * than 6 dB reaches the engine at 6 dB/s like a gain. Double-click: pass calibratedChannels[idx].harmonicTrims[h − 1].
   */
  setHarmonicTrim(idx: 0 | 1, h: number, trim: HarmonicTrim): void {
    if (!Number.isInteger(h) || h < 1 || h > MAX_HARMONICS) return;
    const c = this.params.channels[idx];
    const harmonicTrims = c.harmonicTrims.map((t, k) => (k === h - 1 ? { gainDb: trim.gainDb, phaseDeg: trim.phaseDeg } : { ...t }));
    this.setChannel(idx, { ...c, harmonicTrims }, 'harmonicTrims');
  }

  /**
   * «Azzera armoniche» on channel idx: every harmonic trim of that channel to 0 — and of the other channel too
   * while «Collega L/R» is on (link.ts clearTrims: an absolute reset is copied, never applied as a delta).
   */
  clearHarmonicTrims(idx: 0 | 1): void {
    this.update({ channels: clearTrims(this.params.channels, idx, this.linked) });
  }

  setLinked(linked: boolean): void {
    this.linked = linked;
    this.changed('state');
  }

  setMaster(db: number): void {
    this.update({ masterDb: db });
  }

  setCeiling(db: number): void {
    this.update({ ceilingDb: db });
  }

  setFilters(filters: FilterParams): void {
    this.update({ filters: { ...filters } });
  }

  setPhaseMode(mode: PhaseMode): void {
    this.update({ phaseMode: mode });
  }

  setFRef(hz: number): void {
    this.update({ fRef: hz });
  }

  setTonal(tonal: TonalParams): void {
    this.update({ tonal: { ...tonal } });
  }

  /** +12 dB channel gain unlock (never saved in presets). Locking clamps gains back to 0 dB. */
  setGainUnlocked(on: boolean): void {
    this.update({ gainUnlocked: on });
  }

  /**
   * Input that feeds the anti-noise. Changing it while armed disarms (like a new device): another
   * microphone would drive the speaker at the gain and delay calibrated for the old one. The
   * calibration's feedback FIRs only apply to the input it was made with.
   */
  setRefChannel(ch: 0 | 1): void {
    const changed = ch !== this.params.refChannel;
    const err = this.params.errorChannel === ch ? -1 : this.params.errorChannel;
    if (changed && this.armed) {
      this.disarmInternal();
      this.setMessage(
        'warn',
        "Canale del riferimento cambiato: uscita disattivata. L'allineamento era per l'altro ingresso: ripeti la " +
          "calibrazione, poi riattiva l'uscita.",
      );
    }
    this.update({ refChannel: ch, errorChannel: err });
    if (changed) this.applyCalibrationFirs();
    this.revalidate();
    this.changed('state');
  }

  /** Error mic input (-1 = none). A mono track is re-opened asking for two channels. */
  setErrorChannel(ch: 0 | 1 | -1): void {
    const target = ch === this.params.refChannel ? -1 : ch;
    this.update({ errorChannel: target });
    this.att?.reset();
    this.resetErrorSpectra();
    const track = this.engine.trackInfo;
    if (target >= 0 && this.engine.state === 'mic' && track !== null && (track.channelCount ?? 1) < 2) {
      void this.reopenForErrorMic();
    }
    this.revalidate();
    this.changed('state');
  }

  setSpeakers(cfg: SpeakerConfig): void {
    this.speakers = cfg;
    // One speaker: the R output carries nothing (no signal to an output that is not used).
    if (cfg === 'single' && this.params.channels[1].enabled) {
      const channels: [ChannelParams, ChannelParams] = [{ ...this.params.channels[0] }, { ...this.params.channels[1], enabled: false }];
      this.update({ channels });
    } else if (cfg !== 'single' && !this.params.channels[1].enabled) {
      const channels: [ChannelParams, ChannelParams] = [{ ...this.params.channels[0] }, { ...this.params.channels[1], enabled: true }];
      this.update({ channels });
    } else {
      this.changed('state');
    }
  }

  get speakerConfig(): SpeakerConfig {
    return this.speakers;
  }

  /** Values a double-click on a control returns to (null = none); clears calibratedHarmonics. */
  setCalibratedChannels(ch: [ChannelParams, ChannelParams] | null): void {
    const q = ch ? clampParams({ ...this.params, channels: [ch[0], ch[1]] }) : null;
    this.calibrated = q ? [q.channels[0], q.channels[1]] : null;
    this.calibratedHarmonics = null;
    this.changed('state');
  }

  // ===================================================================== tonal analysis

  /** "Aggancia al picco": tonal f0 = current estimate. */
  snapF0ToPeak(): void {
    const f = this.f0;
    if (!f || !Number.isFinite(f.f0)) return;
    this.update({ tonal: { ...this.params.tonal, f0: f.f0 } });
  }

  /** In tonal mode, follow small f0 drifts automatically (default on). */
  setF0Follow(on: boolean): void {
    this.f0Follow = on;
    this.changed('state');
  }

  // ===================================================================== calibration hooks (wizard)

  get calibrationData(): unknown {
    return this.calibration;
  }

  /**
   * Plain-JSON calibration data (saved with presets), made with the reference input it records (the
   * current one when it records none). Applies or clears the feedback FIRs; an audible output is muted
   * when the data has a loopback signature.
   */
  setCalibration(cal: unknown): void {
    this.calibration = cal ?? null;
    this.calRefChannel = calibrationRef(cal) ?? this.params.refChannel;
    this.applyCalibrationFirs();
    const loopback = this.armed && !this.muted ? (this.refInputRefusal() ?? this.loopbackRefusal()) : null;
    if (loopback !== null) {
      this.engine.send({ type: 'mute', muted: true });
      this.muted = true;
      this.captureMuteToken = 0;
      this.setMessage('warn', loopback);
      this.emitEvent({ type: 'muted' });
    }
    this.changed('state');
  }

  /**
   * Feedback-neutraliser length (FEEDBACK_TAPS_RANGE, remembered); re-applies the calibration's FIRs. A
   * shorter neutraliser leaves more of the loop: the solve's stability margin is checked again for the
   * new length and the channel gains come down by any excess before the shorter FIRs are sent.
   */
  setFeedbackTaps(taps: number): void {
    const n = clampTaps(taps);
    if (n === this.feedbackTaps) return;
    this.feedbackTaps = n;
    this.savePrefs({ feedbackTaps: n });
    const cap = this.loopMargin(this.params);
    if (cap.limitedDb > 0) {
      this.lowerCalibratedGains(cap.limitedDb);
      // The per-harmonic attenuation came with the gains just lowered: it no longer holds (as after an autotune
      // that capLoopGain lowered).
      this.calibratedHarmonics = null;
      this.update({ channels: cap.params.channels }, false);
      this.setMessage('warn', `Neutralizzatore a ${n} tap: ${loopMarginText(cap.limitedDb)}`);
    }
    this.applyCalibrationFirs();
    if (cap.limitedDb > 0) this.afterParamsChange();
    else this.changed('state');
  }

  /** Feedback-neutralisation FIR for an output channel (null disables it). */
  setFeedbackFir(ch: 0 | 1, fir: FeedbackFir | null): void {
    this.sendFir(ch, fir);
    this.changed('state');
  }

  private sendFir(ch: 0 | 1, fir: FeedbackFir | null): void {
    // Any FIR message re-routes the reference through the reconfigure envelope: skip no-ops.
    if (fir === null && !this.firActive[ch]) return;
    const wasCond = this.refFromCond();
    this.engine.send({
      type: 'feedbackFir',
      channel: ch,
      bulkDelay: fir ? fir.bulkDelay : 0,
      taps: fir ? Float64Array.from(fir.taps) : null,
    });
    this.firActive[ch] = fir !== null;
    this.att?.setRefSource(this.refFromCond() ? 'cond' : 'interp');
    // The per-state reference spectra must all come from one source.
    if (this.refFromCond() !== wasCond) {
      this.specRefOn?.reset();
      this.specRefOff?.reset();
    }
  }

  /** The reference of the attenuation is x̂ ('cond') while a neutralisation FIR runs. */
  private refFromCond(): boolean {
    return this.firActive[0] || this.firActive[1];
  }

  private resetErrorSpectra(): void {
    for (const a of [this.specErr, this.specErrOff, this.specRefOn, this.specRefOff]) a?.reset();
  }

  /** MeasureFn for calibration.ts: plays a stimulus (output must be armed, may be muted). */
  readonly measure: MeasureFn = async (stimulus, channel, postRollFrames) => {
    if (!this.armed) return { ref: new Float32Array(0), err: null, masterLin: 0, aborted: true };
    this.measuring = true;
    this.measureProgress = 0;
    this.changed('state');
    try {
      return await this.engine.measure(stimulus, channel, postRollFrames);
    } finally {
      this.measuring = false;
      this.measureProgress = null;
      this.changed('state');
    }
  };

  /** Stops a running measurement at once (its result comes back aborted). */
  abortMeasure(): void {
    this.engine.send({ type: 'abortMeasure' });
  }

  /**
   * CaptureFn for calibration.ts: records `seconds` of ambient reference / error from the tap stream
   * with the anti-noise muted (muted here if needed, un-muted afterwards with the soft-start).
   * Rejects on PANIC, un-mute, or if no audio arrives.
   */
  readonly capture: CaptureFn = async (seconds) => {
    const fs = this.fs;
    if (this.engine.state !== 'mic' || fs === null) {
      throw new Error('Microfono non aperto: impossibile registrare il rumore ambiente.');
    }
    if (this.captureJob !== null) throw new Error('Una registrazione del rumore ambiente è già in corso.');
    const s = Number.isFinite(seconds) ? Math.min(CAPTURE_MAX_S, Math.max(0.05, seconds)) : 1;
    const frames = Math.max(1, Math.round(s * fs));
    let token = 0;
    if (this.armed && !this.muted) {
      this.setMuted(true);
      token = ++this.captureMuteToken;
    }
    const wantErr = this.errAvailable();
    const result = new Promise<{ ref: Float32Array; err: Float32Array | null }>((resolve, reject) => {
      const job: CaptureJob = {
        ref: new Float32Array(frames),
        err: wantErr ? new Float32Array(frames) : null,
        filled: 0,
        // Nothing is recorded until the mute ramp, the delay line and the room have settled.
        skipBlocks: Number.POSITIVE_INFINITY,
        resolve,
        reject,
        timer: null,
      };
      this.captureJob = job;
      job.timer = setTimeout(() => {
        if (this.captureJob !== job) return;
        // The block in flight may still hold audio from before the settling time.
        job.skipBlocks = 1;
        job.timer = setTimeout(() => {
          if (this.captureJob === job) this.failCapture('Nessun audio ricevuto durante la registrazione.');
        }, s * 1000 + 3000);
      }, CAPTURE_SETTLE_MS);
    });
    this.changed('state');
    try {
      return await result;
    } finally {
      if (token !== 0 && token === this.captureMuteToken && this.armed && this.muted && !this.panicked) {
        this.setMuted(false);
      }
      this.changed('state');
    }
  };

  // ===================================================================== presets

  /** Re-reads the stored presets (e.g. after another tab changed them). */
  refreshPresets(): void {
    this.presetList = this.store.list();
    this.changed('state');
  }

  savePreset(name: string): void {
    try {
      const p = this.store.save(name, this.params, {
        deviceKey: this.currentDeviceKey(),
        sampleRate: this.fs ?? 0,
        ...(this.calibration !== null ? { calibration: this.calibration } : {}),
      });
      this.activePresetId = p.id;
      const err = this.store.lastError;
      this.presetMessage = err ? { level: 'warn', text: err } : { level: 'info', text: `Preset «${p.name}» salvato.` };
    } catch (e) {
      this.presetMessage = { level: 'error', text: e instanceof Error ? e.message : 'Salvataggio non riuscito.' };
    }
    this.presetList = this.store.list();
    this.changed('state');
  }

  /**
   * Loads a preset: an audible output is muted, the (sanitised) values applied and the un-mute
   * soft-starts it. The +12 dB unlock is never restored; headphones follow the current validation.
   */
  loadPreset(id: string): void {
    const p = this.store.get(id);
    if (!p) {
      this.presetMessage = { level: 'error', text: 'Preset non trovato.' };
      this.changed('state');
      return;
    }
    if (this.measuring) {
      this.presetMessage = { level: 'warn', text: 'Attendi la fine della misura prima di caricare un preset.' };
      this.changed('state');
      return;
    }
    const warnings = mismatchWarnings(p, { deviceKey: this.currentDeviceKey(), sampleRate: this.fs ?? 0 });
    let next = clampParams({ ...p.params, gainUnlocked: false, headphones: this.validation.headphones });
    const audible = this.armed && !this.muted;
    if (audible) this.engine.send({ type: 'mute', muted: true });
    this.calibration = p.calibration ?? null;
    // The preset's calibration was measured with the input it records, else the preset's reference input.
    this.calRefChannel = calibrationRef(this.calibration) ?? next.refChannel;
    // Presets do not store the neutraliser length: the stability margin is checked for the one in use.
    const cap = this.loopMargin(next);
    if (cap.limitedDb > 0) {
      next = cap.params;
      warnings.push(`Con ${this.feedbackTaps} tap del neutralizzatore ${loopMarginText(cap.limitedDb)}`);
    }
    this.sendSequence(next, false);
    this.applyCalibrationFirs();
    // The new calibration may be for the other input or carry a loopback signature: then the output stays muted.
    const refusal = audible ? (this.refInputRefusal() ?? this.loopbackRefusal()) : null;
    if (audible && refusal === null) this.engine.send({ type: 'mute', muted: false });
    if (refusal !== null) {
      this.muted = true;
      this.captureMuteToken = 0;
      warnings.push(refusal);
      this.emitEvent({ type: 'muted' });
    }
    this.calibrated = [copyChannel(next.channels[0]), copyChannel(next.channels[1])];
    this.calibratedHarmonics = null;
    this.activePresetId = p.id;
    const how = !audible
      ? 'valori applicati'
      : refusal === null
        ? 'uscita silenziata, valori applicati e ripartenza con il soft-start'
        : 'uscita silenziata e valori applicati, anti-rumore non riattivato';
    this.presetMessage = warnings.length
      ? { level: 'warn', text: `Preset «${p.name}» caricato (${how}). ${warnings.join(' ')}` }
      : { level: 'info', text: `Preset «${p.name}» caricato (${how}).` };
    this.emitEvent({ type: 'preset-loaded', detail: p.id });
    this.afterParamsChange();
  }

  deletePreset(id: string): void {
    const p = this.store.get(id);
    this.store.remove(id);
    if (this.activePresetId === id) this.activePresetId = null;
    this.presetMessage = { level: 'info', text: p ? `Preset «${p.name}» eliminato.` : 'Preset eliminato.' };
    this.presetList = this.store.list();
    this.changed('state');
  }

  /** JSON text and a file name for one preset (id) or all of them (null). */
  exportPresets(id: string | null): { filename: string; json: string } | null {
    if (id === null) return { filename: 'controfase-presets.json', json: this.store.exportJson() };
    const p = this.store.get(id);
    if (!p) return null;
    const slug = p.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'preset';
    return {
      filename: `controfase-preset-${slug}.json`,
      json: JSON.stringify({ format: EXPORT_FORMAT, version: 1, presets: [p] }, null, 2),
    };
  }

  /** Imports presets from a file's text (sanitised; nothing is applied). */
  importPresets(json: string): void {
    const r = this.store.importJson(json);
    const err = this.store.lastError;
    if (r.imported === 0) {
      this.presetMessage = { level: 'error', text: 'Nessun preset valido nel file.' };
    } else {
      const rej = r.rejected > 0 ? ` ${r.rejected} voci non valide sono state scartate.` : '';
      this.presetMessage = {
        level: r.rejected > 0 || err ? 'warn' : 'info',
        text: `Importati ${r.imported} preset (valori riportati ai limiti sicuri, sblocco +12 dB escluso).${rej}${err ? ` ${err}` : ''}`,
      };
    }
    this.presetList = this.store.list();
    this.changed('state');
  }

  // ===================================================================== internals: engine events

  private onEngineEvent(e: AudioEngineEvent): void {
    switch (e.type) {
      case 'ready':
        this.setFs(e.sampleRate);
        this.changed('state');
        break;
      case 'meters':
        this.meters = e;
        if (this.clock() >= this.ch2AcceptAfter && e.ch2 !== this.engineCh2) {
          this.engineCh2 = e.ch2;
          this.revalidate();
          this.changed('state');
        }
        this.changed('meters');
        break;
      case 'tap':
        this.handleTap(e.block, e.frames);
        break;
      case 'measureProgress':
        this.measureProgress = e.fraction;
        this.changed('meters');
        break;
      case 'measureResult':
        break;
      case 'channelCheck':
        this.engineCh2 = e.status;
        this.ch2AcceptAfter = 0;
        this.revalidate();
        this.changed('state');
        break;
      case 'howl':
        this.muted = true;
        this.emitEvent({ type: 'howl', detail: `${e.reason}${e.freqHz !== null ? ` @ ${Math.round(e.freqHz)} Hz` : ''}` });
        this.changed('state');
        break;
      case 'autoMute':
        this.muted = true;
        this.setMessage('error', e.reason);
        this.changed('state');
        break;
      case 'fault':
        this.muted = true;
        this.setMessage('error', e.message);
        this.emitEvent({ type: 'fault', detail: e.message });
        this.changed('state');
        break;
      case 'lifecycle':
        this.onLifecycle(e.reason, e.autoMuted, e.cause);
        break;
      case 'stats':
        this.onStats(e.stats);
        break;
    }
  }

  private onLifecycle(reason: string, autoMuted: boolean, cause: LifecycleCause | undefined): void {
    const wasArmed = this.armed;
    if (autoMuted) {
      this.muted = true;
      // "Uscita silenziata" covers everything the output plays: a calibration sweep or the
      // amplifier-level noise stops too, and a recording across the event is not used.
      if (this.measuring) this.abortMeasure();
      this.failCapture(
        cause === 'track-ended' || cause === 'track-muted' ? 'Il microfono si è interrotto.' : `Acquisizione interrotta: ${reason}`,
      );
    }
    this.captureMuteToken = 0;
    if (cause === 'devicechange') void this.refreshDevices();
    if (this.engine.state === 'error') this.micError = this.engine.error;
    this.emitEvent({ type: 'lifecycle', detail: `${cause ?? 'other'}: ${reason}` });
    this.revalidate();
    // The specific reason wins over revalidate()'s generic "validation no longer passes".
    if (wasArmed || cause === 'track-ended' || (cause === 'context-state' && this.engine.state === 'error')) {
      this.setMessage(cause === 'track-ended' || this.engine.state === 'error' ? 'error' : 'warn', reason);
    }
    this.changed('state');
  }

  private onStats(s: AudioStats): void {
    const prev = this.stats;
    this.stats = s;
    if (s.sampleRate > 0) this.setFs(s.sampleRate);
    const relevant =
      prev.sampleRate !== s.sampleRate ||
      prev.outputLatencyMs !== s.outputLatencyMs ||
      prev.trackSettings?.channelCount !== s.trackSettings?.channelCount ||
      prev.trackSettings?.sampleRate !== s.trackSettings?.sampleRate ||
      prev.trackSettings?.echoCancellation !== s.trackSettings?.echoCancellation ||
      prev.trackSettings?.noiseSuppression !== s.trackSettings?.noiseSuppression ||
      prev.trackSettings?.autoGainControl !== s.trackSettings?.autoGainControl;
    const engineError = this.engine.state === 'error' ? this.engine.error : null;
    if (relevant || (engineError !== null && engineError !== this.micError)) {
      if (engineError !== null) this.micError = engineError;
      this.revalidate();
      this.changed('state');
    }
    this.changed('stats');
  }

  private setFs(fs: number): void {
    if (!(fs > 0) || fs === this.fs) return;
    this.fs = fs;
    this.ensureAnalysis(fs);
    // FIRs are sample-indexed: re-applied (or cleared) for the new rate.
    this.applyCalibrationFirs();
  }

  private ensureAnalysis(fs: number): void {
    if (this.analysisFs === fs) return;
    this.analysisFs = fs;
    this.specRef = new SpectrumAccumulator({ fftSize: SPECTRUM_FFT_SIZE });
    this.specErr = new SpectrumAccumulator({ fftSize: SPECTRUM_FFT_SIZE });
    this.specErrOff = new SpectrumAccumulator({ fftSize: SPECTRUM_FFT_SIZE });
    this.specRefOn = new SpectrumAccumulator({ fftSize: SPECTRUM_FFT_SIZE });
    this.specRefOff = new SpectrumAccumulator({ fftSize: SPECTRUM_FFT_SIZE });
    this.tracker = new F0Window(fs, F0_WINDOW_S, 20, 2000);
    this.att = new AttenuationEstimator(fs);
    this.att.setRefSource(this.refFromCond() ? 'cond' : 'interp');
  }

  // ===================================================================== internals: taps

  private errAvailable(): boolean {
    return this.params.errorChannel >= 0 && this.ch2() === 'ok';
  }

  /** ANC state of the audio arriving now, 'transition' until it has been stable for PHASE_SETTLE_MS. */
  private ancPhase(t: number): AncPhase {
    const measuringNow = this.measuring || this.meters?.measuring === true;
    const on =
      this.armed && !this.muted && !this.bypass && !this.panicked && !measuringNow && (this.meters?.softStart ?? 0) >= 0.999;
    const off = !this.armed || this.panicked || ((this.muted || this.bypass) && !measuringNow);
    const kind: AncPhase = on ? 'on' : off ? 'off' : 'transition';
    if (kind !== this.phaseKind) {
      this.phaseKind = kind;
      this.phaseSince = t;
    }
    return t - this.phaseSince >= PHASE_SETTLE_MS ? kind : 'transition';
  }

  private handleTap(block: Float32Array, frames: number): void {
    if (!(frames > 0) || block.length < TAP_N * frames) return;
    const t = this.clock();
    this.lastTapAt = t;
    for (const cb of [...this.tapListeners]) {
      try {
        cb(block, frames);
      } catch (err) {
        console.error(err);
      }
    }
    this.feedCapture(block, frames);
    const fs = this.fs;
    if (fs === null) return;
    this.ensureAnalysis(fs);
    const errOk = this.errAvailable();
    const phase = this.ancPhase(t);

    let spectrum = this.specRef!.push(block, T_REF * frames, frames);
    // Each error spectrum only averages audio of its own ANC state (transitions feed neither), and the
    // reference of that state is averaged over the same windows.
    const errSpec = !errOk ? null : phase === 'on' ? this.specErr! : phase === 'off' ? this.specErrOff! : null;
    if (errSpec !== null && errSpec.push(block, T_ERR * frames, frames)) spectrum = true;
    const refSpec = !errOk ? null : phase === 'on' ? this.specRefOn! : phase === 'off' ? this.specRefOff! : null;
    if (refSpec !== null && refSpec.push(block, (this.refFromCond() ? T_COND : T_REF) * frames, frames)) spectrum = true;
    this.tracker!.push(block.subarray(T_COND * frames, (T_COND + 1) * frames));

    if (errOk) {
      const tripped = this.att!.push(
        phase,
        frames,
        meanSquare(block, T_ERR * frames, frames),
        meanSquare(block, T_REF * frames, frames),
        meanSquare(block, T_COND * frames, frames),
      );
      if (tripped) this.onAncWorse();
    }
    if (spectrum) this.changed('spectrum');
    if (this.f0Timer === null) this.f0Timer = setInterval(() => this.updateF0(), F0_INTERVAL_MS);
  }

  private onAncWorse(): void {
    if (!this.armed || this.muted) return;
    this.engine.send({ type: 'mute', muted: true });
    this.muted = true;
    this.setMessage('error', ANC_WORSE_TEXT);
    this.emitEvent({ type: 'anc-worse' });
    this.changed('state');
  }

  private analyzer(): F0Analyzer {
    if (this.f0Analyzer === null) this.f0Analyzer = this.ownAnalyzer = createAnalyzer();
    return this.f0Analyzer;
  }

  /** Every F0_INTERVAL_MS: estimates f0 of the last F0_WINDOW_S of 'cond' in the analysis worker. */
  private updateF0(): void {
    const tracker = this.tracker;
    if (!tracker || this.f0Busy || this.clock() - this.lastTapAt > 1500) return;
    this.f0Busy = true;
    let job: Promise<F0Estimate | null>;
    try {
      job = tracker.estimate(this.analyzer());
    } catch (err) {
      job = Promise.reject(err);
    }
    void job
      .catch(() => null)
      .then((est) => {
        this.f0Busy = false;
        // Stale after a sample-rate change (new window) or dispose() (the timer is gone).
        if (tracker === this.tracker && this.f0Timer !== null) this.applyF0(est);
      });
  }

  private applyF0(est: F0Estimate | null): void {
    this.f0 = est;
    if (
      est !== null &&
      this.f0Follow &&
      this.params.phaseMode === 'tonal' &&
      est.confidence >= F0_FOLLOW_MIN_CONFIDENCE &&
      est.dominanceDb >= TONAL_DOMINANCE_DB
    ) {
      const cur = this.params.tonal.f0;
      const ratio = Math.max(est.f0, cur) / Math.min(est.f0, cur);
      if (ratio > 1 + 1e-5 && ratio <= F0_FOLLOW_MAX_RATIO) {
        this.update({ tonal: { ...this.params.tonal, f0: est.f0 } });
        return;
      }
    }
    this.changed('state');
  }

  private feedCapture(block: Float32Array, frames: number): void {
    const job = this.captureJob;
    if (!job) return;
    if (job.skipBlocks > 0) {
      job.skipBlocks--;
      return;
    }
    const n = Math.min(frames, job.ref.length - job.filled);
    job.ref.set(block.subarray(T_REF * frames, T_REF * frames + n), job.filled);
    if (job.err) job.err.set(block.subarray(T_ERR * frames, T_ERR * frames + n), job.filled);
    job.filled += n;
    if (job.filled >= job.ref.length) {
      if (job.timer !== null) clearTimeout(job.timer);
      this.captureJob = null;
      job.resolve({ ref: job.ref, err: job.err });
    }
  }

  private failCapture(text: string): void {
    const job = this.captureJob;
    if (!job) return;
    if (job.timer !== null) clearTimeout(job.timer);
    this.captureJob = null;
    job.reject(new Error(text));
  }

  // ===================================================================== internals: devices

  private async initPermission(): Promise<void> {
    if (!this.devices.permission) return;
    try {
      const p = await this.devices.permission();
      if (this.engine.state !== 'mic' && this.permission === 'unknown') {
        this.permission = p;
        this.changed('state');
      }
    } catch {
      // Unknown: the device panel offers the button anyway.
    }
  }

  private async afterMicOpened(): Promise<void> {
    this.permission = 'granted';
    this.micError = null;
    this.engineCh2 = 'pending';
    this.ch2AcceptAfter = this.clock() + CH2_STALE_MS;
    this.engine.send({ type: 'tap', enabled: true });
    this.stats = this.engine.stats;
    if (this.stats.sampleRate > 0) this.setFs(this.stats.sampleRate);
    await this.refreshDevices();
    this.inputId = this.matchInputId(this.engine.trackInfo);
    if (this.engine.sinkSupported) {
      const out = this.pickOutput();
      if (out !== null) {
        try {
          await this.engine.setOutput(out);
          // Only an output the engine actually switched to is shown and validated.
          this.outputId = out;
        } catch {
          this.setMessage('warn', "Impossibile selezionare l'uscita salvata: scegli l'uscita nel pannello Dispositivi.");
        }
      }
    }
    try {
      await this.engine.ensureContextMatchesTrack();
    } catch {
      // Validation reports the sample-rate mismatch.
    }
    this.stats = this.engine.stats;
    if (this.stats.sampleRate > 0) this.setFs(this.stats.sampleRate);
    this.revalidate();
    this.emitEvent({ type: 'mic-opened', detail: this.engine.trackInfo?.label ?? '' });
    this.changed('state');
  }

  private onMicError(err: unknown): void {
    const code = (err as { code?: unknown } | null)?.code;
    if (code === 'aborted') return;
    const text =
      err instanceof Error && err.message
        ? err.message
        : 'Impossibile aprire il microfono: controlla il permesso del browser e il collegamento dell’interfaccia.';
    if (code === 'denied') {
      this.permission = 'denied';
      // Chrome also reports a dismissed prompt as NotAllowedError: when the browser would ask again,
      // it is not a denial. ('granted' here means the system privacy settings block the browser.)
      void this.devices.permission?.().then(
        (p) => {
          if (p === 'prompt' && this.permission === 'denied' && this.engine.state !== 'mic') {
            this.permission = 'prompt';
            this.changed('state');
          }
        },
        () => undefined,
      );
    } else if (this.engine.state !== 'mic') this.permission = this.permission === 'granted' ? 'prompt' : this.permission;
    this.micError = text;
    this.setMessage('error', text);
    this.emitEvent({ type: 'mic-error', detail: text });
    this.revalidate();
  }

  private async reopenForErrorMic(): Promise<void> {
    this.disarmForDeviceChange();
    this.opening = true;
    this.changed('state');
    try {
      await this.engine.openMic(this.inputId, { wantErrorMic: true });
      await this.afterMicOpened();
    } catch (err) {
      this.onMicError(err);
    } finally {
      this.opening = false;
      this.changed('state');
    }
  }

  private matchInputId(track: TrackInfo | null): string | null {
    if (!track) return null;
    if (track.deviceId !== null && this.inputs.some((d) => d.deviceId === track.deviceId)) return track.deviceId;
    const byLabel = this.inputs.find((d) => !d.pseudo && d.label === track.label);
    return byLabel ? byLabel.deviceId : track.deviceId;
  }

  private resolvedInput(): DeviceOption | null {
    const track = this.engine.trackInfo;
    if (this.engine.state !== 'mic' || track === null) return null;
    const id = track.deviceId ?? '';
    const listed = this.inputs.find((d) => d.deviceId === id);
    const d: DeviceOption = listed ?? {
      deviceId: id,
      groupId: track.groupId ?? '',
      label: track.label,
      kind: 'audioinput',
      pseudo: isPseudoId(id),
    };
    return resolvePseudo(this.inputs, d);
  }

  /** null = system default output that cannot be identified (or no setSinkId). */
  private resolvedOutput(): DeviceOption | null {
    if (!this.engine.sinkSupported) return null;
    const id = this.outputId ?? 'default';
    const d = this.outputs.find((o) => o.deviceId === id);
    return d ? resolvePseudo(this.outputs, d) : null;
  }

  /** Saved choice, else the output of the same interface as the input, else the system default. */
  private pickOutput(): string | null {
    const usable = this.outputs.filter((o) => !describeDevice(resolvePseudo(this.outputs, o)).blocked);
    const saved = this.outputId ?? this.prefs.outputId;
    if (saved !== null && usable.some((o) => o.deviceId === saved)) return saved;
    const input = this.resolvedInput();
    if (input && input.groupId !== '') {
      const same = usable.find((o) => !o.pseudo && o.groupId === input.groupId);
      if (same) return same.deviceId;
    }
    if (this.outputs.some((o) => o.deviceId === 'default')) return 'default';
    return usable[0]?.deviceId ?? null;
  }

  private currentDeviceKey(): string {
    const input = this.resolvedInput();
    const output = this.resolvedOutput();
    if (!input && !output) return '';
    return deviceKey(input, output, [...this.inputs, ...this.outputs]);
  }

  private disarmForDeviceChange(): void {
    this.overrides.clear();
    this.headphonesConfirmed = false;
    if (this.armed) {
      this.disarmInternal();
      this.setMessage('info', 'Dispositivi cambiati: uscita disattivata. Controlla la validazione e riattiva l’uscita.');
    }
  }

  private readPrefs(): DevicePrefs {
    const d: DevicePrefs = { inputId: null, outputId: null, latencyMode: 'normal', feedbackTaps: DEFAULT_FEEDBACK_TAPS };
    let prefs = d;
    let hasTaps = false;
    try {
      const raw = this.prefsStorage?.getItem(PREFS_KEY);
      if (raw) {
        const o = JSON.parse(raw) as Record<string, unknown>;
        const str = (v: unknown): string | null => (typeof v === 'string' && v.length > 0 && v.length < 512 ? v : null);
        hasTaps = typeof o['feedbackTaps'] === 'number' && Number.isFinite(o['feedbackTaps']);
        prefs = {
          inputId: str(o['inputId']),
          outputId: str(o['outputId']),
          latencyMode: o['latencyMode'] === 'ultra' ? 'ultra' : 'normal',
          feedbackTaps: hasTaps ? clampTaps(o['feedbackTaps'] as number) : DEFAULT_FEEDBACK_TAPS,
        };
      }
    } catch {
      prefs = d;
    }
    return this.migrateLegacyTaps(prefs, hasTaps);
  }

  /**
   * One-time migration of LEGACY_FEEDBACK_TAPS_KEY: when the prefs have no neutraliser length, a valid old
   * value (clamped to FEEDBACK_TAPS_RANGE) becomes it and is saved in the prefs; the old key is removed in
   * any case (the prefs are the only place now).
   */
  private migrateLegacyTaps(prefs: DevicePrefs, hasTaps: boolean): DevicePrefs {
    const s = this.prefsStorage;
    if (!s) return prefs;
    let raw: string | null = null;
    try {
      raw = s.getItem(LEGACY_FEEDBACK_TAPS_KEY);
    } catch {
      return prefs;
    }
    if (raw === null) return prefs;
    let next = prefs;
    if (!hasTaps) {
      // A number as text ("512"), possibly JSON-quoted; blank is not 0.
      const num = (x: unknown): number =>
        typeof x === 'number' ? x : typeof x === 'string' && x.trim() !== '' ? Number(x) : Number.NaN;
      let v = Number.NaN;
      try {
        v = num(JSON.parse(raw));
      } catch {
        v = num(raw);
      }
      if (Number.isFinite(v)) {
        next = { ...prefs, feedbackTaps: clampTaps(v) };
        try {
          s.setItem(PREFS_KEY, JSON.stringify(next));
        } catch {
          // Not remembered: the old key stays for the next start.
          return next;
        }
      }
    }
    try {
      s.removeItem(LEGACY_FEEDBACK_TAPS_KEY);
    } catch {
      // Harmless: it is ignored once the prefs have the value.
    }
    return next;
  }

  private savePrefs(patch: Partial<DevicePrefs>): void {
    this.prefs = { ...this.prefs, ...patch };
    try {
      this.prefsStorage?.setItem(PREFS_KEY, JSON.stringify(this.prefs));
    } catch {
      // Not remembered: harmless.
    }
  }

  // ===================================================================== internals: validation

  private ch2(): Ch2Status {
    const n = this.engine.trackInfo?.channelCount ?? null;
    if (this.engine.state === 'mic' && n !== null && n < 2) return 'absent';
    if (this.engine.state !== 'mic') return 'pending';
    return this.input2Loopback && this.engineCh2 === 'silent' ? 'pending' : this.engineCh2;
  }

  private computeValidation(): ValidationResult {
    const s = this.stats;
    const track = this.engine.trackInfo;
    const info: SetupInfo = {
      input: this.resolvedInput(),
      output: this.resolvedOutput(),
      trackSettings: toTrackSettings(s.trackSettings ?? track?.settings ?? null),
      ctxSampleRate: s.sampleRate > 0 ? s.sampleRate : null,
      outputLatencyS: s.outputLatencyMs !== null ? s.outputLatencyMs / 1000 : null,
      sinkSupported: this.engine.sinkSupported,
      chromium: this.chromium,
      platform: this.platform,
      wantErrorMic: this.params.errorChannel >= 0,
      errorChannel: this.params.errorChannel,
      ch2Status: this.ch2(),
    };
    const v = validateSetup(info, new Set([...this.overrides, ...this.forcedOverrides]));
    if (this.engine.state === 'mic' && this.params.refChannel === 1 && (track?.channelCount ?? 2) < 2) {
      v.issues.unshift({
        id: REF_CHANNEL_ABSENT,
        level: 'block',
        text: "Il microfono di riferimento è impostato sull'ingresso 2, ma il browser consegna un solo canale: scegli «Ingresso 1» come canale del riferimento.",
        overridable: false,
      });
      v.ok = false;
    }
    return v;
  }

  private revalidate(): void {
    const prevOk = this.validation.ok;
    this.validation = this.computeValidation();
    if (this.validation.ok !== prevOk) this.emitEvent({ type: 'validation', detail: this.validation.ok ? 'ok' : 'failed' });
    if (this.validation.headphones !== this.params.headphones) {
      this.update({ headphones: this.validation.headphones }, false);
      if (this.validation.headphones) {
        // New headphones need their own confirmation (e.g. the default output moved to them while
        // armed): an audible output is muted, and un-muting waits for the confirmation.
        this.headphonesConfirmed = false;
        if (this.armed && !this.muted) {
          this.engine.send({ type: 'mute', muted: true });
          this.muted = true;
          this.captureMuteToken = 0;
          this.emitEvent({ type: 'muted' });
        }
        this.setMessage(
          'warn',
          this.armed
            ? "L'uscita ora è un paio di cuffie: anti-rumore silenziato e limite forzato a −24 dBFS. Conferma le cuffie per riattivarlo."
            : "Uscita cuffie: il limite d'uscita è forzato a −24 dBFS.",
        );
      }
    }
    if (this.armed && !this.validation.ok) {
      this.disarmInternal();
      const p = firstProblem(this.validation);
      this.setMessage('error', `Uscita disattivata: la validazione non è più superata. ${p ? p.text : ''}`.trim());
    }
    this.snapshot = null;
  }

  // ===================================================================== internals: params and state

  /** Patch + clamp + send. `notify` false: caller notifies. */
  private update(patch: Partial<EngineParams>, notify = true): void {
    this.params = clampParams({ ...this.params, ...patch, headphones: patch.headphones ?? this.validation.headphones });
    this.pushParams();
    if (notify) this.afterParamsChange();
    else this.snapshot = null;
  }

  private afterParamsChange(): void {
    this.revalidate();
    this.changed('state');
  }

  private pushParams(): void {
    this.engine.send({ type: 'params', params: this.params });
  }

  private sendSequence(next: EngineParams, audible: boolean): void {
    this.params = next;
    for (const m of safeApplySequence(next, audible)) this.engine.send(m);
  }

  /**
   * Sends the calibration's feedback FIRs, or clears them. They are sample-indexed and measured with
   * one reference input: at another sample rate they would predict the feedback at the wrong time
   * (adding it instead of removing it), with another input they model the wrong microphone. Unknown
   * rates (no audio yet, no fs in the data) pass: re-checked as soon as the rate is known, and a FIR
   * does nothing while the output is silent.
   */
  private applyCalibrationFirs(): void {
    const cal = this.calibration;
    let firs: [FeedbackFir | null, FeedbackFir | null] = [null, null];
    if (cal !== null && this.firsFromCalibration && this.params.refChannel === this.calRefChannel) {
      firs = this.firsFromCalibration(cal, this.feedbackTaps);
      const calFs = calibrationFs(cal);
      if (calFs !== null && this.fs !== null && calFs !== this.fs && (firs[0] !== null || firs[1] !== null)) {
        firs = [null, null];
        const k = (hz: number): string => (hz / 1000).toString().replace('.', ',');
        this.setMessage(
          'warn',
          `La calibrazione è a ${k(calFs)} kHz, l'audio ora a ${k(this.fs)} kHz: neutralizzazione del feedback ` +
            'disattivata (vale solo alla frequenza di campionamento della misura). Ripeti la calibrazione.',
        );
      }
    }
    this.sendFir(0, firs[0] ?? null);
    this.sendFir(1, firs[1] ?? null);
  }

  /**
   * The loop-gain stability margin of the solve (neutralised loop ≤ LOOP_GAIN_DANGER_DB, calibration.ts
   * capLoopGain) for `p` with the loaded calibration and neutraliser length: the params with every channel
   * gain lowered by the excess, and the excess (0 when there is nothing to check: no calibration or
   * feedback data, no FIRs applied for this reference input, unknown or other rate).
   */
  private loopMargin(p: EngineParams): { params: EngineParams; limitedDb: number } {
    const raw = this.calibration;
    const none = { params: p, limitedDb: 0 };
    if (raw === null || !this.firsFromCalibration || p.refChannel !== this.calRefChannel) return none;
    const cal = sanitizeCalibration(raw);
    const fs = this.fs ?? cal?.fs ?? null;
    if (cal === null || fs === null) return none;
    const r = capLoopGain(cal, p, fs, this.feedbackTaps);
    return r.limitedDb > 0 ? { params: r.params, limitedDb: r.limitedDb } : none;
  }

  /** The double-click values follow a stability reduction of the gains (they would undo it). */
  private lowerCalibratedGains(db: number): void {
    const c = this.calibrated;
    if (c === null) return;
    const q = clampParams({ ...this.params, channels: [{ ...c[0], gainDb: c[0].gainDb - db }, { ...c[1], gainDb: c[1].gainDb - db }] });
    this.calibrated = [q.channels[0], q.channels[1]];
  }

  /**
   * Italian reason to refuse an un-mute: the loaded calibration was made with the other reference input
   * (its gain, delay and feedback FIRs model another microphone; the FIRs are already dropped). null = none.
   */
  private refInputRefusal(): string | null {
    if (this.calibration === null || this.params.refChannel === this.calRefChannel) return null;
    return (
      `Anti-rumore non attivato: la calibrazione è stata fatta con il microfono di riferimento sull'ingresso ` +
      `${this.calRefChannel + 1}, ora il riferimento è l'ingresso ${this.params.refChannel + 1}. Torna a quell'ingresso ` +
      'o ripeti la configurazione guidata.'
    );
  }

  /** Italian reason to refuse an un-mute: the calibration has a loopback signature (null = none). */
  private loopbackRefusal(): string | null {
    if (this.calibration === null) return null;
    let names: string[];
    try {
      names = this.loopbackSuspects(this.calibration);
    } catch {
      names = [];
    }
    if (names.length === 0) return null;
    return (
      `Anti-rumore non attivato: le misure ${names.join(', ')} hanno la firma di un cavo di loopback, non di un ` +
      'percorso acustico. Scollega il cavo e ripeti le misure.'
    );
  }

  private disarmInternal(): void {
    this.engine.send({ type: 'arm', armed: false });
    this.armed = false;
    this.captureMuteToken = 0;
    this.emitEvent({ type: 'disarmed' });
  }

  private armFail(reason: ArmFailure, message: string): ArmResult {
    this.setMessage('warn', message);
    this.changed('state');
    return { ok: false, reason, message };
  }

  private setMessage(level: ControllerMessage['level'], text: string): void {
    this.message = { level, text };
    this.snapshot = null;
  }

  private emitEvent(e: ControllerEvent): void {
    for (const cb of [...this.eventListeners]) {
      try {
        cb(e);
      } catch (err) {
        console.error(err);
      }
    }
  }

  private changed(change: ControllerChange): void {
    this.snapshot = null;
    if (this.listeners.size === 0) return;
    const s = this.state;
    for (const cb of [...this.listeners]) {
      try {
        cb(s, change);
      } catch (err) {
        console.error(err);
      }
    }
  }

  private status(): EngineStatus {
    if (this.panicked) return 'panic';
    if (this.engine.state === 'error' && !this.opening) return 'error';
    if (this.opening) return 'starting';
    if (this.engine.state !== 'mic') return 'stopped';
    if (!this.armed) return 'listening';
    if (this.muted) return 'muted';
    if ((this.meters?.softStart ?? 0) < 0.999) return 'starting';
    return 'running';
  }

  private buildState(): ControllerState {
    const micOpen = this.engine.state === 'mic';
    const fs = this.fs;
    const tonalSuggested =
      this.f0 !== null && this.f0.dominanceDb >= TONAL_DOMINANCE_DB && this.params.phaseMode !== 'tonal';
    return {
      engineState: this.engine.state,
      micOpen,
      opening: this.opening,
      permission: micOpen ? 'granted' : this.permission,
      micError: this.micError,
      inputs: this.inputs,
      outputs: this.outputs,
      inputId: this.inputId,
      outputId: this.outputId,
      sinkSupported: this.engine.sinkSupported,
      latencyMode: this.latencyMode,
      trackChannels: this.engine.trackInfo?.channelCount ?? null,
      ch2: this.ch2(),
      fs,
      deviceKey: micOpen ? this.currentDeviceKey() : '',
      validation: this.validation,
      overrides: [...this.overrides],
      canArm: micOpen && this.validation.ok && !this.measuring && !this.opening,
      headphonesConfirmed: this.headphonesConfirmed,
      armed: this.armed,
      muted: this.muted,
      bypass: this.bypass,
      panic: this.panicked,
      status: this.status(),
      ancOn: this.armed && !this.muted && !this.bypass && !this.panicked,
      params: this.params,
      linked: this.linked,
      calibratedChannels: this.calibrated,
      calibratedHarmonics: this.calibratedHarmonics,
      speakers: this.speakers,
      hasCalibration: this.calibration !== null,
      feedbackFirActive: [this.firActive[0], this.firActive[1]],
      feedbackTaps: this.feedbackTaps,
      stats: this.stats,
      meters: this.meters,
      f0: this.f0,
      f0Follow: this.f0Follow,
      tonalSuggested,
      attenuationDb: this.errAvailable() ? (this.att?.attenuationDb ?? null) : null,
      attenuationRefAgeS: this.errAvailable() ? (this.att?.referenceAgeS ?? null) : null,
      measuring: this.measuring,
      measureProgress: this.measureProgress,
      capturing: this.captureJob !== null,
      message: this.message,
      presets: {
        list: this.presetList,
        activeId: this.activePresetId,
        message: this.presetMessage,
        persistent: this.store.persistent,
        storageError: this.store.lastError,
      },
      ancLock: this.ancLock,
    };
  }

  /** Spectra for the spectrum view (null arrays until data arrives). */
  get spectra(): SpectraSnapshot {
    const errOk = this.errAvailable();
    return {
      fs: this.fs ?? 48000,
      fftSize: SPECTRUM_FFT_SIZE,
      ref: this.specRef && this.specRef.frameCount > 0 ? this.specRef.db : null,
      err: errOk && this.specErr && this.specErr.frameCount > 0 ? this.specErr.db : null,
      errBypass: errOk && this.specErrOff && this.specErrOff.frameCount > 0 ? this.specErrOff.db : null,
      refOn: errOk && this.specRefOn && this.specRefOn.frameCount > 0 ? this.specRefOn.db : null,
      refOff: errOk && this.specRefOff && this.specRefOff.frameCount > 0 ? this.specRefOff.db : null,
    };
  }
}
