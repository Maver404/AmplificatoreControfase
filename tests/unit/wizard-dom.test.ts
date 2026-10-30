/**
 * The guided setup mounted on the fake DOM with a real AppController and a fake audio engine:
 * the amplifier-level noise only starts once the engine's rate-limited master has reached 0 dB
 * (safety-1), closing the wizard keeps the keyboard focus (ui-13), and reusing a saved calibration
 * gives the saved master back and re-checks the paths, correcting gain and delay (ui-2, calib-9).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AudioEngineEvent, AudioStats, LatencyMode, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import { inProcessAnalyzer } from '../../src/audio/analysis-client.ts';
import { createCalibration, defaultSweepBand, measurePath } from '../../src/audio/calibration.ts';
import type { MeasureFn, PathData } from '../../src/audio/calibration.ts';
import { deviceKey } from '../../src/audio/devices.ts';
import type { DeviceOption } from '../../src/audio/devices.ts';
import type { WorkletInMsg } from '../../src/audio/protocol.ts';
import { AppController } from '../../src/app/controller.ts';
import type { DeviceProvider, EngineLike } from '../../src/app/controller.ts';
import { PRESETS_STORAGE_KEY } from '../../src/app/presets.ts';
import { clampParams, DEFAULT_PARAMS, dbToLin } from '../../src/dsp/params.ts';
import { mulberry32 } from '../../src/dsp/sweep.ts';
import { masterSettleMs, toPathData } from '../../src/ui/wizard/logic.ts';
import type { StepEnv } from '../../src/ui/wizard/env.ts';
import { createWizardData } from '../../src/ui/wizard/flow.ts';
import { createMeasureStep } from '../../src/ui/wizard/step-measure.ts';
import { createWizard } from '../../src/ui/wizard/wizard.ts';
import { FakeEvent, installFakeDom } from './fake-dom.ts';
import type { FakeDom, FakeElement } from './fake-dom.ts';

const dev = (deviceId: string, kind: DeviceOption['kind'], label: string): DeviceOption => ({ deviceId, kind, label, groupId: 'g-usb', pseudo: false });
const INPUT = dev('usb-in', 'audioinput', 'Scarlett 2i2 USB');
const OUTPUT = dev('usb-out', 'audiooutput', 'Scarlett 2i2 USB');
const STATS: AudioStats = {
  sampleRate: 48000,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: 48000, channelCount: 2, deviceId: 'usb-in' },
  sinkId: '',
};

class FakeEngine implements EngineLike {
  sent: WorkletInMsg[] = [];
  measureAt: number[] = [];
  state: 'idle' | 'mic' | 'error' = 'idle';
  error: string | null = null;
  stats: AudioStats = { ...STATS, sampleRate: 0, outputLatencyMs: null, trackSettings: null };
  trackInfo: TrackInfo | null = null;
  sinkSupported = true;
  async openMic(): Promise<void> {
    this.state = 'mic';
    this.stats = { ...STATS };
    this.trackInfo = { deviceId: 'usb-in', groupId: 'g-usb', label: INPUT.label, settings: STATS.trackSettings!, channelCount: 2 };
  }
  async setOutput(): Promise<void> {}
  async ensureContextMatchesTrack(): Promise<boolean> {
    return false;
  }
  async setLatencyMode(_m: LatencyMode): Promise<void> {}
  measure(s: Float32Array, _c: 0 | 1, p: number): Promise<MeasureResult> {
    this.measureAt.push(performance.now());
    // Never resolves within the test: one noise chunk "plays" for its whole length.
    void s;
    void p;
    return new Promise<MeasureResult>(() => {});
  }
  panic(): void {
    this.send({ type: 'panic' });
  }
  async resume(): Promise<void> {}
  send(msg: WorkletInMsg): void {
    this.sent.push(msg);
  }
  private listeners = new Set<(e: AudioEngineEvent) => void>();
  on(cb: (e: AudioEngineEvent) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
  emit(e: AudioEngineEvent): void {
    for (const cb of [...this.listeners]) cb(e);
  }
}

const devices: DeviceProvider = {
  list: async () => ({ inputs: [INPUT], outputs: [OUTPUT] }),
  platform: () => 'mac',
  chromium: () => true,
};

let dom: FakeDom;
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance', 'Date'] });
  dom = installFakeDom();
});
afterEach(() => {
  dom.restore();
  vi.useRealTimers();
});

async function setup(): Promise<{ c: AppController; e: FakeEngine; host: FakeElement }> {
  const e = new FakeEngine();
  const c = new AppController({ engine: e, devices, presetStorage: null, prefsStorage: null });
  await c.openMic();
  await c.setOutput('usb-out');
  expect(c.arm({ muted: true }).ok).toBe(true);
  const host = dom.mount();
  createWizard(host as unknown as HTMLElement, c, { autoOpen: true, storage: null });
  return { c, e, host };
}

const button = (host: FakeElement, text: string): FakeElement => host.all((b) => b.tagName === 'BUTTON' && b.textContent === text)[0]!;

describe('step 5: the loopback texts name the input the cable borrows (calib-ui-6)', () => {
  it('input 2, or input 1 when the reference microphone is on input 2', async () => {
    const e = new FakeEngine();
    const c = new AppController({ engine: e, devices, presetStorage: null, prefsStorage: null });
    await c.openMic();
    const env = {
      controller: c,
      data: createWizardData({ hasSaved: false }),
      analyzer: () => inProcessAnalyzer,
      startOp: () => ({ signal: new AbortController().signal, end() {} }),
      abortOp() {},
      busy: null,
      masterReadyInMs: () => 0,
      refresh() {},
      savePrefs() {},
      goTo() {},
      setMode() {},
      complete() {},
    } satisfies StepEnv;
    const view = createMeasureStep(env);
    const text = (): string => (view.element as unknown as FakeElement).textContent;
    view.update(c.state);
    expect(text()).toMatch(/Il cavo va dall’uscita scelta qui sotto all’ingresso 2 \(mai all’ingresso del microfono di riferimento\)/);
    c.setRefChannel(1);
    view.update(c.state);
    expect(text()).toMatch(/Il cavo va dall’uscita scelta qui sotto all’ingresso 1 \(mai all’ingresso del microfono di riferimento\)/);
    expect(text()).toMatch(/L’alimentazione phantom \(48 V\) dell’ingresso 1 è SPENTA/);
    expect(text()).toMatch(/Ho scollegato il cavo di loopback dall’ingresso 1\./);
    expect(text()).toMatch(/Un cavo dall’uscita all’ingresso 1 misura la latenza/);
    expect(text()).not.toMatch(/ingresso 2/);
  });
});

describe('step 2: the level noise waits for the master ramp (safety-1)', () => {
  it('raises the master to 0 dB on entering and posts the first noise chunk only once it has settled', async () => {
    const { c, e, host } = await setup();
    expect(c.state.params.masterDb).toBe(-30);
    button(host, 'Avanti').click();
    expect(host.textContent).toMatch(/Passo 2 di 7/);
    // Entering the step starts the 6 dB/s ramp, silently (anti-noise muted, nothing playing).
    expect(c.state.params.masterDb).toBe(0);
    for (const cb of host.all((x) => x.tagName === 'INPUT' && x.type === 'checkbox').slice(0, 3)) {
      cb.checked = true;
      cb.dispatchEvent(new FakeEvent('change'));
    }
    await vi.advanceTimersByTimeAsync(50);
    const play = button(host, 'Fai suonare il rumore di prova');
    expect(play.disabled).toBe(true);
    expect(host.textContent).toMatch(/Il volume master sta salendo a 0 dB/);
    await vi.advanceTimersByTimeAsync(3000);
    expect(play.disabled).toBe(true);
    play.click();
    expect(e.measureAt).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(masterSettleMs(-30, 0));
    expect(play.disabled).toBe(false);
    const t0 = performance.now();
    play.click();
    await vi.advanceTimersByTimeAsync(10);
    expect(e.measureAt).toHaveLength(1);
    expect(e.measureAt[0]).toBeGreaterThanOrEqual(t0);
  });

  it('a master changed meanwhile delays the noise again', async () => {
    const { c, e, host } = await setup();
    button(host, 'Avanti').click();
    for (const cb of host.all((x) => x.tagName === 'INPUT' && x.type === 'checkbox').slice(0, 3)) {
      cb.checked = true;
      cb.dispatchEvent(new FakeEvent('change'));
    }
    await vi.advanceTimersByTimeAsync(masterSettleMs(-30, 0) + 100);
    // The user turns the master down from the top bar and presses play right away.
    c.setMaster(-20);
    await vi.advanceTimersByTimeAsync(20);
    expect(c.state.params.masterDb).toBe(0);
    const play = button(host, 'Fai suonare il rumore di prova');
    expect(play.disabled).toBe(true);
    await vi.advanceTimersByTimeAsync(masterSettleMs(0, -20) + masterSettleMs(-20, 0) + 100);
    expect(play.disabled).toBe(false);
    play.click();
    await vi.advanceTimersByTimeAsync(10);
    expect(e.measureAt).toHaveLength(1);
  });
});

describe('an auto-mute stops the running operation (safety-5, platform-3)', () => {
  it('a lifecycle event ends the wizard operation at once, not only its measurement', async () => {
    const { c, e, host } = await setup();
    button(host, 'Avanti').click();
    for (const cb of host.all((x) => x.tagName === 'INPUT' && x.type === 'checkbox').slice(0, 3)) {
      cb.checked = true;
      cb.dispatchEvent(new FakeEvent('change'));
    }
    await vi.advanceTimersByTimeAsync(masterSettleMs(-30, 0) + 100);
    button(host, 'Fai suonare il rumore di prova').click();
    await vi.advanceTimersByTimeAsync(50);
    expect(e.measureAt).toHaveLength(1);
    const busy = host.all((x) => x.classList.contains('wz-busy'))[0]!;
    expect(busy.textContent).toMatch(/in corso/);
    expect(button(host, 'Avanti').disabled).toBe(true);
    e.sent = [];
    e.emit({ type: 'lifecycle', reason: 'I dispositivi audio sono cambiati: uscita silenziata.', autoMuted: true, cause: 'devicechange' });
    await vi.advanceTimersByTimeAsync(50);
    expect(c.state.muted).toBe(true);
    expect(e.sent.some((m) => m.type === 'abortMeasure')).toBe(true);
    expect(busy.textContent).toBe('');
  });
});

describe('closing the guided setup (ui-13)', () => {
  it('moves the focus to the card button instead of <body>', async () => {
    const { host } = await setup();
    const close = button(host, 'Chiudi');
    close.focus();
    close.click();
    const active = dom.document.activeElement;
    expect(active.tagName).toBe('BUTTON');
    expect(active.textContent).toMatch(/configurazione guidata/);
    expect(host.contains(active)).toBe(true);
  });
});

// ------------------------------------------------------------------------------------- reuse

class MemoryStorage implements Storage {
  private m = new Map<string, string>();
  get length(): number {
    return this.m.size;
  }
  clear(): void {
    this.m.clear();
  }
  getItem(k: string): string | null {
    return this.m.get(k) ?? null;
  }
  key(i: number): string | null {
    return [...this.m.keys()][i] ?? null;
  }
  removeItem(k: string): void {
    this.m.delete(k);
  }
  setItem(k: string, v: string): void {
    this.m.set(k, String(v));
  }
}

const FS = 48000;

/**
 * Recording of the speaker → reference path: the output (stimulus × master) delayed and scaled, a
 * room reflection 2 ms later (an acoustic path is not flat like a loopback cable; `reflection` 0 makes
 * it one) and a faint noise.
 */
function pathRecording(stim: Float32Array, post: number, masterLin: number, delay: number, gain: number, reflection = 0.5): Float32Array {
  const rnd = mulberry32(5);
  const out = new Float32Array(stim.length + post);
  const at = (j: number): number => (j >= 0 && j < stim.length ? stim[j]! : 0);
  for (let i = 0; i < out.length; i++) {
    out[i] = gain * masterLin * (at(i - delay) + reflection * at(i - delay - 96)) + 1e-5 * (2 * rnd() - 1);
  }
  return out;
}

/** The fake engine plays stimuli through a speaker → reference path of `delay` samples and `gain`. */
class PathEngine extends FakeEngine {
  masterDb = DEFAULT_PARAMS.masterDb;
  /** Strength of the 2 ms reflection (0: a flat path, the signature of a loopback cable). */
  reflection = 0.5;
  constructor(
    private readonly delay: number,
    private readonly gain: number,
  ) {
    super();
  }
  override send(msg: WorkletInMsg): void {
    super.send(msg);
    if (msg.type === 'params') this.masterDb = msg.params.masterDb;
  }
  override measure(s: Float32Array, _c: 0 | 1, p: number): Promise<MeasureResult> {
    this.measureAt.push(performance.now());
    const m = dbToLin(this.masterDb);
    return Promise.resolve({ ref: pathRecording(s, p, m, this.delay, this.gain, this.reflection), err: null, masterLin: m, aborted: false });
  }
}

describe('reusing a saved calibration (ui-2, calib-9, plan-4)', () => {
  it('re-levels at 0 dB, gives the saved master back, and corrects gain and delay from a quick sweep', async () => {
    const params = clampParams(DEFAULT_PARAMS);
    const band = defaultSweepBand(params, FS);
    // The saved calibration: speaker → reference 1300 samples, gain 0.3, measured at master 0 dB.
    const measureSaved: MeasureFn = async (stim, _ch, post) => ({ ref: pathRecording(stim, post, 1, 1300, 0.3), err: null, masterLin: 1, aborted: false });
    const saved = await measurePath(measureSaved, inProcessAnalyzer, {
      fs: FS,
      channel: 0,
      band,
      levelDbfs: -20,
      repetitions: 2,
      mic: 'ref',
      mode: 'acoustic',
      previewFirst: true,
    });
    expect(saved.ok).toBe(true);
    const cal = createCalibration(FS, band, 'single');
    cal.feedback[0] = toPathData(saved as PathData);
    const presetParams = { ...params, masterDb: -12, channels: [{ ...params.channels[0], gainDb: -12, delayMs: 4 }, { ...params.channels[1], enabled: false }] };
    const storage = new MemoryStorage();
    storage.setItem(
      PRESETS_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        presets: [
          {
            id: 'p1',
            name: 'Sera',
            createdAt: Date.now() - 3_600_000,
            deviceKey: deviceKey(INPUT, OUTPUT, [INPUT, OUTPUT]),
            sampleRate: FS,
            params: presetParams,
            calibration: JSON.parse(JSON.stringify(cal)) as unknown,
          },
        ],
      }),
    );
    // Today: the amplifier is 3 dB louder (re-levelled by ear) and the latency 1 ms longer.
    const e = new PathEngine(1348, 0.3 * dbToLin(3));
    const c = new AppController({ engine: e, devices, presetStorage: storage, prefsStorage: null });
    await c.openMic();
    await c.setOutput('usb-out');
    const host = dom.mount();
    createWizard(host as unknown as HTMLElement, c, { autoOpen: true, storage: null });
    button(host, 'No, usa la calibrazione salvata').click();
    expect(c.state.params.masterDb).toBe(-12);
    button(host, 'Abilita l’uscita per le misure').click();
    await vi.advanceTimersByTimeAsync(50);
    button(host, 'Avanti').click();
    expect(host.textContent).toMatch(/Passo 2 di 7/);
    expect(c.state.params.masterDb).toBe(0);
    for (const cb of host.all((x) => x.tagName === 'INPUT' && x.type === 'checkbox').slice(0, 3)) {
      cb.checked = true;
      cb.dispatchEvent(new FakeEvent('change'));
    }
    await vi.advanceTimersByTimeAsync(masterSettleMs(-12, 0) + 200);
    button(host, 'Fai suonare il rumore di prova').click();
    await vi.advanceTimersByTimeAsync(500);
    const leveled = host.all((x) => x.tagName === 'LABEL' && x.textContent.startsWith('Ho regolato l’amplificatore del canale L'))[0]!;
    const box = leveled.all((x) => x.tagName === 'INPUT')[0]!;
    box.checked = true;
    box.dispatchEvent(new FakeEvent('change'));
    await vi.advanceTimersByTimeAsync(50);
    button(host, 'Avanti').click();
    expect(host.textContent).toMatch(/Controllo rapido/);
    // Leaving step 2 gives the saved master back (the anti-noise level saved with the preset).
    expect(c.state.params.masterDb).toBe(-12);
    // Without the check the setup cannot go on.
    await vi.advanceTimersByTimeAsync(50);
    expect(button(host, 'Avanti').disabled).toBe(true);
    button(host, 'Controlla le casse').click();
    for (let i = 0; i < 50 && c.state.params.channels[0].gainDb === -12; i++) await vi.advanceTimersByTimeAsync(100);
    const ch = c.state.params.channels[0];
    expect(ch.gainDb).toBeCloseTo(-15, 1);
    expect(ch.delayMs).toBeCloseTo(3, 2);
    // The saved feedback path is replaced (the neutralisation FIRs follow the new latency).
    const now = c.calibrationData as { feedback: [PathData | null, PathData | null] };
    expect(now.feedback[0]!.delaySamples).toBeCloseTo(1348, 0);
    await vi.advanceTimersByTimeAsync(50);
    expect(host.textContent).toMatch(/Cassa L — corretta/);
    expect(button(host, 'Avanti').disabled).toBe(false);

    // v-safety S3 / calib-ui-1: back at step 2 the level noise plays again (the amplifier may be turned):
    // the check made for the old amplifier level no longer counts.
    button(host, 'Indietro').click();
    expect(host.textContent).toMatch(/Passo 2 di 7/);
    await vi.advanceTimersByTimeAsync(masterSettleMs(-12, 0) + 200);
    button(host, 'Fai suonare il rumore di prova').click();
    await vi.advanceTimersByTimeAsync(500);
    button(host, 'Stop').click();
    await vi.advanceTimersByTimeAsync(50);
    button(host, 'Avanti').click();
    expect(host.textContent).toMatch(/Controllo rapido/);
    await vi.advanceTimersByTimeAsync(50);
    expect(host.textContent).not.toMatch(/Calibrazione verificata/);
    expect(button(host, 'Avanti').disabled).toBe(true);
    // A new sweep against the corrected values: unchanged since, so confirmed.
    button(host, 'Controlla le casse').click();
    for (let i = 0; i < 50 && !/Cassa L — confermata/.test(host.textContent); i++) await vi.advanceTimersByTimeAsync(100);
    expect(host.textContent).toMatch(/Cassa L — confermata/);
    expect(c.state.params.channels[0].gainDb).toBeCloseTo(-15, 1);
    expect(button(host, 'Avanti').disabled).toBe(false);
  }, 30000);

  it('a loopback signature in the quick check names the reference input the sweep was recorded on (calib-ui-5)', async () => {
    const params = clampParams(DEFAULT_PARAMS);
    const band = defaultSweepBand(params, FS);
    const measureSaved: MeasureFn = async (stim, _ch, post) => ({ ref: pathRecording(stim, post, 1, 1300, 0.3), err: null, masterLin: 1, aborted: false });
    const saved = await measurePath(measureSaved, inProcessAnalyzer, { fs: FS, channel: 0, band, levelDbfs: -20, repetitions: 2, mic: 'ref', mode: 'acoustic', previewFirst: true });
    const cal = createCalibration(FS, band, 'single');
    cal.feedback[0] = toPathData(saved as PathData);
    const presetParams = { ...params, masterDb: -12, channels: [{ ...params.channels[0], gainDb: -12, delayMs: 4 }, { ...params.channels[1], enabled: false }] };
    const storage = new MemoryStorage();
    storage.setItem(
      PRESETS_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        presets: [
          { id: 'p1', name: 'Sera', createdAt: Date.now() - 3_600_000, deviceKey: deviceKey(INPUT, OUTPUT, [INPUT, OUTPUT]), sampleRate: FS, params: presetParams, calibration: JSON.parse(JSON.stringify(cal)) as unknown },
        ],
      }),
    );
    // Today a cable runs from the output into the reference input (input 1): a flat, pure delay.
    const e = new PathEngine(300, 0.3);
    e.reflection = 0;
    const c = new AppController({ engine: e, devices, presetStorage: storage, prefsStorage: null });
    await c.openMic();
    await c.setOutput('usb-out');
    expect(c.state.params.refChannel).toBe(0);
    const host = dom.mount();
    createWizard(host as unknown as HTMLElement, c, { autoOpen: true, storage: null });
    button(host, 'No, usa la calibrazione salvata').click();
    button(host, 'Abilita l’uscita per le misure').click();
    await vi.advanceTimersByTimeAsync(50);
    button(host, 'Avanti').click();
    for (const cb of host.all((x) => x.tagName === 'INPUT' && x.type === 'checkbox').slice(0, 3)) {
      cb.checked = true;
      cb.dispatchEvent(new FakeEvent('change'));
    }
    await vi.advanceTimersByTimeAsync(masterSettleMs(-12, 0) + 200);
    button(host, 'Fai suonare il rumore di prova').click();
    await vi.advanceTimersByTimeAsync(500);
    const leveled = host.all((x) => x.tagName === 'LABEL' && x.textContent.startsWith('Ho regolato l’amplificatore del canale L'))[0]!;
    const box = leveled.all((x) => x.tagName === 'INPUT')[0]!;
    box.checked = true;
    box.dispatchEvent(new FakeEvent('change'));
    await vi.advanceTimersByTimeAsync(50);
    button(host, 'Avanti').click();
    await vi.advanceTimersByTimeAsync(50);
    button(host, 'Controlla le casse').click();
    for (let i = 0; i < 50 && !/firma di un cavo/.test(host.textContent); i++) await vi.advanceTimersByTimeAsync(100);
    expect(host.textContent).toMatch(/firma di un cavo di loopback: scollega il cavo dall’ingresso 1 \(microfono di riferimento\)/);
    expect(button(host, 'Avanti').disabled).toBe(true);
  }, 30000);
});
