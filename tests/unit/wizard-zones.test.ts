/**
 * Step 6 with two separate zones, on the fake DOM with the real solver on seeded plants (loop-margin-plants.ts):
 * the noise capture is the plant's primary path (calibration.ts measurePrimary mocked), the rest is step-align's
 * own pipeline (solveZones, mergeSeparateZones, the summary, the advice, the per-harmonic list, the autotune
 * block). A zone the other speaker's feedback leaves below 1 dB keeps its speaker off (seed 488): the summary
 * names the one zone attenuated, the list and the loop statement are that zone's speaker's, and the autotune only
 * offers that zone, also once the step is built again. The summary's stability cut is a channel's total (its zone's
 * own and the merged one: seed 399), and its causality the playing zones' (seed 518, delay mode). A saved solution
 * with a speaker off keeps it off when it is reused (step-start).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AudioEngineEvent, AudioStats, LatencyMode, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import { checkLoop, LOOP_GAIN_DANGER_DB, solveFromCalibration } from '../../src/audio/calibration.ts';
import type { Analyzer, PathData, PathResult, PrimaryData } from '../../src/audio/calibration.ts';
import type { WorkletInMsg } from '../../src/audio/protocol.ts';
import { AppController } from '../../src/app/controller.ts';
import type { CalibratedHarmonic, ControllerState, DeviceProvider, EngineLike } from '../../src/app/controller.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import type { StepEnv } from '../../src/ui/wizard/env.ts';
import { ampKey, createWizardData, unmuteBlockers } from '../../src/ui/wizard/flow.ts';
import type { WizardData } from '../../src/ui/wizard/flow.ts';
import { createAlignStep, tuneZoneOffText } from '../../src/ui/wizard/step-align.ts';
import { FakeEvent, installFakeDom } from './fake-dom.ts';
import type { FakeDom, FakeElement } from './fake-dom.ts';
import { plantCal, plantParams, randomPlant } from './loop-margin-plants.ts';
import { mergeSeparateZones } from '../../src/ui/wizard/logic.ts';
import { createStartStep } from '../../src/ui/wizard/step-start.ts';
import { createCheckStep } from '../../src/ui/wizard/step-check.ts';

const captured = vi.hoisted(() => ({ primary: null as PrimaryData | null, calls: 0, paths: [null, null] as (PathData | null)[], swept: [] as number[] }));

vi.mock('../../src/audio/calibration.ts', async (importOriginal) => {
  const real = await importOriginal<typeof import('../../src/audio/calibration.ts')>();
  // The noise recording of either zone: the plant's primary path (the same for both zones, as in the probes); the quick
  // check's sweep of a speaker: its saved feedback path (nothing moved).
  return {
    ...real,
    measurePrimary: async (): Promise<PrimaryData> => {
      captured.calls++;
      return captured.primary!;
    },
    measurePath: async (_m: unknown, _a: unknown, o: { channel: 0 | 1 }): Promise<PathResult> => {
      captured.swept.push(o.channel);
      return { ...captured.paths[o.channel]!, ok: true } as unknown as PathResult;
    },
  };
});

let dom: FakeDom;
beforeEach(() => {
  dom = installFakeDom();
});
afterEach(() => {
  dom.restore();
});

const STATS: AudioStats = {
  sampleRate: 48000,
  renderQuantum: 128,
  baseLatencyMs: 2.7,
  outputLatencyMs: 12,
  trackLatencyMs: 5,
  underruns: 0,
  trackSettings: null,
  sinkId: '',
};

/** Just enough engine for a controller whose state the step starts from. */
class QuietEngine implements EngineLike {
  state: 'idle' | 'mic' | 'error' = 'idle';
  error: string | null = null;
  stats: AudioStats = STATS;
  trackInfo: TrackInfo | null = null;
  sinkSupported = true;
  async openMic(): Promise<void> {}
  async setOutput(): Promise<void> {}
  async ensureContextMatchesTrack(): Promise<boolean> {
    return false;
  }
  async setLatencyMode(_m: LatencyMode): Promise<void> {}
  measure(): Promise<MeasureResult> {
    return new Promise<MeasureResult>(() => {});
  }
  panic(): void {}
  async resume(): Promise<void> {}
  send(_msg: WorkletInMsg): void {}
  on(_cb: (e: AudioEngineEvent) => void): () => void {
    return () => {};
  }
}

const devices: DeviceProvider = {
  list: async () => ({ inputs: [], outputs: [] }),
  platform: () => 'mac',
  chromium: () => true,
};

/** The controller as step 6 uses it: its state, and what the step applies or un-mutes. */
function fakeController(state: ControllerState): {
  state: ControllerState;
  readonly currentParams: EngineParams;
  applied: { params: EngineParams; harmonics: readonly CalibratedHarmonic[] | null | undefined }[];
  unmuted: number;
} {
  const f = {
    state,
    get currentParams(): EngineParams {
      return f.state.params;
    },
    applied: [] as { params: EngineParams; harmonics: readonly CalibratedHarmonic[] | null | undefined }[],
    unmuted: 0,
    capture: () => Promise.reject(new Error('not used')),
    measure: () => Promise.reject(new Error('not used')),
    setCalibration(): void {},
    applyParams(params: EngineParams, o?: { harmonics?: readonly CalibratedHarmonic[] | null }): void {
      f.applied.push({ params, harmonics: o?.harmonics });
      f.state = { ...f.state, params };
    },
    onEvent: () => () => {},
    setAncLock(): void {},
    setMuted(muted: boolean): boolean {
      if (!muted) f.unmuted++;
      return true;
    },
  };
  return f;
}

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
const byClass = (root: FakeElement, cls: string): FakeElement[] => root.all((e) => e.classList.contains(cls));
const items = (list: FakeElement): string[] => list.children.map((li) => li.textContent);

/**
 * Step 6 for plant `seed` (separate zones, error mic on input 2, every unmute condition met but the solution),
 * after recording the noise in both zones and pressing «Calcola e applica».
 */
async function solvedStep(seed: number, mode: 'tonal' | 'delay' = 'tonal'): Promise<{
  root: FakeElement;
  view: ReturnType<typeof createAlignStep>;
  ctl: ReturnType<typeof fakeController>;
  data: WizardData;
  refreshes: () => number;
  env: StepEnv;
}> {
  const pl = randomPlant(seed);
  expect(pl.speakers).toBe('separate');
  const base = new AppController({ engine: new QuietEngine(), devices, presetStorage: null, prefsStorage: null }).state;
  const state: ControllerState = {
    ...base,
    params: { ...plantParams(pl), errorChannel: 1, phaseMode: mode },
    speakers: 'separate',
    fs: 48000,
    feedbackTaps: 256,
    micOpen: true,
    armed: true,
    panic: false,
    validation: { ...base.validation, ok: true },
  };
  const ctl = fakeController(state);
  const data = createWizardData({ hasSaved: false });
  data.calibration = plantCal(pl);
  data.amp.leveled = [true, true];
  data.amp.key = [ampKey(state), ampKey(state)];
  captured.primary = plantCal(pl).primary;
  captured.calls = 0;
  let refreshes = 0;
  const env: StepEnv = {
    controller: ctl as unknown as AppController,
    data,
    analyzer: () => ({}) as Analyzer,
    startOp: () => ({ signal: new AbortController().signal, end() {} }),
    abortOp() {},
    busy: null,
    // Not re-rendered by itself: the test calls update() (a click between two renders sees the last one).
    refresh() {
      refreshes++;
    },
    savePrefs() {},
    goTo() {},
    setMode() {},
    complete() {},
  };
  const view = createAlignStep(env);
  const root = view.element as unknown as FakeElement;
  view.update(ctl.state);
  for (let z = 0; z < 2; z++) {
    root.byText('Registra il rumore ambiente', 'button')!.click();
    await flush();
    view.update(ctl.state);
  }
  expect(captured.calls).toBe(2);
  const solve = root.byText('Calcola e applica', 'button')!;
  expect(solve.disabled).toBe(false);
  solve.click();
  view.update(ctl.state);
  return { root, view, ctl, data, refreshes: () => refreshes, env };
}

describe('step 6 with two separate zones', () => {
  it('a zone the other speaker leaves below 1 dB: its speaker off, the summary, list and autotune about the other zone (seed 488)', async () => {
    const { root, view, ctl, data, env } = await solvedStep(488);
    // Applied: speaker L off, zone R's values; no per-harmonic prediction for the controller (separate zones).
    expect(ctl.applied).toHaveLength(1);
    const [applied] = ctl.applied;
    expect(applied!.params.channels.map((c) => c.enabled)).toEqual([false, true]);
    expect(applied!.harmonics).toBeNull();
    expect(data.solution).toMatchObject({ method: 'error-mic', applied: true });
    expect(data.solution!.predictedDb!).toBeCloseTo(1.11, 2);
    expect(data.solution!.harmonics!.every((x) => x.zone === 1)).toBe(true);
    // The summary names the one zone attenuated, as a warning.
    const outcome = root.all((e) => e.classList.contains('wz-note') && e.textContent.startsWith('Valori applicati'))[0]!;
    expect(outcome.className).toContain('wz-note--warn');
    expect(outcome.textContent).toContain(
      'Attenuazione prevista 1,1 dB. Solo la zona R è attenuata: la cassa L resta spenta (vedi i suggerimenti). Guadagno d’anello (senza neutralizzazione) ',
    );
    // The advice: why speaker L is off, then speaker R's loop alone.
    const advice = items(byClass(root, 'wz-advice')[0]!);
    const drop = advice.findIndex((a) => a.startsWith('Zona L: con entrambe le casse attive') && a.endsWith('la cassa L resta spenta.'));
    const loop = advice.findIndex((a) => a.startsWith("Con la sola cassa R attiva il guadagno d'anello"));
    expect(drop).toBeGreaterThanOrEqual(0);
    expect(loop).toBeGreaterThan(drop);
    expect(advice.some((a) => a.startsWith('Con entrambe le casse attive'))).toBe(false);
    expect(advice.filter((a) => a.includes('resta spenta'))).toHaveLength(1);
    // The channels line and the per-harmonic list: zone R only.
    const params = root.all((e) => e.tagName === 'P' && e.textContent.startsWith('L: '))[0]!;
    expect(params.textContent).toMatch(/^L: spento · R: ritardo \d+,\d{3} ms, guadagno −?\d+,\d dB, polarità /);
    const [harm] = byClass(root, 'wz-harmonics');
    expect(items(harm!).length).toBeGreaterThan(0);
    expect(items(harm!).every((t) => t.startsWith('Zona R · '))).toBe(true);
    // The autotune: zone R chosen, zone L not offered, and why.
    const tune = byClass(root, 'wz-autotune')[0]!;
    expect(tune.hidden).toBe(false);
    const [inL, inR] = tune.all((e) => e.tagName === 'INPUT');
    expect([inL!.disabled, inL!.checked, inR!.disabled, inR!.checked]).toEqual([true, false, false, true]);
    const help = tune.all((e) => e.tagName === 'P' && e.textContent.startsWith('La cassa L è spenta'))[0]!;
    expect(help.hidden).toBe(false);
    expect(help.textContent).toBe('La cassa L è spenta: l’autotune regola solo la zona R, con il mic d’errore in quella zona.');
    expect(unmuteBlockers(ctl.state, data)).toEqual([]);
    const start = tune.byText('Avvia l’autotune', 'button')!;
    expect(start.disabled).toBe(false);
    // Zone L chosen anyway (a click before the next render): the autotune does not start, nothing is un-muted.
    inL!.checked = true;
    inL!.dispatchEvent(new FakeEvent('change'));
    start.click();
    expect(ctl.unmuted).toBe(0);
    view.update(ctl.state);
    expect(tune.textContent).toContain(tuneZoneOffText(0));
    // The next render selects the zone that plays again.
    expect([inL!.checked, inR!.checked]).toEqual([false, true]);
    expect(start.disabled).toBe(false);
    // Back from step 7 the wizard builds the step again (it starts on zone L): zone R is selected again.
    const again = createAlignStep(env);
    again.update(ctl.state);
    const tune2 = byClass(again.element as unknown as FakeElement, 'wz-autotune')[0]!;
    const [l2, r2] = tune2.all((e) => e.tagName === 'INPUT');
    expect([l2!.disabled, l2!.checked, r2!.disabled, r2!.checked]).toEqual([true, false, false, true]);
    expect(tune2.byText('Avvia l’autotune', 'button')!.disabled).toBe(false);
  });

  it('the stability cut in the summary is what a channel lost: its zone’s own cut plus the merged one (seed 399)', async () => {
    // Zone R's own cut 8.2 dB, the merged cut 2.3 dB: channel R lost 10.5 dB (the summary said 8,2 dB).
    const pl = randomPlant(399);
    const outs = ([0, 1] as const).map((z) => solveFromCalibration(plantCal(pl, z), plantParams(pl), 48000, { feedbackTaps: 256 }));
    const { root, ctl } = await solvedStep(399);
    const applied = ctl.applied[0]!.params;
    expect(applied.channels.map((c) => c.enabled)).toEqual([true, true]);
    const lost = ([0, 1] as const).map((z) => (outs[z].loopLimitedDb ?? 0) + outs[z].params.channels[z].gainDb - applied.channels[z].gainDb);
    expect(lost[1]).toBeGreaterThan(lost[0]);
    const outcome = root.all((e) => e.classList.contains('wz-note') && e.textContent.startsWith('Valori applicati'))[0]!;
    expect(outcome.textContent).toContain(`guadagni ridotti di ${lost[1]!.toFixed(1).replace('.', ',')} dB per la stabilità`);
  });

  it('causality is the playing zones’: a non-causal zone whose speaker is off is no warning in the summary (seed 518, delay mode)', async () => {
    const pl = randomPlant(518);
    const p = { ...plantParams(pl), phaseMode: 'delay' as const };
    const outs = ([0, 1] as const).map((z) => solveFromCalibration(plantCal(pl, z), p, 48000, { feedbackTaps: 256 }));
    expect([outs[0].result.causal, outs[1].result.causal]).toEqual([false, true]);
    const { root, ctl, data } = await solvedStep(518, 'delay');
    expect(ctl.applied[0]!.params.channels.map((c) => c.enabled)).toEqual([false, true]);
    expect(data.solution!.causal).toBe(true);
    const outcome = root.all((e) => e.classList.contains('wz-note') && e.textContent.startsWith('Valori applicati'))[0]!;
    expect(outcome.textContent).not.toContain('non causale');
    // Zone L's own advice on its alignment stays (it is about that zone), and its refusal reads as the speaker kept off.
    const advice = items(byClass(root, 'wz-advice')[0]!);
    expect(advice.some((a) => a.startsWith('Zona L: ') && a.includes('i parametri non sono stati cambiati'))).toBe(false);
    expect(advice.some((a) => a.startsWith('Zona L: ') && a.includes(': la cassa L resta spenta'))).toBe(true);
    expect(advice.filter((a) => a.includes('resta spenta'))).toHaveLength(1);
  });

  it('both zones keep at least 1 dB together: both speakers play, both zones listed and offered (seed 1)', async () => {
    const { root, ctl, data } = await solvedStep(1);
    expect(ctl.applied[0]!.params.channels.map((c) => c.enabled)).toEqual([true, true]);
    expect(ctl.applied[0]!.harmonics).toBeNull();
    const outcome = root.all((e) => e.classList.contains('wz-note') && e.textContent.startsWith('Valori applicati'))[0]!;
    expect(outcome.textContent).not.toContain('Solo la zona');
    expect(items(byClass(root, 'wz-advice')[0]!).some((a) => a.startsWith("Con entrambe le casse attive il guadagno d'anello"))).toBe(true);
    expect([...new Set(data.solution!.harmonics!.map((x) => x.zone))]).toEqual([0, 1]);
    const tune = byClass(root, 'wz-autotune')[0]!;
    const inputs = tune.all((e) => e.tagName === 'INPUT');
    expect(inputs.map((i) => i.disabled)).toEqual([false, false]);
    expect(tune.textContent).not.toContain('è spenta');
  });
});

/** In-memory Storage for the presets. */
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

describe('a saved separate-zones solution with a speaker off', () => {
  /** Step 6's merged params for plant `seed` (zone R without a useful solution: speaker R off), saved as a preset. */
  function saved(seed: number): { c: AppController; cal: ReturnType<typeof plantCal>; params: EngineParams } {
    const pl = randomPlant(seed);
    const p = plantParams(pl);
    const cal = plantCal(pl);
    const outs = ([0, 1] as const).map((z) => solveFromCalibration(plantCal(pl, z), p, 48000, { feedbackTaps: 256 }));
    const m = mergeSeparateZones(cal, p, outs, 48000, 256, []);
    expect(m.params.channels.map((x) => x.enabled)).toEqual([true, false]);
    const c = new AppController({ engine: new QuietEngine(), devices, presetStorage: new MemoryStorage(), prefsStorage: null });
    c.setSpeakers('separate');
    c.setCalibration(JSON.parse(JSON.stringify(cal)) as unknown);
    c.applyParams(m.params, { calibrated: true, harmonics: null });
    c.savePreset('zona L sola');
    return { c, cal, params: m.params };
  }

  it('reused through the start question, the speaker stays off and the loop within the margin (seed 492)', () => {
    // Before: loadPreset, then setSpeakers('separate') switched speaker R on again with values never solved (delay
    // 0 ms, gain −20 dB): the neutralised loop went from −6.0 to −3.2 dB, and nothing checked it before step 7.
    const { c, cal, params } = saved(492);
    c.applyParams({ ...params, channels: [params.channels[0], { ...params.channels[1], enabled: true }] });
    const data = createWizardData({ hasSaved: true });
    let mode: string | null = null;
    const env = {
      controller: c,
      data,
      analyzer: () => ({}),
      startOp: () => ({ signal: new AbortController().signal, end() {} }),
      abortOp() {},
      busy: null,
      refresh() {},
      savePrefs() {},
      goTo() {},
      setMode(x: string) {
        mode = x;
      },
      complete() {},
    } as unknown as StepEnv;
    const view = createStartStep(env);
    view.update(c.state);
    (view.element as unknown as FakeElement).byText('No, usa la calibrazione salvata', 'button')!.click();
    expect(mode).toBe('reuse');
    expect(c.state.speakers).toBe('separate');
    expect(c.state.params.channels.map((x) => x.enabled)).toEqual([true, false]);
    expect(checkLoop(cal, c.state.params, 48000, 256)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 1e-6);
  });

  it('«Controllo rapido» does not sweep or judge the speaker that is off (seed 492)', async () => {
    const { c, cal } = saved(492);
    const ctl = fakeController({ ...c.state, fs: 48000, armed: true, panic: false, micOpen: true, validation: { ...c.state.validation, ok: true } });
    const data = createWizardData({ hasSaved: true });
    data.calibration = cal;
    captured.paths = [cal.feedback[0], cal.feedback[1]];
    captured.swept = [];
    const env = {
      controller: ctl as unknown as AppController,
      data,
      analyzer: () => ({}) as Analyzer,
      startOp: () => ({ signal: new AbortController().signal, end() {} }),
      abortOp() {},
      busy: null,
      refresh() {},
      savePrefs() {},
      goTo() {},
      setMode() {},
      complete() {},
    } as unknown as StepEnv;
    const view = createCheckStep(env);
    view.update(ctl.state);
    (view.element as unknown as FakeElement).byText('Controlla le casse', 'button')!.click();
    await flush();
    await flush();
    expect(captured.swept).toEqual([0]);
    expect(data.check.channels[0]!.state).toBe('ok');
    expect(data.check.channels[1]).toEqual({ state: 'ok', text: 'Cassa spenta nella calibrazione salvata: non usata, nessun controllo.' });
  });
});
