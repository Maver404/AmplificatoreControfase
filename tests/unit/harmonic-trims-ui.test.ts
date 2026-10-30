/**
 * Per-harmonic trims in the UI (tonal mode), in node with the fake DOM: the channel strip's «Armoniche»
 * section (rows, keyboard, reset to the calibrated value, «Azzera armoniche», total phase h·Δ + θ_h,
 * no rewrites at meter rate), the scope legend that leaves the trims out, and the wizard's per-harmonic
 * prediction (step 6), autotune outcome and measured attenuation (step 7).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AudioEngineEvent, AudioStats, LatencyMode, MeasureResult, TrackInfo } from '../../src/audio/audio-engine.ts';
import { EXACT_NULL_DB, GEOMETRY_ACCURACY_DB, createCalibration } from '../../src/audio/calibration.ts';
import type { SolveOutcome } from '../../src/audio/calibration.ts';
import type { MeterSnapshot, WorkletInMsg } from '../../src/audio/protocol.ts';
import { AppController } from '../../src/app/controller.ts';
import type { ControllerState, DeviceProvider, EngineLike, SpectraSnapshot } from '../../src/app/controller.ts';
import { clampParams, DEFAULT_PARAMS, MAX_HARMONICS, neutralTrims } from '../../src/dsp/params.ts';
import type { ChannelParams, EngineParams, HarmonicTrim } from '../../src/dsp/params.ts';
import { copyTrims, createChannelStrip } from '../../src/ui/channel-strip.ts';
import type { ChannelStripInfo } from '../../src/ui/channel-strip.ts';
import {
  createScopeView,
  EXPECTED_LEGEND,
  EXPECTED_LEGEND_TRIMMED,
  TRIMS_NOTE,
  trimmedChannels,
} from '../../src/ui/scope-view.ts';
import { bankHarmonics, harmonicTotalPhaseDeg, totalPhaseDeg } from '../../src/ui/units.ts';
import type { StepEnv } from '../../src/ui/wizard/env.ts';
import { createWizardData } from '../../src/ui/wizard/flow.ts';
import type { HarmonicPrediction, SolutionInfo } from '../../src/ui/wizard/flow.ts';
import {
  autotuneHarmonicTexts,
  calibratedMeasurements,
  calibratedPredictions,
  ceilingText,
  expectedHarmonic,
  gainLimitedAttenuationDb,
  harmonicMeasuredText,
  harmonicPredictions,
  harmonicPredictionText,
  LOOP_GUARD_TOL_DB,
  mergeZoneSolutions,
  predictedDbText,
  predictedOutcomeText,
  solvedWithErrorMic,
} from '../../src/ui/wizard/logic.ts';
import { applyAutotuneResult } from '../../src/ui/wizard/ops.ts';
import { createAlignStep } from '../../src/ui/wizard/step-align.ts';
import { createVerifyStep } from '../../src/ui/wizard/step-verify.ts';
import { FakeElement, FakeEvent, installFakeDom } from './fake-dom.ts';
import type { FakeDom } from './fake-dom.ts';

let dom: FakeDom;
beforeEach(() => {
  dom = installFakeDom();
});
afterEach(() => {
  dom.restore();
  vi.useRealTimers();
});

const asEl = (e: FakeElement): HTMLElement => e as unknown as HTMLElement;
const byClass = (root: FakeElement, cls: string): FakeElement[] => root.all((e) => e.classList.contains(cls));
const press = (e: FakeElement, key: string, mods: { shiftKey?: boolean; altKey?: boolean } = {}): void =>
  void e.dispatchEvent(new FakeEvent('keydown', { key, ...mods }));

const trims = (over: Record<number, Partial<HarmonicTrim>> = {}): HarmonicTrim[] =>
  neutralTrims().map((t, i) => ({ ...t, ...(over[i + 1] ?? {}) }));

const channel = (over: Partial<ChannelParams> = {}): ChannelParams => ({ ...clampParams(DEFAULT_PARAMS).channels[0], ...over });

const tonalInfo = (over: Partial<ChannelStripInfo> = {}): ChannelStripInfo => ({
  fs: 48000,
  speedOfSound: 343,
  phaseRefHz: 100,
  calibrated: null,
  phaseMode: 'tonal',
  gainUnlocked: false,
  linked: false,
  harmonics: 4,
  ...over,
});

function mountStrip(initial: ChannelParams = channel()): {
  host: FakeElement;
  seen: { next: ChannelParams; changed: keyof ChannelParams }[];
  strip: ReturnType<typeof createChannelStrip>;
} {
  const host = dom.mount();
  const seen: { next: ChannelParams; changed: keyof ChannelParams }[] = [];
  const strip = createChannelStrip(asEl(host), { channel: 'L', initial, onChange: (next, changed) => seen.push({ next, changed }), onLinkChange: () => {} });
  return { host, seen, strip };
}

const section = (host: FakeElement): FakeElement => byClass(host, 'strip-harm')[0]!;
const rows = (host: FakeElement): FakeElement[] => byClass(host, 'strip-harm-row');
const visibleRows = (host: FakeElement): FakeElement[] => rows(host).filter((r) => !r.hidden);

// ------------------------------------------------------------------------------------- units

describe('total phase at a harmonic and the bank harmonics', () => {
  it('polarity + h·Δ + θ_h, wrapped to [0, 360) like the «Fase totale» readout', () => {
    expect(harmonicTotalPhaseDeg(true, 10, 3, 20)).toBeCloseTo(230, 12);
    expect(harmonicTotalPhaseDeg(true, 0, 1, 0)).toBe(totalPhaseDeg(true, 0));
    expect(harmonicTotalPhaseDeg(false, 30, 1, 0)).toBe(totalPhaseDeg(false, 30));
    // 180 − 400 − 50 = −270 → 90.
    expect(harmonicTotalPhaseDeg(true, -100, 4, -50)).toBeCloseTo(90, 12);
    expect(harmonicTotalPhaseDeg(false, 0, 2, -180)).toBe(180);
    expect(harmonicTotalPhaseDeg(true, 90, 8, 0)).toBe(180);
    expect(harmonicTotalPhaseDeg(true, Number.NaN, 2, Number.NaN)).toBe(180);
  });

  it('harmonics 1..n below 0.45·fs, at most MAX_HARMONICS', () => {
    expect(bankHarmonics(4, 100, 48000)).toEqual([1, 2, 3, 4]);
    expect(bankHarmonics(8, 2000, 44100)).toHaveLength(8);
    expect(bankHarmonics(8, 2000, 22050)).toEqual([1, 2, 3, 4]);
    expect(bankHarmonics(8, 2205, 44100)).toEqual([1, 2, 3, 4, 5, 6, 7, 8].filter((h) => h * 2205 < 0.45 * 44100));
    expect(bankHarmonics(20, 100, 48000)).toHaveLength(MAX_HARMONICS);
    expect(bankHarmonics(3, 0, 48000)).toEqual([]);
    expect(bankHarmonics(Number.NaN, 100, 48000)).toEqual([1]);
  });
});

// ------------------------------------------------------------------------------------- channel strip

describe('channel strip: «Armoniche» (tonal mode only)', () => {
  it('is hidden in delay and Hilbert modes and lists harmonics 1..N with their frequency in tonal mode', () => {
    const { host, strip } = mountStrip();
    const p = channel();
    strip.update(p, null, tonalInfo({ phaseMode: 'delay' }));
    expect(section(host).hidden).toBe(true);
    strip.update(p, null, tonalInfo({ phaseMode: 'hilbert' }));
    expect(section(host).hidden).toBe(true);
    strip.update(p, null, tonalInfo());
    expect(section(host).hidden).toBe(false);
    expect(section(host).tagName).toBe('DETAILS');
    expect(byClass(host, 'strip-harm')[0]!.all((e) => e.tagName === 'SUMMARY')[0]!.textContent).toBe('Armoniche');
    const vis = visibleRows(host);
    expect(vis).toHaveLength(4);
    expect(vis.map((r) => byClass(r, 'strip-harm-title')[0]!.textContent)).toEqual([
      'Fondamentale · 100,0 Hz',
      'Armonica 2 · 200,0 Hz',
      'Armonica 3 · 300,0 Hz',
      'Armonica 4 · 400,0 Hz',
    ]);
    // Each row is a named group; its controls say which harmonic they trim.
    const r2 = vis[1]!;
    expect(r2.getAttribute('role')).toBe('group');
    expect(r2.getAttribute('aria-labelledby')).toBe(byClass(r2, 'strip-harm-title')[0]!.id);
    expect(byClass(r2, 'pc-label').map((l) => l.textContent)).toEqual(['Guadagno armonica 2', 'Fase armonica 2']);
    expect(byClass(r2, 'pc-num')[0]!.getAttribute('aria-label')).toBe('Guadagno armonica 2, valore in dB');
    // More harmonics: more rows; f0 change: new frequencies.
    strip.update(p, null, tonalInfo({ harmonics: 6, phaseRefHz: 50 }));
    expect(visibleRows(host)).toHaveLength(6);
    expect(byClass(visibleRows(host)[5]!, 'strip-harm-title')[0]!.textContent).toBe('Armonica 6 · 300,0 Hz');
  });

  it('hides harmonics the bank skips (≥ 0.45·fs) and says so', () => {
    const { host, strip } = mountStrip();
    strip.update(channel(), null, tonalInfo({ phaseRefHz: 2000, harmonics: 8, fs: 22050 }));
    expect(visibleRows(host)).toHaveLength(4);
    const note = section(host).all((e) => e.tagName === 'P' && /45 %/.test(e.textContent))[0]!;
    expect(note.hidden).toBe(false);
    expect(note.textContent).toBe(
      'Le ultime 4 armoniche superano il 45 % della frequenza di campionamento (9923 Hz): il banco non le genera.',
    );
    strip.update(channel(), null, tonalInfo({ phaseRefHz: 2000, harmonics: 5, fs: 22050 }));
    expect(note.textContent).toMatch(/^L’ultima armonica supera il 45 %/);
    strip.update(channel(), null, tonalInfo());
    expect(note.hidden).toBe(true);
  });

  it('arrow keys trim gain and phase like the other controls; the strip emits a copy of all the trims', () => {
    const initial = channel({ harmonicTrims: trims({ 3: { gainDb: -2, phaseDeg: 15 } }) });
    const { host, seen, strip } = mountStrip(initial);
    strip.update(initial, null, tonalInfo());
    const [gain2, phase2] = byClass(visibleRows(host)[1]!, 'pc-slider');
    press(gain2!, 'ArrowUp');
    expect(seen).toHaveLength(1);
    expect(seen[0]!.changed).toBe('harmonicTrims');
    expect(seen[0]!.next.harmonicTrims).toHaveLength(MAX_HARMONICS);
    expect(seen[0]!.next.harmonicTrims[1]).toEqual({ gainDb: 0.01, phaseDeg: 0 });
    // The others are kept (harmonic 3's trim too).
    expect(seen[0]!.next.harmonicTrims[2]).toEqual({ gainDb: -2, phaseDeg: 15 });
    // A copy: neither the caller's array nor the strip's own.
    expect(seen[0]!.next.harmonicTrims).not.toBe(initial.harmonicTrims);
    seen[0]!.next.harmonicTrims[1]!.gainDb = 99;
    press(gain2!, 'ArrowUp', { shiftKey: true });
    expect(seen[1]!.next.harmonicTrims[1]!.gainDb).toBeCloseTo(0.11, 12);
    press(phase2!, 'ArrowDown');
    expect(seen[2]!.next.harmonicTrims[1]).toEqual({ gainDb: 0.11, phaseDeg: -0.1 });
    press(phase2!, 'PageUp');
    expect(seen[3]!.next.harmonicTrims[1]!.phaseDeg).toBeCloseTo(9.9, 12);
    // Home/End never jump a gain to its limit.
    press(gain2!, 'End');
    expect(seen).toHaveLength(4);
    // A typed value is clamped to the range (+6 dB).
    const num = byClass(visibleRows(host)[1]!, 'pc-num')[0]!;
    num.value = '+20';
    press(num, 'Enter');
    expect(seen[4]!.next.harmonicTrims[1]!.gainDb).toBe(6);
    expect(initial.harmonicTrims[1]).toEqual({ gainDb: 0, phaseDeg: 0 });
  });

  it('shows the total phase at each harmonic: polarity + h·Δ + θ_h', () => {
    const p = channel({ polarityInvert: true, phaseDeltaDeg: 10, harmonicTrims: trims({ 3: { phaseDeg: 20 } }) });
    const { host, strip } = mountStrip(p);
    strip.update(p, null, tonalInfo());
    const totals = visibleRows(host).map((r) => byClass(r, 'strip-harm-total')[0]!.textContent);
    expect(totals).toEqual(['Fase totale 190,0°', 'Fase totale 200,0°', 'Fase totale 230,0°', 'Fase totale 220,0°']);
    // The «Fase totale» readout at f0 is the same number.
    expect(byClass(host, 'phase-readout')[0]!.textContent).toBe('Fase totale: 190,0° @ 100,0 Hz');
  });

  it('the «Fase totale» readout at f0 includes the fundamental’s trim, like its row (one total phase per frequency)', () => {
    const p = channel({ polarityInvert: true, phaseDeltaDeg: 10, harmonicTrims: trims({ 1: { phaseDeg: 20 } }) });
    const { host, strip } = mountStrip(p);
    strip.update(p, null, tonalInfo());
    expect(byClass(visibleRows(host)[0]!, 'strip-harm-total')[0]!.textContent).toBe('Fase totale 210,0°');
    expect(byClass(host, 'phase-readout')[0]!.textContent).toBe('Fase totale: 210,0° @ 100,0 Hz');
    // a new θ_1 alone updates it; outside tonal mode the trims do not count
    const q = channel({ polarityInvert: true, phaseDeltaDeg: 10, harmonicTrims: trims({ 1: { phaseDeg: -30 } }) });
    strip.update(q, null, tonalInfo());
    expect(byClass(host, 'phase-readout')[0]!.textContent).toBe('Fase totale: 160,0° @ 100,0 Hz');
    strip.update(q, null, tonalInfo({ phaseMode: 'delay' }));
    expect(byClass(host, 'phase-readout')[0]!.textContent).toBe('Fase totale: 190,0° @ 100,0 Hz');
  });

  it('the help says the trims come on top of the delay («in aggiunta al ritardo»)', () => {
    const { host, strip } = mountStrip();
    strip.update(channel(), null, tonalInfo());
    const help = byClass(section(host), 'fp-help')[0]!.textContent;
    expect(help).toContain('in aggiunta al ritardo');
    expect(help).not.toContain('in più del');
  });

  it('double-click resets a trim to the calibrated value, or to no trim without a calibration', () => {
    const p = channel({ harmonicTrims: trims({ 2: { gainDb: 3, phaseDeg: 40 } }) });
    const { host, seen, strip } = mountStrip(p);
    strip.update(p, null, tonalInfo());
    const [gainLabel, phaseLabel] = byClass(visibleRows(host)[1]!, 'pc-label');
    gainLabel!.dispatchEvent(new FakeEvent('dblclick'));
    expect(seen.at(-1)!.next.harmonicTrims[1]).toEqual({ gainDb: 0, phaseDeg: 40 });
    const reset = byClass(visibleRows(host)[1]!, 'pc-reset')[0]!;
    expect(reset.title).toMatch(/nessuna correzione/);
    const cal = channel({ harmonicTrims: trims({ 2: { gainDb: -4.5, phaseDeg: -12 } }) });
    strip.update(p, null, tonalInfo({ calibrated: cal }));
    expect(reset.title).toMatch(/valore calibrato \(−4,50 dB\)/);
    gainLabel!.dispatchEvent(new FakeEvent('dblclick'));
    expect(seen.at(-1)!.next.harmonicTrims[1]).toEqual({ gainDb: -4.5, phaseDeg: 40 });
    phaseLabel!.dispatchEvent(new FakeEvent('dblclick'));
    expect(seen.at(-1)!.next.harmonicTrims[1]!.phaseDeg).toBe(-12);
  });

  it('«Azzera armoniche» clears every trim (also of harmonics not in use) and is off when there is nothing to clear', () => {
    const p = channel({ harmonicTrims: trims({ 6: { gainDb: -3 } }) });
    const { host, seen, strip } = mountStrip(p);
    strip.update(p, null, tonalInfo({ harmonics: 4 }));
    const btn = byClass(host, 'strip-harm-reset')[0]!;
    expect(btn.textContent).toBe('Azzera armoniche');
    expect(btn.getAttribute('aria-label')).toBe('Azzera le correzioni delle armoniche del canale L');
    // Harmonic 6 is not in use (4 harmonics): not counted, but still cleared.
    expect(byClass(host, 'strip-harm-count')[0]!.textContent).toBe('');
    expect(btn.disabled).toBe(false);
    btn.click();
    expect(seen.at(-1)!.changed).toBe('harmonicTrims');
    expect(seen.at(-1)!.next.harmonicTrims).toEqual(neutralTrims());
    strip.update(channel(), null, tonalInfo());
    expect(btn.disabled).toBe(true);
    const q = channel({ harmonicTrims: trims({ 1: { phaseDeg: 5 }, 3: { gainDb: 1 } }) });
    strip.update(q, null, tonalInfo());
    expect(byClass(host, 'strip-harm-count')[0]!.textContent).toBe(' · 2 corrette');
    expect(btn.disabled).toBe(false);
  });

  it('«Azzera armoniche» goes to onClearTrims (an absolute reset) when given; its label names both channels when linked', () => {
    const host = dom.mount();
    const seen: (keyof ChannelParams)[] = [];
    let cleared = 0;
    const strip = createChannelStrip(asEl(host), {
      channel: 'R',
      onChange: (_next, changed) => seen.push(changed),
      onLinkChange: () => {},
      onClearTrims: () => cleared++,
    });
    const p = channel({ harmonicTrims: trims({ 2: { gainDb: -6 } }) });
    strip.update(p, null, tonalInfo());
    const btn = byClass(host, 'strip-harm-reset')[0]!;
    expect(btn.getAttribute('aria-label')).toBe('Azzera le correzioni delle armoniche del canale R');
    btn.click();
    expect(cleared).toBe(1);
    expect(seen).toEqual([]);
    // shown at once, before the controller answers
    expect(byClass(visibleRows(host)[1]!, 'pc-num')[0]!.value).toBe('0,00');
    strip.update(p, null, tonalInfo({ linked: true }));
    expect(btn.getAttribute('aria-label')).toBe('Azzera le correzioni delle armoniche dei canali L e R (collegati)');
    // R-L5: linked, it clears the other channel too: enabled while either one has trims.
    const bare = channel({ harmonicTrims: trims({}) });
    strip.update(bare, null, tonalInfo({ linked: true, otherTrimmed: true }));
    expect(btn.disabled).toBe(false);
    btn.click();
    expect(cleared).toBe(2);
    strip.update(bare, null, tonalInfo({ linked: true, otherTrimmed: false }));
    expect(btn.disabled).toBe(true);
    // Unlinked, only this channel's trims count.
    strip.update(bare, null, tonalInfo({ linked: false, otherTrimmed: true }));
    expect(btn.disabled).toBe(true);
    expect(btn.getAttribute('aria-label')).toBe('Azzera le correzioni delle armoniche del canale R');
  });

  it('follows the controller, and is not rewritten at meter rate', () => {
    const p = channel({ harmonicTrims: trims({ 2: { gainDb: -1.25 } }) });
    const { host, strip } = mountStrip(p);
    const meters: MeterSnapshot = {
      ref: { rmsDb: -40, peakDb: -30 },
      err: null,
      outL: { rmsDb: -40, peakDb: -30 },
      outR: { rmsDb: -40, peakDb: -30 },
      limiterGrDb: [0, 0],
      armed: true,
      muted: false,
      bypass: false,
      softStart: 1,
      delaySamples: [5, 5],
      delayClamped: [false, false],
      blockSize: 128,
      ch2: 'absent',
      measuring: false,
      masterDb: -30,
    };
    strip.update(p, meters, tonalInfo());
    const num = byClass(visibleRows(host)[1]!, 'pc-num')[0]!;
    expect(num.value).toBe('−1,25');
    expect(num.textWrites).toBe(0);
    const titles = byClass(host, 'strip-harm-title');
    const writes = titles.map((t) => t.textWrites);
    const countWrites = byClass(host, 'strip-harm-count')[0]!.textWrites;
    for (let i = 0; i < 30; i++) strip.update({ ...p, harmonicTrims: copyTrims(p.harmonicTrims) }, { ...meters }, tonalInfo());
    expect(titles.map((t) => t.textWrites)).toEqual(writes);
    expect(byClass(host, 'strip-harm-count')[0]!.textWrites).toBe(countWrites);
    // A value from the controller (e.g. the wizard's solution) shows up.
    strip.update({ ...p, harmonicTrims: trims({ 2: { gainDb: 4 } }) }, meters, tonalInfo());
    expect(num.value).toBe('4,00');
  });
});

// ------------------------------------------------------------------------------------- scope

describe('scope: the expected trace leaves the harmonic trims out', () => {
  it('trimmedChannels: tonal mode and a trim on a harmonic in use', () => {
    const p = clampParams(DEFAULT_PARAMS);
    expect(trimmedChannels(p)).toEqual([false, false]);
    const t: EngineParams = clampParams({ ...p, channels: [{ ...p.channels[0], harmonicTrims: trims({ 2: { phaseDeg: 10 } }) }, p.channels[1]] });
    expect(trimmedChannels(t)).toEqual([false, false]);
    const tonal = clampParams({ ...t, phaseMode: 'tonal' });
    expect(trimmedChannels(tonal)).toEqual([true, false]);
    // A trim above the harmonics in use does nothing: no note.
    expect(trimmedChannels(clampParams({ ...tonal, tonal: { ...tonal.tonal, harmonics: 1 } }))).toEqual([false, false]);
  });

  it('the legend and a note say so for the channel shown', () => {
    // Canvas, observers and media queries: inert stand-ins (nothing is drawn here).
    const g = globalThis as Record<string, unknown>;
    const saved = { ro: g['ResizeObserver'], mo: g['MutationObserver'] };
    class Inert {
      observe(): void {}
      disconnect(): void {}
    }
    g['ResizeObserver'] = Inert;
    g['MutationObserver'] = Inert;
    Object.assign(window, { matchMedia: () => ({ addEventListener() {}, removeEventListener() {} }), devicePixelRatio: 1 });
    (FakeElement.prototype as unknown as { getContext: () => null }).getContext = () => null;
    const host = dom.mount();
    const scope = createScopeView(asEl(host), { fs: 48000 });
    const legendE = byClass(host, 'legend-e')[0]!;
    const note = byClass(host, 'scope-trim-note')[0]!;
    expect(legendE.textContent).toBe(EXPECTED_LEGEND);
    expect(note.hidden).toBe(true);
    scope.update({ trimmed: [false, true] });
    expect(legendE.textContent).toBe(EXPECTED_LEGEND);
    // The user picks R: its trims are noted.
    const sel = host.all((e) => e.tagName === 'SELECT')[0]!;
    sel.value = '1';
    sel.dispatchEvent(new FakeEvent('change'));
    expect(legendE.textContent).toBe(EXPECTED_LEGEND_TRIMMED);
    expect(note.hidden).toBe(false);
    expect(note.textContent).toBe(TRIMS_NOTE);
    expect(TRIMS_NOTE).toMatch(/non le include/);
    const w = note.textWrites;
    for (let i = 0; i < 10; i++) scope.update({ trimmed: [false, true] });
    expect(note.textWrites).toBe(w);
    scope.update({ trimmed: [false, false] });
    expect(legendE.textContent).toBe(EXPECTED_LEGEND);
    expect(note.hidden).toBe(true);
    scope.destroy();
    delete (FakeElement.prototype as unknown as { getContext?: unknown }).getContext;
    g['ResizeObserver'] = saved.ro;
    g['MutationObserver'] = saved.mo;
  });
});

// ------------------------------------------------------------------------------------- spectra

const FFT = 8192;
const FS = 48000;
const BIN = FS / FFT;

/** A dB spectrum: a floor plus tones (Hann main lobe: −6 dB on the neighbours) at the given levels. */
function spectrum(floorDb: number, tones: [number, number][]): Float32Array {
  const p = new Float64Array(FFT / 2 + 1).fill(Math.pow(10, floorDb / 10));
  for (const [f, db] of tones) {
    const k = Math.round(f / BIN);
    const total = Math.pow(10, db / 10);
    // Main lobe of a Hann window (power): 1/4, 1, 1/4 of the centre bin, normalised to the tone's power.
    for (const [d, w] of [[-1, 0.25], [0, 1], [1, 0.25]] as const) p[k + d]! += (total * w) / 1.5;
  }
  return Float32Array.from(p, (v) => 10 * Math.log10(v));
}

describe('per-harmonic attenuation in step 7', () => {
  it('the measured attenuation at each harmonic, next to what the calibrated values expect', () => {
    expect(harmonicMeasuredText(1, 100, 19.96, { db: 21.04, source: 'predicted' })).toBe('Fondamentale (100,0 Hz): +20,0 dB (prevista 21,0 dB)');
    expect(harmonicMeasuredText(2, 200, 12, { db: 13, source: 'measured' })).toBe('Armonica 2 (200,0 Hz): +12,0 dB (autotune 13,0 dB)');
    expect(harmonicMeasuredText(3, 300, -2.04, null)).toBe('Armonica 3 (300,0 Hz): −2,0 dB');
    expect(harmonicMeasuredText(2, 200, null, null)).toBe('Armonica 2 (200,0 Hz): —');
  });

  it('a prediction at its model’s ceiling is shown as that limit next to the measured attenuation', () => {
    const solution = (method: SolutionInfo['method']): Pick<SolutionInfo, 'method'> => ({ method });
    const predicted = (attenuationDb: number) => ({ calibratedHarmonics: [{ h: 1, freqHz: 100, attenuationDb, source: 'predicted' as const }] });
    const exact = expectedHarmonic(predicted(EXACT_NULL_DB), solution('error-mic'), 1);
    expect(exact).toEqual({ db: EXACT_NULL_DB, source: 'predicted', ceiling: 'exact' });
    expect(harmonicMeasuredText(1, 100, 41.04, exact)).toBe(
      'Fondamentale (100,0 Hz): +41,0 dB (prevista 60 dB, limite di calcolo per un nullo esatto del modello misurato)',
    );
    // A geometric solution's 30 dB is its accuracy limit; the same 30 dB from the error mic is a figure.
    const geo = expectedHarmonic(predicted(GEOMETRY_ACCURACY_DB), solution('two-position'), 1);
    expect(harmonicMeasuredText(1, 100, 22, geo)).toBe(
      'Fondamentale (100,0 Hz): +22,0 dB (prevista 30 dB, limite di precisione del modello senza mic d’errore)',
    );
    expect(expectedHarmonic(predicted(GEOMETRY_ACCURACY_DB), solution('error-mic'), 1)).toEqual({ db: GEOMETRY_ACCURACY_DB, source: 'predicted' });
    // The controller's calibrated prediction is judged with the solution's model; a measurement never is.
    const cal = predicted(30);
    expect(expectedHarmonic(cal, solution('geometry'), 1)!.ceiling).toBe('geometry');
    expect(expectedHarmonic({ calibratedHarmonics: [{ ...cal.calibratedHarmonics[0]!, attenuationDb: 60 }] }, null, 1)!.ceiling).toBe('exact');
    expect(expectedHarmonic({ calibratedHarmonics: [{ ...cal.calibratedHarmonics[0]!, attenuationDb: 60, source: 'measured' }] }, null, 1)).toEqual({
      db: 60,
      source: 'measured',
    });
  });

  it('expects the calibrated values’ attenuation only: never step 6’s prediction once the controller dropped it', () => {
    const solution: SolutionInfo = {
      method: 'error-mic',
      applied: true,
      predictedDb: 15,
      loopGainDb: null,
      causal: true,
      advice: [],
      geometryKey: null,
      harmonics: [
        { zone: null, h: 1, freqHz: 100, predictedDb: 21, untrimmedDb: 15, capped: false },
        { zone: null, h: 2, freqHz: 200, predictedDb: 17, untrimmedDb: 9, capped: false },
      ],
    };
    const cal = { calibratedHarmonics: [{ h: 1, freqHz: 100, attenuationDb: 18, source: 'measured' as const }] };
    expect(expectedHarmonic(cal, solution, 1)).toEqual({ db: 18, source: 'measured' });
    // R-L2: the calibrated harmonics are null after a capped autotune, a preset or setCalibratedChannels: the
    // solve's prediction (made for other trims) is no expectation any more.
    expect(expectedHarmonic(cal, solution, 2)).toBeNull();
    expect(expectedHarmonic({ calibratedHarmonics: null }, solution, 1)).toBeNull();
    expect(expectedHarmonic({}, solution, 1)).toBeNull();
  });

  it('drops the expectation once f0 moved off it or the channel differs at that harmonic (R-L3)', () => {
    const p = clampParams(DEFAULT_PARAMS);
    p.phaseMode = 'tonal';
    p.tonal = { f0: 100, harmonics: 4, q: 30 };
    p.channels[0].harmonicTrims[0] = { gainDb: 4, phaseDeg: 25 };
    const q = clampParams(p);
    const calibratedChannels = [q.channels[0], q.channels[1]] as const;
    const calibratedHarmonics = [
      { h: 1, freqHz: 100, attenuationDb: 28, source: 'predicted' as const },
      { h: 2, freqHz: 200, attenuationDb: 25, source: 'predicted' as const },
    ];
    const state = (over: (x: typeof q) => void) => {
      const params = clampParams(q);
      over(params);
      return { calibratedHarmonics, calibratedChannels, params: clampParams(params) };
    };
    const sol = { method: 'error-mic' as const };
    expect(expectedHarmonic(state(() => {}), sol, 1)).toEqual({ db: 28, source: 'predicted' });
    // The follower's drift within the bank's tolerance (min(f/2Q, 1 %·f)) keeps it…
    expect(expectedHarmonic(state((x) => (x.tonal.f0 = 100.8)), sol, 2)).toEqual({ db: 25, source: 'predicted' });
    // …«Aggancia al picco» to 150 Hz does not (the trims are per harmonic number: made for 100/200 Hz).
    expect(expectedHarmonic(state((x) => (x.tonal.f0 = 150)), sol, 1)).toBeNull();
    expect(expectedHarmonic(state((x) => (x.tonal.f0 = 150)), sol, 2)).toBeNull();
    // «Azzera armoniche»: harmonic 1's trim is gone, its 28 dB was for the trim; harmonic 2 had none.
    expect(expectedHarmonic(state((x) => (x.channels[0].harmonicTrims[0] = { gainDb: 0, phaseDeg: 0 })), sol, 1)).toBeNull();
    expect(expectedHarmonic(state((x) => (x.channels[0].harmonicTrims[0] = { gainDb: 0, phaseDeg: 0 })), sol, 2)).toEqual({ db: 25, source: 'predicted' });
    // The channel gain edited: every harmonic.
    expect(expectedHarmonic(state((x) => (x.channels[0].gainDb -= 3)), sol, 2)).toBeNull();
  });

  it('the controller gets the one-zone prediction, or the autotune’s measurement, with the calibrated values', () => {
    const one: HarmonicPrediction[] = [
      { zone: null, h: 1, freqHz: 100, predictedDb: 21, untrimmedDb: 15, capped: false },
      { zone: null, h: 3, freqHz: 300, predictedDb: 4, untrimmedDb: 3, capped: true },
    ];
    expect(calibratedPredictions(one)).toEqual([
      { h: 1, freqHz: 100, attenuationDb: 21, source: 'predicted' },
      { h: 3, freqHz: 300, attenuationDb: 4, source: 'predicted', capped: true },
    ]);
    expect(calibratedPredictions([])).toBeNull();
    expect(calibratedPredictions(one.map((x) => ({ ...x, zone: 1 as const })))).toBeNull();
    expect(calibratedMeasurements({ harmonics: [{ h: 2, freqHz: 200, startDb: -50, endDb: -60, bypassDb: -42 }] })).toEqual([
      { h: 2, freqHz: 200, attenuationDb: 18, source: 'measured' },
    ]);
    expect(calibratedMeasurements({})).toBeNull();
    // R-L2 (a): capped by up to LOOP_GUARD_TOL_DB (a result at the margin) the measurement still holds, at most
    // at what that gain error leaves; beyond it, it is of other params.
    const deep = { harmonics: [{ h: 1, freqHz: 100, startDb: -50, endDb: -100, bypassDb: -42 }] };
    expect(calibratedMeasurements(deep, 0.02)![0]!.attenuationDb).toBeCloseTo(gainLimitedAttenuationDb(0.02), 9);
    expect(calibratedMeasurements({ harmonics: [{ h: 1, freqHz: 100, startDb: -50, endDb: -60, bypassDb: -42 }] }, LOOP_GUARD_TOL_DB)![0]!.attenuationDb).toBe(18);
    expect(calibratedMeasurements(deep, LOOP_GUARD_TOL_DB + 0.01)).toBeNull();
  });
});

// ------------------------------------------------------------------------------------- step 6 helpers

const outcome = (useful: boolean, harmonics: SolveOutcome['harmonics']): SolveOutcome =>
  ({
    params: clampParams(DEFAULT_PARAMS),
    result: {
      channels: [
        { use: useful, delaySamples: 0, gainLin: 0.1, polarityInvert: true },
        { use: false, delaySamples: 0, gainLin: 0, polarityInvert: true },
      ],
      predictedAttenuationDb: useful ? 15 : 0,
      causal: true,
      requiredExtraLeadSeconds: 0,
    },
    advice: [],
    loopGainDb: null,
    ...(harmonics ? { harmonics } : {}),
  }) as SolveOutcome;

describe('per-harmonic prediction of the solution (step 6)', () => {
  const h = (n: number, predictedDb: number, untrimmedDb: number, capped = false) => ({ h: n, freqHz: n * 100, predictedDb, untrimmedDb, capped });

  it('one zone: the outcome’s harmonics; separate zones: each with its zone; useless zones left out', () => {
    expect(harmonicPredictions([outcome(true, [h(1, 22, 18), h(2, 19, 9)])])).toEqual([
      { zone: null, ...h(1, 22, 18) },
      { zone: null, ...h(2, 19, 9) },
    ]);
    const two = harmonicPredictions([outcome(true, [h(1, 20, 15)]), outcome(true, [h(1, 12, 11, true)])]);
    expect(two.map((x) => x.zone)).toEqual([0, 1]);
    expect(harmonicPredictions([outcome(false, [h(1, 3, 3)]), outcome(true, [h(1, 12, 11)])]).map((x) => x.zone)).toEqual([1]);
    // Outside tonal mode there are none.
    expect(harmonicPredictions([outcome(true, undefined)])).toEqual([]);
    expect(harmonicPredictions([null])).toEqual([]);
  });

  it('the merged loop limit caps them like the broadband prediction', () => {
    const [x] = harmonicPredictions([outcome(true, [h(1, 30, 25)]), outcome(true, [])], 3);
    // Both gains 3 dB low: at most ≈ 10,7 dB of attenuation is left.
    expect(x!.predictedDb).toBeCloseTo(10.7, 1);
    expect(x!.untrimmedDb).toBeCloseTo(10.7, 1);
  });

  it('texts', () => {
    const one: HarmonicPrediction = { zone: null, h: 2, freqHz: 200, predictedDb: 19.04, untrimmedDb: 9.0, capped: false };
    expect(harmonicPredictionText(one, true)).toBe('Armonica 2 (200,0 Hz): 19,0 dB (solo ritardo e guadagno: 9,0 dB)');
    expect(harmonicPredictionText(one, false)).toBe('Armonica 2 (200,0 Hz): 19,0 dB (solo ritardo e guadagno: 9,0 dB)');
    expect(harmonicPredictionText({ ...one, zone: 1, h: 1, freqHz: 100, capped: true }, true)).toBe(
      'Zona R · Fondamentale (100,0 Hz): 19,0 dB (solo ritardo e guadagno: 9,0 dB) · correzione al limite del campo: attenuazione parziale',
    );
  });

  it('a prediction at its model’s ceiling is worded as that limit, not as a figure', () => {
    // Without the error mic the solve's accuracy limit (30 dB) is no lower bound: it is how far the model reaches.
    const geo: HarmonicPrediction = { zone: null, h: 1, freqHz: 100, predictedDb: GEOMETRY_ACCURACY_DB, untrimmedDb: 12, capped: false };
    expect(harmonicPredictionText(geo, false)).toBe(
      'Fondamentale (100,0 Hz): 30 dB, limite di precisione del modello senza mic d’errore (solo ritardo e guadagno: 12,0 dB)',
    );
    // 0,5 dB below it still is the limit; further down, or with the error mic (a coherence of 0.999), a figure.
    expect(harmonicPredictionText({ ...geo, predictedDb: 29.6 }, false)).toMatch(/: 30 dB, limite di precisione del modello/);
    expect(harmonicPredictionText({ ...geo, predictedDb: 29.4 }, false)).toMatch(/: 29,4 dB \(solo/);
    expect(harmonicPredictionText(geo, true)).toMatch(/: 30,0 dB \(solo/);
    // With the error mic an exact null of the measured model stops at 60 dB: the calculation's limit.
    const exact: HarmonicPrediction = { ...geo, predictedDb: EXACT_NULL_DB };
    expect(harmonicPredictionText(exact, true)).toBe(
      'Fondamentale (100,0 Hz): 60 dB, limite di calcolo per un nullo esatto del modello misurato (solo ritardo e guadagno: 12,0 dB)',
    );
    expect(predictedDbText(59.2, true)).toBe('59,2 dB');
    expect(ceilingText('geometry')).toBe('30 dB, limite di precisione del modello senza mic d’errore');
    // Step 6's broadband line.
    expect(predictedOutcomeText(18.04, true)).toBe('Attenuazione prevista 18,0 dB');
    expect(predictedOutcomeText(18.04, false)).toBe('Attenuazione prevista 18,0 dB (stima: senza mic d’errore non è misurabile)');
    expect(predictedOutcomeText(GEOMETRY_ACCURACY_DB, false)).toBe(
      'Attenuazione prevista 30 dB, limite di precisione del modello senza mic d’errore (stima non misurabile)',
    );
    expect(predictedOutcomeText(EXACT_NULL_DB, true)).toBe('Attenuazione prevista 60 dB, limite di calcolo per un nullo esatto del modello misurato');
    expect(predictedOutcomeText(null, true)).toBe('Attenuazione prevista —');
    expect(predictedOutcomeText(null, false)).toBe('Attenuazione prevista — (stima: senza mic d’errore non è misurabile)');
    // Which model a solution's predictions come from.
    expect(solvedWithErrorMic({ method: 'error-mic' })).toBe(true);
    expect(solvedWithErrorMic({ method: 'two-position' })).toBe(false);
    expect(solvedWithErrorMic({ method: 'geometry' })).toBe(false);
    expect(solvedWithErrorMic(null)).toBe(true);
  });

  it('the per-harmonic autotune outcome: attenuation against the bypass, before and after', () => {
    expect(
      autotuneHarmonicTexts({
        harmonics: [
          { h: 1, freqHz: 100, startDb: -50, endDb: -58.04, bypassDb: -40 },
          { h: 3, freqHz: 300, startDb: -41, endDb: -46, bypassDb: -43 },
        ],
      }),
    ).toEqual(['Fondamentale (100,0 Hz): attenuazione +18,0 dB (prima +10,0 dB)', 'Armonica 3 (300,0 Hz): attenuazione +3,0 dB (prima −2,0 dB)']);
    expect(autotuneHarmonicTexts({})).toEqual([]);
    // «prima» against the bypass level at the start window (the ambient rose 6 dB during the run).
    expect(
      autotuneHarmonicTexts({ harmonics: [{ h: 1, freqHz: 100, startDb: -50, endDb: -52, bypassDb: -34, startBypassDb: -40 }] }),
    ).toEqual(['Fondamentale (100,0 Hz): attenuazione +18,0 dB (prima +10,0 dB)']);
    // Aborted: the last window is the probe that stopped the search, the start values are back: no list.
    expect(
      autotuneHarmonicTexts({ aborted: true, harmonics: [{ h: 1, freqHz: 100, startDb: -25, endDb: -0, bypassDb: -10 }] }),
    ).toEqual([]);
  });

  it('merging separate zones keeps each zone’s trims, as copies', () => {
    const base = clampParams(DEFAULT_PARAMS);
    const z0 = clampParams({ ...base, channels: [{ ...base.channels[0], harmonicTrims: trims({ 1: { gainDb: 2 } }) }, base.channels[1]] });
    const z1 = clampParams({ ...base, channels: [base.channels[0], { ...base.channels[1], harmonicTrims: trims({ 2: { phaseDeg: -30 } }) }] });
    const m = mergeZoneSolutions(base, [z0, z1]);
    expect(m.channels[0].harmonicTrims[0]).toEqual({ gainDb: 2, phaseDeg: 0 });
    expect(m.channels[1].harmonicTrims[1]).toEqual({ gainDb: 0, phaseDeg: -30 });
    expect(m.channels[0].harmonicTrims).not.toBe(z0.channels[0].harmonicTrims);
    expect(m.channels[1].harmonicTrims).not.toBe(z1.channels[1].harmonicTrims);
    const off = mergeZoneSolutions(base, [z0, null]);
    expect(off.channels[1].enabled).toBe(false);
    expect(off.channels[1].harmonicTrims).not.toBe(base.channels[1].harmonicTrims);
  });
});

// ------------------------------------------------------------------------------------- wizard steps

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

/** Just enough engine for a controller whose state the steps read. */
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

function envFor(controller: AppController, data = createWizardData({ hasSaved: false })): StepEnv {
  return {
    controller,
    data,
    analyzer: () => {
      throw new Error('not used');
    },
    startOp: () => ({ signal: new AbortController().signal, end() {} }),
    abortOp() {},
    busy: null,
    refresh() {},
    savePrefs() {},
    goTo() {},
    setMode() {},
    complete() {},
  };
}

describe('step 6 applies an autotune result', () => {
  const tonalStart = (): EngineParams =>
    clampParams({ ...DEFAULT_PARAMS, phaseMode: 'tonal', masterDb: -10, tonal: { f0: 100, harmonics: 2, q: 30 } });

  it('only its channels, over the live params: the follower’s f0 and the user’s master stay (R-L4)', () => {
    const c = new AppController({ engine: new QuietEngine(), devices, presetStorage: null, prefsStorage: null });
    c.applyParams(tonalStart());
    // runAutotune's copy of the start, its result's trims tuned…
    const start = c.currentParams;
    const tuned = clampParams(start);
    tuned.channels[0].harmonicTrims[0] = { gainDb: -2, phaseDeg: 70 };
    // …while the follower moved f0 4 % (in steps) and the user lowered the master.
    c.setTonal({ ...start.tonal, f0: 104 });
    c.setMaster(-25);
    const r = {
      params: tuned,
      aborted: false,
      improved: true,
      harmonics: [{ h: 1, freqHz: 100, startDb: -40, endDb: -60, bypassDb: -30, startBypassDb: -30 }],
    };
    const cal = createCalibration(FS, [20, 500], 'single');
    expect(applyAutotuneResult(c, cal, r, { fs: FS, taps: 256, separateZones: false })).toBe(true);
    const s = c.state;
    expect(s.params.tonal.f0).toBe(104);
    expect(s.params.masterDb).toBe(-25);
    expect(s.params.channels[0].harmonicTrims[0]).toEqual({ gainDb: -2, phaseDeg: 70 });
    expect(s.calibratedChannels![0].harmonicTrims[0]).toEqual({ gainDb: -2, phaseDeg: 70 });
    expect(s.calibratedHarmonics).toEqual([{ h: 1, freqHz: 100, attenuationDb: 30, source: 'measured' }]);
    // Separate zones: the channels, without a measurement; not improved or aborted: nothing.
    c.applyParams(tonalStart(), { calibrated: true });
    expect(applyAutotuneResult(c, cal, r, { fs: FS, taps: 256, separateZones: true })).toBe(true);
    expect(c.state.calibratedHarmonics).toBeNull();
    const before = c.currentParams;
    expect(applyAutotuneResult(c, cal, { ...r, improved: false }, { fs: FS, taps: 256, separateZones: false })).toBe(false);
    expect(applyAutotuneResult(c, cal, { ...r, aborted: true }, { fs: FS, taps: 256, separateZones: false })).toBe(false);
    expect(c.currentParams).toEqual(before);
  });
});

describe('wizard steps 6 and 7 in tonal mode', () => {
  it('step 6 lists the applied solution’s prediction harmonic by harmonic', () => {
    const c = new AppController({ engine: new QuietEngine(), devices, presetStorage: null, prefsStorage: null });
    const env = envFor(c);
    const view = createAlignStep(env);
    const root = view.element as unknown as FakeElement;
    view.update(c.state);
    const title = root.all((e) => e.tagName === 'H4' && e.textContent === 'Attenuazione prevista per armonica')[0]!;
    expect(title.hidden).toBe(true);
    env.data.solution = {
      method: 'error-mic',
      applied: true,
      predictedDb: 18,
      loopGainDb: null,
      causal: true,
      advice: [],
      geometryKey: null,
      harmonics: [
        { zone: null, h: 1, freqHz: 100, predictedDb: 24, untrimmedDb: 17, capped: false },
        { zone: null, h: 4, freqHz: 400, predictedDb: 16, untrimmedDb: 10, capped: false },
      ],
    };
    view.update(c.state);
    expect(title.hidden).toBe(false);
    const list = byClass(root, 'wz-harmonics')[0]!;
    expect(list.hidden).toBe(false);
    expect(list.children.map((li) => li.textContent)).toEqual([
      'Fondamentale (100,0 Hz): 24,0 dB (solo ritardo e guadagno: 17,0 dB)',
      'Armonica 4 (400,0 Hz): 16,0 dB (solo ritardo e guadagno: 10,0 dB)',
    ]);
    expect(root.textContent).toMatch(/sezione «Armoniche» dei canali/);
    // A geometric solution at its model's accuracy limit says so instead of promising 30 dB.
    env.data.solution = {
      ...env.data.solution,
      method: 'geometry',
      harmonics: [{ zone: null, h: 1, freqHz: 100, predictedDb: GEOMETRY_ACCURACY_DB, untrimmedDb: 17, capped: false }],
    };
    view.update(c.state);
    expect(list.children.map((li) => li.textContent)).toEqual([
      'Fondamentale (100,0 Hz): 30 dB, limite di precisione del modello senza mic d’errore (solo ritardo e guadagno: 17,0 dB)',
    ]);
    env.data.solution = null;
    view.update(c.state);
    expect(title.hidden).toBe(true);
    expect(list.hidden).toBe(true);
  });

  it('step 7 measures each harmonic on the error-mic spectra', () => {
    vi.useFakeTimers({ toFake: ['performance'] });
    vi.advanceTimersByTime(5000);
    const real = new AppController({ engine: new QuietEngine(), devices, presetStorage: null, prefsStorage: null });
    const base = real.state;
    const params = clampParams({ ...base.params, phaseMode: 'tonal', errorChannel: 1, tonal: { f0: 100, harmonics: 2, q: 30 } });
    // The calibrated values came with the solve's prediction for the fundamental (applyParams `harmonics`).
    const state: ControllerState = {
      ...base,
      params,
      ch2: 'ok',
      fs: FS,
      calibratedChannels: [params.channels[0], params.channels[1]],
      calibratedHarmonics: [{ h: 1, freqHz: 100, attenuationDb: 21, source: 'predicted' }],
    };
    const subs: ((s: ControllerState, change: string) => void)[] = [];
    const tones = (a: number, b: number): [number, number][] => [
      [100, a],
      [200, b],
    ];
    const spectra: SpectraSnapshot = {
      fs: FS,
      fftSize: FFT,
      ref: null,
      err: spectrum(-130, tones(-60, -52)),
      errBypass: spectrum(-130, tones(-40, -45)),
      refOn: spectrum(-130, tones(-30, -30)),
      refOff: spectrum(-130, tones(-30, -30)),
    };
    const fake = {
      state,
      spectra,
      onEvent: () => () => {},
      subscribe: (cb: (s: ControllerState, change: string) => void) => {
        subs.push(cb);
        return () => {};
      },
    } as unknown as AppController;
    const data = createWizardData({ hasSaved: false });
    data.solution = {
      method: 'error-mic',
      applied: true,
      predictedDb: 15,
      loopGainDb: null,
      causal: true,
      advice: [],
      geometryKey: null,
      harmonics: [{ zone: null, h: 1, freqHz: 100, predictedDb: 21, untrimmedDb: 15, capped: false }],
    };
    const view = createVerifyStep(envFor(fake, data));
    const root = view.element as unknown as FakeElement;
    for (const cb of subs) cb(state, 'spectrum');
    view.update(state);
    const list = byClass(root, 'wz-harmonics')[0]!;
    expect(list.children.map((li) => li.textContent)).toEqual(['Fondamentale (100,0 Hz): +20,0 dB (prevista 21,0 dB)', 'Armonica 2 (200,0 Hz): +7,0 dB']);
    expect(root.all((e) => e.tagName === 'H4' && e.textContent === 'Attenuazione per armonica')[0]!.hidden).toBe(false);
    // An exact null of the measured model: the calculation's limit, next to what the error mic measures.
    state.calibratedHarmonics = [{ h: 1, freqHz: 100, attenuationDb: EXACT_NULL_DB, source: 'predicted' }];
    vi.advanceTimersByTime(2000);
    for (const cb of subs) cb(state, 'spectrum');
    view.update(state);
    expect(list.children[0]!.textContent).toBe(
      'Fondamentale (100,0 Hz): +20,0 dB (prevista 60 dB, limite di calcolo per un nullo esatto del modello misurato)',
    );
    // Delay mode: no harmonic list.
    const delay: ControllerState = { ...state, params: { ...params, phaseMode: 'delay' } };
    vi.advanceTimersByTime(2000);
    for (const cb of subs) cb(delay, 'spectrum');
    view.update(delay);
    expect(list.hidden).toBe(true);
    view.leave();
  });
});
