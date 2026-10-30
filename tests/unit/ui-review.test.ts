/**
 * Console views fixed after the review, in node with the fake DOM: status chip and «Avvio» (ui-3,
 * ui-4), live regions not rewritten at meter rate (ui-8), preset list kept between identical updates
 * (ui-9), precision control units and double-click (ui-6, ui-10), drift sensitivity with the
 * resonators (calib-5), neutraliser taps (ui-11), per-band attenuation (plan-9), guide styles and
 * warning contrast (ui-12, ui-17).
 */
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { DEFAULT_FEEDBACK_TAPS, FEEDBACK_TAPS_RANGE } from '../../src/audio/calibration.ts';
import { clampParams, computeChannelDelay, DEFAULT_PARAMS, minTonalQ } from '../../src/dsp/params.ts';
import { channelResponse } from '../../src/dsp/response.ts';
import { createChannelStrip } from '../../src/ui/channel-strip.ts';
import { createDevicePanel } from '../../src/ui/device-panel.ts';
import type { DevicePanelData } from '../../src/ui/device-panel.ts';
import { FEEDBACK_TAP_CHOICES } from '../../src/ui/feedback-taps.ts';
import { createFiltersPanel, driftSensitivity, qHint } from '../../src/ui/filters-panel.ts';
import type { FiltersPanelData } from '../../src/ui/filters-panel.ts';
import { createPrecisionControl } from '../../src/ui/precision-control.ts';
import { createPresetsPanel } from '../../src/ui/presets-panel.ts';
import type { PresetsPanelData } from '../../src/ui/presets-panel.ts';
import { bandAttenuationDb } from '../../src/ui/spectrum-view.ts';
import { createStatusBar, startControl, statusOverride } from '../../src/ui/status-bar.ts';
import type { StatusBarData } from '../../src/ui/status-bar.ts';
import { delayUnits, parseNumberWithUnit, typedUnitFactor } from '../../src/ui/units.ts';
import { FakeElement, FakeEvent, installFakeDom } from './fake-dom.ts';
import type { FakeDom } from './fake-dom.ts';

let dom: FakeDom;
beforeEach(() => {
  dom = installFakeDom();
});
afterEach(() => dom.restore());

const asEl = (e: FakeElement): HTMLElement => e as unknown as HTMLElement;
const byClass = (root: FakeElement, cls: string): FakeElement => root.all((e) => e.classList.contains(cls))[0]!;

// ------------------------------------------------------------------------------------- status bar

describe('status chip and «Avvio» (ui-3, ui-4)', () => {
  const base = { panic: false, armed: true, muted: true, bypass: false, measuring: false };

  it('does not say «Muto» while a test stimulus plays, nor «Attivo» in bypass', () => {
    expect(statusOverride({ ...base, measuring: true })).toEqual({ statusText: 'Suono di prova in corso', statusState: 'measuring' });
    expect(statusOverride({ ...base, muted: false, bypass: true }).statusState).toBe('bypass');
    expect(statusOverride(base)).toEqual({});
    expect(statusOverride({ ...base, panic: true, measuring: true })).toEqual({});
    expect(statusOverride({ ...base, armed: false, measuring: true })).toEqual({});
  });

  it('«Avvio» arms whenever «Attiva uscita» could, and says the real blocker otherwise', () => {
    const ready = { armed: false, panic: false, canArm: true, micOpen: true, opening: false, measuring: false, problem: null, headphonesPending: false };
    // After «Stop» with the validation passed (it used to be disabled).
    expect(startControl(ready)).toEqual({ canStart: true, hint: '' });
    expect(startControl({ ...ready, armed: true, panic: true, canArm: false })).toEqual({ canStart: true, hint: '' });
    expect(startControl({ ...ready, micOpen: false, canArm: false }).hint).toMatch(/consenti il microfono/);
    expect(startControl({ ...ready, canArm: false, problem: 'Uscita Bluetooth.' }).hint).toBe('Uscita bloccata: Uscita Bluetooth.');
    expect(startControl({ ...ready, headphonesPending: true })).toEqual({ canStart: false, hint: expect.stringMatching(/cuffie/) as unknown as string });
    expect(startControl({ ...ready, measuring: true, canArm: false }).canStart).toBe(false);
  });
});

describe('status bar view', () => {
  const data: StatusBarData = {
    status: 'muted',
    running: true,
    canStart: false,
    muted: true,
    bypass: false,
    masterDb: 0,
    sampleRate: 48000,
    renderQuantum: 128,
    baseLatencyMs: 2,
    outputLatencyMs: 10,
    trackLatencyMs: 5,
    underruns: 0,
    message: null,
  };
  it('shows the measuring state, the start hint and a warnings badge; the live chip is rewritten only on change (ui-8)', () => {
    const host = dom.mount();
    const noop2 = (): void => {};
    const sb = createStatusBar(asEl(host), { onStartStop: noop2, onPanic: noop2, onMuteToggle: noop2, onBypassToggle: noop2, onMasterChange: noop2, themeToggle: false });
    const chip = byClass(host, 'sb-state');
    const badge = byClass(host, 'sb-warn-badge');
    const start = byClass(host, 'sb-start');
    sb.update({ ...data, ...statusOverride({ panic: false, armed: true, muted: true, bypass: false, measuring: true }), warnings: { count: 3, level: 'warn' } });
    expect(chip.textContent).toBe('Suono di prova in corso');
    expect(chip.dataset['state']).toBe('measuring');
    expect(badge.hidden).toBe(false);
    expect(badge.textContent).toBe('3 avvisi');
    expect(badge.href).toBe('#warnings');
    expect(badge.dataset['level']).toBe('warn');
    const before = chip.textWrites;
    for (let i = 0; i < 30; i++) sb.update({ ...data, statusText: 'Suono di prova in corso', statusState: 'measuring', warnings: { count: 3, level: 'warn' } });
    expect(chip.textWrites).toBe(before);
    sb.update({ ...data, running: false, canStart: false, startHint: 'Uscita bloccata: Uscita Bluetooth.', warnings: { count: 0, level: null } });
    expect(start.disabled).toBe(true);
    expect(start.title).toBe('Uscita bloccata: Uscita Bluetooth.');
    expect(badge.hidden).toBe(true);
    sb.update({ ...data, running: false, canStart: true });
    expect(start.disabled).toBe(false);
    expect(start.title).toBe('');
  });

  it('locks the master while a stimulus plays (a master change would abort it)', () => {
    const host = dom.mount();
    const masters: number[] = [];
    const noop2 = (): void => {};
    const sb = createStatusBar(asEl(host), {
      onStartStop: noop2,
      onPanic: noop2,
      onMuteToggle: noop2,
      onBypassToggle: noop2,
      onMasterChange: (db) => masters.push(db),
      themeToggle: false,
    });
    const box = byClass(host, 'sb-master');
    const field = box.all((e) => e.tagName === 'INPUT' && e.type !== 'range')[0]!;
    sb.update({ ...data, masterLocked: true });
    expect(field.disabled).toBe(true);
    expect(box.title).toMatch(/suono di prova/);
    sb.update({ ...data, masterLocked: false });
    expect(field.disabled).toBe(false);
    expect(box.title).toBe('');
    expect(masters).toEqual([]);
  });
});

// ------------------------------------------------------------------------------------- live regions

describe('live regions are not rewritten with the same text (ui-8)', () => {
  it('device panel: permission and input-2 lines', () => {
    const host = dom.mount();
    const noop = (): void => {};
    const panel = createDevicePanel(asEl(host), {
      onAllowMic: noop,
      onInputChange: noop,
      onOutputChange: noop,
      onRefChannelChange: noop,
      onErrorChannelChange: noop,
      onLatencyModeChange: noop,
    });
    const d: DevicePanelData = {
      permission: 'granted',
      inputs: [{ deviceId: 'a', label: 'USB' }],
      outputs: [{ deviceId: 'b', label: 'USB' }],
      inputId: 'a',
      outputId: 'b',
      sinkIdSupported: true,
      refChannel: 0,
      errorChannel: -1,
      latencyMode: 'normal',
      ch2: 'ok',
      trackChannels: 2,
    };
    panel.update(d);
    const perm = byClass(host, 'dp-perm-text');
    const ch2 = byClass(host, 'dp-ch2');
    const [p0, c0] = [perm.textWrites, ch2.textWrites];
    for (let i = 0; i < 30; i++) panel.update({ ...d });
    expect(perm.textWrites).toBe(p0);
    expect(ch2.textWrites).toBe(c0);
    panel.update({ ...d, ch2: 'silent' });
    expect(ch2.textWrites).toBe(c0 + 1);
  });

  it('channel strip: the clamped-delay warning', () => {
    const host = dom.mount();
    const p = clampParams(DEFAULT_PARAMS);
    const strip = createChannelStrip(asEl(host), { channel: 'L', initial: p.channels[0], onChange: () => {}, onLinkChange: () => {} });
    const level = { rmsDb: -40, peakDb: -30 };
    const meters = {
      ref: level,
      err: null,
      outL: level,
      outR: level,
      limiterGrDb: [0, 0] as [number, number],
      armed: true,
      muted: false,
      bypass: false,
      softStart: 1,
      delaySamples: [5, 5] as [number, number],
      delayClamped: [true, false] as [boolean, boolean],
      blockSize: 128,
      ch2: 'absent' as const,
      measuring: false,
    };
    const info = { fs: 48000, speedOfSound: 343, phaseRefHz: 100, calibrated: null, phaseMode: 'delay' as const, gainUnlocked: false, linked: false };
    strip.update(p.channels[0], meters, info);
    const warn = host.all((e) => e.getAttribute('role') === 'status' && e.textContent.startsWith('Ritardo limitato'))[0]!;
    const w0 = warn.textWrites;
    for (let i = 0; i < 30; i++) strip.update(p.channels[0], { ...meters }, info);
    expect(warn.textWrites).toBe(w0);
  });
});

// ------------------------------------------------------------------------------------- presets

describe('preset list (ui-9)', () => {
  it('is rebuilt only when what it shows changes', () => {
    const host = dom.mount();
    const noop = (): void => {};
    const panel = createPresetsPanel(asEl(host), { onSave: noop, onLoad: noop, onDelete: noop, onExport: noop, onImport: noop });
    const d: PresetsPanelData = {
      presets: [
        { id: 'a', name: 'Sera', savedAt: 1_700_000_000_000, detail: 'USB' },
        { id: 'b', name: 'Mattina', savedAt: 1_700_000_100_000 },
      ],
      activeId: 'a',
      message: null,
    };
    panel.update(d);
    const list = byClass(host, 'pp-list');
    const first = list.children[0];
    const load = list.all((e) => e.dataset['key'] === 'load:b')[0]!;
    load.focus();
    for (let i = 0; i < 20; i++) panel.update({ ...d, presets: d.presets.map((p) => ({ ...p })), message: { level: 'info', text: `n ${i}` } });
    expect(list.children[0]).toBe(first);
    expect(dom.document.activeElement).toBe(load);
    expect(byClass(host, 'pp-message').textContent).toBe('n 19');
    panel.update({ ...d, activeId: 'b' });
    expect(list.children[0]).not.toBe(first);
    // The focused action is restored on a real rebuild.
    expect(dom.document.activeElement.dataset['key']).toBe('load:b');
  });
});

// ------------------------------------------------------------------------------------- precision control

describe('precision control', () => {
  const press = (e: FakeElement, key: string): void => void e.dispatchEvent(new FakeEvent('keydown', { key }));

  it('double-click resets on the label, never on the slider (ui-6)', () => {
    const host = dom.mount();
    const seen: number[] = [];
    createPrecisionControl(asEl(host), { label: 'Δ fase fine', value: 30, min: -180, max: 180, step: 0.1, unit: '°', resetValue: 0, onChange: (v) => seen.push(v) });
    const slider = byClass(host, 'pc-slider');
    const label = byClass(host, 'pc-label');
    expect(slider.hasListener('dblclick')).toBe(false);
    label.dispatchEvent(new FakeEvent('dblclick'));
    expect(seen).toEqual([0]);
    expect(byClass(host, 'pc-reset').title).toMatch(/^Doppio click sul nome o ↺/);
  });

  it('converts a typed unit, and rejects one the control does not have (ui-10)', () => {
    const host = dom.mount();
    const seen: number[] = [];
    createPrecisionControl(asEl(host), {
      label: 'Ritardo',
      value: 1,
      min: 0,
      max: 500,
      step: 0.001,
      units: delayUnits(48000, 343),
      unitId: 'cm',
      onChange: (v) => seen.push(v),
    });
    const num = byClass(host, 'pc-num');
    num.value = '5 ms';
    press(num, 'Enter');
    expect(seen).toEqual([5]);
    num.value = '96 campioni';
    press(num, 'Enter');
    expect(seen[1]).toBeCloseTo(2, 12);
    num.value = '5 kg';
    press(num, 'Enter');
    expect(seen).toHaveLength(2);
    expect(num.getAttribute('aria-invalid')).toBe('true');
    // No unit: the one shown by the select (cm).
    num.value = '34,3';
    press(num, 'Enter');
    expect(seen[2]).toBeCloseTo(1, 12);
  });

  it('accepts the spellings of a fixed unit', () => {
    const host = dom.mount();
    const seen: number[] = [];
    createPrecisionControl(asEl(host), { label: 'Limite', value: -12, min: -24, max: -1, step: 0.5, unit: 'dBFS', onChange: (v) => seen.push(v) });
    const num = byClass(host, 'pc-num');
    for (const t of ['−6 dB', '-7dBFS', '-8', '-9 Hz']) {
      num.value = t;
      press(num, 'Enter');
    }
    expect(seen).toEqual([-6, -7, -8]);
  });
});

describe('typed units (ui-10)', () => {
  it('parses the unit text after the number', () => {
    expect(parseNumberWithUnit('5 ms')).toEqual({ value: 5, unit: 'ms' });
    expect(parseNumberWithUnit('−3,5dB')).toEqual({ value: -3.5, unit: 'db' });
    expect(parseNumberWithUnit('12 μs')).toEqual({ value: 12, unit: 'µs' });
    expect(parseNumberWithUnit('12 us')).toEqual({ value: 12, unit: 'µs' });
    expect(parseNumberWithUnit('-∞ dB')).toEqual({ value: -Infinity, unit: 'db' });
    expect(parseNumberWithUnit('7')).toEqual({ value: 7, unit: '' });
    expect(parseNumberWithUnit('abc')).toBeNull();
  });

  it('maps it to the factor of one of the control units, or refuses it', () => {
    const u = delayUnits(48000, 343);
    expect(typedUnitFactor('', u)).toBeNull();
    expect(typedUnitFactor('ms', u)).toBe(1);
    expect(typedUnitFactor('µs', u)).toBe(1000);
    expect(typedUnitFactor('campioni', u)).toBe(48);
    expect(typedUnitFactor('cm', u)).toBeCloseTo(34.3, 12);
    expect(typedUnitFactor('db', u)).toBeUndefined();
    expect(typedUnitFactor('khz', [], 'Hz')).toBe(0.001);
    expect(typedUnitFactor('deg', [], '°')).toBe(1);
    expect(typedUnitFactor('x', [], undefined)).toBeUndefined();
  });
});

// ------------------------------------------------------------------------------------- filters panel

describe('tonal drift sensitivity includes the resonator bank (calib-5)', () => {
  it('matches the phase change of the real tonal channel for a 0.1 Hz drift', () => {
    const fs = 48000;
    const p = clampParams(undefined);
    p.masterDb = 0;
    p.phaseMode = 'tonal';
    p.tonal = { f0: 100, harmonics: 4, q: 30 };
    p.channels[0].delayMs = 4;
    const tapS = computeChannelDelay(p, fs, 0).samples / fs;
    const sens = driftSensitivity(p.filters, p.tonal, fs, tapS);
    expect(sens.degPerHz).toHaveLength(4);
    const df = 0.1;
    for (const h of [1, 4]) {
      const a = channelResponse(p, fs, h * 100, 0);
      const b = channelResponse(p, fs, h * (100 + df), 0);
      let lag = -((Math.atan2(b.im, b.re) - Math.atan2(a.im, a.re)) * 180) / Math.PI;
      lag -= 360 * Math.round(lag / 360);
      expect(Math.abs(sens.degPerHz[h - 1]! * df - lag)).toBeLessThan(0.1 * Math.abs(lag));
    }
    // ≈ 34°/Hz from Q 30 at 100 Hz alone: ±0,17 Hz for −20 dB, not ±0,5.
    expect(sens.tolHz).toBeLessThan(0.2);
  });

  it('shows it in the panel, with the neutraliser taps under «Avanzate» (ui-11)', () => {
    const host = dom.mount();
    const taps: number[] = [];
    const panel = createFiltersPanel(asEl(host), {
      onPhaseModeChange: () => {},
      onFRefChange: () => {},
      onTonalChange: () => {},
      onFiltersChange: () => {},
      onSnapToPeak: () => {},
      onFeedbackTapsChange: (n) => taps.push(n),
    });
    const p = clampParams(DEFAULT_PARAMS);
    const d: FiltersPanelData = {
      phaseMode: 'tonal',
      fRef: p.fRef,
      tonal: { f0: 100, harmonics: 4, q: 30 },
      filters: p.filters,
      fs: 48000,
      f0Estimate: { f0: 100.01, confidence: 0.9, dominanceDb: 20 },
      driftAlign: { seconds: 0.03, bound: false },
      feedbackTaps: 256,
      feedbackActive: true,
    };
    panel.update(d);
    expect(host.textContent).toMatch(/risonatori inclusi/);
    expect(host.textContent).toMatch(/Per −20 dB f0 deve essere giusta entro ±0,\d{3} Hz\./);
    // Without a calibration the degrees are an upper bound and the tolerance a lower one (calib-ui-8).
    panel.update({ ...d, driftAlign: { seconds: 0.03, bound: true } });
    expect(host.textContent).toMatch(/1 Hz di errore su f0 = al più \d+,\d° all’armonica 1 · al più \d+,\d° all’armonica 4/);
    expect(host.textContent).toMatch(/Per −20 dB basta che f0 sia giusta entro ±0,\d{3} Hz \(stima prudente: /);
    expect(host.textContent).not.toMatch(/\(al più:/);
    // Fewer taps than the solve's: the controller lowers the gains (v-safety S4, calib-ui-3), the help says so.
    expect(host.textContent).toMatch(/Con meno tap il guadagno d’anello sale: se supera il margine di stabilità, i guadagni dei canali vengono ridotti da soli\./);
    panel.update(d);
    const radios = host.all((e) => e.tagName === 'INPUT' && e.type === 'radio' && e.name.endsWith('-taps'));
    expect(radios.map((r) => Number(r.value))).toEqual([...FEEDBACK_TAP_CHOICES]);
    expect(radios.find((r) => r.checked)?.value).toBe('256');
    const r512 = radios.find((r) => r.value === '512')!;
    r512.checked = true;
    r512.dispatchEvent(new FakeEvent('change'));
    expect(taps).toEqual([512]);
    // The f0 readout is not a live region rewritten every update.
    const est = host.all((e) => e.tagName === 'P' && e.textContent.startsWith('Stima f0'))[0]!;
    expect(est.getAttribute('aria-live')).toBe('off');
    const w = est.textWrites;
    for (let i = 0; i < 10; i++) panel.update({ ...d });
    expect(est.textWrites).toBe(w);
  });

  it('the resonator Q cannot go below the minimum of the number of harmonics (dsp-4)', () => {
    const host = dom.mount();
    const tonal: Array<{ harmonics: number; q: number }> = [];
    const panel = createFiltersPanel(asEl(host), {
      onPhaseModeChange: () => {},
      onFRefChange: () => {},
      onTonalChange: (t) => tonal.push({ harmonics: t.harmonics, q: t.q }),
      onFiltersChange: () => {},
      onSnapToPeak: () => {},
    });
    const p = clampParams(DEFAULT_PARAMS);
    const d: FiltersPanelData = { phaseMode: 'tonal', fRef: p.fRef, tonal: { f0: 100, harmonics: 4, q: 2 }, filters: p.filters, fs: 48000 };
    panel.update(d);
    const qField = (): FakeElement =>
      host.all((e) => e.tagName === 'INPUT' && e.getAttribute('aria-label') === 'Q dei risonatori, valore')[0]!;
    expect(host.textContent).not.toMatch(/Q minimo/);
    // 8 harmonics from the controller (clampParams already raised Q to 6).
    panel.update({ ...d, tonal: { f0: 100, harmonics: 8, q: minTonalQ(8) } });
    expect(host.textContent).toMatch(/Con 8 armoniche il Q minimo è 6: sotto, le bande si sovrappongono troppo/);
    // Typing a lower Q is clamped to the minimum.
    const f = qField();
    f.value = '3';
    f.dispatchEvent(new FakeEvent('blur'));
    expect(parseNumberWithUnit(f.value)?.value).toBe(6);
    expect(tonal.every((t) => t.q >= minTonalQ(t.harmonics))).toBe(true);
    // Raising the harmonics from the panel sends a Q already at the new minimum.
    panel.update({ ...d, tonal: { f0: 100, harmonics: 4, q: 2 } });
    const h = host.all((e) => e.tagName === 'INPUT' && e.getAttribute('aria-label') === 'Armoniche, valore')[0]!;
    h.value = '7';
    h.dispatchEvent(new FakeEvent('blur'));
    expect(tonal.at(-1)).toEqual({ harmonics: 7, q: minTonalQ(7) });
    expect(parseNumberWithUnit(qField().value)?.value).toBe(minTonalQ(7));
    expect(qHint(4)).not.toMatch(/Q minimo/);
    expect(qHint(5)).toMatch(/Con 5 armoniche il Q minimo è 3/);
    expect(qHint(6)).toMatch(/Con 6 armoniche il Q minimo è 4/);
  });

  it('offers tap counts the controller accepts, the default among them', () => {
    expect(FEEDBACK_TAP_CHOICES).toContain(DEFAULT_FEEDBACK_TAPS);
    for (const n of FEEDBACK_TAP_CHOICES) {
      expect(n).toBeGreaterThanOrEqual(FEEDBACK_TAPS_RANGE.min);
      expect(n).toBeLessThanOrEqual(FEEDBACK_TAPS_RANGE.max);
    }
  });
});

// ------------------------------------------------------------------------------------- attenuation per band

describe('per-band attenuation normalised by the reference (plan-9)', () => {
  const fs = 48000;
  const n = 8192;
  const flat = (db: number): Float32Array => new Float32Array(n / 2 + 1).fill(db);

  it('a source 6 dB quieter since the bypass is not read as 6 dB of attenuation', () => {
    const a = bandAttenuationDb({ errOff: flat(-40), refOff: flat(-30), errOn: flat(-46), refOn: flat(-36) }, fs, n, 100);
    expect(a).toBeCloseTo(0, 9);
    const b = bandAttenuationDb({ errOff: flat(-40), refOff: flat(-30), errOn: flat(-56), refOn: flat(-30) }, fs, n, 100);
    expect(b).toBeCloseTo(16, 9);
    expect(bandAttenuationDb({ errOff: null, refOff: flat(-30), errOn: flat(-56), refOn: flat(-30) }, fs, n, 100)).toBeNull();
  });
});

// ------------------------------------------------------------------------------------- styles

describe('styles (ui-12, ui-17)', () => {
  const css = readFileSync(new URL('../../src/styles.css', import.meta.url), 'utf8');

  it('styles the in-app guide', () => {
    for (const sel of ['.guide-panel', '.guide-section', '.guide-table', '.guide-note-warning', '.guide-header']) expect(css).toContain(sel);
  });

  it('light-theme warning text reaches 4.5:1 on its background', () => {
    const light = css.slice(css.indexOf(':root {'), css.indexOf('@media (prefers-color-scheme: dark)'));
    const hex = (name: string): string => new RegExp(`${name}:\\s*(#[0-9a-f]{6})`, 'i').exec(light)![1]!;
    const lum = (h: string): number => {
      const c = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
      return 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;
    };
    const ratio = (a: string, b: string): number => (Math.max(lum(a), lum(b)) + 0.05) / (Math.min(lum(a), lum(b)) + 0.05);
    expect(ratio(hex('--warn'), hex('--warn-bg'))).toBeGreaterThanOrEqual(4.5);
  });
});
