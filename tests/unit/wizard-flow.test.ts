import { describe, expect, it } from 'vitest';
import { calibrationFreqs, createCalibration } from '../../src/audio/calibration.ts';
import type { PathData } from '../../src/audio/calibration.ts';
import { deviceKey } from '../../src/audio/devices.ts';
import type { DeviceOption } from '../../src/audio/devices.ts';
import type { ValidationResult } from '../../src/app/validation.ts';
import { DEFAULT_PARAMS, clampParams } from '../../src/dsp/params.ts';
import {
  ampKey,
  amplifierReleveled,
  bannerFor,
  checkPassed,
  createWizardData,
  flowSteps,
  nextStep,
  prevStep,
  REF_CHANGED_TEXT,
  refInputChanged,
  relockReason,
  resetMeasurements,
  reuseDeviceMatch,
  stalePaths,
  stepGate,
  unmuteBlockers,
} from '../../src/ui/wizard/flow.ts';
import type { GateState, StepId, WizardData } from '../../src/ui/wizard/flow.ts';
import { activeChannels, geometryKey, positionsKey } from '../../src/ui/wizard/logic.ts';
import { errorMicNote } from '../../src/ui/wizard/step-layout.ts';

const FS = 48000;
const okValidation = (): ValidationResult => ({ ok: true, issues: [], headphones: false, sameDevice: true });

const dev = (deviceId: string, kind: DeviceOption['kind'], label: string): DeviceOption => ({ deviceId, kind, label, groupId: `g-${label}`, pseudo: false });
const USB_IN = dev('usb-in', 'audioinput', 'Scarlett 2i2 USB');
const USB_OUT = dev('usb-out', 'audiooutput', 'Scarlett 2i2 USB');
const UMC_IN = dev('umc-in', 'audioinput', 'UMC202HD');
const UMC_OUT = dev('umc-out', 'audiooutput', 'UMC202HD');
/** The controller's device key for an input and output of these lists (as presets store it). */
const keyOf = (inputId: string, outputId: string | null): string =>
  deviceKey(
    [USB_IN, UMC_IN].find((d) => d.deviceId === inputId) ?? null,
    [USB_OUT, UMC_OUT].find((d) => d.deviceId === (outputId ?? 'default')) ?? null,
    [USB_IN, UMC_IN, USB_OUT, UMC_OUT],
  );

/** `inputId`: the open input, from which the device key is derived (unless given). */
function gate(over: Partial<GateState> & { inputId?: string } = {}): GateState {
  const { inputId = 'usb-in', ...rest } = over;
  const s: GateState = {
    micOpen: true,
    opening: false,
    armed: true,
    panic: false,
    validation: okValidation(),
    headphonesConfirmed: false,
    outputId: 'usb-out',
    fs: FS,
    ch2: 'absent',
    measuring: false,
    capturing: false,
    params: clampParams(DEFAULT_PARAMS),
    speakers: 'single',
    deviceKey: '',
    ...rest,
  };
  return rest.deviceKey !== undefined ? s : { ...s, deviceKey: s.micOpen ? keyOf(inputId, s.outputId) : '' };
}

/** Acoustic-looking path (low-frequency roll-off, not flat) at `delay` samples. */
function acousticPath(delay: number, band: [number, number] = [20, 1000]): PathData {
  const freqs = [...calibrationFreqs(FS, band)];
  const re: number[] = [];
  const im: number[] = [];
  for (const f of freqs) {
    const mag = (f / 200) ** 2 / (1 + (f / 200) ** 2);
    const a = (-2 * Math.PI * f * delay) / FS;
    re.push(mag * Math.cos(a));
    im.push(mag * Math.sin(a));
  }
  return { delaySamples: delay, sign: 1, qualityDb: 30, stdSamples: 0.1, freqs, re, im, ir: [], irOffset: 0 };
}

/** Electrical loopback: flat |H| and a pure delay. */
function loopbackPath(delay: number, band: [number, number] = [20, 1000]): PathData {
  const freqs = [...calibrationFreqs(FS, band)];
  return {
    delaySamples: delay,
    sign: 1,
    qualityDb: 40,
    stdSamples: 0.05,
    freqs,
    re: freqs.map((f) => 0.5 * Math.cos((-2 * Math.PI * f * delay) / FS)),
    im: freqs.map((f) => 0.5 * Math.sin((-2 * Math.PI * f * delay) / FS)),
    ir: [],
    irOffset: 0,
  };
}

/** Wizard data with everything done for a single speaker without error mic. */
function doneData(s: GateState): WizardData {
  const d = createWizardData({ hasSaved: false });
  d.amp.checklist = [true, true, true];
  d.amp.playedS = [10, 10];
  d.amp.leveled = [true, true];
  d.amp.key = [ampKey(s), ampKey(s)];
  d.geometry = { micToZoneM: '6', speakerToZoneM: ['1,2', '1,5'], sourceAngleDeg: '0', tempC: '20' };
  const cal = createCalibration(FS, [20, 1000], s.speakers);
  cal.feedback = [acousticPath(1300), acousticPath(1350)];
  d.calibration = cal;
  d.measure.feedback = [
    { state: 'ok', reason: null, path: cal.feedback[0], posKey: positionsKey(d.geometry, 0) },
    { state: 'ok', reason: null, path: cal.feedback[1], posKey: positionsKey(d.geometry, 1) },
  ];
  d.solution = {
    method: 'geometry',
    applied: true,
    predictedDb: 12,
    loopGainDb: -30,
    causal: true,
    advice: [],
    geometryKey: geometryKey(d.geometry, activeChannels(s.speakers)),
  };
  return d;
}

/** Saved calibration reused on `s`'s devices, quick check passed. */
function reuseData(s: GateState): WizardData {
  const d = doneData(s);
  resetMeasurements(d);
  d.mode = 'reuse';
  d.calibration = createCalibration(FS, [20, 1000], s.speakers);
  d.calibration.feedback = [acousticPath(1300), acousticPath(1350)];
  d.solution = { method: 'preset', applied: true, predictedDb: null, loopGainDb: null, causal: null, advice: [], geometryKey: null };
  d.reuse = { presetId: 'p1', name: 'Sera', deviceKey: deviceKey(USB_IN, USB_OUT, [USB_IN, USB_OUT]), masterDb: -12 };
  d.check = { key: ampKey(s), channels: [{ state: 'ok', text: '' }, { state: 'corrected', text: '' }], acceptUnverified: false };
  return d;
}

describe('flowSteps', () => {
  it('runs all seven steps on the first configuration', () => {
    expect(flowSteps('full', false)).toEqual(['devices', 'amp', 'layout', 'placement', 'measure', 'align', 'verify']);
  });

  it('asks the start question first when a calibration is saved', () => {
    expect(flowSteps('full', true)[0]).toBe('start');
    expect(flowSteps('recal', true)).toEqual(['start', 'devices', 'amp', 'placement', 'measure', 'align', 'verify']);
    expect(flowSteps('reuse', true)).toEqual(['start', 'devices', 'amp', 'check', 'verify']);
  });

  it('never skips the safety steps: devices and amplifier level always come before the verification', () => {
    for (const mode of ['full', 'recal', 'reuse'] as const) {
      for (const saved of [false, true]) {
        const s = flowSteps(mode, saved);
        expect(s.indexOf('devices')).toBeGreaterThanOrEqual(0);
        expect(s.indexOf('amp')).toBeGreaterThan(s.indexOf('devices'));
        expect(s.indexOf('verify')).toBe(s.length - 1);
        expect(s.indexOf('amp')).toBeLessThan(s.indexOf('verify'));
      }
    }
  });

  it('moves forward and back within the list', () => {
    const s = flowSteps('reuse', true);
    expect(nextStep(s, 'start')).toBe('devices');
    expect(nextStep(s, 'amp')).toBe('check');
    expect(nextStep(s, 'check')).toBe('verify');
    expect(nextStep(s, 'verify')).toBeNull();
    expect(prevStep(s, 'devices')).toBe('start');
    expect(prevStep(s, 'start')).toBeNull();
    expect(nextStep(s, 'layout')).toBeNull();
  });
});

describe('stepGate', () => {
  it('devices: needs the mic, validation, headphones confirmation and the output armed', () => {
    const d = createWizardData({ hasSaved: false });
    expect(stepGate('devices', gate({ micOpen: false, armed: false }), d).ok).toBe(false);
    const failing: ValidationResult = {
      ok: false,
      issues: [{ id: 'bluetooth-output', level: 'block', text: 'Uscita Bluetooth.', overridable: false }],
      headphones: false,
      sameDevice: false,
    };
    const g = stepGate('devices', gate({ validation: failing, armed: false }), d);
    expect(g.ok).toBe(false);
    expect(g.reason).toContain('Uscita Bluetooth.');
    const hp = { ...okValidation(), headphones: true };
    expect(stepGate('devices', gate({ validation: hp }), d).reason).toContain('cuffie');
    expect(stepGate('devices', gate({ validation: hp, headphonesConfirmed: true }), d).ok).toBe(true);
    expect(stepGate('devices', gate({ armed: false }), d).reason).toContain('Abilita');
    expect(stepGate('devices', gate({ panic: true }), d).ok).toBe(false);
    expect(stepGate('devices', gate(), d)).toEqual({ ok: true, reason: null });
  });

  it('amp: needs the checklist and channel L leveled on the current output and sample rate', () => {
    const s = gate();
    const d = createWizardData({ hasSaved: false });
    expect(stepGate('amp', s, d).ok).toBe(false);
    d.amp.checklist = [true, true, true];
    expect(stepGate('amp', s, d).ok).toBe(false);
    d.amp.leveled[0] = true;
    d.amp.key[0] = ampKey({ outputId: 'other', fs: FS });
    expect(stepGate('amp', s, d).ok).toBe(false);
    d.amp.key[0] = ampKey(s);
    expect(stepGate('amp', s, d).ok).toBe(true);
    // A different sample rate invalidates the level.
    expect(stepGate('amp', gate({ fs: 44100 }), d).ok).toBe(false);
    // The output must still be armed.
    expect(stepGate('amp', gate({ armed: false }), d).ok).toBe(false);
    // First configuration: R is checked at the layout step, once the speakers are chosen.
    expect(stepGate('amp', gate({ speakers: 'shared' }), d).ok).toBe(true);
    // Saved configuration with two speakers (no layout step): R is needed here.
    d.mode = 'reuse';
    expect(stepGate('amp', gate({ speakers: 'shared' }), d).reason).toContain('canale R');
    expect(stepGate('amp', gate({ speakers: 'single' }), d).ok).toBe(true);
  });

  it('layout: two speakers need channel R leveled; the error mic needs a working input 2', () => {
    const s = gate({ speakers: 'shared' });
    const d = doneData(s);
    d.amp.leveled[1] = false;
    expect(stepGate('layout', s, d).reason).toContain('canale R');
    expect(stepGate('layout', gate({ speakers: 'single' }), d).ok).toBe(true);
    d.amp.leveled[1] = true;
    expect(stepGate('layout', s, d).ok).toBe(true);
    const withErr = { ...clampParams(DEFAULT_PARAMS), errorChannel: 1 as const };
    expect(stepGate('layout', gate({ speakers: 'shared', params: withErr, ch2: 'silent' }), d).ok).toBe(false);
    expect(stepGate('layout', gate({ speakers: 'shared', params: withErr, ch2: 'silent' }), d).reason).toContain('sull’ingresso 2 ');
    expect(stepGate('layout', gate({ speakers: 'shared', params: withErr, ch2: 'ok' }), d).ok).toBe(true);
    // Reference on input 2: the engine checks input 1 for the error mic.
    const swapped = { ...clampParams(DEFAULT_PARAMS), refChannel: 1 as const, errorChannel: 0 as const };
    expect(stepGate('layout', gate({ speakers: 'shared', params: swapped, ch2: 'silent' }), d).reason).toContain('sull’ingresso 1 ');
    expect(stepGate('layout', gate({ speakers: 'shared', params: swapped, ch2: 'pending' }), d).reason).toContain('dell’ingresso 1 ');
    expect(errorMicNote('silent', 0)).toMatch(/^L’ingresso 1 è muto/);
    expect(errorMicNote('duplicate', 0)).toBe('L’ingresso 1 duplica l’ingresso 2: il mic d’errore non è utilizzabile.');
    expect(errorMicNote('ok', 1)).toBe('Ingresso 2 attivo: il mic d’errore è utilizzabile.');
  });

  it('placement: needs valid distances for the active channels', () => {
    const d = createWizardData({ hasSaved: false });
    const g = stepGate('placement', gate(), d);
    expect(g.ok).toBe(false);
    expect(g.reason).toMatch(/microfono/);
    d.geometry = { micToZoneM: '6,5', speakerToZoneM: ['1', ''], sourceAngleDeg: '10', tempC: '18' };
    expect(stepGate('placement', gate(), d).ok).toBe(true);
    expect(stepGate('placement', gate({ speakers: 'separate' }), d).ok).toBe(false);
  });

  it('measure: feedback per active channel (or accepted failure), secondary with the error mic, loopback cleared', () => {
    const s = gate({ speakers: 'shared' });
    const d = doneData(s);
    expect(stepGate('measure', s, d).ok).toBe(true);
    d.measure.feedback[1] = { state: 'failed', reason: 'Nessun segnale', path: null, posKey: positionsKey(d.geometry, 1) };
    expect(stepGate('measure', s, d).reason).toContain('cassa R');
    d.measure.acceptWeakFeedback[1] = true;
    expect(stepGate('measure', s, d).ok).toBe(true);
    const withErr = { ...clampParams(DEFAULT_PARAMS), errorChannel: 1 as const };
    expect(stepGate('measure', gate({ speakers: 'shared', params: withErr }), d).reason).toContain('mic d’errore');
    d.measure.loopback = 'measured';
    expect(stepGate('measure', s, d).reason).toContain('loopback');
    d.measure.loopback = 'verified';
    expect(stepGate('measure', gate({ speakers: 'shared', fs: 44100 }), d).ok).toBe(false);
  });

  it('align: needs an applied solution; any running measurement blocks every step', () => {
    const s = gate();
    const d = doneData(s);
    expect(stepGate('align', s, d).ok).toBe(true);
    d.solution = { ...d.solution!, applied: false };
    expect(stepGate('align', s, d).ok).toBe(false);
    const all: StepId[] = ['start', 'devices', 'amp', 'layout', 'placement', 'measure', 'align', 'verify'];
    for (const step of all) expect(stepGate(step, gate({ measuring: true }), doneData(s)).ok).toBe(false);
  });

  it('start: needs an answer', () => {
    const d = createWizardData({ hasSaved: true });
    expect(stepGate('start', gate(), d).ok).toBe(false);
    d.modeChosen = true;
    expect(stepGate('start', gate(), d).ok).toBe(true);
  });

  it('resetMeasurements clears paths and the solution', () => {
    const s = gate();
    const d = doneData(s);
    resetMeasurements(d);
    expect(d.calibration).toBeNull();
    expect(d.solution).toBeNull();
    expect(d.measure.feedback[0].state).toBe('idle');
  });
});

describe('unmuteBlockers', () => {
  it('is empty when every safety condition holds', () => {
    const s = gate({ speakers: 'shared' });
    expect(unmuteBlockers(s, doneData(s))).toEqual([]);
  });

  it('requires the amplifier level of every active channel in this session', () => {
    const s = gate({ speakers: 'shared' });
    const d = doneData(s);
    d.amp.leveled[1] = false;
    expect(unmuteBlockers(s, d).join(' ')).toContain('canale R');
    // Another output device: the levels no longer count.
    expect(unmuteBlockers(gate({ speakers: 'single', outputId: 'other' }), doneData(s)).length).toBeGreaterThan(0);
  });

  it('requires an applied solution, an armed validated output and no PANIC', () => {
    const s = gate();
    const d = doneData(s);
    d.solution = null;
    expect(unmuteBlockers(s, d).join(' ')).toContain('passo 6');
    expect(unmuteBlockers(gate({ armed: false }), doneData(s)).join(' ')).toContain('passo 1');
    expect(unmuteBlockers(gate({ panic: true }), doneData(s)).join(' ')).toContain('PANIC');
    expect(unmuteBlockers(gate({ measuring: true }), doneData(s)).length).toBeGreaterThan(0);
  });

  it('refuses a calibration made at another sample rate', () => {
    const s = gate();
    expect(unmuteBlockers(gate({ fs: 44100 }), doneData(s)).join(' ')).toMatch(/kHz/);
  });

  it('refuses while a loopback cable may be connected or a path has the loopback signature', () => {
    const s = gate();
    const d = doneData(s);
    d.measure.loopback = 'connect';
    expect(unmuteBlockers(s, d).join(' ')).toContain('loopback');
    const d2 = doneData(s);
    d2.calibration!.feedback[0] = loopbackPath(200);
    expect(unmuteBlockers(s, d2).join(' ')).toContain('loopback');
  });
});

describe('distances edited after measuring and aligning (ui-1)', () => {
  it('measurements and a geometry solution only count for the distances they were made with', () => {
    const s = gate({ speakers: 'shared' });
    const d = doneData(s);
    expect(stepGate('measure', s, d).ok).toBe(true);
    expect(stepGate('align', s, d).ok).toBe(true);
    // The mic is moved 20 m farther upstream (step 4 edited after step 6).
    d.geometry.micToZoneM = '26';
    expect(stalePaths(s, d)).toHaveLength(2);
    expect(stepGate('measure', s, d).reason).toMatch(/Hai cambiato le distanze/);
    expect(stepGate('align', s, d).reason).toMatch(/Calcola e applica/);
    const blockers = unmuteBlockers(s, d).join(' ');
    expect(blockers).toMatch(/passo 5/);
    expect(blockers).toMatch(/passo 6/);
    // Back to the same numbers (a typo corrected): everything counts again.
    d.geometry.micToZoneM = '6,0';
    expect(unmuteBlockers(s, d)).toEqual([]);
  });

  it('angle or temperature only invalidate a geometry-based solution, a speaker distance only its channel', () => {
    const s = gate({ speakers: 'shared' });
    const d = doneData(s);
    d.geometry.tempC = '30';
    expect(stalePaths(s, d)).toEqual([]);
    expect(stepGate('align', s, d).ok).toBe(false);
    d.solution = { ...d.solution!, method: 'error-mic', geometryKey: null };
    expect(stepGate('align', s, d).ok).toBe(true);
    d.geometry.speakerToZoneM[1] = '2';
    expect(stalePaths(s, d)).toEqual(['cassa R → microfono di riferimento']);
  });
});

describe('reuse of a saved calibration (calib-9, plan-3, plan-4)', () => {
  it('needs the quick check of every active channel, on the current output and rate', () => {
    const s = gate({ speakers: 'shared' });
    const d = reuseData(s);
    expect(checkPassed(s, d)).toBe(true);
    expect(stepGate('check', s, d).ok).toBe(true);
    expect(unmuteBlockers(s, d)).toEqual([]);
    d.check.channels[1] = { state: 'mismatch', text: 'Rifai le misure.' };
    expect(stepGate('check', s, d).reason).toContain('Rifai le misure.');
    expect(unmuteBlockers(s, d).join(' ')).toMatch(/Controllo rapido/);
    d.check.channels[1] = { state: 'unverified', text: '' };
    expect(stepGate('check', s, d).ok).toBe(false);
    d.check.acceptUnverified = true;
    expect(stepGate('check', s, d).ok).toBe(true);
    // Checked on another output: does not count.
    expect(checkPassed(gate({ speakers: 'shared', outputId: 'umc-out' }), d)).toBe(false);
    const fresh = reuseData(s);
    fresh.check.key = null;
    expect(unmuteBlockers(s, fresh).join(' ')).toMatch(/Controllo rapido/);
  });

  it('blocks a calibration measured with another interface, after the microphone is open', () => {
    const s = gate();
    const d = reuseData(s);
    expect(reuseDeviceMatch(d, s)).toBe('same');
    const other = gate({ inputId: 'umc-in', outputId: 'umc-out' });
    expect(reuseDeviceMatch(d, other)).toBe('different');
    expect(stepGate('devices', other, d).reason).toMatch(/altra interfaccia/);
    expect(unmuteBlockers(other, d).join(' ')).toMatch(/altra interfaccia/);
    // Unknown before the microphone opens (the start question loads the preset first).
    expect(reuseDeviceMatch(d, gate({ micOpen: false }))).toBe('unknown');
    // Outside the reuse flow the key is not compared.
    d.mode = 'recal';
    expect(reuseDeviceMatch(d, other)).toBe('same');
  });
});

describe('relockReason (safety-9)', () => {
  it('locks a completed setup again when the output, the rate or the interfaces change', () => {
    const s = gate();
    const d = doneData(s);
    d.doneDeviceKey = deviceKey(USB_IN, USB_OUT, [USB_IN, UMC_IN, USB_OUT, UMC_OUT]);
    expect(relockReason(s, d)).toBeNull();
    expect(relockReason(gate({ outputId: 'umc-out' }), d)).toMatch(/livello dell’amplificatore/);
    expect(relockReason(gate({ fs: 44100 }), d)).not.toBeNull();
    expect(relockReason(gate({ inputId: 'umc-in' }), d)).not.toBeNull();
    // Nothing to compare while the mic is closed or reopening (no false lock).
    expect(relockReason(gate({ micOpen: false, fs: null }), d)).toBeNull();
    expect(relockReason(gate({ opening: true, outputId: 'umc-out' }), d)).toBeNull();
  });
});

describe('amplifier levelled again at step 2 (v-safety S3, calib-ui-1)', () => {
  it('reuse: the quick check must be repeated; the saved solution stays', () => {
    const s = gate({ speakers: 'shared' });
    const d = reuseData(s);
    expect(unmuteBlockers(s, d)).toEqual([]);
    amplifierReleveled(d, 1);
    expect(checkPassed(s, d)).toBe(false);
    expect(stepGate('check', s, d).ok).toBe(false);
    expect(unmuteBlockers(s, d).join(' ')).toMatch(/Controllo rapido/);
    expect(d.solution?.applied).toBe(true);
    expect(d.calibration?.feedback[1]).not.toBeNull();
  });

  it('full flow: the paths of that channel and the solution go; the other channel keeps its paths', () => {
    const s = gate({ speakers: 'shared' });
    const d = doneData(s);
    expect(unmuteBlockers(s, d)).toEqual([]);
    amplifierReleveled(d, 1);
    expect(d.measure.feedback[1].state).toBe('idle');
    expect(d.calibration!.feedback[1]).toBeNull();
    expect(d.measure.feedback[0].state).toBe('ok');
    expect(d.solution).toBeNull();
    expect(stepGate('measure', s, d).reason).toMatch(/cassa R → microfono di riferimento/);
    expect(unmuteBlockers(s, d).join(' ')).toMatch(/Nessun allineamento/);
  });

  it('before anything was measured it changes nothing', () => {
    const d = createWizardData({ hasSaved: false });
    amplifierReleveled(d, 0);
    expect(d.measure.feedback[0].state).toBe('idle');
    expect(d.solution).toBeNull();
  });
});

describe('reference input changed after the measurements (v-safety S2)', () => {
  const withRef = (s: GateState, ref: 0 | 1): GateState => ({ ...s, params: clampParams({ ...s.params, refChannel: ref }) });

  it('full flow: measured paths and the applied solution stop counting, and step 7 cannot un-mute', () => {
    const s = gate();
    const d = doneData(s);
    d.calibration!.refChannel = 0;
    expect(refInputChanged(s, d)).toBe(false);
    expect(unmuteBlockers(s, d)).toEqual([]);
    const moved = withRef(s, 1);
    expect(refInputChanged(moved, d)).toBe(true);
    expect(unmuteBlockers(moved, d)).toContain(REF_CHANGED_TEXT);
    expect(stepGate('measure', moved, d).reason).toBe(REF_CHANGED_TEXT);
    expect(stepGate('align', moved, d).reason).toBe(REF_CHANGED_TEXT);
    // Back on the input it was measured with: valid again.
    expect(unmuteBlockers(withRef(moved, 0), d)).toEqual([]);
  });

  it('reuse: the quick check made on the other input does not count', () => {
    const s = gate({ speakers: 'shared' });
    const d = reuseData(s);
    d.calibration!.refChannel = 0;
    expect(stepGate('check', s, d).ok).toBe(true);
    const moved = withRef(s, 1);
    expect(stepGate('check', moved, d).reason).toBe(REF_CHANGED_TEXT);
    expect(unmuteBlockers(moved, d)).toContain(REF_CHANGED_TEXT);
  });

  it('after completion the console locks the anti-noise again', () => {
    const s = gate();
    const d = doneData(s);
    d.calibration!.refChannel = 0;
    d.doneDeviceKey = s.deviceKey;
    expect(relockReason(s, d)).toBeNull();
    expect(relockReason(withRef(s, 1), d)).toMatch(/Canale del riferimento/);
    // Even before the rate is known again (the switch re-opens nothing, but no false "valid").
    expect(relockReason({ ...withRef(s, 1), fs: null }, d)).not.toBeNull();
  });

  it('calibrations saved without the input are not compared', () => {
    const s = gate();
    const d = doneData(s);
    delete d.calibration!.refChannel;
    expect(refInputChanged(withRef(s, 1), d)).toBe(false);
  });
});

describe('bannerFor', () => {
  it('offers to resume after a PANIC on every step', () => {
    for (const step of ['devices', 'measure'] as const) {
      expect(bannerFor(step, { ...gate({ panic: true }), canArm: true })?.action).toBe('resume');
    }
  });

  it('asks to re-arm a disarmed output, or to go back to the devices', () => {
    expect(bannerFor('measure', { ...gate({ armed: false }), canArm: true })?.action).toBe('arm');
    expect(bannerFor('measure', { ...gate({ armed: false }), canArm: false })?.action).toBe('devices');
    expect(bannerFor('align', { ...gate({ micOpen: false, armed: false }), canArm: false })?.action).toBe('devices');
    expect(bannerFor('devices', { ...gate({ armed: false }), canArm: true })).toBeNull();
    expect(bannerFor('measure', { ...gate(), canArm: true })).toBeNull();
  });
});
