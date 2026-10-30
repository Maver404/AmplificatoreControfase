import { describe, expect, it } from 'vitest';
import { ISSUE, errorMicInput, validateSetup, type Issue, type SetupInfo } from '../../src/app/validation.ts';
import type { DeviceOption } from '../../src/audio/devices.ts';

const dev = (label: string, kind: 'audioinput' | 'audiooutput', groupId = 'g-scarlett', deviceId = `${kind}-${label}`): DeviceOption => ({
  deviceId,
  groupId,
  label,
  kind,
  pseudo: false,
});

const SCARLETT_IN = dev('Scarlett 2i2 USB (1235:8210)', 'audioinput');
const SCARLETT_OUT = dev('Scarlett 2i2 USB (1235:8210)', 'audiooutput');

/** A setup that passes with no issues at all. */
function good(over: Partial<SetupInfo> = {}): SetupInfo {
  return {
    input: SCARLETT_IN,
    output: SCARLETT_OUT,
    trackSettings: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      sampleRate: 48000,
      channelCount: 2,
    },
    ctxSampleRate: 48000,
    outputLatencyS: 0.012,
    sinkSupported: true,
    chromium: true,
    platform: 'mac',
    wantErrorMic: false,
    ch2Status: 'pending',
    ...over,
  };
}

const find = (issues: Issue[], id: string): Issue | undefined => issues.find((i) => i.id === id);
const ids = (issues: Issue[]): string[] => issues.map((i) => i.id);

describe('validateSetup', () => {
  it('accepts a correct single-interface setup with no issues', () => {
    const r = validateSetup(good());
    expect(r).toEqual({ ok: true, issues: [], headphones: false, sameDevice: true });
  });

  it('every issue text is non-empty Italian prose', () => {
    const r = validateSetup({
      input: null,
      output: null,
      trackSettings: { echoCancellation: true },
      ctxSampleRate: null,
      outputLatencyS: null,
      sinkSupported: false,
      chromium: false,
      platform: 'windows',
      wantErrorMic: true,
      ch2Status: 'absent',
    });
    for (const i of r.issues) expect(i.text.length).toBeGreaterThan(20);
  });

  describe('blocking rules (never overridable)', () => {
    const allOverrides = new Set(Object.values(ISSUE));

    it('no input', () => {
      const r = validateSetup(good({ input: null }), allOverrides);
      const i = find(r.issues, ISSUE.noInput)!;
      expect(i).toMatchObject({ level: 'block', overridable: false });
      expect(i.overridden).toBeUndefined();
      expect(r.ok).toBe(false);
      expect(r.sameDevice).toBe(false);
    });

    it('Bluetooth output', () => {
      const out = dev('AirPods Pro', 'audiooutput', 'g-airpods');
      const r = validateSetup(good({ output: out }), allOverrides);
      expect(find(r.issues, ISSUE.bluetoothOutput)).toMatchObject({ level: 'block', overridable: false });
      expect(find(r.issues, ISSUE.bluetoothOutput)!.text).toContain('AirPods Pro');
      expect(r.ok).toBe(false);
      expect(r.headphones).toBe(true);
    });

    it('Bluetooth output behind a "Predefinito" pseudo device', () => {
      const out: DeviceOption = {
        deviceId: 'default',
        groupId: 'g-bt',
        label: 'Predefinito - Headphones (WH-1000XM4 Stereo)',
        kind: 'audiooutput',
        pseudo: true,
      };
      const r = validateSetup(good({ output: out }));
      expect(find(r.issues, ISSUE.bluetoothOutput)!.text).toContain('«Headphones (WH-1000XM4 Stereo)»');
      expect(r.ok).toBe(false);
    });

    it('AirPlay output', () => {
      const r = validateSetup(good({ output: dev('AirPlay', 'audiooutput', 'g-air') }), allOverrides);
      expect(find(r.issues, ISSUE.airplayOutput)).toMatchObject({ level: 'block', overridable: false });
      expect(r.ok).toBe(false);
    });

    it('Bluetooth input', () => {
      const r = validateSetup(good({ input: dev('AirPods di Mario', 'audioinput', 'g-airpods') }), allOverrides);
      expect(find(r.issues, ISSUE.bluetoothInput)).toMatchObject({ level: 'block', overridable: false });
      expect(r.ok).toBe(false);
    });

    it('built-in mic with built-in speakers', () => {
      const r = validateSetup(
        good({
          input: dev('MacBook Pro Microphone (Built-in)', 'audioinput', 'g-mac-in'),
          output: dev('MacBook Pro Speakers (Built-in)', 'audiooutput', 'g-mac-out'),
        }),
        allOverrides,
      );
      expect(find(r.issues, ISSUE.builtinPair)).toMatchObject({ level: 'block', overridable: false });
      expect(r.ok).toBe(false);
    });

    it('built-in mic alone or built-in speakers alone are not blocked', () => {
      const a = validateSetup(good({ input: dev('Microfono MacBook Air', 'audioinput', 'g-mac') }));
      expect(find(a.issues, ISSUE.builtinPair)).toBeUndefined();
      const b = validateSetup(good({ output: dev('Altoparlanti (Realtek(R) Audio)', 'audiooutput', 'g-rt') }));
      expect(find(b.issues, ISSUE.builtinPair)).toBeUndefined();
      // Built-in mic + headphone jack: no acoustic loop through the speakers.
      const c = validateSetup(
        good({
          input: dev('Microfono (Realtek(R) Audio)', 'audioinput', 'g-rt'),
          output: dev('Cuffie (Realtek(R) Audio)', 'audiooutput', 'g-rt'),
        }),
      );
      expect(find(c.issues, ISSUE.builtinPair)).toBeUndefined();
      expect(c.headphones).toBe(true);
    });

    it.each([
      ['echoCancellation', "cancellazione dell'eco"],
      ['noiseSuppression', 'soppressione del rumore'],
      ['autoGainControl', 'controllo automatico del guadagno'],
    ] as const)('%s still active', (key, words) => {
      const info = good();
      info.trackSettings = { ...info.trackSettings, [key]: true };
      const r = validateSetup(info, allOverrides);
      const i = find(r.issues, ISSUE.trackProcessing)!;
      expect(i).toMatchObject({ level: 'block', overridable: false });
      expect(i.text).toContain(words);
      expect(r.ok).toBe(false);
    });

    it('lists every active processing in one issue', () => {
      const r = validateSetup(
        good({ trackSettings: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }),
      );
      expect(r.issues.filter((i) => i.id === ISSUE.trackProcessing)).toHaveLength(1);
      expect(find(r.issues, ISSUE.trackProcessing)!.text).toMatch(/eco.*rumore.*guadagno/);
    });
  });

  describe('overridable errors', () => {
    it('output latency above 40 ms', () => {
      const r = validateSetup(good({ outputLatencyS: 0.0456 }));
      const i = find(r.issues, ISSUE.outputLatency)!;
      expect(i).toMatchObject({ level: 'error', overridable: true });
      expect(i.text).toContain('45,6 ms');
      expect(r.ok).toBe(false);
      const o = validateSetup(good({ outputLatencyS: 0.0456 }), new Set([ISSUE.outputLatency]));
      expect(o.ok).toBe(true);
      expect(find(o.issues, ISSUE.outputLatency)!.overridden).toBe(true);
    });

    it('exactly 40 ms is accepted', () => {
      expect(find(validateSetup(good({ outputLatencyS: 0.04 })).issues, ISSUE.outputLatency)).toBeUndefined();
    });

    it('unknown output latency is only information', () => {
      for (const v of [null, 0, Number.NaN]) {
        const r = validateSetup(good({ outputLatencyS: v }));
        expect(find(r.issues, ISSUE.outputLatencyUnknown)?.level).toBe('info');
        expect(r.ok).toBe(true);
      }
    });

    it('track sample rate different from the context', () => {
      const info = good({ ctxSampleRate: 48000 });
      info.trackSettings = { ...info.trackSettings, sampleRate: 44100 };
      const r = validateSetup(info);
      const i = find(r.issues, ISSUE.sampleRateMismatch)!;
      expect(i).toMatchObject({ level: 'error', overridable: true });
      expect(i.text).toContain('44,1 kHz');
      expect(i.text).toContain('48 kHz');
      expect(r.ok).toBe(false);
      expect(validateSetup(info, new Set([ISSUE.sampleRateMismatch])).ok).toBe(true);
    });

    it('missing sample rates are not a mismatch', () => {
      const info = good({ ctxSampleRate: null });
      expect(find(validateSetup(info).issues, ISSUE.sampleRateMismatch)).toBeUndefined();
      const info2 = good();
      info2.trackSettings = { echoCancellation: false };
      expect(find(validateSetup(info2).issues, ISSUE.sampleRateMismatch)).toBeUndefined();
    });

    it('each error needs its own override', () => {
      const info = good({ outputLatencyS: 0.08, ctxSampleRate: 44100 });
      expect(validateSetup(info, new Set([ISSUE.outputLatency])).ok).toBe(false);
      expect(validateSetup(info, new Set([ISSUE.sampleRateMismatch])).ok).toBe(false);
      expect(validateSetup(info, new Set([ISSUE.outputLatency, ISSUE.sampleRateMismatch])).ok).toBe(true);
    });

    it('an override never unlocks a block', () => {
      const info = good({ outputLatencyS: 0.08, output: dev('AirPlay', 'audiooutput', 'g-air') });
      const r = validateSetup(info, new Set([ISSUE.outputLatency, ISSUE.airplayOutput]));
      expect(r.ok).toBe(false);
      expect(find(r.issues, ISSUE.airplayOutput)!.overridden).toBeUndefined();
    });
  });

  describe('error mic', () => {
    it.each(['absent', 'silent', 'duplicate'] as const)('%s channel 2 → non-overridable error', (status) => {
      const r = validateSetup(good({ wantErrorMic: true, ch2Status: status }), new Set([ISSUE.errorMicUnavailable]));
      const i = find(r.issues, ISSUE.errorMicUnavailable)!;
      expect(i).toMatchObject({ level: 'error', overridable: false });
      expect(i.overridden).toBeUndefined();
      expect(i.text).toContain("Mic d'errore disattivato");
      expect(r.ok).toBe(false);
    });

    it('names the input the engine checks (input 1 when the reference is on input 2)', () => {
      const on2 = validateSetup(good({ wantErrorMic: true, errorChannel: 1, ch2Status: 'silent' }));
      expect(find(on2.issues, ISSUE.errorMicUnavailable)!.text).toMatch(/l'ingresso 2 è muto\. Collega il microfono d'errore all'ingresso 2 /);
      const on1 = validateSetup(good({ wantErrorMic: true, errorChannel: 0, ch2Status: 'silent' }));
      expect(find(on1.issues, ISSUE.errorMicUnavailable)!.text).toMatch(/l'ingresso 1 è muto\. Collega il microfono d'errore all'ingresso 1 /);
      const dup = validateSetup(good({ wantErrorMic: true, errorChannel: 0, ch2Status: 'duplicate' }));
      expect(find(dup.issues, ISSUE.errorMicUnavailable)!.text).toContain("l'ingresso 1 riceve lo stesso segnale dell'ingresso 2");
      const pending = validateSetup(good({ wantErrorMic: true, errorChannel: 0, ch2Status: 'pending' }));
      expect(find(pending.issues, ISSUE.errorMicPending)!.text).toBe("Verifica dell'ingresso 1 (mic d'errore) in corso…");
      expect(errorMicInput(undefined)).toBe(2);
      expect(errorMicInput(-1)).toBe(2);
    });

    it('turning the option off clears the error', () => {
      const r = validateSetup(good({ wantErrorMic: false, ch2Status: 'absent' }));
      expect(r.ok).toBe(true);
      expect(r.issues).toEqual([]);
    });

    it('ok and pending channel 2', () => {
      expect(validateSetup(good({ wantErrorMic: true, ch2Status: 'ok' })).issues).toEqual([]);
      const p = validateSetup(good({ wantErrorMic: true, ch2Status: 'pending' }));
      expect(ids(p.issues)).toEqual([ISSUE.errorMicPending]);
      expect(p.ok).toBe(true);
    });
  });

  describe('warnings and information', () => {
    it('headphones: warning and flag, still ok', () => {
      const out = dev('Headset Earphone (Jabra EVOLVE 20 MS)', 'audiooutput', 'g-jabra');
      const inp = dev('Headset Microphone (Jabra EVOLVE 20 MS)', 'audioinput', 'g-jabra');
      const r = validateSetup(good({ input: inp, output: out }));
      expect(find(r.issues, ISSUE.headphones)).toMatchObject({ level: 'warn', overridable: false });
      expect(find(r.issues, ISSUE.headphones)!.text).toContain('−24 dBFS');
      expect(r.headphones).toBe(true);
      expect(r.ok).toBe(true);
    });

    it('input and output on different devices: clock drift warning', () => {
      const r = validateSetup(good({ output: dev('Babyface Pro (70785634)', 'audiooutput', 'g-rme') }));
      expect(find(r.issues, ISSUE.clockDrift)?.level).toBe('warn');
      expect(r.sameDevice).toBe(false);
      expect(r.ok).toBe(true);
    });

    it('unknown groupId: cannot tell, information only', () => {
      const r = validateSetup(good({ input: { ...SCARLETT_IN, groupId: '' } }));
      expect(find(r.issues, ISSUE.clockDrift)).toBeUndefined();
      expect(find(r.issues, ISSUE.sameDeviceUnknown)?.level).toBe('info');
      expect(r.sameDevice).toBe(false);
    });

    it('no explicit output: cannot be verified', () => {
      const r = validateSetup(good({ output: null, sinkSupported: false }));
      expect(find(r.issues, ISSUE.outputUnverified)?.level).toBe('warn');
      expect(find(r.issues, ISSUE.sinkUnsupported)?.level).toBe('info');
      expect(r.ok).toBe(true);
    });

    it('non-Chromium browser', () => {
      const r = validateSetup(good({ chromium: false }));
      expect(ids(r.issues)).toEqual([ISSUE.nonChromium]);
      expect(r.issues[0]!.level).toBe('warn');
      expect(r.ok).toBe(true);
    });

    it('setSinkId unsupported', () => {
      const r = validateSetup(good({ sinkSupported: false }));
      expect(ids(r.issues)).toEqual([ISSUE.sinkUnsupported]);
      expect(r.issues[0]!.level).toBe('info');
    });

    it('Windows reminder about 48 kHz and enhancements', () => {
      const r = validateSetup(good({ platform: 'windows' }));
      const i = find(r.issues, ISSUE.windowsSetup)!;
      expect(i.level).toBe('info');
      expect(i.text).toContain('48 kHz');
      expect(i.text).toContain('miglioramenti audio');
      expect(find(validateSetup(good({ platform: 'mac' })).issues, ISSUE.windowsSetup)).toBeUndefined();
    });
  });

  it('sorts issues by severity', () => {
    const r = validateSetup({
      input: dev('MacBook Pro Microphone (Built-in)', 'audioinput', 'g1'),
      output: dev('Altoparlanti MacBook Pro', 'audiooutput', 'g2'),
      trackSettings: { echoCancellation: true, sampleRate: 44100 },
      ctxSampleRate: 48000,
      outputLatencyS: 0.09,
      sinkSupported: false,
      chromium: false,
      platform: 'windows',
      wantErrorMic: true,
      ch2Status: 'silent',
    });
    const rank = { block: 0, error: 1, warn: 2, info: 3 } as const;
    const levels = r.issues.map((i) => rank[i.level]);
    expect(levels).toEqual([...levels].sort((a, b) => a - b));
    expect(r.issues[0]!.level).toBe('block');
    expect(r.issues.at(-1)!.level).toBe('info');
    expect(r.ok).toBe(false);
  });
});
