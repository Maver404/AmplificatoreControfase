import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  classifyLabel,
  compareDeviceKeys,
  detectPlatform,
  deviceKey,
  isChromium,
  listDevices,
  mediaDevicesAvailable,
  normalizeLabel,
  onDeviceChange,
  parseDeviceKey,
  resolvePseudo,
  supportsSetSinkId,
  toDeviceOptions,
  type DeviceOption,
  type LabelClass,
} from '../../src/audio/devices.ts';

type Flags = 'bt' | 'air' | 'hp' | 'bi';
const flags = (...f: Flags[]): LabelClass => ({
  bluetooth: f.includes('bt'),
  airplay: f.includes('air'),
  headphones: f.includes('hp'),
  builtin: f.includes('bi'),
});

const LABELS: Array<[string, LabelClass]> = [
  // Apple / Bluetooth
  ['AirPods Pro', flags('bt', 'hp')],
  ['AirPods di Mario', flags('bt', 'hp')],
  ['Predefinito - AirPods Max (Bluetooth)', flags('bt', 'hp')],
  ['Beats Studio Buds', flags('bt', 'hp')],
  ['Powerbeats Pro', flags('bt', 'hp')],
  ['Headphones (WH-1000XM4 Stereo)', flags('bt', 'hp')],
  ['Headset (WH-1000XM4 Hands-Free AG Audio)', flags('bt', 'hp')],
  ['Auricolare (Galaxy Buds2 Pro Hands-Free AG Audio)', flags('bt', 'hp')],
  ['Cuffie (Bose QC35 II Stereo)', flags('bt', 'hp')],
  ['JBL Flip 5', flags('bt')],
  ['Altoparlante Bluetooth', flags('bt')],
  ["Mario's iPhone Microphone", flags('bt')],
  // AirPlay
  ['AirPlay', flags('air')],
  ['HomePod del soggiorno', flags('air')],
  // Built-in, macOS (English and Italian)
  ['MacBook Pro Microphone (Built-in)', flags('bi')],
  ['Default - MacBook Pro Microphone (Built-in)', flags('bi')],
  ['MacBook Air Speakers', flags('bi')],
  ['Predefinito - Altoparlanti MacBook Pro', flags('bi')],
  ['Microfono MacBook Air', flags('bi')],
  ['Microfono iMac', flags('bi')],
  ['Altoparlanti iMac', flags('bi')],
  ['Mac mini Speakers', flags('bi')],
  ['Built-in Output', flags('bi')],
  ['Built-in Microphone', flags('bi')],
  ['Microfono interno', flags('bi')],
  ['Altoparlanti interni', flags('bi')],
  ['Microfono integrato', flags('bi')],
  // Built-in, Windows
  ['Microphone (Realtek(R) Audio)', flags('bi')],
  ['Microfono (Realtek(R) Audio)', flags('bi')],
  ['Altoparlanti (Realtek(R) Audio)', flags('bi')],
  ['Speakers (Realtek High Definition Audio)', flags('bi')],
  ['Speakers/Headphones (Realtek(R) Audio)', flags('bi', 'hp')],
  ['Microphone Array (Intel® Smart Sound Technology for Digital Microphones)', flags('bi')],
  ['Gruppo microfoni (Tecnologia Intel® Smart Sound per microfoni digitali)', flags('bi')],
  ['Speakers (High Definition Audio Device)', flags('bi')],
  ['Microphone (Integrated Webcam)', flags('bi')],
  // Headphones that are neither wireless nor built-in transducers
  ['Headphones (Realtek(R) Audio)', flags('hp')],
  ['Cuffie (Realtek(R) Audio)', flags('hp')],
  ['External Headphones', flags('hp')],
  ['Cuffie esterne', flags('hp')],
  ['Headset Earphone (Jabra EVOLVE 20 MS)', flags('hp')],
  ['Apple USB-C to 3.5mm Headphone Jack Adapter', flags('hp')],
  ['Auricolari (USB Audio Device)', flags('hp')],
  // On-board endpoints that do not drive the computer's own speakers / mic
  ['Line In (Realtek(R) Audio)', flags()],
  ['Realtek Digital Output (Realtek(R) Audio)', flags()],
  ['DELL U2720Q (NVIDIA High Definition Audio)', flags()],
  ['LG ULTRAFINE (Intel(R) Display Audio)', flags()],
  ['External Microphone', flags()],
  ['Realtek USB2.0 Audio', flags()],
  // USB audio interfaces and USB mics: none of the flags
  ['Scarlett 2i2 USB', flags()],
  ['Scarlett 2i2 USB (1235:8210)', flags()],
  ['Predefinito - Scarlett 2i2 USB (1235:8210)', flags()],
  ['Scarlett Solo USB', flags()],
  ['Focusrite USB Audio', flags()],
  ['Microfono (Focusrite USB Audio)', flags()],
  ['Altoparlanti (2- Focusrite USB Audio)', flags()],
  ['Babyface Pro (70785634)', flags()],
  ['Analog (1+2) (RME Babyface Pro)', flags()],
  ['Fireface UCX II (24082736)', flags()],
  ['UMC204HD 192k', flags()],
  ['Speakers (UMC202HD 192k)', flags()],
  ['Microphone (2- UMC404HD 192k)', flags()],
  ['Audient iD14', flags()],
  ['Line (Audient EVO 4)', flags()],
  ['MOTU M2', flags()],
  ['SSL 2+', flags()],
  ['Volt 276', flags()],
  ['M-Audio AIR 192|4', flags()],
  ['Arturia MiniFuse 2', flags()],
  ['Steinberg UR22C', flags()],
  ['PreSonus AudioBox USB 96', flags()],
  ['Universal Audio Apollo Twin', flags()],
  ['Blue Yeti Stereo Microphone', flags()],
  ['Arturia BeatStep', flags()],
  ['Dispositivo aggregato', flags()],
];

describe('classifyLabel', () => {
  it('has a large table', () => {
    expect(LABELS.length).toBeGreaterThanOrEqual(25);
  });

  it.each(LABELS)('%s', (label, expected) => {
    expect(classifyLabel(label)).toEqual(expected);
  });

  it('treats an empty or non-string label as nothing', () => {
    expect(classifyLabel('')).toEqual(flags());
    expect(classifyLabel(undefined as unknown as string)).toEqual(flags());
  });
});

const dev = (
  deviceId: string,
  label: string,
  kind: 'audioinput' | 'audiooutput',
  groupId = 'g',
): DeviceOption => ({ deviceId, groupId, label, kind, pseudo: deviceId === 'default' || deviceId === 'communications' });

describe('toDeviceOptions', () => {
  it('splits, deduplicates, drops video and placeholder entries, marks pseudo devices', () => {
    const raw = [
      { deviceId: 'default', groupId: 'g1', label: 'Predefinito - Scarlett 2i2 USB', kind: 'audioinput' },
      { deviceId: 'communications', groupId: 'g1', label: 'Comunicazioni - Scarlett 2i2 USB', kind: 'audioinput' },
      { deviceId: 'abc', groupId: 'g1', label: 'Scarlett 2i2 USB', kind: 'audioinput' },
      { deviceId: 'abc', groupId: 'g1', label: 'Scarlett 2i2 USB', kind: 'audioinput' },
      { deviceId: 'abc', groupId: 'g1', label: 'Scarlett 2i2 USB', kind: 'audiooutput' },
      { deviceId: 'cam', groupId: 'g9', label: 'FaceTime HD Camera', kind: 'videoinput' },
      { deviceId: '', groupId: '', label: '', kind: 'audioinput' },
    ] as const;
    const { inputs, outputs } = toDeviceOptions(raw as unknown as MediaDeviceInfo[]);
    expect(inputs.map((d) => [d.deviceId, d.pseudo])).toEqual([
      ['default', true],
      ['communications', true],
      ['abc', false],
    ]);
    expect(outputs).toEqual([
      { deviceId: 'abc', groupId: 'g1', label: 'Scarlett 2i2 USB', kind: 'audiooutput', pseudo: false },
    ]);
  });
});

describe('resolvePseudo', () => {
  const list: DeviceOption[] = [
    dev('default', 'Predefinito - Scarlett 2i2 USB (1235:8210)', 'audioinput', 'gS'),
    dev('in-mac', 'MacBook Pro Microphone (Built-in)', 'audioinput', 'gM'),
    dev('in-sc', 'Scarlett 2i2 USB (1235:8210)', 'audioinput', 'gS'),
    dev('default', 'Predefinito - Scarlett 2i2 USB (1235:8210)', 'audiooutput', 'gS'),
    dev('out-sc', 'Scarlett 2i2 USB (1235:8210)', 'audiooutput', 'gS'),
    dev('out-mac', 'MacBook Pro Speakers (Built-in)', 'audiooutput', 'gM'),
  ];

  it('returns non-pseudo devices unchanged', () => {
    expect(resolvePseudo(list, list[1]!)).toBe(list[1]);
  });

  it('resolves default by label without the localised prefix, keeping the kind', () => {
    expect(resolvePseudo(list, list[0]!)).toBe(list[2]);
    expect(resolvePseudo(list, list[3]!)).toBe(list[4]);
  });

  it('falls back to groupId when the label does not match', () => {
    const d = dev('communications', 'Communications - Something else', 'audioinput', 'gM');
    expect(resolvePseudo(list, d)).toBe(list[1]);
  });

  it('uses groupId to break label ties', () => {
    const twins = [
      dev('a', 'UMC204HD 192k', 'audioinput', 'g1'),
      dev('b', 'UMC204HD 192k', 'audioinput', 'g2'),
    ];
    expect(resolvePseudo(twins, dev('default', 'Default - UMC204HD 192k', 'audioinput', 'g2'))).toBe(twins[1]);
  });

  it('returns the pseudo device itself when ambiguous or unknown', () => {
    const twins = [dev('a', 'X', 'audioinput', 'g1'), dev('b', 'Y', 'audioinput', 'g1')];
    const d = dev('default', 'Default - Z', 'audioinput', 'g1');
    expect(resolvePseudo(twins, d)).toBe(d);
    const lonely = dev('default', 'Default - Q', 'audioinput', 'none');
    expect(resolvePseudo(list, lonely)).toBe(lonely);
  });
});

describe('deviceKey', () => {
  const inSc = dev('in-sc', 'Scarlett 2i2 USB (1235:8210)', 'audioinput', 'gS');
  const outSc = dev('out-sc', 'Scarlett 2i2 USB (1235:8210)', 'audiooutput', 'gS');

  it('is independent of groupId (regenerated per document) and of list order', () => {
    const k1 = deviceKey(inSc, outSc);
    const k2 = deviceKey({ ...inSc, groupId: 'other-session' }, { ...outSc, groupId: 'zzz' });
    expect(k2).toBe(k1);
    expect(deviceKey(inSc, outSc, [outSc, inSc])).toBe(k1);
  });

  it('never contains pseudo ids and resolves them through the list', () => {
    const defIn = dev('default', 'Predefinito - Scarlett 2i2 USB (1235:8210)', 'audioinput', 'gS');
    const defOut = dev('default', 'Default - Scarlett 2i2 USB (1235:8210)', 'audiooutput', 'gS');
    const list = [defIn, inSc, defOut, outSc];
    expect(deviceKey(defIn, defOut, list)).toBe(deviceKey(inSc, outSc));
    const unresolved = deviceKey(defIn, defOut);
    expect(unresolved).not.toContain('default');
    expect(parseDeviceKey(unresolved)).toEqual({
      input: { label: 'Scarlett 2i2 USB (1235:8210)', deviceId: '' },
      output: { label: 'Scarlett 2i2 USB (1235:8210)', deviceId: '' },
    });
    // An unresolved default still matches the real device by label.
    expect(compareDeviceKeys(unresolved, deviceKey(inSc, outSc))).toBe('same');
  });

  it('normalises the Windows instance number and whitespace', () => {
    expect(normalizeLabel('Speakers (2- Focusrite  USB Audio) ')).toBe('Speakers (Focusrite USB Audio)');
    expect(normalizeLabel('Analog (1+2) (RME Babyface Pro)')).toBe('Analog (1+2) (RME Babyface Pro)');
    const a = dev('id1', 'Speakers (2- Focusrite USB Audio)', 'audiooutput');
    const b = dev('id1', 'Speakers (Focusrite USB Audio)', 'audiooutput');
    expect(deviceKey(null, a)).toBe(deviceKey(null, b));
  });

  it('round-trips labels with separator characters', () => {
    const weird = dev('x|y@z', 'Mic | @ : % (1)', 'audioinput');
    const parsed = parseDeviceKey(deviceKey(weird, null));
    expect(parsed).toEqual({ input: { label: 'Mic | @ : % (1)', deviceId: 'x|y@z' }, output: null });
    expect(parseDeviceKey('garbage')).toBeNull();
    expect(parseDeviceKey('k1|in:%E0%A4%A@|out:-')).toBeNull();
  });

  it('distinguishes devices, models and missing devices', () => {
    const base = deviceKey(inSc, outSc);
    expect(compareDeviceKeys(base, base)).toBe('same');
    // Same label, another unit (or another port).
    const twin = deviceKey({ ...inSc, deviceId: 'in-sc-2' }, outSc);
    expect(compareDeviceKeys(base, twin)).toBe('same-model');
    // Same id, label changed (e.g. renumbered): still the same endpoint.
    expect(compareDeviceKeys(base, deviceKey({ ...inSc, label: 'Scarlett 2i2' }, outSc))).toBe('same');
    const other = deviceKey(dev('bb', 'Babyface Pro (70785634)', 'audioinput'), outSc);
    expect(compareDeviceKeys(base, other)).toBe('different');
    expect(compareDeviceKeys(base, deviceKey(inSc, null))).toBe('different');
    expect(compareDeviceKeys(deviceKey(null, null), deviceKey(null, null))).toBe('same');
    expect(compareDeviceKeys(base, 'not a key')).toBe('different');
  });
});

describe('platform and browser detection', () => {
  const UA = {
    chromeMac:
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    chromeWin:
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    edgeWin:
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0',
    headless:
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.0.0 Safari/537.36',
    safariMac:
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15',
    firefoxWin: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0',
    chromeIos:
      'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0 Mobile/15E148 Safari/604.1',
    chromeAndroid:
      'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36',
  };

  it('detectPlatform', () => {
    expect(detectPlatform(UA.chromeMac)).toBe('mac');
    expect(detectPlatform(UA.safariMac)).toBe('mac');
    expect(detectPlatform(UA.chromeWin)).toBe('windows');
    expect(detectPlatform(UA.firefoxWin)).toBe('windows');
    expect(detectPlatform(UA.headless)).toBe('other');
    expect(detectPlatform(UA.chromeIos)).toBe('other');
    expect(detectPlatform(UA.chromeAndroid)).toBe('other');
  });

  it('isChromium', () => {
    expect(isChromium(UA.chromeMac)).toBe(true);
    expect(isChromium(UA.chromeWin)).toBe(true);
    expect(isChromium(UA.edgeWin)).toBe(true);
    expect(isChromium(UA.headless)).toBe(true);
    expect(isChromium(UA.safariMac)).toBe(false);
    expect(isChromium(UA.firefoxWin)).toBe(false);
    expect(isChromium(UA.chromeIos)).toBe(false);
  });

  describe('from navigator', () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('prefers UA client hints', () => {
      vi.stubGlobal('navigator', {
        userAgent: UA.safariMac,
        userAgentData: { platform: 'Windows', brands: [{ brand: 'Not.A/Brand' }, { brand: 'Chromium' }] },
      });
      expect(detectPlatform()).toBe('windows');
      expect(isChromium()).toBe(true);
    });

    it('falls back to the user agent string, and treats a touch "Mac" as iPad', () => {
      vi.stubGlobal('navigator', { userAgent: UA.chromeMac, maxTouchPoints: 0 });
      expect(detectPlatform()).toBe('mac');
      expect(isChromium()).toBe(true);
      vi.stubGlobal('navigator', { userAgent: UA.safariMac, maxTouchPoints: 5 });
      expect(detectPlatform()).toBe('other');
      expect(isChromium()).toBe(false);
    });
  });
});

describe('browser access', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('listDevices returns empty lists without mediaDevices (insecure context)', async () => {
    vi.stubGlobal('navigator', { userAgent: 'x' });
    expect(mediaDevicesAvailable()).toBe(false);
    await expect(listDevices()).resolves.toEqual({ inputs: [], outputs: [] });
  });

  it('listDevices enumerates through navigator.mediaDevices', async () => {
    const enumerateDevices = vi.fn(async () => [
      { deviceId: 'default', groupId: 'g', label: 'Default - MOTU M2', kind: 'audioinput' },
      { deviceId: 'm2', groupId: 'g', label: 'MOTU M2', kind: 'audioinput' },
      { deviceId: 'm2o', groupId: 'g', label: 'MOTU M2', kind: 'audiooutput' },
    ]);
    vi.stubGlobal('navigator', { mediaDevices: { enumerateDevices } });
    expect(mediaDevicesAvailable()).toBe(true);
    const { inputs, outputs } = await listDevices();
    expect(enumerateDevices).toHaveBeenCalledOnce();
    expect(inputs.map((d) => d.deviceId)).toEqual(['default', 'm2']);
    expect(inputs[0]!.pseudo).toBe(true);
    expect(outputs.map((d) => d.deviceId)).toEqual(['m2o']);
  });

  it('onDeviceChange subscribes and unsubscribes once', () => {
    const target = new EventTarget();
    const removeSpy = vi.spyOn(target, 'removeEventListener');
    vi.stubGlobal('navigator', { mediaDevices: target });
    const cb = vi.fn();
    const off = onDeviceChange(cb);
    target.dispatchEvent(new Event('devicechange'));
    expect(cb).toHaveBeenCalledTimes(1);
    off();
    off();
    expect(removeSpy).toHaveBeenCalledTimes(1);
    target.dispatchEvent(new Event('devicechange'));
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('onDeviceChange is a no-op without mediaDevices', () => {
    vi.stubGlobal('navigator', {});
    const off = onDeviceChange(() => {});
    expect(() => off()).not.toThrow();
  });

  it('supportsSetSinkId checks AudioContext.prototype', () => {
    vi.stubGlobal('AudioContext', undefined);
    expect(supportsSetSinkId()).toBe(false);
    class WithSink {
      setSinkId(): Promise<void> {
        return Promise.resolve();
      }
    }
    vi.stubGlobal('AudioContext', WithSink);
    expect(supportsSetSinkId()).toBe(true);
    vi.stubGlobal('AudioContext', class NoSink {});
    expect(supportsSetSinkId()).toBe(false);
  });
});
