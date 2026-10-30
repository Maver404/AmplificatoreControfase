/**
 * Audio device enumeration and label heuristics.
 *
 * The pure helpers (classifyLabel, deviceKey, resolvePseudo, toDeviceOptions, detectPlatform with an
 * explicit UA, ...) never touch browser globals so they can be tested in Node; only listDevices,
 * onDeviceChange, supportsSetSinkId and the argument-less platform/browser probes read
 * `navigator` / `AudioContext`, and always defensively.
 */

export type DeviceKind = 'audioinput' | 'audiooutput';

export interface DeviceOption {
  deviceId: string;
  /** Same value for endpoints of the same physical device, but only within the current document. */
  groupId: string;
  label: string;
  kind: DeviceKind;
  /** Chrome's 'default' / 'communications' entries, which alias another (real) device. */
  pseudo: boolean;
}

export interface LabelClass {
  /** Bluetooth or another wireless link (AirPods, hands-free profiles, iPhone Continuity mic, ...). */
  bluetooth: boolean;
  airplay: boolean;
  /** Headphones, earbuds or a headset (the output reaches the ears directly). */
  headphones: boolean;
  /**
   * The computer's own transducers (MacBook/iMac mic and speakers, laptop Realtek/Intel SST arrays).
   * False for the headphone jack, line in/out, digital/HDMI outputs and anything USB.
   */
  builtin: boolean;
}

export type Platform = 'mac' | 'windows' | 'other';

export const PSEUDO_DEVICE_IDS: readonly string[] = ['default', 'communications'];
export const isPseudoId = (deviceId: string): boolean => PSEUDO_DEVICE_IDS.includes(deviceId);

// ---------------------------------------------------------------------------------------------
// Label classification
// ---------------------------------------------------------------------------------------------

/** "USB" anywhere, or the "(vvvv:pppp)" vendor:product suffix Chrome appends to USB device labels. */
const USB_RE = /usb|\([0-9a-f]{4}:[0-9a-f]{4}\)/i;

/** Wireless no matter what else the label says. */
const WIRELESS_EXPLICIT_RE =
  /bluetooth|hands[\s-]?free|vivavoce|\ba2dp\b|\bhfp\b|\bhsp\b|air\s?pods|\biphone\b|\bipad\b/i;

/**
 * Bluetooth-only product lines. Only trusted when the label does not look USB, because a few of
 * these brands also sell USB-wired devices.
 */
const BT_PRODUCT_RE = new RegExp(
  [
    String.raw`\b(?:power)?beats\b`,
    String.raw`buds\d*\b`,
    String.raw`\bw[hf]-[a-z0-9]{2,}`,
    String.raw`\bsoundlink\b`,
    String.raw`\bsoundcore\b`,
    String.raw`\bquietcomfort\b`,
    String.raw`\bqc\s?(?:35|45|earbuds|ultra)\b`,
    String.raw`\bbose\s(?:nc|700|qc|ultra)`,
    String.raw`\bjabra\s?elite\b`,
    String.raw`\bjbl\s(?:flip|charge|go|clip|xtreme|tune|live|boombox|pulse|endurance|reflect|vibe)\b`,
    String.raw`\b(?:wonder|mega|hyper|ue\s?)boom\b`,
    String.raw`\bnothing\s?ear\b`,
    String.raw`\bbeoplay\b`,
    String.raw`\bmomentum\s(?:true wireless|[34])\b`,
  ].join('|'),
  'i',
);

const AIRPLAY_RE = /airplay|\bhome\s?pod|\bapple\s?tv\b/i;

const HEADPHONES_RE = new RegExp(
  [
    String.raw`head\s?phone`,
    String.raw`headset`,
    String.raw`ear\s?phone`,
    String.raw`earbud`,
    String.raw`\bin-ear\b`,
    String.raw`cuffi[ae]`,
    String.raw`auricolar[ei]`,
    String.raw`air\s?pods`,
    String.raw`\b(?:power)?beats\b`,
    String.raw`buds\d*\b`,
    String.raw`\bw[hf]-[a-z0-9]{2,}`,
    String.raw`\bquietcomfort\b`,
    String.raw`\bqc\s?(?:35|45|earbuds|ultra)\b`,
    String.raw`\bjabra\s?elite\b`,
    String.raw`\bnothing\s?ear\b`,
    String.raw`kopfh(?:ö|oe)rer`,
    String.raw`\bcasque\b`,
    String.raw`auriculares`,
  ].join('|'),
  'i',
);

const BUILTIN_RE = new RegExp(
  [
    String.raw`built[\s-]?in`,
    String.raw`\bintern[oiae]\b`,
    String.raw`\binternal\b`,
    String.raw`\bintegrat[oiae]\b`,
    String.raw`integrated`,
    String.raw`\bincorporat[oiae]\b`,
    String.raw`\bmacbook\b`,
    String.raw`\bimac\b`,
    String.raw`\bmac\s?(?:mini|studio|pro)\b`,
    String.raw`\bstudio display\b`,
    String.raw`realtek`,
    String.raw`smart sound`,
    String.raw`\bsst\b`,
    String.raw`conexant`,
    String.raw`\bidt\b`,
    String.raw`cirrus logic`,
    String.raw`synaptics`,
    String.raw`senary`,
    String.raw`high definition audio`,
    String.raw`microphone array`,
    String.raw`(?:array|gruppo|matrice) (?:di )?microfoni`,
  ].join('|'),
  'i',
);

/** Endpoints of an on-board codec that do not drive the computer's own transducers. */
const NOT_BUILTIN_RE =
  /nvidia|hdmi|displayport|display audio|\bamd high definition|digital (?:output|audio)|s\/?pdif|optical|\bottic[oa]\b|\bline\b|\blinea\b|\bexternal\b|\bestern[oiae]\b/i;

const SPEAKER_RE = /speaker|altoparlant|\bcasse\b/i;

/**
 * Heuristic classification of a device label (Italian and English, macOS and Windows, including
 * Chrome's "Default - " / "Predefinito - " prefixes and "(Built-in)" / "(vvvv:pppp)" suffixes).
 * USB audio interfaces (Scarlett, Babyface, UMC, Audient, ...) must come out as none of these.
 */
export function classifyLabel(label: string): LabelClass {
  const l = typeof label === 'string' ? label : '';
  const usb = USB_RE.test(l);
  const bluetooth = WIRELESS_EXPLICIT_RE.test(l) || (!usb && BT_PRODUCT_RE.test(l));
  const airplay = AIRPLAY_RE.test(l);
  const headphones = HEADPHONES_RE.test(l);
  // A Realtek "Speakers/Headphones" endpoint may be the laptop speakers, so it stays built-in;
  // a pure headphone endpoint does not feed sound back into the built-in mic.
  const builtin =
    !bluetooth &&
    !airplay &&
    !usb &&
    BUILTIN_RE.test(l) &&
    !NOT_BUILTIN_RE.test(l) &&
    !(headphones && !SPEAKER_RE.test(l));
  return { bluetooth, airplay, headphones, builtin };
}

// ---------------------------------------------------------------------------------------------
// Device lists, pseudo devices and stable keys
// ---------------------------------------------------------------------------------------------

type RawDevice = Pick<MediaDeviceInfo, 'deviceId' | 'groupId' | 'label' | 'kind'>;

/** Turns enumerateDevices() output into deduplicated input/output lists (video and placeholder entries dropped). */
export function toDeviceOptions(raw: readonly RawDevice[]): { inputs: DeviceOption[]; outputs: DeviceOption[] } {
  const inputs: DeviceOption[] = [];
  const outputs: DeviceOption[] = [];
  const seen = new Set<string>();
  for (const d of raw) {
    if (d.kind !== 'audioinput' && d.kind !== 'audiooutput') continue;
    const deviceId = typeof d.deviceId === 'string' ? d.deviceId : '';
    // Before permission is granted Chrome returns one entry per kind with an empty id: not selectable.
    if (deviceId === '') continue;
    const k = `${d.kind}\u0000${deviceId}`;
    if (seen.has(k)) continue;
    seen.add(k);
    const opt: DeviceOption = {
      deviceId,
      groupId: typeof d.groupId === 'string' ? d.groupId : '',
      label: typeof d.label === 'string' ? d.label : '',
      kind: d.kind,
      pseudo: isPseudoId(deviceId),
    };
    (d.kind === 'audioinput' ? inputs : outputs).push(opt);
  }
  return { inputs, outputs };
}

/** "Predefinito - Scarlett 2i2 USB" → "Scarlett 2i2 USB" (the prefix is localised, the separator is not). */
export function stripPseudoPrefix(label: string): string {
  const i = label.indexOf(' - ');
  return i >= 0 ? label.slice(i + 3) : label;
}

/**
 * Label normalised for comparisons: whitespace collapsed and the Windows instance number removed
 * ("Speakers (2- Scarlett 2i2 USB)" → "Speakers (Scarlett 2i2 USB)"), which changes on reconnection.
 */
export function normalizeLabel(label: string): string {
  return label
    .replace(/\(\d+-\s*/g, '(')
    .replace(/\s+/g, ' ')
    .trim();
}

const sameLabel = (a: string, b: string): boolean => a.toLowerCase() === b.toLowerCase();

/**
 * Maps a 'default'/'communications' entry to the real device it aliases (same kind), first by its
 * label without the localised prefix, then by groupId. Returns `d` itself when it is not a pseudo
 * device or the match is ambiguous.
 */
export function resolvePseudo(list: readonly DeviceOption[], d: DeviceOption): DeviceOption {
  if (!d.pseudo) return d;
  const candidates = list.filter((x) => !x.pseudo && x.kind === d.kind);
  const target = normalizeLabel(stripPseudoPrefix(d.label));
  const byLabel = target === '' ? [] : candidates.filter((x) => sameLabel(normalizeLabel(x.label), target));
  if (byLabel.length === 1) return byLabel[0]!;
  if (byLabel.length > 1 && d.groupId !== '') {
    const both = byLabel.filter((x) => x.groupId === d.groupId);
    if (both.length === 1) return both[0]!;
  }
  if (d.groupId !== '') {
    const byGroup = candidates.filter((x) => x.groupId === d.groupId);
    if (byGroup.length === 1) return byGroup[0]!;
  }
  return d;
}

const KEY_VERSION = 'k1';

function keyPart(d: DeviceOption | null, list: readonly DeviceOption[] | undefined): string {
  if (!d) return '-';
  const r = list ? resolvePseudo(list, d) : d;
  const label = normalizeLabel(r.pseudo ? stripPseudoPrefix(r.label) : r.label);
  const id = r.pseudo || isPseudoId(r.deviceId) ? '' : r.deviceId;
  return `${encodeURIComponent(label)}@${encodeURIComponent(id)}`;
}

/**
 * Stable key of an input/output pair for presets and calibrations, built from the normalised label
 * and the deviceId (persistent per origin in Chrome). Pseudo ids are never part of the key: pass
 * `list` to resolve them to the real device (via label, then groupId); unresolved pseudo devices
 * contribute their label only. groupId is deliberately NOT part of the key: the spec requires it to
 * be generated per document, so it would change at every page load.
 */
export function deviceKey(
  input: DeviceOption | null,
  output: DeviceOption | null,
  list?: readonly DeviceOption[],
): string {
  return `${KEY_VERSION}|in:${keyPart(input, list)}|out:${keyPart(output, list)}`;
}

export interface DeviceKeyPart {
  label: string;
  /** '' when unknown (unresolved pseudo device). */
  deviceId: string;
}

export interface ParsedDeviceKey {
  input: DeviceKeyPart | null;
  output: DeviceKeyPart | null;
}

function parsePart(s: string): DeviceKeyPart | null | undefined {
  if (s === '-') return null;
  const at = s.indexOf('@');
  if (at < 0) return undefined;
  try {
    return { label: decodeURIComponent(s.slice(0, at)), deviceId: decodeURIComponent(s.slice(at + 1)) };
  } catch {
    return undefined;
  }
}

/** Inverse of deviceKey(); null for keys that are not in the current format. */
export function parseDeviceKey(key: string): ParsedDeviceKey | null {
  if (typeof key !== 'string') return null;
  const m = /^k1\|in:([^|]*)\|out:([^|]*)$/.exec(key);
  if (!m) return null;
  const input = parsePart(m[1]!);
  const output = parsePart(m[2]!);
  if (input === undefined || output === undefined) return null;
  return { input, output };
}

/** 'same-model': same label but a different deviceId (another unit, or the same one on another port). */
export type DeviceKeyMatch = 'same' | 'same-model' | 'different';

function comparePart(a: DeviceKeyPart | null, b: DeviceKeyPart | null): DeviceKeyMatch {
  if (a === null || b === null) return a === b ? 'same' : 'different';
  const labelsMatch = a.label !== '' && sameLabel(a.label, b.label);
  if (a.deviceId !== '' && b.deviceId !== '') {
    if (a.deviceId === b.deviceId) return 'same';
    return labelsMatch ? 'same-model' : 'different';
  }
  return labelsMatch ? 'same' : 'different';
}

export function compareDeviceKeys(a: string, b: string): DeviceKeyMatch {
  if (a === b) return 'same';
  const pa = parseDeviceKey(a);
  const pb = parseDeviceKey(b);
  if (!pa || !pb) return 'different';
  const r = [comparePart(pa.input, pb.input), comparePart(pa.output, pb.output)];
  if (r.includes('different')) return 'different';
  return r.includes('same-model') ? 'same-model' : 'same';
}

// ---------------------------------------------------------------------------------------------
// Browser access
// ---------------------------------------------------------------------------------------------

interface UAData {
  brands?: ReadonlyArray<{ brand: string }>;
  platform?: string;
}

interface NavigatorLike {
  mediaDevices?: MediaDevices;
  userAgent?: string;
  userAgentData?: UAData;
  maxTouchPoints?: number;
}

function getNavigator(): NavigatorLike | null {
  try {
    return typeof navigator === 'undefined' ? null : (navigator as unknown as NavigatorLike);
  } catch {
    return null;
  }
}

/** navigator.mediaDevices is missing outside secure contexts (plain http on a LAN address). */
export function mediaDevicesAvailable(): boolean {
  const md = getNavigator()?.mediaDevices;
  return !!md && typeof md.enumerateDevices === 'function';
}

/** Audio inputs and outputs; empty lists when mediaDevices is unavailable. Rejects if enumeration fails. */
export async function listDevices(): Promise<{ inputs: DeviceOption[]; outputs: DeviceOption[] }> {
  const md = getNavigator()?.mediaDevices;
  if (!md || typeof md.enumerateDevices !== 'function') return { inputs: [], outputs: [] };
  return toDeviceOptions(await md.enumerateDevices());
}

/** AudioContext.setSinkId (Chrome 110+) is required to route the anti-noise to a chosen output. */
export function supportsSetSinkId(): boolean {
  try {
    return typeof AudioContext !== 'undefined' && 'setSinkId' in AudioContext.prototype;
  } catch {
    return false;
  }
}

/** Subscribes to 'devicechange'; returns an idempotent unsubscribe function. */
export function onDeviceChange(cb: () => void): () => void {
  const md = getNavigator()?.mediaDevices;
  if (!md || typeof md.addEventListener !== 'function') return () => {};
  const handler = (): void => cb();
  md.addEventListener('devicechange', handler);
  let active = true;
  return () => {
    if (!active) return;
    active = false;
    md.removeEventListener('devicechange', handler);
  };
}

function platformFromUa(ua: string): Platform {
  if (/iphone|ipad|ipod|android/i.test(ua)) return 'other';
  if (/windows|win32|win64/i.test(ua)) return 'windows';
  if (/macintosh|mac os x|macos/i.test(ua)) return 'mac';
  return 'other';
}

/** Desktop platform from a UA string, or from navigator (UA-CH platform first) when omitted. */
export function detectPlatform(ua?: string): Platform {
  if (typeof ua === 'string') return platformFromUa(ua);
  const nav = getNavigator();
  if (!nav) return 'other';
  const chPlatform = nav.userAgentData?.platform;
  let p: Platform;
  if (typeof chPlatform === 'string' && chPlatform !== '') {
    p = /^mac/i.test(chPlatform) ? 'mac' : /^windows/i.test(chPlatform) ? 'windows' : 'other';
  } else {
    p = platformFromUa(nav.userAgent ?? '');
  }
  // iPadOS in desktop mode reports a Mac user agent.
  if (p === 'mac' && (nav.maxTouchPoints ?? 0) > 1) return 'other';
  return p;
}

function chromiumFromUa(ua: string): boolean {
  // Chrome on iOS (CriOS) and other iOS browsers are WebKit, not Chromium.
  if (/crios|fxios|edgios|firefox/i.test(ua)) return false;
  return /chrom(?:e|ium)\/\d/i.test(ua);
}

/** Chromium engine (Chrome, Edge, Brave, Opera, headless Chrome); from `ua` when given. */
export function isChromium(ua?: string): boolean {
  if (typeof ua === 'string') return chromiumFromUa(ua);
  const nav = getNavigator();
  if (!nav) return false;
  const brands = nav.userAgentData?.brands;
  if (Array.isArray(brands) && brands.length > 0) return brands.some((b) => /chromium/i.test(b.brand));
  return chromiumFromUa(nav.userAgent ?? '');
}
