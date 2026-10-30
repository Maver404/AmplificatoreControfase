/**
 * Presets in localStorage. Everything that comes back from storage or from an imported file is
 * untrusted: it goes through sanitizePreset (schema + clampParams), and the +12 dB unlock is never
 * stored or imported. Loading a preset (mute → apply → soft-start) is the caller's job.
 */
import { clampParams, type EngineParams } from '../dsp/params.ts';
import { compareDeviceKeys, parseDeviceKey, type DeviceKeyPart } from '../audio/devices.ts';

export const PRESETS_STORAGE_KEY = 'controfase.presets.v1';
const CORRUPT_BACKUP_KEY = `${PRESETS_STORAGE_KEY}.corrotto`;
const PROBE_KEY = `${PRESETS_STORAGE_KEY}.probe`;
const EXPORT_FORMAT = 'controfase-presets';
const FORMAT_VERSION = 1;

export interface Preset {
  id: string;
  name: string;
  createdAt: number;
  deviceKey: string;
  sampleRate: number;
  params: EngineParams;
  /** Opaque plain-JSON object, validated by the calibration module when it is used. */
  calibration?: unknown;
}

export interface PresetMeta {
  deviceKey: string;
  sampleRate: number;
  calibration?: unknown;
}

const ID_RE = /^[A-Za-z0-9_.-]{1,128}$/;
const MAX_NAME_CHARS = 80;
const MAX_DEVICE_KEY_CHARS = 4096;
const SAMPLE_RATE_RANGE = { min: 8000, max: 768000 } as const;
const MAX_JSON_DEPTH = 64;
const MAX_JSON_NODES = 1_000_000;
const MAX_IMPORT_CHARS = 32 * 1024 * 1024;
const UNNAMED = 'Senza nome';

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

function isPlainObject(v: unknown): v is Record<string, unknown> {
  if (!isObject(v)) return false;
  const proto: unknown = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
}

const INVALID: unique symbol = Symbol('invalid');

/** Deep copy restricted to JSON values (finite numbers only); INVALID for anything else. */
function cloneJson(v: unknown, depth: number, budget: { nodes: number }): unknown {
  if (++budget.nodes > MAX_JSON_NODES || depth > MAX_JSON_DEPTH) return INVALID;
  if (v === null || typeof v === 'string' || typeof v === 'boolean') return v;
  if (typeof v === 'number') return Number.isFinite(v) ? v : INVALID;
  if (Array.isArray(v)) {
    const out: unknown[] = [];
    for (const x of v as unknown[]) {
      const c = cloneJson(x, depth + 1, budget);
      if (c === INVALID) return INVALID;
      out.push(c);
    }
    return out;
  }
  if (isPlainObject(v)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v)) {
      // Assigning '__proto__' would replace the copy's prototype instead of adding a key.
      if (k === '__proto__') continue;
      const x = v[k];
      if (x === undefined) continue;
      const c = cloneJson(x, depth + 1, budget);
      if (c === INVALID) return INVALID;
      out[k] = c;
    }
    return out;
  }
  return INVALID;
}

/** A deep copy of `v` if it is a plain JSON object (not an array), otherwise undefined. */
function cleanCalibration(v: unknown): Record<string, unknown> | undefined {
  if (!isPlainObject(v)) return undefined;
  const c = cloneJson(v, 0, { nodes: 0 });
  return c === INVALID ? undefined : (c as Record<string, unknown>);
}

/** Control and bidi-override characters removed, whitespace collapsed, at most MAX_NAME_CHARS. */
function cleanName(s: string): string {
  const flat = s.replace(/[\u0000-\u001f\u007f-\u009f‪-‮⁦-⁩]/g, ' ').replace(/\s+/g, ' ').trim();
  return Array.from(flat).slice(0, MAX_NAME_CHARS).join('').trim();
}

/**
 * Validates an untrusted preset: rejects non-objects and entries without a valid id, name or params
 * object; strips unknown keys; clamps every parameter with clampParams and forces gainUnlocked=false
 * (so channel gains above 0 dB come back at 0 dB); keeps `calibration` only if it is a plain JSON
 * object (deep-copied). Invalid metadata falls back to neutral values (sampleRate 0 = unknown).
 */
export function sanitizePreset(raw: unknown): Preset | null {
  try {
    if (!isObject(raw)) return null;
    const { id, name, params } = raw;
    if (typeof id !== 'string' || !ID_RE.test(id)) return null;
    if (typeof name !== 'string') return null;
    if (!isObject(params)) return null;

    const createdAt =
      typeof raw.createdAt === 'number' && Number.isFinite(raw.createdAt) && raw.createdAt >= 0
        ? Math.floor(raw.createdAt)
        : 0;
    const deviceKey =
      typeof raw.deviceKey === 'string' && raw.deviceKey.length <= MAX_DEVICE_KEY_CHARS ? raw.deviceKey : '';
    const sr = raw.sampleRate;
    const sampleRate =
      typeof sr === 'number' && Number.isFinite(sr) && sr >= SAMPLE_RATE_RANGE.min && sr <= SAMPLE_RATE_RANGE.max
        ? sr
        : 0;
    const preset: Preset = {
      id,
      name: cleanName(name) || UNNAMED,
      createdAt,
      deviceKey,
      sampleRate,
      params: clampParams({ ...(params as Partial<EngineParams>), gainUnlocked: false }),
    };
    const calibration = cleanCalibration(raw.calibration);
    if (calibration !== undefined) preset.calibration = calibration;
    return preset;
  } catch {
    // Exotic inputs (throwing getters, proxies): reject rather than propagate.
    return null;
  }
}

const clonePreset = (p: Preset): Preset => structuredClone(p);

function randomId(): string {
  const r = new Uint32Array(2);
  try {
    globalThis.crypto.getRandomValues(r);
  } catch {
    r[0] = Math.floor(Math.random() * 0x100000000);
    r[1] = Math.floor(Math.random() * 0x100000000);
  }
  return `p-${Date.now().toString(36)}-${r[0]!.toString(36)}${r[1]!.toString(36)}`;
}

function defaultStorage(): Storage | null {
  try {
    const s = (globalThis as { localStorage?: Storage }).localStorage;
    if (!s) return null;
    s.setItem(PROBE_KEY, '1');
    s.removeItem(PROBE_KEY);
    return s;
  } catch {
    return null;
  }
}

/** Sanitised, id-deduplicated (last wins) presets from a stored or imported list. */
function sanitizeList(entries: readonly unknown[]): { presets: Preset[]; rejected: number } {
  const byId = new Map<string, Preset>();
  let rejected = 0;
  for (const e of entries) {
    const p = sanitizePreset(e);
    if (p) byId.set(p.id, p);
    else rejected++;
  }
  return { presets: [...byId.values()], rejected };
}

/** Accepts { presets: [...] } (stored and exported format), a bare array, or a single preset. */
function entriesOf(data: unknown): unknown[] {
  if (Array.isArray(data)) return data as unknown[];
  if (isObject(data) && Array.isArray(data.presets)) return data.presets as unknown[];
  return [data];
}

const byName = (a: Preset, b: Preset): number =>
  a.name.localeCompare(b.name, 'it', { sensitivity: 'base' }) || a.createdAt - b.createdAt;

const SAVE_FAILED =
  "Impossibile salvare i preset nella memoria del browser (spazio esaurito o accesso negato): restano disponibili solo finché la pagina è aperta. Usa «Esporta» per conservarli.";

export class PresetStore {
  private readonly storage: Storage | null;
  private readonly now: () => number;
  private cache: Preset[] = [];
  /** The last write failed: memory holds data storage does not have, so it wins over re-reads. */
  private unsaved = false;
  private error: string | null = null;

  /** `storage` omitted: localStorage if usable, else memory only. `null`: memory only. */
  constructor(storage?: Storage | null, opts: { now?: () => number } = {}) {
    this.storage = storage === undefined ? defaultStorage() : storage;
    this.now = opts.now ?? Date.now;
    this.cache = this.readStorage() ?? [];
  }

  /** True when presets survive a reload (storage available and the last write succeeded). */
  get persistent(): boolean {
    return this.storage !== null && !this.unsaved;
  }

  /** Italian message about the last failed write, null after a successful one. */
  get lastError(): string | null {
    return this.error;
  }

  /** Copies, sorted by name. */
  list(): Preset[] {
    return this.current().sort(byName).map(clonePreset);
  }

  get(id: string): Preset | null {
    const p = this.current().find((x) => x.id === id);
    return p ? clonePreset(p) : null;
  }

  /**
   * Saves the parameters under `name` (an existing preset with the same name, ignoring case, is
   * replaced and keeps its id). The stored copy is sanitised: gainUnlocked is never persisted.
   */
  save(name: string, params: EngineParams, meta: PresetMeta): Preset {
    const list = this.current();
    const clean = cleanName(typeof name === 'string' ? name : '') || `Preset ${list.length + 1}`;
    const key = clean.toLocaleLowerCase('it');
    const existing = list.find((p) => p.name.toLocaleLowerCase('it') === key);
    let id = existing?.id ?? randomId();
    while (!existing && list.some((p) => p.id === id)) id = randomId();
    const preset = sanitizePreset({
      id,
      name: clean,
      createdAt: this.now(),
      deviceKey: meta.deviceKey,
      sampleRate: meta.sampleRate,
      params,
      calibration: meta.calibration,
    });
    if (!preset) throw new TypeError('Parametri del preset non validi.');
    this.commit([...list.filter((p) => p.id !== id), preset]);
    return clonePreset(preset);
  }

  remove(id: string): void {
    const list = this.current();
    const next = list.filter((p) => p.id !== id);
    if (next.length !== list.length) this.commit(next);
  }

  exportJson(): string {
    return JSON.stringify(
      { format: EXPORT_FORMAT, version: FORMAT_VERSION, presets: this.current().sort(byName) },
      null,
      2,
    );
  }

  /**
   * Imports presets from exportJson() output, a bare array or a single preset. Every entry is
   * sanitised; entries with an existing id replace it. An unreadable file counts as one rejection.
   */
  importJson(json: string): { imported: number; rejected: number } {
    if (typeof json !== 'string' || json.length > MAX_IMPORT_CHARS) return { imported: 0, rejected: 1 };
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return { imported: 0, rejected: 1 };
    }
    const entries = entriesOf(data);
    const { presets, rejected } = sanitizeList(entries);
    const imported = entries.length - rejected;
    if (presets.length > 0) {
      const ids = new Set(presets.map((p) => p.id));
      this.commit([...this.current().filter((p) => !ids.has(p.id)), ...presets]);
    }
    return { imported, rejected };
  }

  /** Fresh view of the presets (another tab may have changed storage), falling back to memory. */
  private current(): Preset[] {
    if (!this.unsaved) {
      const fromStorage = this.readStorage();
      if (fromStorage) this.cache = fromStorage;
    }
    return [...this.cache];
  }

  /** null when there is no storage or it cannot be read. */
  private readStorage(): Preset[] | null {
    if (!this.storage) return null;
    let raw: string | null;
    try {
      raw = this.storage.getItem(PRESETS_STORAGE_KEY);
    } catch {
      return null;
    }
    if (raw === null) return [];
    let data: unknown;
    try {
      data = JSON.parse(raw);
    } catch {
      this.backupCorrupt(raw);
      return [];
    }
    return sanitizeList(entriesOf(data)).presets;
  }

  /** Keeps unreadable data aside once, so the next save does not silently destroy it. */
  private backupCorrupt(raw: string): void {
    try {
      if (this.storage && this.storage.getItem(CORRUPT_BACKUP_KEY) === null) {
        this.storage.setItem(CORRUPT_BACKUP_KEY, raw);
      }
    } catch {
      // Best effort only.
    }
  }

  private commit(list: Preset[]): void {
    this.cache = list;
    if (!this.storage) return;
    try {
      this.storage.setItem(PRESETS_STORAGE_KEY, JSON.stringify({ version: FORMAT_VERSION, presets: list }));
      this.unsaved = false;
      this.error = null;
    } catch {
      this.unsaved = true;
      this.error = SAVE_FAILED;
    }
  }
}

const formatKHz = (sr: number): string => `${String(Number((sr / 1000).toFixed(3))).replace('.', ',')} kHz`;

function describePart(part: DeviceKeyPart | null | undefined): string {
  if (part === null) return 'nessuno';
  if (!part || part.label.trim() === '') return 'sconosciuto';
  return `«${part.label}»`;
}

/**
 * Italian warnings to show before loading a preset made with another device pair or sample rate.
 * Empty when everything matches (or the current device key is unknown).
 */
export function mismatchWarnings(p: Preset, current: { deviceKey: string; sampleRate: number }): string[] {
  const out: string[] = [];
  if (p.deviceKey === '') {
    out.push(
      "Il preset non indica con quale interfaccia audio è stato creato: controlla ritardi e guadagni prima di attivare l'uscita.",
    );
  } else if (current.deviceKey !== '') {
    const match = compareDeviceKeys(p.deviceKey, current.deviceKey);
    if (match === 'different') {
      const saved = parseDeviceKey(p.deviceKey);
      out.push(
        `Il preset è stato creato con un'altra configurazione audio (ingresso ${describePart(saved?.input)}, uscita ${describePart(saved?.output)}): ritardi, guadagni e calibrazione potrebbero non essere validi. Ricalibra prima di attivare l'uscita.`,
      );
    } else if (match === 'same-model') {
      out.push(
        "Il preset è stato creato con lo stesso modello di interfaccia ma su un altro dispositivo, un'altra porta USB o da un altro indirizzo dell'app: verifica la calibrazione prima di attivare l'uscita.",
      );
    }
  }
  if (p.sampleRate > 0 && current.sampleRate > 0 && p.sampleRate !== current.sampleRate) {
    out.push(
      `Il preset è stato creato a ${formatKHz(p.sampleRate)} ma ora l'audio funziona a ${formatKHz(current.sampleRate)}: la latenza e la calibrazione salvata non sono più valide. Ricalibra.`,
    );
  }
  return out;
}
