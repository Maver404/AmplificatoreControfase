import { describe, expect, it } from 'vitest';
import {
  PRESETS_STORAGE_KEY,
  PresetStore,
  mismatchWarnings,
  sanitizePreset,
  type Preset,
} from '../../src/app/presets.ts';
import { DEFAULT_PARAMS, HEADPHONES_CEILING_DB, RANGES, clampParams, neutralTrims, type EngineParams } from '../../src/dsp/params.ts';
import { deviceKey, type DeviceOption } from '../../src/audio/devices.ts';

/** In-memory Storage stub; `fail` makes the chosen operations throw like a full or blocked storage. */
function memStorage(fail: { get?: boolean; set?: boolean } = {}): { storage: Storage; map: Map<string, string> } {
  const map = new Map<string, string>();
  const storage = {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    key: (i: number) => [...map.keys()][i] ?? null,
    getItem: (k: string) => {
      if (fail.get) throw new Error('SecurityError');
      return map.has(k) ? map.get(k)! : null;
    },
    setItem: (k: string, v: string) => {
      if (fail.set) throw new Error('QuotaExceededError');
      map.set(k, String(v));
    },
    removeItem: (k: string) => {
      map.delete(k);
    },
  };
  return { storage: storage as unknown as Storage, map };
}

const params = (over: (p: EngineParams) => void = () => {}): EngineParams => {
  const p = clampParams(DEFAULT_PARAMS);
  over(p);
  return p;
};

const META = { deviceKey: 'k1|in:A@1|out:A@2', sampleRate: 48000 };

const raw = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  id: 'p-1',
  name: 'Finestra cucina',
  createdAt: 1_700_000_000_000,
  deviceKey: META.deviceKey,
  sampleRate: 48000,
  params: params(),
  ...over,
});

describe('sanitizePreset', () => {
  it('keeps a valid preset intact', () => {
    const p = sanitizePreset(raw())!;
    expect(p).toEqual({
      id: 'p-1',
      name: 'Finestra cucina',
      createdAt: 1_700_000_000_000,
      deviceKey: META.deviceKey,
      sampleRate: 48000,
      params: params(),
    });
    expect('calibration' in p).toBe(false);
  });

  it('brings +12 dB channel gain back to 0 dB and never keeps gainUnlocked', () => {
    const p = sanitizePreset(
      raw({
        params: params((x) => {
          x.gainUnlocked = true;
          x.channels[0].gainDb = 12;
          x.channels[1].gainDb = 6;
        }),
      }),
    )!;
    expect(p.params.gainUnlocked).toBe(false);
    expect(p.params.channels[0].gainDb).toBe(RANGES.gainDb.max);
    expect(p.params.channels[1].gainDb).toBe(0);
  });

  it('clamps the ceiling to the locked range and to the headphones limit', () => {
    expect(sanitizePreset(raw({ params: { ...params(), ceilingDb: 0 } }))!.params.ceilingDb).toBe(-1);
    expect(sanitizePreset(raw({ params: { ...params(), ceilingDb: -1 } }))!.params.ceilingDb).toBe(-1);
    expect(sanitizePreset(raw({ params: { ...params(), ceilingDb: -100 } }))!.params.ceilingDb).toBe(-24);
    const hp = sanitizePreset(raw({ params: { ...params(), headphones: true, ceilingDb: -1 } }))!;
    expect(hp.params.ceilingDb).toBe(HEADPHONES_CEILING_DB);
  });

  it('replaces NaN, Infinity and wrong types with safe defaults', () => {
    const p = sanitizePreset(
      raw({
        params: {
          masterDb: Number.NaN,
          ceilingDb: Number.POSITIVE_INFINITY,
          channels: [{ gainDb: Number.NaN, delayMs: 'lots', polarityInvert: 'yes' }, null],
          filters: 42,
          phaseMode: 'magic',
          fRef: -5,
        },
      }),
    )!;
    expect(p.params.masterDb).toBe(DEFAULT_PARAMS.masterDb);
    // Infinity is not finite: it falls back to the default ceiling, never to the loudest one.
    expect(p.params.ceilingDb).toBe(DEFAULT_PARAMS.ceilingDb);
    expect(p.params.channels[0]).toEqual(DEFAULT_PARAMS.channels[0]);
    expect(p.params.channels[1]).toEqual(DEFAULT_PARAMS.channels[1]);
    expect(p.params.filters).toEqual(DEFAULT_PARAMS.filters);
    expect(p.params.phaseMode).toBe('delay');
    expect(p.params.fRef).toBe(RANGES.fRef.min);
    for (const v of [p.params.masterDb, p.params.ceilingDb, p.params.channels[0].gainDb, p.params.channels[0].delayMs]) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('rejects garbage', () => {
    for (const g of [null, undefined, 42, 'preset', true, [], [raw()], () => raw()]) {
      expect(sanitizePreset(g)).toBeNull();
    }
    expect(sanitizePreset(raw({ id: undefined }))).toBeNull();
    expect(sanitizePreset(raw({ id: '' }))).toBeNull();
    expect(sanitizePreset(raw({ id: 'has spaces' }))).toBeNull();
    expect(sanitizePreset(raw({ id: 'x'.repeat(129) }))).toBeNull();
    expect(sanitizePreset(raw({ name: 7 }))).toBeNull();
    expect(sanitizePreset(raw({ params: undefined }))).toBeNull();
    expect(sanitizePreset(raw({ params: 'loud' }))).toBeNull();
    expect(sanitizePreset(raw({ params: [1, 2] }))).toBeNull();
  });

  it('strips unknown keys at every level', () => {
    const p = sanitizePreset(
      raw({
        evil: '<script>',
        gainUnlocked: true,
        params: { ...params(), extra: 1, channels: [{ ...params().channels[0], boost: 99 }, params().channels[1]] },
      }),
    )!;
    expect(Object.keys(p).sort()).toEqual(['createdAt', 'deviceKey', 'id', 'name', 'params', 'sampleRate']);
    expect(Object.keys(p.params).sort()).toEqual(Object.keys(DEFAULT_PARAMS).sort());
    expect(Object.keys(p.params.channels[0]).sort()).toEqual(Object.keys(DEFAULT_PARAMS.channels[0]).sort());
  });

  it('falls back to neutral metadata', () => {
    const p = sanitizePreset(raw({ createdAt: -1, deviceKey: 5, sampleRate: 1e9, name: '  \u0007 ‮ ' }))!;
    expect(p.createdAt).toBe(0);
    expect(p.deviceKey).toBe('');
    expect(p.sampleRate).toBe(0);
    expect(p.name).toBe('Senza nome');
    expect(sanitizePreset(raw({ name: 'x'.repeat(500) }))!.name).toHaveLength(80);
  });

  it('keeps calibration only when it is a plain JSON object, as a deep copy', () => {
    const cal = { version: 1, latency: { samples: 1234.37, sign: -1 }, fir: [0.1, -0.2, 0], ok: true, note: null };
    const p = sanitizePreset(raw({ calibration: cal }))!;
    expect(p.calibration).toEqual(cal);
    expect(p.calibration).not.toBe(cal);
    (cal.fir as number[])[0] = 99;
    expect((p.calibration as { fir: number[] }).fir[0]).toBe(0.1);

    for (const bad of [
      [1, 2, 3],
      'cal',
      42,
      new Float64Array(4),
      new Date(0),
      { fir: new Float64Array(2) },
      { gain: Number.NaN },
      { gain: Number.POSITIVE_INFINITY },
      { f: () => 1 },
      { nested: { deeper: [1, [2, Symbol('x')]] } },
    ]) {
      expect('calibration' in sanitizePreset(raw({ calibration: bad }))!).toBe(false);
    }
  });

  it('drops __proto__ keys from calibration without touching prototypes', () => {
    const fromJson = JSON.parse('{"calibration": {"__proto__": {"polluted": true}, "a": 1}}') as { calibration: unknown };
    const p = sanitizePreset(raw({ calibration: fromJson.calibration }))!;
    expect(p.calibration).toEqual({ a: 1 });
    expect(Object.getPrototypeOf(p.calibration)).toBe(Object.prototype);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it('rejects calibration nested too deeply (or cyclic)', () => {
    const cyclic: Record<string, unknown> = { a: 1 };
    cyclic.self = cyclic;
    expect('calibration' in sanitizePreset(raw({ calibration: cyclic }))!).toBe(false);
  });
});

describe('harmonic trims in presets', () => {
  const trimmed = (): EngineParams =>
    params((p) => {
      p.phaseMode = 'tonal';
      p.channels[0].harmonicTrims[0] = { gainDb: -2.25, phaseDeg: 17.5 };
      p.channels[1].harmonicTrims[3] = { gainDb: 5.5, phaseDeg: -140 };
    });

  it('a preset saved before the trims existed loads with every trim at 0', () => {
    const old = params() as unknown as { channels: Record<string, unknown>[] };
    for (const c of old.channels) delete c['harmonicTrims'];
    const p = sanitizePreset(raw({ params: old }))!;
    for (const c of p.params.channels) expect(c.harmonicTrims).toEqual(neutralTrims());
  });

  it('sanitises trims: clamped, non-finite or wrong types → 0, padded / truncated to 8, copies', () => {
    const bad = params() as unknown as { channels: Record<string, unknown>[] };
    const src = [{ gainDb: 40, phaseDeg: -900 }, { gainDb: 'x', phaseDeg: null }, null, 7, { gainDb: -3 }];
    bad.channels[0]['harmonicTrims'] = src;
    bad.channels[1]['harmonicTrims'] = Array.from({ length: 12 }, () => ({ gainDb: -1, phaseDeg: 2 }));
    const p = sanitizePreset(raw({ params: bad }))!;
    expect(p.params.channels[0].harmonicTrims.slice(0, 5)).toEqual([
      { gainDb: RANGES.trimGainDb.max, phaseDeg: RANGES.trimPhaseDeg.min },
      { gainDb: 0, phaseDeg: 0 },
      { gainDb: 0, phaseDeg: 0 },
      { gainDb: 0, phaseDeg: 0 },
      { gainDb: -3, phaseDeg: 0 },
    ]);
    expect(p.params.channels[0].harmonicTrims).toHaveLength(8);
    expect(p.params.channels[1].harmonicTrims).toHaveLength(8);
    expect(p.params.channels[0].harmonicTrims[0]).not.toBe(src[0]);
    const notArray = params() as unknown as { channels: Record<string, unknown>[] };
    notArray.channels[1]['harmonicTrims'] = { 0: { gainDb: 3, phaseDeg: 3 } };
    expect(sanitizePreset(raw({ params: notArray }))!.params.channels[1].harmonicTrims).toEqual(neutralTrims());
  });

  it('round trip: save → storage → list → export → import keeps the trims exactly', () => {
    const { storage } = memStorage();
    const store = new PresetStore(storage);
    const saved = store.save('Tonale', trimmed(), META);
    expect(saved.params.channels[0].harmonicTrims[0]).toEqual({ gainDb: -2.25, phaseDeg: 17.5 });
    const again = new PresetStore(storage).get(saved.id)!;
    expect(again.params.channels).toEqual(trimmed().channels);
    const other = new PresetStore(null);
    expect(other.importJson(store.exportJson())).toEqual({ imported: 1, rejected: 0 });
    expect(other.get(saved.id)!.params.channels).toEqual(trimmed().channels);
    // Copies: changing one does not change the store.
    again.params.channels[1].harmonicTrims[3].gainDb = -30;
    expect(store.get(saved.id)!.params.channels[1].harmonicTrims[3].gainDb).toBe(5.5);
  });

  it('mismatchWarnings does not depend on the trims', () => {
    const plain = sanitizePreset(raw())!;
    const withTrims = sanitizePreset(raw({ params: trimmed() }))!;
    for (const cur of [{ deviceKey: META.deviceKey, sampleRate: 48000 }, { deviceKey: 'k1|in:B@1|out:B@2', sampleRate: 44100 }]) {
      expect(mismatchWarnings(withTrims, cur)).toEqual(mismatchWarnings(plain, cur));
    }
  });
});

describe('PresetStore with a Storage', () => {
  it('saves, lists, reloads from storage, gets and removes', () => {
    const { storage, map } = memStorage();
    let t = 1000;
    const store = new PresetStore(storage, { now: () => t++ });
    expect(store.list()).toEqual([]);
    expect(store.persistent).toBe(true);

    const loud = params((x) => {
      x.gainUnlocked = true;
      x.channels[0].gainDb = 12;
      x.channels[1].delayMs = 3.25;
    });
    const a = store.save('Zona B', loud, { ...META, calibration: { latencySamples: 1234.5 } });
    const b = store.save('Zona A', params(), { deviceKey: '', sampleRate: 44100 });
    expect(a.params.gainUnlocked).toBe(false);
    expect(a.params.channels[0].gainDb).toBe(0);
    expect(a.params.channels[1].delayMs).toBe(3.25);
    expect(a.createdAt).toBe(1000);
    expect(a.id).not.toBe(b.id);

    // Stored under the documented key, without the unlock.
    const stored = map.get(PRESETS_STORAGE_KEY)!;
    expect(stored).toBeDefined();
    expect(stored).not.toContain('"gainUnlocked":true');

    const reloaded = new PresetStore(storage);
    expect(reloaded.list().map((p) => p.name)).toEqual(['Zona A', 'Zona B']);
    expect(reloaded.get(a.id)).toEqual(a);
    expect(reloaded.get('missing')).toBeNull();

    reloaded.remove(a.id);
    expect(new PresetStore(storage).list().map((p) => p.id)).toEqual([b.id]);
  });

  it('returns copies that cannot corrupt the store', () => {
    const store = new PresetStore(memStorage().storage);
    const p = store.save('X', params(), META);
    p.params.channels[0].gainDb = 12;
    store.list()[0]!.params.masterDb = 0;
    expect(store.get(p.id)!.params.channels[0].gainDb).toBe(DEFAULT_PARAMS.channels[0].gainDb);
    expect(store.get(p.id)!.params.masterDb).toBe(DEFAULT_PARAMS.masterDb);
  });

  it('saving with an existing name replaces that preset and keeps its id', () => {
    const store = new PresetStore(memStorage().storage);
    const first = store.save('Finestra', params(), META);
    const second = store.save('  finestra ', params((x) => (x.masterDb = -40)), META);
    expect(second.id).toBe(first.id);
    expect(store.list()).toHaveLength(1);
    expect(store.list()[0]!.params.masterDb).toBe(-40);
    expect(store.list()[0]!.name).toBe('finestra');
    expect(store.save('', params(), META).name).toBe('Preset 2');
  });

  it('sees changes made by another store on the same storage (another tab)', () => {
    const { storage } = memStorage();
    const tab1 = new PresetStore(storage);
    const tab2 = new PresetStore(storage);
    tab1.save('Uno', params(), META);
    tab2.save('Due', params(), META);
    expect(tab1.list().map((p) => p.name)).toEqual(['Due', 'Uno']);
  });

  it('ignores invalid stored entries and backs up unreadable data', () => {
    const { storage, map } = memStorage();
    map.set(PRESETS_STORAGE_KEY, JSON.stringify({ version: 1, presets: [raw(), { id: 'bad' }, 7] }));
    expect(new PresetStore(storage).list().map((p) => p.id)).toEqual(['p-1']);

    map.set(PRESETS_STORAGE_KEY, '{not json');
    const store = new PresetStore(storage);
    expect(store.list()).toEqual([]);
    expect(map.get(`${PRESETS_STORAGE_KEY}.corrotto`)).toBe('{not json');
    store.save('Nuovo', params(), META);
    expect(new PresetStore(storage).list().map((p) => p.name)).toEqual(['Nuovo']);
  });

  it('keeps working in memory when the storage throws', () => {
    const { storage } = memStorage({ get: true, set: true });
    const store = new PresetStore(storage);
    expect(store.list()).toEqual([]);
    const p = store.save('Solo sessione', params(), META);
    expect(store.lastError).toMatch(/Impossibile salvare/);
    expect(store.persistent).toBe(false);
    expect(store.get(p.id)!.name).toBe('Solo sessione');
    expect(store.list()).toHaveLength(1);
  });

  it('a failed write keeps memory authoritative until a write succeeds', () => {
    const fail = { set: false };
    const { storage, map } = memStorage(fail);
    const store = new PresetStore(storage);
    store.save('A', params(), META);
    fail.set = true;
    store.save('B', params(), META);
    expect(store.list().map((p) => p.name)).toEqual(['A', 'B']);
    fail.set = false;
    store.save('C', params(), META);
    expect(store.lastError).toBeNull();
    expect(new PresetStore(storage).list().map((p) => p.name)).toEqual(['A', 'B', 'C']);
    expect(map.has(PRESETS_STORAGE_KEY)).toBe(true);
  });
});

describe('PresetStore without storage (null)', () => {
  it('round trips in memory', () => {
    const store = new PresetStore(null);
    expect(store.persistent).toBe(false);
    const p = store.save('Memoria', params(), META);
    expect(store.list()).toEqual([p]);
    expect(store.get(p.id)).toEqual(p);
    store.remove(p.id);
    expect(store.list()).toEqual([]);
    expect(store.lastError).toBeNull();
  });

  it('the default constructor never throws (no localStorage in Node)', () => {
    expect(() => new PresetStore().list()).not.toThrow();
  });
});

describe('export / import', () => {
  it('round trips between stores', () => {
    const src = new PresetStore(null);
    src.save('A', params((x) => (x.channels[0].delayMs = 12.5)), { ...META, calibration: { k: [1, 2] } });
    src.save('B', params(), { deviceKey: '', sampleRate: 44100 });
    const json = src.exportJson();
    expect(JSON.parse(json)).toMatchObject({ format: 'controfase-presets', version: 1 });

    const dst = new PresetStore(memStorage().storage);
    expect(dst.importJson(json)).toEqual({ imported: 2, rejected: 0 });
    expect(dst.list()).toEqual(src.list());
    // Importing the same file again replaces by id instead of duplicating.
    expect(dst.importJson(json)).toEqual({ imported: 2, rejected: 0 });
    expect(dst.list()).toHaveLength(2);
  });

  it('imports mixed valid and invalid entries, clamping the valid ones', () => {
    const store = new PresetStore(null);
    const entries = [
      raw({ id: 'ok-1' }),
      raw({ id: 'ok-2', params: { ...params(), gainUnlocked: true, ceilingDb: 0, channels: [{ gainDb: 12 }, { gainDb: Number.NaN }] } }),
      { id: 'no-params', name: 'x' },
      'string',
      null,
      raw({ id: 'bad id!' }),
      raw({ id: 'ok-3', calibration: [1, 2] }),
    ];
    expect(store.importJson(JSON.stringify({ format: 'controfase-presets', version: 1, presets: entries }))).toEqual({
      imported: 3,
      rejected: 4,
    });
    const p2 = store.get('ok-2')!;
    expect(p2.params.gainUnlocked).toBe(false);
    expect(p2.params.channels[0].gainDb).toBe(0);
    expect(p2.params.channels[1].gainDb).toBe(DEFAULT_PARAMS.channels[1].gainDb);
    expect(p2.params.ceilingDb).toBe(-1);
    expect('calibration' in store.get('ok-3')!).toBe(false);
  });

  it('accepts a bare array or a single preset', () => {
    const store = new PresetStore(null);
    expect(store.importJson(JSON.stringify([raw({ id: 'a' }), raw({ id: 'b' })]))).toEqual({ imported: 2, rejected: 0 });
    expect(store.importJson(JSON.stringify(raw({ id: 'c' })))).toEqual({ imported: 1, rejected: 0 });
    expect(store.list()).toHaveLength(3);
  });

  it('counts an unreadable file as one rejection and changes nothing', () => {
    const store = new PresetStore(null);
    store.save('Esistente', params(), META);
    expect(store.importJson('{oops')).toEqual({ imported: 0, rejected: 1 });
    expect(store.importJson('42')).toEqual({ imported: 0, rejected: 1 });
    expect(store.importJson(JSON.stringify({ presets: 'nope' }))).toEqual({ imported: 0, rejected: 1 });
    expect(store.importJson(undefined as unknown as string)).toEqual({ imported: 0, rejected: 1 });
    expect(store.list().map((p) => p.name)).toEqual(['Esistente']);
  });
});

describe('mismatchWarnings', () => {
  const inA: DeviceOption = { deviceId: 'ia', groupId: 'g', label: 'Scarlett 2i2 USB', kind: 'audioinput', pseudo: false };
  const outA: DeviceOption = { deviceId: 'oa', groupId: 'g', label: 'Scarlett 2i2 USB', kind: 'audiooutput', pseudo: false };
  const keyA = deviceKey(inA, outA);
  const preset = (over: Partial<Preset> = {}): Preset => ({ ...sanitizePreset(raw({ deviceKey: keyA }))!, ...over });

  it('is empty when device and sample rate match', () => {
    expect(mismatchWarnings(preset(), { deviceKey: keyA, sampleRate: 48000 })).toEqual([]);
  });

  it('warns about another interface, naming the saved one', () => {
    const other = deviceKey({ ...inA, deviceId: 'x', label: 'Babyface Pro' }, { ...outA, deviceId: 'y', label: 'Babyface Pro' });
    const w = mismatchWarnings(preset(), { deviceKey: other, sampleRate: 48000 });
    expect(w).toHaveLength(1);
    expect(w[0]).toContain("un'altra configurazione audio");
    expect(w[0]).toContain('«Scarlett 2i2 USB»');
  });

  it('warns more softly about the same model on another unit or port', () => {
    const twin = deviceKey({ ...inA, deviceId: 'ib' }, { ...outA, deviceId: 'ob' });
    const w = mismatchWarnings(preset(), { deviceKey: twin, sampleRate: 48000 });
    expect(w).toHaveLength(1);
    expect(w[0]).toContain('stesso modello');
  });

  it('warns about a different sample rate', () => {
    const w = mismatchWarnings(preset({ sampleRate: 44100 }), { deviceKey: keyA, sampleRate: 48000 });
    expect(w).toHaveLength(1);
    expect(w[0]).toContain('44,1 kHz');
    expect(w[0]).toContain('48 kHz');
  });

  it('reports both problems together, and a preset without device information', () => {
    const w = mismatchWarnings(preset({ sampleRate: 96000 }), { deviceKey: deviceKey(null, outA), sampleRate: 48000 });
    expect(w).toHaveLength(2);
    expect(mismatchWarnings(preset({ deviceKey: '' }), { deviceKey: keyA, sampleRate: 48000 })).toHaveLength(1);
    expect(mismatchWarnings(preset({ sampleRate: 0 }), { deviceKey: keyA, sampleRate: 48000 })).toEqual([]);
  });
});
