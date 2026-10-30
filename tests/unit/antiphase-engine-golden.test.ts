import { describe, expect, it } from 'vitest';
import { AntiPhaseEngine } from '../../src/dsp/antiphase-engine.ts';
import { DECIMATE, GOLDEN_SCENARIOS } from './antiphase-engine.golden.ts';
import type { EngineFactory } from './antiphase-engine.golden.ts';
import { GOLDEN_DATA } from './antiphase-engine.golden-data.ts';

const make: EngineFactory = (fs, opts) => new AntiPhaseEngine(fs, opts);

describe('AntiPhaseEngine: with every harmonic trim at 0 the outputs are those of the engine before trims', () => {
  for (const sc of GOLDEN_SCENARIOS) {
    it(sc.name, () => {
      const want = GOLDEN_DATA[sc.name];
      expect(want).toBeDefined();
      const got = sc.run(make);
      expect(got.L.length).toBe(want.L.length);
      // every DECIMATE-th sample within 1e-12 (float32 outputs: in practice bit-identical)
      for (const ch of ['L', 'R'] as const) {
        for (let i = 0; i < want[ch].length; i++) {
          expect(Math.abs(got[ch][i] - want[ch][i]), `${ch}[${i * DECIMATE}]`).toBeLessThan(1e-12);
        }
      }
      // the checksums cover every sample: 1e-12 per sample over ≤ 1.6 s is ≤ 1e-7
      for (const key of ['L', 'R', 'cond', 'measRef'] as const) {
        for (let k = 0; k < 3; k++) {
          expect(Math.abs(got.sums[key][k] - want.sums[key][k]), `${key} checksum ${k}`).toBeLessThan(1e-7);
        }
      }
    });
  }
});
