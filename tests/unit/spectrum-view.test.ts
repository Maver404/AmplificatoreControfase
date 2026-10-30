import { describe, expect, it } from 'vitest';
import { bandLevelDb, freqToX, spectrumReadout, thirdOctaveCenters, thirdOctaveLabel, xToFreq } from '../../src/ui/spectrum-view.ts';

describe('thirdOctaveCenters', () => {
  it('covers 20 Hz … 4 kHz with the base-10 series', () => {
    const c = thirdOctaveCenters();
    expect(c.length).toBe(24);
    expect(c[0]).toBeCloseTo(19.95, 2);
    expect(c[c.length - 1]).toBeCloseTo(3981, 0);
    for (let i = 1; i < c.length; i++) expect((c[i] as number) / (c[i - 1] as number)).toBeCloseTo(10 ** 0.1, 9);
  });
});

describe('third-octave band names (calib-ui-7)', () => {
  it('uses the nominal names of the base-10 centres, in Hz or kHz', () => {
    const names = thirdOctaveCenters().map(thirdOctaveLabel);
    expect(names.slice(0, 8)).toEqual(['20 Hz', '25 Hz', '31,5 Hz', '40 Hz', '50 Hz', '63 Hz', '80 Hz', '100 Hz']);
    expect(names).toContain('125 Hz');
    expect(names).toContain('315 Hz');
    expect(names.slice(-7)).toEqual(['1 kHz', '1,25 kHz', '1,6 kHz', '2 kHz', '2,5 kHz', '3,15 kHz', '4 kHz']);
    expect(names.join(' ')).not.toMatch(/k Hz/);
  });

  it('the hover readout names the band once', () => {
    const fc = thirdOctaveCenters()[6]!; // 79,4 Hz
    const thirds = spectrumReadout(82.4, ['rif. −24,3 dB'], true, fc, 3.21);
    expect(thirds).toBe('Banda 80 Hz · 82,4 Hz · rif. −24,3 dB · attenuazione +3,2 dB');
    const fine = spectrumReadout(1234, ['rif. −24,3 dB'], false, 1259, -1);
    expect(fine).toBe('1234 Hz · rif. −24,3 dB · attenuazione −1,0 dB (banda 1,25 kHz)');
    expect(spectrumReadout(82.4, [], false, fc, null)).toBe('82,4 Hz');
  });
});

describe('bandLevelDb', () => {
  const fs = 48000;
  const n = 8192;
  const binHz = fs / n;

  it('power-sums the bins inside the band', () => {
    const db = new Float64Array(n / 2 + 1).fill(-100);
    const fc = 1000;
    const k0 = Math.ceil((fc * 2 ** (-1 / 6)) / binHz);
    const k1 = Math.ceil((fc * 2 ** (1 / 6)) / binHz) - 1;
    for (let k = k0; k <= k1; k++) db[k] = -40;
    const bins = k1 - k0 + 1;
    expect(bandLevelDb(db, fs, n, fc)).toBeCloseTo(-40 + 10 * Math.log10(bins), 9);
  });

  it('uses the nearest bin when the band is narrower than a bin', () => {
    const coarse = 512;
    const db = new Float64Array(coarse / 2 + 1).fill(-90);
    const k = Math.round(25 / (fs / coarse));
    db[k] = -20;
    expect(bandLevelDb(db, fs, coarse, 25)).toBeCloseTo(-20, 9);
  });

  it('ignores non-finite bins and returns −∞ outside the data', () => {
    const db = [Number.NaN, -Infinity, -30];
    expect(bandLevelDb(db, fs, n, 4000)).toBe(-Infinity);
  });
});

describe('log axis', () => {
  it('maps 20 Hz → 0, 4 kHz → 1 and round-trips', () => {
    expect(freqToX(20)).toBeCloseTo(0, 12);
    expect(freqToX(4000)).toBeCloseTo(1, 12);
    for (const f of [31.5, 100, 1234]) expect(xToFreq(freqToX(f))).toBeCloseTo(f, 9);
  });
});
