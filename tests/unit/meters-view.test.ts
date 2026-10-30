import { describe, expect, it } from 'vitest';
import { ATTENUATION_NO_CHANGE_DB, ATTENUATION_STALE_S, attenuationReport } from '../../src/ui/meters-view.ts';

describe('attenuationReport', () => {
  it('needs the error mic and a measurement', () => {
    expect(attenuationReport(false, 12)).toMatchObject({ text: '—', state: 'none' });
    expect(attenuationReport(true, null)).toMatchObject({ text: '—', state: 'none' });
    expect(attenuationReport(true, Number.NaN)).toMatchObject({ text: '—', state: 'none' });
    // It names the input the error mic can use: 2, or 1 when the reference is on 2 (calib-ui-6).
    expect(attenuationReport(false, null).note).toBe('Serve il mic d’errore (ingresso 2) per misurare l’attenuazione.');
    expect(attenuationReport(false, null, null, 1).note).toBe('Serve il mic d’errore (ingresso 1) per misurare l’attenuazione.');
  });

  it('treats |A| < 0.05 dB as no change, without a negative zero', () => {
    for (const a of [0, -0, 0.04, -0.04, -0.049999, 0.049999]) {
      const r = attenuationReport(true, a);
      expect(r.state).toBe('flat');
      expect(r.text).toBe('0,0 dB');
      expect(r.note).not.toMatch(/aumentato/);
    }
    expect(ATTENUATION_NO_CHANGE_DB).toBe(0.05);
  });

  it('reports a worse zone only from −0.05 dB down', () => {
    const r = attenuationReport(true, -0.05);
    expect(r.state).toBe('worse');
    expect(r.text).toBe('−0,1 dB');
    expect(r.note).toMatch(/^Il rumore nella zona è aumentato/);
    expect(attenuationReport(true, -3.26).text).toBe('−3,3 dB');
  });

  it('grades positive attenuation', () => {
    expect(attenuationReport(true, 0.05)).toMatchObject({ state: 'fair', text: '+0,1 dB' });
    expect(attenuationReport(true, 9.94)).toMatchObject({ state: 'fair', text: '+9,9 dB' });
    expect(attenuationReport(true, 10)).toMatchObject({ state: 'good', text: '+10,0 dB' });
  });

  it('flags a value measured against an old bypass reference (calib-11)', () => {
    expect(attenuationReport(true, 12, 30).stale).toBe(false);
    expect(attenuationReport(true, 12, ATTENUATION_STALE_S).stale).toBe(false);
    const r = attenuationReport(true, 12, 75);
    expect(r).toMatchObject({ text: '+12,0 dB', state: 'good', stale: true });
    expect(r.note).toMatch(/Riferimento di bypass di 75 s fa: premi B per aggiornarlo\.$/);
    expect(attenuationReport(true, -2, 600).note).toMatch(/di 10 min fa/);
    // Nothing to flag without a value.
    expect(attenuationReport(true, null, 600).stale).toBe(false);
    expect(attenuationReport(false, 12, 600).stale).toBe(false);
  });
});
