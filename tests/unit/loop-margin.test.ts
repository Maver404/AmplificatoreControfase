/**
 * The solvers' stability margin in tonal mode (calibration.ts enforceLoopMargin / bestLoopMargin / loopChecker)
 * and the predictions of separate zones played together (predictOutcome), on seeded random plants
 * (loop-margin-plants.ts) checked against evaluators independent of the solver: the neutralised loop on a fine
 * grid (0.25 Hz, and bw/100 around each harmonic) and the closed-loop attenuation at each harmonic.
 * The seeds are the review's repros: each case failed before its fix. Separate zones played together
 * (mergeSeparateZones): a zone the other speaker's feedback leaves below 1 dB keeps a speaker off (seeds 450, 400),
 * a zone whose delay alone beats its trims with both speakers on plays that (seed 244), and the zones' figures on what
 * they attenuate are the merged ones (seed 375).
 */
import { describe, expect, it } from 'vitest';
import { LOOP_GAIN_DANGER_DB, checkLoop, predictOutcome, solveFromCalibration } from '../../src/audio/calibration.ts';
import type { SolveOutcome } from '../../src/audio/calibration.ts';
import { clampParams } from '../../src/dsp/params.ts';
import {
  calibratedPredictions,
  gainLimitedAttenuationDb,
  harmonicPredictionText,
  harmonicPredictions,
  isLoopAdvice,
  mergeSeparateZones,
  mergeZoneSolutions,
  mergedLoopAdvice,
  mergedZonesLoop,
  usefulOutcome,
} from '../../src/ui/wizard/logic.ts';
import { closedLoopDb, fineGrid, fs, maxLoop, plantCal, plantParams, randomPlant, withCoherence } from './loop-margin-plants.ts';
import type { Plant, Taps } from './loop-margin-plants.ts';

/** The solve of plant `seed` on its full calibration. */
function solved(seed: number): { pl: ReturnType<typeof randomPlant>; cal: ReturnType<typeof plantCal>; out: SolveOutcome } {
  const pl = randomPlant(seed);
  const cal = plantCal(pl);
  return { pl, cal, out: solveFromCalibration(cal, plantParams(pl), fs) };
}

/** Closed-loop attenuation of `p` at every harmonic of the solve (dB). */
const truthAt = (s: ReturnType<typeof solved>, p = s.out.params): number[] =>
  s.out.harmonics!.map((x) => closedLoopDb(s.pl, s.cal, p, x.freqHz, s.pl.secondary));

describe('stability margin in tonal mode', () => {
  it('keeps the uniform cap when the per-harmonic cut would cost more in total (R2, seed 435)', () => {
    // The loop is −6.1 dB at harmonic 2's centre but −4.6 dB at a peak in its band that the neighbours' skirts
    // feed too: harmonic 2's trims had to drop 2.15 dB for it, against 1.39 dB off every gain.
    const s = solved(435);
    expect(usefulOutcome(s.out)).toBe(true);
    expect(s.out.harmonics!.every((x) => x.loopLimitedDb === undefined)).toBe(true);
    expect(s.out.loopLimitedDb).toBeCloseTo(1.39, 1);
    // 13.1 dB in total by the solver's model (10.1 dB with the per-harmonic cut), 12.8 dB at the dominant
    // harmonic in the independent closed loop (9.7 dB).
    expect(s.out.result.predictedAttenuationDb).toBeGreaterThan(12.9);
    expect(truthAt(s)[1]).toBeGreaterThan(12.5);
    expect(maxLoop(s.cal, s.out.params, fineGrid(s.pl, s.out.params)).db).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 0.005);
  });

  it('gives a harmonic back its own trim cut once the uniform cut leaves room (R3, seed 446)', () => {
    // Trims of h2–h8 cut, then every gain lowered 4.1 dB for a peak above h8's band: h5's trim cut (4.1 dB)
    // was then redundant, and h3/h4/h6 had room for part of theirs.
    const s = solved(446);
    expect(s.out.loopLimitedDb).toBeCloseTo(4.15, 1);
    const cut = s.out.harmonics!.map((x) => x.loopLimitedDb ?? 0);
    expect(cut[4]).toBe(0);
    const t = truthAt(s);
    // Closed loop before: h3 2.67, h4 1.89, h5 3.26, h6 1.84 dB (cut twice).
    expect(t[2]).toBeGreaterThan(5);
    expect(t[3]).toBeGreaterThan(3);
    expect(t[4]).toBeGreaterThan(7);
    expect(t[5]).toBeGreaterThan(3);
    expect(s.out.result.predictedAttenuationDb).toBeGreaterThan(4.5);
    // Within the margin everywhere (the fine grid: the give-back is checked on the whole loop).
    expect(maxLoop(s.cal, s.out.params, fineGrid(s.pl, s.out.params)).db).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 0.005);
    // The predictions are the closed loop's.
    s.out.harmonics!.forEach((x, i) => expect(Math.abs(x.predictedDb - t[i]), `harmonic ${x.h}`).toBeLessThan(0.5));
  });

  it('the advice says the gains were lowered too, and «le altre armoniche non cambiano» only when true (R4)', () => {
    // h4 and h5 have the delay alone's response back (reconcileTrims): not lowered for the loop.
    const a = solved(446).out.advice.find((x) => x.startsWith('Armoniche 2, 3, 6, 7 e 8'))!;
    expect(a).toContain('oltre alla riduzione di 4,1 dB dei guadagni dei canali');
    expect(a).not.toContain('le altre armoniche non cambiano');
    // Every trimmed harmonic listed: nothing about the others.
    const all = solved(175).out;
    expect(all.advice.some((x) => x.includes('le altre armoniche non cambiano'))).toBe(false);
  });

  it('the loop of a solve stays within the margin between the harmonics’ bands too (R6, seed 569)', () => {
    // A peak at 130.6 Hz between h1's and h2's bands read 0.11 dB over the margin on the 0.25 Hz grid: the
    // calibration grid's 2 Hz cells missed its top.
    const s = solved(569);
    expect(usefulOutcome(s.out)).toBe(true);
    const fine = maxLoop(s.cal, s.out.params, fineGrid(s.pl, s.out.params));
    expect(fine.db).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 0.005);
    // The check itself reads the refined maximum: never below the fine grid's.
    const check = checkLoop(s.cal, s.out.params, fs, 256)!;
    expect(check.afterDb).toBeGreaterThanOrEqual(fine.db - 0.005);
    expect(check.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 1e-6);
  });

  it('step 6 shows a zone’s closed-loop predictions as they are, not clipped to an open-loop bound (R5, seed 257)', () => {
    const s = solved(257);
    expect(s.out.loopLimitedDb).toBeGreaterThan(6);
    const shown = harmonicPredictions([s.out]);
    const t = truthAt(s);
    shown.forEach((x, i) => {
      expect(x.predictedDb).toBe(s.out.harmonics![i].predictedDb);
      expect(Math.abs(x.predictedDb - t[i])).toBeLessThan(0.01);
    });
    // h1: 5.39 dB; the open-loop bound of the 7.8 dB cut (gainLimitedAttenuationDb) would show 4.5 dB.
    expect(shown[0].predictedDb).toBeGreaterThan(gainLimitedAttenuationDb(s.out.loopLimitedDb!) + 0.5);
  });

  it('a harmonic whose trims the margin lowered says so in step 6', () => {
    const s = solved(446);
    const [h7] = harmonicPredictions([s.out]).filter((x) => x.h === 7);
    expect(h7.loopLimitedDb).toBeGreaterThan(0);
    expect(harmonicPredictionText(h7, true)).toMatch(/ · correzione ridotta di \d+,\d dB per la stabilità$/);
  });
});

describe('separate zones played together', () => {
  it('each zone is predicted through the loop of both speakers, not as the exact null of its own solve (R1, seed 1)', () => {
    // Zone L's speaker feeds back at −6 dB (after neutralisation) at harmonics 2 and 4; zone R was solved with
    // speaker R alone and read 60 dB («limite di calcolo per un nullo esatto») there.
    const pl = randomPlant(1);
    expect(pl.speakers).toBe('separate');
    const p = plantParams(pl);
    const outs = ([0, 1] as const).map((z) => solveFromCalibration(plantCal(pl, z), p, fs));
    const full = plantCal(pl);
    const merged = mergedZonesLoop(full, mergeZoneSolutions(p, [outs[0].params, outs[1].params]), fs, 256, []);
    const zoneR = predictOutcome(outs[1], merged.params)!;
    for (const h of [2, 4]) {
      const x = zoneR.harmonics!.find((y) => y.h === h)!;
      const truth = closedLoopDb(pl, full, merged.params, x.freqHz, [null, pl.secondary[1]]);
      expect(outs[1].harmonics!.find((y) => y.h === h)!.predictedDb).toBeGreaterThan(59);
      expect(x.predictedDb).toBeLessThan(1);
      expect(Math.abs(x.predictedDb - truth)).toBeLessThan(0.5);
      expect(x.ceiling).toBeUndefined();
    }
    // What step 6 lists for zone R (step-align: the outcomes recomputed on the merged params).
    const shown = harmonicPredictions(outs.map((o) => ({ ...o, harmonics: predictOutcome(o, merged.params)!.harmonics })));
    const line = harmonicPredictionText(shown.find((x) => x.zone === 1 && x.h === 2)!, true);
    expect(line).toMatch(/^Zona R · Armonica 2 \(80,98 Hz\): −?0,\d dB/);
    expect(line).not.toContain('limite di calcolo');
    // The zone's own params, as solved: the same prediction as its outcome.
    const own = predictOutcome(outs[1], outs[1].params)!;
    expect(own.predictedDb).toBeCloseTo(outs[1].result.predictedAttenuationDb, 9);
    expect(predictOutcome({ ...outs[1] }, merged.params)).toBeNull();
  });

  it('per-zone advice on trims cut for the loop goes with the other per-zone loop figures', () => {
    expect(isLoopAdvice("Armonica 3 (300 Hz): guadagno d'anello cassa → microfono di riferimento −2,7 dB dopo la neutralizzazione")).toBe(true);
    expect(isLoopAdvice("Armoniche 2 e 4 (162 e 324 Hz): guadagno d'anello cassa → microfono di riferimento −1,0 e −2,0 dB …")).toBe(true);
    expect(isLoopAdvice('Armonica 3 (300 Hz): la correzione necessaria esce dal campo del ritocco per armonica')).toBe(false);
    const pl = randomPlant(39);
    const p = plantParams(pl);
    const outs = ([0, 1] as const).map((z) => solveFromCalibration(plantCal(pl, z), p, fs));
    const advice = outs.flatMap((o, z) => o.advice.map((a) => `Zona ${z === 0 ? 'L' : 'R'}: ${a}`));
    expect(advice.some((a) => a.includes('le altre armoniche non cambiano') || /: Armonic(a|he) .*guadagno d'anello/.test(a))).toBe(true);
    const merged = mergedZonesLoop(plantCal(pl), mergeZoneSolutions(p, [outs[0].params, outs[1].params]), fs, 256, advice);
    expect(merged.limitedDb).toBeGreaterThan(4.5);
    expect(merged.advice.some((a) => /^Zona [LR]: Armonic(a|he) .*guadagno d'anello/.test(a))).toBe(false);
    expect(clampParams(merged.params).channels[0].gainDb).toBeLessThan(outs[0].params.channels[0].gainDb);
  });

  /** Plant `seed`'s zones solved one at a time (step-align solveZones), and their advice prefixed by zone. */
  function zones(seed: number): { pl: Plant; p: ReturnType<typeof plantParams>; full: ReturnType<typeof plantCal>; outs: SolveOutcome[]; advice: string[] } {
    const pl = randomPlant(seed);
    expect(pl.speakers).toBe('separate');
    const p = plantParams(pl);
    const outs = ([0, 1] as const).map((z) => solveFromCalibration(plantCal(pl, z), p, fs));
    const advice = outs.flatMap((o, z) => o.advice.map((a) => `Zona ${z === 0 ? 'L' : 'R'}: ${a}`));
    return { pl, p, full: plantCal(pl), outs, advice };
  }

  /** Zone z's closed-loop attenuation at f (its own error mic, coherence included), independent of the solver. */
  const truthIn = (pl: Plant, cal: ReturnType<typeof plantCal>, p: ReturnType<typeof plantParams>, z: 0 | 1, f: number): number =>
    withCoherence(pl, f, closedLoopDb(pl, cal, p, f, z === 0 ? [pl.secondary[0], null] : [null, pl.secondary[1]]));

  const loopLines = (advice: readonly string[]): string[] => advice.filter((a) => /guadagno d'anello/i.test(a));

  it('a zone the other speaker leaves below 1 dB keeps its speaker off, and the other zone plays alone (seed 450)', () => {
    const { pl, p, full, outs, advice } = zones(450);
    expect(outs.every((o) => usefulOutcome(o))).toBe(true);
    // Both playing, zone L would get 0.82 dB (the independent closed loop agrees, harmonic by harmonic).
    const both = mergedZonesLoop(full, mergeZoneSolutions(p, [outs[0].params, outs[1].params]), fs, 256, []);
    const l = predictOutcome(outs[0], both.params)!;
    expect(l.predictedDb).toBeLessThan(1);
    for (const x of l.harmonics!) expect(Math.abs(x.predictedDb - truthIn(pl, full, both.params, 0, x.freqHz))).toBeLessThan(0.05);
    const m = mergeSeparateZones(full, p, outs, fs, 256, advice);
    expect(m.refused).toBe(false);
    expect(m.on).toEqual([1]);
    expect(m.params.channels[0].enabled).toBe(false);
    // Speaker R alone keeps its own solve's margin: its values and its prediction (2.4 dB) as solved.
    expect(m.limitedDb).toBe(0);
    expect(m.params.channels[1]).toEqual(outs[1].params.channels[1]);
    expect(m.shown[0]).toBeNull();
    expect(m.predictedDb).toBeCloseTo(outs[1].result.predictedAttenuationDb, 9);
    expect(m.predictedDb!).toBeGreaterThan(2);
    for (const x of m.shown[1]!.harmonics!) expect(Math.abs(x.predictedDb - truthIn(pl, full, m.params, 1, x.freqHz))).toBeLessThan(0.05);
    // Step 6's list: zone R's harmonics only; no per-harmonic prediction for the controller (separate zones).
    const list = harmonicPredictions(m.shown);
    expect(list).toHaveLength(outs[1].harmonics!.length);
    expect(list.every((x) => x.zone === 1)).toBe(true);
    expect(calibratedPredictions(list)).toBeNull();
    // Why speaker L is off (zone L, the weaker with both on, is also the weaker alone: 1.1 against 2.4 dB), then the
    // loop of speaker R alone (not «entrambe le casse»); zone L's figures on what it attenuates go with its speaker.
    const drop =
      "Zona L: con entrambe le casse attive, che tornano insieme al microfono di riferimento, l'attenuazione prevista scende a 0,8 dB: " +
      'la cassa L resta spenta.';
    const [loop, ...more] = loopLines(m.advice);
    expect(more).toEqual([]);
    expect(loop).toMatch(/^Con la sola cassa R attiva il guadagno d'anello cassa → microfono di riferimento è −6,1 dB a 621 Hz: /);
    expect(loop).toContain('allontana la cassa R dal microfono di riferimento');
    expect(m.advice.indexOf(drop)).toBeGreaterThanOrEqual(0);
    expect(m.advice.indexOf(drop)).toBeLessThan(m.advice.indexOf(loop!));
    expect(m.advice.filter((a) => a.startsWith('Zona L: ') && /attenuazione prevista/i.test(a))).toEqual([drop]);
    expect(m.advice.some((a) => a.startsWith('Zona L: Canale L: serve un guadagno'))).toBe(true);
    expect(m.advice.some((a) => a.includes('nessuna attenuazione utile'))).toBe(false);
    // Zone R's own figure, unchanged alone.
    expect(m.advice).toContain('Zona R: ' + outs[1].advice.find((a) => a.startsWith('Attenuazione prevista solo 2,4 dB'))!);
  });

  it('the zone kept is the one that gives more alone (seed 400): speaker R off, zone L alone at 1.6 dB', () => {
    // With both on zone L falls to 0.78 dB; alone zone L gives 1.64 dB and zone R 1.33 dB: zone L plays, speaker R is
    // off (the old rule switched the weaker zone with both on off: zone R alone, 1.33 dB).
    const { p, full, outs, advice } = zones(400);
    const m = mergeSeparateZones(full, p, outs, fs, 256, advice);
    expect(m.refused).toBe(false);
    expect(m.on).toEqual([0]);
    expect(m.params.channels[1].enabled).toBe(false);
    expect(m.predictedDb).toBeCloseTo(outs[0].result.predictedAttenuationDb, 9);
    expect(m.predictedDb!).toBeGreaterThan(outs[1].result.predictedAttenuationDb);
    expect(m.advice).toContain(
      "Zona R: con entrambe le casse attive, che tornano insieme al microfono di riferimento, l'attenuazione prevista della zona L " +
        'scende a 0,8 dB: la cassa R resta spenta, perché da sola la zona L attenua di più (1,6 dB contro 1,3 dB).',
    );
    expect(loopLines(m.advice).map((a) => a.slice(0, 28))).toEqual(['Con la sola cassa L attiva i']);
    expect(m.advice.filter((a) => a.startsWith('Zona R: ') && /attenuazione prevista solo/i.test(a))).toEqual([]);
    expect(harmonicPredictions(m.shown).every((x) => x.zone === 0)).toBe(true);
  });

  it('both speakers stay on when each zone keeps at least 1 dB with both playing (seed 1)', () => {
    const { p, full, outs, advice } = zones(1);
    const m = mergeSeparateZones(full, p, outs, fs, 256, advice);
    const merged = mergedZonesLoop(full, mergeZoneSolutions(p, [outs[0].params, outs[1].params]), fs, 256, advice);
    expect(m.on).toEqual([0, 1]);
    expect(m.refused).toBe(false);
    expect(m.params).toEqual(merged.params);
    // Zone R gives 6.1 dB alone, 2.7 dB with both on: its weak-attenuation line is the merged figure's.
    const weakR = 'Zona R: Attenuazione prevista solo 2,7 dB: controlla posizionamento, banda dei filtri e causalità prima di attivare l\'uscita.';
    expect(advice.some((a) => a.startsWith('Zona R: Attenuazione prevista solo'))).toBe(false);
    expect(m.advice.filter((a) => a !== weakR)).toEqual(merged.advice);
    expect(m.advice.indexOf(weakR)).toBe(m.advice.findLastIndex((a) => a.startsWith('Zona L: ')) + 1);
    const preds = outs.map((o) => predictOutcome(o, merged.params)!.predictedDb);
    expect(m.shown.map((o) => o!.result.predictedAttenuationDb)).toEqual(preds);
    expect(m.predictedDb).toBe(Math.min(...preds));
    expect(m.predictedDb!).toBeGreaterThanOrEqual(1);
    expect(loopLines(m.advice).map((a) => a.slice(0, 28))).toEqual(['Con entrambe le casse attive']);
    expect(m.advice.some((a) => a.includes('resta spenta'))).toBe(false);
    expect([...new Set(harmonicPredictions(m.shown).map((x) => x.zone))]).toEqual([0, 1]);
  });

  it('with both speakers on, a zone’s lines on what it attenuates are the merged figures, as in the headline and the list (seed 375)', () => {
    // Alone zone R gives 5.7 dB, with both on 4.7 dB; zone L's out-of-range harmonics fall from 7.1–19 dB to 4.5–8.1 dB.
    const { p, full, outs, advice } = zones(375);
    const m = mergeSeparateZones(full, p, outs, fs, 256, advice);
    expect(m.on).toEqual([0, 1]);
    for (const z of [0, 1] as const) {
      const head = `Zona ${z === 0 ? 'L' : 'R'}: `;
      const o = m.shown[z]!;
      const weak = m.advice.filter((a) => a.startsWith(`${head}Attenuazione prevista solo`));
      expect(weak, head).toEqual(
        o.result.predictedAttenuationDb < 6 ? [`${head}Attenuazione prevista solo ${o.result.predictedAttenuationDb.toFixed(1).replace('.', ',')} dB: controlla posizionamento, banda dei filtri e causalità prima di attivare l'uscita.`] : [],
      );
      const capped = o.harmonics!.filter((x) => x.capped);
      const line = m.advice.filter((a) => a.startsWith(head) && a.includes('esce dal campo del ritocco'));
      expect(line.length, head).toBe(capped.length > 0 ? 1 : 0);
      const v = capped.map((x) => x.predictedDb.toFixed(1).replace('.', ','));
      if (capped.length > 0) expect(line[0]).toContain(`attenuazione prevista ${v.length > 1 ? `${v.slice(0, -1).join(', ')} e ${v.at(-1)}` : v[0]} dB`);
    }
    expect(advice.some((a) => a.startsWith('Zona R: Attenuazione prevista solo 5,7 dB'))).toBe(true);
    expect(m.advice.some((a) => a.startsWith('Zona R: Attenuazione prevista solo 4,7 dB'))).toBe(true);
    expect(m.predictedDb!).toBeCloseTo(Math.min(...m.shown.map((o) => o!.result.predictedAttenuationDb)), 12);
  });

  it('a zone whose delay alone beats its trims with both speakers on plays the delay alone (seed 244)', () => {
    // Zone R's trims null its harmonics with speaker R alone (60 dB) but give 2.4 dB with both on; its delay alone,
    // moved onto the merged params, 7.8 dB; zone L keeps 18.9 dB. The plant's own closed loop agrees.
    const { pl, p, full, outs, advice } = zones(244);
    const m = mergeSeparateZones(full, p, outs, fs, 256, advice);
    expect(m.on).toEqual([0, 1]);
    const r = m.shown[1]!;
    expect(r.result.predictedAttenuationDb).toBeGreaterThan(7.5);
    expect(m.shown[0]!.result.predictedAttenuationDb).toBeGreaterThan(18.5);
    expect(m.predictedDb!).toBeGreaterThan(7.5);
    // Zone R's channel is its delay alone (the solve's delayOnly) with the merged cut.
    const d = outs[1].delayOnly!.params.channels[1];
    expect(m.params.channels[1].harmonicTrims).toEqual(d.harmonicTrims);
    expect(m.params.channels[1].delayMs).toBe(d.delayMs);
    expect(m.params.channels[1].gainDb).toBeCloseTo(d.gainDb - m.limitedDb, 9);
    for (const x of r.harmonics!) {
      expect(x.predictedDb).toBe(x.untrimmedDb);
      expect(x.capped).toBe(false);
      expect(Math.abs(x.predictedDb - truthIn(pl, full, m.params, 1, x.freqHz))).toBeLessThan(0.05);
    }
    expect(m.advice).toContain(
      'Zona R: con entrambe le casse attive le correzioni per armonica attenuerebbero 2,4 dB, solo ritardo e guadagno 7,8 dB: ' +
        'la zona R usa solo ritardo e guadagno.',
    );
    expect(checkLoop(full, m.params, fs, 256)!.afterDb).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + 1e-6);
  });

  it('a zone without a useful solution keeps its speaker off, said once; without any, nothing plays', () => {
    const { p, full, outs, advice } = zones(26);
    const weakL: SolveOutcome = { ...outs[0], result: { ...outs[0].result, predictedAttenuationDb: 0.5 } };
    expect(usefulOutcome(weakL)).toBe(false);
    const m = mergeSeparateZones(full, p, [weakL, outs[1]], fs, 256, advice);
    expect(m.on).toEqual([1]);
    expect(m.refused).toBe(false);
    expect(m.params.channels[0].enabled).toBe(false);
    const note = 'Zona L: nessuna attenuazione utile, la cassa L resta spenta.';
    expect(m.advice.filter((a) => a.includes('resta spenta'))).toEqual([note]);
    expect(m.advice.indexOf(note)).toBeLessThan(m.advice.findIndex((a) => a.startsWith('Con la sola cassa R attiva')));
    expect(m.predictedDb).toBeCloseTo(outs[1].result.predictedAttenuationDb, 9);
    const none = mergeSeparateZones(full, p, [null, null], fs, 256, []);
    expect(none).toMatchObject({ on: [], shown: [null, null], predictedDb: null, refused: false, loopGainDb: null, advice: [] });
    expect(none.params.channels.map((c) => c.enabled)).toEqual([false, false]);
  });

  it('refuses only when what is left stays below 1 dB, and then switches no speaker off', () => {
    // A safety net (on the paths they were solved with, a zone alone keeps its own solve's ≥ 1 dB): the zones
    // merged on speaker → reference paths 18 dB louder, whose loop the cap brings back by cutting every gain.
    const { pl, p, outs, advice } = zones(26);
    const louder = (t: Taps | null): Taps | null => (t === null ? null : t.map(([lag, g]) => [lag, 8 * g] as [number, number]));
    const loud = plantCal({ ...pl, feedback: [louder(pl.feedback[0]), louder(pl.feedback[1])] });
    const m = mergeSeparateZones(loud, p, outs, fs, 256, advice);
    expect(m.refused).toBe(true);
    expect(m.on).toEqual([1]);
    expect(m.predictedDb!).toBeLessThan(1);
    expect(m.advice.some((a) => a.includes('resta spenta'))).toBe(false);
    expect(m.advice.at(-1)).toBe(
      "Zona R: tenendo conto di ciò che la cassa rimanda al microfono di riferimento, l'attenuazione prevista scende a 0,6 dB: " +
        'i parametri non sono stati cambiati.',
    );
    // One useful zone from the start: the same, without the note on the other one.
    const one = mergeSeparateZones(loud, p, [null, outs[1]], fs, 256, advice);
    expect(one.refused).toBe(true);
    expect(one.advice.some((a) => a.includes('resta spenta'))).toBe(false);
  });

  it('the merged loop statement names the speakers that play', () => {
    const c = { rawDb: -2, atHz: 120, afterDb: -4 };
    expect(mergedLoopAdvice(c, 256)).toMatch(/^Rischio di innesco: con entrambe le casse attive .* Allontana o orienta le casse lontano/);
    expect(mergedLoopAdvice(c, 256, 1)).toBe(
      "Rischio di innesco: con la sola cassa R attiva il guadagno d'anello cassa → microfono di riferimento è −2,0 dB a 120 Hz e resta " +
        '−4,0 dB anche con la neutralizzazione del feedback. Allontana o orienta la cassa R lontano dal microfono di riferimento, oppure ' +
        'riduci il guadagno.',
    );
    expect(mergedLoopAdvice({ rawDb: -10, atHz: 120, afterDb: -30 }, 256, 0)).toBe(
      "Con la sola cassa L attiva il guadagno d'anello cassa → microfono di riferimento è −10,0 dB a 120 Hz: la neutralizzazione del " +
        'feedback (automatica, 256 tap) lo porta a −30,0 dB.',
    );
    expect(mergedLoopAdvice({ rawDb: -25, atHz: 120, afterDb: -40 }, 256, 0)).toBeNull();
  });
});
