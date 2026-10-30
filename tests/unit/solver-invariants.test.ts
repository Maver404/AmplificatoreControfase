/**
 * Invariants of the tonal solve (calibration.ts finishSolve / tonalSolution: the trims judged against the delay alone
 * under its own stability margin) on the seeded random plants of loop-margin-plants.ts, each zone of separate zones
 * solved alone as step 6 does, checked against evaluators independent of the solver: the plant's own closed loop at
 * every harmonic and in total (closedLoopDbAt, closedLoopTotalDb) and the neutralised loop on a fine grid
 * (fineLoopMax). Over seeds 0–599 (763 tonal solves) before the fix: 38 outcomes that refused («i parametri non sono
 * stati cambiati») read as useful with the delay alone's open-loop figure (seed 126: 11.2 dB, the trims −3.5 dB), 41
 * predicted less in total than the delay alone under its margin (9 of them refused although it gave 1.2–10.4 dB),
 * and 63 left 95 harmonics more than 0.05 dB below it. The sweep samples every SEED_STEP-th seed and the repros (all
 * 600 seeds take about a minute).
 */
import { describe, expect, it } from 'vitest';
import { LOOP_GAIN_DANGER_DB, capLoopGain, predictOutcome, solutionToParams, solveFromCalibration } from '../../src/audio/calibration.ts';
import type { CalibrationData, SolveOutcome } from '../../src/audio/calibration.ts';
import { RANGES, clampParams, withoutTrims } from '../../src/dsp/params.ts';
import type { EngineParams } from '../../src/dsp/params.ts';
import { usefulOutcome } from '../../src/ui/wizard/logic.ts';
import { closedLoopDbAt, closedLoopTotalDb, fineLoopMax, fs, plantCal, plantParams, randomPlant, withCoherence } from './loop-margin-plants.ts';
import type { Plant, Taps } from './loop-margin-plants.ts';

/** Every SEED_STEP-th seed of 0–599 is solved (and the repros below). */
const SEED_STEP = 5;
/**
 * The reviews' repros: refused but read as useful (5 … 341; 169 zone R, 177 both zones), a refusal although the delay
 * alone gave 10.4 dB (126), harmonics below the delay alone (41, 58, 257, 446, 580), the slowest solves (446, 569);
 * then a uniform cut kept after the loop no longer needed it (139, 211, 743 zone R, 1057), advice quoting a loop below
 * the margin as above it (437 zone L, 1057), a give-back whose lowered trims stopped just over the margin (39 zone R,
 * 58 zone L, 277, 505 zone L, 580 zone R), a give-back refused for a cut of 2e-5 dB (492 zone L, 945, 1105 zone R) and
 * a sub-1 dB solve refused as a howl risk for a cut of 1e-4 dB (21).
 */
const REPROS = [
  5, 21, 22, 39, 41, 58, 106, 112, 123, 126, 139, 148, 155, 169, 177, 185, 211, 228, 257, 277, 292, 307, 341, 437, 446, 492, 505,
  569, 580, 743, 945, 1057, 1105,
];
/** Two predictions closer than this are the same, dB (the solver's comparisons are exact; this is the test's slack). */
const SAME_DB = 0.05;
/**
 * Prediction against the plant's closed loop, as amplitude ratios |R|/|P| (with the coherence ceiling): the model holds
 * the very paths the plant stores (exact DTFTs of the same IRs, the same neutraliser), the primary path interpolated
 * between grid points after removing its bulk delay (exact for the plants' pure delay up to the delay estimate). They
 * agree to ~1e-13; only at an exact null does the evaluator's 80 dB clamp read 59.96 dB through the 60 dB ceiling
 * (5e-6 in amplitude). A figure is wrong long before 1e-5 (−100 dB).
 */
const SAME_AMPLITUDE = 1e-5;
/** Allowance of the fine-grid loop over the margin (the solver's check is sampled and refined, the grid is 0.25 Hz). */
const LOOP_ALLOWANCE_DB = 0.005;

interface Case {
  seed: number;
  zone: 0 | 1 | null;
  pl: Plant;
  cal: CalibrationData;
  p: EngineParams;
  out: SolveOutcome;
  /** The zone's speaker → error-mic paths (the other zone's left out). */
  secondary: [Taps | null, Taps | null];
}

let solved: Case[] | null = null;
/** The tonal solves of the sampled plants (each zone of separate zones alone), made once. */
function cases(): Case[] {
  if (solved !== null) return solved;
  const seeds = new Set<number>(REPROS);
  for (let s = 0; s < 600; s += SEED_STEP) seeds.add(s);
  solved = [...seeds]
    .sort((a, b) => a - b)
    .flatMap((seed) => {
      const pl = randomPlant(seed);
      const p = plantParams(pl);
      const zones = pl.speakers === 'separate' ? ([0, 1] as const) : ([null] as const);
      return zones.map((zone): Case => {
        const cal = zone === null ? plantCal(pl) : plantCal(pl, zone);
        const secondary: Case['secondary'] = zone === null ? pl.secondary : zone === 0 ? [pl.secondary[0], null] : [null, pl.secondary[1]];
        return { seed, zone, pl, cal, p, out: solveFromCalibration(cal, p, fs), secondary };
      });
    });
  return solved;
}

const caseOf = (seed: number, zone: 0 | 1 | null = null): Case => cases().find((c) => c.seed === seed && c.zone === zone)!;
const name = (c: Case): string => `seed ${c.seed}${c.zone === null ? '' : ` zone ${c.zone === 0 ? 'L' : 'R'}`}`;
const refusal = (o: SolveOutcome): boolean => o.advice.some((a) => a.includes('i parametri non sono stati cambiati'));
const amplitude = (db: number): number => Math.pow(10, -db / 20);

/**
 * The delay alone under the uniform cut of the margin (capLoopGain), rebuilt with the public API from the solve's
 * delay/gain/polarity without trims: an applied outcome's delay alone without its cuts, or a refused outcome's
 * result (left unscaled). Never better than the solver's own delay alone, whose margin is the better of this and
 * the per-harmonic one.
 */
function uniformDelayOnly(c: Case): EngineParams {
  const bare = c.out.delayOnly !== undefined ? withoutTrims(c.out.delayOnly.params) : withoutTrims(solutionToParams(c.p, c.out.result, fs).params);
  return capLoopGain(c.cal, bare, fs, 256).params;
}

/** The outcome's params with harmonic h given the delay alone's response back (the trims reproducing it), null outside the trims' range. */
function givenBack(o: SolveOutcome, h: number): EngineParams | null {
  const d = o.delayOnly!.params;
  const q = clampParams(o.params);
  for (const ch of [0, 1] as const) {
    if (!q.channels[ch].enabled) continue;
    const gainDb = d.channels[ch].harmonicTrims[h - 1].gainDb + d.channels[ch].gainDb - q.channels[ch].gainDb;
    if (gainDb > RANGES.trimGainDb.max + 1e-9 || gainDb < RANGES.trimGainDb.min - 1e-9) return null;
    q.channels[ch].harmonicTrims[h - 1] = { gainDb, phaseDeg: d.channels[ch].harmonicTrims[h - 1].phaseDeg };
  }
  return clampParams(q);
}

/** Harmonic h's trims of `p` lowered by `db` on every enabled channel. */
function lowerTrim(p: EngineParams, h: number, db: number): EngineParams {
  const q = clampParams(p);
  for (const c of q.channels) if (c.enabled) c.harmonicTrims[h - 1].gainDb -= db;
  return clampParams(q);
}

/** Every enabled channel gain of `p` lowered by `db`. */
function lowered(p: EngineParams, db: number): EngineParams {
  const q = clampParams(p);
  for (const c of q.channels) if (c.enabled) c.gainDb -= db;
  return clampParams(q);
}

describe('tonal solve invariants on seeded plants', () => {
  it('an outcome that leaves the params unchanged is never useful; a useful one is predicted by its own model', () => {
    let refused = 0;
    for (const c of cases()) {
      const o = c.out;
      if (refusal(o)) {
        refused++;
        expect(usefulOutcome(o), name(c)).toBe(false);
        expect(o.result.predictedAttenuationDb, name(c)).toBeLessThan(1);
        expect(o.params, name(c)).toEqual(c.p);
        expect(o.harmonics, name(c)).toBeUndefined();
        expect(predictOutcome(o, o.params), name(c)).toBeNull();
        continue;
      }
      expect(usefulOutcome(o), name(c)).toBe(true);
      const own = predictOutcome(o, o.params);
      expect(own, name(c)).not.toBeNull();
      expect(own!.predictedDb, name(c)).toBe(o.result.predictedAttenuationDb);
      expect(own!.harmonics, name(c)).toEqual(o.harmonics);
    }
    expect(refused).toBeGreaterThan(5);
  }, 60000);

  it('the review’s refusals: the figure refused is the one quoted, and the delay alone is applied where it gives more', () => {
    // Seed 126: refused at −3.5 dB with the trims while reading 11.2 dB (the delay alone, open loop, no margin).
    const s126 = caseOf(126).out;
    expect(usefulOutcome(s126)).toBe(true);
    expect(s126.result.predictedAttenuationDb).toBeGreaterThan(10.3);
    expect(s126.delayOnly!.predictedDb).toBeGreaterThan(10.3);
    // Seeds whose every candidate stays below 1 dB: refused, quoting what was refused.
    for (const c of [caseOf(5), caseOf(106), caseOf(177, 0), caseOf(177, 1), caseOf(228, 1)]) {
      expect(refusal(c.out), name(c)).toBe(true);
      const quoted = c.out.advice.find((a) => a.includes('i parametri non sono stati cambiati'))!;
      expect(quoted, name(c)).toContain(`(prevista ${c.out.result.predictedAttenuationDb.toFixed(1).replace('.', ',').replace('-', '−')} dB)`);
    }
  }, 60000);

  it('never predicts less in total than the delay alone under the margin', () => {
    for (const c of cases()) {
      const o = c.out;
      // The delay alone under a uniform cut, evaluated on the plant itself.
      const uniform = closedLoopTotalDb(c.pl, c.cal, uniformDelayOnly(c), c.secondary);
      if (refusal(o)) {
        // Refused only when the delay alone gives nothing either.
        expect(uniform, name(c)).toBeLessThan(1 + SAME_DB);
        continue;
      }
      expect(o.delayOnly, name(c)).toBeDefined();
      expect(o.result.predictedAttenuationDb, name(c)).toBeGreaterThanOrEqual(o.delayOnly!.predictedDb - SAME_DB);
      expect(o.delayOnly!.predictedDb, name(c)).toBeGreaterThanOrEqual(uniform - SAME_DB);
      expect(closedLoopTotalDb(c.pl, c.cal, o.params, c.secondary), name(c)).toBeGreaterThanOrEqual(uniform - SAME_DB);
    }
  }, 60000);

  it('per harmonic, never below the delay alone unless giving it back would break the margin and a uniform cut would cost more', () => {
    // The rule tested: a harmonic more than SAME_DB below untrimmedDb (the delay alone under its own margin) is one
    // whose delay-alone response, given back on the params returned, needs a trim outside RANGES.trimGainDb, or
    // takes the neutralised loop (fine grid) above the margin, and then lowering every gain by that excess predicts
    // no more in total than the params returned (than the cut itself, GIVE_BACK_CUT_GAIN_DB = SAME_DB at most), and
    // lowering that harmonic's trims just into the margin does not predict more both there and in total.
    let below = 0;
    for (const c of cases()) {
      const o = c.out;
      if (o.harmonics === undefined) continue;
      const own = predictOutcome(o, o.delayOnly!.params)!;
      let fine: ReturnType<typeof fineLoopMax> | null = null;
      o.harmonics.forEach((x, i) => {
        const at = `${name(c)} h${x.h}`;
        // untrimmedDb is the delay alone's prediction at that harmonic.
        expect(x.untrimmedDb, at).toBe(own.harmonics![i].predictedDb);
        if (!(x.predictedDb < x.untrimmedDb - SAME_DB)) return;
        below++;
        const back = givenBack(o, x.h);
        if (back === null) return;
        fine ??= fineLoopMax(c.pl, c.cal, o.params);
        const excess = fine(back).db - LOOP_GAIN_DANGER_DB;
        expect(excess, at).toBeGreaterThan(-LOOP_ALLOWANCE_DB);
        expect(predictOutcome(o, lowered(back, Math.max(0, excess)))!.predictedDb, at).toBeLessThanOrEqual(
          o.result.predictedAttenuationDb + Math.min(SAME_DB, Math.max(0, excess)) + 1e-3,
        );
        // The give-back with h's trims lowered (bisection on the fine grid) until the loop fits.
        let lo = 0;
        let hi = 0.01;
        while (fine(lowerTrim(back, x.h, hi)).db > LOOP_GAIN_DANGER_DB && hi < 30) hi *= 2;
        if (!(hi < 30)) return;
        for (let k = 0; k < 30; k++) {
          const mid = (lo + hi) / 2;
          if (fine(lowerTrim(back, x.h, mid)).db > LOOP_GAIN_DANGER_DB) lo = mid;
          else hi = mid;
        }
        const alt = predictOutcome(o, lowerTrim(back, x.h, hi))!;
        const better = alt.harmonics![i].predictedDb > x.predictedDb + SAME_DB && alt.predictedDb > o.result.predictedAttenuationDb + SAME_DB;
        expect(better, `${at}: ${x.predictedDb.toFixed(2)} dB, lowered give-back ${alt.harmonics![i].predictedDb.toFixed(2)} dB`).toBe(false);
      });
    }
    // A few % of the harmonics, most of them in the repros picked for it (all 600 seeds: 28 of the 763 solves'
    // harmonics, in 20 solves).
    expect(below).toBeLessThan(30);
  }, 60000);

  it('the predictions are the plant’s own closed loop, per harmonic (trims and delay alone) and in total', () => {
    for (const c of cases()) {
      const o = c.out;
      if (o.harmonics === undefined) continue;
      const freqs = Float64Array.from(o.harmonics, (x) => x.freqHz);
      const trimmed = closedLoopDbAt(c.pl, c.cal, o.params, freqs, c.secondary);
      const plain = closedLoopDbAt(c.pl, c.cal, o.delayOnly!.params, freqs, c.secondary);
      o.harmonics.forEach((x, i) => {
        const at = `${name(c)} h${x.h}`;
        expect(Math.abs(amplitude(x.predictedDb) - amplitude(withCoherence(c.pl, x.freqHz, trimmed[i]))), at).toBeLessThan(SAME_AMPLITUDE);
        expect(Math.abs(amplitude(x.untrimmedDb) - amplitude(withCoherence(c.pl, x.freqHz, plain[i]))), at).toBeLessThan(SAME_AMPLITUDE);
      });
      const total = closedLoopTotalDb(c.pl, c.cal, o.params, c.secondary);
      expect(Math.abs(amplitude(o.result.predictedAttenuationDb) - amplitude(total)), name(c)).toBeLessThan(SAME_AMPLITUDE);
    }
  }, 60000);

  it('the loop of the params returned stays within the margin on a fine grid', () => {
    // Where the solver's own check reads the loop within 1 dB of the margin (elsewhere a sampling miss cannot reach it).
    let checked = 0;
    for (const c of cases()) {
      const o = c.out;
      if (!usefulOutcome(o) || !((o.loopGainAfterDb ?? -Infinity) > LOOP_GAIN_DANGER_DB - 1)) continue;
      checked++;
      const m = fineLoopMax(c.pl, c.cal, o.params)(o.params);
      expect(m.db, `${name(c)} at ${m.at.toFixed(2)} Hz`).toBeLessThanOrEqual(LOOP_GAIN_DANGER_DB + LOOP_ALLOWANCE_DB);
    }
    expect(checked).toBeGreaterThan(20);
  }, 60000);
  it('a uniform cut is kept only where the loop needs it, and its advice quotes a loop above the margin', () => {
    // Raising every gain by the slack the loop has left (fine grid) predicts no more, and «sopra −6 dB» is true.
    let cut = 0;
    for (const c of cases()) {
      const o = c.out;
      if (refusal(o)) continue;
      const line = o.advice.find((a) => a.startsWith("Guadagno d'anello cassa → microfono di riferimento") && a.includes('sopra −6 dB'));
      if (line !== undefined) {
        const after = Number(line.match(/, (\S+) dB dopo la neutralizzazione del feedback: sopra/)![1].replace(',', '.').replace('−', '-'));
        // (One decimal: −6,0 is a loop at the margin.)
        expect(after, `${name(c)}: ${line.slice(0, 140)}`).toBeGreaterThan(LOOP_GAIN_DANGER_DB - 0.05);
      }
      if (!((o.loopLimitedDb ?? 0) > 0)) continue;
      cut++;
      expect(line, name(c)).toBeDefined();
      const loop = fineLoopMax(c.pl, c.cal, o.params)(o.params).db;
      if (!(loop < LOOP_GAIN_DANGER_DB - 0.01)) continue;
      const up = Math.min(o.loopLimitedDb!, LOOP_GAIN_DANGER_DB - loop - 1e-3);
      const raised = lowered(o.params, -up);
      // (Unless the raise would leave a harmonic below the delay alone that was not: the per-harmonic rule comes first.)
      const unfair = o.harmonics?.some((x, i) => x.predictedDb >= x.untrimmedDb - 1e-6 && predictOutcome(o, raised)!.harmonics![i].predictedDb < x.untrimmedDb - 1e-6);
      if (unfair === true) continue;
      expect(closedLoopTotalDb(c.pl, c.cal, raised, c.secondary), `${name(c)}: cut ${o.loopLimitedDb!.toFixed(3)} dB, loop ${loop.toFixed(3)} dB`).toBeLessThanOrEqual(
        closedLoopTotalDb(c.pl, c.cal, o.params, c.secondary) + 0.01,
      );
    }
    expect(cut).toBeGreaterThan(10);
  }, 60000);

  it('a solve below 1 dB whose cut is negligible is refused for the attenuation, not as a howl risk (seed 21)', () => {
    const o = caseOf(21).out;
    expect(refusal(o)).toBe(true);
    expect(o.advice.some((a) => a.startsWith('Rischio di innesco'))).toBe(false);
    expect(o.advice).toContain(`Con queste misure non si ottiene attenuazione (prevista ${o.result.predictedAttenuationDb.toFixed(1).replace('.', ',')} dB): i parametri non sono stati cambiati.`);
    for (const c of cases()) {
      const a = c.out.advice.find((x) => x.startsWith('Rischio di innesco') && x.includes('andrebbero ridotti di'));
      if (a !== undefined) expect(a, name(c)).not.toMatch(/ridotti di (0,0|meno di 0,1) dB/);
    }
  }, 60000);
});
