/**
 * Generates the fake-microphone WAV files Chrome plays in a loop (--use-file-for-fake-audio-capture):
 * 16-bit PCM, mono, 48 kHz. Deterministic, so every run sees exactly the same input.
 *  - tone-200hz.wav: 200 Hz sine, −20 dBFS peak; the length is a whole number of periods so the
 *    loop has no discontinuity (a click would put broadband content into the tone);
 *  - noise.wav: seeded white Gaussian noise, −20 dBFS RMS, peaks clipped at ±0.95.
 */
import { mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { FIXTURE_FS, FIXTURES, fixturePath } from './helpers.ts';
import type { FixtureName } from './helpers.ts';

const DURATION_S = 10;
const TONE_HZ = 200;
const TONE_PEAK = 0.1;
const NOISE_RMS = 0.1;
const NOISE_CLIP = 0.95;
const NOISE_SEED = 0x5eedc0de;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function tone(): Float64Array {
  const period = FIXTURE_FS / TONE_HZ;
  const n = Math.round((DURATION_S * FIXTURE_FS) / period) * period;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = TONE_PEAK * Math.sin((2 * Math.PI * TONE_HZ * i) / FIXTURE_FS);
  return x;
}

const clipNoise = (v: number): number => Math.max(-NOISE_CLIP, Math.min(NOISE_CLIP, v * NOISE_RMS));

function noise(): Float64Array {
  const rand = mulberry32(NOISE_SEED);
  const n = DURATION_S * FIXTURE_FS;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    // Box–Muller: two independent N(0, 1) samples per pair of uniforms.
    const r = Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-12)));
    const phi = 2 * Math.PI * rand();
    x[i] = clipNoise(r * Math.cos(phi));
    if (i + 1 < n) x[i + 1] = clipNoise(r * Math.sin(phi));
  }
  return x;
}

function wav16(samples: Float64Array, fs: number): Buffer {
  const dataBytes = samples.length * 2;
  const b = Buffer.alloc(44 + dataBytes);
  b.write('RIFF', 0, 'ascii');
  b.writeUInt32LE(36 + dataBytes, 4);
  b.write('WAVE', 8, 'ascii');
  b.write('fmt ', 12, 'ascii');
  b.writeUInt32LE(16, 16); // fmt chunk size
  b.writeUInt16LE(1, 20); // PCM
  b.writeUInt16LE(1, 22); // mono
  b.writeUInt32LE(fs, 24);
  b.writeUInt32LE(fs * 2, 28); // byte rate
  b.writeUInt16LE(2, 32); // block align
  b.writeUInt16LE(16, 34); // bits per sample
  b.write('data', 36, 'ascii');
  b.writeUInt32LE(dataBytes, 40);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i] as number));
    b.writeInt16LE(Math.round(v * 32767), 44 + 2 * i);
  }
  return b;
}

const GENERATORS: Record<FixtureName, () => Float64Array> = { 'tone-200hz': tone, noise };

export default function globalSetup(): void {
  for (const name of FIXTURES) {
    const file = fixturePath(name);
    mkdirSync(dirname(file), { recursive: true });
    // Write-then-rename: a Chrome instance never reads a half-written file.
    const tmp = file.replace(/\.wav$/, `.${process.pid}.tmp.wav`);
    writeFileSync(tmp, wav16(GENERATORS[name](), FIXTURE_FS));
    renameSync(tmp, file);
  }
}
