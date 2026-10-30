import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const read = (path: string): string => readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8');

describe('favicon', () => {
  it('index.html links the SVG favicon shipped in public/ (no /favicon.ico 404)', () => {
    expect(read('index.html')).toContain('<link rel="icon" href="/favicon.svg" type="image/svg+xml" />');
    const svg = read('public/favicon.svg');
    expect(svg).toMatch(/^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg" viewBox="0 0 32 32">/);
    // Self-contained: no scripts, no external references.
    expect(svg).not.toMatch(/<script|href=|url\(/i);
  });

  it('the production CSP allows same-origin images', () => {
    const conf = read('docker/security-headers.conf');
    const csp = /add_header\s+Content-Security-Policy\s+"([^"]+)"/.exec(conf)![1]!;
    const directive = (name: string): string | undefined =>
      csp
        .split(';')
        .map((d) => d.trim())
        .find((d) => d.startsWith(`${name} `));
    const img = directive('img-src') ?? directive('default-src');
    expect(img?.split(/\s+/)).toContain("'self'");
  });
});
