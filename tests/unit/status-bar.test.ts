import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { THEME_STORAGE_KEY } from '../../src/ui/status-bar.ts';

describe('theme key', () => {
  it('matches the pre-paint script public/theme-init.js', () => {
    const script = readFileSync(new URL('../../public/theme-init.js', import.meta.url), 'utf8');
    expect(script).toContain(`localStorage.getItem('${THEME_STORAGE_KEY}')`);
  });

  it('the pre-paint script is loaded by index.html', () => {
    const html = readFileSync(new URL('../../index.html', import.meta.url), 'utf8');
    expect(html).toMatch(/<script src="\/theme-init\.js"><\/script>/);
  });
});
