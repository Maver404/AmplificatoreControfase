import { afterEach, describe, expect, it } from 'vitest';
import type { Ch2Status } from '../../src/audio/protocol.ts';
import { ch2Text, createDevicePanel, devicePlaceholder, permissionText } from '../../src/ui/device-panel.ts';
import type { DevicePanelData, MicPermission } from '../../src/ui/device-panel.ts';

describe('device panel texts', () => {
  it('says "negato" only when the browser reports denied', () => {
    for (const p of ['unknown', 'prompt', 'granted'] as MicPermission[]) {
      expect(permissionText(p, false)).not.toMatch(/negato/);
      expect(permissionText(p, true)).not.toMatch(/negato/);
    }
    expect(permissionText('denied', false)).toMatch(/^Accesso al microfono negato/);
    expect(permissionText('prompt', false)).toMatch(/Serve il permesso del microfono/);
  });

  it('reports an empty device list once the microphone is open', () => {
    expect(permissionText('granted', true)).toMatch(/^Nessun dispositivo audio trovato/);
    expect(permissionText('granted', false)).toMatch(/^Microfono consentito/);
  });

  it('uses short select placeholders', () => {
    expect(devicePlaceholder('input', false, 0)).toBe('— consenti il microfono —');
    expect(devicePlaceholder('output', false, 3)).toBe('— consenti il microfono —');
    expect(devicePlaceholder('input', true, 2)).toBe('— scegli —');
    expect(devicePlaceholder('input', true, 0)).toBe('— nessun ingresso —');
    expect(devicePlaceholder('output', true, 0)).toBe('— nessuna uscita —');
    for (const kind of ['input', 'output'] as const) {
      for (const open of [false, true]) {
        for (const n of [0, 1]) expect(devicePlaceholder(kind, open, n).length).toBeLessThanOrEqual(26);
      }
    }
  });

  it('does not claim to check input 2 before the microphone is open', () => {
    const before = ch2Text('pending', false, null);
    expect(before).toMatch(/disponibile dopo aver consentito il microfono/);
    expect(before).not.toMatch(/in corso/);
    expect(ch2Text('ok', false, 2)).toBe(before);
    expect(ch2Text('pending', true)).toBe('Verifica dell’ingresso 2 (mic d’errore) in corso…');
    const states: Ch2Status[] = ['pending', 'ok', 'absent', 'silent', 'duplicate'];
    for (const s of states) expect(ch2Text(s, true, 1)).toMatch(/\(canali ricevuti: 1\)$/);
  });

  it('names the input the engine checks: input 1 when the error mic is there (reference on input 2)', () => {
    expect(ch2Text('silent', true, 2, 1)).toMatch(/^Ingresso 2 muto/);
    expect(ch2Text('silent', true, 2, -1)).toMatch(/^Ingresso 2 muto/);
    expect(ch2Text('silent', true, 2, 0)).toMatch(/^Ingresso 1 muto/);
    expect(ch2Text('ok', true, null, 0)).toBe('Ingresso 1 attivo: il mic d’errore è utilizzabile.');
    expect(ch2Text('duplicate', true, null, 0)).toMatch(/^L’ingresso 1 è identico all’ingresso 2/);
    expect(ch2Text('duplicate', true, null, 1)).toMatch(/^L’ingresso 2 è identico all’ingresso 1/);
  });
});

// ------------------------------------------------------------------------------------- DOM

/** Just enough of the DOM for createDevicePanel (tests run in node without a DOM library). */
class FakeElement {
  readonly tagName: string;
  className = '';
  children: FakeElement[] = [];
  parent: FakeElement | null = null;
  readonly dataset: Record<string, string> = {};
  private readonly attrs = new Map<string, string>();
  private readonly classes = new Set<string>();
  private text = '';
  hidden = false;
  disabled = false;
  value = '';
  id = '';
  htmlFor = '';
  type = '';
  name = '';
  checked = false;
  open = false;
  readonly classList = {
    toggle: (c: string, on?: boolean): boolean => {
      const want = on ?? !this.classes.has(c);
      if (want) this.classes.add(c);
      else this.classes.delete(c);
      return want;
    },
    add: (c: string): void => void this.classes.add(c),
    contains: (c: string): boolean => this.classes.has(c),
  };

  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }

  get textContent(): string {
    return this.text + this.children.map((c) => c.textContent).join('');
  }

  set textContent(v: string) {
    this.text = v;
    this.children = [];
  }

  append(...nodes: FakeElement[]): void {
    for (const n of nodes) {
      n.parent = this;
      this.children.push(n);
    }
  }

  replaceChildren(...nodes: FakeElement[]): void {
    this.children = [];
    this.append(...nodes);
  }

  remove(): void {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }

  setAttribute(k: string, v: string): void {
    this.attrs.set(k, v);
  }

  addEventListener(): void {}

  all(pred: (e: FakeElement) => boolean): FakeElement[] {
    const out: FakeElement[] = [];
    const walk = (e: FakeElement): void => {
      if (pred(e)) out.push(e);
      e.children.forEach(walk);
    };
    walk(this);
    return out;
  }
}

describe('createDevicePanel', () => {
  const g = globalThis as unknown as { document?: unknown };
  afterEach(() => {
    delete g.document;
  });

  const base: DevicePanelData = {
    permission: 'prompt',
    inputs: [],
    outputs: [],
    inputId: null,
    outputId: null,
    sinkIdSupported: true,
    refChannel: 0,
    errorChannel: -1,
    latencyMode: 'normal',
    ch2: 'pending',
  };

  function mount(): { panel: ReturnType<typeof createDevicePanel>; root: FakeElement } {
    g.document = { createElement: (tag: string) => new FakeElement(tag) };
    const root = new FakeElement('div');
    const noop = (): void => {};
    const panel = createDevicePanel(root as unknown as HTMLElement, {
      onAllowMic: noop,
      onInputChange: noop,
      onOutputChange: noop,
      onRefChannelChange: noop,
      onErrorChannelChange: noop,
      onLatencyModeChange: noop,
    });
    return { panel, root };
  }

  const firstOption = (sel: FakeElement): string => sel.children[0]!.textContent;

  it('refreshes the placeholders of lists that stay empty when the microphone opens', () => {
    const { panel, root } = mount();
    const [inSel, outSel] = root.all((e) => e.tagName === 'SELECT');
    const perm = root.all((e) => e.className === 'dp-perm-text')[0]!;
    const ch2 = root.all((e) => e.className === 'dp-ch2')[0]!;
    expect(firstOption(inSel!)).toBe('— consenti il microfono —');
    expect(firstOption(outSel!)).toBe('— consenti il microfono —');
    expect(ch2.textContent).toMatch(/disponibile dopo aver consentito il microfono/);
    expect(perm.textContent).not.toMatch(/negato/);

    panel.update({ ...base, permission: 'granted' });
    expect(firstOption(inSel!)).toBe('— nessun ingresso —');
    expect(firstOption(outSel!)).toBe('— nessuna uscita —');
    expect(perm.textContent).toMatch(/^Nessun dispositivo audio trovato/);
    expect(perm.classList.contains('is-error')).toBe(true);
    expect(ch2.textContent).toBe('Verifica dell’ingresso 2 (mic d’errore) in corso…');

    panel.update({ ...base, permission: 'granted', inputs: [{ deviceId: 'a', label: 'USB' }], outputs: [{ deviceId: 'b', label: 'USB' }] });
    expect(firstOption(inSel!)).toBe('— scegli —');
    expect(inSel!.children).toHaveLength(2);
    expect(perm.textContent).toMatch(/^Microfono consentito/);
    expect(perm.classList.contains('is-error')).toBe(false);
  });

  it('shows the denied advice only for a denied permission', () => {
    const { panel, root } = mount();
    const perm = root.all((e) => e.className === 'dp-perm-text')[0]!;
    panel.update({ ...base, permission: 'denied' });
    expect(perm.textContent).toMatch(/^Accesso al microfono negato/);
    expect(perm.classList.contains('is-error')).toBe(true);
    panel.update({ ...base, permission: 'prompt' });
    expect(perm.textContent).not.toMatch(/negato/);
    expect(perm.classList.contains('is-error')).toBe(false);
    panel.destroy();
    expect(root.children).toHaveLength(0);
  });
});
