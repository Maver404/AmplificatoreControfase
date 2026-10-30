/**
 * Just enough DOM for the views in src/ui (tests run in node without a DOM library): elements with
 * attributes, classes, dataset, text, children, events and focus. Every textContent write is
 * counted per element, so tests can check that live regions are not rewritten with the same text.
 */

type Listener = (e: FakeEvent) => void;

export class FakeEvent {
  readonly type: string;
  target: FakeNode | null = null;
  defaultPrevented = false;
  detail = 1;
  key = '';
  shiftKey = false;
  altKey = false;
  button = 0;
  constructor(type: string, init: Partial<Pick<FakeEvent, 'detail' | 'key' | 'shiftKey' | 'altKey' | 'button'>> = {}) {
    this.type = type;
    Object.assign(this, init);
  }
  preventDefault(): void {
    this.defaultPrevented = true;
  }
  stopPropagation(): void {}
}

export class FakeNode {
  parent: FakeElement | null = null;
  readonly listeners = new Map<string, Set<Listener>>();
  protected ownText = '';

  get nodeType(): number {
    return 3;
  }
  get textContent(): string {
    return this.ownText;
  }
  set textContent(v: string) {
    this.ownText = String(v);
  }
  get parentNode(): FakeElement | null {
    return this.parent;
  }
  get parentElement(): FakeElement | null {
    return this.parent;
  }
  get isConnected(): boolean {
    let n: FakeNode | null = this;
    while (n.parent) n = n.parent;
    return n instanceof FakeElement && n.isDocumentRoot;
  }
  remove(): void {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
  addEventListener(type: string, cb: Listener): void {
    let set = this.listeners.get(type);
    if (!set) this.listeners.set(type, (set = new Set()));
    set.add(cb);
  }
  removeEventListener(type: string, cb: Listener): void {
    this.listeners.get(type)?.delete(cb);
  }
  dispatchEvent(e: FakeEvent): boolean {
    if (e.target === null) e.target = this;
    for (const cb of [...(this.listeners.get(e.type) ?? [])]) cb(e);
    return !e.defaultPrevented;
  }
  hasListener(type: string): boolean {
    return (this.listeners.get(type)?.size ?? 0) > 0;
  }
}

export class FakeElement extends FakeNode {
  readonly tagName: string;
  children: FakeNode[] = [];
  readonly dataset: Record<string, string> = {};
  readonly attrs = new Map<string, string>();
  private readonly classes = new Set<string>();
  readonly style = { setProperty: (): void => {}, removeProperty: (): void => {} };
  /** Number of textContent assignments on this element (also the identical ones). */
  textWrites = 0;
  isDocumentRoot = false;
  owner: FakeDocument;
  id = '';
  hidden = false;
  disabled = false;
  value = '';
  checked = false;
  type = '';
  name = '';
  htmlFor = '';
  title = '';
  tabIndex = 0;
  open = false;
  inputMode = '';
  autocomplete = '';
  spellcheck = true;
  placeholder = '';
  maxLength = -1;
  min = '';
  max = '';
  step = '';
  accept = '';
  href = '';
  download = '';
  files: unknown = null;
  readonly classList = {
    add: (...c: string[]): void => c.forEach((x) => this.classes.add(x)),
    remove: (...c: string[]): void => c.forEach((x) => this.classes.delete(x)),
    contains: (c: string): boolean => this.classes.has(c),
    toggle: (c: string, on?: boolean): boolean => {
      const want = on ?? !this.classes.has(c);
      if (want) this.classes.add(c);
      else this.classes.delete(c);
      return want;
    },
  };

  constructor(tag: string, owner: FakeDocument) {
    super();
    this.tagName = tag.toUpperCase();
    this.owner = owner;
  }

  override get nodeType(): number {
    return 1;
  }
  get className(): string {
    return [...this.classes].join(' ');
  }
  set className(v: string) {
    this.classes.clear();
    for (const c of String(v).split(/\s+/)) if (c) this.classes.add(c);
  }
  override get textContent(): string {
    return this.ownText + this.children.map((c) => c.textContent).join('');
  }
  override set textContent(v: string) {
    this.textWrites++;
    this.ownText = String(v);
    for (const c of this.children) c.parent = null;
    this.children = [];
  }
  get childNodes(): FakeNode[] {
    return this.children;
  }
  get firstChild(): FakeNode | null {
    return this.children[0] ?? null;
  }
  append(...nodes: (FakeNode | string)[]): void {
    for (const n of nodes) {
      const node = typeof n === 'string' ? this.owner.createTextNode(n) : n;
      node.remove();
      node.parent = this;
      this.children.push(node);
    }
  }
  appendChild<T extends FakeNode>(n: T): T {
    this.append(n);
    return n;
  }
  replaceChildren(...nodes: (FakeNode | string)[]): void {
    for (const c of this.children) c.parent = null;
    this.children = [];
    this.ownText = '';
    this.append(...nodes);
    // Removing the focused element moves the focus to <body>, like a browser.
    const a = this.owner.activeElement;
    if (a !== this.owner.body && !a.isConnected) this.owner.activeElement = this.owner.body;
  }
  setAttribute(k: string, v: string): void {
    if (k === 'class') this.className = v;
    else if (k === 'id') this.id = v;
    else this.attrs.set(k, String(v));
  }
  getAttribute(k: string): string | null {
    if (k === 'class') return this.className;
    return this.attrs.get(k) ?? null;
  }
  hasAttribute(k: string): boolean {
    return this.attrs.has(k);
  }
  removeAttribute(k: string): void {
    this.attrs.delete(k);
  }
  contains(n: FakeNode | null): boolean {
    for (let x: FakeNode | null = n; x; x = x.parent) if (x === this) return true;
    return false;
  }
  focus(): void {
    this.owner.activeElement = this;
  }
  blur(): void {
    if (this.owner.activeElement === this) this.owner.activeElement = this.owner.body;
  }
  click(): void {
    if (this.disabled) return;
    this.dispatchEvent(new FakeEvent('click', { detail: 0 }));
  }
  select(): void {}
  scrollIntoView(): void {}
  getBoundingClientRect(): { top: number; left: number; width: number; height: number; bottom: number; right: number } {
    return { top: 0, left: 0, width: 0, height: 0, bottom: 0, right: 0 };
  }
  /** Every descendant element (depth first) matching `pred`. */
  all(pred: (e: FakeElement) => boolean): FakeElement[] {
    const out: FakeElement[] = [];
    const walk = (e: FakeElement): void => {
      for (const c of e.children) {
        if (!(c instanceof FakeElement)) continue;
        if (pred(c)) out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }
  /** Supports '.class', 'tag', '[data-x]' and 'tag.class'. */
  querySelectorAll(sel: string): FakeElement[] {
    return this.all((e) => matches(e, sel));
  }
  querySelector(sel: string): FakeElement | null {
    return this.querySelectorAll(sel)[0] ?? null;
  }
  /** First descendant whose own text content equals `text` (buttons, labels). */
  byText(text: string, tag?: string): FakeElement | null {
    return this.all((e) => e.textContent === text && (!tag || e.tagName === tag.toUpperCase()))[0] ?? null;
  }
}

function matches(e: FakeElement, sel: string): boolean {
  const attr = /^\[([\w-]+)\]$/.exec(sel);
  if (attr) {
    const name = attr[1]!;
    if (name.startsWith('data-')) return name.slice(5).replace(/-(\w)/g, (_m, c: string) => c.toUpperCase()) in e.dataset;
    return e.hasAttribute(name);
  }
  const m = /^([a-z0-9]*)((?:\.[\w-]+)*)$/i.exec(sel);
  if (!m) return false;
  if (m[1] && e.tagName !== m[1].toUpperCase()) return false;
  const classes = (m[2] ?? '').split('.').filter(Boolean);
  return classes.every((c) => e.classList.contains(c));
}

class FakeText extends FakeNode {
  constructor(text: string) {
    super();
    this.ownText = text;
  }
}

export class FakeDocument extends FakeNode {
  readonly body: FakeElement;
  readonly documentElement: FakeElement;
  activeElement: FakeElement;
  constructor() {
    super();
    this.documentElement = new FakeElement('html', this);
    this.documentElement.isDocumentRoot = true;
    this.body = new FakeElement('body', this);
    this.documentElement.append(this.body);
    this.activeElement = this.body;
  }
  createElement(tag: string): FakeElement {
    return new FakeElement(tag, this);
  }
  createElementNS(_ns: string, tag: string): FakeElement {
    return new FakeElement(tag, this);
  }
  createTextNode(text: string): FakeNode {
    return new FakeText(text);
  }
  getElementById(id: string): FakeElement | null {
    return this.documentElement.all((e) => e.id === id)[0] ?? null;
  }
  querySelector(sel: string): FakeElement | null {
    return this.documentElement.querySelector(sel);
  }
  querySelectorAll(sel: string): FakeElement[] {
    return this.documentElement.querySelectorAll(sel);
  }
}

export interface FakeDom {
  document: FakeDocument;
  /** A container attached to the document (so isConnected holds for what is mounted in it). */
  mount(): FakeElement;
  restore(): void;
}

/**
 * Installs `document`, `window` (timers, listeners, rAF through setTimeout) and the element classes
 * the views test with `instanceof` on globalThis; restore() removes them.
 */
export function installFakeDom(): FakeDom {
  const g = globalThis as Record<string, unknown>;
  const saved = new Map<string, unknown>();
  const set = (k: string, v: unknown): void => {
    saved.set(k, g[k]);
    g[k] = v;
  };
  const document = new FakeDocument();
  const win = new FakeNode() as FakeNode & Record<string, unknown>;
  Object.assign(win, {
    setTimeout: (cb: () => void, ms?: number) => setTimeout(cb, ms),
    clearTimeout: (t: ReturnType<typeof setTimeout>) => clearTimeout(t),
    setInterval: (cb: () => void, ms?: number) => setInterval(cb, ms),
    clearInterval: (t: ReturnType<typeof setInterval>) => clearInterval(t),
    requestAnimationFrame: (cb: (t: number) => void) => setTimeout(() => cb(Date.now()), 0),
    cancelAnimationFrame: (t: ReturnType<typeof setTimeout>) => clearTimeout(t),
    scrollTo: () => {},
    scrollY: 0,
    localStorage: undefined,
  });
  set('document', document);
  set('window', win);
  set('HTMLElement', FakeElement);
  set('getComputedStyle', () => ({ position: 'static' }));
  return {
    document,
    mount() {
      const c = document.createElement('div');
      document.body.append(c);
      return c;
    },
    restore() {
      for (const [k, v] of saved) {
        if (v === undefined) delete g[k];
        else g[k] = v;
      }
    },
  };
}
