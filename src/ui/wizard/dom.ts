/**
 * Small DOM helpers for the wizard views (no innerHTML: every text goes through textContent).
 */

export type Child = Node | string | null | undefined | false;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string | null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    e.append(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return e;
}

let seq = 0;
/** Unique id for label/aria wiring. */
export const uid = (prefix: string): string => `${prefix}-${++seq}`;

/** Sets text only when it changed (avoids re-announcing live regions at meter rate). */
export function setText(e: HTMLElement, text: string): void {
  if (e.textContent !== text) e.textContent = text;
}

export function button(text: string, onClick: () => void, cls = 'btn'): HTMLButtonElement {
  const b = el('button', cls, text);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

/** Checkbox row with a label; `onChange` receives the new value. */
export function checkbox(text: Child, onChange: (on: boolean) => void): { root: HTMLLabelElement; input: HTMLInputElement } {
  const input = el('input');
  input.type = 'checkbox';
  input.addEventListener('change', () => onChange(input.checked));
  const root = el('label', 'wz-check', input, el('span', null, text));
  return { root, input };
}

/** Group of radio "cards"; returns a setter for the checked value and the disabled state. */
export function radioCards<T extends string>(
  legend: string,
  options: readonly { value: T; title: string; text?: string }[],
  onChange: (v: T) => void,
): { root: HTMLFieldSetElement; set(v: T | null): void; disable(on: boolean): void } {
  const name = uid('wz-radio');
  const root = el('fieldset', 'wz-choices', el('legend', 'wz-legend', legend));
  const inputs: { v: T; input: HTMLInputElement }[] = [];
  for (const o of options) {
    const input = el('input');
    input.type = 'radio';
    input.name = name;
    input.value = o.value;
    input.addEventListener('change', () => {
      if (input.checked) onChange(o.value);
    });
    const card = el('label', 'wz-choice', input, el('span', 'wz-choice-body', el('strong', null, o.title), o.text ? el('span', 'wz-choice-text', o.text) : null));
    root.append(card);
    inputs.push({ v: o.value, input });
  }
  return {
    root,
    set(v) {
      for (const i of inputs) i.input.checked = i.v === v;
    },
    disable(on) {
      for (const i of inputs) i.input.disabled = on;
    },
  };
}

/** Labelled text field for decimal numbers (Italian comma accepted). */
export function numberField(
  label: string,
  unit: string,
  onInput: (text: string) => void,
): { root: HTMLDivElement; input: HTMLInputElement; setInvalid(on: boolean): void } {
  const id = uid('wz-num');
  const input = el('input', 'wz-num');
  input.type = 'text';
  input.inputMode = 'decimal';
  input.id = id;
  input.autocomplete = 'off';
  input.addEventListener('input', () => onInput(input.value));
  const lab = el('label', 'field-label', label);
  lab.htmlFor = id;
  const root = el('div', 'field wz-field', lab, el('span', 'wz-num-row', input, el('span', 'wz-unit', unit)));
  return {
    root,
    input,
    setInvalid(on) {
      input.setAttribute('aria-invalid', String(on));
    },
  };
}

/** Paragraph note: info / warn / error / ok. */
export function note(level: 'info' | 'warn' | 'error' | 'ok', ...children: Child[]): HTMLParagraphElement {
  const p = el('p', `wz-note wz-note--${level}`, ...children);
  if (level === 'error' || level === 'warn') p.setAttribute('role', 'note');
  return p;
}

/** A note whose level and text are updated later (hidden while empty). */
export function liveNote(): { root: HTMLParagraphElement; set(level: 'info' | 'warn' | 'error' | 'ok', text: string): void; clear(): void } {
  const root = el('p', 'wz-note');
  root.setAttribute('role', 'status');
  root.hidden = true;
  return {
    root,
    set(level, text) {
      root.className = `wz-note wz-note--${level}`;
      setText(root, text);
      root.hidden = text === '';
    },
    clear() {
      setText(root, '');
      root.hidden = true;
    },
  };
}

export function list(items: readonly Child[][], ordered = false): HTMLElement {
  const root = el(ordered ? 'ol' : 'ul', 'wz-list');
  for (const i of items) root.append(el('li', null, ...i));
  return root;
}

/** Replaces the children of a list element with one <li> per text (only when the texts changed). */
export function setListItems(root: HTMLElement, texts: readonly string[]): void {
  const key = texts.join('\u0001');
  if (root.dataset['key'] === key) return;
  root.dataset['key'] = key;
  root.replaceChildren(...texts.map((t) => el('li', null, t)));
  root.hidden = texts.length === 0;
}

/** <progress> with a text label; fraction null = indeterminate, hidden when `visible` is false. */
export function progressBar(label: string): { root: HTMLDivElement; set(visible: boolean, fraction: number | null, text?: string): void } {
  const bar = el('progress', 'wz-progress');
  bar.max = 1;
  bar.setAttribute('aria-label', label);
  const text = el('span', 'wz-progress-text');
  const root = el('div', 'wz-progress-row', bar, text);
  root.hidden = true;
  return {
    root,
    set(visible, fraction, t) {
      root.hidden = !visible;
      if (fraction === null) bar.removeAttribute('value');
      else bar.value = Math.min(1, Math.max(0, fraction));
      setText(text, t ?? '');
    },
  };
}

/** Promise that resolves after `ms` or rejects when `signal` aborts. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error('Interrotto.'));
      return;
    }
    const t = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, Math.max(0, ms));
    const onAbort = (): void => {
      clearTimeout(t);
      reject(new Error('Interrotto.'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
