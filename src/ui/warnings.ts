/**
 * Warning list with four severities. Items are keyed by id and diffed, so calling update() at
 * meter rate does not re-announce unchanged items to screen readers.
 */

export type WarningLevel = 'block' | 'error' | 'warn' | 'info';

export interface WarningAction {
  label: string;
  onClick: () => void;
}

export interface WarningItem {
  id: string;
  level: WarningLevel;
  text: string;
  action?: WarningAction;
}

export interface WarningsOptions {
  /** Text shown when there is nothing to report (omit to hide the list when empty). */
  emptyText?: string;
}

export interface WarningsView {
  update(items: readonly WarningItem[]): void;
  destroy(): void;
}

const LEVEL_ORDER: Record<WarningLevel, number> = { block: 0, error: 1, warn: 2, info: 3 };
const LEVEL_PREFIX: Record<WarningLevel, string> = {
  block: 'Bloccante',
  error: 'Errore',
  warn: 'Attenzione',
  info: 'Info',
};
const SVG_NS = 'http://www.w3.org/2000/svg';
const ICON_PATHS: Record<WarningLevel, string> = {
  // Circle with a bar (no entry).
  block: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm-6 8.5h12v3H6z',
  // Octagon with an exclamation mark.
  error: 'M8 2h8l6 6v8l-6 6H8l-6-6V8zm2.5 4.5v8h3v-8zm0 10v3h3v-3z',
  // Triangle with an exclamation mark.
  warn: 'M12 2 1 21h22zm-1.5 7h3v6h-3zm0 7.5h3v3h-3z',
  // Circle with an "i".
  info: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm-1.5 8h3v8h-3zm0-4h3v3h-3z',
};

function icon(level: WarningLevel): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'warning-icon');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const p = document.createElementNS(SVG_NS, 'path');
  p.setAttribute('d', ICON_PATHS[level]);
  p.setAttribute('fill-rule', 'evenodd');
  svg.append(p);
  return svg;
}

interface Row {
  li: HTMLLIElement;
  level: WarningLevel;
  text: string;
  actionLabel: string | null;
  action: WarningAction | undefined;
}

export function createWarnings(container: HTMLElement, opts: WarningsOptions = {}): WarningsView {
  const root = document.createElement('div');
  root.className = 'warnings';
  const list = document.createElement('ul');
  list.className = 'warnings-list';
  list.setAttribute('aria-live', 'polite');
  list.setAttribute('aria-relevant', 'additions text');
  const empty = document.createElement('p');
  empty.className = 'warnings-empty';
  empty.textContent = opts.emptyText ?? '';
  root.append(list, empty);
  container.append(root);

  const rows = new Map<string, Row>();

  function build(item: WarningItem): Row {
    const li = document.createElement('li');
    const row: Row = { li, level: item.level, text: '', actionLabel: null, action: item.action };
    fill(row, item);
    return row;
  }

  function fill(row: Row, item: WarningItem): void {
    const { li } = row;
    li.className = `warning warning--${item.level}`;
    const body = document.createElement('p');
    body.className = 'warning-text';
    const prefix = document.createElement('strong');
    prefix.className = 'warning-level';
    prefix.textContent = `${LEVEL_PREFIX[item.level]}: `;
    body.append(prefix, item.text);
    const parts: Node[] = [icon(item.level), body];
    if (item.action) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn--small warning-action';
      btn.textContent = item.action.label;
      // Reads the latest action at click time: callers may pass a fresh closure on every update.
      btn.addEventListener('click', () => row.action?.onClick());
      parts.push(btn);
    }
    li.replaceChildren(...parts);
    row.level = item.level;
    row.text = item.text;
    row.actionLabel = item.action?.label ?? null;
  }

  function update(items: readonly WarningItem[]): void {
    const sorted = [...items].sort((a, b) => LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level]);
    const seen = new Set<string>();
    let prev: HTMLLIElement | null = null;
    for (const item of sorted) {
      if (seen.has(item.id)) continue;
      seen.add(item.id);
      let row = rows.get(item.id);
      if (!row) {
        row = build(item);
        rows.set(item.id, row);
      } else {
        row.action = item.action;
        if (row.level !== item.level || row.text !== item.text || row.actionLabel !== (item.action?.label ?? null)) {
          fill(row, item);
        }
      }
      const expectedNext: ChildNode | null = prev ? prev.nextSibling : list.firstChild;
      if (expectedNext !== row.li) list.insertBefore(row.li, expectedNext);
      prev = row.li;
    }
    for (const [id, row] of rows) {
      if (!seen.has(id)) {
        row.li.remove();
        rows.delete(id);
      }
    }
    // The live list itself is never hidden: a region that appears together with its content
    // is often not announced.
    empty.hidden = rows.size > 0 || !opts.emptyText;
    root.dataset['worst'] = sorted[0]?.level ?? 'none';
  }

  update([]);

  return {
    update,
    destroy(): void {
      rows.clear();
      root.remove();
    },
  };
}
