/**
 * Presets: list, save under a name, load, delete (two-step confirmation), export JSON and import
 * from a file. Callbacks only: storage, validation and the mute → apply → soft-start sequence
 * belong to the controller. update() comes with every controller state change: the list is
 * rebuilt only when what it shows changed (a rebuild under a pressed button loses the click and
 * moves the focus).
 */

export interface PresetSummary {
  id: string;
  name: string;
  /** Epoch ms of the last save, if known. */
  savedAt?: number | null;
  /** Short secondary text (e.g. the device the preset was made for). */
  detail?: string;
}

export interface PresetsPanelData {
  presets: readonly PresetSummary[];
  activeId?: string | null;
  message?: { level: 'info' | 'warn' | 'error'; text: string } | null;
  /** Disable load/import (e.g. during a measurement). */
  disabled?: boolean;
}

export interface PresetsPanelOptions {
  onSave: (name: string) => void;
  onLoad: (id: string) => void;
  onDelete: (id: string) => void;
  /** id = one preset, null = all presets. */
  onExport: (id: string | null) => void;
  onImport: (file: File) => void;
}

export interface PresetsPanel {
  update(data: PresetsPanelData): void;
  destroy(): void;
}

export const PRESET_NAME_MAX = 60;
const CONFIRM_MS = 4000;

let seq = 0;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function button(text: string, className = 'btn btn--small'): HTMLButtonElement {
  const b = el('button', className, text);
  b.type = 'button';
  return b;
}

const dateFmt = new Intl.DateTimeFormat('it-IT', { dateStyle: 'short', timeStyle: 'short' });

export function createPresetsPanel(container: HTMLElement, opts: PresetsPanelOptions): PresetsPanel {
  const id = `pp${++seq}`;
  let data: PresetsPanelData = { presets: [] };
  let confirmId: string | null = null;
  let confirmTimer = 0;

  const root = el('div', 'presets-panel');

  const form = el('form', 'pp-save');
  const nameLabel = el('label', 'field-label', 'Nome del preset');
  nameLabel.htmlFor = `${id}-name`;
  const nameInput = el('input');
  nameInput.id = `${id}-name`;
  nameInput.type = 'text';
  nameInput.maxLength = PRESET_NAME_MAX;
  nameInput.autocomplete = 'off';
  nameInput.placeholder = 'es. Finestra cucina, sera';
  const saveBtn = el('button', 'btn btn--primary', 'Salva impostazioni attuali');
  saveBtn.type = 'submit';
  const formRow = el('div', 'pp-save-row');
  formRow.append(nameInput, saveBtn);
  form.append(nameLabel, formRow);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = nameInput.value.trim().slice(0, PRESET_NAME_MAX);
    if (!name) {
      nameInput.setAttribute('aria-invalid', 'true');
      nameInput.focus();
      return;
    }
    nameInput.removeAttribute('aria-invalid');
    opts.onSave(name);
    nameInput.value = '';
  });
  nameInput.addEventListener('input', () => nameInput.removeAttribute('aria-invalid'));

  const list = el('ul', 'pp-list');
  list.setAttribute('aria-label', 'Preset salvati');
  const empty = el('p', 'pp-empty', 'Nessun preset salvato.');

  const io = el('div', 'pp-io');
  const exportAll = button('Esporta tutti (JSON)', 'btn');
  exportAll.addEventListener('click', () => opts.onExport(null));
  const fileInput = el('input', 'visually-hidden');
  fileInput.type = 'file';
  fileInput.accept = '.json,application/json';
  fileInput.id = `${id}-file`;
  fileInput.tabIndex = -1;
  fileInput.setAttribute('aria-hidden', 'true');
  const importBtn = button('Importa da file…', 'btn');
  importBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (f) opts.onImport(f);
    fileInput.value = '';
  });
  io.append(exportAll, importBtn, fileInput);

  const note = el(
    'p',
    'fp-help',
    'Il caricamento silenzia l’uscita, applica i valori (riportati ai limiti sicuri) e riparte con il soft-start. Lo sblocco +12 dB non viene mai salvato.',
  );
  const message = el('p', 'pp-message');
  message.setAttribute('role', 'status');

  root.append(form, list, empty, io, message, note);
  container.append(root);

  function clearConfirm(): void {
    window.clearTimeout(confirmTimer);
    confirmId = null;
  }

  /** Everything the list shows; the list is rebuilt only when this changes. */
  const listKey = (d: PresetsPanelData): string =>
    JSON.stringify([d.presets.map((p) => [p.id, p.name, p.savedAt ?? null, p.detail ?? '']), d.activeId ?? null, !!d.disabled, confirmId]);
  let renderedKey: string | null = null;

  function render(): void {
    const d = data;
    const m = d.message ?? null;
    if (message.textContent !== (m ? m.text : '')) message.textContent = m ? m.text : '';
    message.dataset['level'] = m ? m.level : '';
    importBtn.disabled = !!d.disabled;
    const key = listKey(d);
    if (key === renderedKey) return;
    renderedKey = key;
    // The list is rebuilt: remember which action button had focus and restore it afterwards.
    const focused = document.activeElement;
    const focusKey = focused instanceof HTMLElement && list.contains(focused) ? (focused.dataset['key'] ?? null) : null;
    const items = d.presets.map((p) => {
      const li = el('li', 'pp-item');
      if (p.id === d.activeId) li.classList.add('is-active');
      const info = el('div', 'pp-info');
      const name = el('span', 'pp-name', p.name);
      info.append(name);
      const meta: string[] = [];
      if (p.savedAt !== undefined && p.savedAt !== null && Number.isFinite(p.savedAt)) {
        meta.push(`salvato ${dateFmt.format(new Date(p.savedAt))}`);
      }
      if (p.detail) meta.push(p.detail);
      if (p.id === d.activeId) meta.push('in uso');
      if (meta.length) info.append(el('span', 'pp-meta', meta.join(' · ')));

      const actions = el('div', 'pp-actions');
      const load = button('Carica', 'btn btn--small btn--primary');
      load.disabled = !!d.disabled;
      load.setAttribute('aria-label', `Carica il preset ${p.name}`);
      load.addEventListener('click', () => opts.onLoad(p.id));
      load.dataset['key'] = `load:${p.id}`;
      const exp = button('Esporta');
      exp.setAttribute('aria-label', `Esporta il preset ${p.name}`);
      exp.addEventListener('click', () => opts.onExport(p.id));
      exp.dataset['key'] = `export:${p.id}`;
      const confirming = confirmId === p.id;
      const del = button(confirming ? 'Conferma eliminazione' : 'Elimina', `btn btn--small btn--danger${confirming ? ' is-confirming' : ''}`);
      del.setAttribute('aria-label', confirming ? `Conferma eliminazione del preset ${p.name}` : `Elimina il preset ${p.name}`);
      del.addEventListener('click', () => {
        if (confirmId === p.id) {
          clearConfirm();
          opts.onDelete(p.id);
          render();
          return;
        }
        clearConfirm();
        confirmId = p.id;
        confirmTimer = window.setTimeout(() => {
          confirmId = null;
          render();
        }, CONFIRM_MS);
        render();
      });
      del.dataset['key'] = `delete:${p.id}`;
      actions.append(load, exp, del);
      li.append(info, actions);
      return li;
    });
    list.replaceChildren(...items);
    if (focusKey !== null) {
      for (const b of list.querySelectorAll<HTMLElement>('[data-key]')) {
        if (b.dataset['key'] === focusKey) {
          b.focus();
          break;
        }
      }
    }
    list.hidden = items.length === 0;
    empty.hidden = items.length > 0;
    exportAll.disabled = items.length === 0;
  }

  render();

  return {
    update(next: PresetsPanelData): void {
      data = next;
      if (confirmId !== null && !next.presets.some((p) => p.id === confirmId)) clearConfirm();
      render();
    },
    destroy(): void {
      clearConfirm();
      root.remove();
    },
  };
}
