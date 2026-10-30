/**
 * Precision control: slider + numeric field + −/+ buttons, with optional display units.
 * The value is always held and emitted in the base unit; units only change what is shown.
 *
 * Keyboard (slider): arrows = fine step, Shift ×10, Alt ×0.1, PageUp/PageDown ×100.
 * Home/End are swallowed on purpose: a single key must never jump a gain to its maximum.
 * Numeric field: Enter/blur commit (clamped, invalid text is reverted, never NaN); a unit typed
 * after the number is converted ("5 ms" with the field in cm) or, if the control has no such unit,
 * rejected. ArrowUp/Down and PageUp/Down step, Escape reverts the text without stopping propagation
 * (Escape is also the global PANIC shortcut and must keep reaching the controller).
 * Double-click on the label (or ↺) = reset value. Not on the slider: its first click already moves
 * the value to the clicked position, which would reach the engine before the reset.
 */
import {
  SLIDER_RESOLUTION,
  clampFinite,
  decimalsForStep,
  formatAdaptive,
  formatNumber,
  keySteps,
  modifierMultiplier,
  parseNumberWithUnit,
  positionToValue,
  snapToStep,
  stepInteger,
  stepValue,
  typedUnitFactor,
  valueToPosition,
} from './units.ts';
import type { SliderScale, UnitOption } from './units.ts';

export type { UnitOption, SliderScale } from './units.ts';

export interface PrecisionControlOptions {
  label: string;
  value: number;
  min: number;
  max: number;
  /** Fine step in base units (used for snapping and when the unit has no own step). */
  step: number;
  /** Fixed unit suffix when there is no unit list (e.g. 'dB'). */
  unit?: string;
  /** Selectable display units; the first one is the default unless `unitId` is given. */
  units?: readonly UnitOption[];
  unitId?: string;
  /** Decimals for the base unit (default: from `step`). */
  decimals?: number;
  scale?: SliderScale;
  /** Integer-only values (e.g. number of harmonics). */
  integer?: boolean;
  /** Custom text for the numeric field; receives the value already converted to the display unit. */
  format?: (displayValue: number, unit: UnitOption | null) => string;
  /** Secondary text under the control (e.g. the same value in the other units). */
  describe?: (value: number, unit: UnitOption | null) => string;
  /** Double-click (or the ↺ button) sets this value; null/undefined hides the reset. */
  resetValue?: number | null;
  /** What the reset value is, e.g. 'valore calibrato'. */
  resetLabel?: string;
  disabled?: boolean;
  compact?: boolean;
  onChange: (value: number) => void;
  onUnitChange?: (unitId: string) => void;
}

export interface PrecisionControlPatch {
  label?: string;
  min?: number;
  max?: number;
  units?: readonly UnitOption[];
  resetValue?: number | null;
  resetLabel?: string;
  disabled?: boolean;
}

export interface PrecisionControl {
  readonly element: HTMLElement;
  readonly value: number;
  update(value: number, patch?: PrecisionControlPatch): void;
  destroy(): void;
}

const KEY_HELP = 'Frecce: passo fine · Maiusc ×10 · Alt ×0,1 · PagSu/PagGiù ×100';
const REPEAT_DELAY_MS = 400;
const REPEAT_INTERVAL_MS = 60;

let seq = 0;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function createPrecisionControl(container: HTMLElement, opts: PrecisionControlOptions): PrecisionControl {
  const id = `pc${++seq}`;
  let min = opts.min;
  let max = opts.max;
  let value = clampFinite(opts.value, min, max, min);
  let units: UnitOption[] = opts.units ? [...opts.units] : [];
  let unitId: string | null = opts.unitId ?? units[0]?.id ?? null;
  let resetValue: number | null = opts.resetValue ?? null;
  let resetLabel = opts.resetLabel ?? 'valore di riferimento';
  let label = opts.label;
  let disabled = !!opts.disabled;
  /** The user has typed in the numeric field since the last commit: external updates must not overwrite it. */
  let editing = false;
  let dragging = false;
  const scale = opts.scale ?? 'linear';
  const baseDecimals = opts.decimals ?? decimalsForStep(opts.step);
  const stopRepeaters: Array<() => void> = [];

  // ---- DOM ----
  const root = el('div', `pc${opts.compact ? ' pc--compact' : ''}`);
  root.setAttribute('role', 'group');
  root.setAttribute('aria-labelledby', `${id}-label`);

  const head = el('div', 'pc-head');
  const labelEl = el('label', 'pc-label', label);
  labelEl.id = `${id}-label`;
  labelEl.htmlFor = `${id}-slider`;
  const resetBtn = el('button', 'pc-reset icon-btn', '↺');
  resetBtn.type = 'button';
  head.append(labelEl, resetBtn);

  const row = el('div', 'pc-row');
  const minus = el('button', 'pc-step', '−');
  minus.type = 'button';
  const slider = el('input', 'pc-slider');
  slider.type = 'range';
  slider.id = `${id}-slider`;
  slider.min = '0';
  slider.max = String(SLIDER_RESOLUTION);
  slider.step = '1';
  slider.title = KEY_HELP;
  const plus = el('button', 'pc-step', '+');
  plus.type = 'button';

  const entry = el('div', 'pc-entry');
  const num = el('input', 'pc-num');
  num.type = 'text';
  num.inputMode = 'decimal';
  num.autocomplete = 'off';
  num.spellcheck = false;
  num.id = `${id}-num`;
  entry.append(num);

  let unitSelect: HTMLSelectElement | null = null;
  let unitSuffix: HTMLSpanElement | null = null;
  if (units.length > 0) {
    unitSelect = el('select', 'pc-unit');
    entry.append(unitSelect);
  } else if (opts.unit) {
    unitSuffix = el('span', 'pc-suffix', opts.unit);
    unitSuffix.setAttribute('aria-hidden', 'true');
    entry.append(unitSuffix);
  }
  row.append(minus, slider, plus, entry);

  const hint = el('div', 'pc-hint');
  hint.id = `${id}-hint`;
  slider.setAttribute('aria-describedby', hint.id);
  num.setAttribute('aria-describedby', hint.id);

  root.append(head, row, hint);
  container.append(root);

  // ---- helpers ----
  const currentUnit = (): UnitOption | null => units.find((u) => u.id === unitId) ?? units[0] ?? null;
  const unitLabel = (): string => currentUnit()?.label ?? opts.unit ?? '';

  function displayText(v: number): string {
    const u = currentUnit();
    const factor = u?.factor ?? 1;
    const shown = v * factor;
    if (opts.format) return opts.format(shown, u);
    if (opts.integer && !u) return formatNumber(shown, 0);
    const decimals = u ? (u.decimals ?? decimalsForStep(u.step ?? opts.step * factor)) : baseDecimals;
    return formatAdaptive(shown, decimals);
  }

  function renderUnits(): void {
    if (!unitSelect) return;
    const sel = unitSelect;
    const ids = units.map((u) => `${u.id}:${u.label}`).join('|');
    if (sel.dataset['ids'] !== ids) {
      sel.replaceChildren(
        ...units.map((u) => {
          const o = el('option', undefined, u.label);
          o.value = u.id;
          return o;
        }),
      );
      sel.dataset['ids'] = ids;
    }
    if (!units.some((u) => u.id === unitId)) unitId = units[0]?.id ?? null;
    if (unitId !== null) sel.value = unitId;
  }

  function render(): void {
    const text = displayText(value);
    const ul = unitLabel();
    if (!dragging) slider.value = String(Math.round(valueToPosition(value, min, max, scale) * SLIDER_RESOLUTION));
    slider.setAttribute('aria-valuetext', ul ? `${text} ${ul}` : text);
    if (!editing && num.value !== text) num.value = text;
    if (labelEl.textContent !== label) labelEl.textContent = label;
    num.setAttribute('aria-label', ul ? `${label}, valore in ${ul}` : `${label}, valore`);
    minus.setAttribute('aria-label', `Diminuisci ${label}`);
    plus.setAttribute('aria-label', `Aumenta ${label}`);
    if (unitSelect) unitSelect.setAttribute('aria-label', `Unità di ${label}`);
    const hasReset = resetValue !== null && Number.isFinite(resetValue);
    resetBtn.hidden = !hasReset;
    if (hasReset) {
      const r = `${displayText(resetValue as number)}${ul ? ` ${ul}` : ''}`;
      resetBtn.setAttribute('aria-label', `Ripristina ${label}: ${resetLabel} (${r})`);
      resetBtn.title = `Doppio click sul nome o ↺: ${resetLabel} (${r})`;
    }
    const desc = opts.describe?.(value, currentUnit()) ?? '';
    if (hint.textContent !== desc) hint.textContent = desc;
    hint.hidden = desc === '';
    for (const c of [slider, num, minus, plus, resetBtn]) c.disabled = disabled;
    if (unitSelect) unitSelect.disabled = disabled;
    root.classList.toggle('is-disabled', disabled);
  }

  function commit(next: number): void {
    if (disabled || !Number.isFinite(next)) {
      render();
      return;
    }
    const v = clampFinite(next, min, max, value);
    if (v === value) {
      render();
      return;
    }
    value = v;
    render();
    opts.onChange(value);
  }

  function stepBy(steps: number): void {
    if (steps === 0 || disabled) return;
    if (opts.integer) {
      commit(stepInteger(value, steps, min, max));
      return;
    }
    const u = currentUnit();
    const factor = u?.factor ?? 1;
    const displayStep = u?.step ?? opts.step * factor;
    commit(stepValue(value, steps, displayStep, factor, min, max));
  }

  function commitText(): void {
    // Unchanged text must not re-commit the rounded display value (focus + blur would nudge the value).
    if (num.value.trim() === displayText(value)) return;
    const typed = parseNumberWithUnit(num.value);
    // A typed unit is converted; one the control does not have is an error (never read in another unit).
    const factor = typed === null ? undefined : typedUnitFactor(typed.unit, units, opts.unit);
    if (typed === null || factor === undefined) {
      num.setAttribute('aria-invalid', 'true');
      root.classList.add('is-invalid');
      window.setTimeout(() => {
        num.removeAttribute('aria-invalid');
        root.classList.remove('is-invalid');
      }, 1200);
      num.value = displayText(value);
      return;
    }
    const parsed = typed.value;
    let base: number;
    if (parsed === Infinity) base = max;
    else if (parsed === -Infinity) base = min;
    else base = parsed / (factor ?? currentUnit()?.factor ?? 1);
    if (opts.integer) base = Math.round(base);
    commit(base);
    num.value = displayText(value);
  }

  function reset(): void {
    if (resetValue !== null && Number.isFinite(resetValue)) commit(resetValue);
  }

  // ---- events ----
  slider.addEventListener('keydown', (e) => {
    if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      return;
    }
    const steps = keySteps(e.key, e, true);
    if (steps !== 0) {
      e.preventDefault();
      stepBy(steps);
    }
  });
  slider.addEventListener('pointerdown', () => {
    dragging = true;
  });
  const endDrag = (): void => {
    if (!dragging) return;
    dragging = false;
    render();
  };
  slider.addEventListener('change', endDrag);
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  slider.addEventListener('input', () => {
    const pos = Number(slider.value) / SLIDER_RESOLUTION;
    let v = positionToValue(pos, min, max, scale);
    v = opts.integer ? Math.round(v) : snapToStep(v, opts.step);
    commit(v);
  });
  labelEl.addEventListener('dblclick', reset);
  resetBtn.addEventListener('click', reset);

  num.addEventListener('focus', () => num.select());
  num.addEventListener('input', () => {
    editing = true;
  });
  num.addEventListener('blur', () => {
    commitText();
    editing = false;
    render();
  });
  num.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitText();
      editing = false;
      num.value = displayText(value);
      num.select();
      return;
    }
    if (e.key === 'Escape') {
      editing = false;
      num.value = displayText(value);
      num.select();
      return;
    }
    const steps = keySteps(e.key, e, false);
    if (steps !== 0) {
      e.preventDefault();
      // Stepping from the field discards any half-typed text.
      editing = false;
      stepBy(steps);
      num.value = displayText(value);
      num.select();
    }
  });

  unitSelect?.addEventListener('change', () => {
    unitId = unitSelect?.value ?? unitId;
    render();
    if (unitId !== null) opts.onUnitChange?.(unitId);
  });

  function bindRepeat(btn: HTMLButtonElement, dir: 1 | -1): void {
    let timer = 0;
    let interval = 0;
    const stop = (): void => {
      window.clearTimeout(timer);
      window.clearInterval(interval);
      timer = 0;
      interval = 0;
    };
    btn.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || disabled) return;
      const steps = dir * modifierMultiplier(e);
      stop();
      stepBy(steps);
      timer = window.setTimeout(() => {
        interval = window.setInterval(() => {
          // A disabled or detached button never receives pointerup: stop on our own.
          if (disabled || !btn.isConnected) stop();
          else stepBy(steps);
        }, REPEAT_INTERVAL_MS);
      }, REPEAT_DELAY_MS);
    });
    for (const type of ['pointerup', 'pointerleave', 'pointercancel', 'blur'] as const) btn.addEventListener(type, stop);
    // Keyboard activation (Enter/Space) produces a click with detail 0; pointer clicks were already handled.
    btn.addEventListener('click', (e) => {
      if (e.detail === 0) stepBy(dir * modifierMultiplier(e));
    });
    stopRepeaters.push(stop);
  }
  bindRepeat(minus, -1);
  bindRepeat(plus, 1);
  const stopAll = (): void => stopRepeaters.forEach((s) => s());
  window.addEventListener('pointerup', stopAll);
  window.addEventListener('blur', stopAll);

  renderUnits();
  render();

  return {
    element: root,
    get value() {
      return value;
    },
    update(next: number, patch?: PrecisionControlPatch): void {
      if (patch) {
        if (patch.label !== undefined) label = patch.label;
        if (patch.min !== undefined && Number.isFinite(patch.min)) min = patch.min;
        if (patch.max !== undefined && Number.isFinite(patch.max)) max = patch.max;
        if (patch.units !== undefined) {
          units = [...patch.units];
          renderUnits();
        }
        if (patch.resetValue !== undefined) resetValue = patch.resetValue;
        if (patch.resetLabel !== undefined) resetLabel = patch.resetLabel;
        if (patch.disabled !== undefined) disabled = patch.disabled;
        if (disabled) stopAll();
      }
      // The controller is the source of truth: show its value as is (only NaN is rejected).
      if (Number.isFinite(next)) value = next;
      render();
    },
    destroy(): void {
      stopAll();
      window.removeEventListener('pointerup', endDrag);
      window.removeEventListener('pointercancel', endDrag);
      window.removeEventListener('pointerup', stopAll);
      window.removeEventListener('blur', stopAll);
      root.remove();
    },
  };
}
