/**
 * Global keyboard shortcuts. Escape is PANIC everywhere (form fields, modifiers, auto-repeat
 * included); M (mute) and B (bypass A/B) are single-letter shortcuts, so they are ignored while the
 * user types in a text field, with any modifier held, while an IME is composing, and on auto-repeat
 * (holding M must not toggle the output on and off). Sliders, checkboxes, radios and buttons take
 * no letters: M and B work while they have focus (a slider keeps it after a drag).
 */

export type KeyAction = 'panic' | 'mute' | 'bypass';

export interface KeyLike {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
  repeat?: boolean;
  isComposing?: boolean;
}

/** SELECT keeps its type-ahead; an INPUT only when it takes text (see NON_TEXT_INPUTS). */
const FORM_TAGS = new Set(['INPUT', 'SELECT', 'TEXTAREA']);
/** Input types where a letter types nothing. */
const NON_TEXT_INPUTS = new Set(['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'color', 'file', 'image', 'hidden']);

/** True for text fields, selects and editable content, where letters are text, not shortcuts. */
export function isEditableTarget(t: unknown): boolean {
  if (t === null || typeof t !== 'object') return false;
  const e = t as { tagName?: unknown; isContentEditable?: unknown; type?: unknown };
  if (e.isContentEditable === true) return true;
  if (typeof e.tagName !== 'string') return false;
  const tag = e.tagName.toUpperCase();
  if (!FORM_TAGS.has(tag)) return false;
  // An input without a type attribute is a text field.
  return tag !== 'INPUT' || typeof e.type !== 'string' || !NON_TEXT_INPUTS.has(e.type.toLowerCase());
}

/** Pure mapping of a keydown to an action; `editable` = focus is in a form field. */
export function keyAction(e: KeyLike, editable: boolean): KeyAction | null {
  if (e.key === 'Escape' || e.key === 'Esc') return 'panic';
  if (editable || e.isComposing === true || e.repeat === true) return null;
  if (e.ctrlKey === true || e.metaKey === true || e.altKey === true || e.shiftKey === true) return null;
  if (e.key === 'm' || e.key === 'M') return 'mute';
  if (e.key === 'b' || e.key === 'B') return 'bypass';
  return null;
}

export type KeyHandlers = Record<KeyAction, () => void>;

/**
 * Listens in the capture phase so that no component can swallow Escape before it arrives.
 * Returns an idempotent unbind function.
 */
export function bindKeyboard(target: EventTarget, handlers: KeyHandlers): () => void {
  const listener = (ev: Event): void => {
    const e = ev as KeyboardEvent;
    const action = keyAction(e, isEditableTarget(e.target));
    if (action === null) return;
    // Escape keeps its default behaviour (closing a select, reverting a field); letters do not type.
    if (action !== 'panic') e.preventDefault();
    handlers[action]();
  };
  target.addEventListener('keydown', listener, { capture: true });
  let bound = true;
  return () => {
    if (!bound) return;
    bound = false;
    target.removeEventListener('keydown', listener, { capture: true });
  };
}
