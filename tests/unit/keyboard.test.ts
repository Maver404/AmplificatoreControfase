import { describe, expect, it, vi } from 'vitest';
import { bindKeyboard, isEditableTarget, keyAction } from '../../src/app/keyboard.ts';

describe('keyAction', () => {
  it('Escape is PANIC everywhere: form fields, modifiers, auto-repeat', () => {
    expect(keyAction({ key: 'Escape' }, false)).toBe('panic');
    expect(keyAction({ key: 'Escape' }, true)).toBe('panic');
    expect(keyAction({ key: 'Escape', ctrlKey: true, shiftKey: true, altKey: true, metaKey: true }, true)).toBe('panic');
    expect(keyAction({ key: 'Escape', repeat: true, isComposing: true }, true)).toBe('panic');
    expect(keyAction({ key: 'Esc' }, false)).toBe('panic');
  });

  it('M and B toggle mute and bypass, upper or lower case', () => {
    expect(keyAction({ key: 'm' }, false)).toBe('mute');
    expect(keyAction({ key: 'M' }, false)).toBe('mute');
    expect(keyAction({ key: 'b' }, false)).toBe('bypass');
    expect(keyAction({ key: 'B' }, false)).toBe('bypass');
  });

  it('M and B are ignored in form fields, with modifiers, while composing and on auto-repeat', () => {
    expect(keyAction({ key: 'm' }, true)).toBeNull();
    expect(keyAction({ key: 'b' }, true)).toBeNull();
    for (const mod of ['ctrlKey', 'metaKey', 'altKey', 'shiftKey'] as const) {
      expect(keyAction({ key: 'm', [mod]: true }, false)).toBeNull();
      expect(keyAction({ key: 'B', [mod]: true }, false)).toBeNull();
    }
    expect(keyAction({ key: 'm', isComposing: true }, false)).toBeNull();
    expect(keyAction({ key: 'm', repeat: true }, false)).toBeNull();
  });

  it('other keys do nothing', () => {
    for (const key of ['a', 'n', ' ', 'Enter', 'ArrowUp', 'Tab', 'µ']) expect(keyAction({ key }, false)).toBeNull();
  });
});

describe('isEditableTarget', () => {
  it('recognises inputs, selects, textareas and contenteditable', () => {
    expect(isEditableTarget({ tagName: 'INPUT' })).toBe(true);
    expect(isEditableTarget({ tagName: 'select' })).toBe(true);
    expect(isEditableTarget({ tagName: 'TEXTAREA' })).toBe(true);
    expect(isEditableTarget({ tagName: 'DIV', isContentEditable: true })).toBe(true);
  });

  it('inputs that take no text (sliders, checkboxes, radios, buttons) let M and B through', () => {
    for (const type of ['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'color', 'file', 'RANGE']) {
      expect(isEditableTarget({ tagName: 'INPUT', type, isContentEditable: false })).toBe(false);
      expect(keyAction({ key: 'm' }, isEditableTarget({ tagName: 'INPUT', type }))).toBe('mute');
      expect(keyAction({ key: 'b' }, isEditableTarget({ tagName: 'INPUT', type }))).toBe('bypass');
    }
    // Text entry keeps the letters (no type attribute = text).
    for (const type of ['text', 'number', 'search', 'email', 'tel', 'url', 'password', 'date', 'time']) {
      expect(isEditableTarget({ tagName: 'INPUT', type })).toBe(true);
    }
    expect(keyAction({ key: 'Escape' }, isEditableTarget({ tagName: 'INPUT', type: 'range' }))).toBe('panic');
  });

  it('buttons, the body and non-elements are not editable', () => {
    expect(isEditableTarget({ tagName: 'BUTTON' })).toBe(false);
    expect(isEditableTarget({ tagName: 'BODY', isContentEditable: false })).toBe(false);
    expect(isEditableTarget(null)).toBe(false);
    expect(isEditableTarget(undefined)).toBe(false);
    expect(isEditableTarget('INPUT')).toBe(false);
  });
});

describe('bindKeyboard', () => {
  function keydown(target: EventTarget | null, init: Partial<KeyboardEvent>): Event & { prevented: boolean } {
    const ev = new Event('keydown', { cancelable: true }) as Event & { prevented: boolean };
    Object.assign(ev, init);
    Object.defineProperty(ev, 'target', { value: target });
    ev.prevented = false;
    ev.preventDefault = () => {
      ev.prevented = true;
    };
    return ev;
  }

  it('listens in the capture phase and dispatches the actions', () => {
    const target = new EventTarget();
    const add = vi.spyOn(target, 'addEventListener');
    const handlers = { panic: vi.fn(), mute: vi.fn(), bypass: vi.fn() };
    const unbind = bindKeyboard(target, handlers);
    expect(add.mock.calls[0]?.[2]).toEqual({ capture: true });

    const listener = add.mock.calls[0]![1] as (e: Event) => void;
    const esc = keydown({ tagName: 'INPUT' } as unknown as EventTarget, { key: 'Escape' });
    listener(esc);
    expect(handlers.panic).toHaveBeenCalledTimes(1);
    // Escape keeps its default action (closing a select, reverting a field).
    expect(esc.prevented).toBe(false);

    const m = keydown(null, { key: 'm' });
    listener(m);
    expect(handlers.mute).toHaveBeenCalledTimes(1);
    expect(m.prevented).toBe(true);

    listener(keydown({ tagName: 'TEXTAREA' } as unknown as EventTarget, { key: 'b' }));
    expect(handlers.bypass).not.toHaveBeenCalled();

    const remove = vi.spyOn(target, 'removeEventListener');
    unbind();
    unbind();
    expect(remove).toHaveBeenCalledTimes(1);
  });
});
