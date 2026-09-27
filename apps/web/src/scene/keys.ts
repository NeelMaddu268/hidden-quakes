// The scene's own presenter keys (G: the guided tour; H: the hidden hero) share the shell's rules
// (shell/useKeyboard.ts): text fields and browser shortcuts win, and a key the scene handles is
// prevented so no other handler acts on it.

/** True for targets whose own key handling wins (inputs, textareas, selects, contenteditable). */
export function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT";
}

/** A plain press of a key: not a repeat, not already handled, no Cmd/Ctrl/Alt, not typed into a field. */
export function isPlainPress(event: KeyboardEvent): boolean {
  return (
    !event.repeat &&
    !event.defaultPrevented &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.altKey &&
    !isTextInput(event.target)
  );
}
