import { useEffect } from "react";
import { useDemo, type DemoState } from "@/state/demo";

export interface KeyboardContext {
  /** `meta.scene.heroEventId` once the bundle is ready; E is a no-op while null. */
  heroEventId: string | null;
  /** The reveal beat waits for data: revealing an empty scene would strand the demo mid-reveal. */
  ready: boolean;
}

/**
 * Apply one (lower-cased) key to the store (docs/02 §6 → Keyboard). Returns true when the key was
 * handled, so the caller can preventDefault. The next-beat logic gates on `phase`, never on
 * `revealProgress === 1` (REQ-H3-2): "revealed" arrives only after the scene's settle.
 */
export function applyKey(key: string, store: DemoState, context: KeyboardContext): boolean {
  switch (key) {
    case " ":
      if (store.phase === "public") {
        if (context.ready) store.reveal();
      } else if (store.phase === "revealed") {
        if (store.filter !== "strict") store.setFilter("strict");
        else if (!store.timeMode) store.setTimeMode(true);
      }
      return true;
    case "r":
      store.reset();
      return true;
    case "s":
      store.setFilter(store.filter === "strict" ? "all" : "strict");
      return true;
    case "t":
      store.setTimeMode(!store.timeMode);
      return true;
    case "e":
      if (context.heroEventId !== null) store.select(context.heroEventId);
      return true;
    case "p":
      store.setView(store.view === "plan" ? "oblique" : "plan");
      return true;
    case "escape":
      store.select(null);
      return true;
    default:
      return false;
  }
}

function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/** Presenter keys on `window`, mounted once by the shell. Text fields and browser shortcuts win. */
export function useKeyboard(context: KeyboardContext): void {
  const { heroEventId, ready } = context;
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTextInput(event.target)) return;
      if (applyKey(event.key.toLowerCase(), useDemo.getState(), { heroEventId, ready })) {
        event.preventDefault();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [heroEventId, ready]);
}
