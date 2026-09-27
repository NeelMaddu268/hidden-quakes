"use client";
/**
 * Share links (overnight feature 3, Sat): `?event=<id>` opens that event's evidence drawer once
 * the page has loaded and the reveal has played, and the address bar follows the selection so
 * "Copy link" in the drawer header (H3) can copy `location.href`.
 *
 * The mode is read from the same query string by `ProviderRoot`; this module touches only the
 * `event` parameter and never navigates (history.replaceState, no reload).
 */
import { useEffect } from "react";
import { useDemo } from "@/state/demo";

export const EVENT_PARAM = "event";

/** `?event=` from a query string; null when absent or blank. Unknown ids are filtered by the caller. */
export function parseEventParam(search: string): string | null {
  const value = new URLSearchParams(search).get(EVENT_PARAM);
  return value && value.trim() !== "" ? value.trim() : null;
}

/** The share link for an event: the current page (mode included) with `event=<id>` set. */
export function eventShareLink(eventId: string, href: string = window.location.href): string {
  const url = new URL(href);
  url.searchParams.set(EVENT_PARAM, eventId);
  url.hash = "";
  return url.toString();
}

/** The same URL with the event parameter removed (the drawer closed). */
export function withoutEventParam(href: string): string {
  const url = new URL(href);
  url.searchParams.delete(EVENT_PARAM);
  return url.toString();
}

/**
 * The reveal settles in about seven seconds at full frame rate; on a slow machine the scene's
 * clock (clamped per frame) can take far longer. A shared link should still open its drawer, so
 * after this long the selection is made even if the settle has not been reported.
 */
export const OPEN_FALLBACK_MS = 20_000;

/**
 * Open the drawer for `eventId` after the reveal: starts the reveal when the demo is still on
 * its start frame, then selects once the scene reports "revealed" (or after `OPEN_FALLBACK_MS`,
 * whichever comes first). Returns a cancel; the selection is skipped if that fires first
 * (unmount, or a reset before the settle).
 */
export function openEventAfterReveal(eventId: string, store = useDemo, fallbackMs = OPEN_FALLBACK_MS): () => void {
  const state = store.getState();
  if (state.phase === "revealed") {
    state.select(eventId);
    return () => {};
  }
  if (state.phase === "public") state.reveal();
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    unsubscribe();
    clearTimeout(timer);
  };
  const unsubscribe = store.subscribe((next, prev) => {
    if (next.phase === "revealed" && prev.phase !== "revealed") {
      finish();
      next.select(eventId);
    }
  });
  const timer = setTimeout(() => {
    if (done) return;
    finish();
    const now = store.getState();
    // A reset in the meantime put the demo back on its start frame: nothing to open onto.
    if (now.phase !== "public") now.select(eventId);
  }, fallbackMs);
  return finish;
}

/**
 * Mounted once by the shell. On load, with the bundle ready, `?event=<id>` naming a bundled
 * event opens its drawer after the reveal (an unknown id is ignored, so a stale link from
 * another run degrades to the normal start frame). Afterwards the address bar mirrors the
 * selection: `event=<id>` while a drawer is open, removed when it closes.
 */
export function useShareLink(eventIds: ReadonlySet<string> | null): void {
  const ready = eventIds !== null;
  useEffect(() => {
    if (!ready || typeof window === "undefined") return;
    const requested = parseEventParam(window.location.search);
    const stop = requested && eventIds.has(requested) ? openEventAfterReveal(requested) : () => {};
    const unsubscribe = useDemo.subscribe((next, prev) => {
      if (next.selectedEventId === prev.selectedEventId) return;
      const href = next.selectedEventId
        ? eventShareLink(next.selectedEventId)
        : withoutEventParam(window.location.href);
      if (href !== window.location.href) window.history.replaceState(window.history.state, "", href);
    });
    return () => {
      stop();
      unsubscribe();
    };
    // `eventIds` is rebuilt only when the bundle changes; `ready` is its presence.
  }, [ready, eventIds]);
}
