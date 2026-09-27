"use client";

import { useEffect } from "react";
import { useDemo } from "../state/demo";
import { useBundle } from "./data";
import { selectHiddenHero } from "./hiddenHero";
import { isPlainPress } from "./keys";
import { useTour } from "./tour/store";

/** The key that opens the hidden hero (no shell key uses it; docs/02 §6 → Keyboard, REQ-H3-12). */
export const HIDDEN_HERO_KEY = "h";

/**
 * H opens the evidence drawer on the "hidden" hero (mock judging): the Tier A event with no public-
 * catalog match that most stations agreed on (`hiddenHeroEventId`). Only once the reveal has begun:
 * before it, no candidate event exists on screen to open. While the guided tour plays, H stops the tour
 * like any key (the tour's own listener) and opens nothing. Mounted by the scene; renders nothing.
 */
export function HiddenHeroKey() {
  const bundle = useBundle();
  const events = bundle.status === "ready" ? bundle.events : null;
  useEffect(() => {
    if (events === null) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== HIDDEN_HERO_KEY || !isPlainPress(event)) return;
      if (useTour.getState().running) return;
      if (useDemo.getState().phase === "public") return;
      if (selectHiddenHero(events) !== null) event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [events]);
  return null;
}
