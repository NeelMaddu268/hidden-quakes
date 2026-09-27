// The "hidden" hero (mock judging, H2's request): the demo's E key opens `meta.scene.heroEventId`, a
// public-catalog event, so the demo never showed an event the public regional catalog doesn't have. This
// rule picks one from the data: a Tier A event with no catalog match, the most stations that agreed,
// then the lowest RMS residual, then the earliest reveal slot (deterministic). H4's keyboard binds it.

import { useDemo } from "../state/demo";
import type { SeismicEvent } from "./types";

type HeroCandidate = Pick<SeismicEvent, "id" | "tier" | "catalogMatch" | "revealOrder"> & {
  quality: Pick<SeismicEvent["quality"], "nStations" | "rmsS">;
};

const rms = (e: HeroCandidate) => (Number.isFinite(e.quality.rmsS) ? e.quality.rmsS : Infinity);

/** The id of the Tier A event with no public-catalog match that most stations agreed on, or null. */
export function hiddenHeroEventId(events: readonly HeroCandidate[]): string | null {
  let best: HeroCandidate | null = null;
  for (const e of events) {
    if (e.tier !== "A" || e.catalogMatch != null || !Number.isFinite(e.quality.nStations)) continue;
    if (
      best === null ||
      e.quality.nStations > best.quality.nStations ||
      (e.quality.nStations === best.quality.nStations &&
        (rms(e) < rms(best) || (rms(e) === rms(best) && e.revealOrder < best.revealOrder)))
    ) {
      best = e;
    }
  }
  return best?.id ?? null;
}

/**
 * Opens the drawer on the hidden hero (for H4's keyboard: one call, like E's `select(heroEventId)`).
 * Returns the selected id, or null when the bundle has no such event (nothing changes).
 */
export function selectHiddenHero(events: readonly HeroCandidate[]): string | null {
  const id = hiddenHeroEventId(events);
  if (id !== null) useDemo.getState().select(id);
  return id;
}
