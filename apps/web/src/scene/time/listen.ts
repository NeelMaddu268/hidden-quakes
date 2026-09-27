// Listen (WEB-10): while a clip plays, the audio owns the playhead. The TimeDriver reads `listenTNow()`
// every frame and puts the scene clock exactly where the audio is (tNow = startUtc + currentTime ×
// speed, clamped to the clip window), so sound and visuals cannot drift: a stalled or buffering clip
// stalls the replay with it, and a background tab catches up on its first frame back.

import { audioToTNow, type ListenManifest } from "../../audio/manifest";

export interface ListenClock {
  manifest: Pick<ListenManifest, "startUtc" | "endUtc" | "speed">;
  audio: { readonly currentTime: number };
}

let active: ListenClock | null = null;

/** Hand the playhead to a playing clip (or back to the replay with null). */
export function setListenClock(clock: ListenClock | null): void {
  active = clock;
}

/** The clip currently driving the playhead, if any. */
export function listenClock(): ListenClock | null {
  return active;
}

/** Where the playing clip puts the scene clock (epoch s), or null when no clip drives it. */
export function listenTNow(): number | null {
  return active ? audioToTNow(active.manifest, active.audio.currentTime) : null;
}
