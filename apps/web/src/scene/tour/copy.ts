// The Tour's words (WEB-09). THIS FILE IS THE COPY FILE: H4 (Sri) owns the wording and may edit these
// strings directly (H3 grants that for this one file; see docs/requests/H4.md → REQ-H3-15). H3 owns
// everything else under scene/tour/.
//
// Rules for editing (copy.test.ts enforces the first two and the length; `make check` runs it):
// - No digits. Numbers reach the screen only through placeholders in {braces}, filled from the bundle:
//     {publicCatalogCount}    public regional catalog events in the window    (meta.summary)
//     {recoveredCatalogCount} of those, matched one-to-one by our pipeline    (meta.summary)
//     {candidateCount}        every candidate event we associated             (meta.summary)
//     {additionalCount}       candidate events with no public-catalog match   (meta.summary)
//     {strictQualityCount}    candidate events in the strict tier (Tier A)    (meta.summary)
//     {strictAdditionalCount} strict events with no public-catalog match     (meta.summary)
//     {heroStations}          stations that agreed on the event the tour opens (its quality.nStations)
//     {replayRate}            the time replay's speed, e.g. "1 hour per second" (the scene's playback rate)
// - Language: docs/00's word list. Say "candidate events" and "public regional catalog"; make no claim
//   of cause or forecast, none about any site or operator, none about structures from our depths.
// - Each caption must still read true on the frame it plays over (listed next to each key).
// - Keep captions short (at most 170 characters): they sit in a lower third and play for a few seconds.

export const TOUR_COPY = {
  /** The start frame: only the public regional catalog is drawn. */
  public: "The public regional catalog lists {publicCatalogCount} events here in this window.",
  /** The reveal: the counters climb and every candidate event appears. */
  reveal:
    "Using only public waveforms, we recovered {recoveredCatalogCount} of {publicCatalogCount}, and associated {additionalCount} more candidate events.",
  /** The strict filter: only the strict tier stays bright. */
  strict:
    "{strictQualityCount} pass our strict tier: every quality metric is within the range reached by three-quarters of the public events we recovered.",
  /** The evidence drawer, open on a strict event the public regional catalog does not list. */
  hidden:
    "{strictAdditionalCount} strict events are not in the public regional catalog. Here is one, with its waveforms: {heroStations} stations agree on its arrivals.",
  /** The Validation card, outlined. */
  validation: "How we checked these numbers: every row on the Validation card is computed from this run.",
  /** Time mode: the window replays and each event appears at its origin time. */
  replay: "The window replayed at {replayRate}: each candidate event appears at its origin time.",
  /** Back to every event, no time gating. */
  full: "{candidateCount} candidate events from public waveforms, against {publicCatalogCount} in the public regional catalog.",

  /** The Tour button's label, and its tooltip. */
  button: "Tour",
  buttonTitle: "Play the guided tour (G). Any key or click stops it.",
} as const;

export type TourCopyKey = keyof typeof TOUR_COPY;
