# Expo cheat card (DEMO-05)

One printed page for whoever stands at the table. Every number is a placeholder in braces; print `data/story/expo-card-filled.md` from `make story`, never this file. Language rules are in `docs/demo/language-audit.md`: say "candidate events" and "public regional catalog", and never name a cause for any event.

**Hard max: three minutes per judge.** At two and a half, close with the user line and hand them the link.

## The ninety-second demo

| Time | Do | Say |
| --- | --- | --- |
| 0:00–0:20 | Start frame, nothing pressed | "Enhanced geothermal is expanding near Milford, Utah, and its oversight is built on tiny earthquakes. Operators watch them with dense downhole sensors. Everyone else gets the public regional catalog: {publicCatalogCount} events here on {windowDate}." |
| 0:20–0:35 | **Space** (reveal), let the counter finish | "We rebuilt the catalog from the same public seismometers: {candidateCount} candidate events, {strictQualityCount} at strict quality." |
| 0:35–0:55 | **H**: the drawer opens on a strict event the public catalog doesn't list | "Here's one the public catalog doesn't list. {hiddenHeroStations} stations agreed. The neural picker marks P and S on each seismogram, and the lines are the arrivals its location implies. They agree. That agreement is what makes a dot an event." |
| 0:55–1:15 | **Esc**, point at the Validation card | "We recover {recoveredCatalogCount} of {publicCatalogCount} public events. With every station's clock scrambled, about {meanChanceEvents} chance events show up and none at strict quality, in {nShuffles} scrambles. Our own model, the {confidenceLabel}, tells real timing from scrambled decoys with a held-out ROC AUC of {heldOutRocAuc}." |
| 1:15–1:30 | Full scene, one slow orbit | "Operators can see underground. A county official or a reporter can't. This is an open, public-data-only window into what the public network is already hearing." |

If they want more: **G** plays the guided tour, **P** is plan view, **Event list** is every candidate, **?** is how it works and the keys.

## The eight hardest questions

1. **Are the extra events real?** They're candidate events: each needs consistent picks across stations. We stand behind the {strictQualityCount} strict ones; {strictAdditionalCount} of those aren't in the public catalog.
2. **What's your false-positive rate?** No ground truth exists for events the catalog lacks. Our proxy: scramble every station's clock, and about {meanChanceEvents} chance events appear per scramble, none strict in {nShuffles} scrambles.
3. **Isn't PhaseNet doing everything?** It picks one station at a time. An event exists only when picks agree across stations through a velocity model; association, location, corrections, tiers and matching are ours.
4. **Did you train anything?** Not the picker: pretrained PhaseNet, weights chosen by A/B. We trained the {confidenceLabel} this weekend: held-out ROC AUC {heldOutRocAuc}, {heldOutRocAucEqualStations} even at equal station count. It is not a probability that an event is an earthquake.
5. **Why trust the depths?** We don't claim more than the data gives: ±{medianVErrM} m is the synthetic best case with every station; held out against the public catalog, the median depth difference is {heldOutMedianAbsDzM} m. Depth is our biggest weakness and we say so.
6. **How is this better than STA/LTA?** Same day, same downstream code and tier bars: STA/LTA gives {strictStalta} strict events out of {staltaCandidates} candidates.
7. **Did the geothermal work make these?** We never attribute. Several operations share the region; we show where and when candidate events occur, not why.
8. **Should people nearby worry?** We make no safety claims and no forecasts. What the events mean for hazard is for seismologists and regulators with far more than one day of records.

## Between judges

- Press **R**: back to the start frame, drawer closed, counters reset. Leave the start frame on screen.
- Hands off the keyboard: after a minute idle on the start frame the guided tour plays by itself, so passers-by see the reveal. Any key or click takes control back.
- For a walk-up screen with nobody at the table, open the site with `?kiosk=1`: the tour loops until someone touches it.
- If the browser misbehaves, reload. The live link is on the Devpost.
