// Public surface of the scene (docs/02 §6 → Components). H4's page mounts <Scene/>.
export { Scene } from "./Canvas";
// The "hidden" hero rule and a one-call select for the shell's keyboard (mock-judging request, REQ-H3-12).
export { hiddenHeroEventId, selectHiddenHero } from "./hiddenHero";
// The guided tour (WEB-09): G plays it; H4 mounts the button next to Run details (REQ-H3-15).
export { TourButton, TOUR_KEY } from "./tour/Tour";
export { startTour, stopTour, useTour } from "./tour/store";
