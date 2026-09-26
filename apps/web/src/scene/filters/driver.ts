// The filter look's per-frame driver, free of React so it's testable against the real store:
// eases toward the effective filter and publishes the result as `sceneFx.filterLook`.

import { useDemo } from "../../state/demo";
import { sceneFx } from "../fx";
import { createFilterFade, effectiveFilter, type FilterFade } from "./fade";

export interface FilterLookDriver {
  /** reset(): jump straight to the start look, so the start frame is exact. */
  onReset(): void;
  step(deltaS: number): void;
}

function publish(fade: FilterFade): void {
  const look = sceneFx.filterLook;
  const cur = fade.current;
  look.tierA = cur.tierA;
  look.tierB = cur.tierB;
  look.tierC = cur.tierC;
  look.candidates = cur.candidates;
  look.publicLayer = cur.publicLayer;
  look.halos = cur.halos;
}

export function createFilterLookDriver(durationS?: number): FilterLookDriver {
  const s = useDemo.getState();
  const fade = createFilterFade(effectiveFilter(s.phase, s.filter), durationS);
  publish(fade);
  return {
    onReset() {
      const st = useDemo.getState();
      fade.snap(effectiveFilter(st.phase, st.filter));
      publish(fade);
    },
    step(deltaS) {
      const st = useDemo.getState();
      fade.step(effectiveFilter(st.phase, st.filter), deltaS);
      publish(fade);
    },
  };
}
