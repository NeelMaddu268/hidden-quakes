// The drawer's open-latency log line (WEB-05 acceptance: opens in under 200 ms from preloaded evidence).

/** The budget for select() → traces painted, from docs/lanes/H3 → WEB-05. */
export const OPEN_BUDGET_MS = 200;

/**
 * "[drawer] <id>: traces committed 18.2 ms · frame 21.0 ms · painted 37.5 ms after select() (budget 200 ms)".
 * `committed`: traces in the DOM; `frame`: the frame that paints them starts; `painted`: the next frame
 * starts, so that paint is done. Over budget is flagged, never hidden.
 */
export function formatOpenLatency(eventId: string, committedMs: number, frameMs: number, paintedMs: number): string {
  const f = (v: number) => `${v.toFixed(1)} ms`;
  const over = paintedMs > OPEN_BUDGET_MS ? " — OVER BUDGET" : "";
  return (
    `[drawer] ${eventId}: traces committed ${f(committedMs)} · frame ${f(frameMs)} · painted ${f(paintedMs)} ` +
    `after select() (budget ${OPEN_BUDGET_MS} ms)${over}`
  );
}
