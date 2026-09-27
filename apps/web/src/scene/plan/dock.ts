// The depth section's dock, shared (WEB-13): measured once for the panel (DOM, outside the canvas) and
// read by the plan camera (its reserve) and the feature labels (an obstacle), so all three agree. The
// shell's edges are measured read-only from specific blocks, never from centered modals (Run details,
// the event list, help), so opening one never re-frames the plan.

import { create, type StoreApi, type UseBoundStore } from "zustand";
import { sectionDock, type DockedPanel, type ShellEdges } from "./layout";

/** The shell's edges now (viewport CSS px), from its test ids and H3's own corner stack. */
export function measureShellEdges(doc: Document): ShellEdges {
  const rect = (el: Element | null | undefined) => {
    const r = el?.getBoundingClientRect();
    return r && r.width > 0 && r.height > 0 ? r : null;
  };
  // The title block is the mode line's parent (header); the top-right block holds the counters.
  const header = rect(doc.querySelector('[data-testid="mode-label"]')?.parentElement);
  const card = rect(doc.querySelector('[data-testid="validation-panel"]'));
  const counters = doc.querySelector('[data-testid="counter-public"]');
  const shell = doc.querySelector('[data-testid="mode-label"]')?.parentElement?.parentElement ?? null;
  let topRight: Element | null = counters;
  while (topRight && topRight.parentElement && topRight.parentElement !== shell) topRight = topRight.parentElement;
  const topRightRect = rect(topRight);
  let cornerTop: number | null = null;
  for (const sel of ['[data-testid="scene-legend"]', '[data-testid="vertical-badge"]', '[data-testid="abstract-surface-note"]']) {
    for (const el of Array.from(doc.querySelectorAll(sel))) {
      const r = rect(el);
      if (r && (cornerTop === null || r.top < cornerTop)) cornerTop = r.top;
    }
  }
  return {
    headerBottom: header ? header.bottom : null,
    cardTop: card ? card.top : null,
    topRightBottom: topRightRect ? topRightRect.bottom : null,
    cornerTop,
  };
}

export const usePlanDock: UseBoundStore<StoreApi<{ dock: DockedPanel | null; measured: boolean }>> = create<{
  dock: DockedPanel | null;
  measured: boolean;
}>()(() => ({ dock: null, measured: false }));

const same = (a: DockedPanel | null, b: DockedPanel | null) =>
  a === b ||
  (a !== null && b !== null && a.side === b.side && a.left === b.left && a.top === b.top && a.width === b.width && a.height === b.height);

/**
 * The validation card's last measured distance from the viewport's bottom, per viewport size: before the
 * reveal (no card yet) the dock plans for the card that will appear, so a plan view entered before the
 * reveal doesn't re-frame when the card arrives.
 */
const cardFromBottom = new Map<string, number>();

/** Re-measure and publish the dock (only on change, so readers re-render only when it moves). */
export function updatePlanDock(doc: Document, viewportWidth: number, viewportHeight: number): void {
  const edges = measureShellEdges(doc);
  const key = `${viewportWidth}x${viewportHeight}`;
  if (edges.cardTop !== null) cardFromBottom.set(key, viewportHeight - edges.cardTop);
  else if (cardFromBottom.has(key)) edges.cardTop = viewportHeight - cardFromBottom.get(key)!;
  const next = sectionDock(viewportWidth, viewportHeight, edges);
  const cur = usePlanDock.getState();
  if (cur.measured && same(cur.dock, next)) return;
  usePlanDock.setState({ dock: next, measured: true });
}
