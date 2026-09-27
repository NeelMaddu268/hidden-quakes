// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { measureShellEdges, updatePlanDock, usePlanDock } from "./dock";

// A shell-shaped DOM with fixed rects (jsdom has no layout).
function el(tag: string, rect: { top: number; bottom: number; left?: number; right?: number } | null, attrs: Record<string, string> = {}) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  const r = rect ?? { top: 0, bottom: 0, left: 0, right: 0 };
  const left = r.left ?? 0;
  const right = r.right ?? (rect ? left + 100 : 0);
  e.getBoundingClientRect = () =>
    ({ top: r.top, bottom: r.bottom, left, right, width: right - left, height: r.bottom - r.top, x: left, y: r.top, toJSON() {} }) as DOMRect;
  return e;
}

function shell(opts: { card: boolean }) {
  const root = el("div", { top: 0, bottom: 720 });
  const header = el("header", { top: 24, bottom: 144.4 });
  header.appendChild(el("p", { top: 60, bottom: 76 }, { "data-testid": "mode-label" }));
  const topRight = el("div", { top: 24, bottom: 154.4, left: 880, right: 1256 });
  const counters = el("dl", { top: 24, bottom: 110, left: 880, right: 1256 });
  counters.appendChild(el("dd", { top: 60, bottom: 110 }, { "data-testid": "counter-public" }));
  topRight.appendChild(counters);
  root.append(header, topRight);
  if (opts.card) root.appendChild(el("section", { top: 320.6, bottom: 652 }, { "data-testid": "validation-panel" }));
  document.body.append(root, el("div", { top: 651, bottom: 696, left: 1036, right: 1256 }, { "data-testid": "scene-legend" }));
}

beforeEach(() => usePlanDock.setState({ dock: null, measured: false }));
afterEach(() => {
  document.body.innerHTML = "";
});

describe("measureShellEdges", () => {
  it("reads the title block, the card, the counters' block and the legend", () => {
    shell({ card: true });
    expect(measureShellEdges(document)).toEqual({ headerBottom: 144.4, cardTop: 320.6, topRightBottom: 154.4, cornerTop: 651 });
  });

  it("reports blocks that aren't on screen as null", () => {
    expect(measureShellEdges(document)).toEqual({ headerBottom: null, cardTop: null, topRightBottom: null, cornerTop: null });
  });
});

describe("updatePlanDock", () => {
  it("publishes the dock, and plans for a card measured earlier at this size when it's gone", () => {
    shell({ card: true });
    updatePlanDock(document, 1280, 720);
    const withCard = usePlanDock.getState().dock;
    expect(withCard?.side).toBe("right");
    // The card unmounts (reset to the start frame): the dock stays where the card will be again.
    document.querySelector('[data-testid="validation-panel"]')!.remove();
    updatePlanDock(document, 1280, 720);
    expect(usePlanDock.getState().dock).toEqual(withCard);
  });

  it("publishes only changes", () => {
    shell({ card: true });
    updatePlanDock(document, 1280, 720);
    const first = usePlanDock.getState().dock;
    updatePlanDock(document, 1280, 720);
    expect(usePlanDock.getState().dock).toBe(first);
  });
});
