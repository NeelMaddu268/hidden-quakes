import { afterEach, describe, expect, it } from "vitest";
import { makeRectList } from "./labelPlacement";
import { measureOverlayRects, OverlayObstacles, shellRoot } from "./overlayObstacles";

/** A test-local DOM shaped like the shell: root > header > p[data-testid=mode-label], plus sibling blocks. */
function box<T extends HTMLElement>(el: T, x: number, y: number, w: number, h: number): T {
  el.getBoundingClientRect = () => ({ left: x, top: y, width: w, height: h, right: x + w, bottom: y + h, x, y, toJSON() {} }) as DOMRect;
  return el;
}

function buildShell() {
  const root = document.createElement("div");
  const header = box(document.createElement("header"), 24, 24, 449, 69);
  const label = document.createElement("p");
  label.dataset.testid = "mode-label";
  header.append(label);
  const counters = box(document.createElement("div"), 880, 24, 375, 130);
  const hidden = box(document.createElement("div"), 0, 0, 0, 0); // unmounted/display:none blocks measure 0×0
  root.append(header, counters, hidden);
  document.body.append(root);
  return root;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("overlay obstacles", () => {
  it("finds the shell root through the shell's mode-label test id", () => {
    const root = buildShell();
    expect(shellRoot(document)).toBe(root);
  });

  it("measures the shell's visible blocks and H3's open panels, relative to the canvas origin", () => {
    buildShell();
    const section = box(document.createElement("section"), 24, 136, 360, 284);
    section.dataset.testid = "depth-section";
    const closedDrawer = box(document.createElement("aside"), 768, 0, 512, 720);
    closedDrawer.className = "hqd";
    closedDrawer.dataset.open = "false";
    document.body.append(section, closedDrawer);
    const out = makeRectList(8);
    measureOverlayRects(document, { left: 10, top: 5 }, out);
    const rects = Array.from({ length: out.count }, (_, k) => Array.from(out.rects.slice(k * 4, k * 4 + 4)));
    expect(rects).toEqual([
      [14, 19, 449, 69], // header
      [870, 19, 375, 130], // counters + filters
      [14, 131, 360, 284], // depth section
    ]); // the zero-size block and the closed drawer are skipped
    closedDrawer.dataset.open = "true";
    measureOverlayRects(document, { left: 0, top: 0 }, out);
    expect(out.count).toBe(4);
  });

  it("is empty without a shell and replaces its contents on every measurement", () => {
    const o = new OverlayObstacles(4);
    const canvas = box(document.createElement("canvas"), 0, 0, 1280, 720);
    o.measure(document, canvas);
    expect(o.list.count).toBe(0);
    buildShell();
    o.measure(document, canvas);
    o.measure(document, canvas);
    expect(o.list.count).toBe(2);
  });
});
