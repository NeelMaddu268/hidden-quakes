import { afterEach, describe, expect, it, vi } from "vitest";
import { LabelElements } from "./useLabelElements";

function sized(width: number, height: number): HTMLElement {
  const el = document.createElement("div");
  Object.defineProperty(el, "offsetWidth", { configurable: true, get: () => width });
  Object.defineProperty(el, "offsetHeight", { configurable: true, get: () => height });
  return el;
}

class FakeResizeObserver {
  static last: FakeResizeObserver | null = null;
  observed = new Set<Element>();
  constructor(private readonly cb: (entries: { target: Element }[]) => void) {
    FakeResizeObserver.last = this;
  }
  observe(el: Element) {
    this.observed.add(el);
    this.cb([{ target: el }]); // like the real one: an initial notification per observed element
  }
  unobserve(el: Element) {
    this.observed.delete(el);
  }
  disconnect() {
    this.observed.clear();
  }
  fire(el: Element) {
    this.cb([{ target: el }]);
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  FakeResizeObserver.last = null;
});

describe("LabelElements", () => {
  it("hands out one stable ref callback per index", () => {
    const labels = new LabelElements();
    expect(labels.ref(0)).toBe(labels.ref(0));
    expect(labels.ref(0)).not.toBe(labels.ref(1));
  });

  it("without ResizeObserver, measures once on attach and zeroes on detach", () => {
    vi.stubGlobal("ResizeObserver", undefined);
    const labels = new LabelElements();
    expect(labels.el(0)).toBeNull();
    expect(labels.width(0)).toBe(0);
    const el = sized(120, 14);
    labels.ref(0)(el);
    expect(labels.el(0)).toBe(el);
    expect([labels.width(0), labels.height(0)]).toEqual([120, 14]);
    labels.ref(0)(null);
    expect(labels.el(0)).toBeNull();
    expect([labels.width(0), labels.height(0)]).toEqual([0, 0]);
  });

  it("bumps its version on every attach and detach (so styles are re-applied to a remounted label)", () => {
    vi.stubGlobal("ResizeObserver", undefined);
    const labels = new LabelElements();
    const v0 = labels.version;
    const el = sized(10, 10);
    labels.ref(2)(el);
    expect(labels.version).toBe(v0 + 1);
    labels.ref(2)(el); // same element again: no change
    expect(labels.version).toBe(v0 + 1);
    labels.ref(2)(null);
    expect(labels.version).toBe(v0 + 2);
  });

  it("tracks size changes through ResizeObserver (e.g. the web font loading)", () => {
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    const labels = new LabelElements();
    let width = 100;
    const el = document.createElement("div");
    Object.defineProperty(el, "offsetWidth", { configurable: true, get: () => width });
    Object.defineProperty(el, "offsetHeight", { configurable: true, get: () => 14 });
    labels.ref(0)(el);
    const ro = FakeResizeObserver.last!;
    expect(ro.observed.has(el)).toBe(true);
    expect(labels.width(0)).toBe(100);
    width = 132;
    ro.fire(el);
    expect(labels.width(0)).toBe(132);
    labels.ref(0)(null);
    expect(ro.observed.has(el)).toBe(false);
  });

  it("disconnect stops observing; connect re-observes every attached element", () => {
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    const labels = new LabelElements();
    const a = sized(10, 10);
    const b = sized(20, 10);
    labels.ref(0)(a);
    labels.ref(1)(b);
    const ro = FakeResizeObserver.last!;
    labels.disconnect();
    expect(ro.observed.size).toBe(0);
    labels.connect();
    expect([...ro.observed]).toEqual([a, b]);
  });
});
