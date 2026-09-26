/**
 * Overnight feature 3: `?event=<id>` opens the drawer after the reveal, and the address bar
 * follows the selection. The store is the real demo store; the scene's settle is simulated with
 * `finishReveal()`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { finishReveal, useDemo } from "@/state/demo";
import { eventShareLink, openEventAfterReveal, parseEventParam, withoutEventParam } from "./share";

const state = () => useDemo.getState();

beforeEach(() => {
  state().reset();
});

afterEach(() => {
  state().reset();
});

describe("parseEventParam()", () => {
  it("reads the id, ignoring blanks and other parameters", () => {
    expect(parseEventParam("?mode=showcase&event=hq-1-000301")).toBe("hq-1-000301");
    expect(parseEventParam("?event=%20")).toBeNull();
    expect(parseEventParam("?mode=showcase")).toBeNull();
    expect(parseEventParam("")).toBeNull();
  });
});

describe("eventShareLink() and withoutEventParam()", () => {
  it("sets the event on the current page, keeps the mode, drops any hash", () => {
    const href = "https://example.test/?mode=showcase#x";
    expect(eventShareLink("hq-1-000301", href)).toBe("https://example.test/?mode=showcase&event=hq-1-000301");
    expect(eventShareLink("b", "https://example.test/?event=a")).toBe("https://example.test/?event=b");
    expect(withoutEventParam("https://example.test/?mode=live&event=a")).toBe("https://example.test/?mode=live");
    expect(withoutEventParam("https://example.test/")).toBe("https://example.test/");
  });
});

describe("openEventAfterReveal()", () => {
  it("from the start frame: starts the reveal, then selects once the scene reports revealed", () => {
    const stop = openEventAfterReveal("ev-1");
    expect(state().phase).toBe("revealing");
    expect(state().selectedEventId).toBeNull();
    finishReveal();
    expect(state().phase).toBe("revealed");
    expect(state().selectedEventId).toBe("ev-1");
    stop();
  });

  it("mid-reveal: waits without restarting the reveal", () => {
    state().reveal();
    const stop = openEventAfterReveal("ev-2");
    expect(state().phase).toBe("revealing");
    finishReveal();
    expect(state().selectedEventId).toBe("ev-2");
    stop();
  });

  it("already revealed: selects at once", () => {
    state().reveal();
    finishReveal();
    openEventAfterReveal("ev-3")();
    expect(state().selectedEventId).toBe("ev-3");
  });

  it("falls back to selecting after the timeout when the settle never comes, unless reset meanwhile", () => {
    vi.useFakeTimers();
    try {
      const stop = openEventAfterReveal("ev-slow", useDemo, 1000);
      vi.advanceTimersByTime(999);
      expect(state().selectedEventId).toBeNull();
      vi.advanceTimersByTime(1);
      expect(state().phase).toBe("revealing");
      expect(state().selectedEventId).toBe("ev-slow");
      // The settle arriving afterwards must not select twice or fight a later choice.
      state().select("ev-other");
      finishReveal();
      expect(state().selectedEventId).toBe("ev-other");
      stop();

      state().reset();
      openEventAfterReveal("ev-reset", useDemo, 1000);
      state().reset();
      vi.advanceTimersByTime(1000);
      expect(state().selectedEventId).toBeNull();

      state().reset();
      const cancel = openEventAfterReveal("ev-cancel", useDemo, 1000);
      cancel();
      vi.advanceTimersByTime(1000);
      expect(state().selectedEventId).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("selects nothing after the unsubscribe or after a reset before the settle", () => {
    const stop = openEventAfterReveal("ev-4");
    stop();
    finishReveal();
    expect(state().selectedEventId).toBeNull();

    state().reset();
    const stop2 = openEventAfterReveal("ev-5");
    state().reset();
    // A later reveal that settles must not resurrect the old request once it has been cancelled.
    stop2();
    state().reveal();
    finishReveal();
    expect(state().selectedEventId).toBeNull();
  });
});
