// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COPIED_MS, Header } from "./Header";

// REQ-H4-3: the drawer header's "Copy link" copies the shell's share link for the open event.

const button = () => screen.getByTestId("copy-link");
const flush = () => act(async () => {});

let writeText: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.useFakeTimers();
  window.history.replaceState(null, "", "/?mode=showcase#top");
  writeText = vi.fn(() => Promise.resolve());
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const renderHeader = (eventId = "hq-run-000181") =>
  render(<Header eventId={eventId} event={null} meta={null} onClose={() => {}} />);

describe("Copy link", () => {
  it("copies the page's link with ?event=<id> (mode kept, hash dropped), then says Copied for a moment", async () => {
    renderHeader();
    expect(button().textContent).toBe("Copy link");
    fireEvent.click(button());
    await flush();
    expect(writeText).toHaveBeenCalledWith(`${window.location.origin}/?mode=showcase&event=hq-run-000181`);
    expect(button().textContent).toBe("Copied");
    expect(button().dataset.state).toBe("copied");
    act(() => vi.advanceTimersByTime(COPIED_MS));
    expect(button().textContent).toBe("Copy link");
  });

  it("says so when the clipboard refuses, and logs why", async () => {
    writeText.mockImplementation(() => Promise.reject(new Error("NotAllowedError")));
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    renderHeader();
    fireEvent.click(button());
    await flush();
    expect(button().textContent).toBe("Copy failed");
    expect(error).toHaveBeenCalledWith("Copy link: the clipboard refused the write", expect.any(Error));
  });

  it("says so when the page has no clipboard at all (an insecure origin)", () => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    renderHeader();
    fireEvent.click(button());
    expect(button().textContent).toBe("Copy failed");
    expect(error).toHaveBeenCalledWith("Copy link: the clipboard is unavailable on this page");
  });

  it("starts fresh on another event's drawer", async () => {
    const { rerender } = renderHeader("e1");
    fireEvent.click(button());
    await flush();
    expect(button().textContent).toBe("Copied");
    rerender(<Header eventId="e2" event={null} meta={null} onClose={() => {}} />);
    expect(button().textContent).toBe("Copy link");
    fireEvent.click(button());
    await flush();
    expect(writeText).toHaveBeenLastCalledWith(`${window.location.origin}/?mode=showcase&event=e2`);
  });

  it("carries no digits in its words", () => {
    renderHeader();
    expect(button().textContent).not.toMatch(/\d/);
    expect(button().title).not.toMatch(/\d/);
  });
});
