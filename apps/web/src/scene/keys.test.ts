import { describe, expect, it } from "vitest";
import { isPlainPress, isTextInput } from "./keys";

const key = (init: KeyboardEventInit, target?: EventTarget) => {
  const e = new KeyboardEvent("keydown", { key: "h", cancelable: true, ...init });
  if (target) Object.defineProperty(e, "target", { value: target });
  return e;
};

describe("scene keys", () => {
  it("treats inputs, textareas, selects and contenteditable as text fields", () => {
    for (const tag of ["input", "textarea", "select"]) expect(isTextInput(document.createElement(tag))).toBe(true);
    const div = document.createElement("div");
    expect(isTextInput(div)).toBe(false);
    Object.defineProperty(div, "isContentEditable", { value: true });
    expect(isTextInput(div)).toBe(true);
    expect(isTextInput(null)).toBe(false);
    expect(isTextInput(window)).toBe(false);
  });

  it("accepts only plain presses", () => {
    expect(isPlainPress(key({}))).toBe(true);
    expect(isPlainPress(key({ shiftKey: true }))).toBe(true);
    for (const init of [{ metaKey: true }, { ctrlKey: true }, { altKey: true }, { repeat: true }]) {
      expect(isPlainPress(key(init))).toBe(false);
    }
    const handled = key({});
    handled.preventDefault();
    expect(isPlainPress(handled)).toBe(false);
    expect(isPlainPress(key({}, document.createElement("input")))).toBe(false);
  });
});
