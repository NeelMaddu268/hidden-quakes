import { afterEach, describe, expect, it, vi } from "vitest";
import type { FetchLike } from "../providers/fetch";
import {
  audioToTNow,
  fmtUtcWindow,
  listenFitsRun,
  listenLabel,
  listenTitle,
  loadListenManifest,
  parseListenManifest,
  resetListenManifests,
} from "./manifest";

// ---- Synthetic manifests (rule 5): the shape hq.preprocess.sonify writes, made-up values ---------
function rawClip(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    stationId: "XX.TST1",
    channel: "HHZ",
    startUtc: "2030-01-02T03:00:00Z",
    endUtc: "2030-01-02T04:00:00Z",
    speed: 240,
    durationS: 15,
    note: "Synthetic test clip.",
    source: "Test source",
    clip: "hour",
    runId: "run-x",
    files: {
      ogg: { name: "clip-x.ogg", bytes: 10, sha256: "00" },
      mp3: { name: "clip-x.mp3", bytes: 12, sha256: "11" },
    },
    ...over,
  };
}

const T0 = Date.UTC(2030, 0, 2, 3) / 1000;

describe("parseListenManifest", () => {
  it("reads the fields the player needs, times as epoch seconds, Ogg before MP3", () => {
    const m = parseListenManifest(rawClip(), "/base")!;
    expect(m).toMatchObject({
      stationId: "XX.TST1",
      channel: "HHZ",
      startUtc: T0,
      endUtc: T0 + 3600,
      speed: 240,
      durationS: 15,
      note: "Synthetic test clip.",
      source: "Test source",
      runId: "run-x",
      heroEventId: null,
      eventOriginUtc: null,
    });
    expect(m.sources).toEqual([
      { src: "/base/clip-x.ogg", type: "audio/ogg" },
      { src: "/base/clip-x.mp3", type: "audio/mpeg" },
    ]);
  });

  it("reads the hero fields and keeps sub-second times", () => {
    const m = parseListenManifest(
      rawClip({
        startUtc: "2030-01-02T03:00:00.250Z",
        endUtc: "2030-01-02T03:01:00.250Z",
        heroEventId: "ev-7",
        selection: { eventOriginUtc: "2030-01-02T03:00:30.250Z" },
      }),
    )!;
    expect(m.heroEventId).toBe("ev-7");
    expect(m.startUtc).toBeCloseTo(T0 + 0.25, 6);
    expect(m.eventOriginUtc).toBeCloseTo(T0 + 30.25, 6);
  });

  it("works with a single playable file", () => {
    const m = parseListenManifest(rawClip({ files: { mp3: { name: "only.mp3" } } }))!;
    expect(m.sources).toEqual([{ src: "/audio/only.mp3", type: "audio/mpeg" }]);
  });

  it.each([
    ["not an object", null],
    ["an array", []],
    ["no station", rawClip({ stationId: "" })],
    ["no channel", rawClip({ channel: undefined })],
    ["a bad start", rawClip({ startUtc: "yesterday" })],
    ["end before start", rawClip({ endUtc: "2030-01-02T02:00:00Z" })],
    ["a zero speed", rawClip({ speed: 0 })],
    ["a string speed", rawClip({ speed: "240" })],
    ["no duration", rawClip({ durationS: undefined })],
    ["no files", rawClip({ files: {} })],
    ["a file name with a path", rawClip({ files: { ogg: { name: "../x.ogg" } } })],
  ])("rejects %s", (_, raw) => {
    expect(parseListenManifest(raw)).toBeNull();
  });
});

describe("clip ↔ scene clock", () => {
  const m = parseListenManifest(rawClip())!;

  it("maps audio seconds to data time at the manifest's speed, clamped to the clip window", () => {
    expect(audioToTNow(m, 0)).toBe(T0);
    expect(audioToTNow(m, 1.5)).toBe(T0 + 360);
    expect(audioToTNow(m, 15)).toBe(T0 + 3600);
    expect(audioToTNow(m, 99)).toBe(T0 + 3600);
    expect(audioToTNow(m, -1)).toBe(T0);
    expect(audioToTNow(m, Number.NaN)).toBe(T0);
  });

  it("fits a run only inside its window and, when both name one, the same run", () => {
    const run = { windowStart: T0 - 3 * 3600, windowEnd: T0 + 21 * 3600 };
    expect(listenFitsRun(m, run, "run-x")).toBe(true);
    expect(listenFitsRun(m, run, undefined)).toBe(true);
    expect(listenFitsRun(m, run, "run-y")).toBe(false);
    expect(listenFitsRun(m, { windowStart: T0 + 60, windowEnd: run.windowEnd }, "run-x")).toBe(false);
    expect(listenFitsRun(m, { windowStart: run.windowStart, windowEnd: T0 + 1800 }, "run-x")).toBe(false);
  });
});

describe("labels (every value from the manifest)", () => {
  it("names the station and the speed-up", () => {
    expect(listenLabel(parseListenManifest(rawClip())!)).toBe("Listen: XX.TST1, sped up 240×");
    expect(listenLabel(parseListenManifest(rawClip({ stationId: "YY.B", speed: 1500 }))!)).toBe(
      "Listen: YY.B, sped up 1,500×",
    );
  });

  it("formats the UTC window, repeating the date only across midnight", () => {
    expect(fmtUtcWindow(T0, T0 + 3600)).toBe("2030-01-02 03:00:00–04:00:00 UTC");
    expect(fmtUtcWindow(T0 + 0.25, T0 + 60.25)).toBe("2030-01-02 03:00:00.250–03:01:00.250 UTC");
    expect(fmtUtcWindow(T0 + 20 * 3600, T0 + 22 * 3600)).toBe("2030-01-02 23:00:00 – 2030-01-03 01:00:00 UTC");
  });

  it("hover text carries the channel, the window, the note and the source", () => {
    expect(listenTitle(parseListenManifest(rawClip())!)).toBe(
      "XX.TST1 channel HHZ · 2030-01-02 03:00:00–04:00:00 UTC. Synthetic test clip. Source: Test source.",
    );
    const hero = parseListenManifest(
      rawClip({ note: "", source: undefined, selection: { eventOriginUtc: "2030-01-02T03:30:00.5Z" } }),
    )!;
    expect(listenTitle(hero)).toBe("XX.TST1 channel HHZ · 2030-01-02 03:00:00–04:00:00 UTC · event origin 03:30:00.500 UTC");
  });
});

describe("loadListenManifest", () => {
  afterEach(() => {
    resetListenManifests();
    vi.restoreAllMocks();
  });

  const respond =
    (status: number, body: unknown): FetchLike =>
    async () => ({ ok: status >= 200 && status < 300, status, json: async () => body });

  it("fetches <base>/<name>.json once and parses it", async () => {
    const fetchImpl = vi.fn(respond(200, rawClip()));
    const a = await loadListenManifest("clip-x", fetchImpl, "/snd");
    const b = await loadListenManifest("clip-x", fetchImpl, "/snd");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(fetchImpl.mock.calls[0][0]).toBe("/snd/clip-x.json");
    expect(a?.stationId).toBe("XX.TST1");
    expect(b).toBe(a);
  });

  it("resolves to null (never rejects) when the manifest is missing, broken or unreachable", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await loadListenManifest("missing", respond(404, null))).toBeNull();
    expect(await loadListenManifest("invalid", respond(200, { stationId: "XX.TST1" }))).toBeNull();
    expect(await loadListenManifest("server", respond(500, null))).toBeNull();
    expect(
      await loadListenManifest("offline", async () => {
        throw new TypeError("network down");
      }),
    ).toBeNull();
  });
});
