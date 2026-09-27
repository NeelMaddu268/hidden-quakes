import { afterEach, describe, expect, it, vi } from "vitest";
import type { FetchLike } from "../../providers/fetch";
import {
  loadStationDayManifest,
  parseStationDayManifest,
  STATION_DAY_MANIFEST_URL,
  stationDayImageUrl,
} from "./manifest";
import { syntheticManifest } from "./test-fixture";

const withField = (field: string, value: unknown) => ({ ...syntheticManifest(), [field]: value });
const withLegend = (patch: Record<string, unknown>) => {
  const m = syntheticManifest();
  const legend = m.legend as Record<string, unknown>[];
  return { ...m, legend: [{ ...legend[0], ...patch }, ...legend.slice(1)] };
};

const respond =
  (status: number, body: unknown, calls: string[] = []): FetchLike =>
  async (url) => {
    calls.push(url);
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => {
        if (body instanceof Error) throw body;
        return body;
      },
    };
  };

afterEach(() => {
  vi.restoreAllMocks();
});

describe("parseStationDayManifest", () => {
  it("accepts the contract and keeps every field", () => {
    const raw = syntheticManifest();
    const parsed = parseStationDayManifest({ ...raw, extra: "ignored" });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.manifest).toEqual(raw);
    expect("extra" in parsed.manifest).toBe(false);
  });

  it("accepts a null sensor depth, 60-minute rows, no runners-up and an empty legend", () => {
    const m = {
      ...syntheticManifest(),
      sensorDepthM: null,
      rowMinutes: 60,
      rows: 24,
      legend: [],
      selection: { rule: "r", pickCount: 0, runnersUp: [] },
    };
    expect(parseStationDayManifest(m).ok).toBe(true);
  });

  it.each([
    ["not an object", null],
    ["an array", []],
    ["a string", "station-day.png"],
  ])("rejects %s", (_name, raw) => {
    expect(parseStationDayManifest(raw).ok).toBe(false);
  });

  it.each([
    ["image with a folder", withField("image", "../secret.png")],
    ["image that is not a PNG", withField("image", "station-day.svg")],
    ["missing caption", withField("caption", undefined)],
    ["blank title", withField("title", "  ")],
    ["fractional width", withField("widthPx", 1920.5)],
    ["zero height", withField("heightPx", 0)],
    ["station id without a network", withField("stationId", "SYN01")],
    ["seed id with three parts", withField("seedId", "XX.SYN01.ZZZ")],
    ["sensor depth as text", withField("sensorDepthM", "123")],
    ["day in another format", withField("dayUtc", "03/02/2001")],
    ["start without Z", withField("startUtc", "2001-02-03T00:00:00")],
    ["end before start", withField("endUtc", "2001-02-02T00:00:00Z")],
    ["45-minute rows", withField("rowMinutes", 45)],
    ["zero rows", withField("rows", 0)],
    ["inverted band", withField("filterHz", [20, 2])],
    ["band of one value", withField("filterHz", [2])],
    ["filled gaps", withField("gapsFilled", true)],
    ["negative gap seconds", withField("gapSeconds", -1)],
    ["coverage above one", withField("coverageFraction", 1.2)],
    ["marker time other than origin", withField("markerTime", "pick")],
    ["legend not an array", withField("legend", {})],
    ["unknown legend key", withLegend({ key: "tierD" })],
    ["legend color not #RRGGBB", withLegend({ color: "orange" })],
    ["legend opacity above one", withLegend({ opacity: 2 })],
    ["unknown legend shape", withLegend({ shape: "circle" })],
    ["fractional legend count", withLegend({ count: 1.5 })],
    ["missing selection", withField("selection", undefined)],
    ["bad runner-up", withField("selection", { rule: "r", pickCount: 1, runnersUp: [{ stationId: "X", pickCount: 1 }] })],
  ])("rejects %s", (_name, raw) => {
    const parsed = parseStationDayManifest(raw);
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.errors.length).toBeGreaterThan(0);
  });
});

describe("loadStationDayManifest", () => {
  it("fetches the manifest URL and returns the validated manifest", async () => {
    const calls: string[] = [];
    const m = await loadStationDayManifest(respond(200, syntheticManifest(), calls));
    expect(calls).toEqual([STATION_DAY_MANIFEST_URL]);
    expect(STATION_DAY_MANIFEST_URL).toBe("/helicorder/station-day.json");
    expect(m?.title).toBe("Synthetic station day");
    expect(m && stationDayImageUrl(m)).toBe("/helicorder/station-day.png");
  });

  it("returns null without a manifest (404), on a server error, a network failure or bad JSON", async () => {
    expect(await loadStationDayManifest(respond(404, null))).toBeNull();
    expect(await loadStationDayManifest(respond(500, null))).toBeNull();
    expect(await loadStationDayManifest(respond(200, new Error("bad json")))).toBeNull();
    expect(
      await loadStationDayManifest(async () => {
        throw new Error("offline");
      }),
    ).toBeNull();
  });

  it("returns null, with a console warning, for a manifest that breaks the contract", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await loadStationDayManifest(respond(200, withField("rowMinutes", 15)))).toBeNull();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toMatch(/rowMinutes/);
  });
});
