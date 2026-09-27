// Test-only (rule 5): a synthetic station-day manifest, made-up network, station and values. Never
// served: the app reads the real one from /helicorder/station-day.json.

export function syntheticManifest(): Record<string, unknown> {
  return {
    image: "station-day.png",
    widthPx: 1920,
    heightPx: 1350,
    title: "Synthetic station day",
    caption: "Synthetic caption for XX.SYN01 channel ZZZ; gaps are left blank.",
    runId: "synthetic-run",
    stationId: "XX.SYN01",
    seedId: "XX.SYN01.00.ZZZ",
    channel: "ZZZ",
    stationKind: "borehole",
    sensorDepthM: 123.4,
    dayUtc: "2001-02-03",
    startUtc: "2001-02-03T00:00:00Z",
    endUtc: "2001-02-04T00:00:00Z",
    rowMinutes: 30,
    rows: 48,
    filterHz: [2, 20],
    gapsFilled: false,
    gapSeconds: 17.5,
    coverageFraction: 0.97,
    markerTime: "origin",
    legend: [
      { key: "tierA", label: "synthetic tier A", color: "#FFB547", opacity: 1, shape: "tick", count: 7 },
      { key: "tierB", label: "synthetic tier B", color: "#FFB547", opacity: 0.6, shape: "tick", count: 1234 },
      { key: "tierC", label: "synthetic tier C", color: "#FFB547", opacity: 0.3, shape: "tick", count: 0 },
      { key: "public", label: "synthetic public", color: "#DCE6F2", opacity: 1, shape: "diamond", count: 3 },
    ],
    selection: { rule: "synthetic selection rule", pickCount: 99, runnersUp: [{ stationId: "XX.SYN02", pickCount: 88 }] },
    source: "synthetic source",
    generator: "synthetic generator",
  };
}
